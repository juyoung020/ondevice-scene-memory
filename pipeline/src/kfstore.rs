//! 지도에 넣은 장면 보관 + RTAB-Map 노드 위치: 루프 클로저로 노드가 움직이면 보관한 장면을 고친 위치로 다시 쌓는다.
//! 장면 위치 = 기준 노드의 지금 위치 × (받을 때 기준 노드에서 본 상대 위치). 노드 위치는 스트림 그래프 블록(MMG1)으로 온다.
//! 줄인 보관: 점유 격자·물체에 쓰는 것만 — 깊이는 4 화소 간격, 검출 상자 안은 컬러·깊이 그대로.
//! 점유 격자는 16 화소 간격(8 + 16k)만 읽어 4 화소 간격 깊이와 같은 값이고, 물체 점·사진은 상자 안만 읽는다.
use crate::geom::*;
use crate::seg::Det;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 노드가 마지막으로 쌓은 위치에서 이만큼 넘게 움직이면 다시 쌓을 거리(m)·각도(°). 그 아래 보정은 모아 두었다가 넘을 때 한 번에
pub const MOVE_T: f64 = 0.02;
pub const MOVE_R: f64 = 2.0;

pub struct Kf {
    pub seq: u64, // 보관한 순서(다시 쌓는 동안 들어온 장면을 가려냄)
    pub held: bool, // 맡아 둔 장면으로 들어옴(기준 노드가 믿는 지도와 이어져야 지도에 들어감)
    pub t: f64,
    anchor: u32,
    rel: (M3, V3),
    raw: (M3, V3), // 받은 위치(기준 노드를 모를 때 그대로 씀)
    pub h_cam: f64, // 카메라 높이(바닥 위): 다시 쌓을 때 바닥 높이 = 카메라 z − h_cam
    pub w: usize,
    pub h: usize,
    /// 압축한 컬러·깊이: 장면을 받는 점 작업 스레드가 채움([`Kf::fill`]). 읽는 쪽(다시 쌓기)은 채워질 때까지 기다림
    z: std::sync::OnceLock<KfZ>,
    sparse: bool,
    dets: Option<Vec<DetZ>>,
    /// 점 지도에 넣은 점(거른 뒤 --scene-step 간격 격자의 깊이 mm·색, LZ4): 점 작업 스레드가 채움 → 다시 쌓기는 온 장면 없이 이것만 읽는다
    pub pts: std::sync::OnceLock<Option<PtsZ>>,
}

struct KfZ {
    rgb_z: Vec<u8>,   // 컬러: 채널별 면 + 왼쪽 화소와의 차 → LZ4(줄인 보관이면 검출 마스크 합집합 화소만, 줄 차례로 이어 한 줄로)
    depth_z: Vec<u8>, // 깊이 mm: 왼쪽 화소와의 차 → 아래·위 바이트 면 → LZ4(줄인 보관이면 4 화소 간격)
    box_depth_z: Vec<u8>, // 줄인 보관: 검출 마스크 합집합 화소의 깊이 mm(rgb_z 와 같은 화소 차례)
}

impl KfZ {
    fn bytes(&self) -> usize {
        self.rgb_z.len() + self.depth_z.len() + self.box_depth_z.len()
    }
}

/// 검출 마스크들의 합집합 화소 번호(줄 차례): 줄인 보관이 컬러 · 깊이를 남기는 화소(물체 점과 그 색). 보관한 검출에서 다시 만들어 따로 저장하지 않는다
fn mask_pixels(dets: &[DetZ], w: usize, h: usize) -> Vec<u32> {
    let mut on = vec![false; w * h];
    for d in dets {
        let [x1, y1, x2, _] = d.box_.map(|v| v.max(0) as usize);
        let bw = x2.saturating_sub(x1).max(1);
        for i in 0..d.n {
            if d.bits[i / 8] >> (i % 8) & 1 == 1 {
                let (x, y) = (x1 + i % bw, y1 + i / bw);
                if x < w && y < h {
                    on[y * w + x] = true;
                }
            }
        }
    }
    on.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect()
}

/// 점 격자 압축본: gw×gh(장면 화소 off + step 간격), 깊이 0 = 점 없음
pub struct PtsZ {
    pub step: usize,
    pub off: [usize; 2],
    pub gw: usize,
    pub gh: usize,
    depth_z: Vec<u8>,
    rgb_z: Vec<u8>,
}

impl PtsZ {
    pub fn pack(step: usize, off: [usize; 2], gw: usize, gh: usize, depth: &[f32], rgb: &[u8]) -> PtsZ {
        let d: Vec<u16> = depth.iter().map(|&v| (v * 1000.0).round().clamp(0.0, 65535.0) as u16).collect();
        PtsZ { step, off, gw, gh, depth_z: pack_depth(&d, gw, gh), rgb_z: pack_rgb(rgb, gw, gh) }
    }
    /// (깊이 m, 색)
    pub fn unpack(&self) -> (Vec<f32>, Vec<u8>) {
        (unpack_depth(&self.depth_z, self.gw, self.gh).iter().map(|&v| v as f32 * 1e-3).collect(), unpack_rgb(&self.rgb_z, self.gw, self.gh))
    }
    fn bytes(&self) -> usize {
        self.depth_z.len() + self.rgb_z.len()
    }
}

/// 보관한 검출: 마스크는 화소 8 개를 한 바이트에(받은 bool 마스크의 1/8)
struct DetZ {
    cls: usize,
    score: f32,
    box_: [i32; 4],
    n: usize,
    bits: Vec<u8>,
}

impl DetZ {
    fn pack(d: &Det) -> DetZ {
        let bits = d.mask.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |b, (i, &m)| b | (m as u8) << i)).collect();
        DetZ { cls: d.cls, score: d.score, box_: d.box_, n: d.mask.len(), bits }
    }
    fn unpack(&self) -> Det {
        Det { cls: self.cls, score: self.score, box_: self.box_, mask: (0..self.n).map(|i| self.bits[i / 8] >> (i % 8) & 1 == 1).collect() }
    }
}

const Q: usize = 4; // 줄인 보관의 깊이 화소 간격

impl Kf {
    pub fn depth(&self) -> Vec<f32> {
        if !self.sparse {
            return unpack_depth(&self.z().depth_z, self.w, self.h).iter().map(|&v| v as f32 * 1e-3).collect();
        }
        let (qw, qh) = (self.w.div_ceil(Q), self.h.div_ceil(Q));
        let q = unpack_depth(&self.z().depth_z, qw, qh);
        // 4 화소 간격 깊이를 늘려 채움: 줄마다 Q 칸씩 같은 값을 채우고, 같은 줄 묶음의 나머지 줄은 첫 줄을 복사
        // (화소마다 나눗셈 두 번 하던 것과 같은 값, 다시 쌓기 풀기 시간의 대부분이었음)
        let w = self.w;
        let mut d = vec![0f32; w * self.h];
        for y in 0..self.h {
            if y % Q == 0 {
                let (qrow, row) = (&q[(y / Q) * qw..(y / Q + 1) * qw], &mut d[y * w..(y + 1) * w]);
                for (qx, &v) in qrow.iter().enumerate() {
                    let x0 = qx * Q;
                    row[x0..(x0 + Q).min(w)].fill(v as f32 * 1e-3);
                }
            } else {
                let src = (y - y % Q) * w;
                d.copy_within(src..src + w, y * w);
            }
        }
        let px = self.mask_px();
        if !px.is_empty() {
            let z = unpack_depth(&self.z().box_depth_z, px.len(), 1);
            for (&i, &zv) in px.iter().zip(&z) {
                d[i as usize] = zv as f32 * 1e-3;
            }
        }
        d
    }
    /// 검출 마스크 합집합 화소(줄인 보관의 화소 차례)
    fn mask_px(&self) -> Vec<u32> {
        self.dets.as_deref().map_or_else(Vec::new, |d| mask_pixels(d, self.w, self.h))
    }
    /// 보관한 검출(받은 것과 같은 값)
    pub fn dets(&self) -> Option<Vec<Det>> {
        self.dets.as_ref().map(|v| v.iter().map(DetZ::unpack).collect())
    }
    /// 컬러: 줄인 보관이면 검출 마스크 안 화소만, 그 밖은 0
    pub fn rgb(&self) -> Vec<u8> {
        if !self.sparse {
            return unpack_rgb(&self.z().rgb_z, self.w, self.h);
        }
        let mut rgb = vec![0u8; self.w * self.h * 3];
        let px = self.mask_px();
        if !px.is_empty() {
            let z = unpack_rgb(&self.z().rgb_z, px.len(), 1);
            for (k, &i) in px.iter().enumerate() {
                let o = i as usize * 3;
                rgb[o..o + 3].copy_from_slice(&z[k * 3..k * 3 + 3]);
            }
        }
        rgb
    }
    /// 압축본(채워질 때까지 기다림)
    fn z(&self) -> &KfZ {
        self.z.wait()
    }
    /// 압축이 끝났는가
    pub fn ready(&self) -> bool {
        self.z.get().is_some()
    }
    /// 압축이 끝날 때까지 기다림(점 줄이 채움). 작업 풀 일꾼이면 기다리는 동안 풀의 다른 일(점 줄 포함)을 대신 돈다(일꾼을 붙잡아 생기는 교착 막기)
    pub fn wait_ready(&self) {
        if rayon::current_thread_index().is_some() {
            common::rt::wait_until(|| self.ready());
        }
        let _ = self.z();
    }
    /// 점 격자(점 줄이 채움): wait_ready 와 같은 규칙으로 기다림
    pub fn pts_wait(&self) -> &Option<PtsZ> {
        if rayon::current_thread_index().is_some() {
            common::rt::wait_until(|| self.pts.get().is_some());
        }
        self.pts.wait()
    }
    /// 보관 바이트: 아직 압축 전이면 None
    fn bytes(&self) -> Option<usize> {
        Some(self.z.get()?.bytes() + self.dets.iter().flatten().map(|d| d.bits.len()).sum::<usize>()
            + self.pts.get().and_then(|p| p.as_ref()).map_or(0, |p| p.bytes()))
    }
    /// 압축 전 장면의 바이트 위쪽 한계(LZ4 는 입력보다 조금 커질 수 있음: 넉넉히 두 배)
    fn bound(&self) -> usize {
        2 * self.w * self.h * (3 + 2 + 5) + 4096
    }
    /// 받은 컬러·깊이를 압축해 채움(점 작업 스레드, 밀려 건너뛰는 장면이면 본 스레드). 한 번만
    pub fn fill(&self, rgb: &[u8], depth: &crate::depth::Depth) {
        let (w, h) = (self.w, self.h);
        let z = if self.sparse {
            let (qw, qh) = (w.div_ceil(Q), h.div_ceil(Q));
            let q: Vec<u16> = (0..qw * qh).map(|i| depth.mm_round((i / qw) * Q * w + (i % qw) * Q)).collect();
            // 검출 마스크 합집합 화소만(물체 점과 그 색), 줄 차례로 이어 한 줄로 압축. 마스크 밖 컬러는 버리고 깊이는 4 화소 간격만
            let px = self.mask_px();
            let n = px.len();
            let mut cr = Vec::with_capacity(n * 3);
            let mut cd = Vec::with_capacity(n);
            for &i in &px {
                let i = i as usize;
                cr.extend_from_slice(&rgb[i * 3..i * 3 + 3]);
                cd.push(depth.mm_round(i));
            }
            let (rgb_z, box_depth_z) = if n > 0 { (pack_rgb(&cr, n, 1), pack_depth(&cd, n, 1)) } else { (vec![], vec![]) };
            KfZ { rgb_z, depth_z: pack_depth(&q, qw, qh), box_depth_z }
        } else {
            KfZ { rgb_z: pack_rgb(rgb, w, h), depth_z: pack_depth_it((0..w * h).map(|i| depth.mm_round(i)), w, h), box_depth_z: vec![] }
        };
        let _ = self.z.set(z);
    }
}

pub struct KfStore {
    nodes: HashMap<u32, (M3, V3)>,
    /// 마지막으로 쌓을 때(다시 쌓기 시작 · 처음 받음)의 노드 위치: 움직임은 여기서 잼(작은 보정이 쌓여도 놓치지 않게)
    built: HashMap<u32, (M3, V3)>,
    held: HashSet<u32>, // 믿는 지도와 아직 안 이어진 노드
    kfs: Vec<Arc<Kf>>,
    budget: usize,
    bytes: usize,
    stride: usize, // 장면 n 장마다 하나 보관(예산을 넘을 때마다 두 배)
    count: usize,
    pub dirty: bool, // 보관한 장면의 기준 노드가 움직였음 → 다시 쌓아야 함
    pub moved: (f64, f64), // 다시 쌓기 전까지 가장 크게 움직인 노드(m, °)
}

/// 가만히 있는 동안 보관 간격(초)
const STILL_GAP_S: f64 = 2.0;

/// 두 위치가 1 cm · 1° 안인가
fn still(a: &(M3, V3), b: &(M3, V3)) -> bool {
    let d = [a.1[0] - b.1[0], a.1[1] - b.1[1], a.1[2] - b.1[2]];
    let tr: f64 = (0..3).map(|i| (0..3).map(|j| a.0[i][j] * b.0[i][j]).sum::<f64>()).sum(); // tr(Raᵀ·Rb)
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 0.01 * 0.01 && (tr - 1.0) / 2.0 > 1f64.to_radians().cos()
}

fn inv_mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    let rt = [[a.0[0][0], a.0[1][0], a.0[2][0]], [a.0[0][1], a.0[1][1], a.0[2][1]], [a.0[0][2], a.0[1][2], a.0[2][2]]];
    (mm(&rt, &b.0), mv(&rt, &sub(&b.1, &a.1)))
}

fn mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    (mm(&a.0, &b.0), add(&mv(&a.0, &b.1), &a.1))
}

pub fn angle(a: &M3, b: &M3) -> f64 {
    // 두 회전 사이 각도: trace(aᵀb)
    let mut tr = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            tr += a[j][i] * b[j][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

impl KfStore {
    pub fn new(budget_mb: usize) -> KfStore {
        KfStore { nodes: HashMap::new(), built: HashMap::new(), held: HashSet::new(), kfs: vec![], budget: budget_mb << 20, bytes: 0, stride: 1, count: 0, dirty: false, moved: (0.0, 0.0) }
    }

    pub fn len(&self) -> usize {
        self.kfs.len()
    }

    /// 보관한 장면 바이트(압축 뒤, 압축이 끝날 때까지 기다림)
    pub fn bytes(&mut self) -> usize {
        self.settle(true);
        self.bytes
    }

    /// 예산 맞추기: 압축이 덜 끝난 장면은 위쪽 한계로 어림해 예산 안이면 기다리지 않고, 넘을 수 있으면 압축을 기다려 정확히 센 뒤
    /// 예산을 넘으면 하나 걸러 버리고 앞으로도 두 배 간격으로 보관한다. 보관 목록을 보는 곳(보관 · 그래프 · 다시 쌓기 시작 · 끝) 앞에서 부른다
    pub fn settle(&mut self, exact: bool) {
        let est: usize = self.kfs.iter().map(|k| k.bytes().unwrap_or_else(|| k.bound())).sum();
        if !exact && est <= self.budget {
            return;
        }
        self.bytes = self.kfs.iter().map(|k| {
            k.wait_ready();
            k.bytes().unwrap()
        }).sum(); // 점 격자는 점 작업 스레드가 나중에 채우니 매번 다시 셈
        while self.bytes > self.budget && self.kfs.len() > 2 {
            let old = std::mem::take(&mut self.kfs);
            self.bytes = 0;
            for (i, k) in old.into_iter().enumerate() {
                if i % 2 == 0 {
                    self.bytes += k.bytes().unwrap();
                    self.kfs.push(k);
                }
            }
            self.stride *= 2;
            common::info!("장면 보관: 예산 {} MB 넘음 → {} 장만 남기고 {} 장마다 보관", self.budget >> 20, self.kfs.len(), self.stride);
        }
    }

    /// 그래프 블록: 새 노드는 더하고, 있던 노드가 문턱 넘게 움직였으면 다시 쌓기 표시.
    /// 맡아 둔 노드는 지도에 없으니 움직여도 표시하지 않고, 믿는 노드로 바뀌면(이어짐) 그 노드의 장면을 넣도록 표시
    pub fn apply_graph(&mut self, upd: &[(u32, M3, V3, bool)]) {
        for (id, r, p, held) in upd {
            let old = self.nodes.insert(*id, (*r, *p));
            if *held {
                self.held.insert(*id);
                continue;
            }
            if self.held.remove(id) {
                self.dirty |= self.kfs.iter().any(|k| k.anchor == *id);
                continue;
            }
            if let Some(old) = old {
                let b = *self.built.entry(*id).or_insert(old);
                let (dt, dr) = (norm(&sub(p, &b.1)), angle(r, &b.0));
                if dt > MOVE_T || dr > MOVE_R {
                    self.dirty |= !self.kfs.is_empty();
                    self.moved = (self.moved.0.max(dt), self.moved.1.max(dr));
                }
            }
        }
    }

    /// 이 노드를 기준으로 맡아 둘 수 있는지(맡아 둠 표시로 받은 노드)
    pub fn holds(&self, node: u32) -> bool {
        self.held.contains(&node)
    }

    /// 지금 지도에 넣을 장면인지(기준 노드가 아직 안 이어졌으면 아님)
    pub fn usable(&self, k: &Kf) -> bool {
        !self.held.contains(&k.anchor)
    }

    /// 맡아 둔 장면 수: (이어져 지도에 들어간 것, 아직 안 이어진 것)
    pub fn held_counts(&self) -> (usize, usize) {
        let n = self.kfs.iter().filter(|k| k.held).count();
        let w = self.kfs.iter().filter(|k| k.held && !self.usable(k)).count();
        (n - w, w)
    }

    /// 지도에 넣은 장면 보관(받은 위치 r, p 는 수평 맞춤 전, RTAB-Map 지도 좌표). sparse: 줄인 보관(점유 격자·물체만 다시 쌓음).
    /// 컬러·깊이 압축은 [`Kf::fill`] 로 따로(점 작업 스레드): 여기서는 위치·검출만 담아 바로 돌려준다
    #[allow(clippy::too_many_arguments)]
    pub fn push_lazy(&mut self, t: f64, anchor: u32, r: M3, p: V3, h_cam: f64, w: usize, h: usize, dets: Option<Vec<Det>>, sparse: bool) -> Option<Arc<Kf>> {
        self.settle(false);
        self.count += 1;
        if (self.count - 1) % self.stride != 0 {
            return None;
        }
        let (anchor, rel) = match self.nodes.get(&anchor) {
            Some(n) => (anchor, inv_mul(n, &(r, p))),
            None => (0, (r, p)),
        };
        // 카메라를 가만히 두면 거의 같은 장면이 계속 쌓인다(8 분에 3,679 장 → 다시 쌓기 23 초).
        // 마지막으로 보관한 장면과 같은 기준 노드에서 1 cm · 1° 안이면 2 초에 한 장만 보관한다(그 자리에서 바뀐 것은 2 초 간격으로 남음)
        if let Some(last) = self.kfs.last() {
            if last.anchor == anchor && last.sparse == sparse && (t - last.t).abs() < STILL_GAP_S && still(&last.rel, &rel) {
                return None;
            }
        }
        let kf = Arc::new(Kf {
            seq: self.count as u64, held: self.held.contains(&anchor), t, anchor, rel, raw: (r, p), h_cam, w, h, z: Default::default(), sparse,
            dets: dets.map(|v| v.iter().map(DetZ::pack).collect()), // 장면의 검출은 여기서 끝: 비트로 묶어 보관
            pts: Default::default(),
        });
        self.kfs.push(kf.clone());
        Some(kf)
    }

    /// 보관 + 바로 압축(시험)
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, t: f64, anchor: u32, r: M3, p: V3, h_cam: f64, w: usize, h: usize, rgb: &[u8], depth: &crate::depth::Depth, dets: Option<Vec<Det>>, sparse: bool) -> Option<Arc<Kf>> {
        let kf = self.push_lazy(t, anchor, r, p, h_cam, w, h, dets, sparse)?;
        kf.fill(rgb, depth);
        self.settle(false);
        self.kfs.last().filter(|k| Arc::ptr_eq(k, &kf)).cloned()
    }

    /// 보관한 장면의 지금 위치(수평 맞춤 전): 기준 노드 × 상대 위치
    pub fn pose(&self, k: &Kf) -> (M3, V3) {
        match self.nodes.get(&k.anchor) {
            Some(n) if k.anchor != 0 => mul(n, &k.rel),
            _ => k.raw,
        }
    }

    pub fn kfs(&self) -> &[Arc<Kf>] {
        &self.kfs
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
        self.moved = (0.0, 0.0);
        self.built.clone_from(&self.nodes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 줄 풀기(aarch64 는 NEON 누적 합)가 한 칸씩 더하는 스칼라 식과 같은지(폭 여럿, 감싸기 포함)
    #[test]
    fn unpack_rows_same_as_scalar() {
        let mut s = 3u64;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u8 };
        for w in [1usize, 15, 16, 17, 31, 32, 33, 100, 640] {
            let p: Vec<Vec<u8>> = (0..3).map(|_| (0..w).map(|_| r()).collect()).collect();
            let mut out = vec![0u8; 3 * w];
            unpack_rgb_row([&p[0], &p[1], &p[2]], &mut out);
            for c in 0..3 {
                let mut pv = 0u8;
                for x in 0..w {
                    pv = pv.wrapping_add(p[c][x]);
                    assert_eq!(out[x * 3 + c], pv, "rgb w {w} c {c} x {x}");
                }
            }
            let mut d = vec![0u16; w];
            unpack_depth_row(&p[0], &p[1], &mut d);
            let mut pv = 0u16;
            for x in 0..w {
                pv = pv.wrapping_add(p[0][x] as u16 | (p[1][x] as u16) << 8);
                assert_eq!(d[x], pv, "depth w {w} x {x}");
            }
        }
    }

    /// 묶기 함수가 풀어 쓴 기준 계산과 같은 바이트를 내는지(크기 여럿, 값은 고정 난수)
    #[test]
    fn pack_matches_reference() {
        let ref_rgb = |rgb: &[u8], w: usize, h: usize| {
            let n = w * h;
            let mut b = vec![0u8; n * 3];
            for c in 0..3 {
                for y in 0..h {
                    let mut prev = 0u8;
                    for x in 0..w {
                        let v = rgb[(y * w + x) * 3 + c];
                        b[c * n + y * w + x] = v.wrapping_sub(prev);
                        prev = v;
                    }
                }
            }
            lz4_flex::compress_prepend_size(&b)
        };
        let ref_depth = |d: &[u16], w: usize, h: usize| {
            let n = w * h;
            let mut b = vec![0u8; n * 2];
            for y in 0..h {
                let mut prev = 0u16;
                for x in 0..w {
                    let i = y * w + x;
                    let v = d[i].wrapping_sub(prev);
                    prev = d[i];
                    b[i] = v as u8;
                    b[n + i] = (v >> 8) as u8;
                }
            }
            lz4_flex::compress_prepend_size(&b)
        };
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // 큰 것 다음에 작은 것: 다시 쓰는 작업 버퍼에 앞 장면 값이 남아도 결과가 같아야 함
        for (w, h) in [(640, 480), (1, 1), (7, 3), (160, 120), (0, 0), (33, 1), (1, 29)] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|i| if i % 5 == 0 { rnd() as u8 } else { (i / 7) as u8 }).collect();
            let dm: Vec<f32> = (0..w * h).map(|i| if i % 11 == 0 { 0.0 } else { (rnd() % 70000) as f32 / 1000.0 - 0.5 }).collect();
            let d16: Vec<u16> = dm.iter().map(|&v| mm16(v)).collect();
            assert_eq!(pack_rgb(&rgb, w, h), ref_rgb(&rgb, w, h), "컬러 {w}x{h}");
            assert_eq!(pack_depth(&d16, w, h), ref_depth(&d16, w, h), "깊이 {w}x{h}");
            assert_eq!(pack_depth_m(&dm, w, h), ref_depth(&d16, w, h), "깊이(미터) {w}x{h}");
            assert_eq!(unpack_rgb(&pack_rgb(&rgb, w, h), w, h), rgb);
            assert_eq!(unpack_depth(&pack_depth(&d16, w, h), w, h), d16);
        }
    }


    #[test]
    fn det_bits_roundtrip() {
        for n in [0usize, 1, 7, 8, 9, 20 * 16 + 3] {
            let d = Det { cls: 3, score: 0.7, box_: [1, 2, 3, 4], mask: (0..n).map(|i| (i * 7 + i / 3) % 5 < 2).collect() };
            let u = DetZ::pack(&d).unpack();
            assert_eq!((u.cls, u.score.to_bits(), u.box_, u.mask), (d.cls, d.score.to_bits(), d.box_, d.mask));
        }
    }

    fn rz(a: f64) -> M3 {
        let (s, c) = a.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn follows_anchor() {
        let mut st = KfStore::new(100);
        st.apply_graph(&[(1, rz(0.0), [0.0, 0.0, 0.0], false)]);
        // 노드 1 에서 x 로 1 m 앞의 장면
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &crate::depth::Depth::M(vec![1.0]), None, false);
        assert!(!st.dirty);
        // 노드가 원점에서 90° 돌고 (0, 2, 0) 으로 옮겨짐 → 장면은 (0, 3, 0), 90°
        st.apply_graph(&[(1, rz(std::f64::consts::FRAC_PI_2), [0.0, 2.0, 0.0], false)]);
        assert!(st.dirty);
        let (r, p) = st.pose(&st.kfs()[0]);
        assert!((p[0] - 0.0).abs() < 1e-9 && (p[1] - 3.0).abs() < 1e-9);
        assert!((angle(&r, &rz(std::f64::consts::FRAC_PI_2))).abs() < 1e-6);
    }

    #[test]
    fn held_joins_later() {
        let mut st = KfStore::new(100);
        st.apply_graph(&[(1, rz(0.0), [0.0, 0.0, 0.0], false)]);
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &crate::depth::Depth::M(vec![1.0]), None, false);
        // 새 지도의 노드 2(맡아 둠)와 그 노드에서 x 로 1 m 앞의 장면: 지도에 안 넣음
        st.apply_graph(&[(2, rz(0.0), [5.0, 0.0, 0.0], true)]);
        assert!(st.holds(2) && !st.holds(1));
        st.push(1.0, 2, rz(0.0), [6.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &crate::depth::Depth::M(vec![1.0]), None, false);
        st.apply_graph(&[(2, rz(0.0), [7.0, 0.0, 0.0], true)]);
        assert!(!st.dirty && !st.usable(&st.kfs()[1]) && st.usable(&st.kfs()[0]));
        assert_eq!(st.held_counts(), (0, 1));
        // 이어짐: 노드 2 가 믿는 노드로 (0, 3, 0) 에 → 장면은 (1, 3, 0) 에서 지도에 들어감
        st.apply_graph(&[(2, rz(0.0), [0.0, 3.0, 0.0], false)]);
        assert!(st.dirty && st.usable(&st.kfs()[1]));
        assert_eq!(st.held_counts(), (1, 0));
        let (_, p) = st.pose(&st.kfs()[1]);
        assert!((p[0] - 1.0).abs() < 1e-9 && (p[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn small_move_not_dirty() {
        let mut st = KfStore::new(100);
        st.apply_graph(&[(1, rz(0.0), [0.0, 0.0, 0.0], false)]);
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &crate::depth::Depth::M(vec![1.0]), None, false);
        st.apply_graph(&[(1, rz(0.001), [0.005, 0.0, 0.0], false)]);
        assert!(!st.dirty);
    }

    #[test]
    fn pack_roundtrip_real() {
        // TUM 책상 한 장(있으면): 손실 없이 되돌아오는지, 압축률·시간
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/tum/rgbd_dataset_freiburg1_desk");
        let (Ok(rl), Ok(dl)) = (std::fs::read_to_string(dir.join("rgb.txt")), std::fs::read_to_string(dir.join("depth.txt"))) else { return };
        let first = |s: &str| s.lines().find(|l| !l.starts_with('#')).unwrap().split_whitespace().nth(1).unwrap().to_string();
        let im = image::open(dir.join(first(&rl))).unwrap().into_rgb8();
        let (w, h) = (im.width() as usize, im.height() as usize);
        let rgb = im.into_raw();
        let d: Vec<u16> = image::open(dir.join(first(&dl))).unwrap().into_luma16().into_raw().iter().map(|v| v / 5).collect();
        let t = std::time::Instant::now();
        let (zr, zd) = (pack_rgb(&rgb, w, h), pack_depth(&d, w, h));
        let tp = t.elapsed();
        let t = std::time::Instant::now();
        assert_eq!(unpack_rgb(&zr, w, h), rgb);
        assert_eq!(unpack_depth(&zd, w, h), d);
        common::info!("컬러 {} → {} B, 깊이 {} → {} B, 묶기 {:?}, 풀기 {:?}", rgb.len(), zr.len(), d.len() * 2, zd.len(), tp, t.elapsed());
    }

    #[test]
    fn sparse_keeps_grid_and_masks() {
        // 줄인 보관: 점유 격자 화소(8 + 16k)와 검출 마스크 안은 받은 값 그대로, 마스크 밖 컬러는 0 · 깊이는 4 화소 간격 값
        let (w, h) = (64, 48);
        let d: Vec<f32> = (0..w * h).map(|i| 0.5 + (i % 97) as f32 * 0.01).collect();
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        // 상자 [10, 5, 30, 21] 안 왼쪽 절반만 마스크
        let det = Det { cls: 1, score: 0.9, box_: [10, 5, 30, 21], mask: (0..20 * 16).map(|i| i % 20 < 10).collect() };
        let mut st = KfStore::new(100);
        st.push(0.0, 0, rz(0.0), [0.0; 3], 1.0, w, h, &rgb, &crate::depth::Depth::M(d.clone()), Some(vec![det]), true);
        let kf = &st.kfs()[0];
        let (d2, c2) = (kf.depth(), kf.rgb());
        let mm = |v: f32| (v * 1000.0).round() * 1e-3;
        for v in (8..h).step_by(16) {
            for u in (8..w).step_by(16) {
                assert!((d2[v * w + u] - mm(d[v * w + u])).abs() < 1e-6);
            }
        }
        for v in 5..21 {
            for u in 10..20 {
                assert!((d2[v * w + u] - mm(d[v * w + u])).abs() < 1e-6);
                assert_eq!(c2[(v * w + u) * 3..(v * w + u) * 3 + 3], rgb[(v * w + u) * 3..(v * w + u) * 3 + 3]);
            }
            for u in 20..30 {
                assert_eq!(c2[(v * w + u) * 3..(v * w + u) * 3 + 3], [0, 0, 0]);
                let q = (v - v % Q) * w + (u - u % Q);
                assert!((d2[v * w + u] - mm(d[q])).abs() < 1e-6);
            }
        }
        assert!((d2[3 * w + 2] - mm(d[0])).abs() < 1e-6);
    }
    #[test]
    fn sparse_fill_matches_per_pixel() {
        // 줄 복사로 늘린 깊이가 화소마다 계산하던 값과 비트까지 같은지(가로·세로가 Q 의 배수가 아닌 경우 포함)
        let (w, h) = (63, 47);
        let d: Vec<f32> = (0..w * h).map(|i| 0.3 + (i % 89) as f32 * 0.013).collect();
        let mut st = KfStore::new(100);
        st.push(0.0, 0, rz(0.0), [0.0; 3], 1.0, w, h, &vec![0; w * h * 3], &crate::depth::Depth::M(d.clone()), None, true);
        let kf = &st.kfs()[0];
        let (qw, qh) = (w.div_ceil(Q), h.div_ceil(Q));
        let q = unpack_depth(&kf.z().depth_z, qw, qh);
        let old: Vec<f32> = (0..w * h).map(|i| q[(i / w / Q) * qw + (i % w) / Q] as f32 * 1e-3).collect();
        assert_eq!(kf.depth().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), old.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn still_camera_keeps_one_per_gap() {
        // 가만히 30 장/초로 10 초: 2 초에 한 장만 남는다. 1 cm 넘게 움직이거나 1° 넘게 돌면 바로 보관
        let (w, h) = (16, 12);
        let (rgb, d) = (vec![0u8; w * h * 3], vec![1.0f32; w * h]);
        let mut st = KfStore::new(100);
        for i in 0..300 {
            st.push(i as f64 / 30.0, 0, rz(0.0), [0.0; 3], 1.0, w, h, &rgb, &crate::depth::Depth::M(d.clone()), None, true);
        }
        assert_eq!(st.kfs().len(), 5);
        let n = st.kfs().len();
        st.push(10.01, 0, rz(0.0), [0.04, 0.0, 0.0], 1.0, w, h, &rgb, &crate::depth::Depth::M(d.clone()), None, true);
        st.push(10.02, 0, rz(3f64.to_radians()), [0.04, 0.0, 0.0], 1.0, w, h, &rgb, &crate::depth::Depth::M(d.clone()), None, true);
        st.push(10.03, 0, rz(3f64.to_radians()), [0.04, 0.0, 0.0], 1.0, w, h, &rgb, &crate::depth::Depth::M(d.clone()), None, true);
        assert_eq!(st.kfs().len(), n + 2);
    }

    #[test]
    fn budget_thins() {
        let mut st = KfStore::new(1); // 1 MB
        // 압축이 안 되는 값(의사 난수)
        let mut s = 1u32;
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 24) as u8 };
        let rgb: Vec<u8> = (0..320 * 300 * 3).map(|_| rnd()).collect();
        let d: Vec<f32> = (0..320 * 300).map(|_| rnd() as f32 * 0.1).collect();
        for i in 0..10 {
            st.push(i as f64, 0, rz(0.0), [i as f64 * 0.1, 0.0, 0.0], 1.0, 320, 300, &rgb, &crate::depth::Depth::M(d.clone()), None, false);
        }
        assert!(st.bytes <= 1 << 20);
        assert!(st.stride > 1);
        assert!(st.kfs().windows(2).all(|w| w[0].t < w[1].t));
    }
}


thread_local! {
    /// 묶기 작업 버퍼(스레드마다 하나, 장면마다 다시 씀)
    static PACK_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 컬러 압축: 채널마다 면으로 모으고 줄 안에서 왼쪽 화소와의 차(u8 감싸기) → LZ4. 손실 없음.
pub fn pack_rgb(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    assert_eq!(rgb.len(), n * 3);
    PACK_BUF.with_borrow_mut(|b| {
        b.resize(n * 3, 0);
        if n > 0 {
            let (r, gb) = b.split_at_mut(n);
            let (g, bl) = gb.split_at_mut(n);
            for (((src, r), g), bl) in rgb.chunks_exact(w * 3).zip(r.chunks_exact_mut(w)).zip(g.chunks_exact_mut(w)).zip(bl.chunks_exact_mut(w)) {
                let mut prev = [0u8; 3];
                for (((px, r), g), bl) in src.chunks_exact(3).zip(r).zip(g).zip(bl) {
                    *r = px[0].wrapping_sub(prev[0]);
                    *g = px[1].wrapping_sub(prev[1]);
                    *bl = px[2].wrapping_sub(prev[2]);
                    prev = [px[0], px[1], px[2]];
                }
            }
        }
        lz4_flex::compress_prepend_size(b)
    })
}

pub fn unpack_rgb(z: &[u8], w: usize, h: usize) -> Vec<u8> {
    let b = lz4_flex::decompress_size_prepended(z).expect("보관 컬러 풀기");
    let n = w * h;
    let mut rgb = vec![0u8; n * 3];
    for y in 0..h {
        let r = y * w;
        unpack_rgb_row([&b[r..r + w], &b[n + r..n + r + w], &b[2 * n + r..2 * n + r + w]], &mut rgb[r * 3..(r + w) * 3]);
    }
    rgb
}

/// 한 줄: 면마다 왼쪽부터 누적 합(u8 감싸기) → RGB 섞어 담기. aarch64 는 16 칸씩 NEON 누적 합(정수라 값이 같음)
#[inline]
#[allow(unused_assignments)]
fn unpack_rgb_row(p: [&[u8]; 3], out: &mut [u8]) {
    let w = p[0].len();
    #[allow(unused_mut)]
    let mut prev = [0u8; 3];
    #[allow(unused_mut)]
    let mut x = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: x + 16 ≤ w 인 동안 세 면과 out[3x .. 3x+48] 은 범위 안
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u8(0);
        let scan = |v: uint8x16_t, c: uint8x16_t| {
            let v = vaddq_u8(v, vextq_u8::<15>(z, v));
            let v = vaddq_u8(v, vextq_u8::<14>(z, v));
            let v = vaddq_u8(v, vextq_u8::<12>(z, v));
            let v = vaddq_u8(v, vextq_u8::<8>(z, v));
            vaddq_u8(v, c)
        };
        let (mut c0, mut c1, mut c2) = (z, z, z);
        while x + 16 <= w {
            let r = scan(vld1q_u8(p[0].as_ptr().add(x)), c0);
            let g = scan(vld1q_u8(p[1].as_ptr().add(x)), c1);
            let bb = scan(vld1q_u8(p[2].as_ptr().add(x)), c2);
            vst3q_u8(out.as_mut_ptr().add(3 * x), uint8x16x3_t(r, g, bb));
            c0 = vdupq_laneq_u8::<15>(r);
            c1 = vdupq_laneq_u8::<15>(g);
            c2 = vdupq_laneq_u8::<15>(bb);
            x += 16;
        }
        prev = [vgetq_lane_u8::<0>(c0), vgetq_lane_u8::<0>(c1), vgetq_lane_u8::<0>(c2)];
    }
    for c in 0..3 {
        let mut pv = prev[c];
        for xx in x..w {
            pv = pv.wrapping_add(p[c][xx]);
            out[xx * 3 + c] = pv;
        }
    }
}

/// 한 줄: (아래 바이트 | 위 바이트 << 8) 의 왼쪽부터 누적 합(u16 감싸기). aarch64 는 16 칸씩 NEON(정수라 값이 같음)
#[inline]
#[allow(unused_assignments)]
fn unpack_depth_row(lo: &[u8], hi: &[u8], out: &mut [u16]) {
    let w = out.len();
    #[allow(unused_mut)]
    let mut prev = 0u16;
    #[allow(unused_mut)]
    let mut x = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: x + 16 ≤ w 인 동안 읽기·쓰기 범위 안
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u16(0);
        let scan = |v: uint16x8_t, c: uint16x8_t| {
            let v = vaddq_u16(v, vextq_u16::<7>(z, v));
            let v = vaddq_u16(v, vextq_u16::<6>(z, v));
            let v = vaddq_u16(v, vextq_u16::<4>(z, v));
            vaddq_u16(v, c)
        };
        let mut c = z;
        while x + 16 <= w {
            let (l, h) = (vld1q_u8(lo.as_ptr().add(x)), vld1q_u8(hi.as_ptr().add(x)));
            let a = scan(vreinterpretq_u16_u8(vzip1q_u8(l, h)), c);
            let b = scan(vreinterpretq_u16_u8(vzip2q_u8(l, h)), vdupq_laneq_u16::<7>(a));
            vst1q_u16(out.as_mut_ptr().add(x), a);
            vst1q_u16(out.as_mut_ptr().add(x + 8), b);
            c = vdupq_laneq_u16::<7>(b);
            x += 16;
        }
        prev = vgetq_lane_u16::<0>(c);
    }
    for xx in x..w {
        prev = prev.wrapping_add(lo[xx] as u16 | (hi[xx] as u16) << 8);
        out[xx] = prev;
    }
}


/// 깊이 압축: 줄 안에서 왼쪽 화소와의 차(u16 감싸기)를 아래 바이트 면·위 바이트 면으로 → LZ4. 손실 없음.
pub fn pack_depth(d: &[u16], w: usize, h: usize) -> Vec<u8> {
    assert_eq!(d.len(), w * h);
    pack_depth_it(d.iter().copied(), w, h)
}

/// 깊이(미터) → mm u16 으로 바꾸면서 바로 압축(`pack_depth` 와 같은 바이트). 시험용: 보관은 `push` 가 `Depth` 에서 같은 식으로
#[cfg(test)]
pub fn pack_depth_m(d: &[f32], w: usize, h: usize) -> Vec<u8> {
    assert_eq!(d.len(), w * h);
    pack_depth_it(d.iter().map(|&v| mm16(v)), w, h)
}

#[allow(dead_code)]
fn mm16(d: f32) -> u16 {
    (d * 1000.0).round().clamp(0.0, 65535.0) as u16
}

fn pack_depth_it(mut d: impl Iterator<Item = u16>, w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    PACK_BUF.with_borrow_mut(|b| {
        b.resize(n * 2, 0);
        if n > 0 {
            let (lo, hi) = b.split_at_mut(n);
            for (lo, hi) in lo.chunks_exact_mut(w).zip(hi.chunks_exact_mut(w)) {
                let mut prev = 0u16;
                for (lo, hi) in lo.iter_mut().zip(hi) {
                    let cur = d.next().unwrap();
                    let v = cur.wrapping_sub(prev);
                    prev = cur;
                    *lo = v as u8;
                    *hi = (v >> 8) as u8;
                }
            }
        }
        lz4_flex::compress_prepend_size(b)
    })
}

pub fn unpack_depth(z: &[u8], w: usize, h: usize) -> Vec<u16> {
    let b = lz4_flex::decompress_size_prepended(z).expect("보관 깊이 풀기");
    let n = w * h;
    let mut d = vec![0u16; n];
    for y in 0..h {
        let r = y * w;
        unpack_depth_row(&b[r..r + w], &b[n + r..n + r + w], &mut d[r..r + w]);
    }
    d
}
