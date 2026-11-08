//! 지도에 넣은 장면 보관 + RTAB-Map 노드 위치: 루프 클로저로 노드가 움직이면 보관한 장면을 고친 위치로 다시 쌓는다.
//! 장면 위치 = 기준 노드의 지금 위치 × (받을 때 기준 노드에서 본 상대 위치). 노드 위치는 스트림 그래프 블록(MMG1)으로 온다.
//! 줄인 보관(표면 지도: 표면은 서브맵이 맡음): 점유 격자·물체에 쓰는 것만 — 깊이는 4 화소 간격, 검출 상자 안은 컬러·깊이 그대로.
//! 점유 격자는 16 화소 간격(8 + 16k)만 읽어 4 화소 간격 깊이와 같은 값이고, 물체 점·사진은 상자 안만 읽는다.
use crate::geom::*;
use crate::seg::Det;
use std::collections::HashMap;
use std::sync::Arc;

/// 노드가 이만큼 넘게 움직이면 다시 쌓을 거리(m)·각도(°)
const MOVE_T: f64 = 0.02;
const MOVE_R: f64 = 1.0;

pub struct Kf {
    pub seq: u64, // 보관한 순서(다시 쌓는 동안 들어온 장면을 가려냄)
    pub t: f64,
    anchor: u32,
    rel: (M3, V3),
    raw: (M3, V3), // 받은 위치(기준 노드를 모를 때 그대로 씀)
    pub h_cam: f64, // 카메라 높이(바닥 위): 다시 쌓을 때 바닥 높이 = 카메라 z − h_cam
    pub w: usize,
    pub h: usize,
    rgb_z: Vec<u8>,   // 컬러: 채널별 면 + 왼쪽 화소와의 차 → LZ4(줄인 보관이면 비어 있음)
    depth_z: Vec<u8>, // 깊이 mm: 왼쪽 화소와의 차 → 아래·위 바이트 면 → LZ4(줄인 보관이면 4 화소 간격)
    crops: Vec<Crop>, // 줄인 보관: 검출 상자 안 컬러·깊이
    sparse: bool,
    pub dets: Option<Vec<Det>>,
    /// 점 지도에 넣은 점(거른 뒤 --scene-step 간격 격자의 깊이 mm·색, LZ4): 점 작업 스레드가 채움 → 다시 쌓기는 온 장면 없이 이것만 읽는다
    pub pts: std::sync::OnceLock<Option<PtsZ>>,
}

/// 점 격자 압축본: gw×gh(장면 화소 step 간격), 깊이 0 = 점 없음
pub struct PtsZ {
    pub step: usize,
    pub gw: usize,
    pub gh: usize,
    depth_z: Vec<u8>,
    rgb_z: Vec<u8>,
}

impl PtsZ {
    pub fn pack(step: usize, gw: usize, gh: usize, depth: &[f32], rgb: &[u8]) -> PtsZ {
        let d: Vec<u16> = depth.iter().map(|&v| (v * 1000.0).round().clamp(0.0, 65535.0) as u16).collect();
        PtsZ { step, gw, gh, depth_z: pack_depth(&d, gw, gh), rgb_z: pack_rgb(rgb, gw, gh) }
    }
    /// (깊이 m, 색)
    pub fn unpack(&self) -> (Vec<f32>, Vec<u8>) {
        (unpack_depth(&self.depth_z, self.gw, self.gh).iter().map(|&v| v as f32 * 1e-3).collect(), unpack_rgb(&self.rgb_z, self.gw, self.gh))
    }
    fn bytes(&self) -> usize {
        self.depth_z.len() + self.rgb_z.len()
    }
}

/// 검출 상자 안 화소(줄인 보관)
struct Crop {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    rgb_z: Vec<u8>,
    depth_z: Vec<u8>,
}

const Q: usize = 4; // 줄인 보관의 깊이 화소 간격

impl Kf {
    pub fn depth(&self) -> Vec<f32> {
        if !self.sparse {
            return unpack_depth(&self.depth_z, self.w, self.h).iter().map(|&v| v as f32 * 1e-3).collect();
        }
        let (qw, qh) = (self.w.div_ceil(Q), self.h.div_ceil(Q));
        let q = unpack_depth(&self.depth_z, qw, qh);
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
        for c in &self.crops {
            let z = unpack_depth(&c.depth_z, c.w, c.h);
            for y in 0..c.h {
                for x in 0..c.w {
                    d[(c.y + y) * self.w + c.x + x] = z[y * c.w + x] as f32 * 1e-3;
                }
            }
        }
        d
    }
    pub fn rgb(&self) -> Vec<u8> {
        if !self.sparse {
            return unpack_rgb(&self.rgb_z, self.w, self.h);
        }
        let mut rgb = vec![0u8; self.w * self.h * 3];
        for c in &self.crops {
            let z = unpack_rgb(&c.rgb_z, c.w, c.h);
            for y in 0..c.h {
                let o = ((c.y + y) * self.w + c.x) * 3;
                rgb[o..o + c.w * 3].copy_from_slice(&z[y * c.w * 3..(y + 1) * c.w * 3]);
            }
        }
        rgb
    }
    fn bytes(&self) -> usize {
        self.rgb_z.len() + self.depth_z.len() + self.crops.iter().map(|c| c.rgb_z.len() + c.depth_z.len()).sum::<usize>()
            + self.dets.iter().flatten().map(|d| d.mask.len()).sum::<usize>()
            + self.pts.get().and_then(|p| p.as_ref()).map_or(0, |p| p.bytes())
    }
}

pub struct KfStore {
    nodes: HashMap<u32, (M3, V3)>,
    kfs: Vec<Arc<Kf>>,
    budget: usize,
    bytes: usize,
    stride: usize, // 장면 n 장마다 하나 보관(예산을 넘을 때마다 두 배)
    count: usize,
    pub dirty: bool, // 보관한 장면의 기준 노드가 움직였음 → 다시 쌓아야 함
    pub moved: (f64, f64), // 다시 쌓기 전까지 가장 크게 움직인 노드(m, °)
}

fn inv_mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    let rt = [[a.0[0][0], a.0[1][0], a.0[2][0]], [a.0[0][1], a.0[1][1], a.0[2][1]], [a.0[0][2], a.0[1][2], a.0[2][2]]];
    (mm(&rt, &b.0), mv(&rt, &sub(&b.1, &a.1)))
}

fn mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    (mm(&a.0, &b.0), add(&mv(&a.0, &b.1), &a.1))
}

fn angle(a: &M3, b: &M3) -> f64 {
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
        KfStore { nodes: HashMap::new(), kfs: vec![], budget: budget_mb << 20, bytes: 0, stride: 1, count: 0, dirty: false, moved: (0.0, 0.0) }
    }

    pub fn len(&self) -> usize {
        self.kfs.len()
    }

    /// 보관한 장면 바이트(압축 뒤)
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 그래프 블록: 새 노드는 더하고, 있던 노드가 문턱 넘게 움직였으면 다시 쌓기 표시
    pub fn apply_graph(&mut self, upd: &[(u32, M3, V3)]) {
        for (id, r, p) in upd {
            if let Some(old) = self.nodes.insert(*id, (*r, *p)) {
                let (dt, dr) = (norm(&sub(p, &old.1)), angle(r, &old.0));
                if dt > MOVE_T || dr > MOVE_R {
                    self.dirty |= !self.kfs.is_empty();
                    self.moved = (self.moved.0.max(dt), self.moved.1.max(dr));
                }
            }
        }
    }

    /// 노드의 지금 위치(수평 맞춤 전, RTAB-Map 지도 좌표)
    pub fn node(&self, id: u32) -> Option<(M3, V3)> {
        self.nodes.get(&id).copied()
    }

    /// 장면 위치를 기준 노드에서 본 상대 위치로(노드를 모르면 기준 0 + 받은 위치 그대로)
    pub fn rel_of(&self, anchor: u32, r: M3, p: V3) -> (u32, (M3, V3)) {
        match self.nodes.get(&anchor) {
            Some(n) => (anchor, inv_mul(n, &(r, p))),
            None => (0, (r, p)),
        }
    }

    /// 지도에 넣은 장면 보관(받은 위치 r, p 는 수평 맞춤 전, RTAB-Map 지도 좌표). sparse: 줄인 보관(점유 격자·물체만 다시 쌓음)
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, t: f64, anchor: u32, r: M3, p: V3, h_cam: f64, w: usize, h: usize, rgb: &[u8], depth: &[f32], dets: &Option<Vec<Det>>, sparse: bool) -> Option<Arc<Kf>> {
        self.count += 1;
        if (self.count - 1) % self.stride != 0 {
            return None;
        }
        let (anchor, rel) = match self.nodes.get(&anchor) {
            Some(n) => (anchor, inv_mul(n, &(r, p))),
            None => (0, (r, p)),
        };
        let mm16 = |d: f32| (d * 1000.0).round().clamp(0.0, 65535.0) as u16;
        let (rgb_z, depth_z, crops) = if sparse {
            let (qw, qh) = (w.div_ceil(Q), h.div_ceil(Q));
            let q: Vec<u16> = (0..qw * qh).map(|i| mm16(depth[(i / qw) * Q * w + (i % qw) * Q])).collect();
            let crops = dets.iter().flatten().filter_map(|d| {
                let [x1, y1, x2, y2] = d.box_.map(|v| v as usize);
                let (x2, y2) = (x2.min(w), y2.min(h));
                (x2 > x1 && y2 > y1).then(|| {
                    let (cw, ch) = (x2 - x1, y2 - y1);
                    let mut cr = Vec::with_capacity(cw * ch * 3);
                    let mut cd = Vec::with_capacity(cw * ch);
                    for y in y1..y2 {
                        cr.extend_from_slice(&rgb[(y * w + x1) * 3..(y * w + x2) * 3]);
                        cd.extend(depth[y * w + x1..y * w + x2].iter().map(|&v| mm16(v)));
                    }
                    Crop { x: x1, y: y1, w: cw, h: ch, rgb_z: pack_rgb(&cr, cw, ch), depth_z: pack_depth(&cd, cw, ch) }
                })
            }).collect();
            (vec![], pack_depth(&q, qw, qh), crops)
        } else {
            (pack_rgb(rgb, w, h), pack_depth(&depth.iter().map(|&d| mm16(d)).collect::<Vec<_>>(), w, h), vec![])
        };
        let kf = Kf {
            seq: self.count as u64, t, anchor, rel, raw: (r, p), h_cam, w, h, rgb_z, depth_z, crops, sparse,
            dets: dets.as_ref().map(|v| v.iter().map(|d| Det { cls: d.cls, score: d.score, box_: d.box_, mask: d.mask.clone() }).collect()),
            pts: Default::default(),
        };
        let kf = Arc::new(kf);
        self.kfs.push(kf.clone());
        self.bytes = self.kfs.iter().map(|k| k.bytes()).sum(); // 점 격자는 점 작업 스레드가 나중에 채우니 매번 다시 셈
        // 예산을 넘으면 하나 걸러 버리고 앞으로도 두 배 간격으로 보관
        while self.bytes > self.budget && self.kfs.len() > 2 {
            let old = std::mem::take(&mut self.kfs);
            self.bytes = 0;
            for (i, k) in old.into_iter().enumerate() {
                if i % 2 == 0 {
                    self.bytes += k.bytes();
                    self.kfs.push(k);
                }
            }
            self.stride *= 2;
            eprintln!("장면 보관: 예산 {} MB 넘음 → {} 장만 남기고 {} 장마다 보관", self.budget >> 20, self.kfs.len(), self.stride);
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rz(a: f64) -> M3 {
        let (s, c) = a.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn follows_anchor() {
        let mut st = KfStore::new(100);
        st.apply_graph(&[(1, rz(0.0), [0.0, 0.0, 0.0])]);
        // 노드 1 에서 x 로 1 m 앞의 장면
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &[1.0], &None, false);
        assert!(!st.dirty);
        // 노드가 원점에서 90° 돌고 (0, 2, 0) 으로 옮겨짐 → 장면은 (0, 3, 0), 90°
        st.apply_graph(&[(1, rz(std::f64::consts::FRAC_PI_2), [0.0, 2.0, 0.0])]);
        assert!(st.dirty);
        let (r, p) = st.pose(&st.kfs()[0]);
        assert!((p[0] - 0.0).abs() < 1e-9 && (p[1] - 3.0).abs() < 1e-9);
        assert!((angle(&r, &rz(std::f64::consts::FRAC_PI_2))).abs() < 1e-6);
    }

    #[test]
    fn small_move_not_dirty() {
        let mut st = KfStore::new(100);
        st.apply_graph(&[(1, rz(0.0), [0.0, 0.0, 0.0])]);
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &[1.0], &None, false);
        st.apply_graph(&[(1, rz(0.001), [0.005, 0.0, 0.0])]);
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
        eprintln!("컬러 {} → {} B, 깊이 {} → {} B, 묶기 {:?}, 풀기 {:?}", rgb.len(), zr.len(), d.len() * 2, zd.len(), tp, t.elapsed());
    }

    #[test]
    fn sparse_keeps_grid_and_boxes() {
        // 줄인 보관: 점유 격자 화소(8 + 16k)와 검출 상자 안은 받은 값 그대로, 나머지 깊이는 4 화소 간격 값
        let (w, h) = (64, 48);
        let d: Vec<f32> = (0..w * h).map(|i| 0.5 + (i % 97) as f32 * 0.01).collect();
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let det = Det { cls: 1, score: 0.9, box_: [10, 5, 30, 21], mask: vec![true; 20 * 16] };
        let mut st = KfStore::new(100);
        st.push(0.0, 0, rz(0.0), [0.0; 3], 1.0, w, h, &rgb, &d, &Some(vec![det]), true);
        let kf = &st.kfs()[0];
        let (d2, c2) = (kf.depth(), kf.rgb());
        let mm = |v: f32| (v * 1000.0).round() * 1e-3;
        for v in (8..h).step_by(16) {
            for u in (8..w).step_by(16) {
                assert!((d2[v * w + u] - mm(d[v * w + u])).abs() < 1e-6);
            }
        }
        for v in 5..21 {
            for u in 10..30 {
                assert!((d2[v * w + u] - mm(d[v * w + u])).abs() < 1e-6);
                assert_eq!(c2[(v * w + u) * 3..(v * w + u) * 3 + 3], rgb[(v * w + u) * 3..(v * w + u) * 3 + 3]);
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
        st.push(0.0, 0, rz(0.0), [0.0; 3], 1.0, w, h, &vec![0; w * h * 3], &d, &None, true);
        let kf = &st.kfs()[0];
        let (qw, qh) = (w.div_ceil(Q), h.div_ceil(Q));
        let q = unpack_depth(&kf.depth_z, qw, qh);
        let old: Vec<f32> = (0..w * h).map(|i| q[(i / w / Q) * qw + (i % w) / Q] as f32 * 1e-3).collect();
        assert_eq!(kf.depth().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), old.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
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
            st.push(i as f64, 0, rz(0.0), [0.0; 3], 1.0, 320, 300, &rgb, &d, &None, false);
        }
        assert!(st.bytes <= 1 << 20);
        assert!(st.stride > 1);
        assert!(st.kfs().windows(2).all(|w| w[0].t < w[1].t));
    }
}

/// 컬러 압축: 채널마다 면으로 모으고 줄 안에서 왼쪽 화소와의 차(u8 감싸기) → LZ4. 손실 없음.
pub fn pack_rgb(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
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
}

pub fn unpack_rgb(z: &[u8], w: usize, h: usize) -> Vec<u8> {
    let b = lz4_flex::decompress_size_prepended(z).expect("보관 컬러 풀기");
    let n = w * h;
    let mut rgb = vec![0u8; n * 3];
    for c in 0..3 {
        for y in 0..h {
            let mut prev = 0u8;
            for x in 0..w {
                prev = prev.wrapping_add(b[c * n + y * w + x]);
                rgb[(y * w + x) * 3 + c] = prev;
            }
        }
    }
    rgb
}

/// 깊이 압축: 줄 안에서 왼쪽 화소와의 차(u16 감싸기)를 아래 바이트 면·위 바이트 면으로 → LZ4. 손실 없음.
pub fn pack_depth(d: &[u16], w: usize, h: usize) -> Vec<u8> {
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
}

pub fn unpack_depth(z: &[u8], w: usize, h: usize) -> Vec<u16> {
    let b = lz4_flex::decompress_size_prepended(z).expect("보관 깊이 풀기");
    let n = w * h;
    let mut d = vec![0u16; n];
    for y in 0..h {
        let mut prev = 0u16;
        for x in 0..w {
            let i = y * w + x;
            prev = prev.wrapping_add(b[i] as u16 | (b[n + i] as u16) << 8);
            d[i] = prev;
        }
    }
    d
}
