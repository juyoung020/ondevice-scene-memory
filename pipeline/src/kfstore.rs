//! 지도에 넣은 장면 보관 + RTAB-Map 노드 위치: 루프 클로저로 노드가 움직이면 보관한 장면을 고친 위치로 다시 쌓는다.
//! 장면 위치 = 기준 노드의 지금 위치 × (받을 때 기준 노드에서 본 상대 위치). 노드 위치는 스트림 그래프 블록(MMG1)으로 온다.
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
    rgb_z: Vec<u8>,   // 컬러: 채널별 면 + 왼쪽 화소와의 차 → LZ4
    depth_z: Vec<u8>, // 깊이 mm: 왼쪽 화소와의 차 → 아래·위 바이트 면 → LZ4
    pub dets: Option<Vec<Det>>,
}

impl Kf {
    pub fn depth(&self) -> Vec<f32> {
        unpack_depth(&self.depth_z, self.w, self.h).iter().map(|&v| v as f32 * 1e-3).collect()
    }
    pub fn rgb(&self) -> Vec<u8> {
        unpack_rgb(&self.rgb_z, self.w, self.h)
    }
    fn bytes(&self) -> usize {
        self.rgb_z.len() + self.depth_z.len() + self.dets.iter().flatten().map(|d| d.mask.len()).sum::<usize>()
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

    /// 지도에 넣은 장면 보관(받은 위치 r, p 는 수평 맞춤 전, RTAB-Map 지도 좌표)
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, t: f64, anchor: u32, r: M3, p: V3, h_cam: f64, w: usize, h: usize, rgb: &[u8], depth: &[f32], dets: &Option<Vec<Det>>) {
        self.count += 1;
        if (self.count - 1) % self.stride != 0 {
            return;
        }
        let (anchor, rel) = match self.nodes.get(&anchor) {
            Some(n) => (anchor, inv_mul(n, &(r, p))),
            None => (0, (r, p)),
        };
        let kf = Kf {
            seq: self.count as u64, t, anchor, rel, raw: (r, p), h_cam, w, h, rgb_z: pack_rgb(rgb, w, h),
            depth_z: pack_depth(&depth.iter().map(|&d| (d * 1000.0).round().clamp(0.0, 65535.0) as u16).collect::<Vec<_>>(), w, h),
            dets: dets.as_ref().map(|v| v.iter().map(|d| Det { cls: d.cls, score: d.score, box_: d.box_, mask: d.mask.clone() }).collect()),
        };
        self.bytes += kf.bytes();
        self.kfs.push(Arc::new(kf));
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
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &[1.0], &None);
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
        st.push(0.0, 1, rz(0.0), [1.0, 0.0, 0.0], 1.0, 1, 1, &[0, 0, 0], &[1.0], &None);
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
    fn budget_thins() {
        let mut st = KfStore::new(1); // 1 MB
        // 압축이 안 되는 값(의사 난수)
        let mut s = 1u32;
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 24) as u8 };
        let rgb: Vec<u8> = (0..320 * 300 * 3).map(|_| rnd()).collect();
        let d: Vec<f32> = (0..320 * 300).map(|_| rnd() as f32 * 0.1).collect();
        for i in 0..10 {
            st.push(i as f64, 0, rz(0.0), [0.0; 3], 1.0, 320, 300, &rgb, &d, &None);
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
