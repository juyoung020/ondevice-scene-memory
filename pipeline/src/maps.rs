//! 누적 지도들. 키프레임마다 새로 들어온 점만 해시 칸에 더하고(같은 키프레임은 칸당 한 번),
//! 파일로 쓸 때는 칸을 한 번 훑기만 한다 — 지금까지 쌓인 점을 다시 정렬하지 않는다.
use crate::geom::V3;
use std::collections::{HashMap, HashSet};

/// 칸 좌표 → 키(축마다 21 비트, ±1 백만 칸)
#[inline]
pub fn key(p: &V3, vox: f64) -> u64 {
    let f = |v: f64| (((v / vox).floor() as i64 + (1 << 20)) as u64) & 0x1F_FFFF;
    (f(p[0]) << 42) | (f(p[1]) << 21) | f(p[2])
}
#[inline]
pub fn coarse_of(k: u64, ratio: i64) -> u64 {
    let g = |s: u32| {
        let v = ((k >> s) & 0x1F_FFFF) as i64 - (1 << 20);
        ((v.div_euclid(ratio) + (1 << 20)) as u64) & 0x1F_FFFF
    };
    (g(42) << 42) | (g(21) << 21) | g(0)
}

// ------------------------------------------------------------------ 장면 3D 점 지도

#[derive(Clone, Copy)]
struct SV {
    sum: [f32; 3],
    n: u32,
    col: [u8; 3],
    best: f32, // 이 칸을 본 가장 가까운 거리(색은 그 관측 것)
}

pub struct SceneMap {
    pub vox: f64,
    ratio: i64, // 지지 칸 = vox × ratio
    cells: HashMap<u64, SV>,
    support: HashMap<u64, (u32, u32)>, // 굵은 칸 → (본 키프레임 수, 마지막 키프레임)
    pub min_frames: u32,
}

impl SceneMap {
    pub fn new(vox: f64, sup: f64) -> SceneMap {
        SceneMap { vox, ratio: (sup / vox).round() as i64, cells: HashMap::new(), support: HashMap::new(), min_frames: 2 }
    }
    /// 한 키프레임의 점: (세계 점, 색, 카메라 거리)
    pub fn add(&mut self, kf: u32, pts: &[(V3, [u8; 3], f32)]) {
        for (p, c, d) in pts {
            let k = key(p, self.vox);
            let e = self.cells.entry(k).or_insert(SV { sum: [0.0; 3], n: 0, col: *c, best: f32::MAX });
            e.sum[0] += p[0] as f32;
            e.sum[1] += p[1] as f32;
            e.sum[2] += p[2] as f32;
            e.n += 1;
            if *d < e.best {
                e.best = *d;
                e.col = *c;
            }
            let s = self.support.entry(coarse_of(k, self.ratio)).or_insert((0, u32::MAX));
            if s.1 != kf {
                s.0 += 1;
                s.1 = kf;
            }
        }
    }
    /// 여러 키프레임에서 본 칸만: (점, 색)
    pub fn points(&self) -> Vec<([f32; 3], [u8; 3])> {
        self.cells
            .iter()
            .filter(|(k, _)| self.support.get(&coarse_of(**k, self.ratio)).map_or(0, |s| s.0) >= self.min_frames)
            .map(|(_, v)| ([v.sum[0] / v.n as f32, v.sum[1] / v.n as f32, v.sum[2] / v.n as f32], v.col))
            .collect()
    }
}

// ------------------------------------------------------------------ 2D 점유 격자

pub struct Grid {
    pub res: f64,
    cells: HashMap<(i32, i32), (u32, u32)>, // (막힘으로 본 키프레임 수, 빔으로 본 키프레임 수)
}

pub struct GridImg {
    pub w: usize,
    pub h: usize,
    pub x0: i32,
    pub y0: i32,
    pub px: Vec<u8>, // 254 빔, 0 막힘, 205 모름; 위가 +y
}

impl Grid {
    pub fn new(res: f64) -> Grid {
        Grid { res, cells: HashMap::new() }
    }
    pub fn add(&mut self, cam: (f64, f64), occ: &[(f64, f64)], floor: &[(f64, f64)]) {
        let c = |x: f64, y: f64| ((x / self.res).floor() as i32, (y / self.res).floor() as i32);
        let o: HashSet<(i32, i32)> = occ.iter().map(|p| c(p.0, p.1)).collect();
        let mut f: HashSet<(i32, i32)> = floor.iter().map(|p| c(p.0, p.1)).collect();
        let (cx, cy) = c(cam.0, cam.1);
        for &(x, y) in &o {
            let n = (x - cx).abs().max((y - cy).abs());
            for k in 0..(n - 1).max(0) {
                f.insert((cx + (x - cx) * k / n, cy + (y - cy) * k / n));
            }
        }
        for k in &o {
            self.cells.entry(*k).or_insert((0, 0)).0 += 1;
        }
        for k in f.difference(&o) {
            self.cells.entry(*k).or_insert((0, 0)).1 += 1;
        }
    }
    pub fn image(&self) -> Option<GridImg> {
        if self.cells.is_empty() {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for &(x, y) in self.cells.keys() {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (x0, y0, x1, y1) = (x0 - 5, y0 - 5, x1 + 5, y1 + 5);
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let mut px = vec![205u8; w * h];
        for (&(x, y), &(hit, free)) in &self.cells {
            let i = (h - 1 - (y - y0) as usize) * w + (x - x0) as usize;
            if hit >= 2 && hit >= free {
                px[i] = 0;
            } else if free > 0 {
                px[i] = 254;
            }
        }
        Some(GridImg { w, h, x0, y0, px })
    }
}

// ------------------------------------------------------------------ 물체 기억(트래커)

#[derive(Clone)]
pub struct OV {
    pub sum: [f32; 3],
    pub n: u32,
    pub col: [u8; 3],
    pub best: f32,
}

/// 썸네일: w×h RGB, 마스크(0/255), 깊이(mm)
pub struct Crop {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
    pub mask: Vec<u8>,
    pub depth: Vec<u16>,
}

pub struct Obj {
    pub id: u32,
    pub name: String,
    pub votes: HashMap<String, f32>,
    pub score: f32,
    pub n_obs: u32,
    pub first_pos: V3,
    pub pos: V3,
    pub extent: V3,
    pub last_seen: f64,
    pub miss: u32,
    pub state: &'static str,
    pub cells: HashMap<u64, OV>,
    pub support: HashMap<u64, u32>, // 굵은 칸 → 본 관측 수
    pub crop: Option<Crop>, // 가장 크게 본 장면
    pub crop_area: u32,
    pub dirty: bool,
    pub stamp: u32,
}

pub struct Memory {
    pub objs: Vec<Obj>,
    next: u32,
    pub vox: f64,
    pub sup: f64,
}

impl Memory {
    pub fn new(vox: f64, sup: f64) -> Memory {
        Memory { objs: vec![], next: 0, vox, sup }
    }

    /// 같은 이름(살아 있는) 물체 중 굵은 칸 겹침 25 % 이상이거나 중심이 크기에 맞춘 문턱 안이면 같은 물체
    fn find(&self, name: &str, c: &V3, ext: &V3, keys: &HashSet<u64>) -> Option<usize> {
        let mut best = None;
        let mut bs = -1.0;
        for (i, o) in self.objs.iter().enumerate() {
            if o.state == "gone" || o.name != name {
                continue;
            }
            let d = crate::geom::norm(&crate::geom::sub(&o.pos, c));
            if d > 2.0 {
                continue;
            }
            let ov = keys.iter().filter(|k| o.support.contains_key(k)).count() as f64 / keys.len().max(1) as f64;
            let e = [0, 1, 2].map(|k| o.extent[k].max(ext[k]));
            let gate = (0.6 * crate::geom::norm(&e)).max(0.5);
            if ov >= 0.25 || d < gate {
                let s = ov + (1.0 - d / 2.0) * 0.1;
                if s > bs {
                    bs = s;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// 한 관측(마스크 안 세계 점·색, 카메라 거리)을 기억에 더한다. 반환: 물체 번호
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, name: &str, score: f32, pts: &[(V3, [u8; 3])], dist: f32, stamp: f64,
                  crop: Option<Crop>) -> u32 {
        let n = pts.len() as f64;
        let mut c = [0.0; 3];
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for (p, _) in pts {
            for k in 0..3 {
                c[k] += p[k] / n;
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let ext = [0, 1, 2].map(|k| (hi[k] - lo[k]).max(0.05));
        let ckeys: HashSet<u64> = pts.iter().map(|(p, _)| key(p, self.sup)).collect();
        let i = match self.find(name, &c, &ext, &ckeys) {
            Some(i) => i,
            None => {
                self.objs.push(Obj {
                    id: self.next, name: name.into(), votes: HashMap::new(), score: 0.0, n_obs: 0, first_pos: c, pos: c,
                    extent: ext, last_seen: stamp, miss: 0, state: "seen", cells: HashMap::new(), support: HashMap::new(),
                    crop: None, crop_area: 0, dirty: true, stamp: 0,
                });
                self.next += 1;
                self.objs.len() - 1
            }
        };
        let o = &mut self.objs[i];
        for k in &ckeys {
            *o.support.entry(*k).or_insert(0) += 1;
        }
        for (p, col) in pts {
            let e = o.cells.entry(key(p, self.vox)).or_insert(OV { sum: [0.0; 3], n: 0, col: *col, best: f32::MAX });
            e.sum[0] += p[0] as f32;
            e.sum[1] += p[1] as f32;
            e.sum[2] += p[2] as f32;
            e.n += 1;
            if dist < e.best {
                e.best = dist;
                e.col = *col;
            }
        }
        *o.votes.entry(name.into()).or_insert(0.0) += score;
        o.n_obs += 1;
        o.score = o.score.max(score);
        o.last_seen = stamp;
        o.miss = 0;
        o.state = "seen";
        o.dirty = true;
        if o.n_obs <= 2 {
            o.pos = c;
            o.extent = ext;
        }
        if let Some(cr) = crop {
            let a = cr.w * cr.h;
            if a > o.crop_area {
                o.crop_area = a;
                o.crop = Some(cr);
            }
        }
        o.id
    }

    /// 보여 줄 점: 여러 관측에서 지지받은 칸만, 중심에서 크게 벗어난 점 제외. pos·extent 도 다시 잰다.
    pub fn refresh(&mut self, i: usize) -> Vec<([f32; 3], [u8; 3])> {
        let o = &mut self.objs[i];
        let thr = if o.n_obs < 3 { 1 } else { (0.15 * o.n_obs as f64).ceil().clamp(2.0, 10.0) as u32 };
        let sup = self.sup;
        let mut pts: Vec<([f32; 3], [u8; 3])> = o
            .cells
            .values()
            .map(|v| ([v.sum[0] / v.n as f32, v.sum[1] / v.n as f32, v.sum[2] / v.n as f32], v.col))
            .filter(|(p, _)| o.support.get(&key(&[p[0] as f64, p[1] as f64, p[2] as f64], sup)).copied().unwrap_or(0) >= thr)
            .collect();
        if pts.len() < 20 {
            pts = o.cells.values().map(|v| ([v.sum[0] / v.n as f32, v.sum[1] / v.n as f32, v.sum[2] / v.n as f32], v.col)).collect();
        }
        if pts.is_empty() {
            return pts;
        }
        let med = |k: usize| {
            let mut v: Vec<f32> = pts.iter().map(|p| p.0[k]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let m = [med(0), med(1), med(2)];
        let mut ds: Vec<f32> = pts.iter().map(|p| ((p.0[0] - m[0]).powi(2) + (p.0[1] - m[1]).powi(2) + (p.0[2] - m[2]).powi(2)).sqrt()).collect();
        let mut sd = ds.clone();
        sd.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lim = 3.0 * sd[sd.len() / 2] + 0.05;
        let mut keep = vec![];
        for (p, d) in pts.into_iter().zip(ds.drain(..)) {
            if d <= lim {
                keep.push(p);
            }
        }
        let pct = |k: usize, q: f64| {
            let mut v: Vec<f32> = keep.iter().map(|p| p.0[k]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() - 1) as f64 * q) as usize] as f64
        };
        if !keep.is_empty() {
            o.pos = [m[0] as f64, m[1] as f64, m[2] as f64];
            o.extent = [0, 1, 2].map(|k| (pct(k, 0.95) - pct(k, 0.05)).max(0.05));
        }
        o.name = o.votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0.clone()).unwrap_or_default();
        keep
    }

    /// 보여야 할 자리(화면 안, 깊이 일치)에서 연속 8 키프레임 안 보이면 사라짐
    pub fn check_missing(&mut self, seen: &HashSet<u32>, k: [f64; 4], rwc: &crate::geom::M3, pwc: &V3, depth: &[f32], w: usize, h: usize) {
        for o in self.objs.iter_mut() {
            if seen.contains(&o.id) || o.state == "gone" || o.n_obs < 5 {
                continue;
            }
            let d = crate::geom::sub(&o.pos, pwc);
            let pc = [0, 1, 2].map(|j| rwc[0][j] * d[0] + rwc[1][j] * d[1] + rwc[2][j] * d[2]); // Rᵀ·d
            if pc[2] < 0.3 || pc[2] > 4.0 {
                continue;
            }
            let u = k[0] * pc[0] / pc[2] + k[2];
            let v = k[1] * pc[1] / pc[2] + k[3];
            if !(0.1 * w as f64..0.9 * w as f64).contains(&u) || !(0.1 * h as f64..0.9 * h as f64).contains(&v) {
                continue;
            }
            let dz = depth[v as usize * w + u as usize] as f64;
            if dz <= 0.0 || (dz - pc[2]).abs() > 0.3 {
                continue;
            }
            o.miss += 1;
            if o.miss >= 8 {
                o.state = "gone";
            }
        }
    }
}
