//! 누적 지도들. 키프레임마다 새로 들어온 점만 해시 칸에 더하고(같은 키프레임은 칸당 한 번),
//! 파일로 쓸 때는 칸을 한 번 훑기만 한다 — 지금까지 쌓인 점을 다시 정렬하지 않는다.
use crate::geom::V3;
use crate::geom::{FastMap, FastSet};

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
    cells: FastMap<u64, SV>,
    support: FastMap<u64, (u32, u32)>, // 굵은 칸 → (본 키프레임 수, 마지막 키프레임)
    pub min_frames: u32,
    sent: FastSet<u64>, // 이미 조각 파일로 내보낸 칸
}

impl SceneMap {
    pub fn new(vox: f64, sup: f64) -> SceneMap {
        SceneMap { vox, ratio: (sup / vox).round() as i64, cells: FastMap::default(), support: FastMap::default(), min_frames: 2, sent: FastSet::default() }
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
    /// 지난번 이후 새로 확정된 칸만(조각 파일용): 뷰어가 지도 전체를 다시 받지 않게
    pub fn new_points(&mut self) -> Vec<([f32; 3], [u8; 3])> {
        let mut out = vec![];
        for (k, v) in &self.cells {
            if !self.sent.contains(k) && self.support.get(&coarse_of(*k, self.ratio)).map_or(0, |s| s.0) >= self.min_frames {
                out.push(([v.sum[0] / v.n as f32, v.sum[1] / v.n as f32, v.sum[2] / v.n as f32], v.col));
                self.sent.insert(*k);
            }
        }
        out
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
    cells: FastMap<(i32, i32), (u32, u32)>, // (막힘으로 본 키프레임 수, 빔으로 본 키프레임 수)
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
        Grid { res, cells: FastMap::default() }
    }
    pub fn add(&mut self, cam: (f64, f64), occ: &[(f64, f64)], floor: &[(f64, f64)]) {
        let c = |x: f64, y: f64| ((x / self.res).floor() as i32, (y / self.res).floor() as i32);
        let o: FastSet<(i32, i32)> = occ.iter().map(|p| c(p.0, p.1)).collect();
        let mut f: FastSet<(i32, i32)> = floor.iter().map(|p| c(p.0, p.1)).collect();
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
    pub votes: FastMap<String, f32>,
    pub score: f32,
    pub n_obs: u32,
    pub first_pos: V3,
    pub pos: V3,
    pub extent: V3,
    pub last_seen: f64,
    pub miss: u32,
    pub state: &'static str,
    pub cells: FastMap<u64, OV>,
    pub support: FastMap<u64, u32>, // 굵은 칸 → 본 관측 수
    pub crop: Option<Crop>, // 가장 크게 본 장면
    pub crop_area: u32,
    pub crop_ver: u32, // 사진이 바뀔 때마다 +1(뷰어는 이게 바뀔 때만 다시 받음)
    pub dirty: bool,
    pub stamp: u32,
    pub n_pts: usize, // 마지막으로 내보낸 점 수
    pub crop_saved: u32, // 파일로 쓴 사진의 crop_ver(같으면 다시 안 씀)
    pub cells_saved: usize, // 점 파일을 쓸 때의 칸 수(5 % 넘게 늘어야 다시 씀)
}

pub struct Memory {
    pub objs: Vec<Obj>,
    pub next: u32, // 다음 물체 id(다시 쌓을 때 이어서 써 뷰어가 옛 물체 파일과 헷갈리지 않게)
    pub vox: f64,
    pub sup: f64,
    pub merged: u32, // 병합 후처리로 합친 수(누적)
}

impl Memory {
    pub fn new(vox: f64, sup: f64) -> Memory {
        Memory { objs: vec![], next: 0, vox, sup, merged: 0 }
    }

    /// 같은 이름(살아 있는) 물체 중 굵은 칸 겹침 25 % 이상이거나 중심이 크기에 맞춘 문턱 안이면 같은 물체
    /// cam: 관측한 카메라 위치(시선 방향 거리 오차를 따로 봄)
    fn find(&self, name: &str, c: &V3, ext: &V3, keys: &FastSet<u64>, cam: Option<&V3>) -> Option<usize> {
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
            // 딥러닝 깊이는 거리(시선 방향)가 주로 틀린다: 옆 방향은 같은 문턱, 시선 방향은 거리의 25 % 까지 더 허용
            let along_ok = cam.is_some_and(|cp| {
                let ray = crate::geom::sub(c, cp);
                let rn = crate::geom::norm(&ray).max(1e-6);
                let u = ray.map(|x| x / rn);
                let diff = crate::geom::sub(&o.pos, c);
                let along = crate::geom::dot(&diff, &u);
                let lat = crate::geom::norm(&crate::geom::sub(&diff, &u.map(|x| x * along)));
                lat < gate && along.abs() < gate.max(0.25 * rn)
            });
            if ov >= 0.25 || d < gate || along_ok {
                let s = ov + (1.0 - d / 2.0) * 0.1;
                if s > bs {
                    bs = s;
                    best = Some(i);
                }
            }
        }
        best
    }

    /// 한 관측(마스크 안 세계 점·색, 카메라 거리)을 기억에 더한다. 사진(crop)은 넓이(crop_area)가 지금 사진보다 클 때만 만든다. 반환: 물체 번호
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, name: &str, score: f32, pts: &[(V3, [u8; 3])], dist: f32, stamp: f64,
                  crop_area: u32, crop: impl FnOnce() -> Crop, cam: Option<V3>) -> u32 {
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
        let ckeys: FastSet<u64> = pts.iter().map(|(p, _)| key(p, self.sup)).collect();
        let i = match self.find(name, &c, &ext, &ckeys, cam.as_ref()) {
            Some(i) => i,
            None => {
                self.objs.push(Obj {
                    id: self.next, name: name.into(), votes: FastMap::default(), score: 0.0, n_obs: 0, first_pos: c, pos: c,
                    extent: ext, last_seen: stamp, miss: 0, state: "seen", cells: FastMap::default(), support: FastMap::default(),
                    crop: None, crop_area: 0, crop_ver: 0, dirty: true, stamp: 0, n_pts: 0, crop_saved: 0, cells_saved: 0,
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
        if crop_area > o.crop_area {
            o.crop_area = crop_area;
            o.crop = Some(crop());
            o.crop_ver += 1;
        }
        o.id
    }

    /// 병합 후처리(HOV-SG·ConceptGraphs 방식): 따로 생긴 두 물체가 같은 것이면 합친다.
    /// 기하: 굵은 칸(4 cm) 3D IoU, 작은 쪽 기준 겹침 비율. 의미: 이름 투표 분포 코사인. 생김새: 점 색 히스토그램 코사인.
    /// 합치는 조건: IoU ≥ 0.5, 또는 겹침 ≥ 0.6 이면서 (이름 ≥ 0.5 또는 생김새 ≥ 0.8). 반환: 합친 수
    /// 잠깐 잡혔다 사라진 검출 지우기: 두 번 이하로 보였고 3 초 넘게 다시 안 보인 물체(한두 장면의 잘못된 검출이 화면을 어지럽힘)
    pub fn prune(&mut self, now: f64) -> usize {
        let n0 = self.objs.len();
        self.objs.retain(|o| !(o.n_obs <= 2 && now - o.last_seen > 3.0));
        n0 - self.objs.len()
    }

    pub fn merge_pass(&mut self) -> usize {
        let hist = |o: &Obj| -> Vec<f32> {
            let mut h = vec![0f32; 64];
            for v in o.cells.values() {
                h[(v.col[0] as usize >> 6) * 16 + (v.col[1] as usize >> 6) * 4 + (v.col[2] as usize >> 6)] += 1.0;
            }
            let n = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            h.iter().map(|x| x / n).collect()
        };
        let votes_cos = |a: &Obj, b: &Obj| -> f32 {
            let na = a.votes.values().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            let nb = b.votes.values().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            a.votes.iter().map(|(k, v)| v * b.votes.get(k).copied().unwrap_or(0.0)).sum::<f32>() / (na * nb)
        };
        let mut merged = 0;
        loop {
            let alive: Vec<usize> = (0..self.objs.len()).filter(|&i| self.objs[i].state != "gone").collect();
            let mut pair = None;
            'outer: for (ai, &i) in alive.iter().enumerate() {
                for &j in &alive[ai + 1..] {
                    let (a, b) = (&self.objs[i], &self.objs[j]);
                    // 중심이 크기에 비해 아주 멀면 건너뜀(pos·extent 는 점 파일을 쓸 때만 다시 재므로 넉넉하게)
                    if crate::geom::norm(&crate::geom::sub(&a.pos, &b.pos)) > (crate::geom::norm(&a.extent) + crate::geom::norm(&b.extent)) / 2.0 + 0.5 {
                        continue;
                    }
                    let (sa, sb) = if a.support.len() <= b.support.len() { (&a.support, &b.support) } else { (&b.support, &a.support) };
                    let inter = sa.keys().filter(|k| sb.contains_key(k)).count() as f32;
                    if inter == 0.0 {
                        continue;
                    }
                    let iou = inter / ((sa.len() + sb.len()) as f32 - inter).max(1.0);
                    let ovl = inter / sa.len().max(1) as f32;
                    let sem = votes_cos(a, b);
                    let app = if ovl >= 0.6 { hist(a).iter().zip(hist(b)).map(|(x, y)| x * y).sum::<f32>() } else { 0.0 };
                    if iou >= 0.5 || (ovl >= 0.6 && (sem >= 0.5 || app >= 0.8)) {
                        pair = Some((i, j));
                        break 'outer;
                    }
                }
            }
            let Some((i, j)) = pair else { break };
            // 관측이 많은 쪽으로 합침
            let (keep, drop) = if self.objs[i].n_obs >= self.objs[j].n_obs { (i, j) } else { (j, i) };
            let b = self.objs.remove(drop);
            let keep = if drop < keep { keep - 1 } else { keep };
            let a = &mut self.objs[keep];
            for (k, v) in b.cells {
                let e = a.cells.entry(k).or_insert(OV { sum: [0.0; 3], n: 0, col: v.col, best: f32::MAX });
                for c in 0..3 {
                    e.sum[c] += v.sum[c];
                }
                e.n += v.n;
                if v.best < e.best {
                    e.best = v.best;
                    e.col = v.col;
                }
            }
            for (k, v) in b.support {
                *a.support.entry(k).or_insert(0) += v;
            }
            for (k, v) in b.votes {
                *a.votes.entry(k).or_insert(0.0) += v;
            }
            a.n_obs += b.n_obs;
            a.score = a.score.max(b.score);
            if b.last_seen > a.last_seen {
                a.last_seen = b.last_seen;
                a.miss = b.miss;
                a.state = b.state;
            }
            if b.crop_area > a.crop_area {
                a.crop_area = b.crop_area;
                a.crop = b.crop;
                a.crop_ver = a.crop_ver.max(b.crop_ver) + 1;
            }
            a.dirty = true;
            a.cells_saved = 0; // 합쳤으면 점 파일을 다시
            merged += 1;
        }
        self.merged += merged as u32;
        merged
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
        // 중앙값·분위수는 정렬 대신 선택(같은 값, 큰 물체에서 수십 ms 걸리던 것)
        let nth = |mut v: Vec<f32>, i: usize| *v.select_nth_unstable_by(i, |a, b| a.partial_cmp(b).unwrap()).1;
        let med = |k: usize| {
            let v: Vec<f32> = pts.iter().map(|p| p.0[k]).collect();
            let i = v.len() / 2;
            nth(v, i)
        };
        let m = [med(0), med(1), med(2)];
        let mut ds: Vec<f32> = pts.iter().map(|p| ((p.0[0] - m[0]).powi(2) + (p.0[1] - m[1]).powi(2) + (p.0[2] - m[2]).powi(2)).sqrt()).collect();
        let lim = 3.0 * nth(ds.clone(), ds.len() / 2) + 0.05;
        let mut keep = vec![];
        for (p, d) in pts.into_iter().zip(ds.drain(..)) {
            if d <= lim {
                keep.push(p);
            }
        }
        let pct = |k: usize, q: f64| {
            let v: Vec<f32> = keep.iter().map(|p| p.0[k]).collect();
            let i = ((v.len() - 1) as f64 * q) as usize;
            nth(v, i) as f64
        };
        if !keep.is_empty() {
            o.pos = [m[0] as f64, m[1] as f64, m[2] as f64];
            o.extent = [0, 1, 2].map(|k| (pct(k, 0.95) - pct(k, 0.05)).max(0.05));
        }
        o.name = o.votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0.clone()).unwrap_or_default();
        keep
    }

    /// 보여야 할 자리(화면 안, 40 화소 이상, 깊이 일치)에서 그 자리에 아무것도 검출되지 않은 채 15 장 연속이면 사라짐
    pub fn check_missing(&mut self, seen: &std::collections::HashSet<u32>, k: [f64; 4], rwc: &crate::geom::M3, pwc: &V3, depth: &[f32], w: usize, h: usize) {
        let miss_need = 15u32; // 보여야 할 자리에서 15 장 연속 안 보이면 사라짐(8 장은 검출이 잠깐 빠져도 사라짐이 됐음)
        // 이번 장면에 보인 물체 위치: 그 자리에서 다른 물체가 검출됐으면(중복이거나 이름이 바뀌어 검출됨, 예: 곰 인형 → 고양이) 놓친 걸로 세지 않음
        let seen_at: Vec<V3> = self.objs.iter().filter(|o| seen.contains(&o.id)).map(|o| o.pos).collect();
        for o in self.objs.iter_mut() {
            if seen.contains(&o.id) || o.state == "gone" || o.n_obs < 5 {
                continue;
            }
            let near = (0.5 * o.extent.iter().cloned().fold(0.0, f64::max)).max(0.3);
            if seen_at.iter().any(|p| crate::geom::norm(&crate::geom::sub(p, &o.pos)) < near) {
                continue;
            }
            let d = crate::geom::sub(&o.pos, pwc);
            let pc = [0, 1, 2].map(|j| rwc[0][j] * d[0] + rwc[1][j] * d[1] + rwc[2][j] * d[2]); // Rᵀ·d
            if pc[2] < 0.3 || pc[2] > 4.0 {
                continue;
            }
            // 화면에서 40 화소보다 작게 보이면 검출이 원래 자주 빠지므로 세지 않음
            let size_px = o.extent.iter().cloned().fold(0.0, f64::max) * k[0] / pc[2];
            if size_px < 40.0 {
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
            }            o.miss += 1;
            if o.miss >= miss_need {
                o.state = "gone";
            }
        }
    }
}
