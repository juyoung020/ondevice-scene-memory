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

// ------------------------------------------------------------------ 2D 점유 격자

/// 격자 조각 한 변(칸): 칸 값을 조각마다 한 덩어리로(칸마다 해시를 찾지 않음)
const GT: i32 = 64;
/// 칸 값: (막힘으로 본 키프레임 수, 빔으로 본 키프레임 수, 그중 가구보다 높은 막힘을 본 수). 셋 다 0 = 아직 안 본 칸
type GCell = (u32, u32, u32);

#[derive(Clone)]
pub struct Grid {
    pub res: f64,
    tiles: FastMap<(i32, i32), Box<[GCell]>>, // 조각 (x>>6, y>>6) → 64×64 칸(줄 = y)
}

pub struct GridImg {
    pub w: usize,
    pub h: usize,
    pub x0: i32,
    pub y0: i32,
    /// PGM 파일 바이트 그대로(머리 + 화소): 기록할 때 복사 없이 넘김. 화소는 [`GridImg::px`]
    pub pgm: Vec<u8>,
    hdr: usize,
    pub wall: Vec<bool>, // 막힌 칸 중 가구보다 높은 것(벽·문틀·키 큰 장) — 방은 이것으로만 나뉜다
}

impl GridImg {
    /// 화소(254 빔, 0 막힘, 205 모름; 위가 +y)
    pub fn px(&self) -> &[u8] {
        &self.pgm[self.hdr..]
    }
}

impl Grid {
    pub fn new(res: f64) -> Grid {
        Grid { res, tiles: FastMap::default() }
    }
    #[inline]
    fn cell_mut(&mut self, x: i32, y: i32) -> &mut GCell {
        let t = self.tiles.entry((x >> 6, y >> 6)).or_insert_with(|| vec![(0, 0, 0); (GT * GT) as usize].into_boxed_slice());
        &mut t[((y & (GT - 1)) * GT + (x & (GT - 1))) as usize]
    }
    /// occ: (x, y, 가구보다 높은 점인지). 이 장면이 닿는 칸을 작은 판(장면 범위)에 표시한 뒤 표시한 칸만 더한다 —
    /// 칸 집합(해시) 셋과 시선 칸마다의 해시 넣기를 배열 표시로(더하는 값은 칸 집합 때와 같음)
    pub fn add(&mut self, cam: (f64, f64), occ: &[(f64, f64, bool)], floor: &[(f64, f64)]) {
        let c = |x: f64, y: f64| ((x / self.res).floor() as i32, (y / self.res).floor() as i32);
        let (cx, cy) = c(cam.0, cam.1);
        // 칸 번호·표시 판은 스레드에 하나 두고 다시 씀(장면마다 할당 없음)
        thread_local! { static GS: std::cell::Cell<(Vec<(i32, i32, bool)>, Vec<(i32, i32)>, Vec<u8>)> = const { std::cell::Cell::new((Vec::new(), Vec::new(), Vec::new())) }; }
        let (mut oc, mut fc, mut fl) = GS.with(|g| g.take());
        oc.clear();
        oc.extend(occ.iter().map(|p| { let (x, y) = c(p.0, p.1); (x, y, p.2) }));
        fc.clear();
        fc.extend(floor.iter().map(|p| c(p.0, p.1)));
        let (mut x0, mut y0, mut x1, mut y1) = (cx, cy, cx, cy);
        for (x, y) in oc.iter().map(|p| (p.0, p.1)).chain(fc.iter().copied()) {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        let (bw, bh) = ((x1 as i64 - x0 as i64 + 1) as usize, (y1 as i64 - y0 as i64 + 1) as usize);
        if bw * bh > 1 << 24 {
            GS.with(|g| g.set((oc, fc, fl)));
            return self.add_sets(cam, occ, floor); // 장면 범위가 터무니없이 크면(위치가 튐) 칸 집합으로
        }
        const O: u8 = 1;
        const T: u8 = 2;
        const F: u8 = 4;
        fl.clear();
        fl.resize(bw * bh, 0);
        let at = |x: i32, y: i32| (y - y0) as usize * bw + (x - x0) as usize;
        for &(x, y, t) in &oc {
            fl[at(x, y)] |= O | if t { T } else { 0 };
        }
        for &(x, y) in &fc {
            fl[at(x, y)] |= F;
        }
        // 막힌 칸까지의 시선(같은 칸은 한 번만)
        for &(x, y, _) in &oc {
            let i = at(x, y);
            if fl[i] & 8 != 0 {
                continue;
            }
            fl[i] |= 8;
            let n = (x - cx).abs().max((y - cy).abs());
            for k in 0..(n - 1).max(0) {
                fl[at(cx + (x - cx) * k / n, cy + (y - cy) * k / n)] |= F;
            }
        }
        // 줄마다 타일 경계로 나눈 구간: 표시가 하나도 없으면 건너뛰고(비교 고리는 NEON 으로 묶임), 있으면 타일을 한 번만 찾는다
        for yy in 0..bh {
            let y = y0 + yy as i32;
            let row = &fl[yy * bw..(yy + 1) * bw];
            let mut xx = 0usize;
            while xx < bw {
                let x = x0 + xx as i32;
                let end = (bw).min(xx + (GT - (x & (GT - 1))) as usize);
                let seg = &row[xx..end];
                if seg.iter().any(|&f| f & (O | F) != 0) {
                    let t = self.tiles.entry((x >> 6, y >> 6)).or_insert_with(|| vec![(0, 0, 0); (GT * GT) as usize].into_boxed_slice());
                    let base = ((y & (GT - 1)) * GT + (x & (GT - 1))) as usize;
                    for (j, &f) in seg.iter().enumerate() {
                        if f & (O | F) == 0 {
                            continue;
                        }
                        let e = &mut t[base + j];
                        if f & O != 0 {
                            e.0 += 1;
                            if f & T != 0 {
                                e.2 += 1;
                            }
                        } else {
                            e.1 += 1;
                        }
                    }
                }
                xx = end;
            }
        }
        GS.with(|g| g.set((oc, fc, fl)));
    }
    /// add 와 같은 값을 칸 집합으로(장면 범위가 너무 클 때만)
    fn add_sets(&mut self, cam: (f64, f64), occ: &[(f64, f64, bool)], floor: &[(f64, f64)]) {
        let c = |x: f64, y: f64| ((x / self.res).floor() as i32, (y / self.res).floor() as i32);
        let o: FastSet<(i32, i32)> = occ.iter().map(|p| c(p.0, p.1)).collect();
        let tall: FastSet<(i32, i32)> = occ.iter().filter(|p| p.2).map(|p| c(p.0, p.1)).collect();
        let mut f: FastSet<(i32, i32)> = floor.iter().map(|p| c(p.0, p.1)).collect();
        let (cx, cy) = c(cam.0, cam.1);
        for &(x, y) in &o {
            let n = (x - cx).abs().max((y - cy).abs());
            for k in 0..(n - 1).max(0) {
                f.insert((cx + (x - cx) * k / n, cy + (y - cy) * k / n));
            }
        }
        for k in &o {
            self.cell_mut(k.0, k.1).0 += 1;
        }
        for k in &tall {
            self.cell_mut(k.0, k.1).2 += 1;
        }
        for k in f.difference(&o) {
            self.cell_mut(k.0, k.1).1 += 1;
        }
    }
    /// 본 칸 모두: ((x, y), 값)
    fn cells(&self) -> impl Iterator<Item = ((i32, i32), GCell)> + '_ {
        self.tiles.iter().flat_map(|(&(tx, ty), t)| {
            t.iter().enumerate().filter(|(_, v)| **v != (0, 0, 0)).map(move |(i, &v)| ((tx * GT + i as i32 % GT, ty * GT + i as i32 / GT), v))
        })
    }
    pub fn image(&self) -> Option<GridImg> {
        if self.tiles.is_empty() {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for ((x, y), _) in self.cells() {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (x0, y0, x1, y1) = (x0 - 5, y0 - 5, x1 + 5, y1 + 5);
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let mut pgm = Vec::with_capacity(24 + w * h);
        std::io::Write::write_fmt(&mut pgm, format_args!("P5\n{w} {h}\n255\n")).unwrap();
        let hdr = pgm.len();
        pgm.resize(hdr + w * h, 205);
        let px = &mut pgm[hdr..];
        let mut wall = vec![false; w * h];
        for ((x, y), (hit, free, tall)) in self.cells() {
            let i = (h - 1 - (y - y0) as usize) * w + (x - x0) as usize;
            if hit >= 2 && hit >= free {
                px[i] = 0;
                wall[i] = tall >= 2;
            } else if free > 0 {
                px[i] = 254;
            }
        }
        Some(GridImg { w, h, x0, y0, pgm, hdr, wall })
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
/// 사진 벡터로 합치는 문턱(fr1_room 실측, docs/performance_log.md): 같은 물체의 두 이름 0.92, 의자와 그 위 인형 0.85, 겹친 다른 물체 0.74–0.82
/// 약한 물체: 이보다 적게 보였고 분할 점수가 이보다 낮으면 3 초 안 보일 때 지움
pub const WEAK_OBS: u32 = 6;
pub const WEAK_SCORE: f32 = 0.6;
pub const LOOK_COS: f32 = 0.88;
/// 상자 겹침으로 합칠 때의 사진 코사인
pub const BOX_COS: f32 = 0.9;
/// 두 물체 상자(중심 ± 크기/2)의 겹친 부피 / 작은 상자 부피
fn box_overlap(a: &Obj, b: &Obj) -> f64 {
    let mut inter = 1.0;
    for k in 0..3 {
        let lo = (a.pos[k] - a.extent[k] / 2.0).max(b.pos[k] - b.extent[k] / 2.0);
        let hi = (a.pos[k] + a.extent[k] / 2.0).min(b.pos[k] + b.extent[k] / 2.0);
        inter *= (hi - lo).max(0.0);
    }
    let vol = |o: &Obj| o.extent.iter().map(|x| x.max(0.01)).product::<f64>();
    inter / vol(a).min(vol(b)).max(1e-9)
}
/// 작은 쪽 굳은 칸 중 큰 쪽 굳은 칸과 겹쳐야 하는 비율(같은 물체 0.33, 나란한 모니터 둘 0.14)
pub const LOOK_OVL: f32 = 0.25;

#[derive(Clone)]
pub struct Crop {
    pub w: u32,
    pub h: u32,
    pub rgb: std::sync::Arc<Vec<u8>>, // 기록 스레드·사진 벡터와 복사 없이 나눠 씀
    pub mask: std::sync::Arc<Vec<u8>>, // 기록 스레드·사진 벡터와 복사 없이 나눠 씀
    pub depth: std::sync::Arc<Vec<u16>>, // 기록 스레드·사진 벡터와 복사 없이 나눠 씀
}

#[derive(Clone)]
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
    pub last_seg: u32, // 마지막으로 본 때의 Memory::n_seg
    /// 병합 검사가 필요함: 점 · 상태 · 사진 벡터 · 위치 · 이름이 바뀐 물체. 병합 단계는 이 물체와 그 이웃만 본다
    mdirty: bool,
    /// 굳은 칸(여러 관측이 지지한 굵은 칸)과 색 히스토그램: 처음 쓸 때 만들고 점이 바뀌면 비운다
    solid: std::sync::OnceLock<FastSet<u64>>,
    hist: std::sync::OnceLock<[f32; 64]>,
    /// 사진 벡터: 그 물체 사진 벡터들의 평균 방향(길이 1). 사진 벡터 스레드가 만든 것을 나눠 씀(복사 없음)
    emb: Option<std::sync::Arc<[f32]>>,
    /// 중심 칸(Memory::centers 의 키)
    gcell: (i32, i32, i32),
}

impl Obj {
    /// 굳은 칸 문턱(refresh 가 보여 주는 점과 같은 문턱): 가장자리 번짐 · 한두 장면 오검출 칸을 뺀 물체 몸통
    fn solid_thr(&self) -> u32 {
        if self.n_obs < 3 { 1 } else { (0.15 * self.n_obs as f64).ceil().clamp(2.0, 10.0) as u32 }
    }
    fn solid(&self) -> &FastSet<u64> {
        self.solid.get_or_init(|| {
            let t = self.solid_thr();
            self.support.iter().filter(|(_, n)| **n >= t).map(|(k, _)| *k).collect()
        })
    }
    /// 점 색 히스토그램(64 칸, 길이 1)
    fn hist(&self) -> &[f32; 64] {
        self.hist.get_or_init(|| {
            let mut h = [0f32; 64];
            for v in self.cells.values() {
                h[(v.col[0] as usize >> 6) * 16 + (v.col[1] as usize >> 6) * 4 + (v.col[2] as usize >> 6)] += 1.0;
            }
            let n = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            h.map(|x| x / n)
        })
    }
    /// 점 · 관측 수가 바뀜: 캐시를 비우고 병합 검사 대상으로
    fn changed(&mut self) {
        self.solid = std::sync::OnceLock::new();
        self.hist = std::sync::OnceLock::new();
        self.mdirty = true;
    }
}

/// 굵은 칸 → 그 칸을 지지하는 물체 id. 값 하나 = id 하나(위 비트 0) 또는 여러 id 목록 자리(위 비트 1):
/// 칸 대부분은 물체 하나만 지지하므로 목록 할당 없이 u32 하나. 표는 칸 좌표 아래 비트로 64 개로 나눔:
/// 표가 두 배로 커질 때 옮기는 칸이 1/64 라 관측 한 번이 오래 멈추지 않는다(한 표일 때 물체 1,000 개에서 최대 8 ms)
#[derive(Clone)]
struct KeyIndex {
    one: Vec<FastMap<u64, u32>>,
    many: Vec<Vec<u32>>,
    free: Vec<u32>,
}

const MANY: u32 = 1 << 31;

/// 칸 키 → 표 번호(x · y · z 아래 2 비트씩)
#[inline]
fn shard(k: u64) -> usize {
    ((k & 3) | ((k >> 21) & 3) << 2 | ((k >> 42) & 3) << 4) as usize
}

impl Default for KeyIndex {
    fn default() -> Self {
        KeyIndex { one: (0..64).map(|_| FastMap::default()).collect(), many: vec![], free: vec![] }
    }
}

impl KeyIndex {
    /// id 가 아직 이 칸에 없을 때만 부른다
    fn add(&mut self, k: u64, id: u32) {
        use std::collections::hash_map::Entry;
        match self.one[shard(k)].entry(k) {
            Entry::Vacant(e) => {
                e.insert(id);
            }
            Entry::Occupied(mut e) => {
                let v = *e.get();
                if v & MANY != 0 {
                    self.many[(v & !MANY) as usize].push(id);
                } else {
                    let l = vec![v, id];
                    let s = match self.free.pop() {
                        Some(s) => {
                            self.many[s as usize] = l;
                            s
                        }
                        None => {
                            self.many.push(l);
                            (self.many.len() - 1) as u32
                        }
                    };
                    e.insert(s | MANY);
                }
            }
        }
    }
    fn remove(&mut self, k: u64, id: u32) {
        let Some(v) = self.one[shard(k)].get_mut(&k) else { return };
        if *v & MANY == 0 {
            if *v == id {
                self.one[shard(k)].remove(&k);
            }
            return;
        }
        let s = (*v & !MANY) as usize;
        let l = &mut self.many[s];
        if let Some(p) = l.iter().position(|&x| x == id) {
            l.swap_remove(p);
        }
        if l.len() == 1 {
            *v = l[0];
            l.clear();
            self.free.push(s as u32);
        }
    }
    #[inline]
    fn each(&self, k: u64, mut f: impl FnMut(u32)) {
        if let Some(&v) = self.one[shard(k)].get(&k) {
            if v & MANY == 0 {
                f(v)
            } else {
                self.many[(v & !MANY) as usize].iter().for_each(|&x| f(x))
            }
        }
    }
}

/// 중심 칸 한 변(m)
const CG: f64 = 0.5;
fn gcell(p: &V3) -> (i32, i32, i32) {
    ((p[0] / CG).floor() as i32, (p[1] / CG).floor() as i32, (p[2] / CG).floor() as i32)
}

/// 중심 0.5 m 칸 → 그 칸에 중심이 있는 물체 id
#[derive(Clone, Default)]
struct CenterGrid(FastMap<(i32, i32, i32), Vec<u32>>);

impl CenterGrid {
    fn add(&mut self, c: (i32, i32, i32), id: u32) {
        self.0.entry(c).or_default().push(id);
    }
    fn remove(&mut self, c: (i32, i32, i32), id: u32) {
        if let Some(v) = self.0.get_mut(&c) {
            if let Some(p) = v.iter().position(|&x| x == id) {
                v.swap_remove(p);
            }
            if v.is_empty() {
                self.0.remove(&c);
            }
        }
    }
    /// 중심이 p 에서 r 안일 수 있는 물체(칸 단위로 넉넉하게). 볼 칸이 칸 표보다 많으면 표를 통째로
    fn near(&self, p: &V3, r: f64, mut f: impl FnMut(u32)) {
        let lo = [0, 1, 2].map(|k| ((p[k] - r) / CG).floor() as i64);
        let hi = [0, 1, 2].map(|k| ((p[k] + r) / CG).floor() as i64);
        let n: i64 = (0..3).map(|k| hi[k] - lo[k] + 1).product();
        if n as usize > self.0.len() {
            for (c, v) in &self.0 {
                if (0..3).all(|k| { let x = [c.0, c.1, c.2][k] as i64; lo[k] <= x && x <= hi[k] }) {
                    v.iter().for_each(|&id| f(id));
                }
            }
            return;
        }
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    if let Some(v) = self.0.get(&(x as i32, y as i32, z as i32)) {
                        v.iter().for_each(|&id| f(id));
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Memory {
    pub objs: Vec<Obj>,
    pub next: u32, // 다음 물체 id(다시 쌓을 때 이어서 써 뷰어가 옛 물체 파일과 헷갈리지 않게)
    pub vox: f64,
    pub sup: f64,
    pub merged: u32, // 병합 후처리로 합친 수(누적)
    pub n_seg: u32, // 분할을 돌린 장면 수(분할을 건너뛴 장면은 세지 않음)
    /// 굵은 칸 → 그 칸을 지지하는 물체(관측 · 병합이 겹치는 물체를 칸 수만큼만 찾아 셈)
    index: KeyIndex,
    /// 중심 칸 → 물체(중심 거리 규칙의 후보)
    centers: CenterGrid,
    /// 물체 id → objs 자리
    at: FastMap<u32, u32>,
    /// 사진 벡터 스레드에서 마지막으로 가져온 순번(embed::fresh_since)
    emb_gen: u64,
}

impl Memory {
    pub fn new(vox: f64, sup: f64) -> Memory {
        Memory { objs: vec![], next: 0, vox, sup, merged: 0, n_seg: 0, index: KeyIndex::default(), centers: CenterGrid::default(), at: FastMap::default(), emb_gen: 0 }
    }

    /// objs 자리가 바뀐 뒤(지우기) id → 자리 표를 다시
    fn reseat(&mut self) {
        self.at.clear();
        for (i, o) in self.objs.iter().enumerate() {
            self.at.insert(o.id, i as u32);
        }
    }

    /// 칸 색인 · 중심 칸 · 자리 표를 처음부터(물체 id 가 바뀐 뒤)
    fn reindex(&mut self) {
        let Memory { objs, index, centers, .. } = self;
        *index = KeyIndex::default();
        *centers = CenterGrid::default();
        for o in objs.iter_mut() {
            for k in o.support.keys() {
                index.add(*k, o.id);
            }
            o.gcell = gcell(&o.pos);
            centers.add(o.gcell, o.id);
        }
        self.reseat();
    }

    /// 관측 칸과 겹친 물체 → 겹친 칸 수: 칸 색인으로 관측 칸 수만큼만 찾음
    fn overlaps(&self, keys: &FastSet<u64>) -> FastMap<u32, u32> {
        let mut cnt: FastMap<u32, u32> = FastMap::default();
        for k in keys {
            self.index.each(*k, |id| *cnt.entry(id).or_insert(0) += 1);
        }
        cnt
    }

    /// 물체 i 의 중심이 바뀜: 중심 칸 옮김
    fn moved(&mut self, i: usize) {
        let o = &mut self.objs[i];
        let c = gcell(&o.pos);
        if c != o.gcell {
            self.centers.remove(o.gcell, o.id);
            o.gcell = c;
            self.centers.add(c, o.id);
        }
    }

    /// 같은 이름(살아 있는) 물체 중 굵은 칸 겹침 25 % 이상이거나 중심이 크기에 맞춘 문턱 안이면 같은 물체.
    /// cam: 관측한 카메라 위치(시선 방향 거리 오차를 따로 봄). cnt: 관측 칸과 겹친 물체 id → 겹친 칸 수(칸 색인으로 센 것).
    /// 후보는 겹친 물체와 중심이 문턱 안일 수 있는 물체(중심 칸)뿐 — 모든 물체를 돌지 않는다. 고르는 규칙 · 점수 · 같은 점수일 때
    /// 앞 자리를 고르는 것은 모두 돌던 때와 같다
    fn find(&self, name: &str, c: &V3, ext: &V3, nkeys: usize, cnt: &FastMap<u32, u32>, cam: Option<&V3>) -> Option<usize> {
        // 겹침 없이 같은 물체로 볼 수 있는 가장 먼 중심 거리
        let r = if name == "object" {
            let g0 = 0.5 * crate::geom::norm(ext);
            match cam {
                Some(cp) => {
                    let rn = crate::geom::norm(&crate::geom::sub(c, cp)).max(1e-6);
                    (g0 * g0 + g0.max(0.25 * rn).powi(2)).sqrt()
                }
                None => g0,
            }
            .min(2.0)
        } else {
            2.0
        };
        let mut cand: Vec<(u32, u32)> = cnt.iter().map(|(&id, &n)| (id, n)).collect();
        self.centers.near(c, r, |id| {
            if !cnt.contains_key(&id) {
                cand.push((id, 0));
            }
        });
        let mut best: Option<usize> = None;
        let mut bs = -1.0;
        for (id, n) in cand {
            let i = self.at[&id] as usize;
            let o = &self.objs[i];
            if o.state == "gone" || o.name != name {
                continue;
            }
            let d = crate::geom::norm(&crate::geom::sub(&o.pos, c));
            if d > 2.0 {
                continue;
            }
            let ov = n as f64 / nkeys.max(1) as f64;
            let e = [0, 1, 2].map(|k| o.extent[k].max(ext[k]));
            // 이름이 하나뿐인 분할 모델(object)은 이름으로 옆 물체를 가를 수 없다: 중심 문턱을 둘 중 작은 쪽 크기의 절반으로 좁힌다
            let gate = if name == "object" {
                0.5 * crate::geom::norm(&[0, 1, 2].map(|k| o.extent[k].min(ext[k])))
            } else {
                (0.6 * crate::geom::norm(&e)).max(0.5)
            };
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
                if s > bs || (s == bs && best.is_some_and(|b| i < b)) {
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
        let (c, ext, ckeys) = summarize(pts, self.sup);
        let cnt = self.overlaps(&ckeys);
        let i = match self.find(name, &c, &ext, ckeys.len(), &cnt, cam.as_ref()) {
            Some(i) => i,
            None => {
                let id = self.next;
                let g = gcell(&c);
                self.objs.push(Obj {
                    id, name: name.into(), votes: FastMap::default(), score: 0.0, n_obs: 0, first_pos: c, pos: c,
                    extent: ext, last_seen: stamp, miss: 0, state: "seen", cells: FastMap::default(), support: FastMap::default(),
                    crop: None, crop_area: 0, crop_ver: 0, dirty: true, stamp: 0, n_pts: 0, crop_saved: 0, cells_saved: 0, last_seg: self.n_seg,
                    mdirty: true, solid: Default::default(), hist: Default::default(), emb: None, gcell: g,
                });
                self.centers.add(g, id);
                self.at.insert(id, (self.objs.len() - 1) as u32);
                self.next += 1;
                self.objs.len() - 1
            }
        };
        let ns = self.n_seg;
        let Memory { objs, index, .. } = self;
        let o = &mut objs[i];
        let oid = o.id;
        for k in &ckeys {
            match o.support.entry(*k) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(1);
                    index.add(*k, oid);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => *e.get_mut() += 1,
            }
        }
        // 바로 앞 점과 같은 칸이면 표를 다시 찾지 않고 그 칸에 더함
        let mut cur: Option<(u64, &mut OV)> = None;
        for (p, col) in pts {
            let k = key(p, self.vox);
            if !matches!(&cur, Some((ck, _)) if *ck == k) {
                cur = Some((k, o.cells.entry(k).or_insert(OV { sum: [0.0; 3], n: 0, col: *col, best: f32::MAX })));
            }
            let e = &mut *cur.as_mut().unwrap().1;
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
        o.last_seg = ns;
        o.miss = 0;
        o.state = "seen";
        o.dirty = true;
        o.changed();
        let first = o.n_obs <= 2;
        if first {
            o.pos = c;
            o.extent = ext;
        }
        if crop_area > o.crop_area {
            o.crop_area = crop_area;
            o.crop = Some(crop());
            o.crop_ver += 1;
        }
        if first {
            self.moved(i);
        }
        let i = self.merge_at_obs(i, &cnt, ckeys.len());
        self.objs[i].id
    }

    /// 관측 때 바로 합치기: 이 관측이 다른 물체와도 크게 겹치면(관측 칸의 25 % 이상 = find 가 같은 물체로 보는 겹침) 같은 물체를
    /// 둘로 잡았다는 직접 증거 — 주기 병합과 같은 규칙([`same_thing`])을 그 자리에서 보고 맞으면 합친다. 반환: 관측이 든 물체의 자리
    fn merge_at_obs(&mut self, mut i: usize, cnt: &FastMap<u32, u32>, nkeys: usize) -> usize {
        let me = self.objs[i].id;
        let mut cand: Vec<usize> = cnt.iter().filter(|&(&id, &n)| id != me && n as usize * 4 >= nkeys).map(|(id, _)| self.at[id] as usize).collect();
        if cand.is_empty() {
            return i;
        }
        cand.sort_unstable();
        let ids: Vec<u32> = cand.iter().map(|&j| self.objs[j].id).collect();
        for jid in ids {
            let Some(&j) = self.at.get(&jid) else { continue };
            let j = j as usize;
            if j == i {
                continue;
            }
            let (a, b) = (&self.objs[i], &self.objs[j]);
            if !near_enough(a, b) || !same_thing(a, b, inter(a, b), self.sup) {
                continue;
            }
            let (lo, hi) = (i.min(j), i.max(j));
            let (keep, drop) = if self.objs[lo].n_obs >= self.objs[hi].n_obs { (lo, hi) } else { (hi, lo) };
            self.absorb(keep, drop);
            self.objs.remove(drop);
            self.reseat();
            self.merged += 1;
            i = if drop < keep { keep - 1 } else { keep };
        }
        i
    }

    /// 잠깐 잡혔다 사라진 검출 지우기: 두 번 이하로 보였고 3 초 넘게 다시 안 보인 물체(한두 장면의 잘못된 검출이 화면을 어지럽힘)
    /// 다시 쌓은 물체에 옛 물체의 id 를 물려준다(같은 이름, 가장 가까운 것, 크기 절반 + 15 cm 안, 한 번씩만).
    /// 뷰어는 같은 물체를 계속 같은 것으로 보고(선택 유지), 사진이 그대로면 PNG 세 장을 다시 짜지·받지 않는다.
    /// 판 번호(stamp·crop_ver)는 옛 값에서 이어 가 뷰어가 바뀐 파일을 다시 받는다. 짝이 없는 물체는 받은 id 그대로.
    pub fn adopt_ids(&mut self, old: &Memory) -> usize {
        let mut taken = vec![false; old.objs.len()];
        let mut n = 0;
        for o in self.objs.iter_mut() {
            let reach = 0.5 * o.extent.iter().cloned().fold(0.0, f64::max) + 0.15;
            let best = old
                .objs
                .iter()
                .enumerate()
                .filter(|(j, p)| !taken[*j] && p.name == o.name)
                .map(|(j, p)| (j, crate::geom::norm(&crate::geom::sub(&p.pos, &o.pos))))
                .filter(|&(_, d)| d < reach)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let Some((j, _)) = best else { continue };
            taken[j] = true;
            let p = &old.objs[j];
            o.id = p.id;
            o.stamp = p.stamp;
            // 사진은 실시간에 본 것을 이어받는다: 보관 장면은 마스크 안 컬러만 남아(배경 검정) 그 사진으로 사진 벡터 · 이름을 다시 내면 틀린다
            // (fr1_desk: mouse · teddy bear → glove · pillow). 실시간 사진은 보관 장면을 포함한 모든 장면 가운데 가장 크게 본 것이라 더 작지 않다
            if p.crop.is_some() {
                (o.crop, o.crop_area, o.crop_ver, o.crop_saved) = (p.crop.clone(), p.crop_area, p.crop_ver, p.crop_saved);
            } else {
                (o.crop_ver, o.crop_saved) = (p.crop_ver + 1, p.crop_saved);
            }
            // 사진 벡터는 id 에 딸림: 옛 물체의 것을 이어받고 다음 병합에서 이웃과 다시 견줌
            if p.emb.is_some() {
                o.emb = p.emb.clone();
                o.mdirty = true;
            }
            n += 1;
        }
        // 옛 기억이 마지막으로 가져온 뒤의 사진 벡터부터 이어 가져옴
        self.emb_gen = old.emb_gen;
        self.reindex();
        n
    }

    pub fn prune(&mut self, now: f64) -> usize {
        let n0 = self.objs.len();
        // 두 번 이하로 본 물체는 3 초가 지나고 분할한 장면 8 장에서도 안 보였을 때 지움(분할을 건너뛴 장면은 못 본 것으로 세지 않음)
        let ns = self.n_seg;
        // 몇 번 안 보였고 분할 점수도 낮은 물체(책상 모서리 · 뭉개진 조각: 이름을 붙여도 대개 틀림)도 같은 때 지움
        // 이름이 붙은 물체(사진으로 확실한 이름)는 지우지 않음: 몇 번만 본 진짜 물체(로봇 등)를 기억에서 잃지 않게
        let weak = |o: &Obj| o.n_obs <= 2 || (o.n_obs < WEAK_OBS && o.score < WEAK_SCORE && crate::embed::name(o.id, o.name != "object").is_none());
        let del: Vec<bool> = self.objs.iter().map(|o| weak(o) && now - o.last_seen > 3.0 && ns.saturating_sub(o.last_seg) >= 8).collect();
        if !del.contains(&true) {
            return 0;
        }
        for (o, _) in self.objs.iter().zip(&del).filter(|(_, d)| **d) {
            for k in o.support.keys() {
                self.index.remove(*k, o.id);
            }
            self.centers.remove(o.gcell, o.id);
        }
        let mut it = del.iter();
        self.objs.retain(|_| !*it.next().unwrap());
        self.reseat();
        n0 - self.objs.len()
    }

    /// 병합(HOV-SG·ConceptGraphs 방식): 따로 생긴 두 물체가 같은 것이면 합친다. 규칙은 [`same_thing`].
    /// 바뀐 물체(mdirty: 점 추가 · 사진 벡터 도착 · 상태 · 위치 변화)와 그 이웃만 본다 — 안 바뀐 두 물체는 지난번에 안 맞았으면 지금도 안 맞는다.
    /// 반환: 합친 수
    pub fn merge_pass(&mut self) -> usize {
        self.sync_emb();
        self.merge_dirty()
    }

    /// 시험용 merge_pass: emb 가 물체 번호 → 사진 벡터(길이 1). 지금 벡터와 다르면 바꾸고 병합 검사 대상으로
    #[cfg(test)]
    pub fn merge_pass_with(&mut self, emb: &dyn Fn(u32) -> Option<Vec<f32>>) -> usize {
        for o in self.objs.iter_mut() {
            let v = emb(o.id);
            if v.as_deref() != o.emb.as_deref() {
                o.emb = v.map(|v| v.into());
                o.mdirty = true;
            }
        }
        self.merge_dirty()
    }

    /// 사진 벡터 스레드가 지난번 뒤에 새로 낸 벡터(물체마다 평균 방향)를 물체에 넣는다(바뀐 물체만, 잠금 한 번)
    fn sync_emb(&mut self) {
        let (g, fresh) = crate::embed::fresh_since(self.emb_gen);
        self.emb_gen = g;
        for (id, v) in fresh {
            if let Some(&i) = self.at.get(&id) {
                let o = &mut self.objs[i as usize];
                o.emb = Some(v);
                o.mdirty = true;
            }
        }
    }

    /// 바뀐 물체마다 칸 색인으로 칸을 함께 지지하는 물체(겹친 칸 수까지)를, 사진 벡터나 맞닿음 규칙이 있으면 중심 칸으로 가까운 물체를
    /// 후보로 모아 규칙을 본다. 맞는 짝을 모두 모은 뒤 앞 자리 짝부터 서로 겹치지 않는 것을 한꺼번에 합치고, 합친 물체만 다음 판에 다시 본다
    /// (사슬 a–b–c 를 한 번에 묶지 않고 합친 물체로 다시 견줌 = 하나씩 합치던 때와 같은 판정). 지운 물체는 끝에 한 번에 빼며 차례는 그대로
    fn merge_dirty(&mut self) -> usize {
        let n = self.objs.len();
        let mut dead = vec![false; n];
        let mut merged = 0;
        // 중심 거리 미리 거르기의 가장 큰 반경용: 물체 크기(대각선) 최댓값
        let max_en = self.objs.iter().map(|o| crate::geom::norm(&o.extent)).fold(0.0, f64::max);
        let mut round = vec![false; n];
        let mut cnt: FastMap<u32, u32> = FastMap::default();
        loop {
            let dirty: Vec<usize> = (0..n).filter(|&i| !dead[i] && self.objs[i].mdirty).collect();
            if dirty.is_empty() {
                break;
            }
            round.iter_mut().for_each(|r| *r = false);
            for &i in &dirty {
                self.objs[i].mdirty = false;
                round[i] = true;
            }
            let mut pairs: Vec<(usize, usize)> = vec![];
            for &i in &dirty {
                let a = &self.objs[i];
                cnt.clear();
                for k in a.support.keys() {
                    self.index.each(*k, |id| {
                        if id != a.id {
                            *cnt.entry(id).or_insert(0) += 1;
                        }
                    });
                }
                // 칸이 안 겹친 짝은 맞닿음(전형 크기가 있는 이름) · 상자 겹침(사진 벡터)으로만 합친다
                if a.emb.is_some() || typical(&a.name).is_some() {
                    let r = crate::geom::norm(&a.extent) / 2.0 + max_en / 2.0 + 0.5;
                    self.centers.near(&a.pos, r, |id| {
                        if id != a.id {
                            cnt.entry(id).or_insert(0);
                        }
                    });
                }
                for (&id, &inter) in &cnt {
                    let j = self.at[&id] as usize;
                    // 둘 다 바뀐 짝은 앞 자리 쪽에서 한 번만
                    if dead[j] || (round[j] && j < i) {
                        continue;
                    }
                    let b = &self.objs[j];
                    if near_enough(a, b) && same_thing(a, b, inter as usize, self.sup) {
                        pairs.push((i.min(j), i.max(j)));
                    }
                }
            }
            if pairs.is_empty() {
                break;
            }
            pairs.sort_unstable();
            let mut used = vec![false; n];
            for (i, j) in pairs {
                if used[i] || used[j] {
                    continue; // 합친 물체가 다음 판에 다시 견줌
                }
                (used[i], used[j]) = (true, true);
                // 관측이 많은 쪽으로 합침
                let (keep, drop) = if self.objs[i].n_obs >= self.objs[j].n_obs { (i, j) } else { (j, i) };
                self.absorb(keep, drop);
                dead[drop] = true;
                merged += 1;
            }
        }
        if merged > 0 {
            let mut it = dead.iter();
            self.objs.retain(|_| !*it.next().unwrap());
            self.reseat();
        }
        self.merged += merged as u32;
        merged
    }

    /// 물체 drop 을 keep 에 합친다(칸 · 지지 · 이름 투표 · 관측 수 · 상태 · 사진). drop 은 비고 색인 · 중심 칸에서 빠진다(자리는 부르는 쪽이 지움)
    fn absorb(&mut self, keep: usize, drop: usize) {
        let Memory { objs, index, centers, .. } = self;
        let (a, b) = if keep < drop {
            let (l, r) = objs.split_at_mut(drop);
            (&mut l[keep], &mut r[0])
        } else {
            let (l, r) = objs.split_at_mut(keep);
            (&mut r[0], &mut l[drop])
        };
        centers.remove(b.gcell, b.id);
        for (k, v) in std::mem::take(&mut b.cells) {
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
        for (k, v) in std::mem::take(&mut b.support) {
            index.remove(k, b.id);
            match a.support.entry(k) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(v);
                    index.add(k, a.id);
                }
                std::collections::hash_map::Entry::Occupied(mut e) => *e.get_mut() += v,
            }
        }
        for (k, v) in std::mem::take(&mut b.votes) {
            *a.votes.entry(k).or_insert(0.0) += v;
        }
        a.n_obs += b.n_obs;
        a.score = a.score.max(b.score);
        if b.last_seen > a.last_seen {
            a.last_seen = b.last_seen;
            a.last_seg = a.last_seg.max(b.last_seg);
            a.miss = b.miss;
            a.state = b.state;
        }
        if b.crop_area > a.crop_area {
            a.crop_area = b.crop_area;
            a.crop = b.crop.take();
            a.crop_ver = a.crop_ver.max(b.crop_ver) + 1;
        }
        a.dirty = true;
        a.cells_saved = 0; // 합쳤으면 점 파일을 다시
        a.changed();
    }

    /// 보여 줄 점: 여러 관측에서 지지받은 칸만, 중심에서 크게 벗어난 점 제외. pos·extent 도 다시 잰다.
    pub fn refresh(&mut self, i: usize) -> Vec<([f32; 3], [u8; 3])> {
        let o = &mut self.objs[i];
        let thr = o.solid_thr();
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
        let before = (o.pos, o.extent);
        if !keep.is_empty() {
            o.pos = [m[0] as f64, m[1] as f64, m[2] as f64];
            o.extent = [0, 1, 2].map(|k| (pct(k, 0.95) - pct(k, 0.05)).max(0.05));
        }
        let name = o.votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0.clone()).unwrap_or_default();
        // 위치 · 크기 · 이름은 병합 규칙이 보는 값: 바뀌면 다음 병합에서 이웃과 다시 견줌
        if before != (o.pos, o.extent) || name != o.name {
            o.name = name;
            o.mdirty = true;
            self.moved(i);
        }
        keep
    }

    /// 보여야 할 자리(화면 안, 40 화소 이상, 깊이 일치)에서 그 자리에 아무것도 검출되지 않은 채 15 장 연속이면 사라짐
    pub fn check_missing(&mut self, seen: &std::collections::HashSet<u32>, k: [f64; 4], rwc: &crate::geom::M3, pwc: &V3, depth: &crate::depth::Depth, w: usize, h: usize) {
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
            let dz = depth.at(v as usize * w + u as usize) as f64;
            if dz <= 0.0 || (dz - pc[2]).abs() > 0.3 {
                continue;
            }
            o.miss += 1;
            if o.miss >= miss_need {
                o.state = "gone";
                o.mdirty = true; // 사라진 물체가 낀 짝은 규칙이 다름
            }
        }
    }
}


/// 관측 점 → (중심, 크기(축마다 5 cm 이상), 굵은 칸 키)
fn summarize(pts: &[(V3, [u8; 3])], sup: f64) -> (V3, V3, FastSet<u64>) {
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
    // 이웃 점은 대개 같은 굵은 칸: 바로 앞과 같은 키는 다시 넣지 않음(이미 든 키 넣기는 표를 바꾸지 않으니 같은 집합·같은 차례)
    let mut ckeys: FastSet<u64> = FastSet::default();
    let mut last = None;
    for (p, _) in pts {
        let k = key(p, sup);
        if last != Some(k) {
            ckeys.insert(k);
            last = Some(k);
        }
    }
    (c, ext, ckeys)
}

// ------------------------------------------------------------------ 병합 규칙(두 물체가 같은 것인가)

/// 중심이 크기에 비해 아주 멀면 같은 물체가 아니다(pos·extent 는 점 파일을 쓸 때만 다시 재므로 넉넉하게)
fn near_enough(a: &Obj, b: &Obj) -> bool {
    crate::geom::norm(&crate::geom::sub(&a.pos, &b.pos)) <= (crate::geom::norm(&a.extent) + crate::geom::norm(&b.extent)) / 2.0 + 0.5
}

/// 두 물체가 함께 지지한 굵은 칸 수
fn inter(a: &Obj, b: &Obj) -> usize {
    let (s, l) = if a.support.len() <= b.support.len() { (&a.support, &b.support) } else { (&b.support, &a.support) };
    s.keys().filter(|k| l.contains_key(k)).count()
}

fn votes_cos(a: &Obj, b: &Obj) -> f32 {
    let na = a.votes.values().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    let nb = b.votes.values().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    a.votes.iter().map(|(k, v)| v * b.votes.get(k).copied().unwrap_or(0.0)).sum::<f32>() / (na * nb)
}

/// 이름의 전형 크기: 큰 순서 세 변(m), 넉넉하게(수평은 비스듬히 놓인 대각선까지). 모르는 이름은 None(맞닿음 규칙을 쓰지 않음)
fn typical(name: &str) -> Option<[f64; 3]> {
    Some(match name {
        "chair" => [1.2, 0.9, 0.9],
        "couch" => [2.4, 1.1, 1.1],
        "bottle" => [0.4, 0.15, 0.15],
        "cup" => [0.2, 0.15, 0.15],
        "book" => [0.45, 0.35, 0.15],
        "tv" => [1.0, 0.7, 0.3],
        "laptop" => [0.5, 0.4, 0.35],
        "keyboard" => [0.55, 0.25, 0.1],
        "teddy bear" => [0.7, 0.5, 0.5],
        "potted plant" => [1.0, 0.7, 0.7],
        _ => return None,
    })
}

/// 맞닿은 조각: 한 물체가 보는 방향마다 다른 부분(의자 등받이·앉는 판)만 잡혀 칸이 거의 안 겹친 채 둘셋으로 남는다.
/// 같은 이름이고 굳은 칸끼리 맞닿아 있으며(사이 빈 칸 없이 이웃 칸 5 개 이상) 합친 굳은 칸 상자(축마다 5–95 %)가 그 이름의 전형 크기 안이면 같은 물체.
/// 나란한 의자 둘(중심 1.4 m)은 합친 상자가 의자 하나보다 커서 남는다
fn touching(a: &Obj, b: &Obj, sup: f64) -> bool {
    let Some(lim) = typical(&a.name) else { return false };
    let (sa, sb) = (a.solid(), b.solid());
    if a.name != b.name || sa.len() < 3 || sb.len() < 3 {
        return false;
    }
    let unkey = |k: u64| [42u32, 21, 0].map(|s| ((k >> s) & 0x1F_FFFF) as i64 - (1 << 20));
    // 맞닿음: 작은 쪽 굳은 칸 중 큰 쪽 굳은 칸과 같거나 이웃(26 방향)인 칸
    let (s, l) = if sa.len() <= sb.len() { (sa, sb) } else { (sb, sa) };
    let near = s
        .iter()
        .filter(|&&k| {
            let c = unkey(k);
            (-1i64..=1).any(|x| (-1i64..=1).any(|y| (-1i64..=1).any(|z| {
                let p = [(c[0] + x) as f64 + 0.5, (c[1] + y) as f64 + 0.5, (c[2] + z) as f64 + 0.5].map(|v| v * sup);
                l.contains(&key(&p, sup))
            })))
        })
        .count();
    if near < 5 {
        return false;
    }
    let cs: Vec<[i64; 3]> = sa.iter().chain(sb.iter()).map(|&k| unkey(k)).collect();
    let mut ext: Vec<f64> = (0..3)
        .map(|ax| {
            let mut v: Vec<i64> = cs.iter().map(|c| c[ax]).collect();
            v.sort_unstable();
            (v[v.len() * 19 / 20] - v[v.len() / 20] + 1) as f64 * sup
        })
        .collect();
    ext.sort_by(|x, y| y.partial_cmp(x).unwrap());
    (0..3).all(|k| ext[k] <= lim[k])
}

/// 두 사진 벡터의 코사인(길이가 다르면 None)
fn emb_cos(a: &Obj, b: &Obj) -> Option<f32> {
    let (ea, eb) = (a.emb.as_deref()?, b.emb.as_deref()?);
    (ea.len() == eb.len()).then(|| ea.iter().zip(eb).map(|(x, y)| x * y).sum::<f32>())
}

/// 두 물체가 같은 것인가(near_enough 를 거친 짝, inter = 함께 지지한 굵은 칸 수).
/// 기하: 굵은 칸(4 cm) 3D IoU, 작은 쪽 기준 겹침 비율. 의미: 이름 투표 분포 코사인. 생김새: 점 색 히스토그램 코사인.
/// 같음: IoU ≥ 0.5, 또는 겹침 ≥ 0.6 이면서 (이름 ≥ 0.5 또는 생김새 ≥ 0.8), 또는 이름 ≥ 0.5 이고 작은 쪽 굳은 칸 70 % 가 큰 쪽 칸 안(겹침 ≥ 0.2).
/// 사라짐 표시된 물체가 낀 짝은 이름 ≥ 0.5 이고 기하가 맞을 때만(정지 장면에서 잘못 사라짐 처리된 물체 자리에 같은 물체가 새로 생긴 경우).
/// 맞닿은 조각([`touching`])도 같음. 사진 벡터(있을 때): 이름이 달라도 코사인 ≥ LOOK_COS 이고 작은 쪽 굳은 칸의 LOOK_OVL 이상이 큰 쪽 굳은 칸과
/// 겹치면 같음(한 물체가 보는 방향에 따라 다른 이름으로 잡혀 둘로 남은 경우), 칸은 안 겹쳐도 상자가 작은 쪽의 60 % 이상 겹치고
/// 코사인 ≥ BOX_COS 면 같음. 벡터는 종류까지만 가려서(나란한 모니터 둘도 0.92) 겹침 없이 벡터만으로는 합치지 않는다
fn same_thing(a: &Obj, b: &Obj, inter: usize, sup: f64) -> bool {
    let (frag_in, frag_ovl) = (0.7f32, 0.2f32); // 조각: 굳은 칸 70 % 가 상대 칸 안, 전체 칸 겹침 20 % 이상
    // 싼 검사부터(결과는 같음): 전형 크기 있는 같은 이름인지 → 이름 투표, 상자 겹침 → 사진 벡터
    let touch = || a.name == b.name && typical(&a.name).is_some() && votes_cos(a, b) >= 0.5 && touching(a, b, sup);
    let boxed = || a.emb.is_some() && b.emb.is_some() && box_overlap(a, b) >= 0.6 && emb_cos(a, b).is_some_and(|c| c >= BOX_COS);
    if inter == 0 {
        return touch() || boxed();
    }
    let inter = inter as f32;
    let (ns, nl) = (a.support.len().min(b.support.len()), a.support.len().max(b.support.len()));
    let iou = inter / ((ns + nl) as f32 - inter).max(1.0);
    let ovl = inter / ns.max(1) as f32;
    let sem = votes_cos(a, b);
    let app = if ovl >= 0.6 { a.hist().iter().zip(b.hist()).map(|(x, y)| x * y).sum::<f32>() } else { 0.0 };
    let any_gone = a.state == "gone" || b.state == "gone";
    // 조각: 작은 쪽 굳은 칸의 대부분이 큰 쪽이 본 칸 안(같은 이름). 전체 칸 겹침은 번짐 칸이 분모를 키워 낮게 나온다
    let frag = || {
        let (s, l) = if a.support.len() <= b.support.len() { (a, b) } else { (b, a) };
        let sc = s.solid();
        sc.len() >= 3 && sc.iter().filter(|k| l.support.contains_key(k)).count() as f32 >= frag_in * sc.len() as f32
    };
    // 사진 벡터가 닮았고 몸통이 실제로 겹침(이름 무관)
    let look = || {
        let Some(cos) = emb_cos(a, b) else { return false };
        if cos < LOOK_COS {
            return false;
        }
        // 상자가 크게 겹치고 사진이 아주 닮음(같은 물체를 다른 조각으로 두 번 잡음): 작은 상자의 60 % 이상이 겹침
        if cos >= BOX_COS && box_overlap(a, b) >= 0.6 {
            common::info!("merge(box): {}#{} + {}#{} cos {cos:.2}", a.name, a.id, b.name, b.id);
            return true;
        }
        let (ca, cb) = (a.solid(), b.solid());
        let (s, l) = if ca.len() <= cb.len() { (ca, cb) } else { (cb, ca) };
        let hit = s.iter().filter(|k| l.contains(k)).count() as f32;
        let ok = s.len() >= 3 && hit >= LOOK_OVL * s.len() as f32;
        if ok {
            // 다른 조건이 모두 아닐 때만 여기까지 온다: 사진 벡터가 정한 병합
            common::info!("merge(look): {}#{} + {}#{} cos {cos:.2} overlap {:.2}", a.name, a.id, b.name, b.id, hit / s.len() as f32);
        }
        ok
    };
    let geo = iou >= 0.5 || ovl >= 0.6 || (ovl >= frag_ovl && frag());
    (if any_gone { sem >= 0.5 && geo } else { iou >= 0.5 || (ovl >= 0.6 && (sem >= 0.5 || app >= 0.8)) || (sem >= 0.5 && ovl >= frag_ovl && frag()) }) || touch() || look()
}


#[cfg(test)]
#[path = "maps_bench.rs"]
mod bench;

#[cfg(test)]
mod tests {
    use super::*;

    /// x0 부터 nx 칸 × 5 × 5 칸(4 cm) 덩어리 한 번 관측
    fn block(m: &mut Memory, name: &str, x0: i32, nx: i32) -> u32 {
        let mut pts = vec![];
        for x in x0..x0 + nx {
            for y in 0..5 {
                for z in 0..5 {
                    pts.push(([x as f64 * 0.04 + 0.02, y as f64 * 0.04 + 0.02, z as f64 * 0.04 + 0.02], [128u8, 128, 128]));
                }
            }
        }
        m.update(name, 0.9, &pts, 1.0, 0.0, 1, || Crop { w: 1, h: 1, rgb: vec![0; 3].into(), mask: vec![255].into(), depth: vec![0].into() }, None)
    }

    /// 두 벡터의 코사인이 c 가 되게
    fn vecs(a: u32, b: u32, c: f32) -> impl Fn(u32) -> Option<Vec<f32>> {
        move |id| if id == a { Some(vec![1.0, 0.0]) } else if id == b { Some(vec![c, (1.0 - c * c).sqrt()]) } else { None }
    }

    #[test]
    fn look_merges_two_names_of_one_thing() {
        // 이름이 다른 두 덩어리, 작은 쪽의 40 % 가 겹침
        let mut m = Memory::new(0.012, 0.04);
        let (a, b) = (block(&mut m, "teddy bear", 0, 10), block(&mut m, "chair", 8, 5));
        assert_eq!(m.objs.len(), 2);
        assert_eq!(m.merge_pass_with(&|_| None), 0, "벡터가 없으면 그대로");
        assert_eq!(m.merge_pass_with(&vecs(a, b, 0.80)), 0, "덜 닮은 사진은 그대로");
        assert_eq!(m.merge_pass_with(&vecs(a, b, 0.95)), 1);
        assert_eq!(m.objs.len(), 1);
    }

    #[test]
    fn look_alone_does_not_merge_neighbours() {
        // 닮은 사진이어도 몸통이 거의 안 겹치면(작은 쪽의 20 %) 둘로 둔다: 나란한 모니터
        let mut m = Memory::new(0.012, 0.04);
        let (a, b) = (block(&mut m, "tv", 0, 10), block(&mut m, "laptop", 9, 5));
        assert_eq!(m.merge_pass_with(&vecs(a, b, 0.95)), 0);
        assert_eq!(m.objs.len(), 2);
    }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// 상자(중심 c, 변 s) 겉면의 점 n 개, 색은 상자마다 하나
    fn box_pts(r: &mut Rng, c: V3, s: V3, n: usize, col: [u8; 3]) -> Vec<(V3, [u8; 3])> {
        (0..n)
            .map(|_| {
                let ax = (r.f() * 3.0) as usize % 3;
                let mut p = [0.0; 3];
                for k in 0..3 {
                    p[k] = c[k] + s[k] * if k == ax { if r.f() < 0.5 { -0.5 } else { 0.5 } } else { r.f() - 0.5 };
                }
                (p, col)
            })
            .collect()
    }

    /// 예전 find(모든 물체를 차례로 돔): 새 find 가 같은 답을 내는지 견줄 기준
    fn find_ref(m: &Memory, name: &str, c: &V3, ext: &V3, keys: &FastSet<u64>, cam: Option<&V3>) -> Option<usize> {
        let mut best = None;
        let mut bs = -1.0;
        for (i, o) in m.objs.iter().enumerate() {
            if o.state == "gone" || o.name != name {
                continue;
            }
            let d = crate::geom::norm(&crate::geom::sub(&o.pos, c));
            if d > 2.0 {
                continue;
            }
            let ov = keys.iter().filter(|k| o.support.contains_key(k)).count() as f64 / keys.len().max(1) as f64;
            let e = [0, 1, 2].map(|k| o.extent[k].max(ext[k]));
            let gate = if name == "object" { 0.5 * crate::geom::norm(&[0, 1, 2].map(|k| o.extent[k].min(ext[k]))) } else { (0.6 * crate::geom::norm(&e)).max(0.5) };
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

    /// 색인 · 중심 칸 · 자리 표가 물체와 맞는지(처음부터 다시 만든 것과 같은지)
    fn check_index(m: &Memory) {
        let mut want: FastMap<u64, Vec<u32>> = FastMap::default();
        for o in &m.objs {
            for k in o.support.keys() {
                want.entry(*k).or_default().push(o.id);
            }
            assert_eq!(m.at[&o.id] as usize, m.objs.iter().position(|p| p.id == o.id).unwrap());
            assert_eq!(o.gcell, gcell(&o.pos), "O{} 중심 칸", o.id);
            assert!(m.centers.0[&o.gcell].contains(&o.id));
        }
        assert_eq!(m.at.len(), m.objs.len());
        assert_eq!(m.centers.0.values().map(|v| v.len()).sum::<usize>(), m.objs.len());
        assert_eq!(m.index.one.iter().map(|s| s.len()).sum::<usize>(), want.len());
        for (k, v) in &mut want {
            let mut got = vec![];
            m.index.each(*k, |id| got.push(id));
            got.sort_unstable();
            v.sort_unstable();
            assert_eq!(&got, v, "칸 {k:x}");
        }
    }

    /// 흩어진 물체 여럿을 여러 번 보고(일부는 같은 물체를 어긋나게 둘로 잡음 · 일부는 이름이 다름) 사라짐 표시도 섞은 기억
    fn scene(seed: u64, n: usize, names: &[&str]) -> (Memory, Rng) {
        let mut r = Rng(seed);
        let mut m = Memory::new(0.012, 0.04);
        let side = (n as f64).sqrt().ceil() as usize;
        let objs: Vec<(V3, V3, [u8; 3])> = (0..n)
            .map(|i| {
                let c = [(i % side) as f64 * 0.7 + r.f() * 0.2, (i / side) as f64 * 0.7 + r.f() * 0.2, 0.5 + r.f() * 0.3];
                let s = [0.1 + r.f() * 0.4, 0.1 + r.f() * 0.4, 0.1 + r.f() * 0.4];
                (c, s, [(r.f() * 255.0) as u8, (r.f() * 255.0) as u8, (r.f() * 255.0) as u8])
            })
            .collect();
        for t in 0..4 * n {
            let (c, s, col) = objs[(r.f() * n as f64) as usize % n];
            // 가끔 크게 어긋난 관측(같은 물체를 따로 잡음)
            let j = if r.f() < 0.15 { 0.12 } else { 0.02 };
            let c = [c[0] + (r.f() - 0.5) * j, c[1] + (r.f() - 0.5) * j, c[2] + (r.f() - 0.5) * j];
            let name = names[(r.f() * names.len() as f64) as usize % names.len()];
            let pts = box_pts(&mut r, c, s, 600, col);
            m.update(name, 0.5 + r.f() as f32 * 0.5, &pts, 1.0, t as f64 * 0.1, 1, || Crop { w: 1, h: 1, rgb: vec![0; 3].into(), mask: vec![255].into(), depth: vec![0].into() }, None);
            if r.f() < 0.02 {
                let i = (r.f() * m.objs.len() as f64) as usize % m.objs.len();
                m.objs[i].state = "gone";
                m.objs[i].mdirty = true;
            }
        }
        (m, r)
    }

    #[test]
    fn find_same_as_scanning_all() {
        for (seed, names) in [(1u64, &["object"][..]), (2, &["object", "chair", "cup"][..])] {
            let (mut m, mut r) = scene(seed, 40, names);
            check_index(&m);
            for _ in 0..400 {
                let c = [r.f() * 4.5, r.f() * 4.5, 0.5 + r.f() * 0.3];
                let s = [0.05 + r.f() * 0.5, 0.05 + r.f() * 0.5, 0.05 + r.f() * 0.5];
                let pts = box_pts(&mut r, c, s, 400, [9, 9, 9]);
                let (oc, ext, keys) = summarize(&pts, m.sup);
                let cam = [oc[0] + (r.f() - 0.5) * 6.0, oc[1] + (r.f() - 0.5) * 6.0, 1.5];
                let name = names[(r.f() * names.len() as f64) as usize % names.len()];
                let cnt = m.overlaps(&keys);
                for cam in [None, Some(&cam)] {
                    assert_eq!(m.find(name, &oc, &ext, keys.len(), &cnt, cam), find_ref(&m, name, &oc, &ext, &keys, cam));
                }
            }
            // 물체를 바꾼 뒤(병합 · 지우기 · 위치 다시 재기)에도 색인이 맞음
            m.merge_pass_with(&|_| None);
            check_index(&m);
            for i in 0..m.objs.len() {
                m.refresh(i);
            }
            m.prune(1e9);
            check_index(&m);
        }
    }

    #[test]
    fn merge_reaches_fixpoint() {
        // 바뀐 물체만 본 병합이 끝난 뒤 모든 짝을 다 봐도 더 합칠 짝이 없음(사진 벡터 · 사라짐 · 여러 이름 섞음)
        for seed in 3..6u64 {
            let (mut m, _) = scene(seed, 60, &["object", "chair", "bottle"]);
            let emb = |id: u32| Some((0..8).map(|k| if k == (id as usize * 7) % 3 { 1.0 } else { 0.05 }).collect::<Vec<f32>>()).map(|mut v: Vec<f32>| {
                crate::embed::normalize(&mut v);
                v
            });
            m.merge_pass_with(&emb);
            check_index(&m);
            for i in 0..m.objs.len() {
                for j in i + 1..m.objs.len() {
                    let (a, b) = (&m.objs[i], &m.objs[j]);
                    assert!(!(near_enough(a, b) && same_thing(a, b, inter(a, b), m.sup)), "seed {seed}: O{} O{} 남음", a.id, b.id);
                }
            }
            // 바뀐 것이 없으면 다시 돌려도 0
            assert_eq!(m.merge_pass_with(&emb), 0);
        }
    }

    #[test]
    fn observation_spanning_two_objects_merges_them() {
        // 같은 덩어리를 두 번 따로 잡은 두 물체(사라짐 표시로 find 가 하나를 건너뜀) → 둘을 다 덮는 관측이 들어오면 그 자리에서 하나로
        let mut m = Memory::new(0.012, 0.04);
        block(&mut m, "object", 0, 10);
        (m.objs[0].state, m.objs[0].last_seen) = ("gone", -1.0);
        block(&mut m, "object", 0, 10);
        assert_eq!(m.objs.len(), 1, "같은 칸을 다 덮은 관측이 사라진 물체와 바로 합쳐짐");
        assert_eq!(m.objs[0].state, "seen");
        assert_eq!(m.objs[0].n_obs, 2);
        check_index(&m);
    }
}
