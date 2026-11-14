//! 누적 지도들. 키프레임마다 새로 들어온 점만 해시 칸에 더하고(같은 키프레임은 칸당 한 번),
//! 파일로 쓸 때는 칸을 한 번 훑기만 한다 — 지금까지 쌓인 점을 다시 정렬하지 않는다.
use crate::geom::V3;
use crate::geom::{FastMap, FastSet};

/// x / n(정수 나눗셈, x ∈ [−255, 255], n ∈ 1..=16)을 표로: 칸마다 나눗셈 여섯 번이 넣기 시간의 큰 몫이었음
const DIV: [[i16; 511]; 17] = {
    let mut t = [[0i16; 511]; 17];
    let mut n = 1;
    while n < 17 {
        let mut x = 0;
        while x < 511 {
            t[n][x] = ((x as i32 - 255) / n as i32) as i16;
            x += 1;
        }
        n += 1;
    }
    t
};
#[inline]
fn div(x: i32, n: i32) -> i32 {
    DIV[n as usize][(x + 255) as usize] as i32
}

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

// ------------------------------------------------------------------ 장면 3D 점 지도 + 옥트리(LOD)

/// 칸 하나(13 바이트): 칸 안 평균 위치(칸 크기의 1/255 단위), 가까이서 본 색의 평균, 위치·색 평균에 넣은 점 수, 본 키프레임 수·마지막 키프레임, 가장 가까운 거리(mm)
#[derive(Clone, Copy)]
struct SV {
    off: [u8; 3],
    col: [u8; 3],
    n: u8,
    nc: u8,
    nf: u8,
    last: u16,
    best: u16,
}

impl SV {
    const EMPTY: SV = SV { off: [0; 3], col: [0; 3], n: 0, nc: 0, nf: 0, last: 0, best: 0 };
}

/// Potree 식 옥트리: 수준 L 의 점 간격 = vox·2^(DEPTH−L), 노드 한 변 = 간격 × G. 점은 자기 칸(수준 L 간격)이 아직 비어 있는
/// 가장 굵은 수준의 노드 하나에만 들어간다(겹침 없음) → 뷰어는 굵은 노드부터 받아 화면 크기 순으로 더 고운 노드를 더한다.
pub const OCT_DEPTH: u32 = 5;
pub const OCT_G: i64 = 32;
/// 최근 칸 표(hot)에 남기는 키프레임 수
const HOT_AGE: u32 = 24;

pub struct Node {
    keys: Vec<u64>, // 고운 칸 키
    pub ver: u32,
    pub written: usize, // 마지막으로 파일에 쓴 점 수
}

/// 최근 칸 덩어리: 칸 4×4×4 를 한 곳에(장면의 이웃 화소는 대개 같은 덩어리 → 해시는 덩어리가 바뀔 때만 찾고 칸 값이 붙어 있어 캐시를 덜 놓침)
#[derive(Clone)]
struct Brick {
    occ: u64,  // 이 덩어리에 들어 있는 칸
    last: u16, // 마지막으로 본 키프레임(덩어리 통째로 큰 표로 옮길 때를 정함)
    v: [SV; 64],
}

/// 한동안 안 본 덩어리: 든 칸만 차례로(자리 = occ 에서 그 칸 아래 비트 수)
struct ColdBrick {
    occ: u64,
    v: Box<[SV]>,
}

/// 칸 키 → (덩어리 키, 덩어리 안 자리)
#[inline]
fn brick_of(k: u64) -> (u64, usize) {
    let (x, y, z) = ((k >> 42) & 0x1F_FFFF, (k >> 21) & 0x1F_FFFF, k & 0x1F_FFFF);
    (((x >> 2) << 38) | ((y >> 2) << 19) | (z >> 2), ((x & 3) | ((y & 3) << 2) | ((z & 3) << 4)) as usize)
}

pub struct SceneMap {
    pub vox: f64,
    cells: FastMap<u64, ColdBrick>, // 한동안 안 본 덩어리(큰 표: 덩어리마다 캐시·TLB 를 놓침)
    hot: FastMap<u64, u32>,  // 최근 HOT_AGE 키프레임 안에 본 덩어리 → bricks 자리(장면마다 같은 칸을 거듭 보므로 넣기가 거의 여기서 끝남)
    bricks: Vec<Brick>,
    free: Vec<u32>,          // 비운 덩어리 자리
    hot_kf: u32,             // 마지막으로 hot 을 비운 키프레임
    pub min_obs: u8, // 고운 칸을 본 키프레임 수 하한
    pending: Vec<u64>, // 막 확정된 칸(본 키프레임 수가 min_obs 에 닿은 칸): 다음 place 에서 옥트리로
    occ: Vec<FastSet<u64>>, // 수준별 차지한 칸
    pub nodes: FastMap<(u32, u64), Node>,
    pub run: u64, // 이 지도의 번호: 새로 쌓으면 바뀌어 뷰어가 받은 노드를 비움
    pub n_placed: usize,
}

/// SceneMap::adder 손잡이: put 으로 점을 넣고, 놓을 때(drop) 오래된 칸을 큰 표로
pub struct Adder<'a> {
    m: &'a mut SceneMap,
    kf: u32,
    last_b: (u64, usize),
    near: Box<[(u64, u32); NEAR]>, // 이 장면에서 찾은 덩어리(자리 = 덩어리 좌표 아래 비트): 윗줄 화소가 본 덩어리를 해시 없이 다시 찾음
}

/// Adder 의 최근 덩어리 표 크기(덩어리 좌표 x·y·z 아래 3 비트씩)
const NEAR: usize = 512;

impl Adder<'_> {
    #[inline(always)]
    pub fn put(&mut self, p: V3, c: [u8; 3], d: f32) {
        self.m.put(&mut self.last_b, &mut self.near, self.kf, p, c, d);
    }
}

impl Drop for Adder<'_> {
    fn drop(&mut self) {
        self.m.settle(self.kf);
    }
}

static RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl SceneMap {
    pub fn new(vox: f64, _sup: f64) -> SceneMap {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
        let run = t * 16 + RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16;
        SceneMap {
            vox, cells: FastMap::default(), hot: FastMap::default(), bricks: vec![], free: vec![], hot_kf: 0, min_obs: 2, pending: vec![],
            occ: (0..OCT_DEPTH).map(|_| FastSet::default()).collect(), nodes: FastMap::default(), run, n_placed: 0,
        }
    }
    /// 한 키프레임의 점((세계 점, 색, 카메라 거리))을 하나씩 넣는 손잡이(점을 만드는 반복문 안에서 바로 부름 → 점 벡터·반복자 호출 없이). 놓을 때 오래된 칸을 큰 표로
    pub fn adder(&mut self, kf: u32) -> Adder<'_> {
        Adder { m: self, kf, last_b: (u64::MAX, 0), near: Box::new([(u64::MAX, 0); NEAR]) }
    }
    /// 덩어리 bk 의 bricks 자리(없으면 만들고, 큰 표에 있던 덩어리면 칸을 제자리로 펼침). 이 장면에 봤다고 표시
    #[inline(never)]
    fn brick_slot(&mut self, bk: u64, kf: u32) -> u32 {
        let (cold, bricks, free) = (&mut self.cells, &mut self.bricks, &mut self.free);
        let bi = *self.hot.entry(bk).or_insert_with(|| {
            let mut b = Brick { occ: 0, last: 0, v: [SV::EMPTY; 64] };
            if let Some(cb) = cold.remove(&bk) {
                b.occ = cb.occ;
                let mut m = cb.occ;
                for v in cb.v.iter() {
                    b.v[m.trailing_zeros() as usize] = *v;
                    m &= m - 1;
                }
            }
            match free.pop() {
                Some(i) => {
                    bricks[i as usize] = b;
                    i
                }
                None => {
                    bricks.push(b);
                    (bricks.len() - 1) as u32
                }
            }
        });
        bricks[bi as usize].last = kf as u16;
        bi
    }
    /// 점 하나 넣기. last_b: 마지막 덩어리(이웃 점은 대개 같은 덩어리), near: 이 장면에서 찾은 덩어리 — 둘 다 맞으면 해시를 찾지 않음
    #[inline(always)]
    fn put(&mut self, last_b: &mut (u64, usize), near: &mut [(u64, u32); NEAR], kf: u32, p: V3, c: [u8; 3], d: f32) {
        // 칸 좌표·칸 안 위치를 한 번에(key() 와 같은 식)
        let vox = self.vox;
        let f = [p[0] / vox, p[1] / vox, p[2] / vox];
        let fl = f.map(|v| v.floor());
        let g = |v: f64| ((v as i64 + (1 << 20)) as u64) & 0x1F_FFFF;
        let k = (g(fl[0]) << 42) | (g(fl[1]) << 21) | g(fl[2]);
        let dmm = (d * 1000.0).clamp(0.0, 65535.0) as u16;
        let q = |i: usize| (((f[i] - fl[i]) * 255.0).round() as i32).clamp(0, 255);
        let (bk, slot) = brick_of(k);
        if bk != last_b.0 {
            let ni = ((bk & 7) | ((bk >> 19) & 7) << 3 | ((bk >> 38) & 7) << 6) as usize;
            if near[ni].0 != bk {
                near[ni] = (bk, self.brick_slot(bk, kf));
            }
            *last_b = (bk, near[ni].1 as usize);
        }
        let b = &mut self.bricks[last_b.1];
        let bit = 1u64 << slot;
        if b.occ & bit == 0 {
            b.occ |= bit;
            b.v[slot] = SV { off: [0; 3], col: c, n: 0, nc: 0, nf: 0, last: u16::MAX, best: u16::MAX };
        }
        let e = &mut b.v[slot];
        // 칸 안 평균: 이어 더하는 평균(16 번 넘으면 최근 쪽 무게를 둔 평균)
        let n = (e.n as i32 + 1).min(16);
        for i in 0..3 {
            let m = e.off[i] as i32;
            e.off[i] = (m + div(q(i) - m, n)).clamp(0, 255) as u8;
        }
        e.n = n as u8;
        if e.last != kf as u16 {
            e.last = kf as u16;
            e.nf = e.nf.saturating_add(1);
            if e.nf == self.min_obs {
                self.pending.push(k);
            }
        }
        // 색: 가장 가까운 거리의 2 배 안에서 본 관측의 평균(가까운 쪽 사진이 선명하고, 비슷한 거리 여러 장의 평균이 노출·잡음을 고름).
        // 훨씬 가까운 관측이 오면 그 색에서 다시 시작
        if (dmm as u32) * 2 < e.best as u32 || e.best == u16::MAX {
            e.best = dmm;
            e.col = c;
            e.nc = 1;
        } else if dmm as u32 <= (e.best as u32) * 2 {
            e.nc = e.nc.saturating_add(1).min(16);
            for i in 0..3 {
                let m = e.col[i] as i32;
                e.col[i] = (m + div(c[i] as i32 - m, e.nc as i32)) as u8;
            }
            e.best = e.best.min(dmm);
        }
    }
    /// HOT_AGE 장 넘게 안 본 덩어리는 큰 표로(장면마다가 아니라 덩어리가 시야를 떠날 때 한 번만 큰 표를 건드림).
    /// 칸 값은 어느 표에 있든 같다 — 옮기는 때는 빠르기만 바꾼다
    fn settle(&mut self, kf: u32) {
        let (cold, bricks, free) = (&mut self.cells, &mut self.bricks, &mut self.free);
        if kf.wrapping_sub(self.hot_kf) >= HOT_AGE / 2 || kf < self.hot_kf {
            self.hot_kf = kf;
            let now = kf as u16;
            self.hot.retain(|bk, bi| {
                let b = &bricks[*bi as usize];
                let keep = now.wrapping_sub(b.last) < HOT_AGE as u16;
                if !keep {
                    let mut m = b.occ;
                    let v: Box<[SV]> = (0..b.occ.count_ones()).map(|_| {
                        let s = m.trailing_zeros() as usize;
                        m &= m - 1;
                        b.v[s]
                    }).collect();
                    cold.insert(*bk, ColdBrick { occ: b.occ, v });
                    free.push(*bi);
                }
                keep
            });
        }
    }
    fn cell(&self, k: &u64) -> &SV {
        let (bk, s) = brick_of(*k);
        match self.hot.get(&bk) {
            Some(&bi) => &self.bricks[bi as usize].v[s],
            None => {
                let cb = &self.cells[&bk];
                &cb.v[(cb.occ & ((1u64 << s) - 1)).count_ones() as usize]
            }
        }
    }
    fn pos(&self, k: u64, v: &SV) -> [f32; 3] {
        let f = |s: u32, i: usize| ((((k >> s) & 0x1F_FFFF) as i64 - (1 << 20)) as f64 + v.off[i] as f64 / 255.0) * self.vox;
        [f(42, 0) as f32, f(21, 1) as f32, f(0, 2) as f32]
    }
    /// 새로 확정된 칸을 옥트리 노드에 넣는다. 돌려줌: 넣은 점 수
    pub fn place(&mut self) -> usize {
        let pend = std::mem::take(&mut self.pending);
        let mut n = 0;
        for k in pend {
            let mut lv = OCT_DEPTH;
            for l in 0..OCT_DEPTH {
                if self.occ[l as usize].insert(coarse_of(k, 1 << (OCT_DEPTH - l))) {
                    lv = l;
                    break;
                }
            }
            let nk = coarse_of(k, OCT_G << (OCT_DEPTH - lv));
            self.nodes.entry((lv, nk)).or_insert(Node { keys: vec![], ver: 0, written: 0 }).keys.push(k);
            n += 1;
        }
        self.n_placed += n;
        n
    }
    /// 수준 L 노드 한 변 길이(m)
    pub fn node_size(&self, l: u32) -> f64 {
        self.vox * (OCT_G << (OCT_DEPTH - l)) as f64
    }
    /// 노드 파일 바이트: 점마다 u16×3 위치(노드 원점에서 한 변의 1/65535 단위) 다음 RGB(전체: 위치 블록 뒤 색 블록)
    pub fn node_bytes(&self, l: u32, nk: u64) -> Vec<u8> {
        let nd = &self.nodes[&(l, nk)];
        let size = self.node_size(l);
        let o = node_coord(nk).map(|c| c as f64 * size);
        let mut b = Vec::with_capacity(nd.keys.len() * 9);
        let mut cols = Vec::with_capacity(nd.keys.len() * 3);
        for k in &nd.keys {
            let v = self.cell(k);
            let p = self.pos(*k, v);
            for i in 0..3 {
                let q = ((p[i] as f64 - o[i]) / size * 65535.0).round().clamp(0.0, 65535.0) as u16;
                b.extend_from_slice(&q.to_le_bytes());
            }
            cols.extend_from_slice(&v.col);
        }
        b.extend_from_slice(&cols);
        b
    }
    /// 다시 쓸 노드(쓸 때 판 번호 +1): 아직 안 쓴 노드가 먼저, 그다음 점이 많이 는 순. 점이 10 % 넘게(또는 처음) 는 노드만,
    /// 고른 노드의 점 합이 budget 을 넘으면 멈춤(남은 노드는 다음 번에) — 본 스레드가 한 번에 오래 멈추지 않게. budget 0 = 바뀐 노드 모두
    pub fn take_dirty(&mut self, budget: usize) -> Vec<(u32, u64)> {
        let mut c: Vec<(f64, (u32, u64))> = self.nodes.iter().filter_map(|(k, n)| {
            let (a, b) = (n.keys.len(), n.written);
            let grow = if b == 0 { f64::MAX } else { (a - b) as f64 / b as f64 };
            (a != b && (budget == 0 || grow >= 0.1)).then_some((grow, *k))
        }).collect();
        c.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        let mut out = vec![];
        let mut sum = 0;
        for (_, k) in c {
            if budget > 0 && sum >= budget {
                break;
            }
            let n = self.nodes.get_mut(&k).unwrap();
            sum += n.keys.len();
            n.written = n.keys.len();
            n.ver += 1;
            out.push(k);
        }
        out
    }
    /// 노드 목록(JSON 용): [수준, x, y, z, 점 수, 판]
    pub fn index(&self) -> Vec<[i64; 6]> {
        self.nodes.iter().filter(|(_, n)| n.written > 0).map(|((l, k), n)| {
            let c = node_coord(*k);
            [*l as i64, c[0], c[1], c[2], n.written as i64, n.ver as i64]
        }).collect()
    }
    /// 여러 키프레임에서 본 칸 수
    pub fn n_points(&self) -> usize {
        self.n_placed
    }
    pub fn bytes(&self) -> usize {
        self.cells.capacity() * 33 + self.cells.values().map(|c| c.v.len() * std::mem::size_of::<SV>() + 16).sum::<usize>()
            + self.hot.capacity() * 13 + self.bricks.capacity() * std::mem::size_of::<Brick>() + self.occ.iter().map(|s| s.capacity() * 9).sum::<usize>()
            + self.nodes.values().map(|n| n.keys.capacity() * 8 + 48).sum::<usize>() + self.pending.capacity() * 8
    }
}

/// 키 → 칸 좌표(부호 있는 정수)
pub fn node_coord(k: u64) -> [i64; 3] {
    [42, 21, 0].map(|s| ((k >> s) & 0x1F_FFFF) as i64 - (1 << 20))
}

// ------------------------------------------------------------------ 2D 점유 격자

/// 격자 조각 한 변(칸): 칸 값을 조각마다 한 덩어리로(칸마다 해시를 찾지 않음)
const GT: i32 = 64;
/// 칸 값: (막힘으로 본 키프레임 수, 빔으로 본 키프레임 수, 그중 가구보다 높은 막힘을 본 수). 셋 다 0 = 아직 안 본 칸
type GCell = (u32, u32, u32);

pub struct Grid {
    pub res: f64,
    tiles: FastMap<(i32, i32), Box<[GCell]>>, // 조각 (x>>6, y>>6) → 64×64 칸(줄 = y)
}

pub struct GridImg {
    pub w: usize,
    pub h: usize,
    pub x0: i32,
    pub y0: i32,
    pub px: Vec<u8>, // 254 빔, 0 막힘, 205 모름; 위가 +y
    pub wall: Vec<bool>, // 막힌 칸 중 가구보다 높은 것(벽·문틀·키 큰 장) — 방은 이것으로만 나뉜다
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
        let oc: Vec<(i32, i32, bool)> = occ.iter().map(|p| { let (x, y) = c(p.0, p.1); (x, y, p.2) }).collect();
        let fc: Vec<(i32, i32)> = floor.iter().map(|p| c(p.0, p.1)).collect();
        let (mut x0, mut y0, mut x1, mut y1) = (cx, cy, cx, cy);
        for (x, y) in oc.iter().map(|p| (p.0, p.1)).chain(fc.iter().copied()) {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        let (bw, bh) = ((x1 as i64 - x0 as i64 + 1) as usize, (y1 as i64 - y0 as i64 + 1) as usize);
        if bw * bh > 1 << 24 {
            return self.add_sets(cam, occ, floor); // 장면 범위가 터무니없이 크면(위치가 튐) 칸 집합으로
        }
        const O: u8 = 1;
        const T: u8 = 2;
        const F: u8 = 4;
        let mut fl = vec![0u8; bw * bh];
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
        for yy in 0..bh {
            for xx in 0..bw {
                let f = fl[yy * bw + xx];
                if f & (O | F) == 0 {
                    continue;
                }
                let e = self.cell_mut(x0 + xx as i32, y0 + yy as i32);
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
        let mut px = vec![205u8; w * h];
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
        Some(GridImg { w, h, x0, y0, px, wall })
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
pub const LOOK_COS: f32 = 0.88;
/// 작은 쪽 굳은 칸 중 큰 쪽 굳은 칸과 겹쳐야 하는 비율(같은 물체 0.33, 나란한 모니터 둘 0.14)
pub const LOOK_OVL: f32 = 0.25;

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
    pub last_seg: u32, // 마지막으로 본 때의 Memory::n_seg
}

pub struct Memory {
    pub objs: Vec<Obj>,
    pub next: u32, // 다음 물체 id(다시 쌓을 때 이어서 써 뷰어가 옛 물체 파일과 헷갈리지 않게)
    pub vox: f64,
    pub sup: f64,
    pub merged: u32, // 병합 후처리로 합친 수(누적)
    pub n_seg: u32, // 분할을 돌린 장면 수(분할을 건너뛴 장면은 세지 않음)
}

impl Memory {
    pub fn new(vox: f64, sup: f64) -> Memory {
        Memory { objs: vec![], next: 0, vox, sup, merged: 0, n_seg: 0 }
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
        // 이웃 점은 대개 같은 굵은 칸: 바로 앞과 같은 키는 다시 넣지 않음(이미 든 키 넣기는 표를 바꾸지 않으니 같은 집합·같은 차례)
        let mut ckeys: FastSet<u64> = FastSet::default();
        let mut last = None;
        for (p, _) in pts {
            let k = key(p, self.sup);
            if last != Some(k) {
                ckeys.insert(k);
                last = Some(k);
            }
        }
        let i = match self.find(name, &c, &ext, &ckeys, cam.as_ref()) {
            Some(i) => i,
            None => {
                self.objs.push(Obj {
                    id: self.next, name: name.into(), votes: FastMap::default(), score: 0.0, n_obs: 0, first_pos: c, pos: c,
                    extent: ext, last_seen: stamp, miss: 0, state: "seen", cells: FastMap::default(), support: FastMap::default(),
                    crop: None, crop_area: 0, crop_ver: 0, dirty: true, stamp: 0, n_pts: 0, crop_saved: 0, cells_saved: 0, last_seg: self.n_seg,
                });
                self.next += 1;
                self.objs.len() - 1
            }
        };
        let ns = self.n_seg;
        let o = &mut self.objs[i];
        for k in &ckeys {
            *o.support.entry(*k).or_insert(0) += 1;
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
            let same_crop = match (&o.crop, &p.crop) {
                (Some(a), Some(b)) => a.w == b.w && a.h == b.h && a.rgb == b.rgb && a.mask == b.mask && a.depth == b.depth,
                (None, None) => true,
                _ => false,
            };
            if same_crop && p.crop_saved == p.crop_ver {
                (o.crop_ver, o.crop_saved) = (p.crop_ver, p.crop_saved);
            } else {
                (o.crop_ver, o.crop_saved) = (p.crop_ver + 1, p.crop_saved);
            }
            n += 1;
        }
        n
    }

    pub fn prune(&mut self, now: f64) -> usize {
        let n0 = self.objs.len();
        // 두 번 이하로 본 물체는 3 초가 지나고 분할한 장면 8 장에서도 안 보였을 때 지움(분할을 건너뛴 장면은 못 본 것으로 세지 않음)
        let ns = self.n_seg;
        self.objs.retain(|o| !(o.n_obs <= 2 && now - o.last_seen > 3.0 && ns.saturating_sub(o.last_seg) >= 8));
        n0 - self.objs.len()
    }

    /// 병합 후처리(HOV-SG·ConceptGraphs 방식): 따로 생긴 두 물체가 같은 것이면 합친다.
    /// 기하: 굵은 칸(4 cm) 3D IoU, 작은 쪽 기준 겹침 비율. 의미: 이름 투표 분포 코사인. 생김새: 점 색 히스토그램 코사인.
    /// 합치는 조건: IoU ≥ 0.5, 또는 겹침 ≥ 0.6 이면서 (이름 ≥ 0.5 또는 생김새 ≥ 0.8), 또는 이름 ≥ 0.5 이고 작은 쪽 굳은 칸 70 % 가 큰 쪽 칸 안(겹침 ≥ 0.2).
    /// 사라짐 표시된 물체도 후보(이름 ≥ 0.5 일 때만).
    /// 사진 벡터(embed, 있을 때): 이름이 달라도 두 사진 벡터 코사인 ≥ LOOK_COS 이고 작은 쪽 굳은 칸의 LOOK_OVL 이상이 큰 쪽 굳은 칸과 겹치면 합침
    /// (한 물체가 보는 방향에 따라 다른 이름으로 잡혀 둘로 남은 경우). 벡터는 종류까지만 가려서(나란한 모니터 둘도 0.92) 겹침 없이 벡터만으로는 합치지 않는다.
    /// 반환: 합친 수
    pub fn merge_pass(&mut self) -> usize {
        self.merge_pass_with(&crate::embed::get)
    }

    /// merge_pass 본체. emb: 물체 번호 → 사진 벡터(길이 1)
    pub fn merge_pass_with(&mut self, emb: &dyn Fn(u32) -> Option<Vec<f32>>) -> usize {
        let (frag_in, frag_ovl) = (0.7f32, 0.2f32); // 조각: 굳은 칸 70 % 가 상대 칸 안, 전체 칸 겹침 20 % 이상
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
        // 굳은 칸: 여러 관측이 지지한 굵은 칸(refresh 가 보여 주는 점과 같은 문턱). 가장자리 번짐·한두 장면 오검출 칸을 뺀 물체 몸통
        let solid = |o: &Obj| -> Vec<u64> {
            let thr = if o.n_obs < 3 { 1 } else { (0.15 * o.n_obs as f64).ceil().clamp(2.0, 10.0) as u32 };
            o.support.iter().filter(|(_, n)| **n >= thr).map(|(k, _)| *k).collect()
        };
        // 맞닿은 조각: 한 물체가 보는 방향마다 다른 부분(의자 등받이·앉는 판)만 잡혀 칸이 거의 안 겹친 채 둘셋으로 남는다.
        // 같은 이름이고 굳은 칸끼리 맞닿아 있으며(사이 빈 칸 없이 이웃 칸 5 개 이상) 합친 굳은 칸 상자(축마다 5–95 %)가 그 이름의 전형 크기 안이면 합침.
        // 나란한 의자 둘(중심 1.4 m)은 합친 상자가 의자 하나보다 커서 남는다. 전형 크기를 모르는 이름은 이 규칙을 쓰지 않음
        let sup = self.sup;
        let typical = |name: &str| -> Option<[f64; 3]> {
            // 큰 순서 세 변(m), 넉넉하게: 수평은 비스듬히 놓인 대각선까지
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
        };
        let unkey = |k: u64| [42u32, 21, 0].map(|s| ((k >> s) & 0x1F_FFFF) as i64 - (1 << 20));
        let touching = |a: &Obj, b: &Obj, sa: &[u64], sb: &[u64]| -> bool {
            let Some(lim) = typical(&a.name) else { return false };
            if a.name != b.name || sa.len() < 3 || sb.len() < 3 {
                return false;
            }
            // 맞닿음: 작은 쪽 굳은 칸 중 큰 쪽 굳은 칸과 같거나 이웃(26 방향)인 칸
            let (s, l) = if sa.len() <= sb.len() { (sa, sb) } else { (sb, sa) };
            let lset: crate::geom::FastSet<u64> = l.iter().copied().collect();
            let near = s
                .iter()
                .filter(|&&k| {
                    let c = unkey(k);
                    (-1i64..=1).any(|x| (-1i64..=1).any(|y| (-1i64..=1).any(|z| {
                        let p = [(c[0] + x) as f64 + 0.5, (c[1] + y) as f64 + 0.5, (c[2] + z) as f64 + 0.5].map(|v| v * sup);
                        lset.contains(&key(&p, sup))
                    })))
                })
                .count();
            if near < 5 {
                return false;
            }
            let cs: Vec<[i64; 3]> = sa.iter().chain(sb).map(|&k| unkey(k)).collect();
            let mut ext: Vec<f64> = (0..3)
                .map(|ax| {
                    let mut v: Vec<i64> = cs.iter().map(|c| c[ax]).collect();
                    v.sort_unstable();
                    (v[v.len() * 19 / 20] - v[v.len() / 20] + 1) as f64 * sup
                })
                .collect();
            ext.sort_by(|x, y| y.partial_cmp(x).unwrap());
            (0..3).all(|k| ext[k] <= lim[k])
        };
        let mut merged = 0;
        loop {
            // 사라짐으로 표시된 물체도 후보: 정지 장면에서 잘못 사라짐 처리된 물체 자리에 같은 물체가 새로 생기면 둘로 남았다(중복 대부분이 이 경우).
            // 사라진 물체가 낀 짝은 이름이 같아야(생김새만으로는 안 합침), 합친 물체는 더 최근에 본 쪽 상태를 따른다
            let alive: Vec<usize> = (0..self.objs.len()).collect();
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
                    let touch = || votes_cos(a, b) >= 0.5 && touching(a, b, &solid(a), &solid(b));
                    if inter == 0.0 {
                        if touch() {
                            pair = Some((i, j));
                            break 'outer;
                        }
                        continue;
                    }
                    let iou = inter / ((sa.len() + sb.len()) as f32 - inter).max(1.0);
                    let ovl = inter / sa.len().max(1) as f32;
                    let sem = votes_cos(a, b);
                    let app = if ovl >= 0.6 { hist(a).iter().zip(hist(b)).map(|(x, y)| x * y).sum::<f32>() } else { 0.0 };
                    let any_gone = a.state == "gone" || b.state == "gone";
                    // 조각: 작은 쪽 굳은 칸의 대부분이 큰 쪽이 본 칸 안(같은 이름). 전체 칸 겹침은 번짐 칸이 분모를 키워 낮게 나온다
                    let frag = || {
                        let (s, l) = if a.support.len() <= b.support.len() { (a, b) } else { (b, a) };
                        let sc = solid(s);
                        sc.len() >= 3 && sc.iter().filter(|k| l.support.contains_key(k)).count() as f32 >= frag_in * sc.len() as f32
                    };
                    // 사진 벡터가 닮았고 몸통이 실제로 겹침(이름 무관)
                    let look = || {
                        let (Some(ea), Some(eb)) = (emb(a.id), emb(b.id)) else { return false };
                        let cos = ea.iter().zip(&eb).map(|(x, y)| x * y).sum::<f32>();
                        if ea.len() != eb.len() || cos < LOOK_COS {
                            return false;
                        }
                        let (ca, cb) = (solid(a), solid(b));
                        let (s, l) = if ca.len() <= cb.len() { (ca, cb) } else { (cb, ca) };
                        let lset: FastSet<u64> = l.into_iter().collect();
                        let hit = s.iter().filter(|k| lset.contains(k)).count() as f32;
                        let ok = s.len() >= 3 && hit >= LOOK_OVL * s.len() as f32;
                        if ok {
                            // 다른 조건이 모두 아닐 때만 여기까지 온다: 사진 벡터가 정한 병합
                            eprintln!("merge(look): {}#{} + {}#{} cos {cos:.2} overlap {:.2}", a.name, a.id, b.name, b.id, hit / s.len() as f32);
                        }
                        ok
                    };
                    let geo = iou >= 0.5 || ovl >= 0.6 || (ovl >= frag_ovl && frag());
                    if if any_gone { sem >= 0.5 && geo } else { iou >= 0.5 || (ovl >= 0.6 && (sem >= 0.5 || app >= 0.8)) || (sem >= 0.5 && ovl >= frag_ovl && frag()) } || touch() || look() {
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
                a.last_seg = a.last_seg.max(b.last_seg);
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

impl SceneMap {
    /// 노드를 모두 다시 쓰게 표시(지도를 새로 쌓은 뒤)
    pub fn mark_all(&mut self) {
        for n in self.nodes.values_mut() {
            n.written = 0;
        }
    }
}

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
        m.update(name, 0.9, &pts, 1.0, 0.0, 1, || Crop { w: 1, h: 1, rgb: vec![0; 3], mask: vec![255], depth: vec![0] }, None)
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
}
