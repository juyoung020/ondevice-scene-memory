//! 방 나누기·방 이름(Hydra room finder 의 2D 판).
//! Hydra(RSS 2022)는 장애물까지 거리의 문턱을 키워 가며 빈 공간을 끊고, 오래 따로 남는 덩이를 방 씨앗으로 삼는다.
//! 여기서는 바닥 점유 격자의 빈칸을 그대로 쓰고, 씨앗은 덩이마다 수명으로 고른다(지속성 나누기 — 크기 다른 방이 섞여도 됨).
//!   1) 빈칸 / 막힘 / 모름. 막힘은 가구보다 높은 것(벽)만 — 책상·의자 자리는 방 안 빈칸으로 본다. 안쪽의 작은 모름 구멍은 빈칸으로, 빈칸에 둘러싸인 작은 막힘 점(의자 다리·깊이 잡음)은 지운다.
//!   2) 빈칸의 정확한 유클리드 거리 변환 → 벽 면까지 여유 c = 칸 중심 거리 − res/2.
//!   3) 한 번 쓸기 거름 + 붙이기: 여유가 dil_max 넘는 칸은 연결 성분으로 묶고, 나머지는 dil_step 층으로 나눠 여유 큰 칸부터 넣는다.
//!      칸은 이미 들어온 이웃 중 여유가 가장 큰 칸의 씨앗을 물려받는다. 두 성분이 만나면 어린 쪽 수명(태어난 층 − 만난 층)이
//!      min_life 이상이고 둘 다 min_seed 이상일 때 둘 다 방 씨앗으로 굳히고, 아니면 짧게 산 쪽을 건너편 씨앗에 합친다.
//!      dil_min 아래에서는 새 씨앗이 없다: 큰 홑성분은 씨앗, 작은 것은 만나는 대로 건너편에 붙고, 씨앗 없는 덩이는 min_room 이상이면 방.
//!      → 문 반폭 ≈ dil_max − min_life 보다 좁은 통로로만 이어진 곳이 서로 다른 방.
//!   4) 이음매(두 방이 맞닿은 칸 쌍)가 max_door 보다 길면 문이 아니므로 합친다. min_room 보다 작은 방은 이음매가 가장 긴 이웃에.
//!   5) 남은 이음매 = 문(위치, 폭 = 2·최대 여유 + res).
//!   6) 방 번호 유지: 이전 나눔과 칸 겹침(세계 칸 좌표)으로 짝짓고, 짝 없는 방만 새 번호.
//! 물체 배정: 물체 바닥 자리(xy 상자 + 여유)의 방 칸 다수결, 없으면 obj_search 안 가장 가까운 방 칸.
//! 이름: 방 안 물체 이름의 머리 명사 규칙으로 종류별 점수 → 확률(모름 몫 포함)·확신·근거. 같은 종류가 여럿이면 번호를 붙인다.
use crate::maps::GridImg;
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Params {
    pub dil_min: f64, // 거름 문턱 범위(벽 면까지 여유, m)
    pub dil_max: f64,
    pub dil_step: f64,
    pub min_life: f64,         // 씨앗이 되려면 이만큼(문턱 m) 따로 살아야
    pub min_seed_m2: f64,      // 씨앗 최소 넓이(만날 때)
    pub min_room_m2: f64,      // 방 최소 넓이
    pub max_door_m: f64,       // 이음매가 이보다 길면 문이 아님 → 합침
    pub hole_m2: f64,          // 이보다 작은 안쪽 모름 구멍 → 빈칸
    pub speck_m2: f64,         // 빈칸에 둘러싸인 이 이하 막힘 덩이 → 빈칸
    pub obj_search_m: f64,     // 물체 자리에 방 칸이 없을 때 찾는 반경
    pub footprint_margin: f64, // 물체 바닥 자리 여유
    pub match_min: f64,        // 번호 잇기: 겹침 ≥ match_min × 작은 쪽 넓이
}

impl Default for Params {
    fn default() -> Self {
        Params { dil_min: 0.30, dil_max: 0.60, dil_step: 0.025, min_life: 0.10, min_seed_m2: 0.5, min_room_m2: 2.0, max_door_m: 1.6,
                 hole_m2: 0.25, speck_m2: 0.01, obj_search_m: 1.0, footprint_margin: 0.15, match_min: 0.3 }
    }
}

#[derive(Clone)]
pub struct Room {
    pub id: u32,
    pub n_cells: usize,
    pub area_m2: f64,
    pub centroid: [f64; 2],
    pub max_clear: f64, // 가장 넓은 곳의 벽까지 여유(m)
}

#[derive(Clone)]
pub struct Door {
    pub a: u32, // a < b(방 번호)
    pub b: u32,
    pub pos: [f64; 2], // 이음매 가운데
    pub width: f64,
    pub seam: usize, // 맞닿은 칸 쌍 수
}

/// 칸 (x, y) = 세계 칸 (gx0 + x, gy0 + y), 줄 0 이 아래(−y).
#[derive(Clone, Default)]
pub struct Seg {
    pub w: usize,
    pub h: usize,
    pub gx0: i32,
    pub gy0: i32,
    pub res: f64,
    pub ids: Vec<u32>, // 칸마다 방 번호(0 = 없음)
    pub rooms: Vec<Room>, // 번호 오름차순
    pub doors: Vec<Door>,
}

impl Seg {
    pub fn index(&self, id: u32) -> Option<usize> {
        self.rooms.iter().position(|r| r.id == id)
    }
}

fn find(p: &mut [i32], mut a: i32) -> i32 {
    while p[a as usize] != a {
        p[a as usize] = p[p[a as usize] as usize];
        a = p[a as usize];
    }
    a
}

/// 작은 모름 구멍 → 빈칸, 빈칸에 둘러싸인 작은 막힘 점 → 빈칸. cls: 0 모름, 1 빈칸, 2 막힘.
fn clean_free(cls: &[u8], w: usize, h: usize, hole: usize, speck: usize) -> Vec<bool> {
    let pw = w + 2;
    let mut pc = vec![3u8; pw * (h + 2)]; // 덧댄 격자(밖 = 3), |0x80 = 본 칸
    for y in 0..h {
        pc[(y + 1) * pw + 1..(y + 1) * pw + 1 + w].copy_from_slice(&cls[y * w..(y + 1) * w]);
    }
    let mut free: Vec<bool> = cls.iter().map(|&c| c == 1).collect();
    let p = pw as isize;
    let (o4, od) = ([-1, 1, -p, p], [-p - 1, -p + 1, p - 1, p + 1]);
    let (mut stack, mut comp) = (vec![], vec![]);
    for y in 1..=h {
        for x in 1..=w {
            let s0 = y * pw + x;
            let want = pc[s0];
            if want == 1 || want & 0x80 != 0 {
                continue;
            }
            let eight = want == 2; // 막힘 점은 대각선으로도 한 덩이
            let limit = if want == 0 { hole } else { speck };
            stack.clear();
            stack.push(s0);
            pc[s0] |= 0x80;
            comp.clear();
            let (mut border, mut keep) = (false, limit > 0);
            let (mut nfree, mut nother) = (0usize, 0usize);
            while let Some(c) = stack.pop() {
                if keep {
                    comp.push(c);
                    keep = comp.len() <= limit;
                }
                for d in o4 {
                    let j = (c as isize + d) as usize;
                    let v = pc[j] & 0x7f;
                    if v == want {
                        if pc[j] & 0x80 == 0 {
                            pc[j] |= 0x80;
                            stack.push(j);
                        }
                    } else if v == 3 {
                        border = true;
                    } else if v == 1 {
                        nfree += 1;
                    } else {
                        nother += 1;
                    }
                }
                if eight {
                    for d in od {
                        let j = (c as isize + d) as usize;
                        if pc[j] == want {
                            pc[j] |= 0x80;
                            stack.push(j);
                        } else if pc[j] & 0x7f == 3 {
                            border = true;
                        }
                    }
                }
            }
            if !keep || border || comp.is_empty() {
                continue;
            }
            // 구멍: 둘레가 주로 빈칸. 점: 둘레가 빈칸뿐
            if if want == 0 { nfree > 0 && nfree >= nother } else { nother == 0 } {
                for &c in &comp {
                    free[(c / pw - 1) * w + (c % pw - 1)] = true;
                }
            }
        }
    }
    free
}

/// Felzenszwalb–Huttenlocher 1D 제곱 거리 변환
fn dt1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let isect = |q: usize, r: usize| ((f[q] + (q * q) as f32) - (f[r] + (r * r) as f32)) / (2.0 * (q as f32 - r as f32));
    let mut k = 0;
    v[0] = 0;
    z[0] = -1e20;
    z[1] = 1e20;
    for q in 1..n {
        let mut s = isect(q, v[k]);
        while s <= z[k] && k > 0 {
            k -= 1;
            s = isect(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = 1e20;
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dq = q as f32 - v[k] as f32;
        d[q] = dq * dq + f[v[k]];
    }
}

/// 빈칸마다 벽 면까지 여유(m), 빈칸 아니면 0. 정확한 유클리드 거리: 세로(위·아래 쓸기) + 가로 아래 봉투, 격자 밖 = 막힘.
fn clearance(free: &[bool], w: usize, h: usize, res: f64) -> Vec<f32> {
    let pw = w + 2;
    let mut g = vec![0f32; h * pw];
    let mut run = vec![0u32; w];
    for y in 0..h {
        for x in 0..w {
            run[x] = if free[y * w + x] { run[x] + 1 } else { 0 };
            g[y * pw + x + 1] = run[x] as f32;
        }
    }
    run.fill(0);
    let mut clear = vec![0f32; w * h];
    let (mut d, mut v, mut z) = (vec![0f32; pw], vec![0usize; pw], vec![0f32; pw + 1]);
    let (fres, half) = (res as f32, (res / 2.0) as f32);
    for y in (0..h).rev() {
        let row = &mut g[y * pw..(y + 1) * pw];
        for x in 0..w {
            run[x] = if free[y * w + x] { run[x] + 1 } else { 0 };
            let m = row[x + 1].min(run[x] as f32);
            row[x + 1] = m * m;
        }
        // 빈칸 토막마다 따로 봉투: 양 끝 막힌 칸(0)이 닻이라 토막 밖 포물선은 토막 안에서 이길 수 없다
        let mut x = 0;
        while x < w {
            if !free[y * w + x] {
                x += 1;
                continue;
            }
            let x0 = x;
            while x < w && free[y * w + x] {
                x += 1;
            }
            let n = x - x0 + 2; // 닻 + 토막 + 닻
            dt1d(&row[x0..x0 + n], &mut d, &mut v, &mut z);
            for q in 0..n - 2 {
                clear[y * w + x0 + q] = d[q + 1].sqrt() * fres - half;
            }
        }
    }
    clear
}

/// 한 번 나누기. cls: 칸마다 0 모름, 1 빈칸, 2 막힘(줄 0 이 아래). 방 번호 = 1..n.
pub fn segment(cls: &[u8], w: usize, h: usize, res: f64, gx0: i32, gy0: i32, p: &Params) -> Seg {
    let n_cells = w * h;
    let mut s = Seg { w, h, gx0, gy0, res, ids: vec![0; n_cells], rooms: vec![], doors: vec![] };
    if w < 3 || h < 3 {
        return s;
    }
    let a1 = res * res;
    let mut free = clean_free(cls, w, h, (p.hole_m2 / a1) as usize, (p.speck_m2 / a1 + 1e-9) as usize);
    // 테두리 빈칸은 잠시 막음(이웃 범위 검사 없이) — 끝에서 안쪽 이웃의 방을 받음
    let mut rim = vec![];
    for i in (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((1..h - 1).flat_map(|y| [y * w, y * w + w - 1])) {
        if free[i] {
            free[i] = false;
            rim.push(i);
        }
    }
    // 거리 변환 → 여유, 층 통
    let step = p.dil_step.max(1e-3);
    let k_layers = ((((p.dil_max - p.dil_min) / step + 1e-9).floor() as i32) + 1).clamp(1, 120);
    let life = ((p.min_life / step).round() as i32).max(1);
    let min_seed = ((p.min_seed_m2 / a1) as i32).max(1);
    let min_room = (p.min_room_m2 / a1) as i32;
    let lmin = (-1).min((((res / 2.0 - p.dil_min) / step - 1e-5).ceil() as i32) - 2); // 층 0 아래 통이 늘 하나는 있음
    let top = (k_layers - lmin - 1) as usize; // 맨 위 통 = 층 K − 1 이상(c > dil_max − step) 전부
    let clear = clearance(&free, w, h, res);
    let mut key = vec![-1i32; n_cells];
    let mut bucket: Vec<Vec<u32>> = vec![vec![]; top];
    let (fmin, inv) = (p.dil_min as f32, (1.0 / step) as f32);
    for i in 0..n_cells {
        if free[i] {
            let l = ((clear[i] - fmin) * inv - 1e-5).ceil() as i32 - 1;
            let k = (l - lmin).clamp(0, top as i32);
            key[i] = k;
            if (k as usize) < top {
                bucket[k as usize].push(i as u32);
            }
        }
    }
    // 태어남 번호마다: 성분 부모·크기·태어난 층·홑씨앗(−1 = 섞임), 씨앗 부모, 씨앗이 방이 되는지
    let mut b = vec![-1i32; n_cells];
    let (mut cpar, mut csz, mut cbirth, mut ccs, mut spar): (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) = (vec![], vec![], vec![], vec![], vec![]);
    let mut strong: Vec<bool> = vec![];
    // 맨 위 층: 한 층이라 만남은 언제나 수명 0 → 그냥 연결 성분
    let mut stack = vec![];
    for s0 in 0..n_cells {
        if key[s0] != top as i32 || b[s0] >= 0 {
            continue;
        }
        let id = cpar.len() as i32;
        let mut n = 0;
        b[s0] = id;
        stack.push(s0);
        while let Some(c) = stack.pop() {
            n += 1;
            for j in [c - 1, c + 1, c - w, c + w] {
                if key[j] == top as i32 && b[j] < 0 {
                    b[j] = id;
                    stack.push(j);
                }
            }
        }
        cpar.push(id);
        spar.push(id);
        csz.push(n);
        cbirth.push(k_layers - 1);
        ccs.push(id);
        strong.push(false);
    }
    // dil_min 아래로 내려갈 때: 섞인 성분의 씨앗·큰 홑성분의 씨앗만 방이 된다
    let go_late = |cpar: &mut [i32], spar: &[i32], csz: &[i32], ccs: &[i32], strong: &mut [bool]| {
        for i in 0..spar.len() {
            if spar[i] == i as i32 {
                let r = find(cpar, i as i32) as usize;
                strong[i] = ccs[r] < 0 || csz[r] >= min_seed;
            }
        }
    };
    let mut late = false;
    for k in (0..top).rev() {
        let l = k as i32 + lmin;
        if l < 0 && !late {
            go_late(&mut cpar, &spar, &csz, &ccs, &mut strong);
            late = true;
        }
        for &c in &bucket[k] {
            let c = c as usize;
            let nb = [c - 1, c + 1, c - w, c + w];
            let (mut best, mut bc) = (usize::MAX, -1f32);
            for j in nb {
                if b[j] >= 0 && clear[j] > bc {
                    bc = clear[j];
                    best = j;
                }
            }
            if best == usize::MAX {
                // 새 성분 = 새 씨앗(늦은 단계면 약함)
                let id = cpar.len() as i32;
                b[c] = id;
                cpar.push(id);
                spar.push(id);
                csz.push(1);
                cbirth.push(l);
                ccs.push(id);
                strong.push(false);
                continue;
            }
            b[c] = b[best];
            let mut rc = find(&mut cpar, b[c]) as usize;
            csz[rc] += 1;
            for j in nb {
                if j == best || b[j] < 0 {
                    continue;
                }
                let mut rb = find(&mut cpar, b[j]) as usize;
                if rb == rc {
                    continue;
                }
                let mut ra = rc;
                let (mut side_a, mut side_b) = (find(&mut spar, b[c]), find(&mut spar, b[j])); // 만난 자리 양쪽 씨앗
                // ra = 나이 많은 쪽(태어난 층 높음, 같으면 큰 쪽)
                if cbirth[rb] > cbirth[ra] || (cbirth[rb] == cbirth[ra] && csz[rb] > csz[ra]) {
                    std::mem::swap(&mut ra, &mut rb);
                    std::mem::swap(&mut side_a, &mut side_b);
                }
                let (sa, sb) = (ccs[ra], ccs[rb]);
                let sig = |r: usize| csz[r] >= min_seed && cbirth[r] - l >= life;
                let mut out = -1;
                if !late {
                    if sa >= 0 && sb >= 0 {
                        let (ga, gb) = (sig(ra), sig(rb));
                        if !(ga && gb) {
                            // 짧게 산 쪽을 흡수(둘 다 짧으면 어린 쪽을 나이 많은 쪽에)
                            let (keep, lose) = if ga || !gb { (sa, sb) } else { (sb, sa) };
                            spar[lose as usize] = keep;
                            out = keep;
                        }
                    } else if sa >= 0 {
                        // 홑성분이 섞인 성분을 만남: 짧게 살았으면 건너편 씨앗에
                        if !sig(ra) {
                            spar[sa as usize] = side_b;
                        }
                    } else if sb >= 0 && !sig(rb) {
                        spar[sb as usize] = side_a;
                    }
                } else {
                    let (st_a, st_b) = (sa < 0 || strong[sa as usize], sb < 0 || strong[sb as usize]);
                    if st_a && !st_b {
                        spar[sb as usize] = side_a;
                        out = sa;
                    } else if !st_a && st_b {
                        spar[sa as usize] = side_b;
                        out = sb;
                    } else if !st_a && !st_b {
                        spar[sb as usize] = sa;
                        out = sa;
                    }
                }
                cpar[rb] = ra as i32;
                csz[ra] += csz[rb];
                ccs[ra] = out;
                rc = ra;
            }
        }
    }
    if !late {
        go_late(&mut cpar, &spar, &csz, &ccs, &mut strong);
    }
    // 씨앗 → 방 번호: 방이 되는 씨앗(태어남 순), 그다음 끝까지 약한 덩이 중 min_room 이상
    let nbirth = cpar.len();
    let mut reg = vec![0u32; nbirth];
    let mut nreg = 0u32;
    for i in 0..nbirth {
        if spar[i] == i as i32 && strong[i] {
            nreg += 1;
            reg[i] = nreg;
        }
    }
    for i in 0..nbirth {
        if spar[i] == i as i32 && !strong[i] && csz[find(&mut cpar, i as i32) as usize] >= min_room {
            nreg += 1;
            reg[i] = nreg;
        }
    }
    for i in 0..nbirth {
        reg[i] = reg[find(&mut spar, i as i32) as usize];
    }
    for i in 0..n_cells {
        if b[i] >= 0 {
            s.ids[i] = reg[b[i] as usize];
        }
    }
    for &c in &rim {
        let (x, y) = ((c % w).clamp(1, w - 2), (c / w).clamp(1, h - 2));
        s.ids[c] = s.ids[y * w + x];
    }
    // 방마다 모음(넓이·무게중심·최대 여유)과 이음매(맞닿은 칸 쌍)
    #[derive(Clone, Default)]
    struct Acc {
        n: usize,
        sx: f64,
        sy: f64,
        cmax: f32,
    }
    let nr = nreg as usize;
    let mut acc = vec![Acc::default(); nr + 1];
    let mut seams: BTreeMap<(u32, u32), Acc> = BTreeMap::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let a = s.ids[i];
            if a == 0 {
                continue;
            }
            let r = &mut acc[a as usize];
            r.n += 1;
            r.sx += x as f64;
            r.sy += y as f64;
            r.cmax = r.cmax.max(clear[i]);
            for (ok, j, px, py) in [(x + 1 < w, i + 1, x as f64 + 0.5, y as f64), (y + 1 < h, i + w, x as f64, y as f64 + 0.5)] {
                if ok && s.ids[j] != 0 && s.ids[j] != a {
                    let t = seams.entry((a.min(s.ids[j]), a.max(s.ids[j]))).or_default();
                    t.n += 1;
                    t.sx += px;
                    t.sy += py;
                    t.cmax = t.cmax.max(clear[i].min(clear[j]));
                }
            }
        }
    }
    // 합치기: 긴 이음매, 그다음 가장 작은 방부터 이음매가 가장 긴 이웃에(이웃 없으면 버림)
    let mut rg: Vec<i32> = (0..=nr as i32).collect();
    let mut area: Vec<usize> = acc.iter().map(|a| a.n).collect();
    let max_door = if p.max_door_m > 0.0 { (p.max_door_m / res).round() as usize } else { usize::MAX };
    for (&(a, bb), t) in &seams {
        if t.n >= max_door {
            let (a, bb) = (find(&mut rg, a as i32), find(&mut rg, bb as i32));
            if a != bb {
                rg[bb as usize] = a;
                area[a as usize] += area[bb as usize];
            }
        }
    }
    let mut dropped = vec![false; nr + 1];
    loop {
        let mut small = 0usize;
        for r in 1..=nr {
            if find(&mut rg, r as i32) == r as i32 && !dropped[r] && area[r] < min_room as usize && (small == 0 || area[r] < area[small]) {
                small = r;
            }
        }
        if small == 0 {
            break;
        }
        let mut nbn: BTreeMap<i32, usize> = BTreeMap::new();
        for (&(a, bb), t) in &seams {
            let (a, bb) = (find(&mut rg, a as i32), find(&mut rg, bb as i32));
            if a == bb {
                continue;
            }
            if a == small as i32 {
                *nbn.entry(bb).or_default() += t.n;
            } else if bb == small as i32 {
                *nbn.entry(a).or_default() += t.n;
            }
        }
        match nbn.iter().max_by_key(|&(&r, &n)| (n, -r)) {
            Some((&best, _)) => {
                rg[small] = best;
                area[best as usize] += area[small];
            }
            None => dropped[small] = true,
        }
    }
    // 다시 번호(1..n)
    let mut remap = vec![0u32; nr + 1];
    let mut n = 0u32;
    for r in 1..=nr {
        let root = find(&mut rg, r as i32) as usize;
        if dropped[root] {
            continue;
        }
        if remap[root] == 0 {
            n += 1;
            remap[root] = n;
        }
        remap[r] = remap[root];
    }
    for v in s.ids.iter_mut() {
        *v = remap[*v as usize];
    }
    let mut ra = vec![Acc::default(); n as usize];
    for r in 1..=nr {
        if remap[r] != 0 {
            let (t, q) = (&mut ra[remap[r] as usize - 1], &acc[r]);
            t.n += q.n;
            t.sx += q.sx;
            t.sy += q.sy;
            t.cmax = t.cmax.max(q.cmax);
        }
    }
    s.rooms = ra.iter().enumerate().map(|(i, q)| Room {
        id: i as u32 + 1,
        n_cells: q.n,
        area_m2: q.n as f64 * a1,
        centroid: [(gx0 as f64 + q.sx / q.n.max(1) as f64 + 0.5) * res, (gy0 as f64 + q.sy / q.n.max(1) as f64 + 0.5) * res],
        max_clear: q.cmax as f64,
    }).collect();
    // 문: 합친 방 안쪽 이음매는 사라지고, 남은 이음매를 새 번호로 모음
    let mut m2: BTreeMap<(u32, u32), Acc> = BTreeMap::new();
    for (&(a, bb), q) in &seams {
        let (a, bb) = (remap[a as usize], remap[bb as usize]);
        if a == 0 || bb == 0 || a == bb {
            continue;
        }
        let t = m2.entry((a.min(bb), a.max(bb))).or_default();
        t.n += q.n;
        t.sx += q.sx;
        t.sy += q.sy;
        t.cmax = t.cmax.max(q.cmax);
    }
    s.doors = m2.iter().filter(|(_, t)| t.n >= 2).map(|(&(a, b), t)| Door {
        a,
        b,
        pos: [(gx0 as f64 + t.sx / t.n as f64 + 0.5) * res, (gy0 as f64 + t.sy / t.n as f64 + 0.5) * res],
        width: 2.0 * t.cmax as f64 + res,
        seam: t.n,
    }).collect();
    s
}

/// 이전 나눔과 칸 겹침으로 cur 의 방 번호를 잇는다(겹침 큰 짝부터, 짝 없으면 새 번호).
pub fn match_ids(cur: &mut Seg, prev: Option<&Seg>, next_id: &mut u32, match_min: f64) {
    let n = cur.rooms.len();
    let mut nid = vec![0u32; n + 1];
    if let Some(prev) = prev.filter(|q| !q.rooms.is_empty()) {
        let m = prev.rooms.len();
        let pidx: BTreeMap<u32, usize> = prev.rooms.iter().enumerate().map(|(j, r)| (r.id, j)).collect();
        let mut ov = vec![0usize; n * m];
        let (dx, dy) = ((cur.gx0 - prev.gx0) as i64, (cur.gy0 - prev.gy0) as i64);
        for y in 0..cur.h {
            let py = y as i64 + dy;
            if py < 0 || py >= prev.h as i64 {
                continue;
            }
            for x in 0..cur.w {
                let a = cur.ids[y * cur.w + x];
                let px = x as i64 + dx;
                if a == 0 || px < 0 || px >= prev.w as i64 {
                    continue;
                }
                if let Some(&j) = pidx.get(&prev.ids[py as usize * prev.w + px as usize]) {
                    ov[(a as usize - 1) * m + j] += 1;
                }
            }
        }
        let mut pairs = vec![];
        for i in 0..n {
            for j in 0..m {
                let o = ov[i * m + j];
                if o > 0 && o as f64 >= match_min * cur.rooms[i].n_cells.min(prev.rooms[j].n_cells) as f64 {
                    pairs.push((o, i, j));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let (mut usedc, mut usedp) = (vec![false; n], vec![false; m]);
        for (_, i, j) in pairs {
            if !usedc[i] && !usedp[j] {
                usedc[i] = true;
                usedp[j] = true;
                nid[i + 1] = prev.rooms[j].id;
            }
        }
    }
    for v in nid.iter_mut().skip(1) {
        if *v == 0 {
            *v = *next_id;
            *next_id += 1;
        }
    }
    for v in cur.ids.iter_mut() {
        *v = nid[*v as usize];
    }
    for r in cur.rooms.iter_mut() {
        r.id = nid[r.id as usize];
    }
    for d in cur.doors.iter_mut() {
        let (a, b) = (nid[d.a as usize], nid[d.b as usize]);
        (d.a, d.b) = (a.min(b), a.max(b));
    }
    cur.rooms.sort_by_key(|r| r.id);
    cur.doors.sort_by_key(|d| (d.a, d.b));
}

/// 물체가 든 방 번호(0 = 없음): 바닥 자리의 방 칸 다수결, 없으면 가까운 방 칸.
pub fn assign_object(s: &Seg, pos: [f64; 3], ext: [f64; 3], p: &Params) -> u32 {
    if s.ids.is_empty() {
        return 0;
    }
    let res = s.res;
    let cellx = |v: f64| (v / res).floor() as i64 - s.gx0 as i64;
    let celly = |v: f64| (v / res).floor() as i64 - s.gy0 as i64;
    let (hx, hy) = ((ext[0] / 2.0).max(0.1) + p.footprint_margin, (ext[1] / 2.0).max(0.1) + p.footprint_margin);
    let (x0, x1) = (cellx(pos[0] - hx).max(0), cellx(pos[0] + hx).min(s.w as i64 - 1));
    let (y0, y1) = (celly(pos[1] - hy).max(0), celly(pos[1] + hy).min(s.h as i64 - 1));
    let mut votes: BTreeMap<u32, usize> = BTreeMap::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let v = s.ids[y as usize * s.w + x as usize];
            if v != 0 {
                *votes.entry(v).or_default() += 1;
            }
        }
    }
    if let Some((&id, _)) = votes.iter().max_by_key(|&(&id, &n)| (n, std::cmp::Reverse(id))) {
        return id;
    }
    // 가까운 방 칸(고리 넓히기)
    let (cx, cy) = (cellx(pos[0]), celly(pos[1]));
    let rmax = (p.obj_search_m / res).ceil() as i64;
    let (mut best, mut bd) = (0u32, i64::MAX);
    for r in 1..=rmax {
        for y in cy - r..=cy + r {
            for x in cx - r..=cx + r {
                if (x - cx).abs().max((y - cy).abs()) != r || x < 0 || y < 0 || x >= s.w as i64 || y >= s.h as i64 {
                    continue;
                }
                let v = s.ids[y as usize * s.w + x as usize];
                let d = (x - cx).pow(2) + (y - cy).pow(2);
                if v != 0 && d < bd && d <= rmax * rmax {
                    bd = d;
                    best = v;
                }
            }
        }
        if best != 0 && bd <= r * r {
            break; // 다음 고리는 더 멀다
        }
    }
    best
}

/// 머리 명사 규칙(이름 == key 이거나 " key" 로 끝남, 끝 's' 무시): (key, 방 종류, 점수). 한 물체가 여러 key 에 맞으면 가장 긴 key.
const RULES: &[(&str, &str, f64)] = &[
    ("refrigerator", "kitchen", 2.0), ("fridge", "kitchen", 3.0), ("oven", "kitchen", 2.0), ("stove", "kitchen", 3.0), ("cooktop", "kitchen", 3.0),
    ("range hood", "kitchen", 2.0), ("microwave", "kitchen", 2.0), ("dishwasher", "kitchen", 3.0), ("toaster", "kitchen", 2.0), ("kettle", "kitchen", 1.0),
    ("coffee maker", "kitchen", 1.0), ("kitchen sink", "kitchen", 3.0), ("sink", "kitchen", 1.5), ("bowl", "kitchen", 0.5), ("cup", "kitchen", 0.3),
    ("bottle", "kitchen", 0.3), ("knife", "kitchen", 0.5), ("spoon", "kitchen", 0.5), ("fork", "kitchen", 0.5),
    ("toilet", "bathroom", 3.0), ("bathtub", "bathroom", 3.0), ("bath", "bathroom", 2.0), ("shower", "bathroom", 3.0), ("bathroom sink", "bathroom", 3.0),
    ("sink", "bathroom", 1.0), ("toothbrush", "bathroom", 2.0), ("hair drier", "bathroom", 2.0),
    ("bed", "bedroom", 3.0), ("nightstand", "bedroom", 2.0), ("wardrobe", "bedroom", 1.0), ("dresser", "bedroom", 1.0), ("pillow", "bedroom", 1.0),
    ("teddy bear", "bedroom", 0.5), ("clock", "bedroom", 0.3),
    ("sofa", "living room", 3.0), ("couch", "living room", 2.0), ("tv", "living room", 0.8), ("television", "living room", 2.0),
    ("coffee table", "living room", 2.0), ("armchair", "living room", 1.0), ("fireplace", "living room", 1.0), ("remote", "living room", 1.0),
    ("potted plant", "living room", 0.5), ("vase", "living room", 0.5),
    ("desk", "office", 2.0), ("monitor", "office", 2.0), ("office chair", "office", 2.0), ("computer", "office", 1.0), ("printer", "office", 1.0),
    ("tv", "office", 1.0), ("laptop", "office", 1.5), ("keyboard", "office", 1.5), ("mouse", "office", 1.5), ("book", "office", 0.5),
    ("chair", "office", 0.5), ("cell phone", "office", 0.5),
    ("dining table", "dining room", 1.5), ("chair", "dining room", 0.5), ("wine glass", "dining room", 1.0), ("cup", "dining room", 0.3),
    ("bowl", "dining room", 0.3),
];
const PRIOR: f64 = 1.5; // 모름 몫(점수 단위)
const NAME_MIN: f64 = 2.0; // 이 점수 이상이어야 종류 이름

fn norm(name: &str) -> String {
    name.split(".n.").next().unwrap_or("").replace('_', " ").to_lowercase().trim().to_string()
}

fn head(n: &str, key: &str) -> bool {
    let ends = |a: &str| a == key || a.strip_suffix(key).is_some_and(|rest| rest.ends_with(' '));
    ends(n) || (n.len() > 1 && n.strip_suffix('s').is_some_and(ends))
}

/// 물체 이름이 가리키는 방 종류들: (종류, 점수)
pub fn cues(name: &str) -> Vec<(&'static str, f64)> {
    let n = norm(name);
    let len = RULES.iter().filter(|r| head(&n, r.0)).map(|r| r.0.len()).max().unwrap_or(0);
    RULES.iter().filter(|r| r.0.len() == len && head(&n, r.0)).map(|r| (r.1, r.2)).collect()
}

#[derive(Clone, Default)]
pub struct Label {
    pub name: String,                     // "kitchen", "kitchen 2", "room 7"
    pub kind: String,                     // "kitchen" …, "" = 모름
    pub conf: f32,                        // 가장 그럴듯한 종류의 확률
    pub probs: BTreeMap<String, f64>,     // 종류별 확률 + "unknown"
    pub evidence: Vec<(u32, String, &'static str, f64)>, // (물체 id, 물체 이름, 가리킨 종류, 점수)
    pub objects: Vec<u32>,
}

/// objs: (id, 이름, 위치, 크기). 반환: 방마다 이름표(s.rooms 와 같은 순서), 물체마다 방 번호(0 = 없음).
pub fn name_rooms(s: &Seg, objs: &[(u32, &str, [f64; 3], [f64; 3])], p: &Params) -> (Vec<Label>, Vec<u32>) {
    let mut labels = vec![Label::default(); s.rooms.len()];
    let mut score: Vec<BTreeMap<&'static str, f64>> = vec![BTreeMap::new(); s.rooms.len()];
    let mut obj_room = vec![0u32; objs.len()];
    for (i, o) in objs.iter().enumerate() {
        let rid = assign_object(s, o.2, o.3, p);
        obj_room[i] = rid;
        let Some(ri) = s.index(rid) else { continue };
        labels[ri].objects.push(o.0);
        for (kind, w) in cues(o.1) {
            *score[ri].entry(kind).or_default() += w;
            labels[ri].evidence.push((o.0, o.1.to_string(), kind, w));
        }
    }
    let mut used: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, l) in labels.iter_mut().enumerate() {
        let tot = PRIOR + score[i].values().sum::<f64>();
        let (mut best, mut bt) = (0.0, "");
        for (&t, &v) in &score[i] {
            l.probs.insert(t.to_string(), v / tot);
            if v > best {
                (best, bt) = (v, t);
            }
        }
        l.probs.insert("unknown".into(), PRIOR / tot);
        l.conf = (best / tot) as f32;
        if best >= NAME_MIN {
            let k = used.entry(bt).or_default();
            *k += 1;
            l.kind = bt.to_string();
            l.name = if *k == 1 { bt.to_string() } else { format!("{bt} {k}") };
        } else {
            l.name = format!("room {}", s.rooms[i].id);
        }
    }
    (labels, obj_room)
}

/// 방 번호를 실행 내내 잇는 상태: 지난 나눔과 다음 새 번호.
#[derive(Default)]
pub struct Tracker {
    pub params: Params,
    prev: Option<Seg>,
    next_id: u32,
    seen: (Vec<u8>, Vec<bool>), // 지난번에 나눈 지도 그림
}

impl Tracker {
    /// 점유 지도 그림(위가 +y)으로 다시 나누고 지난 나눔과 번호를 잇는다. 지도가 지난번과 같으면 지난 나눔 그대로.
    pub fn update(&mut self, g: &GridImg, res: f64) -> &Seg {
        let (w, h) = (g.w, g.h);
        if self.prev.as_ref().is_some_and(|p| (p.w, p.h, p.gx0, p.gy0) == (w, h, g.x0, g.y0)) && self.seen.0 == g.px() && self.seen.1 == g.wall {
            return self.prev.as_ref().unwrap();
        }
        self.seen.0.clear();
        self.seen.0.extend_from_slice(g.px());
        self.seen.1.clone_from(&g.wall);
        let mut cls = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (h - 1 - y) * w + x;
                cls[y * w + x] = match g.px()[i] {
                    254 => 1,
                    0 => 2 - !g.wall[i] as u8, // 낮은 막힘(가구)은 방 안
                    _ => 0,
                };
            }
        }
        let mut s = segment(&cls, w, h, res, g.x0, g.y0, &self.params);
        self.next_id = self.next_id.max(1);
        match_ids(&mut s, self.prev.as_ref(), &mut self.next_id, self.params.match_min);
        self.prev.insert(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // cargo test --release -- --ignored --nocapture
    fn timing() {
        let (w, h) = (402, 402); // 20 m × 20 m, 방 넷
        let c = plan(w, h, &[(200, 0, 201, 180), (200, 197, 201, 401), (0, 200, 90, 201), (107, 200, 290, 201), (307, 200, 401, 201)]);
        let t = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..20 {
            n = segment(&c, w, h, 0.05, 0, 0, &Params::default()).rooms.len();
        }
        println!("방 {n} 개, 한 번 {:.2} ms", t.elapsed().as_secs_f64() * 50.0);
    }

    /// 벽으로 둘러싼 w×h 칸 빈 방. walls: 막을 칸 상자 (x0, y0, x1, y1) 포함 범위
    fn plan(w: usize, h: usize, walls: &[(usize, usize, usize, usize)]) -> Vec<u8> {
        let mut c = vec![1u8; w * h];
        for &(x0, y0, x1, y1) in [(0, 0, w - 1, 0), (0, h - 1, w - 1, h - 1), (0, 0, 0, h - 1), (w - 1, 0, w - 1, h - 1)].iter().chain(walls) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    c[y * w + x] = 2;
                }
            }
        }
        c
    }

    #[test]
    fn clearance_is_exact_distance() {
        // 9×9 빈칸 한가운데: 가장 가까운 막힌 칸 중심까지 5 칸 → 여유 5·res − res/2
        let c = plan(11, 11, &[]);
        let free: Vec<bool> = c.iter().map(|&v| v == 1).collect();
        let cl = clearance(&free, 11, 11, 0.05);
        assert!((cl[5 * 11 + 5] - 0.225).abs() < 1e-5);
        assert!((cl[11 + 1] - 0.025).abs() < 1e-5);
        assert_eq!(cl[0], 0.0);
    }

    #[test]
    fn door_splits_two_rooms() {
        // 4.5 m × 5 m 방 둘, 가운데 벽에 0.8 m 문
        let (w, h) = (182, 102);
        let c = plan(w, h, &[(90, 0, 91, 42), (90, 59, 91, 101)]);
        let s = segment(&c, w, h, 0.05, 0, 0, &Params::default());
        assert_eq!(s.rooms.len(), 2);
        assert_eq!(s.doors.len(), 1);
        let d = &s.doors[0];
        assert!((d.width - 0.8).abs() < 0.15, "문 폭 {}", d.width);
        assert!((d.pos[0] - 4.55).abs() < 0.3 && (d.pos[1] - 2.55).abs() < 0.3, "문 위치 {:?}", d.pos);
        assert_ne!(s.ids[50 * w + 40], 0);
        assert_ne!(s.ids[50 * w + 40], s.ids[50 * w + 140]);
        // 문 없는 한 방, 그리고 2 m 로 뚫린 벽(문보다 넓음)은 방 하나
        assert_eq!(segment(&plan(w, h, &[]), w, h, 0.05, 0, 0, &Params::default()).rooms.len(), 1);
        let wide = plan(w, h, &[(90, 0, 91, 30), (90, 71, 91, 101)]);
        assert_eq!(segment(&wide, w, h, 0.05, 0, 0, &Params::default()).rooms.len(), 1);
    }

    #[test]
    fn small_holes_and_specks_do_not_split() {
        let (w, h) = (122, 82);
        let mut c = plan(w, h, &[]);
        for (x, y) in [(40, 40), (41, 41), (80, 20)] {
            c[y * w + x] = 2; // 의자 다리
        }
        for y in 30..34 {
            for x in 60..64 {
                c[y * w + x] = 0; // 못 본 바닥
            }
        }
        let s = segment(&c, w, h, 0.05, 0, 0, &Params::default());
        assert_eq!(s.rooms.len(), 1);
        assert!((s.rooms[0].area_m2 - 120.0 * 80.0 * 0.0025).abs() < 0.05);
    }

    #[test]
    fn ids_survive_growth_and_shift() {
        let (w, h) = (182, 102);
        let p = Params::default();
        let mut next = 1;
        // 처음엔 왼쪽 방만 보임
        let mut a = segment(&plan(90, h, &[]), 90, h, 0.05, 0, 0, &p);
        match_ids(&mut a, None, &mut next, p.match_min);
        assert_eq!(a.rooms.iter().map(|r| r.id).collect::<Vec<_>>(), [1]);
        // 격자가 왼쪽·아래로 넓어지고(원점 이동) 오른쪽 방이 새로 보임
        let mut big = vec![0u8; (w + 20) * (h + 10)];
        let c = plan(w, h, &[(90, 0, 91, 42), (90, 59, 91, 101)]);
        for y in 0..h {
            big[(y + 10) * (w + 20) + 20..(y + 10) * (w + 20) + 20 + w].copy_from_slice(&c[y * w..(y + 1) * w]);
        }
        let mut b = segment(&big, w + 20, h + 10, 0.05, -20, -10, &p);
        match_ids(&mut b, Some(&a), &mut next, p.match_min);
        assert_eq!(b.rooms.len(), 2);
        assert_eq!(b.ids[(50 + 10) * (w + 20) + 40 + 20], 1);
        assert_eq!(b.ids[(50 + 10) * (w + 20) + 140 + 20], 2);
        assert_eq!((b.doors[0].a, b.doors[0].b), (1, 2));
    }

    #[test]
    fn names_from_objects() {
        let (w, h) = (182, 102);
        let c = plan(w, h, &[(90, 0, 91, 42), (90, 59, 91, 101)]);
        let p = Params::default();
        let s = segment(&c, w, h, 0.05, 0, 0, &p);
        let e = [0.3, 0.3, 0.3];
        let (left, right) = (s.ids[50 * w + 40], s.ids[50 * w + 140]);
        let objs = [(1, "refrigerator", [1.0, 1.0, 0.5], e), (2, "sink", [2.0, 1.0, 0.5], e), (3, "oven", [7.0, 2.0, 0.5], e),
                    (4, "wine_glass.n.01", [7.5, 2.0, 0.5], e), (5, "chairs", [8.0, 3.0, 0.5], e)];
        let (lab, room) = name_rooms(&s, &objs, &p);
        assert_eq!(room, [left, left, right, right, right]);
        let (l, r) = (&lab[s.index(left).unwrap()], &lab[s.index(right).unwrap()]);
        assert_eq!((l.name.as_str(), r.name.as_str()), if left < right { ("kitchen", "kitchen 2") } else { ("kitchen 2", "kitchen") });
        // 왼쪽: kitchen 2 + 1.5, bathroom 1, 모름 1.5
        assert!((l.probs["kitchen"] - 3.5 / 6.0).abs() < 1e-9 && (l.probs["unknown"] - 0.25).abs() < 1e-9);
        assert!((r.probs["dining room"] - 1.5 / 5.5).abs() < 1e-9); // wine glass 1 + chair 0.5 / (모름 1.5 + oven 2 + 1 + chair 0.5·2)
        assert_eq!(l.evidence.len(), 3);
        // 단서 없는 방은 번호 이름
        let (lab, _) = name_rooms(&s, &[(1, "person", [1.0, 1.0, 0.5], e)], &p);
        assert_eq!(lab[0].name, format!("room {}", s.rooms[0].id));
        assert_eq!(lab[0].conf, 0.0);
        // 가장 긴 key: office chair 는 chair 규칙을 타지 않음
        assert_eq!(cues("office chair"), [("office", 2.0)]);
        assert_eq!(cues("armchair"), [("living room", 1.0)]);
        assert!(cues("bedside lamp").is_empty());
    }
}
