//! Walls in 2D: occupancy grid → wall segments → robot-frame numbers (wall state vector).
//!
//! Grid layout: cells[y·w + x] is cell (x, y), its lower-left corner = origin + (x, y)·res; −1 unknown, 0..100 occupied %.
//! Occupied = cells ≥ OCC_MIN (65 %).
//!
//! Wall state vector (f32, length STATE_LEN = SECTORS + SEGMENTS·5), robot frame (x forward, y left):
//!   [0 : SECTORS]           distance to the first occupied cell in sector i / MAX_RANGE. Sector 0 = ahead, counter-clockwise,
//!                           spacing 2π/SECTORS. 1.0 = nothing in range.
//!   [SECTORS + 5j : +5]     j-th nearest wall segment: ax, ay, bx, by (m / MAX_RANGE, clamped to [−1, 1]), valid (0/1). Unused = 0.
//!
//! Floating-point work keeps the exact types and operation order of the reference (scenemap walls.cpp), so the numbers
//! match it to the last bit; min/max/clamp below follow std:: semantics (first argument wins on ties / NaN).

pub const OCC_MIN: i8 = 65;
pub const MIN_LEN: f64 = 0.5; // shortest wall segment [m]
pub const MAX_THICK: f64 = 0.5; // thicker blobs are furniture / clutter, not walls [m]
pub const SECTORS: usize = 16;
pub const SEGMENTS: usize = 8;
pub const MAX_RANGE: f64 = 4.0; // [m]
pub const STATE_LEN: usize = SECTORS + SEGMENTS * 5;
const OVERLAP: f64 = 0.6;
const ALIGN_TOL: f64 = 1.0 * 3.14159265358979323846 / 180.0;
const ALIGN_GAIN: f64 = 1.08; // wall_angle: below this multiple of the 0° projection score the map counts as axis-aligned
const PI: f64 = std::f64::consts::PI;

/// Occupancy grid view (layout above).
#[derive(Clone, Copy)]
pub struct Grid<'a> {
    pub cells: &'a [i8],
    pub w: i32,
    pub h: i32,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

#[derive(Clone, Copy)]
struct Seg {
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
}

#[inline]
fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}
#[inline]
fn max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}
#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if hi < v { hi } else { v }
}
#[inline]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { lo } else if hi < v { hi } else { v }
}
/// int(std::floor(x)) without a libm call (same value for every x in the i32 range)
#[inline]
fn floor_i(x: f64) -> i32 {
    let t = x as i32;
    if t as f64 > x { t - 1 } else { t }
}
/// int(std::lround(x)): half away from zero, without a libm call (x − trunc(x) is exact, same value for every x in the i32 range)
#[inline]
fn lround(x: f64) -> i32 {
    let t = x as i32;
    let f = x - t as f64;
    if f >= 0.5 { t + 1 } else if f <= -0.5 { t - 1 } else { t }
}

#[derive(Clone, Copy)]
struct Run {
    a: i32,
    b: i32,
}
#[derive(Clone, Copy)]
struct Group {
    r0: i32,
    r1: i32,
    x0: i32,
    x1: i32,
}

/// Occupied cells packed per row (64 cells = one word). Row order is the PGM one: row 0 = top (max y).
struct Bits {
    nw: usize,
    v: Vec<u64>, // [iy·nw + k]
}

impl Bits {
    fn build(g: &Grid) -> Bits {
        let (w, h) = (g.w as usize, g.h as usize);
        let nw = (w + 63) / 64;
        let mut v = vec![0u64; h * nw];
        for iy in 0..h {
            let src = &g.cells[(h - 1 - iy) * w..(h - iy) * w];
            let dst = &mut v[iy * nw..(iy + 1) * nw];
            for (k, ch) in src.chunks(64).enumerate() {
                let mut m = 0u64;
                for (b, &c) in ch.iter().enumerate() {
                    m |= ((c >= OCC_MIN) as u64) << b;
                }
                dst[k] = m;
            }
        }
        Bits { nw, v }
    }
    fn row(&self, iy: usize) -> &[u64] {
        &self.v[iy * self.nw..(iy + 1) * self.nw]
    }
}

/// Appends the occupied runs of one row (nw words), left to right. Starts/ends come word-wise from neighbour bits (empty words skipped).
fn row_runs(row: &[u64], min_run: i32, out: &mut Vec<Run>) {
    let nw = row.len();
    let mut open = -1i32;
    for k in 0..nw {
        let cur = row[k];
        let before = if k > 0 { row[k - 1] >> 63 } else { 0 };
        let after = if k + 1 < nw { row[k + 1] << 63 } else { 0 };
        if cur == 0 {
            continue;
        }
        let mut st = cur & !((cur << 1) | before);
        let mut en = cur & !((cur >> 1) | after);
        while st | en != 0 {
            let sb = if st != 0 { st.trailing_zeros() as i32 } else { 64 };
            let eb = if en != 0 { en.trailing_zeros() as i32 } else { 64 };
            if sb <= eb {
                // a one-cell run starts and ends on the same bit: start first
                open = k as i32 * 64 + sb;
                st &= st - 1;
            }
            if sb >= eb {
                let e = k as i32 * 64 + eb;
                if open >= 0 && e - open + 1 >= min_run {
                    out.push(Run { a: open, b: e });
                }
                open = -1;
                en &= en - 1;
            }
        }
    }
}

/// Joins the runs of `rows` rows (split by `off`) with the run of the row above when they overlap enough.
fn group_runs(runs: &[Run], off: &[i32], rows: usize, overlap: f64, done: &mut Vec<Group>) {
    let mut active: Vec<Group> = Vec::new();
    let mut nxt: Vec<Group> = Vec::new();
    let mut used: Vec<bool> = Vec::new();
    for r in 0..rows {
        let r = r as i32;
        nxt.clear();
        used.clear();
        used.resize(active.len(), false);
        for rn in &runs[off[r as usize] as usize..off[r as usize + 1] as usize] {
            let mut hit = usize::MAX;
            for (gi, ag) in active.iter().enumerate() {
                if used[gi] || ag.r1 != r - 1 {
                    continue;
                }
                let ov = rn.b.min(ag.x1) - rn.a.max(ag.x0) + 1;
                if ov as f64 >= overlap * ((rn.b - rn.a + 1).min(ag.x1 - ag.x0 + 1)) as f64 {
                    hit = gi;
                    break;
                }
            }
            if hit == usize::MAX {
                nxt.push(Group { r0: r, r1: r, x0: rn.a, x1: rn.b });
            } else {
                let mut g = active[hit];
                used[hit] = true;
                g.r1 = r;
                g.x0 = g.x0.min(rn.a);
                g.x1 = g.x1.max(rn.b);
                nxt.push(g);
            }
        }
        for (gi, ag) in active.iter().enumerate() {
            if !used[gi] {
                done.push(*ag);
            }
        }
        std::mem::swap(&mut active, &mut nxt);
    }
    done.extend_from_slice(&active);
}

/// Axis-aligned (horizontal, vertical) wall centre lines from the occupied cells.
fn axis_segments(g: &Grid, min_len: f64, max_thick: f64, overlap: f64) -> Vec<Seg> {
    let mut out = Vec::new();
    if g.w <= 0 || g.h <= 0 {
        return out;
    }
    let min_run = 1.max(lround(min_len / g.res));
    let bits = Bits::build(g);
    let (w, h) = (g.w as usize, g.h as usize);
    let emit = |out: &mut Vec<Seg>, gr: &Group, transpose: bool| {
        if (gr.r1 - gr.r0 + 1) as f64 * g.res > max_thick {
            return;
        }
        let rc = (gr.r0 + gr.r1) as f64 / 2.0;
        let mut p = [[0f64; 2]; 2];
        for (k, c) in [gr.x0, gr.x1].into_iter().enumerate() {
            let (iy, ix) = if transpose { (c as f64, rc) } else { (rc, c as f64) };
            p[k][0] = g.ox + (ix + 0.5) * g.res;
            p[k][1] = g.oy + ((g.h - 1) as f64 - iy + 0.5) * g.res;
        }
        out.push(Seg { ax: p[0][0], ay: p[0][1], bx: p[1][0], by: p[1][1] });
    };
    let mut groups = Vec::new();
    // horizontal walls: runs per row
    {
        let mut runs = Vec::new();
        let mut off = vec![0i32; h + 1];
        for iy in 0..h {
            row_runs(bits.row(iy), min_run, &mut runs);
            off[iy + 1] = runs.len() as i32;
        }
        group_runs(&runs, &off, h, overlap, &mut groups);
        for gr in &groups {
            emit(&mut out, gr, false);
        }
    }
    // vertical walls: runs per column without transposing. Walking down the rows, only the columns that start/end here are picked as bits.
    {
        struct VRun {
            x: i32,
            a: i32,
            b: i32,
        }
        let mut vr: Vec<VRun> = Vec::new();
        let mut start = vec![0i32; w + 64];
        let zeros = vec![0u64; bits.nw];
        for iy in 0..=h {
            let cur = if iy < h { bits.row(iy) } else { &zeros[..] };
            let prev = if iy > 0 { bits.row(iy - 1) } else { &zeros[..] };
            for k in 0..bits.nw {
                let mut st = cur[k] & !prev[k];
                let mut en = prev[k] & !cur[k];
                while st != 0 {
                    start[k * 64 + st.trailing_zeros() as usize] = iy as i32;
                    st &= st - 1;
                }
                while en != 0 {
                    let x = k * 64 + en.trailing_zeros() as usize;
                    if iy as i32 - start[x] >= min_run {
                        vr.push(VRun { x: x as i32, a: start[x], b: iy as i32 - 1 });
                    }
                    en &= en - 1;
                }
            }
        }
        // by column (rows of the transposed grid); within a column they are already top to bottom
        let mut voff = vec![0i32; w + 1];
        for r in &vr {
            voff[r.x as usize + 1] += 1;
        }
        for x in 0..w {
            voff[x + 1] += voff[x];
        }
        let mut vruns = vec![Run { a: 0, b: 0 }; vr.len()];
        let mut pos: Vec<i32> = voff[..w].to_vec();
        for r in &vr {
            vruns[pos[r.x as usize] as usize] = Run { a: r.a, b: r.b };
            pos[r.x as usize] += 1;
        }
        groups.clear();
        group_runs(&vruns, &voff, w, overlap, &mut groups);
        for gr in &groups {
            emit(&mut out, gr, true);
        }
    }
    out
}

/// Main orthogonal wall direction θ (−45°..45°, rad): the angle whose projection histograms are sharpest,
/// 0 unless it is ALIGN_GAIN times sharper than at 0°.
fn wall_angle(g: &Grid) -> f64 {
    let mut px: Vec<f32> = Vec::new();
    let mut py: Vec<f32> = Vec::new();
    for y in 0..g.h {
        for x in 0..g.w {
            if g.cells[y as usize * g.w as usize + x as usize] >= OCC_MIN {
                px.push(x as f32);
                py.push(y as f32);
            }
        }
    }
    if px.len() < 20 {
        return 0.0;
    }
    // sum of squares of the x / y projection histograms (cell bins) of the points turned by −θ: larger when walls are axis-parallel.
    // 90° periodic, so only −45°..45°.
    let dd = ((g.w as f64).hypot(g.h as f64)).ceil() as i32 + 2;
    let nb = (2 * dd + 1) as usize;
    let (mut hx, mut hy) = (vec![0i32; nb], vec![0i32; nb]);
    let mut score = |th: f64| -> f64 {
        hx.fill(0);
        hy.fill(0);
        let c = th.cos() as f32;
        let s = th.sin() as f32;
        let d = dd as f32 + 0.5f32; // d: round without negatives (int truncation)
        let top = nb - 1; // bins stay in 0..=2·D (|turned point| ≤ D − 2); the min only lets the compiler drop the bounds checks
        for (&x, &y) in px.iter().zip(&py) {
            // SAFETY: finite and within 0..=2·D + 1 (points are cell indices, |c|, |s| ≤ 1), far inside the i32 range
            let (bx, by) = unsafe { ((c * x + s * y + d).to_int_unchecked::<i32>(), (-s * x + c * y + d).to_int_unchecked::<i32>()) };
            hx[(bx as usize).min(top)] += 1;
            hy[(by as usize).min(top)] += 1;
        }
        let mut v = 0f64;
        for i in 0..nb {
            v += hx[i] as f64 * hx[i] as f64 + hy[i] as f64 * hy[i] as f64;
        }
        v
    };
    let deg = PI / 180.0;
    let v0 = score(0.0);
    let (mut best, mut bv) = (0.0f64, v0);
    // 1° steps → 0.25° → 0.05°
    for k in -45..45 {
        let th = k as f64 * deg;
        let v = score(th);
        if v > bv {
            bv = v;
            best = th;
        }
    }
    for step in [0.25 * deg, 0.05 * deg] {
        let c0 = best;
        for k in -4..=4 {
            let th = c0 + k as f64 * step;
            let v = score(th);
            if v > bv {
                bv = v;
                best = th;
            }
        }
    }
    if best >= PI / 4.0 {
        best -= PI / 2.0;
    }
    if best < -PI / 4.0 {
        best += PI / 2.0;
    }
    // if turning sharpens it by less than 8 % the map counts as axis-aligned (gt maps: clutter can score −1°..−1.25° slightly higher,
    // ratio ≤ 1.05; tilted slam maps 1.13..3.3). Those take the axis-aligned extraction.
    if best.abs() > ALIGN_TOL && bv < ALIGN_GAIN * v0 {
        return 0.0;
    }
    best
}

/// The grid with the occupied cells inside the ignore boxes (map frame, x0 y0 x1 y1 …) set free. None = no boxes (use g.cells).
fn mask_ignore(g: &Grid, ignore: &[f64]) -> Option<Vec<i8>> {
    if ignore.len() < 4 {
        return None;
    }
    let mut m = g.cells[..g.w as usize * g.h as usize].to_vec();
    for r in ignore.chunks_exact(4) {
        let cx0 = 0.max(floor_i((r[0] - g.ox) / g.res));
        let cx1 = (g.w - 1).min(floor_i((r[2] - g.ox) / g.res));
        let cy0 = 0.max(floor_i((r[1] - g.oy) / g.res));
        let cy1 = (g.h - 1).min(floor_i((r[3] - g.oy) / g.res));
        for y in cy0..=cy1 {
            for x in cx0..=cx1 {
                let i = y as usize * g.w as usize + x as usize;
                if m[i] >= OCC_MIN {
                    m[i] = 0;
                }
            }
        }
    }
    Some(m)
}

/// Wall segments (flat ax ay bx by …, map frame [m]) and the wall direction θ [rad] they were found at.
/// ignore: boxes (x0 y0 x1 y1 …, map frame) whose occupied cells are not walls — furniture on the floor.
/// Axis-aligned maps (|θ| ≤ 1°) take the axis-aligned extraction; tilted maps (slam: map frame = start pose) are turned by −θ,
/// extracted there, and each segment is fitted back onto the occupied band of the original grid before turning it back by θ.
pub fn wall_segments(g: &Grid, ignore: &[f64]) -> (Vec<f64>, f64) {
    if g.w <= 0 || g.h <= 0 {
        return (Vec::new(), 0.0);
    }
    // the ignore boxes are axis boxes of the original frame, so they are cleared first (furniture stays out of the angle too)
    let masked = mask_ignore(g, ignore);
    let gm = Grid { cells: masked.as_deref().unwrap_or(g.cells), ..*g };
    let th = wall_angle(&gm);
    let segs = if th.abs() <= ALIGN_TOL { axis_segments(&gm, MIN_LEN, MAX_THICK, OVERLAP) } else { rotated_segments(&gm, th, MIN_LEN, MAX_THICK, OVERLAP) };
    let mut flat = Vec::with_capacity(segs.len() * 4);
    for s in segs {
        flat.extend_from_slice(&[s.ax, s.ay, s.bx, s.by]);
    }
    (flat, th)
}

#[derive(Clone, Copy)]
struct Line {
    hz: bool, // along u (v = q, u = a0..a1), else along v (u = q, v = a0..a1)
    q: f64,
    a0: f64,
    a1: f64,
}

fn rotated_segments(g: &Grid, th: f64, min_len: f64, max_thick: f64, overlap: f64) -> Vec<Seg> {
    let src = g.cells;
    // turned coordinates u = R(−θ)·p. The four corners of the map, turned, give the new grid extent.
    let (c, s, res) = (th.cos(), th.sin(), g.res);
    let (mut u0, mut v0, mut u1, mut v1) = (1e300f64, 1e300f64, -1e300f64, -1e300f64);
    for k in 0..4 {
        let x = g.ox + (if k & 1 != 0 { g.w } else { 0 }) as f64 * res;
        let y = g.oy + (if k & 2 != 0 { g.h } else { 0 }) as f64 * res;
        let u = c * x + s * y;
        let v = -s * x + c * y;
        u0 = min(u0, u);
        u1 = max(u1, u);
        v0 = min(v0, v);
        v1 = max(v1, v);
    }
    let ww = ((u1 - u0) / res).ceil() as i32 + 1;
    let hh = ((v1 - v0) / res).ceil() as i32 + 1;
    let (gw, gh) = (g.w, g.h);
    let mut rot = vec![-1i8; ww as usize * hh as usize];
    // each new cell centre goes back to the original grid: occupied if any of the 4 surrounding cells (bilinear footprint) is,
    // else the nearest cell's value. Nearest-only would break staircase diagonal walls in the turned grid.
    // c·u and s·u per column are the same products for every row: computed once
    let cu: Vec<f64> = (0..ww).map(|i| c * (u0 + (i as f64 + 0.5) * res)).collect();
    let su: Vec<f64> = (0..ww).map(|i| s * (u0 + (i as f64 + 0.5) * res)).collect();
    for (j, row) in rot.chunks_mut(ww as usize).enumerate() {
        let v = v0 + (j as f64 + 0.5) * res;
        let (sv, cv) = (s * v, c * v);
        for i in 0..ww {
            let fx = (cu[i as usize] - sv - g.ox) / res - 0.5;
            let fy = (su[i as usize] + cv - g.oy) / res - 0.5;
            let x0 = floor_i(fx);
            let y0 = floor_i(fy);
            let mut val = -1i8;
            let mut occ = false;
            for q in 0..4 {
                let x = x0 + (q & 1);
                let y = y0 + (q >> 1);
                if x < 0 || y < 0 || x >= gw || y >= gh {
                    continue;
                }
                if src[y as usize * gw as usize + x as usize] >= OCC_MIN {
                    occ = true;
                }
            }
            if occ {
                val = 100;
            } else {
                let x = lround(fx);
                let y = lround(fy);
                if x >= 0 && y >= 0 && x < gw && y < gh {
                    val = if src[y as usize * gw as usize + x as usize] >= 0 { 0 } else { -1 };
                }
            }
            row[i as usize] = val;
        }
    }
    let gr = Grid { cells: &rot, w: ww, h: hh, res, ox: u0, oy: v0 };
    let raw = axis_segments(&gr, min_len, max_thick, overlap);
    // segments from the turned grid are the middle row of a blob (group_runs), so with staircase-swollen walls, double parallel walls (slam)
    // or blobs cut by max_thick they sit beside the wall band or run past the wall end into free space. So they are refitted on the
    // original grid (furniture cleared):
    //  1) snap: shift parallel within ±3 cells in 1/4-cell steps, measure the fraction of points on the line that hit occupied cells,
    //     and move to the weighted centre of the highest band (connected range ≥ half the peak). Peak < 0.25 → dropped.
    //  2) clip: walk the moved line every half cell (±half a cell across); gaps longer than 4 cells split it, ends shrink to the last
    //     occupied point. Pieces stay if ≥ min_len with occupied fraction ≥ 0.7 (lines through rooms, short spikes across walls drop here).
    //  3) merge: same direction, ≤ 2 cells apart, ranges overlapping or within 1 cell → one line (length-weighted position, then 1·2 again).
    // original cell (x, y) = point (ox + x·res, oy + y·res) plus one cell — corner based, no half-cell shift.
    let occ_uv = |u: f64, v: f64| -> bool {
        let x = c * u - s * v;
        let y = s * u + c * v;
        let ix = floor_i((x - g.ox) / res);
        let iy = floor_i((y - g.oy) / res);
        ix >= 0 && iy >= 0 && ix < gw && iy < gh && src[iy as usize * gw as usize + ix as usize] >= OCC_MIN
    };
    let occ_l = |l: &Line, t: f64, d: f64| if l.hz { occ_uv(t, l.q + d) } else { occ_uv(l.q + d, t) };
    let sstep = res / 2.0;
    let nsamp = |l: &Line| 1.max(lround((l.a1 - l.a0) / sstep));
    let snap = |l: &mut Line| -> bool {
        const K: i32 = 12; // ±3 cells, 1/4-cell steps
        let n = nsamp(l);
        let mut f = [0f64; 2 * K as usize + 1];
        for k in 0..=2 * K {
            let d = (k - K) as f64 * res / 4.0;
            let mut cnt = 0i32;
            for i in 0..=n {
                cnt += occ_l(l, l.a0 + (l.a1 - l.a0) * i as f64 / n as f64, d) as i32;
            }
            f[k as usize] = cnt as f64 / (n + 1) as f64;
        }
        let mut b = K;
        for k in 0..=2 * K {
            if f[k as usize] - 0.01 * (k - K).abs() as f64 > f[b as usize] - 0.01 * (b - K).abs() as f64 {
                b = k;
            }
        }
        let fb = f[b as usize];
        if fb < 0.25 {
            return false;
        }
        let (mut lo, mut hi) = (b, b);
        while lo > 0 && f[lo as usize - 1] >= 0.5 * fb {
            lo -= 1;
        }
        while hi < 2 * K && f[hi as usize + 1] >= 0.5 * fb {
            hi += 1;
        }
        let (mut sw, mut sd) = (0f64, 0f64);
        for k in lo..=hi {
            sw += f[k as usize];
            sd += f[k as usize] * (k - K) as f64 * res / 4.0;
        }
        l.q += sd / sw;
        true
    };
    let clip = |l: &Line, out: &mut Vec<Line>| {
        let n = nsamp(l);
        let max_gap = 8; // 4 cells = 8 half-cell points
        let on = |t: f64| occ_l(l, t, 0.0) || occ_l(l, t, -res / 2.0) || occ_l(l, t, res / 2.0);
        let (mut first, mut last, mut cnt, mut gap) = (-1i32, -1i32, 0i32, 0i32);
        let close = |first: &mut i32, last: &mut i32, cnt: &mut i32, gap: &mut i32, out: &mut Vec<Line>| {
            if *first >= 0 {
                let t0 = l.a0 + (l.a1 - l.a0) * *first as f64 / n as f64;
                let t1 = l.a0 + (l.a1 - l.a0) * *last as f64 / n as f64;
                if t1 - t0 >= min_len - res - 1e-9 && *cnt as f64 >= 0.7 * (*last - *first + 1) as f64 {
                    out.push(Line { hz: l.hz, q: l.q, a0: t0, a1: t1 });
                }
            }
            *first = -1;
            *last = -1;
            *cnt = 0;
            *gap = 0;
        };
        for i in 0..=n {
            if on(l.a0 + (l.a1 - l.a0) * i as f64 / n as f64) {
                if first < 0 {
                    first = i;
                }
                last = i;
                cnt += 1;
                gap = 0;
            } else if first >= 0 && {
                gap += 1;
                gap > max_gap
            } {
                close(&mut first, &mut last, &mut cnt, &mut gap, out);
            }
        }
        close(&mut first, &mut last, &mut cnt, &mut gap, out);
    };
    let refine = |mut l: Line, out: &mut Vec<Line>| {
        if !snap(&mut l) {
            return;
        }
        let mut pcs = Vec::new();
        clip(&l, &mut pcs);
        for mut p in pcs {
            if snap(&mut p) {
                out.push(p); // the clipped pieces are snapped once more
            }
        }
    };
    let mut lines: Vec<Line> = Vec::new();
    for w in &raw {
        let hz = w.ay == w.by;
        let l = if hz { Line { hz: true, q: w.ay, a0: min(w.ax, w.bx), a1: max(w.ax, w.bx) } } else { Line { hz: false, q: w.ax, a0: min(w.ay, w.by), a1: max(w.ay, w.by) } };
        refine(l, &mut lines);
    }
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..lines.len() {
            for j in i + 1..lines.len() {
                let (a, b) = (lines[i], lines[j]);
                if a.hz != b.hz || (a.q - b.q).abs() > 2.0 * res + 1e-9 || min(a.a1, b.a1) - max(a.a0, b.a0) < -res - 1e-9 {
                    continue;
                }
                let la = a.a1 - a.a0 + res;
                let lb = b.a1 - b.a0 + res;
                let m = Line { hz: a.hz, q: (a.q * la + b.q * lb) / (la + lb), a0: min(a.a0, b.a0), a1: max(a.a1, b.a1) };
                let mut pcs = Vec::new();
                refine(m, &mut pcs);
                lines.remove(j);
                lines.remove(i);
                lines.splice(i..i, pcs);
                changed = true; // each merge drops a line (≤ 1 piece) or splits into pieces > 4 cells apart that never merge again
                break 'outer;
            }
        }
    }
    let all: Vec<Seg> = lines.iter().map(|l| if l.hz { Seg { ax: l.a0, ay: l.q, bx: l.a1, by: l.q } } else { Seg { ax: l.q, ay: l.a0, bx: l.q, by: l.a1 } }).collect();
    // short parallel leftovers: dropped when within max_thick/2 across of a longer parallel segment and inside its range (1 cell slack)
    let mut segs = Vec::new();
    {
        let tol = max_thick / 2.0;
        let slack = res;
        let horiz = |s: &Seg| s.ay == s.by;
        for (i, a) in all.iter().enumerate() {
            let ha = horiz(a);
            let alen = if ha { (a.bx - a.ax).abs() } else { (a.by - a.ay).abs() };
            let a0 = if ha { min(a.ax, a.bx) } else { min(a.ay, a.by) };
            let a1 = if ha { max(a.ax, a.bx) } else { max(a.ay, a.by) };
            let mut shadow = false;
            for (j, b) in all.iter().enumerate() {
                if shadow {
                    break;
                }
                if j == i || horiz(b) != ha {
                    continue;
                }
                let blen = if ha { (b.bx - b.ax).abs() } else { (b.by - b.ay).abs() };
                if blen < alen || (blen == alen && j > i) {
                    continue;
                }
                let off = if ha { (a.ay - b.ay).abs() } else { (a.ax - b.ax).abs() };
                let b0 = if ha { min(b.ax, b.bx) } else { min(b.ay, b.by) };
                let b1 = if ha { max(b.ax, b.bx) } else { max(b.ay, b.by) };
                shadow = off <= tol && a0 >= b0 - slack && a1 <= b1 + slack;
            }
            if !shadow {
                segs.push(*a);
            }
        }
    }
    for w in segs.iter_mut() {
        // p = R(θ)·u
        *w = Seg { ax: c * w.ax - s * w.ay, ay: s * w.ax + c * w.ay, bx: c * w.bx - s * w.by, by: s * w.bx + c * w.by };
    }
    segs
}

/// Distance [m] along n rays from pose (x, y, yaw) to the first occupied cell, max_range if none (grid DDA, one step per cell boundary).
fn ray_distances(g: &Grid, pose: &[f64; 3], out: &mut [f32], max_range: f64) {
    let (x, y, yaw) = (pose[0], pose[1], pose[2]);
    let n = out.len();
    for i in 0..n {
        let th = yaw + 2.0 * PI * i as f64 / n as f64;
        let (dx, dy) = (th.cos(), th.sin());
        let fx = (x - g.ox) / g.res;
        let fy = (y - g.oy) / g.res;
        let mut cx = floor_i(fx);
        let mut cy = floor_i(fy);
        let sx = if dx > 0.0 { 1 } else { -1 };
        let sy = if dy > 0.0 { 1 } else { -1 };
        let inf = f64::INFINITY;
        let tdx = if dx != 0.0 { (g.res / dx).abs() } else { inf };
        let tdy = if dy != 0.0 { (g.res / dy).abs() } else { inf };
        let mut tx = if dx != 0.0 { ((if sx > 0 { (cx + 1) as f64 - fx } else { fx - cx as f64 }) * g.res) / dx.abs() } else { inf };
        let mut ty = if dy != 0.0 { ((if sy > 0 { (cy + 1) as f64 - fy } else { fy - cy as f64 }) * g.res) / dy.abs() } else { inf };
        let mut best = max_range as f32;
        let mut t = 0f64;
        while t <= max_range {
            if cx >= 0 && cx < g.w && cy >= 0 && cy < g.h && g.cells[cy as usize * g.w as usize + cx as usize] >= OCC_MIN {
                best = max(t, 0.0) as f32;
                break;
            }
            if tx < ty {
                t = tx;
                tx += tdx;
                cx += sx;
            } else {
                t = ty;
                ty += tdy;
                cy += sy;
            }
        }
        out[i] = best;
    }
}

#[derive(Clone, Copy)]
struct Cand {
    d: f64,
    a: [f32; 4],
}

// std::partial_sort as in libstdc++ (heap select + heap sort) with comp a.d < b.d, so ties keep the same order as the reference.
fn adjust_heap(v: &mut [Cand], mut hole: usize, len: usize, value: Cand) {
    let top = hole;
    let mut second = hole;
    while len >= 1 && second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if v[second].d < v[second - 1].d {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len & 1 == 0 && len >= 2 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    // push_heap
    while hole > top {
        let parent = (hole - 1) / 2;
        if !(v[parent].d < value.d) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = value;
}

fn partial_sort(v: &mut [Cand], m: usize) {
    if m >= 2 {
        let mut parent = (m - 2) / 2;
        loop {
            let value = v[parent];
            adjust_heap(v, parent, m, value);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    for i in m..v.len() {
        if v[i].d < v[0].d {
            let value = v[i];
            v[i] = v[0];
            adjust_heap(v, 0, m, value);
        }
    }
    let mut last = m;
    while last > 1 {
        last -= 1;
        let value = v[last];
        v[last] = v[0];
        adjust_heap(v, 0, last, value);
    }
}

/// The k nearest segments in the robot frame. out: k·4 (ax, ay, bx, by) [m], dist: k; unused = 0 / +inf.
fn segments_robot_frame(segs: &[f64], pose: &[f64; 3], out: &mut [f32], dist: &mut [f32]) {
    out.fill(0.0);
    dist.fill(f32::INFINITY);
    if segs.len() < 4 {
        return;
    }
    let k = dist.len();
    let c = (-pose[2]).cos();
    let s = (-pose[2]).sin();
    let mut cand: Vec<Cand> = segs
        .chunks_exact(4)
        .map(|w| {
            let (ax, ay, bx, by) = (w[0] - pose[0], w[1] - pose[1], w[2] - pose[0], w[3] - pose[1]);
            let (rax, ray, rbx, rby) = (c * ax - s * ay, s * ax + c * ay, c * bx - s * by, s * bx + c * by);
            let (abx, aby) = (rbx - rax, rby - ray);
            let t = clamp(-(rax * abx + ray * aby) / max(abx * abx + aby * aby, 1e-12), 0.0, 1.0);
            Cand { d: (rax + abx * t).hypot(ray + aby * t), a: [rax as f32, ray as f32, rbx as f32, rby as f32] }
        })
        .collect();
    let m = k.min(cand.len());
    partial_sort(&mut cand, m);
    for j in 0..m {
        out[4 * j..4 * j + 4].copy_from_slice(&cand[j].a);
        dist[j] = cand[j].d as f32;
    }
}

/// Wall state vector (layout at the top) for segments (flat ax ay bx by …) seen from pose (x, y, yaw).
pub fn wall_state(g: &Grid, segs: &[f64], pose: &[f64; 3]) -> [f32; STATE_LEN] {
    let mut out = [0f32; STATE_LEN];
    let mut rays = [0f32; SECTORS];
    ray_distances(g, pose, &mut rays, MAX_RANGE);
    for i in 0..SECTORS {
        out[i] = (rays[i] as f64 / MAX_RANGE) as f32;
    }
    let mut seg = [0f32; SEGMENTS * 4];
    let mut dist = [0f32; SEGMENTS];
    segments_robot_frame(segs, pose, &mut seg, &mut dist);
    for j in 0..SEGMENTS {
        let o = &mut out[SECTORS + 5 * j..SECTORS + 5 * j + 5];
        let valid = dist[j].is_finite() && dist[j] as f64 <= MAX_RANGE * 1.5;
        for q in 0..4 {
            o[q] = if valid { clampf((seg[4 * j + q] as f64 / MAX_RANGE) as f32, -1.0, 1.0) } else { 0.0 };
        }
        o[4] = if valid { 1.0 } else { 0.0 };
    }
    out
}
