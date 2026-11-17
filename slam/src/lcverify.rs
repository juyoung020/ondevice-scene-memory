//! 루프 · 재위치 후보의 기하 검증: 특징 짝 + 두 장의 깊이(3D) + IMU 중력 → 중력 고정 4 자유도 RANSAC(방위 + 이동)
//! → 4 자유도 재투영 가우스-뉴턴 → 6 자유도 재투영 다듬기(PnP 다듬기). 결과 = 대상(옛 키프레임) 카메라 → 질의 카메라 변환.
//!
//! - 두 카메라의 중력 방향 g_q · g_d(카메라 축, IMU)를 알면 기울기(roll · pitch)가 정해지므로 남는 것은 방위 1 + 이동 3.
//!   3D-3D 두 짝이면 가설 하나(최소 표본 2 점, 6 자유도 PnP 는 3 – 4 점).
//! - 가설 평가 · 맞은 점: 대상 3D 점을 질의 사진에 투영한 오차 < px(기본 2 화소, RTAB-Map PnP 와 같은 기준).
//! - 결정적: 표본 고르기는 고정 씨앗의 LCG(같은 입력 → 같은 결과). 식의 계산 순서는 파이썬 기준 구현과 같게 둔다
//!   (같은 기계에서 비트까지 같은지 시험, docs/loop_closure_npu.md).
//!
//! 카메라 축: 광학 축(x 오른쪽, y 아래, z 앞). 깊이 화소 = 컬러 화소(정렬된 깊이).

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

#[derive(Clone, Copy, Debug)]
pub struct Intr {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

#[derive(Clone, Debug)]
pub struct Params {
    /// RANSAC 가설 수
    pub iters: u32,
    /// 맞은 점 재투영 오차(화소)
    pub px: f64,
    /// 가설 평가 · 4 자유도 다듬기의 오차(화소): 중력 1° 오차는 화면에서 약 f·0.017 ≈ 9 화소라 가설은 넓게 보고,
    /// 6 자유도 다듬기에서 px_hyp → 2·px → px 로 좁힌다
    pub px_hyp: f64,
    /// 받아들일 맞은 점 수(RTAB-Map Vis/MinInliers)
    pub min_inliers: usize,
    /// 깊이 유효 범위(m)
    pub zmin: f64,
    pub zmax: f64,
    /// 두 점 사이 가로 거리가 이보다 짧으면 방위를 정하지 않음(m), 두 장의 길이 · 높이 차 허용(m)
    pub min_base: f64,
    pub len_tol: f64,
    /// 가우스-뉴턴 반복(4 자유도 두 번, 6 자유도 한 번)
    pub gn_iters: u32,
    /// 6 자유도 다듬기(중력 오차 흡수)
    pub refine6: bool,
    pub seed: u64,
}
impl Default for Params {
    fn default() -> Self {
        Params { iters: 200, px: 2.0, px_hyp: 8.0, min_inliers: 20, zmin: 0.1, zmax: 4.0, min_base: 0.05, len_tol: 0.05, gn_iters: 5, refine6: true, seed: 0x5eed }
    }
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub r: M3,
    pub t: V3,
    /// 마지막 맞은 점 수
    pub inliers: usize,
    /// RANSAC 가장 좋은 가설의 맞은 점 수
    pub hyp_inliers: usize,
    pub accepted: bool,
}

/// 깊이(정렬된 u16, per_m = 값/미터)에서 화소(반올림)의 3D 점. 범위 밖 · 0 이면 None
pub fn backproject(depth: &[u16], w: usize, h: usize, per_m: f64, k: &Intr, p: [f32; 2], zmin: f64, zmax: f64) -> Option<V3> {
    let u = (p[0] as f64).round_ties_even().clamp(0.0, (w - 1) as f64) as usize;
    let v = (p[1] as f64).round_ties_even().clamp(0.0, (h - 1) as f64) as usize;
    let z = depth[v * w + u] as f64 / per_m;
    if !(z > zmin && z < zmax) {
        return None;
    }
    Some([(p[0] as f64 - k.cx) * z / k.fx, (p[1] as f64 - k.cy) * z / k.fy, z])
}

#[inline]
fn mv(m: &M3, x: &V3) -> V3 {
    [m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2], m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2], m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2]]
}
#[inline]
fn mm(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}
#[inline]
fn tr(a: &M3) -> M3 {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}

/// 단위 중력 g 를 (0,0,1) 로 돌리는 회전(최소 회전, 로드리게스)
pub fn align_to_z(g: V3) -> M3 {
    let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let g = [g[0] / n, g[1] / n, g[2] / n];
    let (vx, vy) = (g[1], -g[0]);
    let c = g[2];
    if c < -1.0 + 1e-9 {
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    let k: M3 = [[0.0, 0.0, vy], [0.0, 0.0, -vx], [-vy, vx, 0.0]];
    let k2 = mm(&k, &k);
    let f = 1.0 / (1.0 + c);
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            a[i][j] = id + k[i][j] + k2[i][j] * f;
        }
    }
    a
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// 투영 오차 제곱(카메라 앞 0.05 m 밖이면 None)
#[inline]
fn reproj_err2(r: &M3, t: &V3, x: &V3, k: &Intr, obs: [f32; 2]) -> Option<f64> {
    let p = mv(r, x);
    let c = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
    if c[2] <= 0.05 {
        return None;
    }
    let u = k.fx * c[0] / c[2] + k.cx;
    let v = k.fy * c[1] / c[2] + k.cy;
    let (du, dv) = (u - obs[0] as f64, v - obs[1] as f64);
    Some(du * du + dv * dv)
}

fn count(r: &M3, t: &V3, xd: &[Option<V3>], kq: &[[f32; 2]], k: &Intr, px2: f64, inl: Option<&mut Vec<usize>>) -> usize {
    let mut n = 0;
    let mut list = inl;
    if let Some(l) = list.as_deref_mut() {
        l.clear();
    }
    for (i, x) in xd.iter().enumerate() {
        if let Some(x) = x
            && let Some(e2) = reproj_err2(r, t, x, k, kq[i])
            && e2 < px2
        {
            n += 1;
            if let Some(l) = list.as_deref_mut() {
                l.push(i);
            }
        }
    }
    n
}

fn rz(th: f64) -> M3 {
    let (s, c) = th.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// n×n 선형식 풀기(부분 피벗 가우스 소거, 계산 순서 고정)
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for c in 0..N {
        let mut p = c;
        for r in c + 1..N {
            if a[r][c].abs() > a[p][c].abs() {
                p = r;
            }
        }
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..N {
            let f = a[r][c] / a[c][c];
            for j in c..N {
                a[r][j] -= f * a[c][j];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; N];
    for c in (0..N).rev() {
        let mut s = b[c];
        for j in c + 1..N {
            s -= a[c][j] * x[j];
        }
        x[c] = s / a[c][c];
    }
    Some(x)
}

/// 4 자유도 가우스-뉴턴 한 번: 중력 정렬 공간의 (방위 th, 이동 tp)
fn gn4(th: &mut f64, tp: &mut V3, aq_t: &M3, pd: &[V3], obs: &[[f32; 2]], k: &Intr) {
    let mut hh = [[0.0; 4]; 4];
    let mut g = [0.0; 4];
    let (s, c) = th.sin_cos();
    let rzm = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    let drz = [[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]];
    for (p, o) in pd.iter().zip(obs) {
        let a = mv(&rzm, p);
        let q = mv(aq_t, &[a[0] + tp[0], a[1] + tp[1], a[2] + tp[2]]);
        if q[2] <= 0.05 {
            continue;
        }
        let iz = 1.0 / q[2];
        let u = k.fx * q[0] * iz + k.cx;
        let v = k.fy * q[1] * iz + k.cy;
        let r = [u - o[0] as f64, v - o[1] as f64];
        // d(u,v)/dq
        let du = [k.fx * iz, 0.0, -k.fx * q[0] * iz * iz];
        let dv = [0.0, k.fy * iz, -k.fy * q[1] * iz * iz];
        let dth = mv(aq_t, &mv(&drz, p));
        let cols: [V3; 4] = [dth, [aq_t[0][0], aq_t[1][0], aq_t[2][0]], [aq_t[0][1], aq_t[1][1], aq_t[2][1]], [aq_t[0][2], aq_t[1][2], aq_t[2][2]]];
        let mut ju = [0.0; 4];
        let mut jv = [0.0; 4];
        for j in 0..4 {
            ju[j] = du[0] * cols[j][0] + du[1] * cols[j][1] + du[2] * cols[j][2];
            jv[j] = dv[0] * cols[j][0] + dv[1] * cols[j][1] + dv[2] * cols[j][2];
        }
        for a in 0..4 {
            for b in 0..4 {
                hh[a][b] += ju[a] * ju[b] + jv[a] * jv[b];
            }
            g[a] += ju[a] * r[0] + jv[a] * r[1];
        }
    }
    if let Some(d) = solve(hh, [-g[0], -g[1], -g[2], -g[3]]) {
        *th += d[0];
        tp[0] += d[1];
        tp[1] += d[2];
        tp[2] += d[3];
    }
}

fn skew(w: &V3) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// 로드리게스 exp([w]×)
pub fn expso3(w: &V3) -> M3 {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let k = skew(w);
    let k2 = mm(&k, &k);
    let (a, b) = if th < 1e-12 { (1.0, 0.5) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            r[i][j] = id + a * k[i][j] + b * k2[i][j];
        }
    }
    r
}

/// 6 자유도 가우스-뉴턴 한 번(왼쪽 섭동: X' = exp(w)(R X + t) + dt)
fn gn6(r: &mut M3, t: &mut V3, xs: &[V3], obs: &[[f32; 2]], k: &Intr) {
    let mut hh = [[0.0; 6]; 6];
    let mut g = [0.0; 6];
    for (x, o) in xs.iter().zip(obs) {
        let p = mv(r, x);
        let q = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
        if q[2] <= 0.05 {
            continue;
        }
        let iz = 1.0 / q[2];
        let u = k.fx * q[0] * iz + k.cx;
        let v = k.fy * q[1] * iz + k.cy;
        let res = [u - o[0] as f64, v - o[1] as f64];
        let du = [k.fx * iz, 0.0, -k.fx * q[0] * iz * iz];
        let dv = [0.0, k.fy * iz, -k.fy * q[1] * iz * iz];
        // dq/dw = −[q]×, dq/dt = I
        let nq = skew(&q);
        let mut ju = [0.0; 6];
        let mut jv = [0.0; 6];
        for j in 0..3 {
            ju[j] = -(du[0] * nq[0][j] + du[1] * nq[1][j] + du[2] * nq[2][j]);
            jv[j] = -(dv[0] * nq[0][j] + dv[1] * nq[1][j] + dv[2] * nq[2][j]);
            ju[3 + j] = du[j];
            jv[3 + j] = dv[j];
        }
        for a in 0..6 {
            for b in 0..6 {
                hh[a][b] += ju[a] * ju[b] + jv[a] * jv[b];
            }
            g[a] += ju[a] * res[0] + jv[a] * res[1];
        }
    }
    if let Some(d) = solve(hh, [-g[0], -g[1], -g[2], -g[3], -g[4], -g[5]]) {
        let e = expso3(&[d[0], d[1], d[2]]);
        *r = mm(&e, r);
        let tt = mv(&e, t);
        *t = [tt[0] + d[3], tt[1] + d[4], tt[2] + d[5]];
    }
}

/// 짝(kq[i] ↔ kd[i])의 3D 점(xq · xd: 깊이로 만든 것, 없으면 None)과 두 카메라의 중력 방향으로 검증
pub fn verify(kq: &[[f32; 2]], xq: &[Option<V3>], xd: &[Option<V3>], gq: V3, gd: V3, k: &Intr, p: &Params) -> Option<Verdict> {
    let n = kq.len();
    assert!(xq.len() == n && xd.len() == n);
    let aq = align_to_z(gq);
    let ad = align_to_z(gd);
    let aq_t = tr(&aq);
    // 두 장 모두 깊이가 있는 짝: 중력 정렬 공간의 점
    let mut cand = Vec::new();
    let mut pq = Vec::new();
    let mut pd = Vec::new();
    for i in 0..n {
        if let (Some(a), Some(b)) = (&xq[i], &xd[i]) {
            cand.push(i);
            pq.push(mv(&aq, a));
            pd.push(mv(&ad, b));
        }
    }
    if cand.len() < 2 {
        return None;
    }
    let px2 = p.px * p.px;
    let ph2 = p.px_hyp * p.px_hyp;
    let mut rng = Lcg(p.seed);
    let nc = cand.len() as u64;
    let mut best: Option<(usize, f64, V3)> = None;
    for _ in 0..p.iters {
        let a = (rng.next() % nc) as usize;
        let b = (rng.next() % nc) as usize;
        if a == b {
            continue;
        }
        let dd = [pd[a][0] - pd[b][0], pd[a][1] - pd[b][1], pd[a][2] - pd[b][2]];
        let dq = [pq[a][0] - pq[b][0], pq[a][1] - pq[b][1], pq[a][2] - pq[b][2]];
        let hd = (dd[0] * dd[0] + dd[1] * dd[1]).sqrt();
        let hq = (dq[0] * dq[0] + dq[1] * dq[1]).sqrt();
        if hd < p.min_base || hq < p.min_base || (hd - hq).abs() > p.len_tol || (dd[2] - dq[2]).abs() > p.len_tol {
            continue;
        }
        let th = (dd[0] * dq[1] - dd[1] * dq[0]).atan2(dd[0] * dq[0] + dd[1] * dq[1]);
        let rzm = rz(th);
        let ra = mv(&rzm, &pd[a]);
        let rb = mv(&rzm, &pd[b]);
        let tp = [((pq[a][0] - ra[0]) + (pq[b][0] - rb[0])) * 0.5, ((pq[a][1] - ra[1]) + (pq[b][1] - rb[1])) * 0.5, ((pq[a][2] - ra[2]) + (pq[b][2] - rb[2])) * 0.5];
        let r = mm(&aq_t, &mm(&rzm, &ad));
        let t = mv(&aq_t, &tp);
        let c = count(&r, &t, xd, kq, k, ph2, None);
        if best.as_ref().is_none_or(|b| c > b.0) {
            best = Some((c, th, tp));
        }
    }
    let (hyp, mut th, mut tp) = best?;
    // 4 자유도 다듬기: 가설 문턱의 맞은 점으로 GN → 맞은 점 다시 → GN
    let mut inl = Vec::new();
    let pose = |th: f64, tp: &V3| (mm(&aq_t, &mm(&rz(th), &ad)), mv(&aq_t, tp));
    for _ in 0..2 {
        let (r, t) = pose(th, &tp);
        count(&r, &t, xd, kq, k, ph2, Some(&mut inl));
        if inl.len() < 3 {
            break;
        }
        let pdi: Vec<V3> = inl.iter().map(|&i| mv(&ad, xd[i].as_ref().unwrap())).collect();
        let obs: Vec<[f32; 2]> = inl.iter().map(|&i| kq[i]).collect();
        for _ in 0..p.gn_iters {
            gn4(&mut th, &mut tp, &aq_t, &pdi, &obs, k);
        }
    }
    let (mut r, mut t) = pose(th, &tp);
    if p.refine6 {
        // 6 자유도: 문턱 px_hyp → 2·px → px 로 좁히며 맞은 점으로 GN
        for th2 in [ph2, 4.0 * px2, px2] {
            count(&r, &t, xd, kq, k, th2, Some(&mut inl));
            if inl.len() < 6 {
                break;
            }
            let xs: Vec<V3> = inl.iter().map(|&i| xd[i].unwrap()).collect();
            let obs: Vec<[f32; 2]> = inl.iter().map(|&i| kq[i]).collect();
            for _ in 0..p.gn_iters {
                gn6(&mut r, &mut t, &xs, &obs, k);
            }
        }
    }
    let ninl = count(&r, &t, xd, kq, k, px2, Some(&mut inl));
    Some(Verdict { r, t, inliers: ninl, hyp_inliers: hyp, accepted: ninl >= p.min_inliers })
}

/// 두 변환의 차(이동 m, 회전 °): a⁻¹ b
pub fn pose_error(ra: &M3, ta: &V3, rb: &M3, tb: &V3) -> (f64, f64) {
    let rat = tr(ra);
    let e = mm(&rat, rb);
    let d = [tb[0] - ta[0], tb[1] - ta[1], tb[2] - ta[2]];
    let et = mv(&rat, &d);
    let c = ((e[0][0] + e[1][1] + e[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    ((et[0] * et[0] + et[1] * et[1] + et[2] * et[2]).sqrt(), c.acos().to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_maps_gravity_to_z() {
        for g in [[0.1, -0.98, 0.2], [0.0, 0.0, 1.0], [0.3, 0.3, -0.9], [0.0, 0.0, -1.0]] {
            let a = align_to_z(g);
            let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            let z = mv(&a, &[g[0] / n, g[1] / n, g[2] / n]);
            assert!((z[0]).abs() < 1e-12 && (z[1]).abs() < 1e-12 && (z[2] - 1.0).abs() < 1e-12, "{z:?}");
            let i = mm(&tr(&a), &a);
            for r in 0..3 {
                for c in 0..3 {
                    assert!((i[r][c] - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
                }
            }
        }
    }

    /// 합성 장면: 알려진 변환으로 만든 짝 + 외톨이 40 % → 변환을 되찾음
    #[test]
    fn recovers_synthetic_pose() {
        let k = Intr { fx: 517.3, fy: 516.5, cx: 318.6, cy: 255.3 };
        let r_true = expso3(&[0.05, -0.3, 0.08]);
        let t_true = [0.12, -0.04, 0.2];
        let g_world = [0.0, 1.0, 0.0];
        // 대상 카메라 = 세계, 질의 카메라 = r_true · 세계 + t_true → 질의 중력 = r_true g
        let gd = g_world;
        let gq = mv(&r_true, &g_world);
        let mut rng = Lcg(7);
        let mut f = || (rng.next() % 10_000) as f64 / 10_000.0;
        let (mut kq, mut xq, mut xd) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200 {
            let x = [(f() - 0.5) * 2.0, (f() - 0.5) * 1.5, 1.0 + 2.0 * f()];
            let q = {
                let p = mv(&r_true, &x);
                [p[0] + t_true[0], p[1] + t_true[1], p[2] + t_true[2]]
            };
            let mut obs = [(k.fx * q[0] / q[2] + k.cx) as f32, (k.fy * q[1] / q[2] + k.cy) as f32];
            let mut qq = q;
            if i % 5 < 2 {
                obs = [(f() * 640.0) as f32, (f() * 480.0) as f32];
                qq = [q[0] + f() - 0.5, q[1], q[2] + f()];
            }
            kq.push(obs);
            xq.push(Some(qq));
            xd.push(Some(x));
        }
        let v = verify(&kq, &xq, &xd, gq, gd, &k, &Params::default()).unwrap();
        let (dt, da) = pose_error(&r_true, &t_true, &v.r, &v.t);
        assert!(v.accepted && v.inliers >= 110, "inliers {}", v.inliers);
        assert!(dt < 1e-3 && da < 0.05, "dt {dt} da {da}");
    }
}
