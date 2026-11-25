//! ORB-SLAM3 `MLPnPsolver`(src/MLPnPsolver.cpp): 재위치에 쓰는 PnP(MLPnP, 6 점) + RANSAC.
//!
//! 원본과 비트 같게 옮겼다. 원본은 GCC 13 `-O3 -march=native`(AVX-512 · FMA)라 Eigen 커널마다 곱 다음 덧셈이
//! FMA 로 합쳐지는 자리가 다르다 — 원본 오브젝트를 `-fdump-tree-optimized` 로 덤프한 GIMPLE 을 읽어 자리마다
//! 같은 식(`mul_add` 위치 · 더하는 순서)으로 적었다. 직선 코드인 `mlpnpJacs` 는 GIMPLE 을 그대로 옮긴 생성 코드
//! (`mlpnp_jacs.rs`, 하네스 gen/gimple_to_rust.py), 4x4 역행렬(Eigen `compute_inverse_size4` SSE 판)도 생성 코드.
//!
//! 원본과 다른 점(결과 비트는 같음):
//! - `Refine()` 의 `computePose` 는 결과를 버리는 코드라(원본에서 mRi 를 갱신하지 않음) 부르지 않는다.
//!   `Refine` 의 `CheckInliers` 는 같은 mRi 로 다시 세는 것이라 직전 결과를 그대로 쓴다.
//! - 난수는 호출자가 넘긴다(`DUtils::Random::RandomInt(a, b)` 와 같은 함수). 원본은 glibc `rand()` 하나를
//!   Sim3Solver 등과 나눠 쓰므로 통합 쪽이 같은 생성기를 넘겨야 같은 수열이 된다.

use super::g2o_pose::{ldlt_back6, ldlt_factor6};
use super::mlpnp_jacs::mlpnp_jacs;
use super::pinhole::Pinhole;

unsafe extern "C" {
    fn sincos(x: f64, s: *mut f64, c: *mut f64);
}

/// GCC 가 sin · cos 쌍을 합친 `__builtin_cexpi`(glibc `sincos`): (sin, cos)
#[inline]
pub(super) fn csincos(x: f64) -> (f64, f64) {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    unsafe { sincos(x, &mut s, &mut c) };
    (s, c)
}

/// glibc `pow`
#[inline]
pub(super) fn cpow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

const MINP: f64 = f64::MIN_POSITIVE;

// ---------------------------------------------------------------------------------------------
// Eigen JacobiSVD(정사각, 사전 QR 없음) — `JacobiSVD<MatrixXd, ColPivHouseholderQRPreconditioner>::compute`
// ---------------------------------------------------------------------------------------------

/// 열 우선 정사각 행렬 `a[열][행]`
type Sq<const N: usize> = [[f64; N]; N];

fn identity<const N: usize>() -> Sq<N> {
    let mut m = [[0.0; N]; N];
    for (i, c) in m.iter_mut().enumerate() {
        c[i] = 1.0;
    }
    m
}

/// `internal::real_2x2_jacobi_svd`: (왼쪽 회전 (c, s), 오른쪽 회전 (c, s))
#[inline]
fn real_2x2_jacobi_svd<const N: usize>(w: &Sq<N>, p: usize, q: usize) -> ((f64, f64), (f64, f64)) {
    let m00 = w[p][p];
    let m01 = w[q][p];
    let m10 = w[p][q];
    let m11 = w[q][q];
    let d = m10 - m01;
    let (c1, s1, x, y, z);
    if d.abs() < MINP {
        (c1, s1, x, y, z) = (1.0, 0.0, m00, m01, m11);
    } else {
        let t = m11 + m00;
        let u = t / d;
        let tmp = fma(u, u, 1.0).sqrt();
        let s = 1.0 / tmp;
        let c = u / tmp;
        if c == 1.0 && s == 0.0 {
            (c1, s1, x, y, z) = (c, s, m00, m01, m11);
        } else {
            // m.applyOnTheLeft(0, 1, rot1): 2 칸 벡터 FMA
            (c1, s1) = (c, s);
            x = fma(m00, c, m10 * s);
            y = fma(m01, c, m11 * s);
            z = fma(m01, -s, m11 * c);
        }
    }
    // makeJacobi(x, y, z)
    let deno = y.abs() * 2.0;
    let (cr, sr) = if deno < MINP {
        (1.0, 0.0)
    } else {
        let tau = (x - z) / deno;
        let w = fma(tau, tau, 1.0).sqrt();
        let t = if tau > 0.0 { 1.0 / (w + tau) } else { 1.0 / (tau - w) };
        let nsign = if t > 0.0 { -1.0 } else { 1.0 };
        let n = 1.0 / fma(t, t, 1.0).sqrt();
        (n, n * (((y / y.abs()) * nsign) * t.abs()))
    };
    // j_left = rot1 * j_right.transpose()
    let ms = -sr;
    let cl = fma(-s1, ms, c1 * cr);
    let sl = fma(c1, ms, s1 * cr);
    ((cl, sl), (cr, sr))
}

/// 특이값 분해. `want_u` 면 U 도. 반환 (특이값 내림차순, U, V) — Eigen 처럼 정렬은 열 맞바꿈.
fn jacobi_svd<const N: usize>(mat: &Sq<N>, want_u: bool) -> ([f64; N], Sq<N>, Sq<N>) {
    let mut scale = 0.0f64;
    for c in mat {
        for &v in c {
            scale = scale.max(v.abs());
        }
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let mut w = *mat;
    for c in w.iter_mut() {
        for v in c.iter_mut() {
            *v /= scale;
        }
    }
    let mut u = identity::<N>();
    let mut v = identity::<N>();
    let precision = 2.0 * f64::EPSILON;
    let mut max_diag = 0.0f64;
    for i in 0..N {
        max_diag = max_diag.max(w[i][i].abs());
    }
    let mut finished = false;
    while !finished {
        finished = true;
        for p in 1..N {
            for q in 0..p {
                let thr = MINP.max(precision * max_diag);
                if w[q][p].abs() > thr || w[p][q].abs() > thr {
                    finished = false;
                    let ((cl, sl), (cr, sr)) = real_2x2_jacobi_svd(&w, p, q);
                    // W.applyOnTheLeft(p, q, j_left): 행 p · q
                    if !(cl == 1.0 && sl == 0.0) {
                        for c in w.iter_mut() {
                            let (x, y) = (c[p], c[q]);
                            c[p] = fma(cl, x, sl * y);
                            c[q] = fma(-sl, x, cl * y);
                        }
                        if want_u {
                            // U.applyOnTheRight(p, q, j_left.transpose()): applyOnTheRight 가 다시 전치해 j_left 그대로
                            for i in 0..N {
                                let (x, y) = (u[p][i], u[q][i]);
                                u[p][i] = fma(cl, x, sl * y);
                                u[q][i] = fma(-sl, x, cl * y);
                            }
                        }
                    }
                    // W.applyOnTheRight(p, q, j_right) · V.applyOnTheRight(p, q, j_right): 회전 (c, -s)
                    if !(cr == 1.0 && sr == 0.0) {
                        let ms = -sr;
                        for i in 0..N {
                            let (x, y) = (w[p][i], w[q][i]);
                            w[p][i] = fma(cr, x, ms * y);
                            w[q][i] = fma(sr, x, cr * y);
                        }
                        for i in 0..N {
                            let (x, y) = (v[p][i], v[q][i]);
                            v[p][i] = fma(cr, x, ms * y);
                            v[q][i] = fma(sr, x, cr * y);
                        }
                    }
                    max_diag = max_diag.max(w[p][p].abs().max(w[q][q].abs()));
                }
            }
        }
    }
    let mut sv = [0.0f64; N];
    for i in 0..N {
        let a = w[i][i];
        sv[i] = a.abs();
        if want_u && a < 0.0 {
            for x in u[i].iter_mut() {
                *x = -*x;
            }
        }
    }
    for s in sv.iter_mut() {
        *s *= scale;
    }
    for i in 0..N {
        let mut pos = i;
        let mut mx = sv[i];
        for (j, &s) in sv.iter().enumerate().skip(i + 1) {
            if s > mx {
                mx = s;
                pos = j;
            }
        }
        if mx == 0.0 {
            break;
        }
        if pos != i {
            sv.swap(i, pos);
            if want_u {
                u.swap(i, pos);
            }
            v.swap(i, pos);
        }
    }
    (sv, u, v)
}

/// `JacobiSVD<MatrixXd, HouseholderQRPreconditioner>(fᵀ, ComputeFullV).matrixV().block(0, 1, 3, 2)`:
/// 1x3 행렬은 Householder QR(3x1) 로 줄여 V = Q. 반환 [열 1, 열 2].
fn nullspace(f: &[f64; 3]) -> [[f64; 3]; 2] {
    let mut scale = f[0].abs().max(f[1].abs()).max(f[2].abs());
    if scale == 0.0 {
        scale = 1.0;
    }
    let (c0, x1, x2) = (f[0] / scale, f[1] / scale, f[2] / scale);
    let tail = fma(x2, x2, x1 * x1);
    if tail <= MINP {
        return [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut beta = fma(c0, c0, tail).sqrt();
    if c0 >= 0.0 {
        beta = -beta;
    }
    let dn = c0 - beta;
    let (e0, e1) = (x1 / dn, x2 / dn);
    let tau = (beta - c0) / beta;
    // householderQ().evalTo: I 에 applyHouseholderOnTheLeft
    let l0 = tau * e0;
    let l1 = tau * e1;
    [
        [-(tau * e0), fma(-l0, e0, 1.0), -(l1 * e0)],
        [-(tau * e1), -(l0 * e1), fma(-l1, e1, 1.0)],
    ]
}

// ---------------------------------------------------------------------------------------------
// 작은 곱 · 행렬식 · 역행렬
// ---------------------------------------------------------------------------------------------

/// Eigen 지연 곱(`MatrixXd * MatrixXd.transpose()` → `Matrix3d`, 깊이 n): 열마다 목적지 16 바이트 정렬에 따라
/// 두 행은 2 칸 Packet(0 에서 시작하는 FMA 사슬), 남은 한 행은 스칼라(첫 곱 뒤 곱 · 덧셈 따로).
/// `dst_aligned` = 목적지 시작이 16 바이트 정렬(원본 스택 배치로 정해짐).
fn lazy_prod3(l: &[[f64; 3]], r: &[[f64; 3]], dst_aligned: bool) -> [f64; 9] {
    // l: 3xn 의 열들, r: 3xn 의 열들(곱은 l · rᵀ)
    let n = l.len();
    let mut out = [0.0f64; 9];
    for j in 0..3 {
        // 열 j 시작 주소 = dst + 24 j → 정렬 여부
        let col_aligned = dst_aligned == (j % 2 == 0);
        for i in 0..3 {
            let packet = if col_aligned { i < 2 } else { i > 0 };
            let v = if n == 0 {
                0.0
            } else if packet {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc = fma(l[k][i], r[k][j], acc);
                }
                acc
            } else {
                let mut acc = l[0][i] * r[0][j];
                for k in 1..n {
                    acc = l[k][i] * r[k][j] + acc;
                }
                acc
            };
            out[j * 3 + i] = v;
        }
    }
    out
}

/// `determinant_impl<Matrix3d, 3>::run`(열 우선 e[0..9])
#[inline]
fn det3(e: &[f64; 9]) -> f64 {
    let a = fma(e[4], e[8], -(e[7] * e[5]));
    let b = fma(e[1], e[8], -(e[2] * e[7]));
    let c = fma(e[0], a, -(e[3] * b));
    let d = fma(e[1], e[5], -(e[2] * e[4]));
    fma(e[6], d, c)
}

/// `compute_inverse_size4<SSE, double>::run`(열 우선 4x4). GIMPLE 생성 코드.
#[allow(clippy::all)]
fn inverse4(m: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    include!("mlpnp_inv4.inc");
    out
}

// ---------------------------------------------------------------------------------------------
// 로드리게스
// ---------------------------------------------------------------------------------------------

/// `MLPnPsolver::rodrigues2rot`(열 우선 3x3)
pub fn rodrigues2rot(w: &[f64; 3]) -> [f64; 9] {
    let n = fma(w[2], w[2], w[0] * w[0] + w[1] * w[1]).sqrt();
    if !(n > f64::EPSILON) {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let (sn, cs) = csincos(n);
    let b = (1.0 - cs) / (n * n);
    let a = sn / n;
    let s = [0.0, w[2], -w[1], -w[2], 0.0, w[0], w[1], -w[0], 0.0];
    let at = |i: usize, j: usize| s[j * 3 + i];
    let bs = |i: usize, k: usize| at(i, k) * b;
    let mut r = [0.0f64; 9];
    for j in 0..3 {
        for i in 0..3 {
            let p = if i < 2 {
                fma(bs(i, 2), at(2, j), fma(bs(i, 1), at(1, j), bs(i, 0) * at(0, j)))
            } else {
                fma(bs(2, 0), at(0, j), fma(bs(2, 2), at(2, j), bs(2, 1) * at(1, j)))
            };
            let id = if i == j { 1.0 } else { 0.0 };
            r[j * 3 + i] = p + fma(at(i, j), a, id);
        }
    }
    r
}

/// `MLPnPsolver::rot2rodrigues`(열 우선 3x3)
pub fn rot2rodrigues(r: &[f64; 9]) -> [f64; 3] {
    let trace = (r[0] + (r[4] + r[8])) - 1.0;
    let wn = (trace * 0.5).acos();
    if wn > f64::EPSILON {
        let sc = wn / (wn.sin() * 2.0);
        [(r[5] - r[7]) * sc, (r[6] - r[2]) * sc, (r[1] - r[3]) * sc]
    } else {
        [0.0; 3]
    }
}

// ---------------------------------------------------------------------------------------------
// computePose(비평면 6 점) · 가우스-뉴턴
// ---------------------------------------------------------------------------------------------

/// Eigen GEMM(gebp) 한 칸: 4 열 묶음 안이면 k < 8 까지 짝 · 홀 k 를 두 누산기로 나눠 더한 뒤 합치고 남은 k 를
/// 이어 FMA, 묶음 밖 열은 한 누산기 FMA 사슬.
#[inline]
fn gebp_dot(a: impl Fn(usize) -> f64, b: impl Fn(usize) -> f64, depth: usize, split: bool) -> f64 {
    let peeled = depth & !7;
    let mut c = 0.0f64;
    let mut k = 0;
    if split {
        let mut d = 0.0f64;
        while k < peeled {
            if k % 2 == 0 {
                c = fma(a(k), b(k), c);
            } else {
                d = fma(a(k), b(k), d);
            }
            k += 1;
        }
        c += d;
    }
    while k < depth {
        c = fma(a(k), b(k), c);
        k += 1;
    }
    // R = C·alpha + 0
    fma(c, 1.0, 0.0)
}

/// 비평면 MLPnP(6 대응). 반환 (R 열 우선, t) — opengv 관례(world → camera).
pub fn compute_pose6(f: &[[f64; 3]; 6], p: &[[f64; 3]; 6]) -> ([f64; 9], [f64; 3]) {
    compute_pose6_tr(f, p, &mut |_| {})
}

/// `compute_pose6` + 중간값 기록(비교 하네스용)
#[inline(always)]
fn compute_pose6_tr(f: &[[f64; 3]; 6], p: &[[f64; 3]; 6], tr: &mut impl FnMut(&[f64])) -> ([f64; 9], [f64; 3]) {
    let ns: [[[f64; 3]; 2]; 6] = std::array::from_fn(|i| nullspace(&f[i]));
    // 1. 평면 검사: Σ p pᵀ 의 랭크가 2 면(세계 원점을 지나는 평면 위 6 점) 평면 경로
    let planar = lazy_prod3(p, p, PLANAR_DST_ALIGNED);
    tr(&planar);
    if fullpiv_rank3(&planar) == 2 {
        return compute_pose6_planar(f, p, &ns, &planar, tr);
    }
    // 3. 설계 행렬 A(12x12): a[행][열]
    let mut a = [[0.0f64; 12]; 12];
    for i in 0..6 {
        for (r, n) in ns[i].iter().enumerate() {
            let row = &mut a[2 * i + r];
            for (c3, nc) in n.iter().enumerate() {
                for (pc, pv) in p[i].iter().enumerate() {
                    row[c3 * 3 + pc] = nc * pv;
                }
                row[9 + c3] = *nc;
            }
        }
    }
    // 4. AᵀA (GEMM, 깊이 12, 열은 4 개 묶음 셋)
    let mut ata: Sq<12> = [[0.0; 12]; 12];
    for j in 0..12 {
        for i in 0..12 {
            ata[j][i] = gebp_dot(|k| a[k][i], |k| a[k][j], 12, true);
        }
    }
    tr(ata.as_flattened());
    let (_, _, v) = jacobi_svd::<12>(&ata, false);
    let r1 = v[11];
    tr(&r1);
    // 비평면: tmp 는 r1[0..9] 를 열 우선으로
    let col_norm = |c: usize| fma(r1[3 * c + 2], r1[3 * c + 2], r1[3 * c] * r1[3 * c] + r1[3 * c + 1] * r1[3 * c + 1]).sqrt();
    let scale = 1.0 / cpow((col_norm(0) * col_norm(1) * col_norm(2)).abs(), 1.0 / 3.0);
    let tmp: Sq<3> = [[r1[0], r1[1], r1[2]], [r1[3], r1[4], r1[5]], [r1[6], r1[7], r1[8]]];
    tr(&[scale]);
    let (_, u3, v3) = jacobi_svd::<3>(&tmp, true);
    tr(u3.as_flattened());
    tr(v3.as_flattened());
    // Rout = U Vᵀ
    // Rout = U Vᵀ: 이 자리는 원본에서 칸마다 곱을 반올림해 차례로 더함(합침 없음, 추적 하네스로 확인)
    let mut rout = [0.0f64; 9];
    for j in 0..3 {
        for i in 0..3 {
            rout[j * 3 + i] = (u3[0][i] * v3[0][j] + u3[1][i] * v3[1][j]) + u3[2][i] * v3[2][j];
        }
    }
    tr(&rout);
    if det3(&rout) < 0.0 {
        for x in rout.iter_mut() {
            *x = -*x;
        }
    }
    // tout = Rout (scale · t)
    let st = [scale * r1[9], scale * r1[10], scale * r1[11]];
    let tout = [
        fma(rout[6], st[2], fma(rout[3], st[1], rout[0] * st[0])),
        fma(rout[7], st[2], fma(rout[4], st[1], rout[1] * st[0])),
        fma(st[0], rout[2], fma(st[2], rout[8], st[1] * rout[5])),
    ];
    tr(&tout);
    // 방향: 4x4 역변환 두 개 중 재투영 오차 작은 쪽
    let mut ti = [[0.0f64; 16]; 2];
    let mut err = [0.0f64; 2];
    for s in 0..2 {
        let mut t4 = [0.0f64; 16];
        for c in 0..3 {
            t4[c * 4..c * 4 + 3].copy_from_slice(&rout[c * 3..c * 3 + 3]);
        }
        let sg = if s == 0 { 1.0 } else { -1.0 };
        t4[12] = sg * tout[0];
        t4[13] = sg * tout[1];
        t4[14] = sg * tout[2];
        t4[15] = 1.0;
        let m = inverse4(&t4);
        let mut e = 0.0f64;
        for k in 0..6 {
            let (x, y, z) = (p[k][0], p[k][1], p[k][2]);
            let v0 = fma(m[8], z, fma(m[4], y, x * m[0])) + m[12];
            let v1 = fma(m[9], z, fma(m[5], y, x * m[1])) + m[13];
            let v2 = fma(x, m[2], fma(z, m[10], y * m[6])) + m[14];
            let nrm = fma(v2, v2, v0 * v0 + v1 * v1).sqrt();
            let (v0, v1, v2) = (v0 / nrm, v1 / nrm, v2 / nrm);
            let dot = fma(f[k][2], v2, f[k][0] * v0 + f[k][1] * v1);
            e += 1.0 - dot;
        }
        err[s] = e;
        tr(&[e]);
        ti[s] = m;
    }
    let pick = if err[0] < err[1] { 0 } else { 1 };
    let tout = [ti[pick][12], ti[pick][13], ti[pick][14]];
    let rout = [ti[0][0], ti[0][1], ti[0][2], ti[0][4], ti[0][5], ti[0][6], ti[0][8], ti[0][9], ti[0][10]];
    // 5. 가우스-뉴턴
    tr(&rout);
    tr(&tout);
    let om = rot2rodrigues(&rout);
    tr(&om);
    let mut x = [om[0], om[1], om[2], tout[0], tout[1], tout[2]];
    gauss_newton(&mut x, p, &ns);
    tr(&x);
    (rodrigues2rot(&[x[0], x[1], x[2]]), [x[3], x[4], x[5]])
}

/// GEMM(gebp) 9 행 중 마지막 행(8): "한 행씩" 경로 — 4 열 묶음은 rhs 를 lhs 자리로 바꾼 Packet8d 계산
/// (k 짝 · 홀을 칸으로 나누고 k 0..8 은 네 누산기를 ((0+1)+(2+3)) 으로 합친 뒤 남은 k 를 FMA, 끝에 짝 + 홀).
/// 묶음 밖 열은 스칼라 FMA 사슬.
#[inline]
fn gebp_dot_lastrow(a: impl Fn(usize) -> f64, b: impl Fn(usize) -> f64, depth: usize, group: bool) -> f64 {
    if !group {
        let mut c = a(0) * b(0);
        for k in 1..depth {
            c = fma(a(k), b(k), c);
        }
        return fma(c, 1.0, 0.0);
    }
    let p = |k: usize| a(k) * b(k);
    let endk4 = depth / 8 * 8;
    let endk = depth / 2 * 2;
    let (mut ev, mut od) = (0.0f64, 0.0f64);
    let mut k = 0;
    if endk4 > 0 {
        // 첫 묶음(k 0..8): 누산기 4 개가 한 번씩
        ev = (p(0) + p(2)) + (p(4) + p(6));
        od = (p(1) + p(3)) + (p(5) + p(7));
        k = 8;
        // 깊이 16 이상은 원본에서 누산기에 FMA 를 더 쌓지만 MLPnP(깊이 12)에는 없음
        debug_assert!(endk4 == 8);
    }
    while k < endk {
        ev = fma(a(k), b(k), ev);
        od = fma(a(k + 1), b(k + 1), od);
        k += 2;
    }
    debug_assert!(endk == depth);
    fma(ev + od, 1.0, 0.0)
}

/// `SelfAdjointEigenSolver<Matrix3d>::compute`(고유벡터, 고윳값 오름차순) — 3x3 닫힌 꼴 삼중 대각화 + QR 단계
fn saes3(a: &[f64; 9]) -> [f64; 9] {
    let lower = [a[0], a[1], a[2], a[4], a[5], a[8]];
    let mut scale = 0.0f64;
    for v in lower {
        scale = scale.max(v.abs());
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let (m00, m10, m20, m11, m21, m22) = (a[0] / scale, a[1] / scale, a[2] / scale, a[4] / scale, a[5] / scale, a[8] / scale);
    let mut diag = [m00, 0.0, 0.0];
    let mut sub;
    let mut q: [f64; 9];
    let v1 = m20 * m20;
    if v1 <= MINP {
        diag[1] = m11;
        diag[2] = m22;
        sub = [m10, m21];
        q = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    } else {
        let beta = fma(m10, m10, v1).sqrt();
        let inv = 1.0 / beta;
        let q01 = inv * m10;
        let q02 = m20 * inv;
        let qq = fma(q01 * 2.0, m21, q02 * (m22 - m11));
        diag[1] = fma(q02, qq, m11);
        diag[2] = fma(-q02, qq, m22);
        sub = [beta, fma(-q01, qq, m21)];
        q = [1.0, 0.0, 0.0, 0.0, q01, q02, 0.0, q02, -q01];
    }
    // computeFromTridiagonal_impl
    let n = 3usize;
    let mut end = n - 1;
    let mut start = 0usize;
    let mut iter = 0usize;
    let precision_inv = 1.0 / f64::EPSILON;
    while end > 0 {
        for i in start..end {
            if sub[i].abs() < MINP {
                sub[i] = 0.0;
            } else {
                let sc = precision_inv * sub[i];
                if sc * sc <= diag[i].abs() + diag[i + 1].abs() {
                    sub[i] = 0.0;
                }
            }
        }
        while end > 0 && sub[end - 1] == 0.0 {
            end -= 1;
        }
        if end == 0 {
            break;
        }
        iter += 1;
        if iter > 30 * n {
            break;
        }
        start = end - 1;
        while start > 0 && sub[start - 1] != 0.0 {
            start -= 1;
        }
        tridiagonal_qr_step(&mut diag, &mut sub, start, end, &mut q);
    }
    if iter <= 30 * n {
        for i in 0..n - 1 {
            let mut k = i;
            for j in i + 1..n {
                if diag[j] < diag[k] {
                    k = j;
                }
            }
            if k != i {
                diag.swap(i, k);
                for r in 0..3 {
                    q.swap(i * 3 + r, k * 3 + r);
                }
            }
        }
    }
    q
}

/// Eigen `tridiagonal_qr_step`(고유벡터 3x3 열 우선 q 에 회전 누적)
fn tridiagonal_qr_step(diag: &mut [f64; 3], sub: &mut [f64; 2], start: usize, end: usize, q: &mut [f64; 9]) {
    let td = (diag[end - 1] - diag[end]) * 0.5;
    let e = sub[end - 1];
    let mut mu = diag[end];
    if td == 0.0 {
        mu -= e.abs();
    } else if e != 0.0 {
        let e2 = e * e;
        let (ae, at) = (e.abs(), td.abs());
        let h = if ae > at { fma(at / ae, at / ae, 1.0).sqrt() * ae } else { fma(ae / at, ae / at, 1.0).sqrt() * at };
        let den = td + if td > 0.0 { h } else { -h };
        if e2 == 0.0 {
            mu -= e / (den / e);
        } else {
            mu -= e2 / den;
        }
    }
    let mut x = diag[start] - mu;
    let mut z = sub[start];
    let mut k = start;
    while k < end && z != 0.0 {
        // makeGivens(x, z)
        let (c, s);
        if x == 0.0 {
            c = 0.0;
            s = if z < 0.0 { 1.0 } else { -1.0 };
        } else if x.abs() > z.abs() {
            let t = z / x;
            let mut u = fma(t, t, 1.0).sqrt();
            if x < 0.0 {
                u = -u;
            }
            c = 1.0 / u;
            s = c * -t;
        } else {
            let t = x / z;
            let mut u = fma(t, t, 1.0).sqrt();
            if z < 0.0 {
                u = -u;
            }
            s = -1.0 / u;
            c = s * -t;
        }
        let (dk, sk, dk1) = (diag[k], sub[k], diag[k + 1]);
        let ssk = s * sk;
        let inner_a = fma(c, dk, -ssk);
        let sdk = fma(c, sk, s * dk);
        let inner_b = fma(c, sk, -(s * dk1));
        let dkp1 = fma(c, dk1, ssk);
        diag[k] = fma(inner_a, c, -(s * inner_b));
        diag[k + 1] = fma(sdk, s, c * dkp1);
        sub[k] = fma(sdk, c, -(s * dkp1));
        if k > start {
            sub[k - 1] = fma(sub[k - 1], c, -(s * z));
        }
        x = sub[k];
        if k + 1 < end {
            z = -s * sub[k + 1];
            sub[k + 1] *= c;
        }
        // q.applyOnTheRight(k, k+1, rot): 회전 (c, -s)
        if !(c == 1.0 && s == 0.0) {
            for r in 0..3 {
                let (xi, yi) = (q[k * 3 + r], q[(k + 1) * 3 + r]);
                q[k * 3 + r] = fma(c, xi, -s * yi);
                q[(k + 1) * 3 + r] = fma(s, xi, c * yi);
            }
        }
        k += 1;
    }
}

/// 평면 경로: 고유 좌표계로 돌린 점의 y · z 만으로 9 열 설계 행렬, 방향 후보 4 개
#[inline(always)]
fn compute_pose6_planar(
    f: &[[f64; 3]; 6],
    p: &[[f64; 3]; 6],
    ns: &[[[f64; 3]; 2]; 6],
    planar: &[f64; 9],
    tr: &mut impl FnMut(&[f64]),
) -> ([f64; 9], [f64; 3]) {
    let ev = saes3(planar);
    tr(&ev);
    // eigenRot = 고유벡터ᵀ (er(i, j) = ev(j, i))
    let er = |i: usize, j: usize| ev[i * 3 + j];
    let q: [[f64; 3]; 6] = std::array::from_fn(|k| {
        let (x, y, z) = (p[k][0], p[k][1], p[k][2]);
        [
            fma(er(0, 2), z, fma(er(0, 1), y, er(0, 0) * x)),
            fma(er(1, 2), z, fma(er(1, 1), y, er(1, 0) * x)),
            fma(x, er(2, 0), fma(z, er(2, 2), y * er(2, 1))),
        ]
    });
    tr(q.as_flattened());
    let mut a = [[0.0f64; 9]; 12];
    for i in 0..6 {
        for (r, n) in ns[i].iter().enumerate() {
            let row = &mut a[2 * i + r];
            for c3 in 0..3 {
                row[2 * c3] = n[c3] * q[i][1];
                row[2 * c3 + 1] = n[c3] * q[i][2];
                row[6 + c3] = n[c3];
            }
        }
    }
    let mut ata: Sq<9> = [[0.0; 9]; 9];
    for j in 0..9 {
        for i in 0..9 {
            ata[j][i] = if i < 8 {
                gebp_dot(|k| a[k][i], |k| a[k][j], 12, j < 8)
            } else {
                gebp_dot_lastrow(|k| a[k][i], |k| a[k][j], 12, j < 8)
            };
        }
    }
    tr(ata.as_flattened());
    let (_, _, v) = jacobi_svd::<9>(&ata, false);
    let r = v[8];
    tr(&r);
    // tmp 의 첫 열 = 둘째 열 × 셋째 열, 뒤집은(transposeInPlace) 뒤 열 j = 원래 행 j
    let cx = fma(r[5], r[2], -(r[4] * r[3]));
    let cy = fma(r[4], r[1], -(r[5] * r[0]));
    let cz = fma(r[3], r[0], -(r[2] * r[1]));
    let tmp: Sq<3> = [[cx, r[0], r[1]], [cy, r[2], r[3]], [cz, r[4], r[5]]];
    let nrm = |c: &[f64; 3]| fma(c[2], c[2], c[0] * c[0] + c[1] * c[1]).sqrt();
    let scale = 1.0 / (nrm(&tmp[2]) * nrm(&tmp[1])).abs().sqrt();
    tr(&[scale]);
    let (_, u3, v3) = jacobi_svd::<3>(&tmp, true);
    // Rout1 = U Vᵀ: 평면 경로 쪽은 지연 곱(목적지 정렬, 두 행 Packet FMA 사슬 · 한 행 스칼라) — 비평면 쪽과 다름(추적 하네스로 확인)
    let mut r1 = lazy_prod3(&u3, &v3, true);
    if det3(&r1) < 0.0 {
        for x in r1.iter_mut() {
            *x = -*x;
        }
    }
    // Rout1 = eigenRotᵀ Rout1
    let mut r2 = [0.0f64; 9];
    for j in 0..3 {
        for i in 0..3 {
            let l = |k: usize| er(k, i);
            let rr = |k: usize| r1[j * 3 + k];
            r2[j * 3 + i] = fma(l(2), rr(2), l(0) * rr(0) + l(1) * rr(1));
        }
    }
    tr(&r2);
    let t = [scale * r[6], scale * r[7], scale * r[8]];
    // transposeInPlace, *= -1, 행렬식 음수면 셋째 열 부호
    let mut ro = [0.0f64; 9];
    for j in 0..3 {
        for i in 0..3 {
            ro[j * 3 + i] = -r2[i * 3 + j];
        }
    }
    if det3(&ro) < 0.0 {
        for x in ro[6..9].iter_mut() {
            *x = -*x;
        }
    }
    let mut rb = ro;
    for x in rb[0..6].iter_mut() {
        *x = -*x;
    }
    let cands = [(ro, t), (ro, t.map(|x| -x)), (rb, t), (rb, t.map(|x| -x))];
    let mut best = 0usize;
    let mut bestv = 0.0f64;
    for (ci, (rm, tv)) in cands.iter().enumerate() {
        let mut norms = 0.0f64;
        for k in 0..6 {
            let (x, y, z) = (p[k][0], p[k][1], p[k][2]);
            let v0 = fma(rm[6], z, fma(rm[3], y, rm[0] * x)) + tv[0];
            let v1 = fma(rm[7], z, fma(rm[4], y, rm[1] * x)) + tv[1];
            let v2 = fma(x, rm[2], fma(z, rm[8], y * rm[5])) + tv[2];
            let nn = fma(v2, v2, v0 * v0 + v1 * v1).sqrt();
            let (v0, v1, v2) = (v0 / nn, v1 / nn, v2 / nn);
            norms += 1.0 - fma(f[k][2], v2, f[k][0] * v0 + f[k][1] * v1);
        }
        tr(&[norms]);
        if ci == 0 || norms < bestv {
            best = ci;
            bestv = norms;
        }
    }
    let (rout, tout) = cands[best];
    let om = rot2rodrigues(&rout);
    let mut x = [om[0], om[1], om[2], tout[0], tout[1], tout[2]];
    gauss_newton(&mut x, p, ns);
    tr(&x);
    (rodrigues2rot(&[x[0], x[1], x[2]]), [x[3], x[4], x[5]])
}

/// `points3 * points3ᵀ` 목적지(planarTest, 원본 computePose 스택)가 16 바이트 정렬인가(추적 하네스로 확인)
const PLANAR_DST_ALIGNED: bool = true;

/// `MLPnPsolver::mlpnp_residuals_and_jacs`: (잔차 12, 야코비안 12x6 a[행][열])
fn residuals_and_jacs(x: &[f64; 6], pts: &[[f64; 3]; 6], ns: &[[[f64; 3]; 2]; 6]) -> ([f64; 12], [[f64; 6]; 12]) {
    let w = [x[0], x[1], x[2]];
    let t = [x[3], x[4], x[5]];
    let r = rodrigues2rot(&w);
    let mut res = [0.0f64; 12];
    let mut jac = [[0.0f64; 6]; 12];
    for i in 0..6 {
        let (px, py, pz) = (pts[i][0], pts[i][1], pts[i][2]);
        let c0 = t[0] + fma(r[6], pz, fma(r[3], py, r[0] * px));
        let c1 = t[1] + fma(r[7], pz, fma(r[4], py, r[1] * px));
        let c2 = t[2] + fma(r[2], px, fma(r[8], pz, r[5] * py));
        let nrm = fma(c2, c2, c0 * c0 + c1 * c1).sqrt();
        let (c0, c1, c2) = (c0 / nrm, c1 / nrm, c2 / nrm);
        let [n0, n1] = &ns[i];
        res[2 * i] = fma(n0[2], c2, fma(n0[0], c0, n0[1] * c1));
        res[2 * i + 1] = fma(c2, n1[2], fma(c0, n1[0], c1 * n1[1]));
        let jc = mlpnp_jacs(&pts[i], n0, n1, &w, &t);
        for k in 0..6 {
            jac[2 * i][k] = jc[2 * k];
            jac[2 * i + 1][k] = jc[2 * k + 1];
        }
    }
    (res, jac)
}

/// `MLPnPsolver::mlpnp_gn`(공분산 없음, 최대 5 회)
fn gauss_newton(x: &mut [f64; 6], pts: &[[f64; 3]; 6], ns: &[[[f64; 3]; 2]; 6]) {
    for _ in 0..5 {
        let (r, jac) = residuals_and_jacs(x, pts, ns);
        // A = Jᵀ J (GEMM 6x6, 깊이 12: 열 0..4 는 4 열 묶음, 4 · 5 는 남은 열)
        let mut h = [0.0f64; 36];
        for j in 0..6 {
            for i in 0..6 {
                h[j * 6 + i] = gebp_dot(|k| jac[k][i], |k| jac[k][j], 12, j < 4);
            }
        }
        // g = Jᵀ r (GEMV: 행마다 0 에서 시작하는 FMA 사슬)
        let g: [f64; 6] = std::array::from_fn(|i| {
            let mut c = 0.0f64;
            for k in 0..12 {
                c = fma(jac[k][i], r[k], c);
            }
            fma(c, 1.0, 0.0)
        });
        let (m, tr, _) = ldlt_factor6(&h);
        let mut dx = [0.0f64; 6];
        ldlt_back6(&m, &tr, &g, &mut dx);
        let mut mx = 0.0f64;
        let mut mn = f64::INFINITY;
        for v in dx {
            mx = mx.max(v.abs());
            mn = mn.min(v.abs());
        }
        if mx > 5.0 || mn > 1.0 {
            break;
        }
        // dl = J dx
        let mut dlmax = 0.0f64;
        for row in &jac {
            let mut c = 0.0f64;
            for k in 0..6 {
                c = fma(row[k], dx[k], c);
            }
            dlmax = dlmax.max(fma(c, 1.0, 0.0).abs());
        }
        for k in 0..6 {
            x[k] -= dx[k];
        }
        if dlmax < 1e-5 {
            break;
        }
    }
}

/// `FullPivHouseholderQR<Matrix3d>(m).rank()`
fn fullpiv_rank3(m: &[f64; 9]) -> usize {
    let mut q = *m; // 열 우선
    let at = |r: usize, c: usize| r + 3 * c;
    let precision = f64::EPSILON * 3.0;
    let mut biggest = 0.0f64;
    let mut nonzero = 3usize;
    let mut maxpivot = 0.0f64;
    let mut temp = [0.0f64; 3];
    for k in 0..3 {
        let (mut br, mut bc, mut bs) = (k, k, -1.0f64);
        for c in k..3 {
            for r in k..3 {
                let s = q[at(r, c)].abs();
                if s > bs {
                    bs = s;
                    br = r;
                    bc = c;
                }
            }
        }
        if k == 0 {
            biggest = bs;
        }
        if bs <= biggest * precision {
            nonzero = k;
            break;
        }
        if br != k {
            for c in k..3 {
                q.swap(at(k, c), at(br, c));
            }
        }
        if bc != k {
            for r in 0..3 {
                q.swap(at(r, k), at(r, bc));
            }
        }
        // makeHouseholderInPlace(col k, 행 k..3)
        let c0 = q[at(k, k)];
        let mut tail = 0.0f64;
        for r in k + 1..3 {
            tail = if r == k + 1 { q[at(r, k)] * q[at(r, k)] } else { fma(q[at(r, k)], q[at(r, k)], tail) };
        }
        let (tau, beta);
        if tail <= MINP {
            tau = 0.0;
            beta = c0;
            for r in k + 1..3 {
                q[at(r, k)] = 0.0;
            }
        } else {
            let mut b = fma(c0, c0, tail).sqrt();
            if c0 >= 0.0 {
                b = -b;
            }
            for r in k + 1..3 {
                q[at(r, k)] /= c0 - b;
            }
            tau = (b - c0) / b;
            beta = b;
        }
        q[at(k, k)] = beta;
        if beta.abs() > maxpivot {
            maxpivot = beta.abs();
        }
        // applyHouseholderOnTheLeft(아래 오른쪽 (3-k)x(2-k))
        if tau != 0.0 {
            for c in k + 1..3 {
                let mut s = q[at(k, c)];
                for r in k + 1..3 {
                    s = fma(q[at(r, k)], q[at(r, c)], s);
                }
                temp[c] = s;
            }
            for c in k + 1..3 {
                q[at(k, c)] = fma(-tau, temp[c], q[at(k, c)]);
                for r in k + 1..3 {
                    q[at(r, c)] = fma(-(tau * q[at(r, k)]), temp[c], q[at(r, c)]);
                }
            }
        }
    }
    let thr = maxpivot.abs() * (f64::EPSILON * 3.0);
    (0..nonzero).filter(|&i| q[at(i, i)].abs() > thr).count()
}

// ---------------------------------------------------------------------------------------------
// RANSAC(MLPnPsolver 상태 · iterate · CheckInliers)
// ---------------------------------------------------------------------------------------------

/// `MLPnPsolver` — 생성자 입력은 맵 점이 붙은(나쁘지 않은) 대응만, 원본 순서대로.
#[derive(Clone, Debug)]
pub struct MlpnpSolver {
    pub cam: Pinhole,
    /// mvP2D(왜곡 보정 키포인트)
    pub p2d: Vec<[f32; 2]>,
    pub sigma2: Vec<f32>,
    pub bearings: Vec<[f64; 3]>,
    pub p3dw: Vec<[f64; 3]>,
    pub kp_idx: Vec<usize>,
    pub max_error: Vec<f32>,
    /// mvpMapPointMatches.size()
    pub n_matches: usize,
    pub ransac_prob: f64,
    pub ransac_min_inliers: i32,
    pub ransac_max_its: i32,
    pub ransac_epsilon: f32,
    pub ransac_min_set: i32,
    // 상태
    pub n_iterations: i32,
    pub best_inliers: Vec<bool>,
    pub n_best_inliers: i32,
    pub best_tcw: [f32; 16],
    pub ri: [[f64; 3]; 3],
    pub ti: [f64; 3],
    pub inliers_i: Vec<bool>,
    pub n_inliers_i: i32,
}

/// `iterate` 결과
#[derive(Clone, Debug, PartialEq)]
pub struct IterateOut {
    pub ok: bool,
    pub no_more: bool,
    pub n_inliers: i32,
    /// 크기 n_matches(없으면 빈 벡터 — 원본 vbInliers.clear())
    pub inliers: Vec<bool>,
    /// 열 우선 4x4
    pub tcw: [f32; 16],
}

const IDENT4F: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

impl MlpnpSolver {
    /// 생성자 + `SetRansacParameters`. 입력 키포인트(왜곡 보정) · 레벨 σ² · 맵 점 세계 좌표(f32) · 원래 인덱스.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cam: Pinhole,
        kps: &[[f32; 2]],
        sigma2: &[f32],
        world: &[[f32; 3]],
        kp_idx: &[usize],
        n_matches: usize,
        probability: f64,
        min_inliers: i32,
        max_iterations: i32,
        min_set: i32,
        epsilon: f32,
        th2: f32,
    ) -> Self {
        let bearings = kps
            .iter()
            .map(|k| {
                let b = cam.unproject(k[0], k[1]);
                // cv_br /= cv_br.z (float)
                [f64::from(b[0] / b[2]), f64::from(b[1] / b[2]), f64::from(b[2] / b[2])]
            })
            .collect();
        let mut s = Self {
            cam,
            p2d: kps.to_vec(),
            sigma2: sigma2.to_vec(),
            bearings,
            p3dw: world.iter().map(|w| w.map(f64::from)).collect(),
            kp_idx: kp_idx.to_vec(),
            max_error: Vec::new(),
            n_matches,
            ransac_prob: 0.0,
            ransac_min_inliers: 0,
            ransac_max_its: 0,
            ransac_epsilon: 0.0,
            ransac_min_set: 0,
            n_iterations: 0,
            best_inliers: Vec::new(),
            n_best_inliers: 0,
            best_tcw: [0.0; 16],
            ri: [[0.0; 3]; 3],
            ti: [0.0; 3],
            inliers_i: Vec::new(),
            n_inliers_i: 0,
        };
        s.set_ransac_parameters(probability, min_inliers, max_iterations, min_set, epsilon, th2);
        s
    }

    pub fn n(&self) -> usize {
        self.p2d.len()
    }

    /// `SetRansacParameters`
    pub fn set_ransac_parameters(&mut self, probability: f64, min_inliers: i32, max_iterations: i32, min_set: i32, epsilon: f32, th2: f32) {
        self.ransac_prob = probability;
        self.ransac_min_inliers = min_inliers;
        self.ransac_max_its = max_iterations;
        self.ransac_epsilon = epsilon;
        self.ransac_min_set = min_set;
        let n = self.p2d.len() as i32;
        self.inliers_i.resize(n as usize, false);
        let mut nmin = (n as f32 * self.ransac_epsilon) as i32;
        if nmin < self.ransac_min_inliers {
            nmin = self.ransac_min_inliers;
        }
        if nmin < min_set {
            nmin = min_set;
        }
        self.ransac_min_inliers = nmin;
        let r = self.ransac_min_inliers as f32 / n as f32;
        if self.ransac_epsilon < r {
            self.ransac_epsilon = r;
        }
        let n_it = if self.ransac_min_inliers == n {
            1
        } else {
            ((1.0 - self.ransac_prob).ln() / (1.0 - f64::from(self.ransac_epsilon).powf(3.0)).ln()).ceil() as i32
        };
        self.ransac_max_its = 1.max(n_it.min(self.ransac_max_its));
        self.max_error = self.sigma2.iter().map(|s| s * th2).collect();
    }

    /// `CheckInliers`(mRi · mti 로)
    fn check_inliers(&mut self) {
        let (r, t) = (&self.ri, &self.ti);
        let mut n = 0;
        for i in 0..self.p2d.len() {
            let p = self.p3dw[i];
            let (x, y, z) = (f64::from(p[0] as f32), f64::from(p[1] as f32), f64::from(p[2] as f32));
            let xc = (fma(z, r[0][2], fma(x, r[0][0], y * r[0][1])) + t[0]) as f32;
            let yc = (fma(z, r[1][2], fma(x, r[1][0], y * r[1][1])) + t[1]) as f32;
            let zc = (fma(z, r[2][2], fma(x, r[2][0], y * r[2][1])) + t[2]) as f32;
            let uv = self.cam.project_f(&[xc, yc, zc]);
            let dx = self.p2d[i][0] - uv[0];
            let dy = self.p2d[i][1] - uv[1];
            let e2 = dx.mul_add(dx, dy * dy);
            let inl = self.max_error[i] > e2;
            self.inliers_i[i] = inl;
            if inl {
                n += 1;
            }
        }
        self.n_inliers_i = n;
    }

    fn tcw_from_ri(&self) -> [f32; 16] {
        let mut t = IDENT4F;
        for r in 0..3 {
            for c in 0..3 {
                t[c * 4 + r] = self.ri[r][c] as f32;
            }
            t[12 + r] = self.ti[r] as f32;
        }
        t
    }

    fn inliers_out(&self, v: &[bool]) -> Vec<bool> {
        let mut o = vec![false; self.n_matches];
        for (i, &b) in v.iter().enumerate() {
            if b {
                o[self.kp_idx[i]] = true;
            }
        }
        o
    }

    /// `iterate(nIterations, ...)`. `rand(a, b)` = `DUtils::Random::RandomInt(a, b)`.
    pub fn iterate(&mut self, n_iterations: i32, rand: &mut impl FnMut(i32, i32) -> i32) -> IterateOut {
        let mut out = IterateOut { ok: false, no_more: false, n_inliers: 0, inliers: Vec::new(), tcw: IDENT4F };
        let n = self.p2d.len() as i32;
        if n < self.ransac_min_inliers {
            out.no_more = true;
            return out;
        }
        let mut cur = 0;
        let min_set = self.ransac_min_set as usize;
        let mut avail: Vec<usize> = Vec::with_capacity(n as usize);
        while self.n_iterations < self.ransac_max_its || cur < n_iterations {
            cur += 1;
            self.n_iterations += 1;
            avail.clear();
            avail.extend(0..n as usize);
            let mut fs = [[0.0f64; 3]; 6];
            let mut ps = [[0.0f64; 3]; 6];
            for i in 0..min_set {
                let ri = rand(0, avail.len() as i32 - 1) as usize;
                let idx = avail[ri];
                if i < 6 {
                    fs[i] = self.bearings[idx];
                    ps[i] = self.p3dw[idx];
                }
                avail[ri] = *avail.last().unwrap();
                avail.pop();
            }
            assert_eq!(min_set, 6, "MLPnP 최소 집합은 6 점만 옮김");
            let (rm, tv) = compute_pose6(&fs, &ps);
            for r in 0..3 {
                for c in 0..3 {
                    self.ri[r][c] = rm[c * 3 + r];
                }
            }
            self.ti = tv;
            self.check_inliers();
            if self.n_inliers_i >= self.ransac_min_inliers {
                if self.n_inliers_i > self.n_best_inliers {
                    self.best_inliers = self.inliers_i.clone();
                    self.n_best_inliers = self.n_inliers_i;
                    self.best_tcw = self.tcw_from_ri();
                }
                // Refine: computePose 결과는 버려지고 CheckInliers 는 같은 mRi 로 같은 결과
                if self.n_inliers_i > self.ransac_min_inliers {
                    out.ok = true;
                    out.n_inliers = self.n_inliers_i;
                    out.inliers = self.inliers_out(&self.inliers_i);
                    out.tcw = self.tcw_from_ri();
                    return out;
                }
            }
        }
        if self.n_iterations >= self.ransac_max_its {
            out.no_more = true;
            if self.n_best_inliers >= self.ransac_min_inliers {
                out.ok = true;
                out.n_inliers = self.n_best_inliers;
                out.inliers = self.inliers_out(&self.best_inliers);
                out.tcw = self.best_tcw;
                return out;
            }
        }
        out
    }
}

#[doc(hidden)]
pub mod testing {
    //! 비교 하네스용(원본 라이브러리 함수와 하나씩 대조)
    pub fn compute_pose6_trace(f: &[[f64; 3]; 6], p: &[[f64; 3]; 6]) -> Vec<f64> {
        let mut v = Vec::new();
        super::compute_pose6_tr(f, p, &mut |x: &[f64]| v.extend_from_slice(x));
        v
    }
    pub fn nullspace(f: &[f64; 3]) -> [[f64; 3]; 2] {
        super::nullspace(f)
    }
    pub fn svd12_v(m: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
        super::jacobi_svd::<12>(m, false).2
    }
    pub fn svd3_uv(m: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let (_, u, v) = super::jacobi_svd::<3>(m, true);
        (u, v)
    }
    pub fn inverse4(m: &[f64; 16]) -> [f64; 16] {
        super::inverse4(m)
    }
    pub fn det3(m: &[f64; 9]) -> f64 {
        super::det3(m)
    }
    pub fn fullpiv_rank3(m: &[f64; 9]) -> usize {
        super::fullpiv_rank3(m)
    }
    pub fn jacs(pt: &[f64; 3], r: &[f64; 3], s: &[f64; 3], w: &[f64; 3], t: &[f64; 3]) -> [f64; 12] {
        super::mlpnp_jacs(pt, r, s, w, t)
    }
    pub fn gauss_newton(x: &mut [f64; 6], pts: &[[f64; 3]; 6], ns: &[[[f64; 3]; 2]; 6]) {
        super::gauss_newton(x, pts, ns)
    }
}
