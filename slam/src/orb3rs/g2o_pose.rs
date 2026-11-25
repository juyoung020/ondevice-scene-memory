//! ORB-SLAM3 Thirdparty/g2o 의 단일 자세(6 자유도) 최적화에 쓰이는 부분을 원본과 비트 같게 옮김.
//!
//! - `SE3Quat`(types/se3quat.h): 사원수 곱(Eigen SSE2 `quat_product` 순서) · 점 사상 · `exp` · `normalizeRotation`
//! - `VertexSE3Expmap::oplusImpl` = `exp(update) * estimate`
//! - `EdgeSE3ProjectXYZOnlyPose`(ORB_SLAM3 OptimizableTypes) · `EdgeStereoSE3ProjectXYZOnlyPose`(g2o) 오차 · 야코비안
//! - `RobustKernelHuber::robustify`, `BaseUnaryEdge::constructQuadraticForm`
//! - `LinearSolverDense`(Eigen `LDLT<MatrixXd>` 비블록 알고리즘 + 풀이)
//! - `OptimizationAlgorithmLevenberg::solve`(Raul 멈춤 조건 포함)
//!
//! 원본은 `-O3 -march=native`(GCC 13, FMA 켜짐) 빌드라 곱 다음 덧셈이 FMA 로 합쳐진다. 어느 곱이 합쳐지는지는
//! 저장소 밖 하네스(`~/opt/orb3_opt_bg/probe.cc`)로 같은 식을 무작위 입력에 돌려 비트 비교해 정했다(`fma` 표시).

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Eigen `Quaterniond` 저장 순서 (x, y, z, w)
pub type Quat = [f64; 4];

/// Eigen SSE2 `quat_product<double>`: t1 = ww·xy + yy·zw, t2 = zz·xy − xx·zw, xy = addsub(t1, swap t2) …
#[inline]
pub fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    // GCC 는 pmul+pmul 덧셈에서 두 번째 곱을 FMA 로 합침(probe 역어셈블로 확인)
    let t1x = fma(ay, bz, aw * bx);
    let t1y = fma(ay, bw, aw * by);
    let t2x = fma(-ax, bz, az * bx);
    let t2y = fma(-ax, bw, az * by);
    let u1z = fma(-ay, bx, aw * bz);
    let u1w = fma(-ay, by, aw * bw);
    let u2z = fma(ax, bx, az * bz);
    let u2w = fma(ax, by, az * bw);
    [t1x - t2y, t1y + t2x, u1z + u2w, u1w - u2z]
}

#[inline(always)]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    // x · y 는 SLP 묶음(두 번째 곱 합침), z 는 스칼라(첫 곱 합침) — probe 역어셈블
    [fma(-a[2], b[1], a[1] * b[2]), fma(-a[0], b[2], a[2] * b[0]), fma(a[0], b[1], -(a[1] * b[0]))]
}

/// `QuaternionBase::_transformVector`: uv = 2·(q.vec × v); v + w·uv + q.vec × uv
#[inline]
pub fn quat_rotate(q: &Quat, v: &[f64; 3]) -> [f64; 3] {
    let qv = [q[0], q[1], q[2]];
    let mut uv = cross(&qv, v);
    for e in &mut uv {
        *e += *e;
    }
    let c = cross(&qv, &uv);
    [fma(q[3], uv[0], v[0]) + c[0], fma(q[3], uv[1], v[1]) + c[1], fma(q[3], uv[2], v[2]) + c[2]]
}

/// Packet4d squaredNorm 의 수평 합 순서 (x²+z²)+(y²+w²)
#[inline(always)]
fn sqnorm4(q: &Quat) -> f64 {
    (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])
}

/// `SE3Quat::normalizeRotation`: w<0 이면 부호 뒤집고 정규화
#[inline]
pub fn normalize_rotation(q: &mut Quat) {
    if q[3] < 0.0 {
        for e in q.iter_mut() {
            *e = -*e;
        }
    }
    let z = sqnorm4(q);
    if z > 0.0 {
        let n = z.sqrt();
        for e in q.iter_mut() {
            *e /= n;
        }
    }
}

/// 3x3 행렬(열 우선)에서 사원수 — Eigen `quaternionbase_assign_impl<Matrix3>`
pub fn quat_from_mat3(m: &[f64; 9]) -> Quat {
    let c = |r: usize, k: usize| m[k * 3 + r];
    let mut q = [0.0; 4];
    // 원본(libORB_SLAM3 역어셈블) 의 대각 합 순서: (R22 + R11) + R00
    let mut t = (c(2, 2) + c(1, 1)) + c(0, 0);
    if t > 0.0 {
        t = (t + 1.0).sqrt();
        q[3] = 0.5 * t;
        t = 0.5 / t;
        q[0] = (c(2, 1) - c(1, 2)) * t;
        q[1] = (c(0, 2) - c(2, 0)) * t;
        q[2] = (c(1, 0) - c(0, 1)) * t;
    } else {
        let mut i = 0;
        if c(1, 1) > c(0, 0) {
            i = 1;
        }
        if c(2, 2) > c(i, i) {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        t = ((c(i, i) - c(j, j)) - c(k, k) + 1.0).sqrt();
        q[i] = 0.5 * t;
        t = 0.5 / t;
        q[3] = (c(k, j) - c(j, k)) * t;
        q[j] = (c(j, i) + c(i, j)) * t;
        q[k] = (c(k, i) + c(i, k)) * t;
    }
    q
}

/// Ω·Ω (Ω = skew(ω), 대각 0). 비대각은 0 아닌 항 하나라 순서 무관, 대각 두 항의 FMA 자리는
/// probe 무작위 탐색으로 정함: (0,0) · (1,1) 은 뒤 항 합침, (2,2) 는 앞 항 합침.
#[inline(always)]
fn skew_sq(om: &[f64; 9]) -> [f64; 9] {
    let o = |r: usize, c: usize| om[c * 3 + r];
    let mut m = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            if i != j {
                let k = 3 - i - j;
                m[j * 3 + i] = o(i, k) * o(k, j);
            }
        }
    }
    m[0] = fma(o(0, 2), o(2, 0), o(0, 1) * o(1, 0));
    m[4] = fma(o(1, 2), o(2, 1), o(1, 0) * o(0, 1));
    m[8] = fma(o(2, 0), o(0, 2), o(2, 1) * o(1, 2));
    m
}

/// V·υ (SE3Quat::exp). 0 · 1 행은 Packet2d 로 열 순서 FMA, 2 행은 스칼라로 (V21·u1) → V22·u2 → V20·u0 순(역어셈블)
#[inline(always)]
fn mat3_vec(a: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    let mut r = [0.0; 3];
    for i in 0..2 {
        let mut s = a[i] * v[0];
        s = fma(a[3 + i], v[1], s);
        s = fma(a[6 + i], v[2], s);
        r[i] = s;
    }
    r[2] = fma(a[2], v[0], fma(a[8], v[2], a[5] * v[1]));
    r
}

/// g2o `SE3Quat`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SE3Quat {
    pub r: Quat,
    pub t: [f64; 3],
}

impl SE3Quat {
    pub fn identity() -> Self {
        Self { r: [0.0, 0.0, 0.0, 1.0], t: [0.0; 3] }
    }

    /// `SE3Quat(const Quaterniond&, const Vector3d&)` — normalizeRotation 포함
    pub fn new(q: Quat, t: [f64; 3]) -> Self {
        let mut r = q;
        normalize_rotation(&mut r);
        Self { r, t }
    }

    /// `map(xyz) = r*xyz + t`
    #[inline]
    pub fn map(&self, x: &[f64; 3]) -> [f64; 3] {
        let p = quat_rotate(&self.r, x);
        [p[0] + self.t[0], p[1] + self.t[1], p[2] + self.t[2]]
    }

    /// `operator*`: t += r*t2; r *= r2; normalizeRotation
    pub fn mul(&self, o: &SE3Quat) -> SE3Quat {
        let p = quat_rotate(&self.r, &o.t);
        let mut r = quat_mul(&self.r, &o.r);
        normalize_rotation(&mut r);
        SE3Quat { r, t: [self.t[0] + p[0], self.t[1] + p[1], self.t[2] + p[2]] }
    }

    /// `SE3Quat::exp(update)` — update = (ω, υ)
    pub fn exp(u: &[f64; 6]) -> SE3Quat {
        let w = [u[0], u[1], u[2]];
        let ups = [u[3], u[4], u[5]];
        let theta = fma(w[2], w[2], w[0] * w[0] + w[1] * w[1]).sqrt();
        // skew(ω) 열 우선
        let om = [0.0, w[2], -w[1], -w[2], 0.0, w[0], w[1], -w[0], 0.0];
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let om2 = skew_sq(&om);
        let mut r = [0.0; 9];
        let v;
        if theta < 0.00001 {
            for k in 0..9 {
                r[k] = (id[k] + om[k]) + om2[k];
            }
            v = r;
        } else {
            let (s, c) = (theta.sin(), theta.cos());
            let a = s / theta;
            let b = (1.0 - c) / (theta * theta);
            let cc = (theta - s) / theta.powf(3.0);
            let mut vv = [0.0; 9];
            for k in 0..9 {
                r[k] = fma(b, om2[k], fma(a, om[k], id[k]));
                vv[k] = fma(cc, om2[k], fma(b, om[k], id[k]));
            }
            v = vv;
        }
        SE3Quat::new(quat_from_mat3(&r), mat3_vec(&v, &ups))
    }
}

/// `RobustKernelHuber::robustify` → (ρ, ρ', ρ'')
#[inline]
pub fn huber(delta: f64, e: f64) -> [f64; 3] {
    // g2o 의 RobustKernelHuber::dsqr 는 float 칸(역어셈블 vcvtss2sd 로 확인)
    let dsqr = f64::from((delta * delta) as f32);
    if e <= dsqr {
        [e, 1.0, 0.0]
    } else {
        let sqrte = e.sqrt();
        let r1 = delta / sqrte;
        [fma(2.0 * sqrte, delta, -dsqr), r1, -0.5 * r1 / e]
    }
}

/// `BaseUnaryEdge<D>::constructQuadraticForm` 의 누적(정보 행렬은 대각 `info`).
/// `jac` 는 D x 6 열 우선. 강건이면 `rho1` 을 곱한 정보 행렬, 아니면 그대로.
#[inline]
pub fn add_quadratic<const D: usize>(jac: &[f64], info: f64, err: &[f64; D], rho1: Option<f64>, h: &mut [f64; 36], b: &mut [f64; 6]) {
    // b -= (ρ'·Aᵀ)·Ω·e  /  Aᵀ·Ω·e
    // H += Aᵀ·(ρ'Ω)·A   /  Aᵀ·Ω·A
    let wo = match rho1 {
        Some(r) => r * info,
        None => info,
    };
    // T = Aᵀ·W (6 x D), W 대각
    let mut t = [[0.0f64; 3]; 6];
    let mut m = [[0.0f64; 3]; 6];
    for i in 0..6 {
        for k in 0..D {
            let a = jac[i * D + k];
            t[i][k] = a * wo;
            m[i][k] = match rho1 {
                Some(r) => (r * a) * info,
                None => a * info,
            };
        }
    }
    for i in 0..6 {
        let mut s = m[i][0] * err[0];
        for k in 1..D {
            s = fma(m[i][k], err[k], s);
        }
        b[i] -= s;
    }
    for j in 0..6 {
        for i in 0..6 {
            let mut s = t[i][0] * jac[j * D];
            for k in 1..D {
                s = fma(t[i][k], jac[j * D + k], s);
            }
            h[j * 6 + i] += s;
        }
    }
}

/// Eigen `LDLT<MatrixXd>`(Lower, 비블록) 분해 후 `solve`. g2o `LinearSolverDense::solve` 와 같이
/// `isPositive()` 가 아니면 false(x 는 건드리지 않음). `h` 는 열 우선 6x6(아래 삼각만 읽음).
pub fn ldlt_solve6(h: &[f64; 36], bv: &[f64; 6], x: &mut [f64; 6]) -> bool {
    let (m, tr, sign) = ldlt_factor6(h);
    if !sign_positive(sign) {
        return false;
    }
    ldlt_back6(&m, &tr, bv, x);
    true
}

const N: usize = 6;
#[inline(always)]
fn at(r: usize, c: usize) -> usize {
    c * N + r
}

/// `ldlt_inplace<Lower>::unblocked`: (대각 D · 아래 L 을 담은 행렬, 전치 표, 부호 0 양 · 1 음 · 2 영 · 3 부정)
pub fn ldlt_factor6(h: &[f64; 36]) -> ([f64; 36], [usize; 6], u8) {
    let mut m = *h;
    let mut tr = [0usize; N];
    let mut tmp = [0.0f64; N];
    // sign: 0 Positive, 1 Negative, 2 Zero, 3 Indefinite
    let mut sign = 2u8;
    for k in 0..N {
        let mut big = k;
        let mut bigv = m[at(k, k)].abs();
        for i in k + 1..N {
            let v = m[at(i, i)].abs();
            if v > bigv {
                bigv = v;
                big = i;
            }
        }
        tr[k] = big;
        if k != big {
            for c in 0..k {
                m.swap(at(k, c), at(big, c));
            }
            for r in big + 1..N {
                m.swap(at(r, k), at(r, big));
            }
            m.swap(at(k, k), at(big, big));
            for i in k + 1..big {
                let t = m[at(i, k)];
                m[at(i, k)] = m[at(big, i)];
                m[at(big, i)] = t;
            }
        }
        let rs = N - k - 1;
        if k > 0 {
            for j in 0..k {
                tmp[j] = m[at(j, j)] * m[at(k, j)];
            }
            // 대각: 곱 반올림 후 순차 덧셈(합침 없음)
            let mut s = m[at(k, 0)] * tmp[0];
            for j in 1..k {
                s += m[at(k, j)] * tmp[j];
            }
            m[at(k, k)] -= s;
            // A21 -= A20·temp: Eigen GEMV — 남은 행이 4 이상이면 앞 4 행 Packet4d, 그 뒤 2 이상이면 2 행 Packet2d
            // (0 에서 시작하는 FMA 사슬), 나머지 행은 스칼라(곱+덧셈). 결과는 res + (−1)·c
            let rs = N - k - 1;
            let mut npk = 0;
            if rs >= 4 {
                npk = 4;
            }
            if rs - npk >= 2 {
                npk += 2;
            }
            for (ri, i) in (k + 1..N).enumerate() {
                let mut c = 0.0;
                if ri < npk {
                    for j in 0..k {
                        c = fma(m[at(i, j)], tmp[j], c);
                    }
                } else {
                    for j in 0..k {
                        c += m[at(i, j)] * tmp[j];
                    }
                }
                m[at(i, k)] = fma(c, -1.0, m[at(i, k)]);
            }
        }
        let akk = m[at(k, k)];
        let valid = akk.abs() > 0.0;
        if k == 0 && !valid {
            // 대각 전부 0: Eigen 은 전치를 항등으로 두고 ZeroSign 으로 바로 끝냄(isPositive 참)
            for (j, t) in tr.iter_mut().enumerate() {
                *t = j;
            }
            sign = 2;
            break;
        }
        if rs > 0 && valid {
            for i in k + 1..N {
                m[at(i, k)] /= akk;
            }
        }
        match sign {
            0 => {
                if akk < 0.0 {
                    sign = 3
                }
            }
            1 => {
                if akk > 0.0 {
                    sign = 3
                }
            }
            2 => {
                if akk > 0.0 {
                    sign = 0
                } else if akk < 0.0 {
                    sign = 1
                }
            }
            _ => {}
        }
    }
    (m, tr, sign)
}

/// `LDLT::_solve_impl_transposed`: x = Pᵀ L⁻ᵀ D⁻¹ L⁻¹ P b
pub fn ldlt_back6(m: &[f64; 36], tr: &[usize; 6], bv: &[f64; 6], x: &mut [f64; 6]) {
    let mut d = *bv;
    for k in 0..N {
        d.swap(k, tr[k]);
    }
    for i in 0..N {
        for j in i + 1..N {
            d[j] = fma(-d[i], m[at(j, i)], d[j]);
        }
    }
    for i in 0..N {
        let di = m[at(i, i)];
        if di.abs() > f64::MIN_POSITIVE {
            d[i] /= di;
        } else {
            d[i] = 0.0;
        }
    }
    // Lᵀ 풀이(행 우선 위 삼각, 단위 대각): 항 1 개면 d − a·x 를 FMA 로, 아니면 첫 곱 반올림 · 길이 5 이상이면
    // 1..=4 항은 곱 반올림 후 덧셈 · 나머지 FMA(GCC 자동 벡터화 모양, 역어셈블)
    for i in (0..N - 1).rev() {
        let k = N - 1 - i;
        if k == 1 {
            d[i] = fma(-m[at(i + 1, i)], d[i + 1], d[i]);
            continue;
        }
        let mut s = m[at(i + 1, i)] * d[i + 1];
        let mut j = 1;
        if k >= 5 {
            while j <= 4 {
                s += m[at(i + 1 + j, i)] * d[i + 1 + j];
                j += 1;
            }
        }
        while j < k {
            s = fma(m[at(i + 1 + j, i)], d[i + 1 + j], s);
            j += 1;
        }
        d[i] -= s;
    }
    for k in (0..N).rev() {
        d.swap(k, tr[k]);
    }
    *x = d;
}

#[inline(always)]
fn sign_positive(s: u8) -> bool {
    s == 0 || s == 2
}
