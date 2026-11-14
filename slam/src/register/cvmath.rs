//! OpenCV 4.10 의 수치 함수 중 PnP 에 쓰는 것을 같은 연산 순서로 옮긴 것(비트 단위로 같게).
//! 원본: core/src/lapack.cpp(JacobiSVDImpl_, SVBkSb, _SVDcompute, invert/solve 의 SVD 경로),
//! core/src/matmul.simd.hpp(MulTransposedR), core/include/opencv2/core/operations.hpp(RNG),
//! calib3d/src/calibration.cpp(cvRodrigues2, cvProjectPoints2Internal), calib3d/src/undistort.dispatch.cpp.
//! 이 빌드(HAVE_LAPACK 없음)는 SVD 를 모두 JacobiSVDImpl_ 로 한다.

/// cv::RNG(곱셈-올림 난수). state 0 이면 0xffffffff
#[derive(Clone, Copy, Debug)]
pub struct CvRng(pub u64);

impl CvRng {
    pub fn new(seed: u64) -> Self {
        CvRng(if seed == 0 { 0xffff_ffff } else { seed })
    }
    #[inline]
    pub fn next(&mut self) -> u32 {
        self.0 = (self.0 as u32 as u64).wrapping_mul(4164903690u64).wrapping_add(self.0 >> 32);
        self.0 as u32
    }
    /// RNG::uniform(int a, int b)
    #[inline]
    pub fn uniform(&mut self, a: i32, b: i32) -> i32 {
        if a == b {
            a
        } else {
            (self.next() % ((b - a) as u32)) as i32 + a
        }
    }
}

/// cvRound(double): 짝수 쪽으로 반올림(lrint)
#[inline]
pub fn cv_round(v: f64) -> i32 {
    v.round_ties_even() as i32
}

/// JacobiSVDImpl_<double>. at: n 행 x m 열(행 간격 astep), vt: n x n(간격 vstep) 또는 없음.
/// 끝나면 at 의 행 = 왼쪽 특이벡터, w = 특이값(내림차순)
pub fn jacobi_svd(at: &mut [f64], astep: usize, w_out: &mut [f64], mut vt: Option<(&mut [f64], usize)>, m: usize, n: usize, n1: usize) {
    let minval = f64::MIN_POSITIVE;
    let eps = f64::EPSILON * 10.0;
    let mut w = [0f64; 16];
    assert!(n <= 16);
    let max_iter = m.max(30);
    for i in 0..n {
        let mut sd = 0f64;
        for k in 0..m {
            let t = at[i * astep + k];
            sd += t * t;
        }
        w[i] = sd;
        if let Some((v, vs)) = vt.as_mut() {
            for k in 0..n {
                v[i * *vs + k] = 0.0;
            }
            v[i * *vs + i] = 1.0;
        }
    }
    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n.saturating_sub(1) {
            for j in i + 1..n {
                let (a, b) = (w[i], w[j]);
                let mut p = 0f64;
                for k in 0..m {
                    p += at[i * astep + k] * at[j * astep + k];
                }
                if p.abs() <= eps * (a * b).sqrt() {
                    continue;
                }
                p *= 2.0;
                let beta = a - b;
                let gamma = p.hypot(beta);
                let (c, s);
                if beta < 0.0 {
                    let delta = (gamma - beta) * 0.5;
                    s = (delta / gamma).sqrt();
                    c = p / (gamma * s * 2.0);
                } else {
                    c = ((gamma + beta) / (gamma * 2.0)).sqrt();
                    s = p / (gamma * c * 2.0);
                }
                let (mut a2, mut b2) = (0f64, 0f64);
                for k in 0..m {
                    let ai = at[i * astep + k];
                    let aj = at[j * astep + k];
                    let t0 = c * ai + s * aj;
                    let t1 = -s * ai + c * aj;
                    at[i * astep + k] = t0;
                    at[j * astep + k] = t1;
                    a2 += t0 * t0;
                    b2 += t1 * t1;
                }
                w[i] = a2;
                w[j] = b2;
                changed = true;
                if let Some((v, vs)) = vt.as_mut() {
                    // VBLAS<double>::givens 는 t0 = a*c + b*s, t1 = b*c - a*s (스칼라와 같은 값)
                    for k in 0..n {
                        let vi = v[i * *vs + k];
                        let vj = v[j * *vs + k];
                        v[i * *vs + k] = vi * c + vj * s;
                        v[j * *vs + k] = vj * c - vi * s;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..n {
        let mut sd = 0f64;
        for k in 0..m {
            let t = at[i * astep + k];
            sd += t * t;
        }
        w[i] = sd.sqrt();
    }
    for i in 0..n.saturating_sub(1) {
        let mut j = i;
        for k in i + 1..n {
            if w[j] < w[k] {
                j = k;
            }
        }
        if i != j {
            w.swap(i, j);
            if let Some((v, vs)) = vt.as_mut() {
                for k in 0..m {
                    at.swap(i * astep + k, j * astep + k);
                }
                for k in 0..n {
                    v.swap(i * *vs + k, j * *vs + k);
                }
            }
        }
    }
    w_out[..n].copy_from_slice(&w[..n]);
    if vt.is_none() {
        return;
    }
    let mut rng = CvRng::new(0x12345678);
    for i in 0..n1 {
        let mut sd = if i < n { w[i] } else { 0.0 };
        let mut ii = 0;
        while ii < 100 && sd <= minval {
            let val0 = 1.0 / m as f64;
            for k in 0..m {
                at[i * astep + k] = if rng.next() & 256 != 0 { val0 } else { -val0 };
            }
            for _ in 0..2 {
                for j in 0..i {
                    let mut sd2 = 0f64;
                    for k in 0..m {
                        sd2 += at[i * astep + k] * at[j * astep + k];
                    }
                    let mut asum = 0f64;
                    for k in 0..m {
                        let t = at[i * astep + k] - sd2 * at[j * astep + k];
                        at[i * astep + k] = t;
                        asum += t.abs();
                    }
                    asum = if asum > eps * 100.0 { 1.0 / asum } else { 0.0 };
                    for k in 0..m {
                        at[i * astep + k] *= asum;
                    }
                }
            }
            sd = 0.0;
            for k in 0..m {
                let t = at[i * astep + k];
                sd += t * t;
            }
            sd = sd.sqrt();
            ii += 1;
        }
        let s = if sd > minval { 1.0 / sd } else { 0.0 };
        for k in 0..m {
            at[i * astep + k] *= s;
        }
    }
}

/// SVD::compute(a, w, u, vt) 결과(정사각 n x n, 작은 행렬용).
/// 반환: (w, ut, vt) — ut 의 행 i = 왼쪽 특이벡터 i(= OpenCV u 의 열 i), vt 의 행 i = 오른쪽 특이벡터 i
pub fn svd_square<const N: usize>(a: &[[f64; N]; N]) -> ([f64; N], [[f64; N]; N], [[f64; N]; N]) {
    // temp_a = transpose(a): temp_a[i][k] = a[k][i]
    let mut at = vec![0f64; N * N];
    for i in 0..N {
        for k in 0..N {
            at[i * N + k] = a[k][i];
        }
    }
    let mut w = [0f64; N];
    let mut v = vec![0f64; N * N];
    jacobi_svd(&mut at, N, &mut w, Some((&mut v, N)), N, N, N);
    let mut ut = [[0f64; N]; N];
    let mut vt = [[0f64; N]; N];
    for i in 0..N {
        for k in 0..N {
            ut[i][k] = at[i * N + k];
            vt[i][k] = v[i * N + k];
        }
    }
    (w, ut, vt)
}

/// MatrAXPY: y[i][0..n] += a[i] * x[0..n]
#[inline]
fn matr_axpy(m: usize, n: usize, x: &[f64], dx: usize, a: &[f64], inca: usize, y: &mut [f64], dy: usize) {
    for i in 0..m {
        let s = a[i * inca];
        for j in 0..n {
            y[i * dy + j] = y[i * dy + j] + s * x[i * dx + j];
        }
    }
}

/// cv::invert(src, dst, DECOMP_SVD) 의 3x3 경우
pub fn invert3_svd(src: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let (w, ut, vt) = svd_square::<3>(src);
    // SVD::backSubst(w, u, vt, 없음): uT=false(u 의 열 i = ut 의 행 i), vT=true
    let eps = f64::EPSILON * 2.0;
    let mut threshold = 0f64;
    for i in 0..3 {
        threshold += w[i];
    }
    threshold *= eps;
    let mut x = [0f64; 9];
    let mut buffer = [0f64; 3];
    for i in 0..3 {
        let mut wi = w[i];
        if wi.abs() <= threshold {
            continue;
        }
        wi = 1.0 / wi;
        for j in 0..3 {
            buffer[j] = ut[i][j] * wi;
        }
        // MatrAXPY(n, nb, buffer, 0, v(행 i), 1, x, ldx)
        let vrow = vt[i];
        matr_axpy(3, 3, &buffer, 0, &vrow, 1, &mut x, 3);
    }
    [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]]
}

/// cv::solve(A(m x n), b(m), DECOMP_SVD), m >= n, 해는 n 개
pub fn solve_svd<const M: usize, const N: usize>(a: &[[f64; N]; M], b: &[f64; M]) -> [f64; N] {
    // a = transpose(src): n x m
    let mut at = [[0f64; M]; N];
    for i in 0..N {
        for k in 0..M {
            at[i][k] = a[k][i];
        }
    }
    let mut flat = vec![0f64; N * M];
    for i in 0..N {
        flat[i * M..(i + 1) * M].copy_from_slice(&at[i]);
    }
    let mut w = [0f64; N];
    let mut v = vec![0f64; N * N];
    jacobi_svd(&mut flat, M, &mut w, Some((&mut v, N)), M, N, N);
    // SVBkSb(m, n, w, u=a(uT), v(vT), b, nb=1)
    let eps = f64::EPSILON * 2.0;
    let nm = M.min(N);
    let mut threshold = 0f64;
    for i in 0..nm {
        threshold += w[i];
    }
    threshold *= eps;
    let mut x = [0f64; N];
    for i in 0..nm {
        let mut wi = w[i];
        if wi.abs() <= threshold {
            continue;
        }
        wi = 1.0 / wi;
        let mut s = 0f64;
        for j in 0..M {
            s += flat[i * M + j] * b[j];
        }
        s *= wi;
        for j in 0..N {
            x[j] = x[j] + s * v[i * N + j];
        }
    }
    x
}

/// cvMulTransposed(src, dst, order=1): dst = srcᵀ·src (윗삼각 계산 후 대칭 복사)
pub fn mul_transposed_ata(src: &[f64], rows: usize, cols: usize, dst: &mut [f64]) {
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0f64;
            for k in 0..rows {
                s += src[k * cols + i] * src[k * cols + j];
            }
            dst[i * cols + j] = s;
        }
    }
    for i in 0..cols {
        for j in 0..i {
            dst[i * cols + j] = dst[j * cols + i];
        }
    }
}

/// cvRodrigues2: 회전 벡터 → 3x3(행 우선)
pub fn rodrigues_v2m(r: [f64; 3]) -> [f64; 9] {
    let theta = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if theta < f64::EPSILON {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;
    let itheta = if theta != 0.0 { 1.0 / theta } else { 0.0 };
    let (x, y, z) = (r[0] * itheta, r[1] * itheta, r[2] * itheta);
    let rrt = [x * x, x * y, x * z, x * y, y * y, y * z, x * z, y * z, z * z];
    let rx = [0.0, -z, y, z, 0.0, -x, -y, x, 0.0];
    let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut out = [0f64; 9];
    for k in 0..9 {
        out[k] = (c * eye[k] + c1 * rrt[k]) + s * rx[k];
    }
    out
}

/// cvRodrigues2: 3x3(행 우선) → 회전 벡터
pub fn rodrigues_m2v(rin: &[f64; 9]) -> [f64; 3] {
    // checkRange(R, -100, 100): NaN 이거나 [-100, 100) 밖이면 0
    if rin.iter().any(|v| v.is_nan() || !(*v >= -100.0 && *v < 100.0)) {
        return [0.0; 3];
    }
    let a = [[rin[0], rin[1], rin[2]], [rin[3], rin[4], rin[5]], [rin[6], rin[7], rin[8]]];
    let (_w, ut, vt) = svd_square::<3>(&a);
    // R = U*Vt, U(i,k) = ut[k][i]
    let mut r = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0f64;
            for k in 0..3 {
                s += ut[k][i] * vt[k][j];
            }
            r[i][j] = s;
        }
    }
    let mut rv = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let s = ((rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]) * 0.25).sqrt();
    let mut c = (r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5;
    c = if c > 1.0 {
        1.0
    } else if c < -1.0 {
        -1.0
    } else {
        c
    };
    let mut theta = c.acos();
    if s < 1e-5 {
        if c > 0.0 {
            rv = [0.0, 0.0, 0.0];
        } else {
            let mut t = (r[0][0] + 1.0) * 0.5;
            rv[0] = t.max(0.0).sqrt();
            t = (r[1][1] + 1.0) * 0.5;
            rv[1] = t.max(0.0).sqrt() * if r[0][1] < 0.0 { -1.0 } else { 1.0 };
            t = (r[2][2] + 1.0) * 0.5;
            rv[2] = t.max(0.0).sqrt() * if r[0][2] < 0.0 { -1.0 } else { 1.0 };
            if rv[0].abs() < rv[1].abs() && rv[0].abs() < rv[2].abs() && (r[1][2] > 0.0) != (rv[1] * rv[2] > 0.0) {
                rv[2] = -rv[2];
            }
            theta /= (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
            rv = [rv[0] * theta, rv[1] * theta, rv[2] * theta];
        }
    } else {
        let mut vth = 1.0 / (2.0 * s);
        vth *= theta;
        rv = [rv[0] * vth, rv[1] * vth, rv[2] * vth];
    }
    rv
}

/// 카메라 내부값(왜곡 없음). projectPoints/undistortPoints 의 K
#[derive(Clone, Copy, Debug)]
pub struct K {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// cvProjectPoints2Internal(왜곡 0): 회전 행렬 r(행 우선) 과 t 로 한 점 투영, 결과는 double(호출 쪽에서 float 로)
#[inline]
pub fn project_one(r: &[f64; 9], t: &[f64; 3], k: &K, p: [f32; 3]) -> (f64, f64) {
    let (xx, yy, zz) = (p[0] as f64, p[1] as f64, p[2] as f64);
    let mut x = r[0] * xx + r[1] * yy + r[2] * zz + t[0];
    let mut y = r[3] * xx + r[4] * yy + r[5] * zz + t[1];
    let z = r[6] * xx + r[7] * yy + r[8] * zz + t[2];
    let z = if z != 0.0 { 1.0 / z } else { 1.0 };
    x *= z;
    y *= z;
    // 왜곡 계수 0: cdist=1, icdist2=1, 기울기 행렬 = 단위 → xd = x*1*1 + 0 ... = x
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let a1 = 2.0 * x * y;
    let a2 = r2 + 2.0 * x * x;
    let a3 = r2 + 2.0 * y * y;
    let cdist = 1.0 + 0.0 * r2 + 0.0 * r4 + 0.0 * r6;
    let icdist2 = 1.0 / (1.0 + 0.0 * r2 + 0.0 * r4 + 0.0 * r6);
    let xd0 = x * cdist * icdist2 + 0.0 * a1 + 0.0 * a2 + 0.0 * r2 + 0.0 * r4;
    let yd0 = y * cdist * icdist2 + 0.0 * a3 + 0.0 * a1 + 0.0 * r2 + 0.0 * r4;
    // matTilt(단위)*Vec3d(xd0, yd0, 1): Matx 곱 = 1*xd0 + 0*yd0 + 0*1
    let vx = 1.0 * xd0 + 0.0 * yd0 + 0.0 * 1.0;
    let vy = 0.0 * xd0 + 1.0 * yd0 + 0.0 * 1.0;
    let vz = 0.0 * xd0 + 0.0 * yd0 + 1.0 * 1.0;
    let inv_proj = if vz != 0.0 { 1.0 / vz } else { 1.0 };
    let xd = inv_proj * vx;
    let yd = inv_proj * vy;
    (xd * k.fx + k.cx, yd * k.fy + k.cy)
}

/// undistortPoints(왜곡 0, R/P 없음): 화소 → 정규 좌표(float 출력)
#[inline]
pub fn undistort_one(u: f32, v: f32, k: &K) -> (f32, f32) {
    let ifx = 1.0 / k.fx;
    let ify = 1.0 / k.fy;
    let x = (u as f64 - k.cx) * ifx;
    let y = (v as f64 - k.cy) * ify;
    // 왜곡 보정 반복은 계수 0 이면 값이 그대로, RR = 단위: xx = 1*x + 0*y + 0
    let xx = 1.0 * x + 0.0 * y + 0.0;
    let yy = 0.0 * x + 1.0 * y + 0.0;
    let ww = 1.0 / (0.0 * x + 0.0 * y + 1.0);
    ((xx * ww) as f32, (yy * ww) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_opencv() {
        // OpenCV: RNG rng((uint64)-1); 첫 next() 값
        let mut r = CvRng::new(u64::MAX);
        let a = r.next();
        let expect = ((0xffff_ffffu64 * 4164903690u64) + 0xffff_ffff) as u32;
        assert_eq!(a, expect);
    }

    #[test]
    fn rodrigues_roundtrip() {
        let r = [0.1, -0.3, 0.25];
        let m = rodrigues_v2m(r);
        let r2 = rodrigues_m2v(&m);
        for i in 0..3 {
            assert!((r[i] - r2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn invert3() {
        let a = [[2.0, 1.0, 0.5], [0.3, 3.0, 0.1], [0.2, 0.4, 1.5]];
        let i = invert3_svd(&a);
        for r in 0..3 {
            for c in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[r][k] * i[k][c];
                }
                assert!((s - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
