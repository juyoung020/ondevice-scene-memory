//! Eigen 3.4.0 `JacobiSVD<Matrix<f32,N,N>>`(정사각, QR 전처리 없음)와 ORB-SLAM3 `GeometricTools::Triangulate` 를
//! 같은 f32 연산 순서로 옮긴 것(기준 빌드 `-ffp-contract=off` 와 비트 동일).
//! 원본: Eigen/src/SVD/JacobiSVD.h(compute), misc/RealSvd2x2.h(real_2x2_jacobi_svd), Jacobi/Jacobi.h(makeJacobi · 회전 적용).
//! 이 파일은 crate 의 다른 모듈을 쓰지 않는다(저장소 밖 하네스가 `#[path]` 로 그대로 끌어와 비교).

/// 행렬은 a[행][열].
pub type Mat<const N: usize> = [[f32; N]; N];

#[derive(Clone, Copy)]
struct Rot {
    c: f32,
    s: f32,
}

impl Rot {
    fn transpose(self) -> Rot {
        Rot { c: self.c, s: -self.s }
    }
    // JacobiRotation::operator*
    fn mul(self, o: Rot) -> Rot {
        Rot { c: self.c * o.c - self.s * o.s, s: self.c * o.s + self.s * o.c }
    }
}

// apply_rotation_in_the_plane(x, y, j): x = c·x + s·y, y = −s·x + c·y (c==1 · s==0 이면 그대로)
#[inline]
fn rot_pair(x: &mut f32, y: &mut f32, j: Rot) {
    let (xi, yi) = (*x, *y);
    *x = j.c * xi + j.s * yi;
    *y = -j.s * xi + j.c * yi;
}

fn apply_left<const N: usize>(m: &mut Mat<N>, p: usize, q: usize, j: Rot) {
    if j.c == 1.0 && j.s == 0.0 {
        return;
    }
    for k in 0..N {
        let (mut x, mut y) = (m[p][k], m[q][k]);
        rot_pair(&mut x, &mut y, j);
        m[p][k] = x;
        m[q][k] = y;
    }
}

fn apply_right<const N: usize>(m: &mut Mat<N>, p: usize, q: usize, j: Rot) {
    let j = j.transpose();
    if j.c == 1.0 && j.s == 0.0 {
        return;
    }
    for r in m.iter_mut() {
        let (mut x, mut y) = (r[p], r[q]);
        rot_pair(&mut x, &mut y, j);
        r[p] = x;
        r[q] = y;
    }
}

// JacobiRotation::makeJacobi(x, y, z)
fn make_jacobi(x: f32, y: f32, z: f32) -> Rot {
    let deno = 2.0f32 * y.abs();
    if deno < f32::MIN_POSITIVE {
        return Rot { c: 1.0, s: 0.0 };
    }
    let tau = (x - z) / deno;
    let w = (tau * tau + 1.0f32).sqrt();
    let t = if tau > 0.0 { 1.0f32 / (tau + w) } else { 1.0f32 / (tau - w) };
    let sign_t = if t > 0.0 { 1.0f32 } else { -1.0f32 };
    let n = 1.0f32 / (t * t + 1.0f32).sqrt();
    Rot { c: n, s: -sign_t * (y / y.abs()) * t.abs() * n }
}

// internal::real_2x2_jacobi_svd
fn real_2x2<const N: usize>(a: &Mat<N>, p: usize, q: usize) -> (Rot, Rot) {
    let mut m: Mat<2> = [[a[p][p], a[p][q]], [a[q][p], a[q][q]]];
    let t = m[0][0] + m[1][1];
    let d = m[1][0] - m[0][1];
    let rot1 = if d.abs() < f32::MIN_POSITIVE {
        Rot { c: 1.0, s: 0.0 }
    } else {
        let u = t / d;
        let tmp = (1.0f32 + u * u).sqrt();
        Rot { c: u / tmp, s: 1.0f32 / tmp }
    };
    apply_left(&mut m, 0, 1, rot1);
    let jr = make_jacobi(m[0][0], m[0][1], m[1][1]);
    (rot1.mul(jr.transpose()), jr)
}

#[inline]
fn maxi(a: f32, b: f32) -> f32 {
    // numext::maxi = std::max(a, b) = (a < b) ? b : a
    if a < b { b } else { a }
}

/// JacobiSVD 결과: 특이값(내림차순), V(ComputeFullV), U(want_u 일 때 ComputeFullU)
pub struct Svd<const N: usize> {
    pub sv: [f32; N],
    pub v: Mat<N>,
    pub u: Option<Mat<N>>,
    /// 0 이 아닌 특이값 수
    pub nonzero: usize,
}

/// `JacobiSVD<Matrix<float,N,N>>(a, ComputeFullV [| ComputeFullU])`. 입력에 NaN · inf 가 있으면 None(원본 InvalidInput).
pub fn jacobi_svd<const N: usize>(a: &Mat<N>, want_u: bool) -> Option<Svd<N>> {
    let precision = 2.0f32 * f32::EPSILON;
    let consider_zero = f32::MIN_POSITIVE;
    // cwiseAbs().maxCoeff<PropagateNaN>()
    let mut scale = 0f32;
    let mut first = true;
    for r in a {
        for &x in r {
            let v = x.abs();
            if v.is_nan() {
                return None;
            }
            if first || v > scale {
                scale = v;
                first = false;
            }
        }
    }
    if !scale.is_finite() {
        return None;
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let mut w = *a;
    for r in w.iter_mut() {
        for x in r.iter_mut() {
            *x /= scale;
        }
    }
    let mut id = [[0f32; N]; N];
    for (i, r) in id.iter_mut().enumerate() {
        r[i] = 1.0;
    }
    let mut u = id;
    let mut v = id;
    let mut max_diag = w[0][0].abs();
    for i in 1..N {
        // maxCoeff: 더 큰 것만 바꿈
        if w[i][i].abs() > max_diag {
            max_diag = w[i][i].abs();
        }
    }
    let mut finished = false;
    while !finished {
        finished = true;
        for p in 1..N {
            for q in 0..p {
                let threshold = maxi(consider_zero, precision * max_diag);
                if w[p][q].abs() > threshold || w[q][p].abs() > threshold {
                    finished = false;
                    let (jl, jr) = real_2x2(&w, p, q);
                    apply_left(&mut w, p, q, jl);
                    if want_u {
                        apply_right(&mut u, p, q, jl.transpose());
                    }
                    apply_right(&mut w, p, q, jr);
                    apply_right(&mut v, p, q, jr);
                    max_diag = maxi(max_diag, maxi(w[p][p].abs(), w[q][q].abs()));
                }
            }
        }
    }
    let mut sv = [0f32; N];
    for i in 0..N {
        let a = w[i][i];
        sv[i] = a.abs();
        if want_u && a < 0.0 {
            for r in u.iter_mut() {
                r[i] = -r[i];
            }
        }
    }
    for s in sv.iter_mut() {
        *s *= scale;
    }
    let mut nonzero = N;
    for i in 0..N {
        // tail(N−i).maxCoeff(&pos): 첫 최댓값 자리
        let mut pos = 0;
        let mut m = sv[i];
        for k in 1..N - i {
            if sv[i + k] > m {
                m = sv[i + k];
                pos = k;
            }
        }
        if m == 0.0 {
            nonzero = i;
            break;
        }
        if pos != 0 {
            let pos = pos + i;
            sv.swap(i, pos);
            if want_u {
                for r in u.iter_mut() {
                    r.swap(i, pos);
                }
            }
            for r in v.iter_mut() {
                r.swap(i, pos);
            }
        }
    }
    Some(Svd { sv, v, u: if want_u { Some(u) } else { None }, nonzero })
}

/// GeometricTools::Triangulate: 정규화 좌표 x_c1 · x_c2(앞 두 칸)와 3×4 Tc1w · Tc2w 로 3D 점. x3Dh(3)==0 이면 None.
/// (NaN 입력은 원본에서 SVD 가 InvalidInput 이라 V 가 단위 행렬로 남는다: 그 경우도 같게)
pub fn triangulate(x1: [f32; 3], x2: [f32; 3], t1: &[[f32; 4]; 3], t2: &[[f32; 4]; 3]) -> Option<[f32; 3]> {
    let mut a = [[0f32; 4]; 4];
    for j in 0..4 {
        a[0][j] = x1[0] * t1[2][j] - t1[0][j];
        a[1][j] = x1[1] * t1[2][j] - t1[1][j];
        a[2][j] = x2[0] * t2[2][j] - t2[0][j];
        a[3][j] = x2[1] * t2[2][j] - t2[1][j];
    }
    let v = match jacobi_svd(&a, false) {
        Some(s) => s.v,
        None => {
            // InvalidInput: m_matrixV 는 할당만 된 채(allocate) — 원본도 정의되지 않은 값이라 같게 맞출 수 없음
            return None;
        }
    };
    let h = [v[0][3], v[1][3], v[2][3], v[3][3]];
    if h[3] == 0.0 {
        return None;
    }
    Some([h[0] / h[3], h[1] / h[3], h[2] / h[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diag_sorted() {
        let a: Mat<3> = [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
        let s = jacobi_svd(&a, true).unwrap();
        assert_eq!(s.sv, [3.0, 2.0, 1.0]);
        assert_eq!(s.v[1][0], 1.0);
    }
}
