//! g2o::SE3Quat · Eigen::Quaterniond 연산을 기준 빌드(-ffp-contract=off, Eigen 3.4.0)의 계산 순서 그대로 옮긴 것.
//! `fma(a, b, c)` = a·b + c 를 곱 · 덧셈 따로 반올림(이름은 식 모양 표시). Eigen 이 명시적으로 FMA 명령을 쓰는 자리는 `fused`.
//! 사원수 저장은 Eigen 과 같은 [x, y, z, w].

#[inline(always)]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

/// 한 번만 반올림하는 a·b + c (Eigen pmadd 가 vfmadd 내장 함수를 직접 쓰는 자리)
#[inline(always)]
pub fn fused(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

pub type Q = [f64; 4];
pub type V3 = [f64; 3];
/// 3×3 열 우선(Eigen 저장 순서): m[c*3 + r]
pub type M3 = [f64; 9];

/// QuaternionBase::_transformVector (q · v · q⁻¹). 원본(공유 라이브러리 함수 · 인라인 모두 같은 순서):
/// uv = 2·(q.vec × v), 외적 0·1 칸 = r(앞 곱) − 뒤 곱 을 한 번에(fnmadd), 2 칸 = 앞 곱 − r(뒤 곱)(fmsub);
/// d = q.vec × uv 같은 방식; 결과 = fma(uv, w, v) + d
#[inline]
pub fn rotate(q: &Q, v: &V3) -> V3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let c0 = fma(-z, v[1], y * v[2]);
    let c1 = fma(-x, v[2], z * v[0]);
    let c2 = fma(x, v[1], -(y * v[0]));
    let uv = [c0 + c0, c1 + c1, c2 + c2];
    let d0 = fma(-z, uv[1], y * uv[2]);
    let d1 = fma(-x, uv[2], z * uv[0]);
    let d2 = fma(x, uv[1], -(y * uv[0]));
    [fma(uv[0], w, v[0]) + d0, fma(uv[1], w, v[1]) + d1, fma(uv[2], w, v[2]) + d2]
}

/// SE3Quat::map: q·p + t
#[inline]
pub fn map(q: &Q, t: &V3, p: &V3) -> V3 {
    let r = rotate(q, p);
    [r[0] + t[0], r[1] + t[1], r[2] + t[2]]
}

/// QuaternionBase::toRotationMatrix (원본 인라인 자리의 축약: 대각은 1 − (r(·)+r(·)) 또는 1 − fma, 비대각은 fma(±t·, w, r(곱)))
#[inline]
pub fn to_rot(q: &Q) -> M3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let tx = x + x;
    let ty = y + y;
    let tz = z + z;
    let tyy = y * ty;
    let tzz = z * tz;
    let txy = x * ty;
    let txz = x * tz;
    let tyz = y * tz;
    let r00 = 1.0 - (tyy + tzz);
    let r11 = 1.0 - fma(x, tx, tzz);
    let r22 = 1.0 - fma(x, tx, tyy);
    let r01 = fma(-tz, w, txy);
    let r10 = fma(tz, w, txy);
    let r02 = fma(ty, w, txz);
    let r20 = fma(-ty, w, txz);
    let r12 = fma(-tx, w, tyz);
    let r21 = fma(tx, w, tyz);
    [r00, r10, r20, r01, r11, r21, r02, r12, r22]
}

/// 사원수 곱 a·b (Eigen SSE quat_product 를 GCC 가 축약한 꼴: addsub 두 쌍)
#[inline]
pub fn qmul(a: &Q, b: &Q) -> Q {
    let x = fma(b[2], a[1], b[0] * a[3]) - fma(-b[3], a[0], b[1] * a[2]);
    let y = fma(b[3], a[1], b[1] * a[3]) + fma(-b[2], a[0], b[0] * a[2]);
    let w = fma(-a[1], b[1], b[3] * a[3]) - fma(b[0], a[0], b[2] * a[2]);
    let z = fma(-a[1], b[0], b[2] * a[3]) + fma(b[1], a[0], b[3] * a[2]);
    [x, y, z, w]
}

/// SE3Quat::normalizeRotation: w < 0 이면 부호를 뒤집고 normalize(제곱합 = (x²+z²)+(y²+w²), 0 보다 크면 √ 로 나눔)
#[inline]
pub fn normalize_rot(q: &mut Q) {
    if 0.0 > q[3] {
        for v in q.iter_mut() {
            *v = -*v;
        }
    }
    let n = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    if n > 0.0 {
        let s = n.sqrt();
        for v in q.iter_mut() {
            *v /= s;
        }
    }
}

/// skew(ω) 열 우선: [0, w2, -w1, -w2, 0, w0, w1, -w0, 0]
#[inline]
fn skew(w: &V3) -> M3 {
    [0.0, w[2], -w[1], -w[2], 0.0, w[0], w[1], -w[0], 0.0]
}

/// 3×3 게으른 곱 a·b (Eigen 고정 크기): 0·1 행 = k 0→2 차례 fma, 2 행 = fma(a20,b0c, fma(a22,b2c, r(a21·b1c)))
#[inline]
pub fn mm3(a: &M3, b: &M3) -> M3 {
    let mut c = [0.0; 9];
    for j in 0..3 {
        for i in 0..2 {
            c[j * 3 + i] = fused(a[6 + i], b[j * 3 + 2], fused(a[3 + i], b[j * 3 + 1], a[i] * b[j * 3]));
        }
        c[j * 3 + 2] = fma(a[2], b[j * 3], fma(a[8], b[j * 3 + 2], a[5] * b[j * 3 + 1]));
    }
    c
}

/// 3×3 · 3 (0·1 행 차례 fma, 2 행 = fma(v0, a20, fma(v2, a22, r(v1·a21))))
#[inline]
pub fn mv3(a: &M3, v: &V3) -> V3 {
    [
        fused(v[2], a[6], fused(v[1], a[3], a[0] * v[0])),
        fused(v[2], a[7], fused(v[1], a[4], a[1] * v[0])),
        fma(v[0], a[2], fma(v[2], a[8], v[1] * a[5])),
    ]
}

/// Quaterniond(const Matrix3d&) (Eigen quaternionbase_assign_impl: 대각합 = (r22 + r11) + r00)
pub fn q_from_r(m: &M3) -> Q {
    let at = |i: usize, j: usize| m[j * 3 + i];
    let tr = (at(2, 2) + at(1, 1)) + at(0, 0);
    let mut q = [0.0; 4];
    if tr > 0.0 {
        let t = (tr + 1.0).sqrt();
        q[3] = 0.5 * t;
        let t = 0.5 / t;
        q[0] = (at(2, 1) - at(1, 2)) * t;
        q[1] = (at(0, 2) - at(2, 0)) * t;
        q[2] = (at(1, 0) - at(0, 1)) * t;
    } else {
        let mut i = 0;
        if at(1, 1) > at(0, 0) {
            i = 1;
        }
        if at(2, 2) > at(i, i) {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let t = ((at(i, i) - at(j, j)) - at(k, k) + 1.0).sqrt();
        q[i] = 0.5 * t;
        let t = 0.5 / t;
        q[3] = (at(k, j) - at(j, k)) * t;
        q[j] = (at(j, i) + at(i, j)) * t;
        q[k] = (at(k, i) + at(i, k)) * t;
    }
    q
}

/// SE3Quat::exp(update = [ω, υ]) → (q, t)
pub fn exp(u: &[f64; 6]) -> (Q, V3) {
    let w = [u[0], u[1], u[2]];
    let ups = [u[3], u[4], u[5]];
    let theta = fma(w[2], w[2], w[0] * w[0] + w[1] * w[1]).sqrt();
    let om = skew(&w);
    let om2 = mm3(&om, &om);
    let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let (r, v) = if theta < 0.00001 {
        let mut r = [0.0; 9];
        for k in 0..9 {
            r[k] = om2[k] + (id[k] + om[k]);
        }
        (r, r)
    } else {
        let (s, c) = theta.sin_cos();
        let a = s / theta;
        let b = (1.0 - c) / (theta * theta);
        let c3 = (theta - s) / theta.powf(3.0);
        let mut r = [0.0; 9];
        let mut v = [0.0; 9];
        for k in 0..9 {
            r[k] = fma(b, om2[k], fma(a, om[k], id[k]));
            v[k] = fma(c3, om2[k], fma(b, om[k], id[k]));
        }
        (r, v)
    };
    let mut q = q_from_r(&r);
    normalize_rot(&mut q);
    (q, mv3(&v, &ups))
}

/// VertexSE3Expmap::oplusImpl: (q, t) ← exp(δ) · (q, t)
pub fn oplus_se3(q: &mut Q, t: &mut V3, d: &[f64; 6]) {
    let (eq, et) = exp(d);
    let rt = rotate(&eq, t);
    *t = [et[0] + rt[0], et[1] + rt[1], et[2] + rt[2]];
    let mut nq = qmul(&eq, q);
    normalize_rot(&mut nq);
    *q = nq;
}
