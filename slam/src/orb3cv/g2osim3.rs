//! g2o::Sim3(Thirdparty/g2o/g2o/types/sim3.h, double) — 루프 닫기 · 지도 합치기 · 본질 그래프 · Sim3 최적화가 쓰는 7 자유도 변환.
//! 기준 빌드(-O3 -march=native -ffp-contract=off)와 비트 같게: 계산 순서는 Eigen 3.4 이 내는 순서 그대로, Eigen 이 pmadd 로
//! 묶는 자리(작은 행렬 곱 · 행렬 × 벡터)만 FMA. 하네스 `~/opt/orb3_cv/sim3`(probe.cc → sim3.bin, Rust `--bin sim3`).
//! 이 파일은 다른 모듈을 쓰지 않는다(하네스가 `#[path]` 로 그대로 끌어감).

/// Eigen Quaterniond 저장 순서 (x, y, z, w)
pub type Quatd = [f64; 4];
/// 3x3 열 우선(Eigen Matrix3d 저장 순서)
pub type M3d = [f64; 9];
pub type V3d = [f64; 3];

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[inline(always)]
fn at(m: &M3d, r: usize, c: usize) -> f64 {
    m[c * 3 + r]
}

/// Eigen cross(스칼라): (a.y b.z − a.z b.y, a.z b.x − a.x b.z, a.x b.y − a.y b.x)
#[inline]
pub fn cross(a: &V3d, b: &V3d) -> V3d {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Eigen SSE2 quat_product<double>(pmul · padd · paddsub)
#[inline]
pub fn quat_mul(a: &Quatd, b: &Quatd) -> Quatd {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    let t1x = aw * bx + ay * bz;
    let t1y = aw * by + ay * bw;
    let t2x = az * bx - ax * bz;
    let t2y = az * by - ax * bw;
    let u1z = aw * bz - ay * bx;
    let u1w = aw * bw - ay * by;
    let u2z = az * bz + ax * bx;
    let u2w = az * bw + ax * by;
    [t1x - t2y, t1y + t2x, u1z + u2w, u1w - u2z]
}

/// Quaternion::conjugate
#[inline]
pub fn quat_conj(q: &Quatd) -> Quatd {
    [-q[0], -q[1], -q[2], q[3]]
}

/// QuaternionBase::_transformVector: uv = 2(q.vec × v); (v + w·uv) + q.vec × uv
#[inline]
pub fn quat_rotate(q: &Quatd, v: &V3d) -> V3d {
    let qv = [q[0], q[1], q[2]];
    let mut uv = cross(&qv, v);
    for e in &mut uv {
        *e += *e;
    }
    let c = cross(&qv, &uv);
    [(v[0] + q[3] * uv[0]) + c[0], (v[1] + q[3] * uv[1]) + c[1], (v[2] + q[3] * uv[2]) + c[2]]
}

/// QuaternionBase::toRotationMatrix
pub fn quat_to_mat(q: &Quatd) -> M3d {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (tx, ty, tz) = (2.0 * x, 2.0 * y, 2.0 * z);
    let (twx, twy, twz) = (tx * w, ty * w, tz * w);
    let (txx, txy, txz) = (tx * x, ty * x, tz * x);
    let (tyy, tyz, tzz) = (ty * y, tz * y, tz * z);
    [
        1.0 - (tyy + tzz),
        txy + twz,
        txz - twy,
        txy - twz,
        1.0 - (txx + tzz),
        tyz + twx,
        txz + twy,
        tyz - twx,
        1.0 - (txx + tyy),
    ]
}

/// Quaterniond(Matrix3d) — Eigen quaternionbase_assign_impl(Shoemake)
pub fn quat_from_mat(m: &M3d) -> Quatd {
    let c = |r: usize, k: usize| at(m, r, k);
    let mut q = [0.0; 4];
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

/// g2o skew(v)
#[inline]
pub fn skew(v: &V3d) -> M3d {
    [0.0, v[2], -v[1], -v[2], 0.0, v[0], v[1], -v[0], 0.0]
}

/// g2o deltaR(R) = (R21 − R12, R02 − R20, R10 − R01)
#[inline]
pub fn delta_r(r: &M3d) -> V3d {
    [at(r, 2, 1) - at(r, 1, 2), at(r, 0, 2) - at(r, 2, 0), at(r, 1, 0) - at(r, 0, 1)]
}

/// Matrix3d × Matrix3d(Eigen 작은 고정 크기 지연 곱): 열마다 행 0 · 1 은 Packet2d pmadd(FMA) 사슬, 행 2 는 스칼라 (a21 b1 + a22 b2) + a20 b0
pub fn mat_mul(a: &M3d, b: &M3d) -> M3d {
    let mut r = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            let (b0, b1, b2) = (at(b, 0, j), at(b, 1, j), at(b, 2, j));
            r[j * 3 + i] = if i < 2 {
                fma(at(a, i, 2), b2, fma(at(a, i, 1), b1, at(a, i, 0) * b0))
            } else {
                (at(a, i, 1) * b1 + at(a, i, 2) * b2) + at(a, i, 0) * b0
            };
        }
    }
    r
}

/// Matrix3d × Vector3d: 행 0 · 1 은 Packet2d pmadd(FMA) 사슬(열 0 → 1 → 2), 행 2 는 스칼라 (a21 v1 + a22 v2) + a20 v0
pub fn mat_vec(a: &M3d, v: &V3d) -> V3d {
    let mut r = [0.0; 3];
    for i in 0..2 {
        r[i] = fma(at(a, i, 2), v[2], fma(at(a, i, 1), v[1], at(a, i, 0) * v[0]));
    }
    r[2] = (at(a, 2, 1) * v[1] + at(a, 2, 2) * v[2]) + at(a, 2, 0) * v[0];
    r
}

/// Vector3d::norm
#[inline]
pub fn norm3(v: &V3d) -> f64 {
    ((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]).sqrt()
}

/// W.lu().solve(t) — Eigen PartialPivLU(3x3, 부분 피벗) + 단위 아래 삼각 · 위 삼각 풀이
pub fn lu_solve(w: &M3d, b: &V3d) -> V3d {
    let mut m = *w;
    let mut perm = [0usize, 1, 2];
    for k in 0..3 {
        // 열 k 의 k.. 행에서 절댓값 최대(같으면 앞)
        let mut p = k;
        let mut best = at(&m, k, k).abs();
        for r in k + 1..3 {
            let v = at(&m, r, k).abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if p != k {
            for c in 0..3 {
                m.swap(c * 3 + k, c * 3 + p);
            }
            perm.swap(k, p);
        }
        if best != 0.0 {
            let piv = at(&m, k, k);
            for r in k + 1..3 {
                m[k * 3 + r] /= piv;
            }
        }
        for c in k + 1..3 {
            let u = at(&m, k, c);
            for r in k + 1..3 {
                m[c * 3 + r] -= at(&m, r, k) * u;
            }
        }
    }
    let mut x = [b[perm[0]], b[perm[1]], b[perm[2]]];
    // 풀이(Eigen 완전 펼침 triangular_solver_unroller): 행마다 x_i −= (앞서 푼 칸과의 내적), 위 삼각은 나눗셈까지
    x[1] -= at(&m, 1, 0) * x[0];
    x[2] -= at(&m, 2, 0) * x[0] + at(&m, 2, 1) * x[1];
    x[2] /= at(&m, 2, 2);
    x[1] -= at(&m, 1, 2) * x[2];
    x[1] /= at(&m, 1, 1);
    x[0] -= at(&m, 0, 1) * x[1] + at(&m, 0, 2) * x[2];
    x[0] /= at(&m, 0, 0);
    x
}

/// g2o::Sim3
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G2oSim3 {
    pub r: Quatd,
    pub t: V3d,
    pub s: f64,
}

impl Default for G2oSim3 {
    fn default() -> Self {
        G2oSim3 { r: [0.0, 0.0, 0.0, 1.0], t: [0.0; 3], s: 1.0 }
    }
}

impl G2oSim3 {
    /// Sim3(Quaterniond, Vector3d, double)
    pub fn new(r: Quatd, t: V3d, s: f64) -> Self {
        G2oSim3 { r, t, s }
    }

    /// Sim3(Matrix3d, Vector3d, double)
    pub fn from_mat(r: &M3d, t: V3d, s: f64) -> Self {
        G2oSim3 { r: quat_from_mat(r), t, s }
    }

    /// Sim3(Vector7d update) — (ω, υ, σ)
    pub fn exp(u: &[f64; 7]) -> Self {
        let (r, w, s) = Self::exp_parts(u);
        G2oSim3 { r, t: mat_vec(&w, &[u[3], u[4], u[5]]), s }
    }

    /// exp 의 (회전, W, s) — t = W·υ
    pub fn exp_parts(u: &[f64; 7]) -> (Quatd, M3d, f64) {
        let omega = [u[0], u[1], u[2]];
        let sigma = u[6];
        let theta = norm3(&omega);
        let om = skew(&omega);
        let s = sigma.exp();
        let om2 = mat_mul(&om, &om);
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eps = 0.00001;
        let (a, b, c);
        let mut r = [0.0; 9];
        let rot_small = |r: &mut M3d| {
            let o3 = mat_mul(&om, &om);
            for k in 0..9 {
                r[k] = (id[k] + om[k]) + o3[k];
            }
        };
        let rot_big = |r: &mut M3d| {
            let ka = theta.sin() / theta;
            let kb = (1.0 - theta.cos()) / (theta * theta);
            for k in 0..9 {
                r[k] = (id[k] + ka * om[k]) + kb * om2[k];
            }
        };
        if sigma.abs() < eps {
            c = 1.0;
            if theta < eps {
                a = 1.0 / 2.0;
                b = 1.0 / 6.0;
                rot_small(&mut r);
            } else {
                let theta2 = theta * theta;
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
                rot_big(&mut r);
            }
        } else {
            c = (s - 1.0) / sigma;
            if theta < eps {
                let sigma2 = sigma * sigma;
                a = ((sigma - 1.0) * s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * s) / (sigma2 * sigma);
                rot_small(&mut r);
            } else {
                rot_big(&mut r);
                let aa = s * theta.sin();
                let bb = s * theta.cos();
                let theta2 = theta * theta;
                let sigma2 = sigma * sigma;
                let cc = theta2 + sigma2;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) * 1.0 / theta2;
            }
        }
        let rq = quat_from_mat(&r);
        let mut w = [0.0; 9];
        for k in 0..9 {
            w[k] = (a * om[k] + b * om2[k]) + c * id[k];
        }
        (rq, w, s)
    }

    /// map(xyz) = s·(r·xyz) + t
    #[inline]
    pub fn map(&self, x: &V3d) -> V3d {
        let p = quat_rotate(&self.r, x);
        [self.s * p[0] + self.t[0], self.s * p[1] + self.t[1], self.s * p[2] + self.t[2]]
    }

    /// log() → (ω, υ, σ)
    pub fn log(&self) -> [f64; 7] {
        let s = self.s;
        let sigma = s.ln();
        let rm = quat_to_mat(&self.r);
        let d = 0.5 * (((at(&rm, 0, 0) + at(&rm, 1, 1)) + at(&rm, 2, 2)) - 1.0);
        let eps = 0.00001;
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let dr = delta_r(&rm);
        let half = |v: &V3d| [0.5 * v[0], 0.5 * v[1], 0.5 * v[2]];
        let (omega, a, b, c);
        if sigma.abs() < eps {
            c = 1.0;
            if d > 1.0 - eps {
                omega = half(&dr);
                a = 1.0 / 2.0;
                b = 1.0 / 6.0;
            } else {
                let theta = d.acos();
                let theta2 = theta * theta;
                let k = theta / (2.0 * (1.0 - d * d).sqrt());
                omega = [k * dr[0], k * dr[1], k * dr[2]];
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
            }
        } else {
            c = (s - 1.0) / sigma;
            if d > 1.0 - eps {
                let sigma2 = sigma * sigma;
                omega = half(&dr);
                a = ((sigma - 1.0) * s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * s) / (sigma2 * sigma);
            } else {
                let theta = d.acos();
                let k = theta / (2.0 * (1.0 - d * d).sqrt());
                omega = [k * dr[0], k * dr[1], k * dr[2]];
                let theta2 = theta * theta;
                let aa = s * theta.sin();
                let bb = s * theta.cos();
                let cc = theta2 + sigma * sigma;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) * 1.0 / theta2;
            }
        }
        let om = skew(&omega);
        // B*Omega*Omega = (B·Ω)·Ω
        let bo: M3d = std::array::from_fn(|k| b * om[k]);
        let bo2 = mat_mul(&bo, &om);
        let mut w = [0.0; 9];
        for k in 0..9 {
            w[k] = (a * om[k] + bo2[k]) + c * id[k];
        }
        let ups = lu_solve(&w, &self.t);
        [omega[0], omega[1], omega[2], ups[0], ups[1], ups[2], sigma]
    }

    /// inverse(): (r̄, r̄·((−1/s)·t), 1/s)
    pub fn inverse(&self) -> Self {
        let rc = quat_conj(&self.r);
        let k = -1.0 / self.s;
        let t = quat_rotate(&rc, &[k * self.t[0], k * self.t[1], k * self.t[2]]);
        G2oSim3 { r: rc, t, s: 1.0 / self.s }
    }

    /// operator*: r·r₂, s·(r·t₂) + t, s·s₂
    pub fn mul(&self, o: &Self) -> Self {
        let p = quat_rotate(&self.r, &o.t);
        G2oSim3 {
            r: quat_mul(&self.r, &o.r),
            t: [self.s * p[0] + self.t[0], self.s * p[1] + self.t[1], self.s * p[2] + self.t[2]],
            s: self.s * o.s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_inverse() {
        let u = [0.3, -0.2, 0.5, 1.0, -2.0, 0.5, 0.2];
        let a = G2oSim3::exp(&u);
        let l = a.log();
        assert!(u.iter().zip(&l).all(|(x, y)| (x - y).abs() < 1e-12));
        let i = a.mul(&a.inverse());
        assert!((i.s - 1.0).abs() < 1e-12 && i.t.iter().all(|x| x.abs() < 1e-12));
        let x = [0.5, 1.5, -2.0];
        let y = a.inverse().map(&a.map(&x));
        assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-12));
    }
}
