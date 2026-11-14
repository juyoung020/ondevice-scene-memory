//! GTSAM 4.2 Pose3 · Rot3(3×3 행렬) · SO3 의 지수·로그 사상과 합성. 빌드 설정 GTSAM_POSE3_EXPMAP · GTSAM_ROT3_EXPMAP (사원수 아님).
//! 원본: gtsam/geometry/Pose3.cpp(Expmap · Logmap · AdjointMap · inverse · operator*), SO3.cpp(ExpmapFunctor · Logmap), Rot3M.cpp.
//! 덧셈 순서는 Eigen 3.4(SSE2, FMA 없음)가 실제로 내는 순서 그대로(probe 로 확인):
//!   3×3 행렬 × 행렬/벡터 = 0·1 행은 앞에서부터 차례로, 2 행은 a0 + (a1 + a2) · 3 벡터 내적 = 차례로 · 대각합 = a00 + (a11 + a22)

pub type M3 = [[f64; 3]; 3];
pub type V3 = [f64; 3];
pub type M6 = [[f64; 6]; 6];

/// 3×3 × 3×3 (Eigen Matrix3d 곱의 덧셈 순서)
#[inline]
pub fn mm(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for j in 0..3 {
        for i in 0..2 {
            c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j]) + a[i][2] * b[2][j];
        }
        c[2][j] = a[2][0] * b[0][j] + (a[2][1] * b[1][j] + a[2][2] * b[2][j]);
    }
    c
}

/// 3×3 × 3 벡터
#[inline]
pub fn mv(a: &M3, v: &V3) -> V3 {
    [
        (a[0][0] * v[0] + a[0][1] * v[1]) + a[0][2] * v[2],
        (a[1][0] * v[0] + a[1][1] * v[1]) + a[1][2] * v[2],
        a[2][0] * v[0] + (a[2][1] * v[1] + a[2][2] * v[2]),
    ]
}

#[inline]
pub fn dot3(a: &V3, b: &V3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

#[inline]
pub fn tr(a: &M3) -> M3 {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}

/// gtsam::skewSymmetric(x, y, z)
#[inline]
pub fn skew(x: f64, y: f64, z: f64) -> M3 {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// SO3::Expmap = so3::ExpmapFunctor(omega).expmap()
pub fn so3_expmap(w: &V3) -> M3 {
    let theta2 = dot3(w, w);
    let theta = theta2.sqrt();
    let wm = skew(w[0], w[1], w[2]);
    if theta2 <= f64::EPSILON {
        let mut r = I3;
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] += wm[i][j];
            }
        }
        return r;
    }
    let sin_theta = theta.sin();
    let s2 = (theta / 2.0).sin();
    let one_minus_cos = 2.0 * s2 * s2;
    let mut k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = wm[i][j] / theta;
        }
    }
    let kk = mm(&k, &k);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (I3[i][j] + sin_theta * k[i][j]) + one_minus_cos * kk[i][j];
        }
    }
    r
}

/// SO3::Logmap (Jacobian 없음)
pub fn so3_logmap(r: &M3) -> V3 {
    let (r11, r12, r13) = (r[0][0], r[0][1], r[0][2]);
    let (r21, r22, r23) = (r[1][0], r[1][1], r[1][2]);
    let (r31, r32, r33) = (r[2][0], r[2][1], r[2][2]);
    let tr = r11 + (r22 + r33);
    // 대각합 ≈ -1(각 ≈ ±π)
    let near_pi = |w: f64, q1: f64, q2: f64, q3: f64| -> f64 {
        let rr = q1.sqrt();
        let one_over_r = 1.0 / rr;
        let norm = (q1 * q1 + q2 * q2 + q3 * q3 + w * w).sqrt();
        let sgn_w = if w < 0.0 { -1.0 } else { 1.0 };
        let mag = std::f64::consts::PI - (2.0 * sgn_w * w) / norm;
        let scale = 0.5 * one_over_r * mag;
        sgn_w * scale
    };
    if tr + 1.0 < 1e-3 {
        if r33 > r22 && r33 > r11 {
            let w = r21 - r12;
            let q1 = 2.0 + 2.0 * r33;
            let q2 = r31 + r13;
            let q3 = r23 + r32;
            let s = near_pi(w, q1, q2, q3);
            [s * q2, s * q3, s * q1]
        } else if r22 > r11 {
            let w = r13 - r31;
            let q1 = 2.0 + 2.0 * r22;
            let q2 = r23 + r32;
            let q3 = r12 + r21;
            let s = near_pi(w, q1, q2, q3);
            [s * q3, s * q1, s * q2]
        } else {
            let w = r32 - r23;
            let q1 = 2.0 + 2.0 * r11;
            let q2 = r12 + r21;
            let q3 = r31 + r13;
            let s = near_pi(w, q1, q2, q3);
            [s * q1, s * q2, s * q3]
        }
    } else {
        let tr_3 = tr - 3.0;
        let magnitude = if tr_3 < -1e-6 {
            let theta = ((tr - 1.0) / 2.0).acos();
            theta / (2.0 * theta.sin())
        } else {
            0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0
        };
        [magnitude * (r32 - r23), magnitude * (r13 - r31), magnitude * (r21 - r12)]
    }
}

/// gtsam::Pose3 (R, t)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose3 {
    pub r: M3,
    pub t: V3,
}

impl Pose3 {
    /// Pose3(Transform::toEigen4d()): float 3×4(행 우선)를 double 로
    pub fn from_tf(d: &[f32; 12]) -> Pose3 {
        let f = |i: usize| d[i] as f64;
        Pose3 {
            r: [[f(0), f(1), f(2)], [f(4), f(5), f(6)], [f(8), f(9), f(10)]],
            t: [f(3), f(7), f(11)],
        }
    }

    /// Transform::fromEigen4d(p.matrix()): double → float
    pub fn to_tf(&self) -> [f32; 12] {
        let (r, t) = (&self.r, &self.t);
        [
            r[0][0] as f32, r[0][1] as f32, r[0][2] as f32, t[0] as f32,
            r[1][0] as f32, r[1][1] as f32, r[1][2] as f32, t[1] as f32,
            r[2][0] as f32, r[2][1] as f32, r[2][2] as f32, t[2] as f32,
        ]
    }

    /// Pose3::inverse: Rt = Rᵀ, (Rt, Rt·(-t))
    pub fn inverse(&self) -> Pose3 {
        let rt = tr(&self.r);
        let nt = [-self.t[0], -self.t[1], -self.t[2]];
        Pose3 { r: rt, t: mv(&rt, &nt) }
    }

    /// Pose3::operator*: (R·R2, t + R·t2)
    pub fn compose(&self, o: &Pose3) -> Pose3 {
        let rt = mv(&self.r, &o.t);
        Pose3 { r: mm(&self.r, &o.r), t: [self.t[0] + rt[0], self.t[1] + rt[1], self.t[2] + rt[2]] }
    }

    /// LieGroup::between = inverse() * g
    pub fn between(&self, o: &Pose3) -> Pose3 {
        self.inverse().compose(o)
    }

    /// Pose3::AdjointMap = [R 0; [t]×R R] (회전 먼저)
    pub fn adjoint(&self) -> M6 {
        let a = mm(&skew(self.t[0], self.t[1], self.t[2]), &self.r);
        let mut m = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = self.r[i][j];
                m[i + 3][j + 3] = self.r[i][j];
                m[i + 3][j] = a[i][j];
            }
        }
        m
    }

    /// Pose3::Expmap(xi) (xi = [ω, v])
    pub fn expmap(xi: &[f64; 6]) -> Pose3 {
        let w = [xi[0], xi[1], xi[2]];
        let v = [xi[3], xi[4], xi[5]];
        let r = so3_expmap(&w);
        let theta2 = dot3(&w, &w);
        if theta2 > f64::EPSILON {
            let wv = dot3(&w, &v);
            let tp = [w[0] * wv, w[1] * wv, w[2] * wv];
            let wxv = [w[1] * v[2] - w[2] * v[1], w[2] * v[0] - w[0] * v[2], w[0] * v[1] - w[1] * v[0]];
            let rw = mv(&r, &wxv);
            let mut t = [0.0; 3];
            for i in 0..3 {
                t[i] = ((wxv[i] - rw[i]) + tp[i]) / theta2;
            }
            Pose3 { r, t }
        } else {
            Pose3 { r, t: v }
        }
    }

    /// Pose3::Logmap
    pub fn logmap(p: &Pose3) -> [f64; 6] {
        let w = so3_logmap(&p.r);
        let tt = p.t;
        let t = dot3(&w, &w).sqrt();
        if t < 1e-10 {
            return [w[0], w[1], w[2], tt[0], tt[1], tt[2]];
        }
        let wm = skew(w[0] / t, w[1] / t, w[2] / t);
        let tan = (0.5 * t).tan();
        let wt = mv(&wm, &tt);
        let wwt = mv(&wm, &wt);
        let h = 0.5 * t;
        let c = 1.0 - t / (2.0 * tan);
        let mut u = [0.0; 3];
        for i in 0..3 {
            u[i] = (tt[i] - h * wt[i]) + c * wwt[i];
        }
        [w[0], w[1], w[2], u[0], u[1], u[2]]
    }

    /// LieGroup::retract = compose(Expmap(v))
    pub fn retract(&self, v: &[f64; 6]) -> Pose3 {
        self.compose(&Pose3::expmap(v))
    }

    /// LieGroup::localCoordinates = Logmap(between(g))
    pub fn local(&self, g: &Pose3) -> [f64; 6] {
        Pose3::logmap(&self.between(g))
    }
}
