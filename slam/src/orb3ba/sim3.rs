//! g2o::Sim3(ORB-SLAM3 판 Thirdparty/g2o/g2o/types/sim3.h)를 기준 빌드의 계산 순서대로.
//! 사원수 [x, y, z, w] · 이동 · 척도. 곱 · exp 는 정규화하지 않는다(원본과 같음).
use super::se3::{mm3, mv3, q_from_r, qmul, rotate, to_rot, M3, Q, V3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim3 {
    pub q: Q,
    pub t: V3,
    pub s: f64,
}

const I3: M3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn skew(w: &V3) -> M3 {
    [0.0, w[2], -w[1], -w[2], 0.0, w[0], w[1], -w[0], 0.0]
}

fn delta_r(r: &M3) -> V3 {
    let a = |i: usize, j: usize| r[j * 3 + i];
    [a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1)]
}

/// 원소별 (a·X + b·Y) + c·I
fn comb(a: f64, x: &M3, b: f64, y: &M3, c: f64) -> M3 {
    std::array::from_fn(|k| (a * x[k] + b * y[k]) + c * I3[k])
}
/// 원소별 (I + a·X) + b·Y
fn rot_of(a: f64, x: &M3, b: f64, y: &M3) -> M3 {
    std::array::from_fn(|k| (I3[k] + a * x[k]) + b * y[k])
}

/// PartialPivLU<Matrix3d>(부분 피벗, 마지막 행 따로) 로 W·x = b 풀기
pub fn lu_solve3(w: &M3, b: &V3) -> V3 {
    let mut lu = *w;
    let at = |m: &M3, i: usize, j: usize| m[j * 3 + i];
    let mut tr = [0usize; 3];
    for k in 0..2 {
        // 가장 큰 |값| 의 행(같으면 앞)
        let mut best = k;
        let mut bv = at(&lu, k, k).abs();
        for i in k + 1..3 {
            let v = at(&lu, i, k).abs();
            if v > bv {
                bv = v;
                best = i;
            }
        }
        tr[k] = best;
        if bv != 0.0 {
            if best != k {
                for j in 0..3 {
                    lu.swap(j * 3 + k, j * 3 + best);
                }
            }
            let p = at(&lu, k, k);
            for i in k + 1..3 {
                lu[k * 3 + i] /= p;
            }
        }
        for j in k + 1..3 {
            let rj = at(&lu, k, j);
            for i in k + 1..3 {
                lu[j * 3 + i] -= rj * lu[k * 3 + i];
            }
        }
    }
    tr[2] = 2;
    // P·b: 치환을 차례로
    let mut x = *b;
    for k in 0..3 {
        x.swap(k, tr[k]);
    }
    // 단위 아래 삼각
    x[1] -= at(&lu, 1, 0) * x[0];
    x[2] -= at(&lu, 2, 0) * x[0] + at(&lu, 2, 1) * x[1];
    // 위 삼각
    x[2] /= at(&lu, 2, 2);
    x[1] -= at(&lu, 1, 2) * x[2];
    x[1] /= at(&lu, 1, 1);
    x[0] -= at(&lu, 0, 1) * x[1] + at(&lu, 0, 2) * x[2];
    x[0] /= at(&lu, 0, 0);
    x
}

impl Sim3 {
    pub fn identity() -> Sim3 {
        Sim3 { q: [0.0, 0.0, 0.0, 1.0], t: [0.0; 3], s: 1.0 }
    }

    /// Sim3(const Vector7d& update): [ω, υ, σ]
    pub fn exp(u: &[f64; 7]) -> Sim3 {
        let omega = [u[0], u[1], u[2]];
        let ups = [u[3], u[4], u[5]];
        let sigma = u[6];
        let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
        let om = skew(&omega);
        let s = sigma.exp();
        let om2 = mm3(&om, &om);
        let eps = 0.00001;
        let (a, b, c, r);
        if sigma.abs() < eps {
            c = 1.0;
            if theta < eps {
                a = 1.0 / 2.0;
                b = 1.0 / 6.0;
                r = rot_of(1.0, &om, 1.0, &mm3(&om, &om));
            } else {
                let theta2 = theta * theta;
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
                r = rot_of(theta.sin() / theta, &om, (1.0 - theta.cos()) / (theta * theta), &om2);
            }
        } else {
            c = (s - 1.0) / sigma;
            if theta < eps {
                let sigma2 = sigma * sigma;
                a = ((sigma - 1.0) * s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * s) / (sigma2 * sigma);
                r = rot_of(1.0, &om, 1.0, &om2);
            } else {
                r = rot_of(theta.sin() / theta, &om, (1.0 - theta.cos()) / (theta * theta), &om2);
                let aa = s * theta.sin();
                let bb = s * theta.cos();
                let theta2 = theta * theta;
                let sigma2 = sigma * sigma;
                let cc = theta2 + sigma2;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) * 1.0 / theta2;
            }
        }
        let q = q_from_r(&r);
        let w = comb(a, &om, b, &om2, c);
        Sim3 { q, t: mv3(&w, &ups), s }
    }

    pub fn log(&self) -> [f64; 7] {
        let s = self.s;
        let sigma = s.ln();
        let r = to_rot(&self.q);
        let d = 0.5 * (((r[0] + r[4]) + r[8]) - 1.0);
        let eps = 0.00001;
        let (omega, a, b, c);
        let scale = |k: f64, v: V3| [k * v[0], k * v[1], k * v[2]];
        if sigma.abs() < eps {
            c = 1.0;
            if d > 1.0 - eps {
                omega = scale(0.5, delta_r(&r));
                a = 1.0 / 2.0;
                b = 1.0 / 6.0;
            } else {
                let theta = d.acos();
                let theta2 = theta * theta;
                omega = scale(theta / (2.0 * (1.0 - d * d).sqrt()), delta_r(&r));
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
            }
        } else {
            c = (s - 1.0) / sigma;
            if d > 1.0 - eps {
                let sigma2 = sigma * sigma;
                omega = scale(0.5, delta_r(&r));
                a = ((sigma - 1.0) * s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * s) / (sigma2 * sigma);
            } else {
                let theta = d.acos();
                omega = scale(theta / (2.0 * (1.0 - d * d).sqrt()), delta_r(&r));
                let theta2 = theta * theta;
                let aa = s * theta.sin();
                let bb = s * theta.cos();
                let cc = theta2 + sigma * sigma;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) * 1.0 / theta2;
            }
        }
        let om = skew(&omega);
        let bom: M3 = std::array::from_fn(|k| b * om[k]);
        let bom2 = mm3(&bom, &om);
        let w: M3 = std::array::from_fn(|k| (a * om[k] + bom2[k]) + c * I3[k]);
        let ups = lu_solve3(&w, &self.t);
        [omega[0], omega[1], omega[2], ups[0], ups[1], ups[2], sigma]
    }

    pub fn inverse(&self) -> Sim3 {
        let qc = [-self.q[0], -self.q[1], -self.q[2], self.q[3]];
        let k = -1.0 / self.s;
        let mt = [k * self.t[0], k * self.t[1], k * self.t[2]];
        Sim3 { q: qc, t: rotate(&qc, &mt), s: 1.0 / self.s }
    }

    pub fn mul(&self, o: &Sim3) -> Sim3 {
        let rt = rotate(&self.q, &o.t);
        Sim3 {
            q: qmul(&self.q, &o.q),
            t: [self.s * rt[0] + self.t[0], self.s * rt[1] + self.t[1], self.s * rt[2] + self.t[2]],
            s: self.s * o.s,
        }
    }

    pub fn map(&self, p: &V3) -> V3 {
        let r = rotate(&self.q, p);
        [self.s * r[0] + self.t[0], self.s * r[1] + self.t[1], self.s * r[2] + self.t[2]]
    }
}

