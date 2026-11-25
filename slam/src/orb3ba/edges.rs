//! LBA 간선 세 종류의 오차 · χ² · 야코비안 · 2차 형식(헤세 블록 · b 누적)을 원본 g2o(ORB-SLAM3 판, 같은 빌드)의
//! 계산 순서대로(기준 빌드; 묶음 곱 자리만 `fused`). 행렬은 Eigen 과 같은 열 우선.
//! - 단안 ORB_SLAM3::EdgeSE3ProjectXYZ(D=2, Pinhole::project · projectJac 는 float 매개변수)
//! - 스테레오 g2o::EdgeStereoSE3ProjectXYZ(D=3, cam_project 의 1/z 는 float)
//! - 중력 EdgeGravityUp(D=3, 단항, 수치 미분 · 강건 커널 없음)
use super::se3::{fma, fused, map, rotate, to_rot, Q, V3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// pCamera->mvParameters[0..4] = fx fy cx cy (float)
    Mono { cam: [f32; 4] },
    Stereo { fx: f64, fy: f64, cx: f64, cy: f64, bf: f64 },
    Gravity,
}

/// 간선 하나. v0 = 점(단항이면 자세), v1 = 자세. info = D×D 열 우선. delta = Huber 문턱(없으면 강건 커널 없음)
#[derive(Clone, Debug)]
pub struct Edge {
    pub kind: Kind,
    pub v0: usize,
    pub v1: usize,
    pub meas: [f64; 3],
    pub info: [f64; 9],
    pub delta: Option<f64>,
}

impl Edge {
    pub fn dim(&self) -> usize {
        match self.kind {
            Kind::Mono { .. } => 2,
            _ => 3,
        }
    }
}

/// RobustKernelHuber::robustify (dsqr 는 float 칸). 돌려줌 (ρ0, ρ1)
#[inline]
pub fn huber(e: f64, delta: f64) -> (f64, f64) {
    let dsqr = (delta * delta) as f32 as f64;
    if e <= dsqr {
        (e, 1.0)
    } else {
        let s = e.sqrt();
        (fma(s + s, delta, -dsqr), delta / s)
    }
}

/// BaseEdge<D>::chi2() (가상 함수로 불리는 쪽: activeRobustChi2 · 이상치 판정)
#[inline]
pub fn chi2(d: usize, e: &[f64; 3], i: &[f64; 9]) -> f64 {
    if d == 2 {
        let y0 = fma(i[2], e[1], i[0] * e[0]);
        let y1 = fma(i[3], e[1], i[1] * e[0]);
        y0 * e[0] + y1 * e[1]
    } else {
        let y0 = fma(i[6], e[2], fma(i[3], e[1], i[0] * e[0]));
        let y1 = fma(i[7], e[2], fma(i[4], e[1], i[1] * e[0]));
        let y2 = fma(e[0], i[2], fma(e[2], i[8], e[1] * i[5]));
        fma(y2, e[2], y0 * e[0] + y1 * e[1])
    }
}

/// 오차. pose = (q, t), pt = 점(단항이면 쓰지 않음)
pub fn error(ed: &Edge, q: &Q, t: &V3, pt: &V3) -> [f64; 3] {
    let z = &ed.meas;
    match ed.kind {
        Kind::Mono { cam } => {
            let p = map(q, t, pt);
            let u = (cam[0] as f64 * p[0]) / p[2] + cam[2] as f64;
            let v = (cam[1] as f64 * p[1]) / p[2] + cam[3] as f64;
            [z[0] - u, z[1] - v, 0.0]
        }
        Kind::Stereo { fx, fy, cx, cy, bf } => {
            let p = map(q, t, pt);
            let invz_f = (1.0 / p[2]) as f32;
            let invz = invz_f as f64;
            let u = fma(p[0] * invz, fx, cx);
            let v = fma(p[1] * invz, fy, cy);
            let ur = u - (invz_f * bf as f32) as f64;
            [z[0] - u, z[1] - v, z[2] - ur]
        }
        Kind::Gravity => {
            let g = rotate(q, &[0.0, 0.0, 1.0]);
            [g[0] - z[0], g[1] - z[1], g[2] - z[2]]
        }
    }
}

/// isDepthPositive: 카메라 좌표 z > 0
pub fn depth_positive(q: &Q, t: &V3, pt: &V3) -> bool {
    map(q, t, pt)[2] > 0.0
}

/// 이항 간선의 해석 야코비안. xi = 점 쪽(D×3), xj = 자세 쪽(D×6), 열 우선(행 수 = D)
pub fn jacobian(ed: &Edge, q: &Q, t: &V3, pt: &V3, xi: &mut [f64; 9], xj: &mut [f64; 18]) {
    match ed.kind {
        Kind::Mono { cam } => {
            let p = map(q, t, pt);
            let (x, y, z) = (p[0], p[1], p[2]);
            // Pinhole::projectJac 를 부호 뒤집어(-projectJac)
            let zz = z * z;
            let p00 = cam[0] as f64 / z;
            let p11 = cam[1] as f64 / z;
            let p02 = ((-cam[0]) as f64 * x) / zz;
            let p12 = ((-cam[1]) as f64 * y) / zz;
            // nP 열: c0 = [-p00, -0], c1 = [-0, -p11], c2 = [-p02, -p12]
            let n = [[-p00, -0.0], [-0.0, -p11], [-p02, -p12]];
            let r = to_rot(q);
            for k in 0..3 {
                for row in 0..2 {
                    xi[k * 2 + row] = fused(r[k * 3 + 2], n[2][row], fused(r[k * 3 + 1], n[1][row], r[k * 3] * n[0][row]));
                }
            }
            // SE3deriv 3×6 (열 우선 d[k*3 + 행])
            let d = [0.0, -z, y, z, 0.0, -x, -y, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            for k in 0..6 {
                for row in 0..2 {
                    xj[k * 2 + row] = fused(n[2][row], d[k * 3 + 2], fused(n[1][row], d[k * 3 + 1], n[0][row] * d[k * 3]));
                }
            }
        }
        Kind::Stereo { fx, fy, bf, .. } => {
            let p = map(q, t, pt);
            let r = to_rot(q);
            let rr = |i: usize, j: usize| r[j * 3 + i];
            let (x, y, z) = (p[0], p[1], p[2]);
            let z2 = z * z;
            let mut a = [0.0; 9];
            for j in 0..3 {
                a[j * 3] = (-fx * rr(0, j)) / z + (fx * x * rr(2, j)) / z2;
                a[j * 3 + 1] = (-fy * rr(1, j)) / z + (fy * y * rr(2, j)) / z2;
                a[j * 3 + 2] = a[j * 3] - (bf * rr(2, j)) / z2;
            }
            *xi = a;
            let mut b = [0.0; 18];
            b[0] = x * y / z2 * fx;
            b[3] = -(1.0 + x * x / z2) * fx;
            b[6] = y / z * fx;
            b[9] = -1.0 / z * fx;
            b[12] = 0.0;
            b[15] = x / z2 * fx;
            b[1] = (1.0 + y * y / z2) * fy;
            b[4] = -(x * y / z2) * fy;
            b[7] = -x / z * fy;
            b[10] = 0.0;
            b[13] = -1.0 / z * fy;
            b[16] = y / z2 * fy;
            b[2] = b[0] - bf * y / z2;
            b[5] = b[3] + bf * x / z2;
            b[8] = b[6];
            b[11] = b[9];
            b[14] = 0.0;
            b[17] = b[15] - bf / z2;
            *xj = b;
        }
        Kind::Gravity => unreachable!("중력 간선은 수치 미분(numeric_jacobian)"),
    }
}

#[inline]
fn dot3t(a: &[f64], b: &[f64]) -> f64 {
    fma(a[2], b[2], a[0] * b[0] + a[1] * b[1])
}
#[inline]
fn dot2(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// 한 간선의 2차 형식을 정점 블록에 더함(BaseBinaryEdge::constructQuadraticForm, 강건 커널 경로).
/// pt_h(3×3) · pt_b(3) = 점, ps_h(6×6) · ps_b(6) = 자세(None = 고정), hpl(6×3, 자세 행 × 점 열) = 섞인 블록
pub fn quad_binary(
    ed: &Edge,
    e: &[f64; 3],
    xi: &[f64; 9],
    xj: &[f64; 18],
    pt_h: &mut [f64; 9],
    pt_b: &mut [f64; 3],
    pose: Option<(&mut [f64; 36], &mut [f64; 6], &mut [f64; 18])>,
) {
    let om = &ed.info;
    let delta = ed.delta.expect("LBA 이항 간선은 Huber");
    if ed.dim() == 3 {
        // omega_r = -Ω·e (0·1 행 = 묶음 fma, 2 행 = 차례 합), 인라인 χ²
        let mut w = [0.0; 3];
        for r in 0..2 {
            w[r] = fma(-om[6 + r], e[2], fma(-om[3 + r], e[1], (-om[r]) * e[0]));
        }
        w[2] = ((-om[8]) * e[2] + (-om[5]) * e[1]) + (-om[2]) * e[0];
        let y0 = fma(om[6], e[2], fma(om[3], e[1], om[0] * e[0]));
        let y1 = fma(om[7], e[2], fma(om[4], e[1], om[1] * e[0]));
        let y2 = (om[8] * e[2] + om[5] * e[1]) + om[2] * e[0];
        let chi = fma(e[2], y2, y0 * e[0] + y1 * e[1]);
        let (_, rho1) = huber(chi, delta);
        let mut wo = [0.0; 9];
        for k in 0..9 {
            wo[k] = rho1 * om[k];
        }
        for v in w.iter_mut() {
            *v *= rho1;
        }
        // 점: b += Aᵀω, H += (AᵀW)·A
        for r in 0..3 {
            pt_b[r] += dot3t(&xi[r * 3..r * 3 + 3], &w);
        }
        let mut t3 = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                t3[j * 3 + i] = dot3t(&xi[i * 3..i * 3 + 3], &wo[j * 3..j * 3 + 3]);
            }
        }
        for c in 0..3 {
            for r in 0..2 {
                pt_h[c * 3 + r] += fused(xi[c * 3 + 2], t3[6 + r], fused(xi[c * 3 + 1], t3[3 + r], xi[c * 3] * t3[r]));
            }
            pt_h[c * 3 + 2] += fma(t3[2], xi[c * 3], fma(t3[8], xi[c * 3 + 2], t3[5] * xi[c * 3 + 1]));
        }
        if let Some((ph, pb, hpl)) = pose {
            let mut t6 = [0.0; 18];
            for i in 0..6 {
                for j in 0..3 {
                    t6[j * 6 + i] = dot3t(&xj[i * 3..i * 3 + 3], &wo[j * 3..j * 3 + 3]);
                }
            }
            for c in 0..3 {
                for r in 0..6 {
                    hpl[c * 6 + r] += fused(xi[c * 3 + 2], t6[12 + r], fused(xi[c * 3 + 1], t6[6 + r], xi[c * 3] * t6[r]));
                }
            }
            for r in 0..6 {
                pb[r] += dot3t(&xj[r * 3..r * 3 + 3], &w);
            }
            for c in 0..6 {
                for r in 0..6 {
                    ph[c * 6 + r] += fused(xj[c * 3 + 2], t6[12 + r], fused(xj[c * 3 + 1], t6[6 + r], xj[c * 3] * t6[r]));
                }
            }
        }
    } else {
        let w0 = fma(-om[2], e[1], (-om[0]) * e[0]);
        let w1 = fma(-om[3], e[1], (-om[1]) * e[0]);
        let y0 = fma(om[2], e[1], om[0] * e[0]);
        let y1 = fma(om[3], e[1], om[1] * e[0]);
        let chi = y0 * e[0] + y1 * e[1];
        let (_, rho1) = huber(chi, delta);
        let wo = [rho1 * om[0], rho1 * om[1], rho1 * om[2], rho1 * om[3]];
        let w = [w0 * rho1, w1 * rho1];
        for r in 0..3 {
            pt_b[r] += dot2(&xi[r * 2..r * 2 + 2], &w);
        }
        let mut t3 = [0.0; 6];
        for i in 0..3 {
            for j in 0..2 {
                t3[j * 3 + i] = dot2(&xi[i * 2..i * 2 + 2], &wo[j * 2..j * 2 + 2]);
            }
        }
        for c in 0..3 {
            for r in 0..2 {
                pt_h[c * 3 + r] += fused(xi[c * 2 + 1], t3[3 + r], xi[c * 2] * t3[r]);
            }
            pt_h[c * 3 + 2] += xi[c * 2] * t3[2] + xi[c * 2 + 1] * t3[5];
        }
        if let Some((ph, pb, hpl)) = pose {
            let mut t6 = [0.0; 12];
            for i in 0..6 {
                for j in 0..2 {
                    t6[j * 6 + i] = dot2(&xj[i * 2..i * 2 + 2], &wo[j * 2..j * 2 + 2]);
                }
            }
            for c in 0..3 {
                for r in 0..6 {
                    hpl[c * 6 + r] += fused(xi[c * 2 + 1], t6[6 + r], xi[c * 2] * t6[r]);
                }
            }
            for r in 0..6 {
                pb[r] += dot2(&xj[r * 2..r * 2 + 2], &w);
            }
            for c in 0..6 {
                for r in 0..6 {
                    ph[c * 6 + r] += fused(xj[c * 2 + 1], t6[6 + r], xj[c * 2] * t6[r]);
                }
            }
        }
    }
}

/// 중력(단항, 강건 커널 없음): b -= (AᵀΩ)·e, H += (AᵀΩ)·A. a = 3×6 열 우선
pub fn quad_gravity(ed: &Edge, e: &[f64; 3], a: &[f64; 18], ph: &mut [f64; 36], pb: &mut [f64; 6]) {
    let om = &ed.info;
    let mut t6 = [0.0; 18];
    for i in 0..6 {
        for j in 0..3 {
            t6[j * 6 + i] = dot3t(&a[i * 3..i * 3 + 3], &om[j * 3..j * 3 + 3]);
        }
    }
    for r in 0..6 {
        pb[r] -= fused(e[2], t6[12 + r], fused(e[1], t6[6 + r], e[0] * t6[r]));
    }
    for c in 0..6 {
        for r in 0..6 {
            ph[c * 6 + r] += fused(a[c * 3 + 2], t6[12 + r], fused(a[c * 3 + 1], t6[6 + r], a[c * 3] * t6[r]));
        }
    }
}
