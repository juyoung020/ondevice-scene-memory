//! ORB-SLAM3 `Optimizer::PoseOptimization`(src/Optimizer.cc) — 한 프레임 자세만 g2o Levenberg 로 다듬고
//! 4 라운드마다 χ² 로 이상치를 다시 가른다. 단안(EdgeSE3ProjectXYZOnlyPose) · RGB-D/스테레오
//! (EdgeStereoSE3ProjectXYZOnlyPose) 관측. 두 번째 카메라(mpCamera2, 어안 스테레오)는 RGB-D 경로에 없어 뺐다.
//!
//! 원본과 비트 같음: 저장소 밖 하네스(`~/opt/orb3_opt_bg`)가 원본 실행 중 실제 호출의 입력 · 출력을 덤프하고
//! 같은 입력으로 이 함수 결과(자세 비트 · 이상치 표시 · 반환값)를 비교. 결과는 docs/orb3_port.md.

use super::g2o_pose::{SE3Quat, add_quadratic, huber, ldlt_solve6};
use super::pinhole::Pinhole;

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Sophus::SE3f 저장 그대로(사원수 x y z w, 이동)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3f {
    pub q: [f32; 4],
    pub t: [f32; 3],
}

/// 맵 점이 붙은 키포인트 하나(Frame 의 mvKeysUn · mvuRight · mvInvLevelSigma2[octave] · MapPoint::GetWorldPos)
#[derive(Clone, Copy, Debug)]
pub struct PoseObs {
    pub u: f32,
    pub v: f32,
    /// mvuRight; 음수면 단안 관측
    pub ur: f32,
    pub inv_sigma2: f32,
    pub xw: [f32; 3],
}

/// Frame 의 카메라 값: mpCamera(Pinhole) 와 스테레오 간선용 fx fy cx cy mbf
#[derive(Clone, Copy, Debug)]
pub struct PoseCam {
    pub cam: Pinhole,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub bf: f32,
}

struct Edge {
    stereo: bool,
    obs: [f64; 3],
    info: f64,
    xw: [f64; 3],
    err: [f64; 3],
    level: u8,
    robust: bool,
    delta: f64,
}

impl Edge {
    #[inline]
    fn compute_error(&mut self, est: &SE3Quat, pc: &PoseCam) {
        let p = est.map(&self.xw);
        if self.stereo {
            // EdgeStereoSE3ProjectXYZOnlyPose::cam_project: invz 는 float
            let invz = (1.0f64 / p[2]) as f32 as f64;
            let (fx, fy, cx, cy, bf) = (pc.fx as f64, pc.fy as f64, pc.cx as f64, pc.cy as f64, pc.bf as f64);
            let r0 = fma(p[0] * invz, fx, cx);
            let r1 = fma(p[1] * invz, fy, cy);
            let r2 = fma(-bf, invz, r0);
            self.err = [self.obs[0] - r0, self.obs[1] - r1, self.obs[2] - r2];
        } else {
            let r = pc.cam.project(&p);
            self.err = [self.obs[0] - r[0], self.obs[1] - r[1], 0.0];
        }
    }

    /// BaseEdge::chi2 = eᵀ(Ω e), Ω 대각
    #[inline]
    fn chi2(&self) -> f64 {
        let e = &self.err;
        if self.stereo {
            fma(e[2], self.info * e[2], e[0] * (self.info * e[0]) + e[1] * (self.info * e[1]))
        } else {
            e[0] * (self.info * e[0]) + e[1] * (self.info * e[1])
        }
    }

    #[inline]
    fn robust_chi2(&self) -> f64 {
        let c = self.chi2();
        if self.robust { huber(self.delta, c)[0] } else { c }
    }

    /// linearizeOplus + constructQuadraticForm
    #[inline]
    fn accumulate(&self, est: &SE3Quat, pc: &PoseCam, h: &mut [f64; 36], b: &mut [f64; 6]) {
        let p = est.map(&self.xw);
        let (x, y, z) = (p[0], p[1], p[2]);
        let rho1 = if self.robust { Some(huber(self.delta, self.chi2())[1]) } else { None };
        if self.stereo {
            let invz = 1.0 / z;
            let invz_2 = invz * invz;
            let (fx, fy, bf) = (pc.fx as f64, pc.fy as f64, pc.bf as f64);
            let j00 = x * y * invz_2 * fx;
            let j01 = -fma(x * x, invz_2, 1.0) * fx;
            let j02 = y * invz * fx;
            let j03 = -invz * fx;
            let j05 = x * invz_2 * fx;
            let j10 = fma(y * y, invz_2, 1.0) * fy;
            let j11 = -x * y * invz_2 * fy;
            let j12 = -x * invz * fy;
            let j14 = -invz * fy;
            let j15 = y * invz_2 * fy;
            let j20 = fma(-(bf * y), invz_2, j00);
            let j21 = fma(bf * x, invz_2, j01);
            let j25 = fma(-bf, invz_2, j05);
            // 3 x 6 열 우선
            let jac = [
                j00, j10, j20, j01, j11, j21, j02, j12, j02, j03, 0.0, j03, 0.0, j14, 0.0, j05, j15, j25,
            ];
            add_quadratic::<3>(&jac, self.info, &self.err, rho1, h, b);
        } else {
            // -projectJac(p) · SE3deriv, Eigen 지연 곱(열마다 행 묶음 FMA 사슬)
            let pj = pc.cam.project_jac(&p);
            let nj = [-pj[0], -pj[1], -pj[2], -pj[3], -pj[4], -pj[5]];
            let dd = [0.0, -z, y, z, 0.0, -x, -y, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let mut jac = [0.0; 12];
            for col in 0..6 {
                for row in 0..2 {
                    let mut s = nj[row] * dd[col * 3];
                    s = fma(nj[2 + row], dd[col * 3 + 1], s);
                    s = fma(nj[4 + row], dd[col * 3 + 2], s);
                    jac[col * 2 + row] = s;
                }
            }
            add_quadratic::<2>(&jac, self.info, &[self.err[0], self.err[1]], rho1, h, b);
        }
    }
}

/// g2o SparseOptimizer::optimize(its) + OptimizationAlgorithmLevenberg (한 정점, 활성 간선 = level 0)
struct Lm {
    x: [f64; 6],
}

impl Lm {
    fn optimize(&mut self, est: &mut SE3Quat, edges: &mut [Edge], pc: &PoseCam, its: i32) -> i32 {
        let active: Vec<usize> = (0..edges.len()).filter(|&i| edges[i].level == 0).collect();
        if active.is_empty() {
            // 활성 정점 없음: g2o 는 -1 을 돌려주고 아무것도 바꾸지 않음
            return -1;
        }
        let mut lambda = 0.0f64;
        let mut ni = 2.0f64;
        let mut nbad = 0;
        let mut iters = 0;
        for it in 0..its {
            for &i in &active {
                edges[i].compute_error(est, pc);
            }
            let mut current_chi = 0.0;
            for &i in &active {
                current_chi += edges[i].robust_chi2();
            }
            let ini_chi = current_chi;
            let mut h = [0.0f64; 36];
            let mut b = [0.0f64; 6];
            for &i in &active {
                edges[i].accumulate(est, pc, &mut h, &mut b);
            }
            if it == 0 {
                let mut maxd = 0.0f64;
                for j in 0..6 {
                    let v = h[j * 7].abs();
                    maxd = if v < maxd { maxd } else { v };
                }
                lambda = 1e-5 * maxd;
                ni = 2.0;
                nbad = 0;
            }
            let mut rho: f64;
            let mut qmax = 0;
            loop {
                let backup = *est;
                // Schur 경로: Hschur = 0 + Hpp, 대각에 λ
                let mut hs = [0.0f64; 36];
                for k in 0..36 {
                    hs[k] = 0.0 + h[k];
                }
                for j in 0..6 {
                    hs[j * 7] = h[j * 7] + lambda;
                    hs[j * 7] = 0.0 + hs[j * 7];
                }
                let ok2 = ldlt_solve6(&hs, &b, &mut self.x);
                *est = SE3Quat::exp(&self.x).mul(est);
                for &i in &active {
                    edges[i].compute_error(est, pc);
                }
                let mut temp_chi = 0.0;
                for &i in &active {
                    temp_chi += edges[i].robust_chi2();
                }
                if !ok2 {
                    temp_chi = f64::MAX;
                }
                rho = current_chi - temp_chi;
                let mut scale = 0.0f64;
                for j in 0..6 {
                    scale = fma(self.x[j], fma(lambda, self.x[j], b[j]), scale);
                }
                scale += 1e-3;
                rho /= scale;
                if rho > 0.0 && temp_chi.is_finite() {
                    let mut alpha = 1.0 - (2.0f64.mul_add(rho, -1.0)).powf(3.0);
                    alpha = if 2.0 / 3.0 < alpha { 2.0 / 3.0 } else { alpha };
                    let sf = if 1.0 / 3.0 < alpha { alpha } else { 1.0 / 3.0 };
                    lambda *= sf;
                    ni = 2.0;
                    current_chi = temp_chi;
                } else {
                    lambda *= ni;
                    ni *= 2.0;
                    *est = backup;
                }
                qmax += 1;
                if !(rho < 0.0 && qmax < 10) {
                    break;
                }
            }
            iters += 1;
            if qmax == 10 || rho == 0.0 {
                break;
            }
            if (ini_chi - current_chi) * 1e3 < ini_chi {
                nbad += 1;
            } else {
                nbad = 0;
            }
            if nbad >= 3 {
                break;
            }
        }
        iters
    }
}

/// 결과: (내점 수 = nInitialCorrespondences − nBad, 새 자세). `outlier` 는 obs 와 같은 길이(Frame::mvbOutlier 중 맵 점 칸).
/// 관측이 3 개 미만이면 원본처럼 0 과 들어온 자세를 돌려줌(이상치 표시는 false).
pub fn pose_optimization(tcw: &Se3f, pc: &PoseCam, obs: &[PoseObs], outlier: &mut [bool]) -> (i32, Se3f) {
    let ninit = obs.len() as i32;
    for o in outlier.iter_mut() {
        *o = false;
    }
    if ninit < 3 {
        return (0, *tcw);
    }
    let delta_mono = (5.991f32).sqrt() as f64;
    let delta_stereo = (7.815f32).sqrt() as f64;
    let mut edges: Vec<Edge> = obs
        .iter()
        .map(|o| {
            let stereo = o.ur >= 0.0;
            Edge {
                stereo,
                obs: [o.u as f64, o.v as f64, if stereo { o.ur as f64 } else { 0.0 }],
                info: o.inv_sigma2 as f64,
                xw: [o.xw[0] as f64, o.xw[1] as f64, o.xw[2] as f64],
                err: [0.0; 3],
                level: 0,
                robust: true,
                delta: if stereo { delta_stereo } else { delta_mono },
            }
        })
        .collect();
    // 단안 간선 먼저, 그다음 스테레오(원본의 이상치 판정 순회 순서; 값은 간선마다 독립이라 순서는 수에 영향 없음)
    let q0 = [tcw.q[0] as f64, tcw.q[1] as f64, tcw.q[2] as f64, tcw.q[3] as f64];
    let t0 = [tcw.t[0] as f64, tcw.t[1] as f64, tcw.t[2] as f64];
    let chi2_mono = 5.991f32;
    let chi2_stereo = 7.815f32;
    let mut lm = Lm { x: [0.0; 6] };
    let mut est = SE3Quat::new(q0, t0);
    let mut nbad = 0;
    for it in 0..4 {
        est = SE3Quat::new(q0, t0);
        lm.optimize(&mut est, &mut edges, pc, 10);
        nbad = 0;
        for (i, e) in edges.iter_mut().enumerate() {
            if outlier[i] {
                e.compute_error(&est, pc);
            }
            let chi2 = e.chi2() as f32;
            let th = if e.stereo { chi2_stereo } else { chi2_mono };
            if chi2 > th {
                outlier[i] = true;
                e.level = 1;
                nbad += 1;
            } else {
                outlier[i] = false;
                e.level = 0;
            }
            if it == 2 {
                e.robust = false;
            }
        }
        if edges.len() < 10 {
            break;
        }
    }
    // SE3Quat → Sophus::SE3f: 사원수를 float 로 내린 뒤 정규화
    let mut q = [est.r[0] as f32, est.r[1] as f32, est.r[2] as f32, est.r[3] as f32];
    let n = ((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])).sqrt();
    for e in &mut q {
        *e /= n;
    }
    (ninit - nbad, Se3f { q, t: [est.t[0] as f32, est.t[1] as f32, est.t[2] as f32] })
}
