//! ORB-SLAM3 Sim3Solver(Horn 1987 닫힌 해 + RANSAC)를 원본 libORB_SLAM3 이 계산하는 순서 그대로 옮긴 것.
//! 원본: src/Sim3Solver.cc(SetRansacParameters · iterate(5 인자) · ComputeCentroid · ComputeSim3 · CheckInliers ·
//! Project · FromCameraToImage), CameraModels/Pinhole.cpp(project), Sophus SO3f::exp, Eigen EigenSolver<Matrix4f>(eig4).
//! FMA 로 합쳐진 자리는 GCC 13 -O3 -march=native GIMPLE 에서 읽은 그대로(eig4.rs 머리말).
//! 생성자(키프레임 · 지도 점 → 카메라 좌표)는 공통 자료 구조 몫이라 여기선 이미 만든 대응점 표를 받는다.

use super::crand::GlibcRand;
use super::eig4::eigen_solver4;

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}
#[inline(always)]
fn fnma(a: f32, b: f32, c: f32) -> f32 {
    (-a).mul_add(b, c)
}

/// 핀홀(Pinhole::project): u = fx*x/z + cx
#[derive(Clone, Copy, Debug)]
pub struct Pinhole {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}
impl Pinhole {
    #[inline]
    pub fn project(&self, p: [f32; 3]) -> [f32; 2] {
        [self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy]
    }
}

/// 3x3 · 4x4 행렬은 열 우선
pub type M3 = [f32; 9];
pub type M4 = [f32; 16];

#[derive(Clone, Debug)]
pub struct Sim3Solver {
    pub x3dc1: Vec<[f32; 3]>,
    pub x3dc2: Vec<[f32; 3]>,
    pub p1im1: Vec<[f32; 2]>,
    pub p2im2: Vec<[f32; 2]>,
    /// 9.210*sigma² 를 size_t 로 자른 값
    pub max_err1: Vec<u64>,
    pub max_err2: Vec<u64>,
    /// 대응 i → 원래 짝 번호(vpMatched12 색인)
    pub indices1: Vec<usize>,
    pub all_indices: Vec<usize>,
    pub n1: usize,
    pub cam1: Pinhole,
    pub cam2: Pinhole,
    pub fix_scale: bool,
    pub ransac_prob: f64,
    pub ransac_min_inliers: i32,
    pub ransac_max_its: i32,
    pub n_iterations: i32,
    pub best_inliers_n: i32,
    pub best_inliers: Vec<bool>,
    pub best_t12: M4,
    pub best_r: M3,
    pub best_t: [f32; 3],
    pub best_s: f32,
    /// 이번 반복 결과
    r12i: M3,
    t12i: [f32; 3],
    s12i: f32,
    t12m: M4,
    t21m: M4,
    inl_i: Vec<bool>,
    n_inl_i: i32,
}

/// iterate 결과
#[derive(Clone, Debug)]
pub struct IterOut {
    pub t: M4,
    pub no_more: bool,
    pub converge: bool,
    pub inliers: Vec<bool>,
    pub n_inliers: i32,
}

impl Sim3Solver {
    #[allow(clippy::too_many_arguments)]
    pub fn from_matches(
        x3dc1: Vec<[f32; 3]>,
        x3dc2: Vec<[f32; 3]>,
        max_err1: Vec<u64>,
        max_err2: Vec<u64>,
        indices1: Vec<usize>,
        n1: usize,
        cam1: Pinhole,
        cam2: Pinhole,
        fix_scale: bool,
    ) -> Self {
        let n = x3dc1.len();
        let p1im1 = x3dc1.iter().map(|&p| cam1.project(p)).collect();
        let p2im2 = x3dc2.iter().map(|&p| cam2.project(p)).collect();
        let mut s = Sim3Solver {
            x3dc1,
            x3dc2,
            p1im1,
            p2im2,
            max_err1,
            max_err2,
            indices1,
            all_indices: (0..n).collect(),
            n1,
            cam1,
            cam2,
            fix_scale,
            ransac_prob: 0.99,
            ransac_min_inliers: 6,
            ransac_max_its: 300,
            n_iterations: 0,
            best_inliers_n: 0,
            best_inliers: vec![],
            best_t12: [0.0; 16],
            best_r: [0.0; 9],
            best_t: [0.0; 3],
            best_s: 0.0,
            r12i: [0.0; 9],
            t12i: [0.0; 3],
            s12i: 0.0,
            t12m: [0.0; 16],
            t21m: [0.0; 16],
            inl_i: vec![false; n],
            n_inl_i: 0,
        };
        s.set_ransac_parameters(0.99, 6, 300);
        s
    }

    pub fn n(&self) -> usize {
        self.x3dc1.len()
    }

    /// SetRansacParameters
    pub fn set_ransac_parameters(&mut self, prob: f64, min_inliers: i32, max_its: i32) {
        self.ransac_prob = prob;
        self.ransac_min_inliers = min_inliers;
        let n = self.n() as i32;
        self.inl_i.resize(n as usize, false);
        let eps = min_inliers as f32 / n as f32;
        let nit: i32 = if min_inliers == n {
            1
        } else {
            ((1.0 - prob).ln() / (1.0 - (eps as f64).powf(3.0)).ln()).ceil() as i32
        };
        self.ransac_max_its = 1.max(nit.min(max_its));
        self.n_iterations = 0;
    }

    /// iterate(nIterations, bNoMore, vbInliers, nInliers, bConverge)
    pub fn iterate(&mut self, n_iterations: i32, rng: &mut GlibcRand) -> IterOut {
        self.iterate_with(n_iterations, &mut |lo, hi| rng.random_int(lo, hi))
    }

    /// 난수를 밖에서 주는 판(원본이 실제 쓴 rand 값으로 대조할 때)
    pub fn iterate_with(&mut self, n_iterations: i32, rand_int: &mut dyn FnMut(i32, i32) -> i32) -> IterOut {
        let mut out = IterOut { t: [0.0; 16], no_more: false, converge: false, inliers: vec![false; self.n1], n_inliers: 0 };
        let n = self.n() as i32;
        if n < self.ransac_min_inliers {
            out.no_more = true;
            out.t = ident4();
            return out;
        }
        let mut best_sim3: M4 = [f32::NAN; 16];
        let mut cur = 0;
        while self.n_iterations < self.ransac_max_its && cur < n_iterations {
            cur += 1;
            self.n_iterations += 1;
            let mut avail = self.all_indices.clone();
            let mut p1 = [0f32; 9];
            let mut p2 = [0f32; 9];
            for i in 0..3 {
                let randi = rand_int(0, avail.len() as i32 - 1) as usize;
                let idx = avail[randi];
                p1[3 * i..3 * i + 3].copy_from_slice(&self.x3dc1[idx]);
                p2[3 * i..3 * i + 3].copy_from_slice(&self.x3dc2[idx]);
                avail[randi] = *avail.last().unwrap();
                avail.pop();
            }
            self.compute_sim3(&p1, &p2);
            self.check_inliers();
            if self.n_inl_i >= self.best_inliers_n {
                self.best_inliers = self.inl_i.clone();
                self.best_inliers_n = self.n_inl_i;
                self.best_t12 = self.t12m;
                self.best_r = self.r12i;
                self.best_t = self.t12i;
                self.best_s = self.s12i;
                if self.n_inl_i > self.ransac_min_inliers {
                    out.n_inliers = self.n_inl_i;
                    for i in 0..n as usize {
                        if self.inl_i[i] {
                            out.inliers[self.indices1[i]] = true;
                        }
                    }
                    out.converge = true;
                    out.t = self.best_t12;
                    return out;
                } else {
                    best_sim3 = self.best_t12;
                }
            }
        }
        if self.n_iterations >= self.ransac_max_its {
            out.no_more = true;
        }
        // 원본은 이번 호출에서 나아지지 않으면 초기화 안 된 행렬을 돌려줌(쓰이지 않음)
        out.t = best_sim3;
        out
    }

    /// ComputeSim3(P1, P2) — 결과는 반복 상태(r12i, t12i, s12i, T12, T21)
    pub fn compute_sim3(&mut self, p1: &M3, p2: &M3) {
        let r = compute_sim3(p1, p2, self.fix_scale);
        self.r12i = r.r;
        self.t12i = r.t;
        self.s12i = r.s;
        self.t12m = r.t12;
        self.t21m = r.t21;
    }

    fn check_inliers(&mut self) {
        let n = self.n();
        self.n_inl_i = 0;
        for i in 0..n {
            let p2im1 = project_t(&self.t12m, self.x3dc2[i], &self.cam1);
            let p1im2 = project_t(&self.t21m, self.x3dc1[i], &self.cam2);
            let d1 = [self.p1im1[i][0] - p2im1[0], self.p1im1[i][1] - p2im1[1]];
            let d2 = [p1im2[0] - self.p2im2[i][0], p1im2[1] - self.p2im2[i][1]];
            let e1 = fma(d1[1], d1[1], d1[0] * d1[0]);
            let e2 = fma(d2[0], d2[0], d2[1] * d2[1]);
            if e1 < self.max_err1[i] as f32 && e2 < self.max_err2[i] as f32 {
                self.inl_i[i] = true;
                self.n_inl_i += 1;
            } else {
                self.inl_i[i] = false;
            }
        }
    }
}

fn ident4() -> M4 {
    let mut m = [0f32; 16];
    for i in 0..4 {
        m[i * 5] = 1.0;
    }
    m
}

/// Project: Rcw*P + tcw (행마다 k=1 곱, k=2 FMA, k=0 FMA, + t) 후 핀홀
#[inline]
fn project_t(t: &M4, p: [f32; 3], cam: &Pinhole) -> [f32; 2] {
    let mut c = [0f32; 3];
    for r in 0..3 {
        let a = t[r + 4] * p[1];
        let a = fma(t[r + 8], p[2], a);
        let a = fma(t[r], p[0], a);
        c[r] = a + t[r + 12];
    }
    cam.project(c)
}

pub struct Sim3Result {
    pub r: M3,
    pub t: [f32; 3],
    pub s: f32,
    pub t12: M4,
    pub t21: M4,
}

/// ComputeCentroid: C = (c0 + (c1 + c2)) / 3, Pr = P - C
fn centroid(p: &M3) -> (M3, [f32; 3]) {
    let mut c = [0f32; 3];
    for r in 0..3 {
        c[r] = ((p[r + 6] + p[r + 3]) + p[r]) / 3.0;
    }
    let mut pr = [0f32; 9];
    for k in 0..3 {
        for r in 0..3 {
            pr[r + 3 * k] = p[r + 3 * k] - c[r];
        }
    }
    (pr, c)
}

pub fn compute_sim3(p1: &M3, p2: &M3, fix_scale: bool) -> Sim3Result {
    let (pr1, o1) = centroid(p1);
    let (pr2, o2) = centroid(p2);
    let n = n_matrix(p1, p2);
    let es = eigen_solver4(&n);
    // maxCoeff: 같은 값이면 앞 번호
    let ev = es.eval_re;
    let mut mi = if ev[0] < ev[1] { 1 } else { 0 };
    let mut mv = ev[mi];
    for k in 2..4 {
        if ev[k] > mv {
            mv = ev[k];
            mi = k;
        }
    }
    let e = &es.evec;
    let v = [e[1 + 4 * mi], e[2 + 4 * mi], e[3 + 4 * mi]];
    let e0 = e[4 * mi];
    let vn = fma(v[0], v[0], fma(v[2], v[2], v[1] * v[1])).sqrt();
    let ang2 = vn.atan2(e0) * 2.0;
    let w = [(ang2 * v[0]) / vn, (ang2 * v[1]) / vn, (ang2 * v[2]) / vn];
    // Sophus SO3f::exp
    let th2 = fma(w[0], w[0], fma(w[2], w[2], w[1] * w[1]));
    let (imag, real) = if th2 < 1e-5f32 * 1e-5f32 {
        let th4 = th2 * th2;
        let im = fma(th4, (1.0f64 / 3840.0) as f32, fnma(th2, (1.0f64 / 48.0) as f32, 0.5));
        let re = fma(th4, (1.0f64 / 384.0) as f32, fnma(th2, 0.125, 1.0));
        (im, re)
    } else {
        let th = th2.sqrt();
        let h = th * 0.5;
        (h.sin() / th, h.cos())
    };
    let (qx, qy, qz, qw) = (imag * w[0], imag * w[1], imag * w[2], real);
    let (tx, ty, tz) = (qx * 2.0, qy * 2.0, qz * 2.0);
    let txy = ty * qx;
    let txz = tz * qx;
    let tyy = ty * qy;
    let tyz = tz * qy;
    let tzz = tz * qz;
    let r00 = 1.0 - (tyy + tzz);
    let r01 = fnma(tz, qw, txy);
    let r02 = fma(ty, qw, txz);
    let r10 = fma(tz, qw, txy);
    let r11 = 1.0 - fma(tx, qx, tzz);
    let r12 = fnma(tx, qw, tyz);
    let r20 = fnma(ty, qw, txz);
    let r21 = fma(tx, qw, tyz);
    let r22 = 1.0 - fma(tx, qx, tyy);
    let rr: M3 = [r00, r10, r20, r01, r11, r21, r02, r12, r22];
    let rm = |r: usize, c: usize| rr[r + 3 * c];
    let s = if fix_scale {
        1.0f32
    } else {
        // P3 = R * Pr2 : 원소 0..7 은 k=2 곱 → k=1 FMA → k=0 FMA, 원소 8(2,2) 는 k=1 곱 → k=2 → k=0
        let mut p3 = [0f32; 9];
        for c in 0..3 {
            for r in 0..3 {
                let b = |k: usize| pr2[k + 3 * c];
                p3[r + 3 * c] = if r == 2 && c == 2 {
                    fma(rm(r, 0), b(0), fma(rm(r, 2), b(2), rm(r, 1) * b(1)))
                } else {
                    fma(rm(r, 0), b(0), fma(rm(r, 1), b(1), rm(r, 2) * b(2)))
                };
            }
        }
        // nom = (Pr1 .* P3).sum(), den = (P3 .* P3).sum() : 4 묶음 두 개 + 수평 합 + 마지막 FMA
        let mut a = [0f32; 4];
        let mut d = [0f32; 4];
        for i in 0..4 {
            a[i] = fma(p3[4 + i], pr1[4 + i], p3[i] * pr1[i]);
            d[i] = fma(p3[i], p3[i], p3[4 + i] * p3[4 + i]);
        }
        let nom = fma(pr1[8], p3[8], (a[0] + a[2]) + (a[1] + a[3]));
        let den = fma(p3[8], p3[8], (d[0] + d[2]) + (d[1] + d[3]));
        nom / den
    };
    let sr: M3 = if fix_scale { rr } else { std::array::from_fn(|i| s * rr[i]) };
    let srm = |r: usize, c: usize| sr[r + 3 * c];
    let mut t = [0f32; 3];
    for r in 0..3 {
        t[r] = o1[r] - fma(o2[0], srm(r, 0), fma(o2[2], srm(r, 2), o2[1] * srm(r, 1)));
    }
    let mut t12 = ident4();
    for c in 0..3 {
        for r in 0..3 {
            t12[r + 4 * c] = srm(r, c);
        }
        t12[c + 12] = t[c];
    }
    let inv = 1.0f32 / s;
    let sri = |r: usize, c: usize| inv * rm(c, r);
    let mut t21 = ident4();
    for c in 0..3 {
        for r in 0..3 {
            t21[r + 4 * c] = sri(r, c);
        }
    }
    for r in 0..2 {
        let a = (-sri(r, 2)) * t[2];
        let a = fnma(sri(r, 1), t[1], a);
        t21[r + 12] = fnma(sri(r, 0), t[0], a);
    }
    let a = (-sri(2, 1)) * t[1];
    let a = fnma(t[2], sri(2, 2), a);
    t21[2 + 12] = fnma(sri(2, 0), t[0], a);
    Sim3Result { r: rr, t, s, t12, t21 }
}

/// ComputeSim3 의 N 행렬(중심 빼기 · M = Pr2 Pr1^T · Horn N)
pub fn n_matrix(p1: &M3, p2: &M3) -> [f32; 16] {
    let (pr1, _) = centroid(p1);
    let (pr2, _) = centroid(p2);
    // M = Pr2 * Pr1^T : M(i,j) = fma(a0,b0, fma(a2,b2, a1*b1))
    let mut m = [0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let a = |k: usize| pr2[i + 3 * k];
            let b = |k: usize| pr1[j + 3 * k];
            m[i + 3 * j] = fma(b(0), a(0), fma(b(2), a(2), b(1) * a(1)));
        }
    }
    let mm = |r: usize, c: usize| m[r + 3 * c];
    let n11 = (mm(0, 0) + mm(1, 1)) + mm(2, 2);
    let n12 = mm(1, 2) - mm(2, 1);
    let n13 = mm(2, 0) - mm(0, 2);
    let n14 = mm(0, 1) - mm(1, 0);
    let n22 = (mm(0, 0) - mm(1, 1)) - mm(2, 2);
    let n23 = mm(0, 1) + mm(1, 0);
    let n24 = mm(2, 0) + mm(0, 2);
    let n33 = (mm(1, 1) - mm(0, 0)) - mm(2, 2);
    let n34 = mm(1, 2) + mm(2, 1);
    let n44 = (-mm(0, 0) - mm(1, 1)) + mm(2, 2);
    let n: [f32; 16] = [n11, n12, n13, n14, n12, n22, n23, n24, n13, n23, n33, n34, n14, n24, n34, n44];
    n
}

/// 생성자 입력: 짝 하나(vpMatched12[i1])의 원본 분기 결과
#[derive(Clone, Copy, Debug)]
pub enum PairIn {
    /// vpMatched12[i1] 이 NULL
    Unmatched,
    /// pKF1 의 i1 번 지도 점 없음
    NoMp1,
    /// 둘 중 하나 isBad
    Bad,
    /// GetIndexInKeyFrame 이 -1(pKF2 에 안 보임: 원본은 vpKeyFrameMatchedMP 를 무시하고 늘 pKF2 로 찾는다)
    NoIndex,
    /// 쓰는 짝: 두 지도 점 세계 좌표 · 두 키포인트 층
    Use { pos1: [f32; 3], pos2: [f32; 3], oct1: i32, oct2: i32 },
}

/// Rcw*X + tcw (생성자 자리, GIMPLE: 행마다 r1·p1 → fma(r2,p2) → fma(r0,p0) → + t)
#[inline]
pub fn rot_add(r: &M3, t: &[f32; 3], p: [f32; 3]) -> [f32; 3] {
    let mut c = [0f32; 3];
    for i in 0..3 {
        let a = r[i + 3] * p[1];
        let a = fma(p[2], r[i + 6], a);
        let a = fma(p[0], r[i], a);
        c[i] = t[i] + a;
    }
    c
}

impl Sim3Solver {
    /// Sim3Solver(pKF1, pKF2, vpMatched12, bFixScale, vpKeyFrameMatchedMP) 생성자.
    /// r/t = 키프레임 mRcw · mTcw.translation(), sigma2 = mvLevelSigma2, pairs = vpMatched12 순서.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r1: &M3,
        t1: &[f32; 3],
        r2: &M3,
        t2: &[f32; 3],
        sigma2_1: &[f32],
        sigma2_2: &[f32],
        pairs: &[PairIn],
        cam1: Pinhole,
        cam2: Pinhole,
        fix_scale: bool,
    ) -> Self {
        let (mut x1, mut x2, mut e1, mut e2, mut ix) = (vec![], vec![], vec![], vec![], vec![]);
        for (i1, p) in pairs.iter().enumerate() {
            if let PairIn::Use { pos1, pos2, oct1, oct2 } = *p {
                e1.push((9.210f64 * sigma2_1[oct1 as usize] as f64) as u64);
                e2.push((9.210f64 * sigma2_2[oct2 as usize] as f64) as u64);
                ix.push(i1);
                x1.push(rot_add(r1, t1, pos1));
                x2.push(rot_add(r2, t2, pos2));
            }
        }
        Self::from_matches(x1, x2, e1, e2, ix, pairs.len(), cam1, cam2, fix_scale)
    }
}
