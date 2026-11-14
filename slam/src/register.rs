//! RTAB-Map 0.24.2 RegistrationVis(특징 짝짓기 + 3D→2D PnP + 두 장면 묶음 조정) 를 옮긴 것.
//! 원본: corelib/src/RegistrationVis.cpp(computeTransformationImpl), Registration.cpp(computeTransformationMod),
//!       util3d_motion_estimation.cpp(estimateMotion3DTo2D, solvePnPRansac), opencv/solvepnp.cpp(cv3::solvePnPRansac),
//!       VWDictionary.cpp(추측 없을 때의 kNN 짝짓기), OptimizerG2O.cpp(optimizeBA).
//! 우리 설정이 지나가는 경로만: Vis/CorType=0(특징 짝짓기), Vis/EstimationType=1(PnP), Vis/BundleAdjustment=1(g2o),
//! Vis/PnPRefineIterations=0, Vis/PnPFlags=0, Reg/RepeatOnce=true, 카메라 하나, Vis/MinDepth·MaxDepth=0.
//!
//! 짝짓기는 두 길이 있다.
//!  - 추측 있음: 기준 장면 3D 점을 추측 위치로 현재 화면에 투영 → 반경 Vis/CorGuessWinSize 안의 특징점 중
//!    해밍 거리 1·2 등 비율(Vis/CorNNDR) 로 고름. 후보가 하나면 거리와 상관없이 그것
//!  - 추측 없음: 현재 장면의 설명자마다 기준 장면에서 가장 가까운 두 개(정확한 해밍 kNN) → 1 등 <= 비율×2 등이면 짝.
//!    같은 기준 점을 둘 이상이 고르면 모두 버림. 성공하면 그 결과를 추측으로 한 번 더(Reg/RepeatOnce)
//!    원본은 여기서 rtflann 무작위 kd-나무(std::random_device 로 섞음)를 써서 실행마다 결과가 다를 수 있다.
//!    같은 입력에 늘 같은 결과가 나오도록 정확한 kNN 을 쓴다(무작위 나무가 찾으려는 답, Vis/CorNNType=3 과 같은 결과).
mod ba;
mod cvmath;
mod epnp;
mod pnp;
pub mod rtf;

use crate::loopclose::flann::{GlibcRand, KdTreeIndex};
use crate::params::Params;
use std::sync::{Arc, Mutex};

/// 프로세스 전체가 함께 쓰는 난수 상태(원본의 glibc rand() 와 std::random_device). 모듈들이 같은 것을 나눠 쓰면
/// 원본 한 프로세스와 같은 순서로 소비된다. rd_calls = 지금까지 부른 random_device 수(k 번째 값 = (k+1)*2654435761)
pub struct ProcessRng {
    pub rand: GlibcRand,
    pub rd_calls: u32,
}
pub type SharedRng = Arc<Mutex<ProcessRng>>;
/// srand(1)·random_device 0 회에서 시작하는 새 공유 상태
pub fn new_shared_rng() -> SharedRng {
    Arc::new(Mutex::new(ProcessRng { rand: GlibcRand::new(1), rd_calls: 0 }))
}
use crate::{Camera, Features, Pose};
use cvmath::K;
pub use rtf::Tf;
thread_local! { #[doc(hidden)] pub static PROF: std::cell::RefCell<[f64; 4]> = const { std::cell::RefCell::new([0.0; 4]) }; }
#[doc(hidden)]
pub use ba::TRACE as BA_TRACE;

/// 한 번의 맞추기 결과(RTAB-Map RegistrationInfo 에서 쓰는 칸)
#[derive(Clone, Debug)]
pub struct RegInfo {
    /// `to` 장면의 `from` 장면 기준 위치(몸체 축). 실패면 None
    pub transform: Option<Pose>,
    /// 같은 값의 float 3x4(RTAB-Map Transform 그대로, 비트 비교용)
    pub tf: Tf,
    pub inliers: usize,
    pub matches: usize,
    /// 6x6 공분산(x y z roll pitch yaw), 원본처럼 대각만 채움
    pub covariance: [[f64; 6]; 6],
    /// 안쪽 점 짝 (from 번호, to 번호), 기준 점 번호 순
    pub inlier_pairs: Vec<(usize, usize)>,
    /// 모든 짝(기준 3D 가 있는 것) (from 번호, to 번호)
    pub match_pairs: Vec<(usize, usize)>,
    /// `to` 장면에 붙은 단어 수(RTAB-Map toSignature.getWords().size()). 키프레임 조건에 쓰임
    pub to_words: usize,
    /// 맞추기 뒤 `to` 장면에 남은 특징점 번호. 추측 없이 시작해 한 번 더 맞춘 경우만 일부(Some).
    /// RTAB-Map 은 두 번째 맞추기에서 이 남은 특징점을 그 장면의 특징으로 바꿔 저장한다(키프레임이 되면 이것을 씀)
    pub to_kept: Option<Vec<usize>>,
    /// 두 번째 판(추측 있음)을 거친 경우 원본이 `from` Signature 에 남기는 단어 순서(원래 특징 번호): 추측 창에 이웃이 있던 투영 점(번호 순) 뒤에 나머지 유한 점(번호 순). 3D 없는 점은 빠짐
    pub from_order: Option<Vec<usize>>,
    pub rejected: &'static str,
}

impl RegInfo {
    fn empty() -> RegInfo {
        RegInfo {
            transform: None,
            tf: Tf::NULL,
            inliers: 0,
            matches: 0,
            covariance: eye6(),
            inlier_pairs: Vec::new(),
            match_pairs: Vec::new(),
            to_words: 0,
            to_kept: None,
            from_order: None,
            rejected: "",
        }
    }
}

fn eye6() -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        c[i][i] = 1.0;
    }
    c
}

/// Registration::COVARIANCE_LINEAR_EPSILON / ANGULAR_EPSILON
const COV_LIN_EPS: f64 = 0.00000001;
const COV_ANG_EPS: f64 = 0.00000003;

fn floor_cov(c: &mut [[f64; 6]; 6]) {
    for i in 0..3 {
        if c[i][i] <= COV_LIN_EPS {
            c[i][i] = COV_LIN_EPS;
        }
    }
    for i in 3..6 {
        if c[i][i] <= COV_ANG_EPS {
            c[i][i] = COV_ANG_EPS;
        }
    }
}

pub struct RegistrationVis {
    min_inliers: usize,
    iterations: i32,
    pnp_reproj_error: f32,
    nndr: f32,
    /// Vis/CorNNType: 1(기본) = rtflann 무작위 kd-나무(원본 그대로), 3 = 정확한 해밍 kNN
    kd_tree: bool,
    rng: SharedRng,
    guess_win_size: i32,
    bundle_adjustment: i32,
    repeat_once: bool,
    var_median_ratio: usize,
    ba_iterations: i32,
    ba_baseline: f64,
    ba_robust_delta: f64,
}

/// 해밍 거리용으로 미리 64 비트 넷으로 바꾼 설명자
#[inline]
fn desc_words(d: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for i in 0..4 {
        w[i] = u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}
#[inline]
fn hamming(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones() + (a[2] ^ b[2]).count_ones() + (a[3] ^ b[3]).count_ones()
}

#[inline]
fn is_finite3(p: &[f32; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
}

/// uIsInBounds(v, lo, hi): 유한하고 lo <= v < hi
#[inline]
fn in_bounds(v: f32, lo: f32, hi: f32) -> bool {
    v.is_finite() && !(v < lo) && !(v >= hi)
}

/// 맞추기 안에서 쓰는 장면 보기: 원래 Features 의 일부 번호(순서 유지)
struct View<'a> {
    f: &'a Features,
    idx: Vec<usize>,
    w: Vec<[u64; 4]>,
}

impl<'a> View<'a> {
    fn new(f: &'a Features, idx: Vec<usize>) -> View<'a> {
        let w = idx.iter().map(|&i| desc_words(&f.desc[i])).collect();
        View { f, idx, w }
    }
    fn len(&self) -> usize {
        self.idx.len()
    }
    #[inline]
    fn p3(&self, k: usize) -> [f32; 3] {
        let p = &self.f.pts3d[self.idx[k]];
        [p.x, p.y, p.z]
    }
    #[inline]
    fn kp(&self, k: usize) -> [f32; 2] {
        let kp = &self.f.kps[self.idx[k]];
        [kp.x, kp.y]
    }
}

/// 짝짓기 결과: from 보기 번호 순의 (from k, to k) 와 단어 개수(원본의 최소 개수 검사에 씀)
struct Corr {
    pairs: Vec<(usize, usize)>,
    from_words: usize,
    to_words: usize,
    /// 추측 없는 길에서 to 단어로 남는 보기 번호(중복 단어 제외)
    to_kept: Option<Vec<usize>>,
    /// 추측 창 길: 원본 from 단어 순서(보기 번호)
    from_order: Option<Vec<usize>>,
}

impl RegistrationVis {
    pub fn new(p: &Params) -> RegistrationVis {
        RegistrationVis {
            min_inliers: p.get("Vis/MinInliers", 20i32).max(0) as usize,
            iterations: p.get("Vis/Iterations", 300i32),
            pnp_reproj_error: p.get("Vis/PnPReprojError", 2.0f32),
            nndr: p.get("Vis/CorNNDR", 0.8f32),
            kd_tree: p.get("Vis/CorNNType", 1i32) != 3,
            rng: new_shared_rng(),
            guess_win_size: p.get("Vis/CorGuessWinSize", 40i32),
            bundle_adjustment: p.get("Vis/BundleAdjustment", 1i32),
            repeat_once: p.get_bool("Reg/RepeatOnce", true),
            var_median_ratio: p.get("Vis/PnPVarianceMedianRatio", 4i32).max(2) as usize,
            ba_iterations: p.get("Optimizer/Iterations", 20i32),
            ba_baseline: p.get("Optimizer/Baseline", 0.075f64),
            ba_robust_delta: p.get("Optimizer/RobustKernelDelta", 8.0f64),
        }
    }

    /// 다른 모듈과 같은 난수 상태를 쓰게 함
    pub fn set_rng(&mut self, rng: SharedRng) {
        self.rng = rng;
    }

    pub fn min_inliers(&self) -> usize {
        self.min_inliers
    }

    /// Registration::computeTransformation(from, to, guess). guess = `to` 의 `from` 기준 추측 위치
    pub fn compute(&self, from: &Features, to: &Features, cam: &Camera, guess: Option<Pose>) -> RegInfo {
        let g = guess.map(|p| Tf::from_pose(&p)).unwrap_or(Tf::NULL);
        self.compute_tf(from, to, cam, g)
    }

    /// float 위치 그대로 받는 판(위치 추적이 씀). guess 가 NULL 이면 추측 없음
    pub fn compute_tf(&self, from: &Features, to: &Features, cam: &Camera, guess: Tf) -> RegInfo {
        self.compute_tf_ids(from, None, to, cam, guess)
    }

    /// from 특징에 원본 단어 id 가 붙어 있는 경우(같은 Signature 를 다시 맞출 때). from_ids[i] = i 번째 특징의 단어 id(겹침 없음).
    /// 짝은 단어 id 순으로 PnP 에 들어가고(원본 estimateMotion3DTo2D 가 id 순), kNN 결과가 id 0 인 단어에 닿으면 그 뒤는 버린다(VWDictionary::addNewWords)
    pub fn compute_tf_ids(&self, from: &Features, from_ids: Option<&[i32]>, to: &Features, cam: &Camera, guess: Tf) -> RegInfo {
        let local = cam_local(cam);
        let k = K { fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy };
        let fv = View::new(from, (0..from.kps.len()).collect());
        let tv = View::new(to, (0..to.kps.len()).collect());
        let (mut info, kept) = self.compute_impl(&fv, from_ids, &tv, cam, &k, &local, guess);
        if self.repeat_once && guess.is_null() && info.transform.is_some() {
            // 결과를 추측으로 한 번 더. to 는 첫 번째에서 단어가 된 특징점만(원본은 그 단어를 특징으로 다시 씀)
            let kept: Vec<usize> = kept.unwrap_or_else(|| (0..to.kps.len()).collect());
            let tv2 = View::new(to, kept.clone());
            let t = info.tf;
            let (mut info2, _) = self.compute_impl(&fv, from_ids, &tv2, cam, &k, &local, t);
            info2.to_kept = Some(kept);
            info = info2;
        }
        floor_cov(&mut info.covariance);
        info
    }

    /// computeTransformationImpl 한 번. 두 번째 값: 추측 없는 길에서 to 단어로 남은 원래 번호
    fn compute_impl(&self, fv: &View, from_ids: Option<&[i32]>, tv: &View, cam: &Camera, k: &K, local: &Tf, guess: Tf) -> (RegInfo, Option<Vec<usize>>) {
        let mut info = RegInfo::empty();
        let guess_set = !guess.is_identity() && !guess.is_null();
        let _p0 = std::time::Instant::now();
        let corr = if fv.len() > 0 && tv.len() > 0 {
            if guess_set && self.guess_win_size > 0 {
                self.match_guess(fv, tv, cam, k, local, &guess)
            } else {
                self.match_knn(fv, from_ids, tv)
            }
        } else {
            Corr { pairs: Vec::new(), from_words: fv.len(), to_words: 0, to_kept: None, from_order: None }
        };
        let mut corr = corr;
        if let Some(ids) = from_ids {
            corr.pairs.sort_by_key(|&(a, _)| ids[fv.idx[a]]);
        }
        info.from_order = corr.from_order.as_ref().map(|v| v.iter().map(|&i| fv.idx[i]).collect());
        let kept = corr.to_kept.as_ref().map(|v| v.iter().map(|&i| tv.idx[i]).collect());
        PROF.with(|p| p.borrow_mut()[if guess_set {0} else {1}] += _p0.elapsed().as_secs_f64());
        info.to_words = corr.to_words;
        if corr.to_words == 0 {
            info.rejected = "짝 없음";
            return (info, kept);
        }
        if corr.from_words < self.min_inliers || corr.to_words < self.min_inliers {
            info.rejected = "특징점 부족";
            return (info, kept);
        }
        // estimateMotion3DTo2D: 3D 가 유한한 짝만(기준 점 번호 순 = 원본의 단어 번호 순)
        let mut op: Vec<[f32; 3]> = Vec::with_capacity(corr.pairs.len());
        let mut ip: Vec<[f32; 2]> = Vec::with_capacity(corr.pairs.len());
        let mut mp: Vec<(usize, usize)> = Vec::with_capacity(corr.pairs.len());
        for &(a, b) in &corr.pairs {
            let p = fv.p3(a);
            if is_finite3(&p) {
                op.push(p);
                ip.push(tv.kp(b));
                mp.push((a, b));
            }
        }
        info.matches = mp.len();
        info.match_pairs = mp.iter().map(|&(a, b)| (fv.idx[a], tv.idx[b])).collect();
        let mut covariance = eye6();
        let mut transform = Tf::NULL;
        let mut inliers: Vec<usize> = Vec::new();
        if mp.len() >= self.min_inliers {
            let mut m1 = Vec::new();
            let mut m2 = Vec::new();
            let _p1 = std::time::Instant::now();
            let rr = pnp::solve_pnp_ransac(&op, &ip, k, self.iterations, self.pnp_reproj_error, &mut m1, &mut m2);
            PROF.with(|p| p.borrow_mut()[2] += _p1.elapsed().as_secs_f64());
            if let Some(r) = rr {
                inliers = r.inliers;
                if inliers.len() >= self.min_inliers {
                    let rm = cvmath::rodrigues_v2m(r.rvec);
                    let pnp_t = Tf([
                        rm[0] as f32,
                        rm[1] as f32,
                        rm[2] as f32,
                        r.tvec[0] as f32,
                        rm[3] as f32,
                        rm[4] as f32,
                        rm[5] as f32,
                        r.tvec[1] as f32,
                        rm[6] as f32,
                        rm[7] as f32,
                        rm[8] as f32,
                        r.tvec[2] as f32,
                    ]);
                    transform = local.mul(&pnp_t).inverse();
                    self.pnp_covariance(tv, cam, local, &transform, &op, &ip, &mp, &inliers, &mut covariance);
                }
            }
        }
        if transform.is_null() {
            info.inliers = inliers.len();
            info.inlier_pairs = inliers.iter().map(|&i| (fv.idx[mp[i].0], tv.idx[mp[i].1])).collect();
            info.covariance = covariance;
            info.rejected = "안쪽 점 부족";
            return (info, kept);
        }
        // 묶음 조정
        if self.bundle_adjustment > 0 && !inliers.is_empty() {
            floor_cov(&mut covariance);
            let local_inv = local.inverse();
            let mut pts = Vec::with_capacity(inliers.len());
            for &i in &inliers {
                let (a, b) = mp[i];
                let p3 = op[i];
                let df = local_inv.transform_point(p3)[2];
                let dt = local_inv.transform_point(tv.p3(b))[2];
                pts.push(ba::BaPoint { p3, kp_from: fv.kp(a), depth_from: df, kp_to: tv.kp(b), depth_to: dt });
            }
            let cov_diag = [covariance[0][0], covariance[1][1], covariance[2][2], covariance[3][3], covariance[4][4], covariance[5][5]];
            ba::TRACE.with(|t| {
                if let Some(v) = t.borrow_mut().as_mut() {
                    let mut s = format!("IN {:e} {:e} {:e} {:e}", cam.fx, cam.fy, cam.cx, cam.cy);
                    for x in transform.0 {
                        s += &format!(" {:e}", x);
                    }
                    for x in cov_diag {
                        s += &format!(" {:e}", x);
                    }
                    s += &format!(" {}", pts.len());
                    for p in &pts {
                        s += &format!(" {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e}", p.p3[0], p.p3[1], p.p3[2], p.kp_from[0], p.kp_from[1], p.depth_from, p.kp_to[0], p.kp_to[1], p.depth_to);
                    }
                    v.push(s);
                }
            });
            let _p2 = std::time::Instant::now();
            let out = ba::optimize_ba(&ba::BaInput {
                fx: cam.fx,
                fy: cam.fy,
                cx: cam.cx,
                cy: cam.cy,
                local: *local,
                transform,
                cov_diag,
                points: &pts,
                iterations: self.ba_iterations,
                baseline: self.ba_baseline,
                robust_delta: self.ba_robust_delta,
            });
            PROF.with(|p| p.borrow_mut()[3] += _p2.elapsed().as_secs_f64());
            match out.transform {
                Some(t) => {
                    let kept_in: Vec<usize> = inliers.iter().enumerate().filter(|(j, _)| !out.outliers_to[*j]).map(|(_, &i)| i).collect();
                    inliers = kept_in;
                    if inliers.len() < self.min_inliers {
                        transform = Tf::NULL;
                        info.rejected = "묶음 조정 뒤 안쪽 점 부족";
                    } else {
                        transform = t;
                    }
                }
                None => {
                    transform = Tf::NULL;
                    info.rejected = "묶음 조정 실패";
                }
            }
        }
        info.inliers = inliers.len();
        info.inlier_pairs = inliers.iter().map(|&i| (fv.idx[mp[i].0], tv.idx[mp[i].1])).collect();
        info.covariance = covariance;
        if !transform.is_null() {
            info.tf = transform;
            info.transform = Some(transform.to_pose());
        }
        (info, kept)
    }

    /// estimateMotion3DTo2D 의 분산(PCL computeVariance 흉내): 거리²·각도의 1/4 지점 값 × 2.1981
    #[allow(clippy::too_many_arguments)]
    fn pnp_covariance(&self, tv: &View, cam: &Camera, local: &Tf, transform: &Tf, op: &[[f32; 3]], ip: &[[f32; 2]], mp: &[(usize, usize)], inliers: &[usize], cov: &mut [[f64; 6]; 6]) {
        let n = inliers.len();
        if n == 0 {
            return;
        }
        let mut ed = Vec::with_capacity(n);
        let mut ea = Vec::with_capacity(n);
        let tcf = transform.mul(local);
        let tcf_inv = tcf.inverse();
        for &i in inliers {
            let obj = op[i];
            let to3 = tv.p3(mp[i].1);
            let newp = if is_finite3(&to3) {
                transform.transform_point(to3)
            } else {
                // 깊이 없는 관측: 광선 위, 기준 점 깊이의 1.1 배
                let objb = tcf_inv.transform_point(obj);
                let (cx, cy) = (cam.cx as f32, cam.cy as f32);
                let cx = if cx > 0.0 { cx } else { (cam.width / 2) as f32 - 0.5 };
                let cy = if cy > 0.0 { cy } else { (cam.height / 2) as f32 - 0.5 };
                let ray = [(ip[i][0] - cx) / cam.fx as f32, (ip[i][1] - cy) / cam.fy as f32, 1.0f32];
                let z = objb[2];
                let p = [ray[0] * z, ray[1] * z, ray[2] * z];
                let p = [(p[0] as f64 * 1.1) as f32, (p[1] as f64 * 1.1) as f32, (p[2] as f64 * 1.1) as f32];
                tcf.transform_point(p)
            };
            let (dx, dy, dz) = (obj[0] - newp[0], obj[1] - newp[1], obj[2] - newp[2]);
            ed.push(dx * dx + dy * dy + dz * dz);
            ea.push(angle3d(obj, newp) as f32);
        }
        ed.sort_by(|a, b| a.total_cmp(b));
        ea.sort_by(|a, b| a.total_cmp(b));
        let lin = 2.1981 * ed[n / self.var_median_ratio] as f64;
        let ang = 2.1981 * ea[n / self.var_median_ratio] as f64;
        for i in 0..3 {
            cov[i][i] *= lin + 1e-6;
        }
        for i in 3..6 {
            cov[i][i] *= ang + 1e-6;
        }
    }

    /// 추측으로 투영해 창 안에서 고르기("match projected to frame")
    fn match_guess(&self, fv: &View, tv: &View, cam: &Camera, k: &K, local: &Tf, guess: &Tf) -> Corr {
        let gcr = guess.mul(local).inverse();
        let mut rd = [0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                rd[r * 3 + c] = gcr.at(r, c) as f64;
            }
        }
        // cv::Rodrigues(R → rvec), projectPoints 는 그 rvec 을 다시 행렬로
        let rvec = cvmath::rodrigues_m2v(&rd);
        let tvec = [gcr.x() as f64, gcr.y() as f64, gcr.z() as f64];
        let rr = cvmath::rodrigues_v2m(rvec);
        let w1 = (cam.width as i32 - 1) as f32;
        let h1 = (cam.height as i32 - 1) as f32;
        let radius_sqr = (self.guess_win_size as f32) * (self.guess_win_size as f32);
        let to_pts: Vec<[f32; 2]> = (0..tv.len()).map(|j| tv.kp(j)).collect();
        let mut added_to = vec![false; tv.len()];
        let mut pairs = Vec::new();
        let mut from_words = 0usize;
        let mut any_projected = false;
        let mut nbr = vec![false; fv.len()];
        for i in 0..fv.len() {
            let p3 = fv.p3(i);
            if !is_finite3(&p3) {
                continue;
            }
            // 투영 안 된 유한 점도 from 단어로 들어간다
            from_words += 1;
            let (px, py) = cvmath::project_one(&rr, &tvec, k, p3);
            let (px, py) = (px as f32, py as f32);
            if !(in_bounds(px, 0.0, w1) && in_bounds(py, 0.0, h1) && gcr.transform_point(p3)[2] > 0.0) {
                continue;
            }
            any_projected = true;
            // 이웃이 하나라도 있으면 원본은 이 점을 from 단어로 넣는다(짝 여부와 상관없이)
            // nanoflann 반경 찾기: (q - p)² 합 < r²(float)
            let (mut b0, mut d0, mut d1, mut nc) = (usize::MAX, u32::MAX, u32::MAX, 0usize);
            let mut only = usize::MAX;
            for (j, q) in to_pts.iter().enumerate() {
                let dx = px - q[0];
                let dy = py - q[1];
                if dx * dx + dy * dy < radius_sqr {
                    nc += 1;
                    only = j;
                    let d = hamming(&fv.w[i], &tv.w[j]);
                    if d < d0 {
                        d1 = d0;
                        d0 = d;
                        b0 = j;
                    } else if d < d1 {
                        d1 = d;
                    }
                }
            }
            nbr[i] = nc > 0;
            let matched = if nc >= 2 {
                if (d0 as f32) < self.nndr * (d1 as f32) { Some(b0) } else { None }
            } else if nc == 1 {
                Some(only)
            } else {
                None
            };
            if let Some(j) = matched {
                if !added_to[j] {
                    added_to[j] = true;
                    pairs.push((i, j));
                }
            }
        }
        if !any_projected {
            // 모든 투영이 화면 밖: 원본은 단어를 하나도 만들지 않음
            return Corr { pairs: Vec::new(), from_words: 0, to_words: 0, to_kept: None, from_order: None };
        }
        let mut order: Vec<usize> = (0..fv.len()).filter(|&i| nbr[i]).collect();
        order.extend((0..fv.len()).filter(|&i| !nbr[i] && is_finite3(&fv.p3(i))));
        Corr { pairs, from_words, to_words: tv.len(), to_kept: None, from_order: Some(order) }
    }

    /// 추측 없이: VWDictionary kNN(정확한 해밍 2-NN) + NNDR + 중복 단어 제거
    fn match_knn(&self, fv: &View, from_ids: Option<&[i32]>, tv: &View) -> Corr {
        let nf = fv.len();
        let mut best = vec![usize::MAX; tv.len()];
        if self.kd_tree {
            // VWDictionary(kNNFlannKdTree): from 설명자로 매번 새 색인(L1, 재균형 없음, 나무 4 = random_device 4 번) → to 마다 2-NN, checks 32
            let rows: Vec<[u8; 32]> = fv.idx.iter().map(|&i| fv.f.desc[i]).collect();
            let qs: Vec<[u8; 32]> = tv.idx.iter().map(|&i| tv.f.desc[i]).collect();
            let mut idx = KdTreeIndex::new(1.0);
            {
                let mut st = self.rng.lock().unwrap();
                idx.rand = st.rand.clone();
                let mut k = st.rd_calls;
                idx.set_seed_source(move || {
                    k = k.wrapping_add(1);
                    k.wrapping_mul(2654435761)
                });
                idx.build(&rows);
                st.rd_calls += crate::loopclose::flann::TREES as u32;
                st.rand = idx.rand.clone();
            }
            let (ri, rd) = idx.knn_search(&qs, 2, 32);
            for j in 0..tv.len() {
                // fullResults(multimap): 거리 >= 0 인 것만, 거리 순(같으면 들어온 순)
                let mut res: [(f32, usize); 2] = [(0.0, 0); 2];
                let mut n = 0;
                for c in 0..2 {
                    let (i, d) = (ri[j * 2 + c], rd[j * 2 + c]);
                    if i < 0 {
                        continue;
                    }
                    if from_ids.is_some_and(|v| v[fv.idx[i as usize]] == 0) {
                        break; // 원본: id 0 단어는 결과에서 빼고 거기서 멈춤
                    }
                    if d >= 0.0 {
                        res[n] = (d, i as usize);
                        n += 1;
                    } else {
                        break;
                    }
                }
                if n == 2 && res[1].0 < res[0].0 {
                    res.swap(0, 1);
                }
                if n >= 2 && !(res[0].0 > self.nndr * res[1].0) {
                    best[j] = res[0].1;
                }
            }
        } else if nf >= 2 {
            for j in 0..tv.len() {
                let (mut b0, mut b1, mut d0, mut d1) = (usize::MAX, usize::MAX, u32::MAX, u32::MAX);
                let tw = &tv.w[j];
                for i in 0..nf {
                    let d = hamming(&fv.w[i], tw);
                    if d < d0 {
                        d1 = d0;
                        b1 = b0;
                        d0 = d;
                        b0 = i;
                    } else if d < d1 {
                        d1 = d;
                        b1 = i;
                    }
                }
                // 원본: id 0 단어는 결과에서 빼고 거기서 멈춤 → 결과가 둘 못 되면 짝 없음
                if from_ids.is_some_and(|v| v[fv.idx[b0]] == 0 || v[fv.idx[b1]] == 0) {
                    continue;
                }
                // 거절 조건: d0 > nndr*d1(제곱 L2 = 해밍, float)
                if !((d0 as f32) > self.nndr * (d1 as f32)) {
                    best[j] = b0;
                }
            }
        }
        let mut cnt = vec![0u32; nf];
        for &b in &best {
            if b != usize::MAX {
                cnt[b] += 1;
            }
        }
        let mut pairs = Vec::new();
        let mut kept = Vec::with_capacity(tv.len());
        for j in 0..tv.len() {
            let b = best[j];
            if b == usize::MAX {
                kept.push(j);
            } else if cnt[b] == 1 {
                kept.push(j);
                pairs.push((b, j));
            }
        }
        pairs.sort_unstable();
        Corr { pairs, from_words: nf, to_words: kept.len(), to_kept: Some(kept), from_order: None }
    }
}

/// pcl::getAngle3D(Vector4f(x,y,z,0), ...): float 정규화·내적(SSE 축약 순서) 뒤 double acos
fn angle3d(a: [f32; 3], b: [f32; 3]) -> f64 {
    let n = |v: [f32; 3]| -> [f32; 4] {
        let s = (v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + 0.0f32 * 0.0);
        if s > 0.0 {
            let r = s.sqrt();
            [v[0] / r, v[1] / r, v[2] / r, 0.0 / r]
        } else {
            [v[0], v[1], v[2], 0.0]
        }
    };
    let (x, y) = (n(a), n(b));
    let d = (x[0] * y[0] + x[2] * y[2]) + (x[1] * y[1] + x[3] * y[3]);
    let mut rad = d as f64;
    if rad < -1.0 {
        rad = -1.0;
    } else if rad > 1.0 {
        rad = 1.0;
    }
    rad.acos()
}

/// Camera.local(몸체 → 광학) 을 float Transform 으로. 광학 회전(0, ±1 만) 은 정확한 값으로 맞춤
pub fn cam_local(cam: &Camera) -> Tf {
    let mut t = Tf::from_pose(&cam.local);
    for v in t.0.iter_mut() {
        for c in [-1.0f32, 0.0, 1.0] {
            if (*v - c).abs() < 1e-6 {
                *v = c;
            }
        }
    }
    t
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::KeyPoint;
    use nalgebra::{Point3, Translation3, UnitQuaternion, Vector3};

    /// 결정적 난수(시험용)
    pub struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        pub fn f(&mut self) -> f64 {
            (self.next() as f64) / ((1u64 << 53) as f64)
        }
    }

    pub fn d435() -> Camera {
        // 광학 회전: 몸체(x 앞, y 왼, z 위) 의 광학 축(x 오른, y 아래, z 앞)
        let m = nalgebra::Matrix3::new(0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0);
        let q = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(m));
        Camera { fx: 525.0, fy: 525.0, cx: 319.5, cy: 239.5, width: 640, height: 480, local: Pose::from_parts(Translation3::new(0.0, 0.0, 0.0), q) }
    }

    /// 몸체 축 3D 점들을 두 위치에서 본 특징. 설명자는 무작위, to 쪽은 8 비트 뒤집음.
    /// motion = to 장면의 from 기준 위치
    pub fn synth_scene(n: usize, cam: &Camera, from_pose: &Pose, to_pose: &Pose, noise_px: f64, outlier_frac: f64, seed: u64, world: &mut Vec<(Point3<f64>, [u8; 32])>) -> (Features, Features) {
        let mut r = Lcg(seed);
        let local = cam.local;
        let mut from = Features::default();
        let mut to = Features::default();
        if world.is_empty() {
            for _ in 0..n * 4 {
                let p = Point3::new(1.0 + 4.0 * r.f(), (r.f() - 0.5) * 4.0, (r.f() - 0.5) * 3.0);
                let mut d = [0u8; 32];
                for b in d.iter_mut() {
                    *b = r.next() as u8;
                }
                world.push((p, d));
            }
        }
        for (pw, d) in world.iter() {
            if from.kps.len() >= n {
                break;
            }
            let pb1 = from_pose.inverse() * pw;
            let pb2 = to_pose.inverse() * pw;
            let pc1 = local.inverse() * pb1;
            let pc2 = local.inverse() * pb2;
            if pc1.z < 0.3 || pc2.z < 0.3 {
                continue;
            }
            let u1 = cam.fx * pc1.x / pc1.z + cam.cx;
            let v1 = cam.fy * pc1.y / pc1.z + cam.cy;
            let mut u2 = cam.fx * pc2.x / pc2.z + cam.cx + (r.f() - 0.5) * 2.0 * noise_px;
            let mut v2 = cam.fy * pc2.y / pc2.z + cam.cy + (r.f() - 0.5) * 2.0 * noise_px;
            if !(u1 >= 0.0 && u1 < 639.0 && v1 >= 0.0 && v1 < 479.0 && u2 >= 0.0 && u2 < 639.0 && v2 >= 0.0 && v2 < 479.0) {
                continue;
            }
            let mut d2 = *d;
            for _ in 0..8 {
                let bit = (r.next() % 256) as usize;
                d2[bit / 8] ^= 1 << (bit % 8);
            }
            if r.f() < outlier_frac {
                u2 = 20.0 + r.f() * 600.0;
                v2 = 20.0 + r.f() * 440.0;
            }
            from.kps.push(KeyPoint { x: u1 as f32, y: v1 as f32, ..Default::default() });
            from.pts3d.push(Point3::new(pb1.x as f32, pb1.y as f32, pb1.z as f32));
            from.desc.push(*d);
            to.kps.push(KeyPoint { x: u2 as f32, y: v2 as f32, ..Default::default() });
            to.pts3d.push(Point3::new(pb2.x as f32, pb2.y as f32, pb2.z as f32));
            to.desc.push(d2);
        }
        (from, to)
    }

    pub fn synth(n: usize, motion: &Pose, noise_px: f64, outlier_frac: f64, seed: u64) -> (Features, Features, Camera) {
        let cam = d435();
        let mut w = Vec::new();
        let (f, t) = synth_scene(n, &cam, &Pose::identity(), motion, noise_px, outlier_frac, seed, &mut w);
        (f, t, cam)
    }

    fn err(a: &Pose, b: &Pose) -> (f64, f64) {
        let d = a.inverse() * b;
        (d.translation.vector.norm(), d.rotation.angle().to_degrees())
    }

    fn motion() -> Pose {
        Pose::from_parts(Translation3::new(0.05, -0.02, 0.01), UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03)))
    }

    #[test]
    fn known_motion_no_guess() {
        let m = motion();
        let (f, t, cam) = synth(300, &m, 0.0, 0.0, 1);
        let reg = RegistrationVis::new(&Params::default());
        let info = reg.compute(&f, &t, &cam, None);
        let tr = info.transform.expect("변환");
        let (dt, da) = err(&tr, &m);
        assert!(dt < 1e-3 && da < 0.1, "dt={dt} da={da} inliers={}", info.inliers);
        assert!(info.inliers > 250, "inliers={}", info.inliers);
    }

    #[test]
    fn known_motion_with_guess() {
        let m = motion();
        let (f, t, cam) = synth(300, &m, 0.0, 0.0, 2);
        let reg = RegistrationVis::new(&Params::default());
        let g = Pose::from_parts(Translation3::new(0.04, -0.015, 0.0), UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -0.015, 0.025)));
        let info = reg.compute(&f, &t, &cam, Some(g));
        let tr = info.transform.expect("변환");
        let (dt, da) = err(&tr, &m);
        assert!(dt < 1e-3 && da < 0.1, "dt={dt} da={da}");
    }

    #[test]
    fn outliers_rejected() {
        let m = motion();
        let (f, t, cam) = synth(300, &m, 0.5, 0.3, 3);
        let reg = RegistrationVis::new(&Params::default());
        let info = reg.compute(&f, &t, &cam, None);
        let tr = info.transform.expect("변환");
        let (dt, da) = err(&tr, &m);
        assert!(dt < 5e-3 && da < 0.3, "dt={dt} da={da}");
        // 바깥값으로 옮긴 관측은 안쪽 점이 아니어야 함
        for &(a, b) in &info.inlier_pairs {
            let (u, v) = (t.kps[b].x as f64, t.kps[b].y as f64);
            let pb = m.inverse() * Point3::new(f.pts3d[a].x as f64, f.pts3d[a].y as f64, f.pts3d[a].z as f64);
            let pc = cam.local.inverse() * pb;
            let (pu, pv) = (cam.fx * pc.x / pc.z + cam.cx, cam.fy * pc.y / pc.z + cam.cy);
            assert!(((u - pu).powi(2) + (v - pv).powi(2)).sqrt() < 5.0);
        }
        assert!(info.inliers < info.matches);
    }

    #[test]
    fn min_inliers_rejection() {
        let m = motion();
        let (f, t, cam) = synth(15, &m, 0.0, 0.0, 4);
        let reg = RegistrationVis::new(&Params::default());
        let info = reg.compute(&f, &t, &cam, None);
        assert!(info.transform.is_none());
        let mut p = Params::default();
        p.set("Vis/MinInliers", "10");
        let reg = RegistrationVis::new(&p);
        let info = reg.compute(&f, &t, &cam, None);
        assert!(info.transform.is_some(), "{}", info.rejected);
    }
}
