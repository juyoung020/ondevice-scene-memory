//! `Tracking::Relocalization`(src/Tracking.cc) 흐름. 후보 질의(`KeyFrameDatabase::DetectRelocalizationCandidates`,
//! 패치판 지도별)와 BoW 계산은 부르는 쪽 몫: 여기는 후보 목록을 받아 원본 순서 그대로
//! SearchByBoW(0.75) → PnP RANSAC 5 번씩 돌아가며 → PoseOptimization → 투영 보강(0.9, 10/100 · 3/64) 을 한다.
//!
//! PnP 는 원본 MLPnPsolver 자리. 포터 OPT 가 Rust 로 옮기는 중이라 트레잇으로 받는다(지금은 원본 FFI 를 끼움).
//! 트레잇은 Sophus::SE3f(eigTcw) 생성까지 끝낸 자세를 돌려준다(행렬→사원수 비트를 원본 쪽에서 정함).

use std::collections::{BTreeMap, BTreeSet};

use crate::orb3rs::matcher::Matcher;
use crate::orb3rs::pinhole::Pinhole;
use crate::orb3rs::pose_opt::{pose_optimization, PoseCam, PoseObs, Se3f};
use crate::orb3rs::{Frame, KeyFrame, KfId, MapPoint, MpId, Se3};

/// MLPnPsolver 하나(후보 키프레임 하나). `iterate(5, ...)` 의 결과.
pub struct PnpStep {
    pub pose: Option<Se3>,
    pub no_more: bool,
    /// 장면 점 수(N) 길이
    pub inliers: Vec<bool>,
}

pub trait PnpSolver {
    fn iterate(&mut self, n: i32) -> PnpStep;
}

/// `new MLPnPsolver(F, matches)` + `SetRansacParameters(0.99,10,300,6,0.5,5.991)`
pub trait PnpFactory {
    type S: PnpSolver;
    fn make(&mut self, f: &Frame, matches: &[Option<MpId>], mps: &BTreeMap<MpId, MapPoint>) -> Self::S;
}

/// `Optimizer::PoseOptimization(&mCurrentFrame)` 를 Frame 위에서: 맵 점 칸만 관측으로, 이상치 표시를 되돌려 쓰고 자세 갱신.
pub fn pose_optimization_frame(f: &mut Frame, mps: &BTreeMap<MpId, MapPoint>) -> i32 {
    let cam = f.ctx.cam;
    let pc = PoseCam { cam: Pinhole::new(cam.fx, cam.fy, cam.cx, cam.cy), fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy, bf: cam.bf };
    let mut idx = Vec::new();
    let mut obs = Vec::new();
    for (i, m) in f.map_points.iter().enumerate() {
        if let Some(id) = m {
            let k = &f.keys[i];
            obs.push(PoseObs { u: k.x, v: k.y, ur: f.u_right[i], inv_sigma2: f.ctx.pyr.inv_sigma2[k.octave as usize], xw: mps[id].pos });
            idx.push(i);
            f.outlier[i] = false;
        }
    }
    let t = f.tcw.unwrap_or_default();
    let mut out = vec![false; obs.len()];
    let (n, p) = pose_optimization(&Se3f { q: t.q, t: t.t }, &pc, &obs, &mut out);
    if obs.len() < 3 {
        return 0;
    }
    for (j, &i) in idx.iter().enumerate() {
        f.outlier[i] = out[j];
    }
    f.set_pose(Se3 { q: p.q, t: p.t });
    n
}

fn drop_outliers(f: &mut Frame) {
    for i in 0..f.keys.len() {
        if f.outlier[i] {
            f.map_points[i] = None;
        }
    }
}

/// 성공하면 true(부르는 쪽이 mnLastRelocFrameId = F.mnId).
pub fn relocalize<P: PnpFactory>(
    f: &mut Frame,
    cands: &[KfId],
    kfs: &BTreeMap<KfId, KeyFrame>,
    mps: &BTreeMap<MpId, MapPoint>,
    pnp: &mut P,
) -> bool {
    if cands.is_empty() {
        return false;
    }
    let matcher = Matcher::new(0.75, true);
    let nk = cands.len();
    let mut solvers: Vec<Option<P::S>> = (0..nk).map(|_| None).collect();
    let mut matches: Vec<Vec<Option<MpId>>> = vec![Vec::new(); nk];
    let mut discarded = vec![false; nk];
    let mut ncand = 0;
    for i in 0..nk {
        let kf = &kfs[&cands[i]];
        if kf.bad {
            discarded[i] = true;
            continue;
        }
        let (n, m) = matcher.search_by_bow_kf_frame(kf, f, mps);
        matches[i] = m;
        if n < 15 {
            discarded[i] = true;
            continue;
        }
        solvers[i] = Some(pnp.make(f, &matches[i], mps));
        ncand += 1;
    }
    let matcher2 = Matcher::new(0.9, true);
    let mut matched = false;
    while ncand > 0 && !matched {
        for i in 0..nk {
            if discarded[i] {
                continue;
            }
            let st = solvers[i].as_mut().unwrap().iterate(5);
            if st.no_more {
                discarded[i] = true;
                ncand -= 1;
            }
            let Some(pose) = st.pose else { continue };
            f.set_pose(pose);
            let mut found: BTreeSet<MpId> = BTreeSet::new();
            for j in 0..st.inliers.len() {
                if st.inliers[j] {
                    f.map_points[j] = matches[i][j];
                    if let Some(id) = matches[i][j] {
                        found.insert(id);
                    }
                } else {
                    f.map_points[j] = None;
                }
            }
            // 원본 sFound 는 NULL 도 넣지만 투영 짝짓기는 맵 점 포인터로만 찾으므로 차이 없음
            let mut ngood = pose_optimization_frame(f, mps);
            if ngood < 10 {
                continue;
            }
            drop_outliers(f);
            if ngood < 50 {
                let kf = &kfs[&cands[i]];
                let nadd = matcher2.search_by_projection_kf(f, kf, &found, mps, 10.0, 100);
                if nadd + ngood >= 50 {
                    ngood = pose_optimization_frame(f, mps);
                    if ngood > 30 && ngood < 50 {
                        found = f.map_points.iter().flatten().copied().collect();
                        let nadd = matcher2.search_by_projection_kf(f, kf, &found, mps, 3.0, 64);
                        if ngood + nadd >= 50 {
                            ngood = pose_optimization_frame(f, mps);
                            drop_outliers(f);
                        }
                    }
                }
            }
            if ngood >= 50 {
                matched = true;
                break;
            }
        }
    }
    matched
}
