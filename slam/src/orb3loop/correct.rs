//! LoopClosing::CorrectLoop(src/LoopClosing.cc). 지도 스레드를 멈춘 뒤의 몸통: 공시 키프레임 Sim3 보정 → 지도 점 옮김 →
//! 루프 짝 점 바꿔 끼움 → SearchAndFuse → 새 연결(LoopConnections) → OptimizeEssentialGraph → 루프 가장자리.
//! 멈춤 · 대기열 비우기 · GBA 멈춤은 부르는 쪽(`orb3sys::threads` 의 StopToken, 대기열은 지도 스레드가 비움).
//! OptimizeEssentialGraph 는 트레잇(BA 몫) — 대조 시험은 원본 실행 결과를 넣는다.
//! std::map<KeyFrame*> 순회는 번호 순(graph.rs 머리말).
use std::collections::{BTreeMap, BTreeSet};

use super::detect::G2oSim3;
use super::graph::{connected_set, update_normal_and_depth};
use crate::orb3rs::mapdata::{kf_update_connections, mp_replace};
use crate::orb3rs::Atlas;
use crate::orb3rs::matcher::Matcher;

use crate::orb3rs::{quat_normalize, KeyFrame, KfId, MapPoint, MpId, Se3};

impl G2oSim3 {
    /// Sophus::SE3d(r, t / s).cast<float>(): double 사원수 정규화 → float 로 내려 다시 정규화
    pub fn to_se3f(&self) -> Se3 {
        let q = &self.r;
        let n = ((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])).sqrt();
        let qd = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
        let qf = quat_normalize(qd.map(|v| v as f32));
        Se3 { q: qf, t: [(self.t[0] / self.s) as f32, (self.t[1] / self.s) as f32, (self.t[2] / self.s) as f32] }
    }
}

/// `Optimizer::OptimizeEssentialGraph(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale)`.
/// `corrected_by` = 지도 점의 (mnCorrectedByKF, mnCorrectedReference)(원본은 지도 점 칸).
pub trait EssentialGraph {
    #[allow(clippy::too_many_arguments)]
    fn optimize(
        &mut self,
        kfs: &mut BTreeMap<KfId, KeyFrame>,
        mps: &mut BTreeMap<MpId, MapPoint>,
        corrected_by: &BTreeMap<MpId, (KfId, KfId)>,
        loop_kf: KfId,
        cur: KfId,
        non_corrected: &BTreeMap<KfId, G2oSim3>,
        corrected: &BTreeMap<KfId, G2oSim3>,
        loop_conn: &BTreeMap<KfId, BTreeSet<KfId>>,
        fix_scale: bool,
    );
}

/// CorrectLoop 입력(LoopClosing 멤버)
pub struct LoopIn<'a> {
    pub cur: KfId,
    pub matched: KfId,
    /// mg2oLoopScw
    pub scw: G2oSim3,
    /// mvpLoopMapPoints
    pub loop_mps: &'a [Option<MpId>],
    /// mvpLoopMatchedMPs
    pub loop_matched: &'a [Option<MpId>],
    pub fix_scale: bool,
    /// 지도가 IMU 초기화됐는지(RGB-D 는 거짓)
    pub imu_init: bool,
}

pub struct LoopOut {
    pub corrected: BTreeMap<KfId, G2oSim3>,
    pub non_corrected: BTreeMap<KfId, G2oSim3>,
    pub loop_conn: BTreeMap<KfId, BTreeSet<KfId>>,
    /// mbRunningGBA 를 켜고 전역 BA 스레드를 띄울지(원본: IMU 초기화 전이거나 키프레임 200 개 미만 · 지도 1 개. 뒤 조건은 IMU 쪽이라 안 옮김)
    pub run_gba: bool,
    /// 루프 양끝 키프레임: 원본은 AddLoopEdge 에서 mbNotErase = true
    pub not_erase: [KfId; 2],
}

/// CorrectLoop 몸통(지도 스레드가 멈춘 뒤). `corrected_by` 는 지도 점 mnCorrectedByKF · mnCorrectedReference(계속 들고 다님)
pub fn correct_loop<E: EssentialGraph>(inp: &LoopIn, a: &mut Atlas, corrected_by: &mut BTreeMap<MpId, (KfId, KfId)>, ess: &mut E) -> LoopOut {
    let cur = inp.cur;
    kf_update_connections(a, cur, false);
    let mut conn_kfs: Vec<KfId> = a.kfs[&cur].ordered_conn.iter().map(|c| c.0).collect();
    conn_kfs.push(cur);
    let mut corrected = BTreeMap::new();
    let mut non = BTreeMap::new();
    corrected.insert(cur, inp.scw);
    let twc = a.kfs[&cur].twc;
    let tcw = a.kfs[&cur].tcw;
    non.insert(cur, G2oSim3::from_se3_raw(&tcw));
    a.kfs.get_mut(&cur).unwrap().set_pose(inp.scw.to_se3f());
    for &k in &conn_kfs {
        if k == cur {
            continue;
        }
        let tiw = a.kfs[&k].tcw;
        let tic = tiw.mul(&twc);
        let csiw = G2oSim3::from_se3(&tic).mul(&inp.scw);
        corrected.insert(k, csiw);
        a.kfs.get_mut(&k).unwrap().set_pose(csiw.to_se3f());
        non.insert(k, G2oSim3::from_se3_raw(&tiw));
    }
    for (&k, csiw) in &corrected {
        let cswi = csiw.inverse();
        let siw = non[&k];
        let pts: Vec<Option<MpId>> = a.kfs[&k].map_points.clone();
        for m in pts.into_iter().flatten() {
            if a.mps[&m].bad || corrected_by.get(&m).is_some_and(|c| c.0 == cur) {
                continue;
            }
            let p = a.mps[&m].pos.map(f64::from);
            let c = cswi.map(&siw.map(&p));
            let mp = a.mps.get_mut(&m).unwrap();
            mp.pos = c.map(|v| v as f32);
            corrected_by.insert(m, (cur, k));
            update_normal_and_depth(mp, &a.kfs);
        }
        // bImuInit 이면 속도 보정(RGB-D 는 없음)
        kf_update_connections(a, k, false);
    }
    for (i, lm) in inp.loop_matched.iter().enumerate() {
        let Some(lm) = *lm else { continue };
        match a.kfs[&cur].map_points[i] {
            Some(c) => mp_replace(a, c, lm),
            None => {
                a.kfs.get_mut(&cur).unwrap().map_points[i] = Some(lm);
                let p = a.mps.get_mut(&lm).unwrap();
                p.add_observation(&a.kfs[&cur], i);
                p.compute_distinctive_descriptors(&a.kfs);
            }
        }
    }
    // SearchAndFuse(CorrectedSim3, mvpLoopMapPoints)
    let lmps: Vec<MpId> = inp.loop_mps.iter().map(|m| m.expect("mvpLoopMapPoints 에 NULL")).collect();
    search_and_fuse(&corrected, &lmps, a);
    crate::orb3sys::mapping::step_digest_atlas(a, inp.cur, "fuse");
    let mut loop_conn: BTreeMap<KfId, BTreeSet<KfId>> = BTreeMap::new();
    for &k in &conn_kfs {
        let prev: Vec<KfId> = a.kfs[&k].ordered_conn.iter().map(|c| c.0).collect();
        kf_update_connections(a, k, false);
        let mut s = connected_set(&a.kfs[&k]);
        for p in &prev {
            s.remove(p);
        }
        for c in &conn_kfs {
            s.remove(c);
        }
        loop_conn.insert(k, s);
    }
    ess.optimize(&mut a.kfs, &mut a.mps, corrected_by, inp.matched, cur, &non, &corrected, &loop_conn, inp.fix_scale);
    a.kfs.get_mut(&inp.matched).unwrap().add_loop_edge(cur);
    a.kfs.get_mut(&cur).unwrap().add_loop_edge(inp.matched);
    LoopOut { corrected, non_corrected: non, loop_conn, run_gba: !inp.imu_init, not_erase: [inp.matched, cur] }
}

/// LoopClosing::SearchAndFuse(KeyFrameAndPose, vpMapPoints): 보정 Sim3 마다 Fuse(th 4) 뒤 바꿀 점을 Replace
pub fn search_and_fuse(corrected: &BTreeMap<KfId, G2oSim3>, pts: &[MpId], a: &mut Atlas) {
    let matcher = Matcher::new(0.8, true);
    for (&k, siw) in corrected {
        let scw = siw.to_sophus();
        let mut rep = vec![None; pts.len()];
        matcher.fuse_sim3(k, &mut a.kfs, &scw, pts, &mut a.mps, 4.0, &mut rep);
        for (i, r) in rep.iter().enumerate() {
            if let Some(r) = *r {
                mp_replace(a, r, pts[i]);
            }
        }
    }
}
