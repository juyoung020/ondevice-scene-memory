//! LoopClosing::CorrectLoop(src/LoopClosing.cc). 지도 스레드를 멈춘 뒤의 몸통: 공시 키프레임 Sim3 보정 → 지도 점 옮김 →
//! 루프 짝 점 바꿔 끼움 → SearchAndFuse → 새 연결(LoopConnections) → OptimizeEssentialGraph → 루프 가장자리.
//! 멈춤 · 대기열 비우기 · GBA 멈춤은 부르는 쪽(`orb3sys::threads` 의 StopToken, 대기열은 지도 스레드가 비움).
//! OptimizeEssentialGraph 는 트레잇(BA 몫) — 대조 시험은 원본 실행 결과를 넣는다.
//! std::map<KeyFrame*> 순회는 번호 순(graph.rs 머리말).
use std::collections::{BTreeMap, BTreeSet};

use super::detect::{quatd_rotate, G2oSim3};
use super::graph::{add_loop_edge, connected_set, update_connections, update_normal_and_depth};
use crate::orb3rs::matcher::{replace_map_point, Matcher};
use crate::orb3rs::{quat_normalize, KeyFrame, KfId, MapPoint, MpId, Se3};

impl G2oSim3 {
    /// inverse: (r̄, r̄·((−1/s)·t), 1/s)
    pub fn inverse(&self) -> G2oSim3 {
        let rc = [-self.r[0], -self.r[1], -self.r[2], self.r[3]];
        let c = -1.0 / self.s;
        let t = quatd_rotate(&rc, &[c * self.t[0], c * self.t[1], c * self.t[2]]);
        G2oSim3 { r: rc, t, s: 1.0 / self.s }
    }
    /// map(p) = s·(r·p) + t
    pub fn map(&self, p: &[f64; 3]) -> [f64; 3] {
        let q = quatd_rotate(&self.r, p);
        [self.s * q[0] + self.t[0], self.s * q[1] + self.t[1], self.s * q[2] + self.t[2]]
    }
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
pub fn correct_loop<E: EssentialGraph>(
    inp: &LoopIn,
    kfs: &mut BTreeMap<KfId, KeyFrame>,
    mps: &mut BTreeMap<MpId, MapPoint>,
    corrected_by: &mut BTreeMap<MpId, (KfId, KfId)>,
    ess: &mut E,
) -> LoopOut {
    let cur = inp.cur;
    update_connections(cur, kfs, mps, false);
    let mut conn_kfs: Vec<KfId> = kfs[&cur].ordered_conn.iter().map(|c| c.0).collect();
    conn_kfs.push(cur);
    let mut corrected = BTreeMap::new();
    let mut non = BTreeMap::new();
    corrected.insert(cur, inp.scw);
    let twc = kfs[&cur].twc;
    let tcw = kfs[&cur].tcw;
    non.insert(cur, G2oSim3::from_se3_raw(&tcw));
    kfs.get_mut(&cur).unwrap().set_pose(inp.scw.to_se3f());
    for &k in &conn_kfs {
        if k == cur {
            continue;
        }
        let tiw = kfs[&k].tcw;
        let tic = tiw.mul(&twc);
        let csiw = G2oSim3::from_se3(&tic).mul(&inp.scw);
        corrected.insert(k, csiw);
        kfs.get_mut(&k).unwrap().set_pose(csiw.to_se3f());
        non.insert(k, G2oSim3::from_se3_raw(&tiw));
    }
    for (&k, csiw) in &corrected {
        let cswi = csiw.inverse();
        let siw = non[&k];
        let pts: Vec<Option<MpId>> = kfs[&k].map_points.clone();
        for m in pts.into_iter().flatten() {
            if mps[&m].bad || corrected_by.get(&m).is_some_and(|c| c.0 == cur) {
                continue;
            }
            let p = mps[&m].pos.map(f64::from);
            let c = cswi.map(&siw.map(&p));
            let mp = mps.get_mut(&m).unwrap();
            mp.pos = c.map(|v| v as f32);
            corrected_by.insert(m, (cur, k));
            update_normal_and_depth(mp, kfs);
        }
        // bImuInit 이면 속도 보정(RGB-D 는 없음)
        update_connections(k, kfs, mps, false);
    }
    for (i, lm) in inp.loop_matched.iter().enumerate() {
        let Some(lm) = *lm else { continue };
        match kfs[&cur].map_points[i] {
            Some(c) => replace_map_point(c, lm, kfs, mps),
            None => {
                kfs.get_mut(&cur).unwrap().map_points[i] = Some(lm);
                let kf = &kfs[&cur];
                let p = mps.get_mut(&lm).unwrap();
                p.add_observation(kf, i);
                p.compute_distinctive_descriptors(kfs);
            }
        }
    }
    // SearchAndFuse(CorrectedSim3, mvpLoopMapPoints)
    let matcher = Matcher::new(0.8, true);
    let lmps: Vec<MpId> = inp.loop_mps.iter().map(|m| m.expect("mvpLoopMapPoints 에 NULL")).collect();
    for (&k, siw) in &corrected {
        let scw = siw.to_sophus();
        let mut rep = vec![None; lmps.len()];
        matcher.fuse_sim3(k, kfs, &scw, &lmps, mps, 4.0, &mut rep);
        for (i, r) in rep.iter().enumerate() {
            if let Some(r) = *r {
                replace_map_point(r, lmps[i], kfs, mps);
            }
        }
    }
    let mut loop_conn: BTreeMap<KfId, BTreeSet<KfId>> = BTreeMap::new();
    for &k in &conn_kfs {
        let prev: Vec<KfId> = kfs[&k].ordered_conn.iter().map(|c| c.0).collect();
        update_connections(k, kfs, mps, false);
        let mut s = connected_set(&kfs[&k]);
        for p in &prev {
            s.remove(p);
        }
        for c in &conn_kfs {
            s.remove(c);
        }
        loop_conn.insert(k, s);
    }
    ess.optimize(kfs, mps, corrected_by, inp.matched, cur, &non, &corrected, &loop_conn, inp.fix_scale);
    add_loop_edge(inp.matched, cur, kfs);
    add_loop_edge(cur, inp.matched, kfs);
    LoopOut { corrected, non_corrected: non, loop_conn, run_gba: !inp.imu_init, not_erase: [inp.matched, cur] }
}
