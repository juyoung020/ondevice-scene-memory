//! OptimizeEssentialGraph(루프 판, RGB-D)를 포터 LOOP CorrectLoop(`orb3loop::correct::EssentialGraph`)에 엮은 판:
//! 그래프 만들기 · 20 번 Levenberg 는 포터 CV(`orb3cv::sim3opt`, 원본 9 호출 비트 동일), 입력 모으기 · 결과 되써 넣기는 원본 순서대로 —
//! 키프레임(vpKFs = GetAllKeyFrames, 번호 순)마다 자세 = Siw(정점) → SE3(R, t/s), 그 뒤 점(vpMPs, 번호 순)마다
//! 고친 기준 키프레임(mnCorrectedByKF == 지금 KF 면 mnCorrectedReference)으로 P' = Scw⁻¹(새) · Srw(옛) · P, UpdateNormalAndDepth.
use crate::orb3cv::sim3opt::{build_eg_loop, eg_loop_points, optimize_essential_graph, sim3_to_se3f, EgKf, EgLoopInput, EgMp};
use crate::orb3loop::correct::EssentialGraph;
use crate::orb3loop::detect::G2oSim3;
use crate::orb3rs::mapdata::mp_update_normal_and_depth;
use crate::orb3rs::types::{Atlas, KeyFrame, KfId, MapPoint, MpId, Se3};
use std::collections::{BTreeMap, BTreeSet};

/// 루프 판 본질 그래프(Rust)
pub struct RsEss {
    /// 지도 mnInitKFid(정점 고정). None 이면 지도의 가장 작은 키프레임 번호(첫 키프레임은 지워지지 않음)
    pub init_kf: Option<KfId>,
}

impl EssentialGraph for RsEss {
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
    ) {
        let map = kfs[&cur].map;
        let ids: Vec<KfId> = kfs.iter().filter(|(_, k)| k.map == map && !k.bad).map(|(id, _)| *id).collect();
        let eg_kfs: Vec<EgKf> = ids
            .iter()
            .map(|id| {
                let k = &kfs[id];
                let mut children = k.children.clone();
                children.sort_unstable();
                let mut loop_edges = k.loop_edges.clone();
                loop_edges.sort_unstable();
                EgKf {
                    id: *id as u32,
                    bad: k.bad,
                    q: k.tcw.q,
                    t: k.tcw.t,
                    corrected: corrected.get(id).copied(),
                    non_corrected: non_corrected.get(id).copied(),
                    parent: k.parent.map(|p| p as u32),
                    children: children.into_iter().map(|c| c as u32).collect(),
                    loop_edges: loop_edges.into_iter().map(|c| c as u32).collect(),
                    covis100: k.covisibles_by_weight(100).into_iter().map(|n| (n as u32, kfs.get(&n).is_none_or(|x| x.bad))).collect(),
                }
            })
            .collect();
        let lc: Vec<(u32, Vec<(u32, i32)>)> =
            loop_conn.iter().map(|(k, s)| (*k as u32, s.iter().map(|n| (*n as u32, kfs[k].weight(*n))).collect())).collect();
        let init = self.init_kf.unwrap_or_else(|| ids[0]);
        let inp = EgLoopInput { cur_kf: cur as u32, loop_kf: loop_kf as u32, init_kf: init as u32, fix_scale, kfs: eg_kfs, loop_conn: lc };
        let mut g = build_eg_loop(&inp);
        optimize_essential_graph(&mut g.vertices, &g.edges, 20);
        for v in &g.vertices {
            let (q, t) = sim3_to_se3f(&v.est);
            if let Some(k) = kfs.get_mut(&(v.id as KfId)) {
                k.set_pose(Se3 { q, t });
            }
        }
        let mp_ids: Vec<MpId> = mps.iter().filter(|(_, p)| p.map == map && !p.bad).map(|(id, _)| *id).collect();
        let eg_mps: Vec<EgMp> = mp_ids
            .iter()
            .map(|id| {
                let p = &mps[id];
                let (cb, cr) = corrected_by.get(id).copied().unwrap_or((0, 0));
                EgMp { bad: p.bad, corrected_by_kf: cb, corrected_reference: cr, ref_kf: p.ref_kf as u32, pos: p.pos }
            })
            .collect();
        let pts = eg_loop_points(&g, cur as u32, &eg_mps);
        // UpdateNormalAndDepth 는 Atlas 판이라 잠깐 옮겨 담음
        let mut a = Atlas { kfs: std::mem::take(kfs), mps: std::mem::take(mps), ..Default::default() };
        let live: Vec<MpId> = mp_ids.into_iter().filter(|id| !a.mps[id].bad).collect();
        for (id, (_, p)) in live.iter().zip(pts) {
            a.mps.get_mut(id).unwrap().pos = p;
            mp_update_normal_and_depth(&mut a, *id);
        }
        *kfs = a.kfs;
        *mps = a.mps;
    }
}
