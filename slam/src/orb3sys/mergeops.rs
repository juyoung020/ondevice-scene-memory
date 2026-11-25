//! MergeLocal 의 BA 두 개를 포터 LOOP `orb3loop::merge` 트레잇에 엮은 판:
//! - 맞춤 BA(Optimizer::LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, &bStop), bStop 은 늘 거짓) = 포터 BA `orb3ba::mlba`.
//! - 본질 그래프(합치기 판 5 인자) = 포터 CV `build_eg_merge` · `optimize_essential_graph`(20 번) + 원본 마무리 순서:
//!   안 고정 키프레임마다(나쁨 빼고, 목록 순) mTcwBefMerge · mTwcBefMerge = 지금 자세, 자세 = SE3d(R, t/s).cast<float>(),
//!   그 뒤 안 고친 점마다(나쁨 빼고) 기준 키프레임이 나쁘면 관측을 지워 가며 새 기준, vpBadPose[기준] 이면 P' = Twr · Tnc_wr⁻¹ · P · UpdateNormalAndDepth.
//! mTcwBefMerge 는 공통 KeyFrame 칸이 아니라 여기 표로 든다: 맞춤 창(고정 고침) 키프레임은 MergeLocal 이 자세를 바꾸기 전 값(합치기 들어올 때 지금 지도 자세).
use std::collections::BTreeMap;

use crate::orb3ba::lba::ObsOrder;
use crate::orb3ba::mlba::{merge_local_ba, KfMpOrder, MergeMarks};
use crate::orb3cv::sim3opt::{build_eg_merge, merge_kf_update, merge_point, optimize_essential_graph, MgInput, MgKf};
use crate::orb3loop::merge::{MergeEssential, MergeLba};
use crate::orb3rs::mapdata::{mp_erase_observation, mp_update_normal_and_depth};
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{Atlas, KfId, MpId, Se3};

pub struct RsMergeLba<'a> {
    pub ext: &'a mut TrackExt,
    pub marks: &'a mut MergeMarks,
}

impl MergeLba for RsMergeLba<'_> {
    fn run(&mut self, a: &mut Atlas, main: KfId, adjust: &[KfId], fixed: &[KfId]) {
        merge_local_ba(main, adjust, fixed, a, self.ext, self.marks, &|| false, ObsOrder(None), KfMpOrder(None));
    }
}

pub struct RsMergeEss {
    /// 합치기 들어올 때 키프레임 자세(Tcw) — 맞춤 창의 mTcwBefMerge
    pub bef: BTreeMap<KfId, Se3>,
}

fn mg_kf(a: &Atlas, k: KfId, tcw_bef: Se3) -> MgKf {
    let kf = &a.kfs[&k];
    let mut children = kf.children.clone();
    children.sort_unstable();
    let mut loop_edges = kf.loop_edges.clone();
    loop_edges.sort_unstable();
    MgKf {
        id: k as u32,
        bad: kf.bad,
        pose: kf.tcw,
        tcw_bef_merge: tcw_bef,
        twc: kf.twc,
        twc_bef_merge: tcw_bef.inverse(),
        parent: kf.parent.map(|p| p as u32),
        children: children.into_iter().map(|c| c as u32).collect(),
        loop_edges: loop_edges.into_iter().map(|c| c as u32).collect(),
        covis100: kf.covisibles_by_weight(100).into_iter().map(|n| (n as u32, a.kfs.get(&n).is_none_or(|x| x.bad))).collect(),
    }
}

impl MergeEssential for RsMergeEss {
    fn run(&mut self, a: &mut Atlas, _cur: KfId, fixed: &[KfId], fixed_corrected: &[KfId], non_fixed: &[KfId], non_corrected_mps: &[MpId]) {
        let inp = MgInput {
            fixed: fixed.iter().map(|&k| mg_kf(a, k, a.kfs[&k].tcw)).collect(),
            fixed_corrected: fixed_corrected.iter().map(|&k| mg_kf(a, k, self.bef.get(&k).copied().unwrap_or(a.kfs[&k].tcw))).collect(),
            non_fixed: non_fixed.iter().map(|&k| mg_kf(a, k, a.kfs[&k].tcw)).collect(),
        };
        let mut g = build_eg_merge(&inp);
        optimize_essential_graph(&mut g.vertices, &g.edges, 20);
        // mTwcBefMerge: 고정 고침은 MergeLocal 이 둔 값, 안 고정은 아래에서
        let mut twc_bef: BTreeMap<KfId, Se3> = fixed_corrected.iter().filter_map(|k| self.bef.get(k).map(|t| (*k, t.inverse()))).collect();
        for &k in non_fixed {
            if a.kfs[&k].bad {
                continue;
            }
            let Ok(i) = g.vertices.binary_search_by_key(&(k as i32), |v| v.id) else { continue };
            let (old, old_twc) = (a.kfs[&k].tcw, a.kfs[&k].twc);
            let (t, _bef, bef_twc, _) = merge_kf_update(&old, &old_twc, &g.vertices[i].est);
            twc_bef.insert(k, bef_twc);
            a.kfs.get_mut(&k).unwrap().set_pose(t);
        }
        for &m in non_corrected_mps {
            if a.mps[&m].bad {
                continue;
            }
            let mut r = a.mps[&m].ref_kf;
            while a.kfs.get(&r).is_some_and(|k| k.bad) {
                mp_erase_observation(a, m, r);
                if a.mps[&m].bad {
                    break;
                }
                r = a.mps[&m].ref_kf;
            }
            if a.mps[&m].bad {
                continue;
            }
            if g.bad_pose.get(&(r as u32)).copied().unwrap_or(false) {
                let Some(tnc) = twc_bef.get(&r) else { continue };
                let twr = a.kfs[&r].twc;
                let p = merge_point(&twr, tnc, &a.mps[&m].pos);
                a.mps.get_mut(&m).unwrap().pos = p;
                mp_update_normal_and_depth(a, m);
            }
        }
    }
}
