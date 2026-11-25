//! 지도 합치기용 `Optimizer::LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag)`(RGB-D) 를 Atlas 위에서.
//! - 고정 키프레임 먼저 · 다음 조정 키프레임(나쁨 · 다른 지도는 빼고 표시도 안 함), 점은 각 키프레임의 GetMapPoints()
//!   (원본 std::set 포인터 순 — 기본은 번호 순, 시험은 덤프 순서) 순서로 처음 본 것만.
//! - 간선: 관측 키프레임이 나쁨 · 최대 번호 초과 · 합치기 표시(mnBALocalForMerge) ≠ 주 키프레임 · 그 칸 점 없음이면 빼고,
//!   왼쪽 칸이 -1 인지 보지 않음(원본 그대로). Huber √5.99 · √7.815.
//! - 5 번 → (멈춤 아니면) 이상치 간선 level 1 · 커널 지움(나쁨 점의 간선은 그대로 둠) → level 0 만 10 번
//!   → 이상치 관측 지우기 → 조정 키프레임 자세 · 점 위치 · UpdateNormalAndDepth.
//! - 표시 칸(mnBALocalForMerge)은 호출 사이에 남으므로 부르는 쪽이 `MergeMarks` 를 들고 있는다.
use std::collections::{BTreeSet, HashMap};

use super::edges::{depth_positive, Edge, Kind};
use super::lba::{mp_update_normal_and_depth, ObsOrder};
use super::se3::normalize_rot;
use super::solver::optimize_stop;
use super::{Est, Problem, Vertex};
use crate::orb3rs::localmap::mp_erase_observation;
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{Atlas, KfId, MpId, Se3};

/// mnBALocalForMerge(키프레임 · 점)
#[derive(Clone, Debug, Default)]
pub struct MergeMarks {
    pub kf: HashMap<KfId, u64>,
    pub mp: HashMap<MpId, u64>,
}

#[derive(Clone, Debug, Default)]
pub struct MlbaStats {
    pub stopped_before: bool,
    pub iterations: [usize; 2],
    pub erased: Vec<(KfId, MpId)>,
}

/// 키프레임의 GetMapPoints() 순회 순서(시험에서 원본 포인터 순서를 넣을 때만 Some)
#[derive(Clone, Copy, Default)]
pub struct KfMpOrder<'a>(pub Option<&'a HashMap<KfId, Vec<MpId>>>);

impl KfMpOrder<'_> {
    fn get(&self, k: KfId, atlas: &Atlas) -> Vec<MpId> {
        match self.0.and_then(|o| o.get(&k)) {
            Some(v) => v.clone(),
            None => atlas.kfs[&k].map_points.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect(),
        }
    }
}

pub struct MlbaGraph {
    pub problem: Problem,
    pub adjust: Vec<(KfId, usize)>,
    pub mps: Vec<(MpId, usize)>,
    /// (간선, 키프레임, 점, 스테레오)
    pub obs_edges: Vec<(usize, KfId, MpId, bool)>,
}

pub fn build_graph(main: KfId, adjust: &[KfId], fixed: &[KfId], atlas: &Atlas, marks: &mut MergeMarks, ord: ObsOrder, kmo: KfMpOrder) -> MlbaGraph {
    let cur_map = atlas.kfs[&main].map;
    let mut p = Problem::default();
    let mut vidx: HashMap<KfId, usize> = HashMap::new();
    let mut max_kf = 0u64;
    let mut vp_mps: Vec<MpId> = Vec::new();
    let mut out_adj = Vec::new();
    for (list, is_fixed) in [(fixed, true), (adjust, false)] {
        for &k in list {
            let kk = &atlas.kfs[&k];
            if kk.bad || kk.map != cur_map {
                continue;
            }
            marks.kf.insert(k, main);
            let mut q = kk.tcw.q.map(|v| v as f64);
            normalize_rot(&mut q);
            vidx.insert(k, p.vertices.len());
            if !is_fixed {
                out_adj.push((k, p.vertices.len()));
            }
            p.vertices.push(Vertex { id: k as i32, fixed: is_fixed, est: Est::Pose { q, t: kk.tcw.t.map(|v| v as f64) } });
            max_kf = max_kf.max(k);
            for mp in kmo.get(k, atlas) {
                let m = &atlas.mps[&mp];
                if !m.bad && m.map == cur_map && marks.mp.get(&mp) != Some(&main) {
                    vp_mps.push(mp);
                    marks.mp.insert(mp, main);
                }
            }
        }
    }
    let th2 = (5.99f32).sqrt() as f64;
    let th3 = (7.815f32).sqrt() as f64;
    let mut out_mps = Vec::new();
    let mut obs_edges = Vec::new();
    for &mp in &vp_mps {
        let m = &atlas.mps[&mp];
        if m.bad {
            continue;
        }
        let pi = p.vertices.len();
        p.vertices.push(Vertex { id: (mp + max_kf + 1) as i32, fixed: false, est: Est::Point(m.pos.map(|v| v as f64)) });
        out_mps.push((mp, pi));
        for (k, (left, _)) in ord.obs(mp, atlas) {
            let Some(kk) = atlas.kfs.get(&k) else { continue };
            if kk.bad || k > max_kf || marks.kf.get(&k) != Some(&main) || kk.map_points.get(left as usize).copied().flatten().is_none() {
                continue;
            }
            let Some(&vk) = vidx.get(&k) else { continue };
            let kp = &kk.keys[left as usize];
            let inv = kk.ctx.pyr.inv_sigma2[kp.octave as usize] as f64;
            let ur = kk.u_right[left as usize];
            let c = &kk.ctx.cam;
            let mut info = [0.0; 9];
            let stereo = !(ur < 0.0);
            let e = if !stereo {
                info[0] = inv;
                info[3] = inv;
                Edge { kind: Kind::Mono { cam: [c.fx, c.fy, c.cx, c.cy] }, v0: pi, v1: vk, meas: [kp.x as f64, kp.y as f64, 0.0], info, delta: Some(th2) }
            } else {
                info[0] = inv;
                info[4] = inv;
                info[8] = inv;
                Edge {
                    kind: Kind::Stereo { fx: c.fx as f64, fy: c.fy as f64, cx: c.cx as f64, cy: c.cy as f64, bf: c.bf as f64 },
                    v0: pi,
                    v1: vk,
                    meas: [kp.x as f64, kp.y as f64, ur as f64],
                    info,
                    delta: Some(th3),
                }
            };
            obs_edges.push((p.edges.len(), k, mp, stereo));
            p.edges.push(e);
        }
    }
    MlbaGraph { problem: p, adjust: out_adj, mps: out_mps, obs_edges }
}

#[allow(clippy::too_many_arguments)]
pub fn merge_local_ba(
    main: KfId,
    adjust: &[KfId],
    fixed: &[KfId],
    atlas: &mut Atlas,
    ext: &mut TrackExt,
    marks: &mut MergeMarks,
    stop: &dyn Fn() -> bool,
    ord: ObsOrder,
    kmo: KfMpOrder,
) -> MlbaStats {
    let g = build_graph(main, adjust, fixed, atlas, marks, ord, kmo);
    let mut st = MlbaStats::default();
    if stop() {
        st.stopped_before = true;
        return st;
    }
    let p1 = &g.problem;
    let o1 = optimize_stop(p1, 5, stop);
    st.iterations[0] = o1.iterations;
    let bad_mp = |mp: MpId, atlas: &Atlas| atlas.mps[&mp].bad;
    let pos_of = |est: &[Est], e: &Edge| {
        let (Est::Point(pt), Est::Pose { q, t }) = (&est[e.v0], &est[e.v1]) else { unreachable!() };
        depth_positive(q, t, pt)
    };
    // 마지막 χ² · 정점 값
    let mut chi2 = o1.chi2.clone();
    let mut est = o1.est.clone();
    if !stop() {
        // level 1 로 뺄 간선 · 커널 지우기(나쁨 점의 간선은 그대로)
        let mut keep = vec![true; p1.edges.len()];
        let mut p2 = Problem { vertices: p1.vertices.clone(), edges: Vec::new(), lambda_init: p1.lambda_init };
        for (v, e) in p2.vertices.iter_mut().zip(&o1.est) {
            v.est = e.clone();
        }
        let mut delta2 = p1.edges.iter().map(|e| e.delta).collect::<Vec<_>>();
        for want_stereo in [false, true] {
            for &(ei, _k, mp, stereo) in &g.obs_edges {
                if stereo != want_stereo || bad_mp(mp, atlas) {
                    continue;
                }
                let th = if stereo { 7.815 } else { 5.991 };
                if o1.chi2[ei] > th || !pos_of(&o1.est, &p1.edges[ei]) {
                    keep[ei] = false;
                }
                delta2[ei] = None;
            }
        }
        let mut map2 = Vec::new();
        for (ei, e) in p1.edges.iter().enumerate() {
            if keep[ei] {
                map2.push(ei);
                let mut e2 = e.clone();
                e2.delta = delta2[ei];
                p2.edges.push(e2);
            }
        }
        let o2 = optimize_stop(&p2, 10, stop);
        st.iterations[1] = o2.iterations;
        // level 1 간선의 χ² 는 첫 최적화 값, 나머지는 두 번째의 마지막 계산값
        let act2 = super::solver::Sys::new(&p2).active;
        for (j, &ei) in map2.iter().enumerate() {
            if act2[j] {
                chi2[ei] = o2.chi2[j];
            }
        }
        est = o2.est;
    }
    let mut to_erase = Vec::new();
    for want_stereo in [false, true] {
        for &(ei, k, mp, stereo) in &g.obs_edges {
            if stereo != want_stereo || bad_mp(mp, atlas) {
                continue;
            }
            let th = if stereo { 7.815 } else { 5.991 };
            if chi2[ei] > th || !pos_of(&est, &p1.edges[ei]) {
                to_erase.push((k, mp));
            }
        }
    }
    for &(k, mp) in &to_erase {
        if let Some(&(l, _)) = atlas.mps[&mp].obs.get(&k) {
            if l != -1 {
                atlas.kfs.get_mut(&k).unwrap().map_points[l as usize] = None;
            }
        }
        let next_ref = if ord.0.is_some() && atlas.mps[&mp].ref_kf == k {
            ord.obs(mp, atlas).into_iter().map(|o| o.0).find(|&o| o != k)
        } else {
            None
        };
        mp_erase_observation(mp, k, atlas, ext);
        if let Some(r) = next_ref {
            atlas.mps.get_mut(&mp).unwrap().ref_kf = r;
        }
    }
    st.erased = to_erase;
    let vi: HashMap<KfId, usize> = g.adjust.iter().copied().collect();
    for &k in adjust {
        if atlas.kfs[&k].bad {
            continue;
        }
        let Some(&i) = vi.get(&k) else { continue };
        let Est::Pose { q, t } = &est[i] else { unreachable!() };
        atlas.kfs.get_mut(&k).unwrap().set_pose(Se3::new(q.map(|v| v as f32), t.map(|v| v as f32)));
    }
    for &(mp, i) in &g.mps {
        if atlas.mps[&mp].bad {
            continue;
        }
        let Est::Point(x) = &est[i] else { unreachable!() };
        atlas.mps.get_mut(&mp).unwrap().pos = x.map(|v| v as f32);
        mp_update_normal_and_depth(mp, atlas, ord);
    }
    st
}
