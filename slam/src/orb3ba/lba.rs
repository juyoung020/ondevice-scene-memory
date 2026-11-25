//! `Optimizer::LocalBundleAdjustment(KeyFrame*, bool*, Map*, ...)`(RGB-D · 관성 없음 · 우리 중력 항) 를
//! `orb3rs::types::Atlas` 위에서: 지역 · 고정 키프레임 고르기 → g2o 그래프(`Problem`) 만들기 → `optimize_stop(10)` →
//! 이상치 관측 지우기 → 자세 · 점 되써 넣기 · `UpdateNormalAndDepth` · 지도 변경 번호 올리기.
//!
//! 원본의 표시 칸(mnBALocalForKF · mnBAFixedForKF)은 이 함수 안에서만 읽고 쓰므로 지역 집합으로 바꿨다(이웃이면
//! 나쁨 · 다른 지도여도 표시되는 것까지 같게). 관측 순회는 원본 `map<KeyFrame*,…>` 의 포인터 순서 자리 — 여기는
//! 키프레임 번호 순(`MapPoint::obs` BTreeMap). 원본에서 할당 순서 = 번호 순이면 같다(결정성 브랜치 pose-parb 의
//! 번호 순 비교자가 확정되면 원본도 같은 순서).
//! 지도 갱신 잠금(mMutexMapUpdate)은 부르는 쪽(`orb3sys::threads::MapUpdate`)이 되써 넣기 동안 쥔다.
use std::collections::{BTreeSet, HashMap};

use super::edges::{depth_positive, Edge, Kind};
use super::se3::normalize_rot;
use super::solver::optimize_stop;
use super::{Est, Problem, Vertex};
use crate::orb3rs::localmap::mp_erase_observation;
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{norm, sub, Atlas, KfId, MpId, Se3, V3};

/// 원본의 출력 인자(num_fixedKF · num_OptKF · num_MPs · num_edges)와 끝난 모양
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LbaStats {
    pub num_fixed_kf: usize,
    pub num_opt_kf: usize,
    pub num_mps: usize,
    pub num_edges: usize,
    /// 고정 키프레임 0 개로 그만둠
    pub no_fixed: bool,
    /// 최적화 전에 멈춤 깃발이 서 있어 그만둠(되써 넣기 없음)
    pub stopped_before: bool,
    pub iterations: usize,
    /// 지운 (키프레임, 점) 관측
    pub erased: Vec<(KfId, MpId)>,
}

/// 그래프만(최적화 전) — 덤프의 Problem 과 대조할 때도 이것을 쓴다.
pub struct LbaGraph {
    pub problem: Problem,
    pub local_kfs: Vec<KfId>,
    pub fixed_kfs: Vec<KfId>,
    pub local_mps: Vec<MpId>,
    pub max_kf_id: u64,
    /// (간선 번호, 키프레임, 점, 스테레오인가)
    pub obs_edges: Vec<(usize, KfId, MpId, bool)>,
    pub num_fixed_kf: usize,
}

/// Vector3f::normalized(): squaredNorm 은 x² + (y² + z²)
fn normalized(v: &V3) -> V3 {
    let n2 = v[0] * v[0] + (v[1] * v[1] + v[2] * v[2]);
    if n2 > 0.0 {
        let n = n2.sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        *v
    }
}

/// 관측(`map<KeyFrame*,…>`) 순회 순서. 기본은 키프레임 번호 순(`MapPoint::obs`). 시험에서 원본 덤프의 포인터 순서를
/// 넣어 나머지 계산을 비트 비교할 때만 `Some`.
#[derive(Clone, Copy, Default)]
pub struct ObsOrder<'a>(pub Option<&'a HashMap<MpId, Vec<KfId>>>);

impl ObsOrder<'_> {
    pub fn obs(&self, mp: MpId, atlas: &Atlas) -> Vec<(KfId, (i32, i32))> {
        let m = &atlas.mps[&mp];
        match self.0.and_then(|o| o.get(&mp)) {
            Some(ord) => ord.iter().filter_map(|k| m.obs.get(k).map(|v| (*k, *v))).collect(),
            None => m.obs.iter().map(|(k, v)| (*k, *v)).collect(),
        }
    }
}

pub fn build_graph(kf: KfId, atlas: &Atlas, ext: &TrackExt, inertial: bool) -> LbaGraph {
    build_graph_ord(kf, atlas, ext, inertial, ObsOrder::default())
}

pub fn build_graph_ord(kf: KfId, atlas: &Atlas, ext: &TrackExt, inertial: bool, ord: ObsOrder) -> LbaGraph {
    let k0 = &atlas.kfs[&kf];
    let cur_map = k0.map;
    let init_kf = ext.map_init_kf.get(&cur_map).copied();

    let mut local_mark: BTreeSet<KfId> = BTreeSet::new();
    let mut local_kfs = vec![kf];
    local_mark.insert(kf);
    for &(n, _) in &k0.ordered_conn {
        local_mark.insert(n);
        let kn = &atlas.kfs[&n];
        if !kn.bad && kn.map == cur_map {
            local_kfs.push(n);
        }
    }

    let mut num_fixed_kf = 0;
    let mut mp_mark: BTreeSet<MpId> = BTreeSet::new();
    let mut local_mps = Vec::new();
    for &k in &local_kfs {
        if Some(k) == init_kf {
            num_fixed_kf = 1;
        }
        for mp in atlas.kfs[&k].map_points.iter().flatten() {
            let m = &atlas.mps[mp];
            if !m.bad && m.map == cur_map && mp_mark.insert(*mp) {
                local_mps.push(*mp);
            }
        }
    }

    let mut fixed_mark: BTreeSet<KfId> = BTreeSet::new();
    let mut fixed_kfs = Vec::new();
    for mp in &local_mps {
        for (k, _) in ord.obs(*mp, atlas) {
            if !local_mark.contains(&k) && fixed_mark.insert(k) {
                let kk = &atlas.kfs[&k];
                if !kk.bad && kk.map == cur_map {
                    fixed_kfs.push(k);
                }
            }
        }
    }
    num_fixed_kf += fixed_kfs.len();

    let mut p = Problem { vertices: Vec::new(), edges: Vec::new(), lambda_init: if inertial { 100.0 } else { 0.0 } };
    let mut graph = LbaGraph { problem: Problem::default(), local_kfs, fixed_kfs, local_mps, max_kf_id: 0, obs_edges: Vec::new(), num_fixed_kf };
    if num_fixed_kf == 0 {
        return graph;
    }

    // g2o::SE3Quat(q, t) 생성자가 normalizeRotation(double) 을 부름
    let pose = |s: &Se3| {
        let mut q = [s.q[0] as f64, s.q[1] as f64, s.q[2] as f64, s.q[3] as f64];
        normalize_rot(&mut q);
        Est::Pose { q, t: [s.t[0] as f64, s.t[1] as f64, s.t[2] as f64] }
    };
    let mut vidx: HashMap<KfId, usize> = HashMap::new();
    let mut max_kf_id = 0u64;
    for &k in &graph.local_kfs {
        let kk = &atlas.kfs[&k];
        let fixed = Some(k) == init_kf && (kk.up_cam.is_none() || graph.fixed_kfs.is_empty());
        vidx.insert(k, p.vertices.len());
        p.vertices.push(Vertex { id: k as i32, fixed, est: pose(&kk.tcw) });
        max_kf_id = max_kf_id.max(k);
    }
    // 중력 항: 1e4 · I, 강건 커널 없음
    for &k in &graph.local_kfs {
        if let Some(up) = atlas.kfs[&k].up_cam {
            let u = normalized(&up);
            let mut info = [0.0; 9];
            info[0] = 1e4;
            info[4] = 1e4;
            info[8] = 1e4;
            p.edges.push(Edge { kind: Kind::Gravity, v0: vidx[&k], v1: 0, meas: [u[0] as f64, u[1] as f64, u[2] as f64], info, delta: None });
        }
    }
    for &k in &graph.fixed_kfs {
        vidx.insert(k, p.vertices.len());
        p.vertices.push(Vertex { id: k as i32, fixed: true, est: pose(&atlas.kfs[&k].tcw) });
        max_kf_id = max_kf_id.max(k);
    }

    let th_mono = (5.991f64).sqrt() as f32 as f64;
    let th_stereo = (7.815f64).sqrt() as f32 as f64;
    for &mp in &graph.local_mps {
        let m = &atlas.mps[&mp];
        let pi = p.vertices.len();
        p.vertices.push(Vertex {
            id: (mp + max_kf_id + 1) as i32,
            fixed: false,
            est: Est::Point([m.pos[0] as f64, m.pos[1] as f64, m.pos[2] as f64]),
        });
        for (k, (left, _right)) in ord.obs(mp, atlas) {
            let kk = &atlas.kfs[&k];
            if kk.bad || kk.map != cur_map || left == -1 {
                continue;
            }
            let kp = &kk.keys[left as usize];
            let inv = kk.ctx.pyr.inv_sigma2[kp.octave as usize] as f64;
            let ur = kk.u_right[left as usize];
            let c = &kk.ctx.cam;
            let stereo = ur >= 0.0;
            let e = if !stereo {
                let mut info = [0.0; 9];
                info[0] = inv;
                info[3] = inv;
                Edge { kind: Kind::Mono { cam: [c.fx, c.fy, c.cx, c.cy] }, v0: pi, v1: vidx[&k], meas: [kp.x as f64, kp.y as f64, 0.0], info, delta: Some(th_mono) }
            } else {
                let mut info = [0.0; 9];
                info[0] = inv;
                info[4] = inv;
                info[8] = inv;
                Edge {
                    kind: Kind::Stereo { fx: c.fx as f64, fy: c.fy as f64, cx: c.cx as f64, cy: c.cy as f64, bf: c.bf as f64 },
                    v0: pi,
                    v1: vidx[&k],
                    meas: [kp.x as f64, kp.y as f64, ur as f64],
                    info,
                    delta: Some(th_stereo),
                }
            };
            graph.obs_edges.push((p.edges.len(), k, mp, stereo));
            p.edges.push(e);
        }
    }
    graph.max_kf_id = max_kf_id;
    graph.problem = p;
    graph
}

/// MapPoint::UpdateNormalAndDepth(오른쪽 영상 없음)
pub fn mp_update_normal_and_depth(mp: MpId, atlas: &mut Atlas, ord: ObsOrder) {
    let m = &atlas.mps[&mp];
    if m.bad || m.obs.is_empty() {
        return;
    }
    let pos = m.pos;
    let mut normal = [0.0f32; 3];
    let mut n = 0;
    for (k, (l, _)) in ord.obs(mp, atlas) {
        if l != -1 {
            let ni = sub(&pos, &atlas.kfs[&k].camera_center());
            let d = norm(&ni);
            normal = [normal[0] + ni[0] / d, normal[1] + ni[1] / d, normal[2] + ni[2] / d];
            n += 1;
        }
    }
    let rk = &atlas.kfs[&m.ref_kf];
    let dist = norm(&sub(&pos, &rk.camera_center()));
    // observations[pRefKF]: 없으면 기본 튜플 (0, 0)
    let (left, _) = m.obs.get(&m.ref_kf).copied().unwrap_or((0, 0));
    let level = rk.keys[left as usize].octave as usize;
    let scale = &rk.ctx.pyr.scale;
    let max_d = dist * scale[level];
    let min_d = max_d / scale[rk.ctx.pyr.nlevels as usize - 1];
    let nf = n as f32;
    let m = atlas.mps.get_mut(&mp).unwrap();
    m.max_dist = max_d;
    m.min_dist = min_d;
    m.normal = [normal[0] / nf, normal[1] / nf, normal[2] / nf];
}

/// LocalBundleAdjustment 전체. `stop` = pbStopFlag(mbAbortBA). `inertial` = pMap->IsInertial()
pub fn local_bundle_adjustment(kf: KfId, atlas: &mut Atlas, ext: &mut TrackExt, inertial: bool, stop: &dyn Fn() -> bool) -> LbaStats {
    local_bundle_adjustment_ord(kf, atlas, ext, inertial, stop, ObsOrder::default())
}

pub fn local_bundle_adjustment_ord(kf: KfId, atlas: &mut Atlas, ext: &mut TrackExt, inertial: bool, stop: &dyn Fn() -> bool, ord: ObsOrder) -> LbaStats {
    let g = build_graph_ord(kf, atlas, ext, inertial, ord);
    let mut st = LbaStats {
        num_fixed_kf: g.num_fixed_kf,
        num_opt_kf: g.local_kfs.len(),
        num_mps: g.local_mps.len(),
        num_edges: g.obs_edges.len(),
        ..Default::default()
    };
    if g.num_fixed_kf == 0 {
        st.no_fixed = true;
        st.num_opt_kf = 0;
        st.num_mps = 0;
        st.num_edges = 0;
        return st;
    }
    if stop() {
        st.stopped_before = true;
        return st;
    }
    let out = optimize_stop(&g.problem, 10, stop);
    st.iterations = out.iterations;

    // 이상치: 단안 먼저, 다음 스테레오(원본 vpEdgesMono · vpEdgesStereo 순서)
    let p = &g.problem;
    let mut to_erase = Vec::new();
    for want_stereo in [false, true] {
        for &(ei, k, mp, stereo) in &g.obs_edges {
            if stereo != want_stereo || atlas.mps[&mp].bad {
                continue;
            }
            let e = &p.edges[ei];
            let (Est::Point(pt), Est::Pose { q, t }) = (&out.est[e.v0], &out.est[e.v1]) else { unreachable!() };
            let th = if stereo { 7.815 } else { 5.991 };
            if out.chi2[ei] > th || !depth_positive(q, t, pt) {
                to_erase.push((k, mp));
            }
        }
    }

    for &(k, mp) in &to_erase {
        // KeyFrame::EraseMapPointMatch(pMP): 점이 아는 이 키프레임의 칸을 비움
        if let Some(&(l, _)) = atlas.mps[&mp].obs.get(&k) {
            if l != -1 {
                atlas.kfs.get_mut(&k).unwrap().map_points[l as usize] = None;
            }
        }
        // mpRefKF 가 지운 키프레임이면 남은 관측의 첫 번째(원본 begin()) — 곧 나쁨이 되어도 기준은 이 값으로 남는다
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

    for (i, &k) in g.local_kfs.iter().enumerate() {
        let Est::Pose { q, t } = &out.est[i] else { unreachable!() };
        let tiw = Se3::new([q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32], [t[0] as f32, t[1] as f32, t[2] as f32]);
        atlas.kfs.get_mut(&k).unwrap().set_pose(tiw);
    }
    let base = g.local_kfs.len() + g.fixed_kfs.len();
    for (j, &mp) in g.local_mps.iter().enumerate() {
        let Est::Point(x) = &out.est[base + j] else { unreachable!() };
        atlas.mps.get_mut(&mp).unwrap().pos = [x[0] as f32, x[1] as f32, x[2] as f32];
        mp_update_normal_and_depth(mp, atlas, ord);
    }
    let map = atlas.kfs[&kf].map;
    if let Some(m) = atlas.maps.iter_mut().find(|m| m.id == map) {
        m.change_id += 1;
    }
    st
}
