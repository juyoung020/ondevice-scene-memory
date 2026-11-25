//! `Optimizer::GlobalBundleAdjustemnt` → `BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust)`
//! (RGB-D, 오른쪽 카메라 없음)를 `orb3rs::types::Atlas` 위에서.
//! - 정점: 나쁘지 않은 키프레임(첫 키프레임만 고정), 점은 나쁘지 않은 것 · id = 점 번호 + 최대 키프레임 번호 + 1.
//!   관측이 하나도 안 걸린 점은 빼고(removeVertex) 되써 넣지도 않는다.
//! - 간선: 점 순서 · 관측 순서대로(관측 순회는 `ObsOrder`, 기본 키프레임 번호 순), 왼쪽 칸 없음(-1)은 간선 없이 nEdges 만 셈.
//!   Huber δ 는 √5.99 · √7.815(float), bRobust 가 거짓이면 커널 없음.
//! - 되써 넣기: nLoopKF 가 지도의 원점 키프레임이면 바로 자세 · 점 위치 · UpdateNormalAndDepth,
//!   아니면 mTcwGBA · mPosGBA · mnBAGlobalForKF 에 해당하는 값을 돌려준다(루프 스레드가 나중에 옮김 — `GbaOut`).
//!   원본의 dist > 1 갈래는 세기만 하고 쓰지 않아 뺐다.
use std::collections::HashMap;

use super::edges::{Edge, Kind};
use super::lba::{mp_update_normal_and_depth, ObsOrder};
use super::se3::normalize_rot;
use super::solver::optimize_stop;
use super::{Est, Problem, Vertex};
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{Atlas, KfId, MapId, MpId, Se3};

/// 전역 BA 결과. `direct` 면 이미 지도에 썼고 아래 목록은 비어 있다.
#[derive(Clone, Debug, Default)]
pub struct GbaOut {
    pub direct: bool,
    pub iterations: usize,
    /// (키프레임, mTcwGBA) — mnBAGlobalForKF = loop_kf
    pub kf_gba: Vec<(KfId, Se3)>,
    /// (점, mPosGBA) — mnBAGlobalForKF = loop_kf
    pub mp_gba: Vec<(MpId, [f32; 3])>,
}

pub struct GbaGraph {
    pub problem: Problem,
    /// 정점 번호 → 키프레임 (나쁘지 않은 것, vpKFs 순서)
    pub kfs: Vec<(KfId, usize)>,
    /// 들어간 점과 그 정점 번호(vpMP 순서)
    pub mps: Vec<(MpId, usize)>,
}

/// Sophus::SE3d(Quaterniond, t).cast<float>(): double 에서 정규화(÷ norm) → float 로 → SE3f 생성자가 다시 정규화
fn se3d_cast(q: &[f64; 4], t: &[f64; 3]) -> Se3 {
    let n = ((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])).sqrt();
    let qn = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    Se3::new(qn.map(|v| v as f32), t.map(|v| v as f32))
}

pub fn build_graph(atlas: &Atlas, init_kf: Option<KfId>, kfs: &[KfId], mps: &[MpId], robust: bool, ord: ObsOrder) -> GbaGraph {
    let mut p = Problem::default();
    let mut vidx: HashMap<KfId, usize> = HashMap::new();
    let mut out_kfs = Vec::new();
    let mut max_kf_id = 0u64;
    for &k in kfs {
        let kk = &atlas.kfs[&k];
        if kk.bad {
            continue;
        }
        let s = &kk.tcw;
        let mut q = s.q.map(|v| v as f64);
        normalize_rot(&mut q);
        vidx.insert(k, p.vertices.len());
        out_kfs.push((k, p.vertices.len()));
        p.vertices.push(Vertex { id: k as i32, fixed: Some(k) == init_kf, est: Est::Pose { q, t: s.t.map(|v| v as f64) } });
        max_kf_id = max_kf_id.max(k);
    }
    let th2 = (5.99f32).sqrt() as f64;
    let th3 = (7.815f32).sqrt() as f64;
    let mut out_mps = Vec::new();
    for &mp in mps {
        let m = &atlas.mps[&mp];
        if m.bad {
            continue;
        }
        let pi = p.vertices.len();
        p.vertices.push(Vertex { id: (mp + max_kf_id + 1) as i32, fixed: false, est: Est::Point(m.pos.map(|v| v as f64)) });
        let mut n_edges = 0;
        for (k, (left, _)) in ord.obs(mp, atlas) {
            let Some(kk) = atlas.kfs.get(&k) else { continue };
            if kk.bad || k > max_kf_id {
                continue;
            }
            let Some(&vk) = vidx.get(&k) else { continue };
            n_edges += 1;
            if left == -1 {
                continue;
            }
            let kp = &kk.keys[left as usize];
            let inv = kk.ctx.pyr.inv_sigma2[kp.octave as usize] as f64;
            let ur = kk.u_right[left as usize];
            let c = &kk.ctx.cam;
            let mut info = [0.0; 9];
            let e = if ur < 0.0 {
                info[0] = inv;
                info[3] = inv;
                Edge { kind: Kind::Mono { cam: [c.fx, c.fy, c.cx, c.cy] }, v0: pi, v1: vk, meas: [kp.x as f64, kp.y as f64, 0.0], info, delta: robust.then_some(th2) }
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
                    delta: robust.then_some(th3),
                }
            };
            p.edges.push(e);
        }
        if n_edges == 0 {
            p.vertices.pop();
        } else {
            out_mps.push((mp, pi));
        }
    }
    GbaGraph { problem: p, kfs: out_kfs, mps: out_mps }
}

/// GlobalBundleAdjustemnt(pMap, n_iter, stop, loop_kf, robust). 키프레임 · 점 순서는 지도의 GetAllKeyFrames · GetAllMapPoints
/// (원본 std::set 포인터 순 자리 — 여기는 번호 순).
pub fn global_bundle_adjustment(atlas: &mut Atlas, ext: &TrackExt, map: MapId, n_iter: usize, stop: &dyn Fn() -> bool, loop_kf: KfId, robust: bool) -> GbaOut {
    let m = atlas.maps.iter().find(|m| m.id == map).expect("지도");
    let mut kfs = m.kfs.clone();
    kfs.sort_unstable();
    let mut mps = m.mps.clone();
    mps.sort_unstable();
    let init = ext.map_init_kf.get(&map).copied();
    let origin = m.origin_kf;
    bundle_adjustment_ord(atlas, init, origin, &kfs, &mps, n_iter, stop, loop_kf, robust, ObsOrder::default())
}

#[allow(clippy::too_many_arguments)]
pub fn bundle_adjustment_ord(
    atlas: &mut Atlas,
    init_kf: Option<KfId>,
    origin_kf: Option<KfId>,
    kfs: &[KfId],
    mps: &[MpId],
    n_iter: usize,
    stop: &dyn Fn() -> bool,
    loop_kf: KfId,
    robust: bool,
    ord: ObsOrder,
) -> GbaOut {
    let g = build_graph(atlas, init_kf, kfs, mps, robust, ord);
    let o = optimize_stop(&g.problem, n_iter, stop);
    let direct = Some(loop_kf) == origin_kf;
    let mut out = GbaOut { direct, iterations: o.iterations, ..Default::default() };
    for &(k, vi) in &g.kfs {
        let Est::Pose { q, t } = &o.est[vi] else { unreachable!() };
        if direct {
            let s = Se3::new(q.map(|v| v as f32), t.map(|v| v as f32));
            atlas.kfs.get_mut(&k).unwrap().set_pose(s);
        } else {
            out.kf_gba.push((k, se3d_cast(q, t)));
        }
    }
    for &(mp, vi) in &g.mps {
        if atlas.mps[&mp].bad {
            continue;
        }
        let Est::Point(x) = &o.est[vi] else { unreachable!() };
        let pos = x.map(|v| v as f32);
        if direct {
            atlas.mps.get_mut(&mp).unwrap().pos = pos;
            mp_update_normal_and_depth(mp, atlas, ord);
        } else {
            out.mp_gba.push((mp, pos));
        }
    }
    out
}
