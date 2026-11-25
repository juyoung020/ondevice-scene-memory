//! LoopClosing::MergeLocal(src/LoopClosing.cc) — 관성 없음 · 단안 아님(RGB-D) 갈래. 지금 지도의 둘레(현재 키프레임 공시 창)를
//! 합칠 지도 좌표로 옮겨 붙이고 → 신장 트리 뒤집기 → SearchAndFuse → 연결 갱신 → 맞춤 BA → (지도 스레드 다시 멈춤) → 본질 그래프 →
//! 남은 지도 통째 옮기기 → 합침 가장자리. BA 두 개는 트레잇(BA · CV 몫).
//! 원본과 다른 점: 맞춤 BA 뒤 지도 스레드를 풀었다가 곧 다시 멈추는 자리를 풀지 않음(그 사이 지도 스레드가 돌면 결과가 실행마다 다름 —
//! `orb3sys::loopflow` 가 멈춤을 쥐고 있음). 지도 번호 바꾸기(ChangeId: 합칠 지도가 지금 지도 번호를 받음)는 끝에서 한 번에.
//! std::set<KeyFrame*/MapPoint*> 순회는 번호 순.
use std::collections::{BTreeMap, BTreeSet};

use super::correct::search_and_fuse;
use super::detect::G2oSim3;
use crate::orb3cv::g2osim3::{quat_mul as quatd_mul, quat_rotate as quatd_rotate};
use crate::orb3rs::mapdata::{kf_change_parent, kf_update_connections};
use crate::orb3rs::{Atlas, KfId, MapId, MpId, Se3};

/// Optimizer::LocalBundleAdjustment(pMainKF, vpAdjustKF, vpFixedKF, &bStop)
pub trait MergeLba {
    fn run(&mut self, a: &mut Atlas, main: KfId, adjust: &[KfId], fixed: &[KfId]);
}
/// Optimizer::OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs)
pub trait MergeEssential {
    fn run(&mut self, a: &mut Atlas, cur: KfId, fixed: &[KfId], fixed_corrected: &[KfId], non_fixed: &[KfId], non_corrected_mps: &[MpId]);
}

pub struct MergeIn {
    pub cur: KfId,
    pub matched: KfId,
    /// mg2oMergeScw
    pub scw: G2oSim3,
    /// 들어올 때 전역 BA 가 돌고 있었는지(bRelaunchBA)
    pub gba_running: bool,
}

pub struct MergeOut {
    pub corrected: BTreeMap<KfId, G2oSim3>,
    pub non_corrected: BTreeMap<KfId, G2oSim3>,
    /// 전역 BA 를 다시 띄울지(관성 없음: bRelaunchBA)
    pub relaunch_gba: bool,
    /// 합친 뒤 지도(번호는 지워진 지금 지도의 번호)
    pub map: MapId,
}

type Se3d = ([f64; 4], [f64; 3]);

/// SE3f.cast<double>() → (q 정규화(double), t)
fn se3d(p: &Se3) -> Se3d {
    let s = G2oSim3::from_se3(p);
    (s.r, s.t)
}

/// Sophus SE3d 곱(사원수 Sophus 식 → 정규화, t = ta + Ra·tb)
fn se3d_mul(a: &Se3d, b: &Se3d) -> Se3d {
    let (p, q) = (&a.0, &b.0);
    let (ax, ay, az, aw) = (p[0], p[1], p[2], p[3]);
    let (bx, by, bz, bw) = (q[0], q[1], q[2], q[3]);
    let r = [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ];
    let n = ((r[0] * r[0] + r[2] * r[2]) + (r[1] * r[1] + r[3] * r[3])).sqrt();
    let r = [r[0] / n, r[1] / n, r[2] / n, r[3] / n];
    let rt = quatd_rotate(&a.0, &b.1);
    (r, [a.1[0] + rt[0], a.1[1] + rt[1], a.1[2] + rt[2]])
}

fn grow_window(a: &Atlas, seed: KfId, n: usize) -> BTreeSet<KfId> {
    let mut s: BTreeSet<KfId> = BTreeSet::new();
    s.insert(seed);
    s.extend(a.kfs[&seed].best_covisibility(n as i32));
    let mut tries = 0;
    while s.len() < n && tries < 5 {
        let mut new = Vec::new();
        for k in &s {
            for c in a.kfs[k].best_covisibility((n / 2) as i32) {
                if !a.kfs[&c].bad && !s.contains(&c) {
                    new.push(c);
                }
            }
        }
        s.extend(new);
        tries += 1;
    }
    s
}

/// KeyFrame::GetMapPoints(나쁘지 않은 점 집합)
fn kf_points(a: &Atlas, k: KfId) -> Vec<MpId> {
    a.kfs[&k].map_points.iter().flatten().copied().filter(|m| !a.mps[m].bad).collect()
}

fn map_idx(a: &Atlas, id: MapId) -> usize {
    a.maps.iter().position(|m| m.id == id).expect("지도 없음")
}

/// MergeLocal. `at_release(a, 합칠 지도 번호, 지금 지도 번호)` 는 원본이 끝에서 지도 스레드를 풀기 직전 상태를 보는 자리(대조용, 번호 바꾸기 전)
pub fn merge_local<L: MergeLba, E: MergeEssential>(
    inp: &MergeIn,
    a: &mut Atlas,
    lba: &mut L,
    ess: &mut E,
    at_release: &mut dyn FnMut(&Atlas, MapId, MapId),
) -> MergeOut {
    const N: usize = 25;
    let cur = inp.cur;
    let cur_map = a.kfs[&cur].map;
    let merge_map = a.kfs[&inp.matched].map;
    kf_update_connections(a, cur, false);
    let window = grow_window(a, cur, N);
    let mut window_mps: BTreeSet<MpId> = BTreeSet::new();
    for &k in &window {
        if a.kfs[&k].bad {
            continue;
        }
        window_mps.extend(kf_points(a, k));
    }
    let merge_kfs = grow_window(a, inp.matched, N);
    let mut fuse_set: BTreeSet<MpId> = BTreeSet::new();
    for &k in &merge_kfs {
        fuse_set.extend(kf_points(a, k));
    }
    let fuse_pts: Vec<MpId> = fuse_set.into_iter().collect();

    let twc = se3d(&a.kfs[&cur].twc);
    let non_swc = G2oSim3::new(twc.0, twc.1, 1.0);
    let mut corrected = BTreeMap::new();
    let mut non = BTreeMap::new();
    corrected.insert(cur, inp.scw);
    non.insert(cur, non_swc.inverse());
    let mut tcw_merge: BTreeMap<KfId, Se3> = BTreeMap::new();
    for &k in &window {
        if a.kfs[&k].bad {
            continue;
        }
        let csiw = if k != cur {
            let tiw = se3d(&a.kfs[&k].tcw);
            non.insert(k, G2oSim3::new(tiw.0, tiw.1, 1.0));
            let tic = se3d_mul(&tiw, &twc);
            let c = G2oSim3::new(tic.0, tic.1, 1.0).mul(&inp.scw);
            corrected.insert(k, c);
            c
        } else {
            inp.scw
        };
        tcw_merge.insert(k, csiw.to_se3f());
    }
    let mut mp_merge: BTreeMap<MpId, ([f32; 3], [f32; 3])> = BTreeMap::new();
    window_mps.retain(|m| {
        let p = &a.mps[m];
        if p.bad {
            return false;
        }
        let Some(c) = corrected.get(&p.ref_kf) else { return false };
        let cswi = c.inverse();
        let nsiw = non[&p.ref_kf];
        let pw = cswi.map(&nsiw.map(&p.pos.map(f64::from)));
        let rcor = quatd_mul(&cswi.r, &nsiw.r);
        let nrm = Se3 { q: rcor.map(|v| v as f32), t: [0.0; 3] }.rotate(&p.normal);
        mp_merge.insert(*m, (pw.map(|v| v as f32), nrm));
        true
    });
    let (ci, mi) = (map_idx(a, cur_map), map_idx(a, merge_map));
    for &k in &window {
        if a.kfs[&k].bad {
            continue;
        }
        let kf = a.kfs.get_mut(&k).unwrap();
        kf.set_pose(tcw_merge[&k]);
        kf.map = merge_map;
        a.maps[mi].add_keyframe(k);
        a.maps[ci].erase_keyframe(k);
    }
    for m in &window_mps {
        if a.mps[m].bad {
            continue;
        }
        let (p, n) = mp_merge[m];
        let mp = a.mps.get_mut(m).unwrap();
        mp.pos = p;
        mp.normal = n;
        mp.map = merge_map;
        a.maps[mi].add_map_point(*m);
        a.maps[ci].erase_map_point(*m);
    }
    a.current = mi;
    a.maps[ci].bad = true;
    a.maps[mi].increase_change_index();
    if let Some(o) = a.maps[ci].origin_kf {
        a.kfs.get_mut(&o).unwrap().first_conn_done = true;
    }
    // 신장 트리 뒤집기
    let mut new_child = a.kfs[&cur].parent;
    let mut new_parent = cur;
    kf_change_parent(a, cur, inp.matched);
    while let Some(c) = new_child {
        a.kfs.get_mut(&c).unwrap().erase_child(new_parent);
        let old = a.kfs[&c].parent;
        kf_change_parent(a, c, new_parent);
        new_parent = c;
        new_child = old;
    }
    kf_update_connections(a, inp.matched, false);
    search_and_fuse(&corrected, &fuse_pts, a);
    for &k in &window {
        if !a.kfs[&k].bad {
            kf_update_connections(a, k, false);
        }
    }
    for &k in &merge_kfs {
        if !a.kfs[&k].bad {
            kf_update_connections(a, k, false);
        }
    }
    let wv: Vec<KfId> = window.iter().copied().collect();
    let mv: Vec<KfId> = merge_kfs.iter().copied().collect();
    lba.run(a, cur, &wv, &mv);
    let ckfs = a.maps[ci].kfs.clone();
    let cmps = a.maps[ci].mps.clone();
    if !ckfs.is_empty() {
        ess.run(a, cur, &mv, &wv, &ckfs, &cmps);
        for &k in &ckfs {
            let kf = &a.kfs[&k];
            if kf.bad || kf.map != cur_map {
                continue;
            }
            a.kfs.get_mut(&k).unwrap().map = merge_map;
            a.maps[mi].add_keyframe(k);
            a.maps[ci].erase_keyframe(k);
        }
        for &m in &cmps {
            if a.mps[&m].bad {
                continue;
            }
            a.mps.get_mut(&m).unwrap().map = merge_map;
            a.maps[mi].add_map_point(m);
            a.maps[ci].erase_map_point(m);
        }
    }
    at_release(a, merge_map, cur_map);
    a.kfs.get_mut(&inp.matched).unwrap().add_merge_edge(cur);
    a.kfs.get_mut(&cur).unwrap().add_merge_edge(inp.matched);
    a.maps[ci].increase_change_index();
    a.maps[mi].increase_change_index();
    // ChangeId(지금 지도 번호) · SetMapBad 된 지금 지도는 RemoveBadMaps 로 사라짐
    for k in a.kfs.values_mut() {
        if k.map == merge_map {
            k.map = cur_map;
        }
    }
    for p in a.mps.values_mut() {
        if p.map == merge_map {
            p.map = cur_map;
        }
    }
    a.maps[mi].id = cur_map;
    a.maps.remove(ci);
    a.current = map_idx(a, cur_map);
    MergeOut { corrected, non_corrected: non, relaunch_gba: inp.gba_running, map: cur_map }
}
