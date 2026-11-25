//! 포터 TRACK — ORB-SLAM3 LocalMapping.cc 의 BA 호출을 뺀 판단들(RGB-D, 관성 없음).
//! MapPointCulling · KeyFrameCulling(KeyFrame/MapPoint::SetBadFlag 가 관측 수에 미치는 효과 포함).
//! 지역 BA 는 opt/orb3-rs-ba 몫이라 부르는 자리만: [`LocalMapperOps::local_ba`].
use super::track::{obs_weight, TrackExt};
use super::types::*;

/// LocalMapping::MapPointCulling. `recent` = mlpRecentAddedMapPoints(순서 유지), 돌려줌: SetBadFlag 부른 점들(순서대로).
/// 점 상태(found · visible · first_kf · bad)는 atlas, nObs 는 ext.
pub fn map_point_culling(recent: &mut Vec<MpId>, cur_kf: KfId, monocular: bool, atlas: &mut Atlas, ext: &mut TrackExt) -> Vec<MpId> {
    let th: i32 = if monocular { 2 } else { 3 };
    let mut set_bad = Vec::new();
    let mut keep = Vec::with_capacity(recent.len());
    for &mp in recent.iter() {
        let m = &atlas.mps[&mp];
        let age = (cur_kf as i32).wrapping_sub(m.first_kf as i32);
        if m.bad {
            continue;
        }
        let ratio = m.found as f32 / m.visible as f32;
        if ratio < 0.25 {
            set_bad.push(mp);
            mp_set_bad(mp, atlas, ext);
        } else if age >= 2 && ext.nobs(mp) <= th {
            set_bad.push(mp);
            mp_set_bad(mp, atlas, ext);
        } else if age >= 3 {
        } else {
            keep.push(mp);
        }
    }
    *recent = keep;
    set_bad
}

/// MapPoint::SetBadFlag: 나쁨 표시, 관측한 키프레임 칸 비움(EraseMapPointMatch)
pub fn mp_set_bad(mp: MpId, atlas: &mut Atlas, _ext: &mut TrackExt) {
    let obs: Vec<(KfId, (i32, i32))> = {
        let m = atlas.mps.get_mut(&mp).unwrap();
        m.bad = true;
        std::mem::take(&mut m.obs).into_iter().collect()
    };
    for (kf, (l, _r)) in obs {
        if let Some(k) = atlas.kfs.get_mut(&kf) {
            if l >= 0 && (l as usize) < k.map_points.len() {
                k.map_points[l as usize] = None;
            }
        }
    }
}

/// MapPoint::EraseObservation(kf): 관측 무게 빼고, nObs ≤ 2 면 SetBadFlag
pub fn mp_erase_observation(mp: MpId, kf: KfId, atlas: &mut Atlas, ext: &mut TrackExt) {
    let bad = {
        let m = atlas.mps.get_mut(&mp).unwrap();
        let Some((l, r)) = m.obs.get(&kf).copied() else { return };
        let mut w = 0;
        if l != -1 {
            w += obs_weight(atlas.kfs[&kf].u_right[l as usize]);
        }
        if r != -1 {
            w += 1;
        }
        let m = atlas.mps.get_mut(&mp).unwrap();
        m.obs.remove(&kf);
        let n = ext.mp_nobs.entry(mp).or_insert(0);
        *n -= w;
        *n <= 2
    };
    if bad {
        mp_set_bad(mp, atlas, ext);
    }
}

/// KeyFrame::SetBadFlag 중 컬링 판단에 닿는 효과: 처음 키프레임 · 지우지 말기(mbNotErase)면 그냥 돌아감,
/// 아니면 나쁨 표시와 자기 지도 점들의 관측 지우기. (신장 트리 다시 잇기 · 공시성 갱신은 판단에 안 닿아 통합 때 옮김)
pub fn kf_set_bad(kf: KfId, atlas: &mut Atlas, ext: &mut TrackExt) -> bool {
    let k = &atlas.kfs[&kf];
    if ext.map_init_kf.get(&k.map) == Some(&kf) {
        return false;
    }
    if ext.kf_not_erase.contains(&kf) {
        return false;
    }
    let mps: Vec<MpId> = k.map_points.iter().flatten().copied().collect();
    for mp in mps {
        mp_erase_observation(mp, kf, atlas, ext);
    }
    atlas.kfs.get_mut(&kf).unwrap().bad = true;
    true
}

/// LocalMapping::KeyFrameCulling(관성 없음). `local` = 지금 키프레임의 공시성 순서 목록(UpdateBestCovisibles 뒤).
/// `abort_ba()` = mbAbortBA(반복 중 바뀔 수 있음). 돌려줌: SetBadFlag 를 부른 키프레임들.
pub fn keyframe_culling(
    local: &[KfId],
    monocular: bool,
    th_depth: &dyn Fn(KfId) -> f32,
    abort_ba: &dyn Fn() -> bool,
    atlas: &mut Atlas,
    ext: &mut TrackExt,
) -> Vec<KfId> {
    let redundant_th: f32 = 0.9;
    let mut count = 0;
    let mut called = Vec::new();
    for &kf in local {
        count += 1;
        let k = &atlas.kfs[&kf];
        let is_init = ext.map_init_kf.get(&k.map) == Some(&kf);
        if !(is_init || k.bad) {
            let thd = th_depth(kf);
            let th_obs = 3;
            let mut n_red = 0;
            let mut n_mps = 0;
            for (i, slot) in k.map_points.iter().enumerate() {
                let Some(mp) = *slot else { continue };
                let m = &atlas.mps[&mp];
                if m.bad {
                    continue;
                }
                if !monocular && (k.depth[i] > thd || k.depth[i] < 0.0) {
                    continue;
                }
                n_mps += 1;
                if ext.nobs(mp) > th_obs {
                    let lvl = k.keys[i].octave;
                    let mut n = 0;
                    for (&okf, &(l, _r)) in m.obs.iter() {
                        if okf == kf {
                            continue;
                        }
                        let lvli = atlas.kfs[&okf].keys[l as usize].octave;
                        if lvli <= lvl + 1 {
                            n += 1;
                            if n > th_obs {
                                break;
                            }
                        }
                    }
                    if n > th_obs {
                        n_red += 1;
                    }
                }
            }
            if (n_red as f32) > redundant_th * n_mps as f32 {
                called.push(kf);
                kf_set_bad(kf, atlas, ext);
            }
        }
        if (count > 20 && abort_ba()) || count > 100 {
            break;
        }
    }
    called
}

/// LocalMapping::Run 한 바퀴의 차례(RGB-D, 관성 없음). 하위 단계는 다른 몫 · 원본.
pub trait LocalMapperOps {
    fn process_new_keyframe(&mut self);
    fn map_point_culling(&mut self);
    fn create_new_map_points(&mut self);
    fn check_new_keyframes(&mut self) -> bool;
    fn search_in_neighbors(&mut self);
    fn stop_requested(&mut self) -> bool;
    fn kfs_in_map(&mut self) -> i32;
    /// Optimizer::LocalBundleAdjustment(opt/orb3-rs-ba)
    fn local_ba(&mut self);
    fn keyframe_culling(&mut self);
    fn insert_in_loop_closer(&mut self);
}

/// LocalMapping::Run 의 키프레임 하나 처리(관성 · IMU 초기화 갈래 뺌)
pub fn run_once(ops: &mut dyn LocalMapperOps, abort_ba: &dyn Fn() -> bool) {
    ops.process_new_keyframe();
    ops.map_point_culling();
    ops.create_new_map_points();
    if !ops.check_new_keyframes() {
        ops.search_in_neighbors();
    }
    if !ops.check_new_keyframes() && !ops.stop_requested() {
        if ops.kfs_in_map() > 2 {
            ops.local_ba();
        }
        let _ = abort_ba;
        ops.keyframe_culling();
    }
    ops.insert_in_loop_closer();
}
