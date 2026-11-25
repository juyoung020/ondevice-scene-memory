//! LOOP 몫에서만 쓰는 지도 갱신: MapPoint::UpdateNormalAndDepth · GetConnectedKeyFrames 집합.
//! 공시성(UpdateConnections · AddConnection · AddLoopEdge)은 `orb3rs::mapdata` 것을 쓴다.
//! 원본 std::map<KeyFrame*> 순회는 번호(mnId) 순(번호 순 비교자 OrderById — 대조는 그 패치를 넣은 기준 빌드 `~/opt/orb3_refid`).
use std::collections::{BTreeMap, BTreeSet};

use crate::orb3rs::{add, norm, sub, KeyFrame, KfId, MapPoint, V3};

/// 원본 GetConnectedKeyFrames(집합)
pub fn connected_set(kf: &KeyFrame) -> BTreeSet<KfId> {
    kf.conn_weights.keys().copied().collect()
}

/// MapPoint::UpdateNormalAndDepth. 원본은 기준 키프레임이 관측에 없으면 observations[pRefKF] 가 (0, 0) 을 넣어 0 번 점의 층을 씀
pub fn update_normal_and_depth(mp: &mut MapPoint, kfs: &BTreeMap<KfId, KeyFrame>) {
    if mp.bad || mp.obs.is_empty() {
        return;
    }
    let pos = mp.pos;
    let mut normal: V3 = [0.0; 3];
    let mut n = 0i32;
    for (k, &(l, r)) in &mp.obs {
        let c = kfs[k].camera_center();
        for idx in [l, r] {
            if idx != -1 {
                let ni = sub(&pos, &c);
                let d = norm(&ni);
                normal = add(&normal, &[ni[0] / d, ni[1] / d, ni[2] / d]);
                n += 1;
            }
        }
    }
    let rk = &kfs[&mp.ref_kf];
    let pc = sub(&pos, &rk.camera_center());
    let dist = norm(&pc);
    let left = mp.obs.get(&mp.ref_kf).map_or(0, |o| o.0);
    let level = rk.keys[left as usize].octave as usize;
    let sc = &rk.ctx.pyr.scale;
    let lsf = sc[level];
    mp.max_dist = dist * lsf;
    mp.min_dist = mp.max_dist / sc[sc.len() - 1];
    let nf = n as f32;
    mp.normal = [normal[0] / nf, normal[1] / nf, normal[2] / nf];
}
