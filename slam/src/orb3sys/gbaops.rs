//! 전역 BA 를 루프 흐름(`loopflow::GbaOps`)에 엮은 판: 계산은 포터 BA 의 `orb3ba::gba::global_bundle_adjustment`(실제 루프 뒤 호출 비트 동일),
//! 결과 넣기는 원본 LoopClosing::RunGlobalBundleAdjustment 뒷부분(관성 없음) 그대로 —
//! 지도 첫 키프레임부터 신장 트리를 따라 BA 에 없던 자식 자세를 부모 보정으로 전파(mTcwGBA = Tchildc · 부모 mTcwGBA),
//! 키프레임 자세 바꾸기(mTcwBefGBA 저장), 점은 BA 값 또는 기준 키프레임 보정(Tcw_after⁻¹ · Tcw_before · X), 큰 변화 표시.
//! 원본 set<KeyFrame*>(자식) 주소 순은 id 순.
use super::loopflow::GbaOps;
use super::mapping::MapState;
use super::threads::MapUpdate;
use crate::orb3ba::gba::{global_bundle_adjustment, GbaOut};
use crate::orb3rs::types::{KfId, MapId, MpId, Se3};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

/// GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false)
pub struct RsGba;

impl GbaOps<MapState> for RsGba {
    type Out = GbaOut;

    fn run(&self, map: &MapUpdate<MapState>, map_id: u64, loop_kf: u64, stop: &AtomicBool) -> GbaOut {
        // 원본은 지도 잠금 없이 g2o 를 푼다(그 사이 지도 스레드가 키프레임을 더함) — Rust 판은 잠금 안에서 떠 간 지도로 풂
        let mut st = {
            let g = map.lock();
            (g.atlas.clone(), g.ext.clone())
        };
        let robust = false;
        global_bundle_adjustment(&mut st.0, &st.1, map_id as MapId, 10, &|| stop.load(Ordering::SeqCst), loop_kf as KfId, robust)
    }

    fn apply(&self, st: &mut MapState, out: GbaOut, map_id: u64, _loop_kf: u64) {
        apply_gba(st, &out, map_id as MapId);
    }
}

/// RunGlobalBundleAdjustment 의 결과 넣기(지도 스레드 멈춤 · 지도 잠금 안)
pub fn apply_gba(st: &mut MapState, out: &GbaOut, map_id: MapId) {
    let a = &mut st.atlas;
    let mut gba: BTreeMap<KfId, Se3> = out.kf_gba.iter().copied().collect();
    let mp_gba: BTreeMap<MpId, [f32; 3]> = out.mp_gba.iter().copied().collect();
    let Some(mi) = a.maps.iter().position(|m| m.id == map_id) else { return };
    let mut bef: BTreeMap<KfId, Se3> = BTreeMap::new();
    let mut todo: VecDeque<KfId> = a.maps[mi].origin_kf.into_iter().collect();
    while let Some(k) = todo.pop_front() {
        let (twc, mut childs) = {
            let kf = &a.kfs[&k];
            (kf.twc, kf.children.clone())
        };
        childs.sort_unstable();
        let k_gba = gba.get(&k).copied();
        for c in childs {
            let Some(ck) = a.kfs.get(&c) else { continue };
            if ck.bad {
                continue;
            }
            if !gba.contains_key(&c) {
                // 부모가 BA 에 없었으면 원본은 비어 있는 mTcwGBA 를 곱함 — 첫 키프레임부터 내려오므로 일어나지 않음
                let pg = k_gba.expect("부모 mTcwGBA");
                let tchildc = ck.tcw.mul(&twc);
                gba.insert(c, tchildc.mul(&pg));
            }
            todo.push_back(c);
        }
        let kf = a.kfs.get_mut(&k).unwrap();
        bef.insert(k, kf.tcw);
        if let Some(p) = k_gba {
            kf.set_pose(p);
        }
    }
    let mps = a.maps[mi].mps.clone();
    for m in mps {
        let Some(p) = a.mps.get(&m) else { continue };
        if p.bad {
            continue;
        }
        if let Some(x) = mp_gba.get(&m) {
            a.mps.get_mut(&m).unwrap().pos = *x;
        } else {
            let r = p.ref_kf;
            if !gba.contains_key(&r) {
                continue;
            }
            let Some(b) = bef.get(&r) else { continue };
            let xc = b.mul_point(&p.pos);
            let w = a.kfs[&r].twc.mul_point(&xc);
            a.mps.get_mut(&m).unwrap().pos = w;
        }
    }
    let m = &mut a.maps[mi];
    m.inform_new_big_change();
    m.increase_change_index();
}
