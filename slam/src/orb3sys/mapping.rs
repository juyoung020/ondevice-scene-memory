//! LocalMapping 스레드(Rust 판): [`Mapper`] 대기열 · 멈춤 위에서 `orb3rs::localmap::run_once`(원본 순서, 포터 TRACK)를 돌리고
//! 지역 BA 자리에 `orb3ba::lba::local_bundle_adjustment`(비트 동일 37 호출)를 끼운다.
//!
//! 키프레임 처리 단계(ProcessNewKeyFrame · MapPointCulling · CreateNewMapPoints · SearchInNeighbors · KeyFrameCulling)는
//! [`KfSteps`] 로 받는다 — 포터 TRACK 이 커밋하는 대로 그 함수들을 이어 붙이는 자리.
//!
//! 잠금: 키프레임 하나를 처리하는 동안 지도 갱신 잠금([`MapUpdate`])을 쥔다. 원본은 대부분 단계를 잠금 없이 하고(추적 스레드와 경합)
//! 지역 BA 결과를 넣을 때만 잠그지만, Rust 판은 공유 자료를 잠금 밖에서 바꿀 수 없어 단계 전체를 잠금 안에서 한다.
//! 지역 BA 의 중단 깃발(mbAbortBA)은 [`Mapper::abort_ba`] — 새 키프레임 · 멈춤 요청이 세운다.
use super::threads::{LoopQueue, MapUpdate, Mapper};
use crate::orb3ba::lba;
use crate::orb3rs::localmap::{run_once, LocalMapperOps};
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{Atlas, KfId};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// 지도 갱신 잠금이 지키는 자료
#[derive(Default)]
pub struct MapState {
    pub atlas: Atlas,
    pub ext: TrackExt,
    /// 지금 지도가 관성(IMU 위쪽 간선 포함)인가
    pub inertial: bool,
}

/// 키프레임 처리 단계(포터 TRACK 판이 구현). `kf` = 지금 키프레임(mpCurrentKeyFrame)
pub trait KfSteps: Send {
    fn process_new_keyframe(&mut self, st: &mut MapState, kf: KfId);
    fn map_point_culling(&mut self, st: &mut MapState, kf: KfId);
    fn create_new_map_points(&mut self, st: &mut MapState, kf: KfId);
    fn search_in_neighbors(&mut self, st: &mut MapState, kf: KfId);
    fn keyframe_culling(&mut self, st: &mut MapState, kf: KfId);
}

/// 실행 통계
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappingStats {
    pub keyframes: u64,
    pub local_ba: u64,
    pub local_ba_iters: u64,
}

struct Ops<'a, S: KfSteps> {
    st: &'a mut MapState,
    steps: &'a mut S,
    mapper: &'a Mapper<KfId>,
    kf: KfId,
    stats: &'a mut MappingStats,
    to_loop: bool,
}

impl<S: KfSteps> LocalMapperOps for Ops<'_, S> {
    fn process_new_keyframe(&mut self) {
        self.steps.process_new_keyframe(self.st, self.kf)
    }
    fn map_point_culling(&mut self) {
        self.steps.map_point_culling(self.st, self.kf)
    }
    fn create_new_map_points(&mut self) {
        self.steps.create_new_map_points(self.st, self.kf)
    }
    fn check_new_keyframes(&mut self) -> bool {
        self.mapper.check_new_keyframes()
    }
    fn search_in_neighbors(&mut self) {
        self.steps.search_in_neighbors(self.st, self.kf)
    }
    fn stop_requested(&mut self) -> bool {
        self.mapper.stop_requested()
    }
    fn kfs_in_map(&mut self) -> i32 {
        let a = &self.st.atlas;
        a.maps.get(a.current).map(|m| m.kfs.len() as i32).unwrap_or(0)
    }
    fn local_ba(&mut self) {
        let abort = self.mapper.abort_ba();
        let s = lba::local_bundle_adjustment(self.kf, &mut self.st.atlas, &mut self.st.ext, self.st.inertial, &|| abort.load(Ordering::SeqCst));
        self.stats.local_ba += 1;
        self.stats.local_ba_iters += s.iterations as u64;
    }
    fn keyframe_culling(&mut self) {
        self.steps.keyframe_culling(self.st, self.kf)
    }
    fn insert_in_loop_closer(&mut self) {
        self.to_loop = true;
    }
}

/// LocalMapping::Run: 끝날 때까지 키프레임을 처리해 루프 대기열로 넘김
pub fn run(mapper: &Arc<Mapper<KfId>>, map: &MapUpdate<MapState>, loopq: &LoopQueue<KfId>, steps: &mut impl KfSteps) -> MappingStats {
    let mut stats = MappingStats::default();
    while let Some(kf) = mapper.next() {
        let to_loop = {
            let mut g = map.lock();
            let mut ops = Ops { st: &mut g, steps, mapper, kf, stats: &mut stats, to_loop: false };
            run_once(&mut ops, &|| mapper.abort_ba().load(Ordering::SeqCst));
            ops.to_loop
        };
        stats.keyframes += 1;
        if to_loop {
            loopq.insert(kf);
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Rec(Arc<Mutex<Vec<String>>>);
    impl KfSteps for Rec {
        fn process_new_keyframe(&mut self, _st: &mut MapState, kf: KfId) {
            self.0.lock().unwrap().push(format!("p{kf}"));
        }
        fn map_point_culling(&mut self, _st: &mut MapState, kf: KfId) {
            self.0.lock().unwrap().push(format!("c{kf}"));
        }
        fn create_new_map_points(&mut self, _st: &mut MapState, kf: KfId) {
            self.0.lock().unwrap().push(format!("n{kf}"));
        }
        fn search_in_neighbors(&mut self, _st: &mut MapState, _kf: KfId) {}
        fn keyframe_culling(&mut self, _st: &mut MapState, kf: KfId) {
            self.0.lock().unwrap().push(format!("k{kf}"));
        }
    }

    /// 지도 스레드가 키프레임마다 원본 순서로 단계를 돌리고 루프 대기열로 넘김(지도가 비어 지역 BA 는 안 부름)
    #[test]
    fn mapping_thread_order_and_loop_handoff() {
        let mapper = Mapper::<KfId>::new();
        let q = LoopQueue::<KfId>::new();
        let map = Arc::new(MapUpdate::new(MapState::default()));
        let log = Arc::new(Mutex::new(Vec::new()));
        let (m2, q2, map2, l2) = (Arc::clone(&mapper), Arc::clone(&q), Arc::clone(&map), Arc::clone(&log));
        let t = std::thread::spawn(move || run(&m2, &map2, &q2, &mut Rec(l2)));
        for k in 0..5 {
            mapper.insert(k);
            mapper.wait_idle();
        }
        mapper.finish();
        let s = t.join().unwrap();
        assert_eq!((s.keyframes, s.local_ba), (5, 0));
        let l = log.lock().unwrap().clone();
        assert_eq!(&l[..4], &["p0", "c0", "n0", "k0"]);
        let got: Vec<KfId> = (0..5).map(|_| q.next().unwrap()).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }
}
