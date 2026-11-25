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
        self.steps.process_new_keyframe(self.st, self.kf);
        step_digest(self.st, self.kf, "pnk");
    }
    fn map_point_culling(&mut self) {
        self.steps.map_point_culling(self.st, self.kf);
        step_digest(self.st, self.kf, "mpc");
    }
    fn create_new_map_points(&mut self) {
        self.steps.create_new_map_points(self.st, self.kf);
        step_digest(self.st, self.kf, "cnmp");
    }
    fn check_new_keyframes(&mut self) -> bool {
        self.mapper.check_new_keyframes()
    }
    fn search_in_neighbors(&mut self) {
        self.steps.search_in_neighbors(self.st, self.kf);
        step_digest(self.st, self.kf, "sin");
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
        step_digest(self.st, self.kf, "lba");
    }
    fn keyframe_culling(&mut self) {
        self.steps.keyframe_culling(self.st, self.kf);
        step_digest(self.st, self.kf, "kfc");
    }
    fn insert_in_loop_closer(&mut self) {
        self.to_loop = true;
    }
}

/// 재생 대조용(ORBSYS_STEPS=파일): 단계 뒤 지금 지도 요약 한 줄 — 원본 쪽 `slam/tools/orb3sys_dump/steps_dump.cc` 와 같은 식
/// (나쁘지 않은 키프레임 · 점, id 순, FNV-1a: 자세 · 연결 · KF 점 칸 · 점 위치/법선/거리 · 관측/nObs/기준 KF)
fn steps_out() -> Option<&'static std::sync::Mutex<std::fs::File>> {
    static OUT: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    OUT.get_or_init(|| std::env::var("ORBSYS_STEPS").ok().map(|p| std::sync::Mutex::new(std::fs::File::create(p).expect("ORBSYS_STEPS")))).as_ref()
}

/// 재생 대조용: 단계 파일에 줄 하나(ORBSYS_STEPS 없으면 아무것도 안 함)
pub fn step_text(s: &str) {
    use std::io::Write;
    if let Some(f) = steps_out() {
        let _ = f.lock().unwrap().write_all(s.as_bytes());
    }
}

/// 자세 7 칸(사원수 x y z w, 이동) 비트 16 진
pub fn se3_hex(p: &crate::orb3rs::types::Se3) -> String {
    p.q.iter().chain(p.t.iter()).map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(" ")
}

pub fn step_digest(st: &MapState, kf: KfId, step: &str) {
    step_digest_atlas(&st.atlas, kf, step)
}

pub fn step_digest_atlas(a: &crate::orb3rs::types::Atlas, kf: KfId, step: &str) {
    use std::io::Write;
    let Some(f) = steps_out() else { return };
    struct H(u64);
    impl H {
        fn b(&mut self, x: &[u8]) {
            for c in x {
                self.0 ^= *c as u64;
                self.0 = self.0.wrapping_mul(1099511628211);
            }
        }
        fn u(&mut self, v: u64) { self.b(&v.to_le_bytes()) }
        fn i(&mut self, v: i32) { self.b(&v.to_le_bytes()) }
        fn f(&mut self, v: f32) { self.b(&v.to_le_bytes()) }
    }
    let mid = a.kfs[&kf].map;
    let Some(m) = a.maps.iter().find(|m| m.id == mid) else { return };
    let ks: Vec<KfId> = m.kfs.iter().copied().filter(|k| a.kfs.get(k).is_some_and(|x| !x.bad)).collect();
    let ps: Vec<u64> = m.mps.iter().copied().filter(|p| a.mps.get(p).is_some_and(|x| !x.bad)).collect();
    let mut ks = ks;
    ks.sort_unstable();
    let mut ps = ps;
    ps.sort_unstable();
    let o = 1469598103934665603u64;
    let (mut hk, mut hc, mut hm, mut hp, mut ho) = (H(o), H(o), H(o), H(o), H(o));
    for k in &ks {
        let x = &a.kfs[k];
        hk.u(*k);
        for v in x.tcw.q { hk.f(v) }
        for v in x.tcw.t { hk.f(v) }
        hc.u(*k);
        for (c, w) in &x.conn_weights { hc.u(*c); hc.i(*w) }
        hc.u(x.parent.unwrap_or(u64::MAX));
        for (c, w) in &x.ordered_conn { hc.u(*c); hc.i(*w) }
        hm.u(*k);
        for p in &x.map_points { hm.u(p.unwrap_or(u64::MAX)) }
    }
    for p in &ps {
        let x = &a.mps[p];
        hp.u(*p);
        for v in x.pos { hp.f(v) }
        for v in x.normal { hp.f(v) }
        hp.f(x.min_dist);
        hp.f(x.max_dist);
        ho.u(*p);
        ho.i(x.n_obs);
        ho.u(x.ref_kf);
        for (k, (l, r)) in &x.obs { ho.u(*k); ho.i(*l); ho.i(*r) }
    }
    if std::env::var_os("ORBSYS_STEPS_FULL").is_some() {
        let mut t = String::new();
        for k in &ks {
            let x = &a.kfs[k];
            t += &format!("  K{k} p{} w", x.parent.map_or(-1, |p| p as i64));
            for (c, w) in &x.conn_weights { t += &format!(" {c}:{w}") }
            t += " o";
            for (c, w) in &x.ordered_conn { t += &format!(" {c}:{w}") }
            t += "\n";
        }
        let _ = f.lock().unwrap().write_all(t.as_bytes());
    }
    let line = format!("{kf} {step} {} {} {:016x} {:016x} {:016x} {:016x} {:016x}
", ks.len(), ps.len(), hk.0, hc.0, hm.0, hp.0, ho.0);
    let _ = f.lock().unwrap().write_all(line.as_bytes());
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
        // LoopClosing::InsertKeyFrame: 원본은 0 번 키프레임을 대기열에 넣지 않음(그래서 KeyFrameDatabase 에도 없음)
        if to_loop && kf != 0 {
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
