//! LoopClosing 스레드 흐름(Run · CorrectLoop · MergeLocal · RunGlobalBundleAdjustment 의 멈춤 · 잠금 · 전역 BA 순서)을
//! [`threads`](super::threads) 의 [`LoopQueue`] · [`StopToken`] · [`MapUpdate`] 위에 엮은 것.
//!
//! 계산(장소 인식 · Sim3 · 고치기 · 합치기 · 전역 BA)은 [`LoopOps`] · [`GbaOps`] 로 받는다 — 포터 LOOP · CV · BA 의 Rust 판이
//! 들어오는 자리. 이 파일은 원본의 순서만 옮긴다:
//! - Run: 대기열의 키프레임마다 `NewDetectCommonRegions` → 합치기가 먼저(받아들이면 MergeLocal, 같이 잡힌 루프는 버림) → 루프(받아들이면 CorrectLoop).
//!   후보 표지 지우기(SetErase · 일치 수 0)는 받아들였든 아니든 한다.
//! - CorrectLoop: 지도 스레드 멈춤 요청 → 돌던 전역 BA 중단(mbStopGBA · mnFullBAIdx++) → 멈춤 기다림 → 지도 갱신 잠금 안에서 고치기
//!   → 잠금 놓고 전역 BA 시작 → 멈춤 놓기(Release). MergeLocal 은 전역 BA 중단을 멈춤 요청보다 먼저 한다(원본 순서).
//! - 전역 BA: 따로 스레드. 끝나면 번호(mnFullBAIdx)가 그대로이고 중단되지 않았을 때만 지도 스레드를 멈추고 잠금 안에서 결과를 넣는다.
//!
//! 원본과 다른 점(교착 고침, 결과 순서는 같음):
//! - 멈춤은 요청자별 토큰이라 전역 BA 의 지도 갱신과 루프 고치기가 서로의 Release 로 풀리지 않는다.
//! - 원본 `EmptyQueue`(루프 스레드가 지도 스레드의 대기열을 대신 처리)는 하지 않는다: 지도 스레드가 대기열을 다 비운 뒤에야 멈추므로
//!   (`Mapper::next`) 멈춤을 기다린 시점의 지도 상태는 같고, 한 키프레임을 두 스레드가 함께 처리하는 경합이 없어진다.
//! - 원본 전역 BA 는 지도 잠금 없이 살아 있는 지도를 읽는다. 여기서는 [`GbaOps::run`] 이 잠금 안에서 필요한 것을 떠 간 뒤 잠금 밖에서 푼다
//!   (원본에서도 그 사이 지역 지도가 바꾼 값은 결과에 섞이지 않고 덮이므로, 결과가 달라질 수 있는 자리는 원본의 경합 자리뿐).
//! - 전역 BA 를 쥔 채 지도 스레드를 기다리는 자리(mMutexGBA 를 쥐고 멈춤 대기)는 번호 확인만 잠금 안에서 하고 멈춤 대기는 잠금 밖에서.
use super::threads::{LoopQueue, MapUpdate, Mapper};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// NewDetectCommonRegions 결과(mbMergeDetected · mbLoopDetected)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Found {
    pub merge: bool,
    pub looped: bool,
}

/// 루프 스레드 계산(포터 LOOP · CV 판이 구현)
pub trait LoopOps<M>: Send {
    /// 키프레임 하나로 공통 영역 찾기. 지도 갱신 잠금이 필요하면 안에서 잡는다
    fn detect(&mut self, map: &MapUpdate<M>, kf: u64) -> Found;
    /// 합치기 받아들임(관성 지도의 척도 0.9~1.1 검사 · IMU 초기화 전 거절). false 면 Run 이 합치기 후보를 버리고 다음 키프레임으로(`continue`)
    fn merge_accept(&mut self, map: &M) -> MergeVerdict;
    /// MergeLocal(지도 스레드 멈춤 · 지도 잠금 안). 돌려준 값이 Some(지도 id) 면 그 지도로 전역 BA
    fn merge_local(&mut self, map: &mut M) -> Option<u64>;
    /// 루프 받아들임(관성 지도의 기울기 검사, mg2oLoopScw 고침 포함)
    fn loop_accept(&mut self, map: &M) -> bool;
    /// CorrectLoop 의 계산 부분(지도 스레드 멈춤 · 지도 잠금 안). Some(지도 id) 면 전역 BA
    fn correct_loop(&mut self, map: &mut M) -> Option<u64>;
    /// 합치기 후보 표지 지우기(SetErase · 일치 수 · 짝 목록 비우기)
    fn clear_merge(&mut self, map: &M);
    /// 루프 후보 표지 지우기
    fn clear_loop(&mut self, map: &M);
    /// 이번 키프레임을 mpLastCurrentKF 로
    fn done(&mut self, kf: u64);
}

/// 합치기 받아들임 결과
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeVerdict {
    /// MergeLocal 진행
    Merge,
    /// IMU 초기화 전: 합치지 않고 표지만 지움(같이 잡힌 루프도 지움)
    Skip,
    /// 척도 나쁨: 합치기 표지 지우고 이 키프레임 끝(`continue`, 루프 표지는 남음)
    Abort,
}

/// 전역 BA(포터 BA 판이 구현)
pub trait GbaOps<M>: Send + Sync + 'static {
    /// 결과(키프레임 · 점 새 값)
    type Out: Send + 'static;
    /// 지도 `map_id` 를 루프 키프레임 `loop_kf` 기준으로 최적화. 잠금은 안에서 필요한 동안만. `stop` 이 서면 일찍 끝내도 됨
    fn run(&self, map: &MapUpdate<M>, map_id: u64, loop_kf: u64, stop: &AtomicBool) -> Self::Out;
    /// 결과 넣기(지도 스레드 멈춤 · 지도 잠금 안): 자식 따라 자세 전파 · 점 고치기. IMU 초기화가 그 사이 됐으면 버릴지도 여기서
    fn apply(&self, map: &mut M, out: Self::Out, map_id: u64, loop_kf: u64);
}

struct GbaState {
    /// mnFullBAIdx
    idx: AtomicU64,
    /// mbStopGBA(이 번호의 BA 용)
    stop: Mutex<Arc<AtomicBool>>,
    /// mbRunningGBA · mbFinishedGBA: 살아 있는 BA 스레드 수
    running: AtomicU64,
    /// 통계: 결과를 넣은 수 · 버린 수
    applied: AtomicU64,
    dropped: AtomicU64,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

/// 루프 스레드 + 전역 BA 스레드
pub struct LoopFlow<M: Send + 'static, G: GbaOps<M>> {
    map: Arc<MapUpdate<M>>,
    mapper: Arc<Mapper<u64>>,
    queue: Arc<LoopQueue<u64>>,
    gba: Arc<G>,
    st: Arc<GbaState>,
}

/// 실행 통계
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopStats {
    pub loops: u64,
    pub merges: u64,
    pub gba_started: u64,
    pub gba_applied: u64,
    pub gba_dropped: u64,
}

impl<M: Send + 'static, G: GbaOps<M>> LoopFlow<M, G> {
    pub fn new(map: Arc<MapUpdate<M>>, mapper: Arc<Mapper<u64>>, queue: Arc<LoopQueue<u64>>, gba: G) -> Self {
        let st = Arc::new(GbaState {
            idx: AtomicU64::new(0),
            stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            running: AtomicU64::new(0),
            applied: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            handles: Mutex::new(Vec::new()),
        });
        LoopFlow { map, mapper, queue, gba: Arc::new(gba), st }
    }

    /// 전역 BA 가 돌고 있나(isRunningGBA)
    pub fn gba_running(&self) -> bool {
        self.st.running.load(Ordering::SeqCst) > 0
    }

    /// 돌던 전역 BA 를 버림(mbStopGBA = true · mnFullBAIdx++ · detach)
    fn abort_gba(&self) {
        if self.gba_running() {
            self.st.stop.lock().unwrap_or_else(|e| e.into_inner()).store(true, Ordering::SeqCst);
            self.st.idx.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn start_gba(&self, map_id: u64, loop_kf: u64) {
        let stop = Arc::new(AtomicBool::new(false));
        *self.st.stop.lock().unwrap_or_else(|e| e.into_inner()) = Arc::clone(&stop);
        let idx = self.st.idx.load(Ordering::SeqCst);
        self.st.running.fetch_add(1, Ordering::SeqCst);
        let (map, mapper, gba, st) = (Arc::clone(&self.map), Arc::clone(&self.mapper), Arc::clone(&self.gba), Arc::clone(&self.st));
        let h = std::thread::Builder::new()
            .name("orb3-gba".into())
            .spawn(move || {
                let out = gba.run(&map, map_id, loop_kf, &stop);
                // 원본은 mMutexGBA 안에서 번호 · 중단을 보고 멈춤을 기다림 — 여기선 멈춘 뒤 번호를 다시 본다
                if st.idx.load(Ordering::SeqCst) == idx && !stop.load(Ordering::SeqCst) {
                    let tok = mapper.request_stop();
                    // 지도 스레드가 끝났으면(isFinished) 원본도 그대로 넣는다
                    tok.wait();
                    if st.idx.load(Ordering::SeqCst) == idx && !stop.load(Ordering::SeqCst) {
                        let mut g = map.lock();
                        gba.apply(&mut g, out, map_id, loop_kf);
                        st.applied.fetch_add(1, Ordering::SeqCst);
                    } else {
                        st.dropped.fetch_add(1, Ordering::SeqCst);
                    }
                    drop(tok);
                } else {
                    st.dropped.fetch_add(1, Ordering::SeqCst);
                }
                st.running.fetch_sub(1, Ordering::SeqCst);
            })
            .expect("전역 BA 스레드");
        self.st.handles.lock().unwrap_or_else(|e| e.into_inner()).push(h);
    }

    /// CorrectLoop 의 멈춤 · 잠금 순서
    fn correct_loop(&self, ops: &mut impl LoopOps<M>, kf: u64) {
        let tok = self.mapper.request_stop();
        self.abort_gba();
        if !tok.wait() {
            return; // 지도 스레드가 끝남(종료 중)
        }
        let gba = {
            let mut g = self.map.lock();
            ops.correct_loop(&mut g)
        };
        if let Some(map_id) = gba {
            self.start_gba(map_id, kf);
        }
        drop(tok);
    }

    /// MergeLocal 의 멈춤 · 잠금 순서(전역 BA 중단이 멈춤 요청보다 먼저)
    fn merge_local(&self, ops: &mut impl LoopOps<M>, kf: u64) {
        self.abort_gba();
        let tok = self.mapper.request_stop();
        if !tok.wait() {
            return;
        }
        let gba = {
            let mut g = self.map.lock();
            ops.merge_local(&mut g)
        };
        drop(tok);
        if let Some(map_id) = gba {
            self.start_gba(map_id, kf);
        }
    }

    /// LoopClosing::Run(대기열이 끝날 때까지). 끝나면 돌던 전역 BA 를 기다림
    pub fn run(&self, mut ops: impl LoopOps<M>) -> LoopStats {
        let mut s = LoopStats::default();
        while let Some(kf) = self.queue.next() {
            let f = ops.detect(&self.map, kf);
            let mut looped = f.looped;
            if f.merge {
                let v = ops.merge_accept(&self.map.lock());
                if v == MergeVerdict::Abort {
                    ops.clear_merge(&self.map.lock());
                    continue;
                }
                if v == MergeVerdict::Merge {
                    let before = self.st.handles.lock().unwrap_or_else(|e| e.into_inner()).len();
                    self.merge_local(&mut ops, kf);
                    s.merges += 1;
                    s.gba_started += (self.st.handles.lock().unwrap_or_else(|e| e.into_inner()).len() - before) as u64;
                }
                ops.clear_merge(&self.map.lock());
                if looped {
                    ops.clear_loop(&self.map.lock());
                    looped = false;
                }
            }
            if looped {
                if ops.loop_accept(&self.map.lock()) {
                    let before = self.st.handles.lock().unwrap_or_else(|e| e.into_inner()).len();
                    self.correct_loop(&mut ops, kf);
                    s.loops += 1;
                    s.gba_started += (self.st.handles.lock().unwrap_or_else(|e| e.into_inner()).len() - before) as u64;
                }
                ops.clear_loop(&self.map.lock());
            }
            ops.done(kf);
        }
        self.join_gba();
        s.gba_applied = self.st.applied.load(Ordering::SeqCst);
        s.gba_dropped = self.st.dropped.load(Ordering::SeqCst);
        s
    }

    /// 끝: 돌던 전역 BA 를 중단 · 기다림
    fn join_gba(&self) {
        self.abort_gba();
        let hs: Vec<_> = std::mem::take(&mut *self.st.handles.lock().unwrap_or_else(|e| e.into_inner()));
        for h in hs {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    /// 가짜 지도: 기록만
    #[derive(Default)]
    struct FakeMap {
        log: Vec<String>,
        poses: u64,
    }

    struct Ops {
        mapper: Arc<Mapper<u64>>,
        plan: Vec<(u64, Found, MergeVerdict, bool)>,
        gba_map: Option<u64>,
        cur: u64,
    }

    impl Ops {
        fn plan(&self, kf: u64) -> (Found, MergeVerdict, bool) {
            self.plan.iter().find(|p| p.0 == kf).map(|p| (p.1, p.2, p.3)).unwrap_or((Found::default(), MergeVerdict::Merge, true))
        }
    }

    impl LoopOps<FakeMap> for Ops {
        fn detect(&mut self, map: &MapUpdate<FakeMap>, kf: u64) -> Found {
            map.lock().log.push(format!("detect {kf}"));
            self.cur = kf;
            self.plan(kf).0
        }
        fn merge_accept(&mut self, _m: &FakeMap) -> MergeVerdict {
            let kf = self.cur;
            self.plan(kf).1
        }
        fn merge_local(&mut self, m: &mut FakeMap) -> Option<u64> {
            assert!(self.mapper.is_stopped(), "합치기 때 지도 스레드가 멈춰 있어야");
            m.log.push("merge".into());
            self.gba_map
        }
        fn loop_accept(&mut self, _m: &FakeMap) -> bool {
            let kf = self.cur;
            self.plan(kf).2
        }
        fn correct_loop(&mut self, m: &mut FakeMap) -> Option<u64> {
            assert!(self.mapper.is_stopped(), "고칠 때 지도 스레드가 멈춰 있어야");
            m.log.push("correct".into());
            m.poses += 1;
            self.gba_map
        }
        fn clear_merge(&mut self, _m: &FakeMap) {}
        fn clear_loop(&mut self, _m: &FakeMap) {}
        fn done(&mut self, _kf: u64) {}
    }

    struct Gba {
        ms: u64,
        mapper: Arc<Mapper<u64>>,
    }
    impl GbaOps<FakeMap> for Gba {
        type Out = u64;
        fn run(&self, map: &MapUpdate<FakeMap>, _id: u64, kf: u64, stop: &AtomicBool) -> u64 {
            let p = map.lock().poses;
            for _ in 0..self.ms {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
            p * 1000 + kf
        }
        fn apply(&self, m: &mut FakeMap, out: u64, _id: u64, _kf: u64) {
            assert!(self.mapper.is_stopped() || self.mapper.is_finished());
            m.log.push(format!("gba {out}"));
        }
    }

    /// 지도 스레드 흉내: 키프레임을 받아 루프 대기열로
    fn spawn_mapper(mapper: Arc<Mapper<u64>>, q: Arc<LoopQueue<u64>>) -> JoinHandle<u64> {
        thread::spawn(move || {
            let mut n = 0;
            while let Some(k) = mapper.next() {
                thread::sleep(Duration::from_micros(200));
                q.insert(k);
                n += 1;
            }
            n
        })
    }

    fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(secs)).expect("시간 안에 안 끝남(교착)")
    }

    fn run_case(n: u64, plan: Vec<(u64, Found, MergeVerdict, bool)>, gba_ms: u64) -> (LoopStats, Vec<String>) {
        within(20, move || {
            let mapper = Mapper::<u64>::new();
            let q = LoopQueue::<u64>::new();
            let map = Arc::new(MapUpdate::new(FakeMap::default()));
            let mt = spawn_mapper(Arc::clone(&mapper), Arc::clone(&q));
            let flow = Arc::new(LoopFlow::new(Arc::clone(&map), Arc::clone(&mapper), Arc::clone(&q), Gba { ms: gba_ms, mapper: Arc::clone(&mapper) }));
            let ops = Ops { mapper: Arc::clone(&mapper), plan, gba_map: Some(0), cur: 0 };
            let f2 = Arc::clone(&flow);
            let lt = thread::spawn(move || f2.run(ops));
            for k in 0..n {
                // 추적 쪽처럼 멈춤 중엔 키프레임을 만들지 않음(원본 NeedNewKeyFrame; 멈춤이 풀릴 때 대기열은 지워진다)
                while mapper.stop_requested() || mapper.is_stopped() {
                    thread::sleep(Duration::from_micros(50));
                }
                mapper.insert(k);
                thread::sleep(Duration::from_micros(300));
            }
            mapper.wait_idle();
            // 루프 대기열이 빌 때까지
            let mut stable = 0;
            let mut last = usize::MAX;
            while stable < 50 {
                let c = map.lock().log.iter().filter(|l| l.starts_with("detect")).count();
                stable = if c == last && !flow.gba_running() && mapper.queue_len() == 0 { stable + 1 } else { 0 };
                last = c;
                thread::sleep(Duration::from_millis(2));
            }
            // 돌던 BA 가 끝나도록 잠시
            while flow.gba_running() {
                thread::sleep(Duration::from_millis(1));
            }
            q.finish();
            let s = lt.join().unwrap();
            mapper.finish();
            mt.join().unwrap();
            let log = std::mem::take(&mut map.lock().log);
            (s, log)
        })
    }

    #[test]
    fn loop_then_gba_applied() {
        let f = Found { merge: false, looped: true };
        let (s, log) = run_case(30, vec![(10, f, MergeVerdict::Merge, true)], 5);
        assert_eq!(s.loops, 1);
        assert_eq!((s.gba_started, s.gba_applied, s.gba_dropped), (1, 1, 0));
        let c = log.iter().position(|l| l == "correct").unwrap();
        let g = log.iter().position(|l| l.starts_with("gba")).unwrap();
        assert!(c < g);
        assert_eq!(log[g], "gba 1010");
    }

    #[test]
    fn second_loop_drops_running_gba() {
        let f = Found { merge: false, looped: true };
        // 첫 BA 는 길게: 두 번째 루프가 중단시킴
        let (s, log) = run_case(40, vec![(5, f, MergeVerdict::Merge, true), (8, f, MergeVerdict::Merge, true)], 400);
        assert_eq!(s.loops, 2);
        assert_eq!(s.gba_started, 2);
        assert_eq!((s.gba_applied, s.gba_dropped), (1, 1));
        let gs: Vec<_> = log.iter().filter(|l| l.starts_with("gba")).collect();
        assert_eq!(gs, vec!["gba 2008"]);
    }

    #[test]
    fn merge_first_clears_loop_and_rejects() {
        let both = Found { merge: true, looped: true };
        let lp = Found { merge: false, looped: true };
        let (s, log) = run_case(
            30,
            vec![(3, both, MergeVerdict::Merge, true), (6, both, MergeVerdict::Abort, true), (9, both, MergeVerdict::Skip, true), (12, lp, MergeVerdict::Merge, false)],
            1,
        );
        // 3: 합치기(루프는 버림) · 6: 척도 나쁨 → continue(루프도 안 함) · 9: IMU 전 → 둘 다 안 함 · 12: 루프 거절
        assert_eq!((s.merges, s.loops), (1, 0));
        assert_eq!(log.iter().filter(|l| *l == "merge").count(), 1);
        assert_eq!(log.iter().filter(|l| *l == "correct").count(), 0);
    }

    #[test]
    fn many_loops_with_running_mapper_finish() {
        let f = Found { merge: false, looped: true };
        let plan: Vec<_> = (0..200).filter(|k| k % 7 == 3).map(|k| (k, f, MergeVerdict::Merge, true)).collect();
        let (s, _log) = run_case(200, plan, 3);
        assert_eq!(s.loops, (0..200).filter(|k| k % 7 == 3).count() as u64);
        assert_eq!(s.gba_applied + s.gba_dropped, s.gba_started);
    }
}
