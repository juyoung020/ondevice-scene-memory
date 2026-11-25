//! System 의 세 스레드(추적 = 부르는 스레드 · 지역 지도 · 루프) 사이 신호와 잠금.
//!
//! 원본(LocalMapping · LoopClosing · 패치 Tracking::GraftMap)의 멈춤 신호는 이렇게 생겼다:
//! - `RequestStop` 이 깃발 하나(mbStopRequested)를 세우고, 부른 쪽은 `while(!isStopped()) usleep(1000)` 로 돈다.
//! - `Release` 는 누가 부르든 깃발 · 멈춤을 함께 내린다 — 루프 닫기와 옮겨 붙이기(추적 스레드)가 같이 멈춰 세웠을 때
//!   먼저 끝난 쪽이 다른 쪽이 아직 쓰는 동안 지도 스레드를 다시 돌린다.
//! - 지도 갱신 잠금(Map::mMutexMapUpdate)을 쥔 채로 멈춤을 기다리는 자리가 있다(패치 DoPendingGraft 는 Track() 안,
//!   지역 BA 는 같은 잠금을 잡으러 감) — 기다리는 쪽과 기다림을 받는 쪽이 서로의 잠금을 기다리면 `Local Mapping STOP` 뒤 진행이 없다.
//!
//! 이 판은 구조로 그 셋을 없앤다:
//! - 멈춤은 [`StopToken`] 하나씩(세는 표): 마지막 토큰이 놓일 때만 지도 스레드가 다시 돈다.
//! - 기다림은 모두 Condvar(바쁜 대기 없음, 끝(finish)도 깨움).
//! - 지도 갱신 잠금은 [`MapUpdate`] 로만 잡고, 그것을 쥔 스레드가 멈춤 · 한가함을 기다리면 바로 패닉(교착 대신 그 자리를 알림).
//!
//! 순서(결과가 원본과 같아야 하는 부분)는 원본 그대로: 지도 스레드는 쌓인 키프레임을 먼저 처리하고 대기열이 비었을 때만 멈춘다,
//! `SetNotStop(true)` 는 이미 멈췄으면 거절, 새 키프레임이 오면 지역 BA 중단 깃발.
use std::cell::Cell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

thread_local! {
    /// 이 스레드가 쥔 지도 갱신 잠금 수
    static MAP_HELD: Cell<u32> = const { Cell::new(0) };
    /// 이 스레드가 지도 갱신 잠금을 기다린 시간(ns, 계측용 — `take_map_wait_ns` 로 읽고 비움)
    static MAP_WAIT_NS: Cell<u64> = const { Cell::new(0) };
}

/// 이 스레드가 지금까지 지도 갱신 잠금을 기다린 시간(ns)을 읽고 0 으로
pub fn take_map_wait_ns() -> u64 {
    MAP_WAIT_NS.with(|c| c.replace(0))
}

fn assert_no_map_lock(what: &str) {
    let n = MAP_HELD.with(|c| c.get());
    assert!(n == 0, "지도 갱신 잠금을 쥔 채 {what} 를 기다림(교착 자리)");
}

/// 지도 갱신 잠금(Map::mMutexMapUpdate). 쥔 동안 이 스레드는 멈춤 · 한가함을 기다릴 수 없다
pub struct MapUpdate<T>(Mutex<T>);

pub struct MapGuard<'a, T>(MutexGuard<'a, T>);

impl<T> MapUpdate<T> {
    pub fn new(v: T) -> MapUpdate<T> {
        MapUpdate(Mutex::new(v))
    }
    pub fn lock(&self) -> MapGuard<'_, T> {
        let g = match self.0.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let t0 = std::time::Instant::now();
                let g = self.0.lock().unwrap_or_else(|e| e.into_inner());
                MAP_WAIT_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
                g
            }
        };
        MAP_HELD.with(|c| c.set(c.get() + 1));
        MapGuard(g)
    }
}

impl<T> Drop for MapGuard<'_, T> {
    fn drop(&mut self) {
        MAP_HELD.with(|c| c.set(c.get() - 1));
    }
}

impl<T> Deref for MapGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for MapGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

struct MState<K> {
    queue: VecDeque<K>,
    /// 지도 스레드가 키프레임 하나를 처리 중
    busy: bool,
    /// mbAcceptKeyFrames
    accept: bool,
    /// 살아 있는 멈춤 토큰 수(mbStopRequested 대신)
    holders: u32,
    /// mbStopped
    stopped: bool,
    /// mbNotStop
    not_stop: bool,
    finish: bool,
    finished: bool,
    /// mbResetRequestedActiveMap
    reset_req: bool,
    /// 지도 스레드가 처리한 대기열 비우기 수(최근 점 목록 비우기 표지)
    reset_gen: u64,
}

/// 지역 지도 스레드의 대기열 · 멈춤(LocalMapping 의 mlNewKeyFrames · mbStopRequested · mbStopped · mbNotStop · mbAbortBA)
pub struct Mapper<K> {
    st: Mutex<MState<K>>,
    cv: Condvar,
    abort_ba: AtomicBool,
}

impl<K> Mapper<K> {
    pub fn new() -> Arc<Mapper<K>> {
        Arc::new(Mapper {
            st: Mutex::new(MState { queue: VecDeque::new(), busy: false, accept: true, holders: 0, stopped: false, not_stop: false, finish: false, finished: false, reset_req: false, reset_gen: 0 }),
            cv: Condvar::new(),
            abort_ba: AtomicBool::new(false),
        })
    }

    fn st(&self) -> MutexGuard<'_, MState<K>> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// InsertKeyFrame
    pub fn insert(&self, k: K) {
        let mut s = self.st();
        s.queue.push_back(k);
        self.abort_ba.store(true, Ordering::SeqCst);
        self.cv.notify_all();
    }

    /// AcceptKeyFrames
    pub fn accept_keyframes(&self) -> bool {
        self.st().accept
    }

    /// KeyframesInQueue
    pub fn queue_len(&self) -> usize {
        self.st().queue.len()
    }

    /// stopRequested
    pub fn stop_requested(&self) -> bool {
        self.st().holders > 0
    }

    /// isStopped
    pub fn is_stopped(&self) -> bool {
        self.st().stopped
    }

    /// isFinished
    pub fn is_finished(&self) -> bool {
        self.st().finished
    }

    /// SetNotStop: 이미 멈췄으면 거절(원본과 같음)
    pub fn set_not_stop(&self, flag: bool) -> bool {
        let mut s = self.st();
        if flag && s.stopped {
            return false;
        }
        s.not_stop = flag;
        self.cv.notify_all();
        true
    }

    /// 지역 BA 중단 깃발(mbAbortBA). BA 가 이것을 본다
    pub fn abort_ba(&self) -> &AtomicBool {
        &self.abort_ba
    }

    /// RequestStop. 토큰이 살아 있는 동안 멈춰 있다(여럿이 함께 쥘 수 있음)
    pub fn request_stop(self: &Arc<Self>) -> StopToken<K> {
        let mut s = self.st();
        s.holders += 1;
        self.abort_ba.store(true, Ordering::SeqCst);
        self.cv.notify_all();
        StopToken { m: Arc::clone(self) }
    }

    /// 추적 쪽 WaitMappingIdle: 대기열이 비고 처리 중이 아니거나, 멈췄거나, 끝날 때까지
    pub fn wait_idle(&self) {
        assert_no_map_lock("지도 스레드 한가함");
        let mut s = self.st();
        while !((s.queue.is_empty() && !s.busy) || s.stopped || s.finished) {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// RequestResetActiveMap: 지도 스레드가 지금 키프레임을 마친 뒤 대기열을 비울 때까지 기다림(끝났으면 바로)
    pub fn reset_active_map(&self) {
        assert_no_map_lock("지도 스레드 비우기");
        let mut s = self.st();
        s.reset_req = true;
        self.cv.notify_all();
        while s.reset_req && !s.finished {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// 지도 스레드가 처리한 대기열 비우기 수
    pub fn reset_gen(&self) -> u64 {
        self.st().reset_gen
    }

    /// RequestFinish: 지도 스레드를 끝냄(멈춤 중이어도 깨움)
    pub fn finish(&self) {
        let mut s = self.st();
        s.finish = true;
        self.cv.notify_all();
    }

    // ---- 지도 스레드 쪽 ----

    /// LocalMapping::Run 의 바깥 반복: 다음 키프레임(없으면 멈춤 요청을 받거나 새 키프레임 · 끝을 기다림). 끝이면 None
    pub fn next(&self) -> Option<K> {
        assert_no_map_lock("새 키프레임");
        let mut s = self.st();
        s.busy = false;
        loop {
            if s.reset_req {
                // ResetIfRequested(지금 지도): 쌓인 키프레임을 버림(최근 점 목록은 reset_gen 을 보고 단계 쪽이 비움)
                s.queue.clear();
                s.reset_req = false;
                s.reset_gen += 1;
                self.cv.notify_all();
            }
            if s.finish {
                s.finished = true;
                s.stopped = false;
                self.cv.notify_all();
                return None;
            }
            if let Some(k) = s.queue.pop_front() {
                // SetAcceptKeyFrames(false) · ProcessNewKeyFrame · 처리 시작에 mbAbortBA = false
                s.busy = true;
                s.accept = false;
                self.abort_ba.store(false, Ordering::SeqCst);
                return Some(k);
            }
            if s.holders > 0 && !s.not_stop {
                // Stop(): 멈추고 모든 토큰이 놓일 때까지
                s.stopped = true;
                s.accept = false;
                self.cv.notify_all();
                while s.holders > 0 && !s.finish {
                    s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                s.stopped = false;
                // Release: 원본은 여기서 대기열을 비움(멈춘 동안 들어온 키프레임은 지움)
                s.queue.clear();
                continue;
            }
            s.accept = true;
            self.cv.notify_all();
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// CheckNewKeyFrames(처리 중에 다음이 와 있나)
    pub fn check_new_keyframes(&self) -> bool {
        !self.st().queue.is_empty()
    }
}

/// 지도 스레드 멈춤(RequestStop ~ Release). 놓이면 Release
pub struct StopToken<K> {
    m: Arc<Mapper<K>>,
}

impl<K> StopToken<K> {
    /// 실제로 멈출 때까지 기다림. 끝나 버렸으면 false
    pub fn wait(&self) -> bool {
        assert_no_map_lock("지도 스레드 멈춤");
        let mut s = self.m.st();
        while !s.stopped && !s.finished {
            s = self.m.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.stopped
    }
}

impl<K> Drop for StopToken<K> {
    fn drop(&mut self) {
        let mut s = self.m.st();
        s.holders -= 1;
        self.m.cv.notify_all();
    }
}

/// 루프 스레드의 대기열(LoopClosing::mlpLoopKeyFrameQueue) · 끝
pub struct LoopQueue<K> {
    st: Mutex<(VecDeque<K>, bool)>,
    cv: Condvar,
    /// 넣었지만 아직 다 처리하지 않은 수(재생 도구가 루프 스레드까지 기다릴 때)
    pending: std::sync::atomic::AtomicUsize,
}

impl<K> LoopQueue<K> {
    pub fn new() -> Arc<LoopQueue<K>> {
        Arc::new(LoopQueue { st: Mutex::new((VecDeque::new(), false)), cv: Condvar::new(), pending: Default::default() })
    }
    /// InsertKeyFrame(지도 스레드가 처리를 끝낸 키프레임)
    pub fn insert(&self, k: K) {
        let mut s = self.st.lock().unwrap_or_else(|e| e.into_inner());
        s.0.push_back(k);
        self.pending.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.cv.notify_all();
    }
    /// 키프레임 하나 처리 끝(루프 흐름이 부름)
    pub fn done(&self) {
        self.pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
    pub fn pending(&self) -> usize {
        self.pending.load(std::sync::atomic::Ordering::SeqCst)
    }
    pub fn finish(&self) {
        let mut s = self.st.lock().unwrap_or_else(|e| e.into_inner());
        s.1 = true;
        self.cv.notify_all();
    }
    /// 다음 키프레임(끝이면 None)
    pub fn next(&self) -> Option<K> {
        assert_no_map_lock("루프 대기열");
        let mut s = self.st.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if s.1 {
                return None;
            }
            if let Some(k) = s.0.pop_front() {
                return Some(k);
            }
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    use std::time::{Duration, Instant};

    /// 끝나야 할 일을 시간 제한 안에 끝내는지(교착이면 실패)
    fn within(secs: u64, f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            f();
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(secs)).expect("시간 안에 안 끝남(교착)");
    }

    #[test]
    fn two_stoppers_release_only_when_both_done() {
        within(10, || {
            let m: Arc<Mapper<u64>> = Mapper::new();
            let mm = Arc::clone(&m);
            let h = thread::spawn(move || while mm.next().is_some() {});
            let a = m.request_stop();
            let b = m.request_stop();
            assert!(a.wait() && b.wait());
            drop(a);
            thread::sleep(Duration::from_millis(20));
            // 원본 Release 라면 여기서 다시 돌았다
            assert!(m.is_stopped());
            drop(b);
            let t = Instant::now();
            while m.is_stopped() {
                assert!(t.elapsed() < Duration::from_secs(5));
                thread::yield_now();
            }
            m.finish();
            h.join().unwrap();
        });
    }

    #[test]
    fn queue_drains_before_stop_and_not_stop_refused() {
        within(10, || {
            let m: Arc<Mapper<u64>> = Mapper::new();
            let done = Arc::new(AtomicU64::new(0));
            let (mm, d) = (Arc::clone(&m), Arc::clone(&done));
            for k in 0..5 {
                m.insert(k);
            }
            let tok = m.request_stop();
            let h = thread::spawn(move || {
                while let Some(_k) = mm.next() {
                    d.fetch_add(1, Ordering::SeqCst);
                }
            });
            assert!(tok.wait());
            assert_eq!(done.load(Ordering::SeqCst), 5, "쌓인 키프레임을 먼저 처리");
            assert!(!m.set_not_stop(true), "멈춘 뒤 SetNotStop(true) 는 거절");
            drop(tok);
            m.wait_idle();
            m.finish();
            h.join().unwrap();
        });
    }

    #[test]
    #[should_panic(expected = "교착 자리")]
    fn waiting_with_map_lock_panics() {
        let map = MapUpdate::new(0u32);
        let m: Arc<Mapper<u64>> = Mapper::new();
        let tok = m.request_stop();
        let _g = map.lock();
        tok.wait();
    }

    /// 추적 · 지도 · 루프 · 옮겨 붙이기가 동시에 멈춤 · 지도 잠금 · 한가함 기다림을 섞어도 끝남
    #[test]
    fn three_threads_stress() {
        within(60, || {
            let map = Arc::new(MapUpdate::new(0u64));
            let m: Arc<Mapper<u64>> = Mapper::new();
            let lq: Arc<LoopQueue<u64>> = LoopQueue::new();
            let (mm, lq2, map2) = (Arc::clone(&m), Arc::clone(&lq), Arc::clone(&map));
            let mapper = thread::spawn(move || {
                while let Some(k) = mm.next() {
                    *map2.lock() += 1; // 지역 BA 가 지도 잠금을 잡음
                    lq2.insert(k);
                }
            });
            let (m3, lq3, map3) = (Arc::clone(&m), Arc::clone(&lq), Arc::clone(&map));
            let looper = thread::spawn(move || {
                while let Some(k) = lq3.next() {
                    if k % 7 == 0 {
                        // CorrectLoop: 멈춤 → 지도 잠금 → 고침 → Release
                        let t = m3.request_stop();
                        if !t.wait() {
                            break;
                        }
                        *map3.lock() += 1000;
                    }
                }
            });
            for f in 0..3000u64 {
                {
                    let mut g = map.lock();
                    *g += 1;
                    if f % 50 == 0 {
                        // 원본 DoPendingGraft 는 여기서(잠금 안) 멈춤을 기다렸다: 이 판은 잠금을 놓은 뒤에 기다림
                    }
                }
                if f % 50 == 0 {
                    let t = m.request_stop();
                    t.wait();
                    *map.lock() += 1;
                }
                if !m.stop_requested() && m.set_not_stop(true) {
                    m.insert(f);
                    m.set_not_stop(false);
                }
                if f % 10 == 0 {
                    m.wait_idle();
                }
            }
            m.wait_idle();
            m.finish();
            mapper.join().unwrap();
            lq.finish();
            looper.join().unwrap();
        });
    }
}
