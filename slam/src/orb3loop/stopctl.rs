//! 지도 스레드(LocalMapping) 멈춤 · 풀기 제어. 원본 `RequestStop/Stop/isStopped/Release` 의 교착 자리를 고친 설계.
//!
//! 원본 문제(docs/orb3_port.md "멈춤 조사"):
//! - 멈춤 요청자가 여럿(CorrectLoop · MergeLocal · GBA 끝 · GraftMap)인데 `Release()` 하나가 `mbStopRequested` 를
//!   통째로 지워, 다른 요청자는 `isStopped()` 를 영영 못 보고 기다린다.
//! - `CorrectLoop` 가 루프 스레드에서 `EmptyQueue()` 를 불러 지도 스레드와 같은 상태를 동시에 쓴다.
//!
//! 여기서는: 요청자마다 표(`StopToken`)를 받아 센다. 지도 스레드는 요청 수가 0 보다 크면 멈추고, 마지막 표가
//! 풀릴 때만 다시 돈다. 큐 비우기는 요청 표에 "비우고 멈춰" 표시만 남기고 실제 처리는 지도 스레드가 한다.
//! 기다림은 모두 종료 확인(`finish`)과 같이 본다.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[derive(Default, Debug)]
struct St {
    requests: u32,
    drain: bool,
    stopped: bool,
    finish: bool,
    not_stop: bool,
}

#[derive(Default)]
pub struct StopCtl {
    m: Mutex<St>,
    cv: Condvar,
}

/// 멈춤 요청 표. 버리면(drop) 풀림.
pub struct StopToken<'a> {
    c: &'a StopCtl,
}

impl Drop for StopToken<'_> {
    fn drop(&mut self) {
        let mut s = self.c.m.lock().unwrap();
        s.requests -= 1;
        if s.requests == 0 {
            s.stopped = false;
        }
        self.c.cv.notify_all();
    }
}

impl<'a> StopToken<'a> {
    /// 지도 스레드가 멈출 때까지(또는 종료) 기다림. 멈췄으면 true, 종료로 빠지면 false.
    pub fn wait_stopped(&self) -> bool {
        let mut s = self.c.m.lock().unwrap();
        loop {
            if s.stopped {
                return true;
            }
            if s.finish {
                return false;
            }
            s = self.c.cv.wait_timeout(s, Duration::from_millis(50)).unwrap().0;
        }
    }
}

impl StopCtl {
    pub fn new() -> Self {
        Self::default()
    }

    /// RequestStop. `drain` = 멈추기 전에 큐를 지도 스레드가 비움(원본 CorrectLoop 의 EmptyQueue 자리).
    pub fn request(&self, drain: bool) -> StopToken<'_> {
        let mut s = self.m.lock().unwrap();
        s.requests += 1;
        if drain {
            // 이미 멈춰 있으면 지도 스레드를 깨워 큐부터 비우게 함(멈춤 상태는 비운 뒤 다시 섬)
            s.drain = true;
            s.stopped = false;
        }
        self.cv.notify_all();
        StopToken { c: self }
    }

    /// 지도 스레드: 큐가 비었거나 요청이 있을 때 부름. 큐를 비워야 하면 Some(true) 를 돌려주고 멈추지 않음
    /// (부른 쪽이 큐를 처리한 뒤 다시 부름). 멈춤 조건이면 표가 다 풀리거나 종료될 때까지 여기서 기다림.
    /// 반환: None = 멈춤 없음(계속 일함), Some(true) = 큐 먼저 비울 것, Some(false) = 멈췄다가 풀림.
    pub fn check_stop(&self, queue_empty: bool) -> Option<bool> {
        let mut s = self.m.lock().unwrap();
        if s.requests == 0 || s.not_stop {
            return None;
        }
        if s.drain && !queue_empty {
            return Some(true);
        }
        s.drain = false;
        s.stopped = true;
        self.cv.notify_all();
        while s.requests > 0 && !s.finish && !s.drain {
            s = self.cv.wait(s).unwrap();
        }
        s.stopped = false;
        Some(false)
    }

    /// SetNotStop(true/false). 켜져 있는 동안 지도 스레드는 멈추지 않음. 이미 멈춰 있으면 false.
    pub fn set_not_stop(&self, v: bool) -> bool {
        let mut s = self.m.lock().unwrap();
        if v && s.stopped {
            return false;
        }
        s.not_stop = v;
        true
    }

    pub fn is_stopped(&self) -> bool {
        self.m.lock().unwrap().stopped
    }

    pub fn stop_requested(&self) -> bool {
        self.m.lock().unwrap().requests > 0
    }

    /// RequestFinish: 모든 기다림을 깨움
    pub fn finish(&self) {
        let mut s = self.m.lock().unwrap();
        s.finish = true;
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// 원본 교착 재현 모양: 요청자 둘, 하나가 먼저 풀어도 다른 하나는 멈춤을 본다.
    #[test]
    fn two_requesters_one_release() {
        let c = Arc::new(StopCtl::new());
        let queue = Arc::new(AtomicUsize::new(5));
        let mapper = {
            let (c, q) = (c.clone(), queue.clone());
            std::thread::spawn(move || {
                let mut stops = 0;
                loop {
                    match c.check_stop(q.load(Ordering::SeqCst) == 0) {
                        Some(true) => {
                            q.fetch_sub(1, Ordering::SeqCst);
                        }
                        Some(false) => stops += 1,
                        None => {
                            if c.m.lock().unwrap().finish {
                                return stops;
                            }
                            std::thread::yield_now();
                        }
                    }
                }
            })
        };
        for _ in 0..200 {
            let a = c.request(true);
            assert!(a.wait_stopped());
            assert_eq!(queue.load(Ordering::SeqCst), 0, "큐는 지도 스레드가 비움");
            let b = c.request(false);
            drop(a);
            assert!(b.wait_stopped(), "앞 요청자가 풀어도 남은 요청자는 멈춤 상태를 봄");
            assert!(c.is_stopped());
            drop(b);
            queue.store(3, Ordering::SeqCst);
        }
        c.finish();
        let stops = mapper.join().unwrap();
        assert!(stops >= 200);
    }

    #[test]
    fn finish_unblocks_waiter() {
        let c = StopCtl::new();
        let t = c.request(false);
        c.finish();
        assert!(!t.wait_stopped());
    }
}
