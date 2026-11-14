//! 단계 사이 "가장 새 장면" 칸: 앞 단계는 넣기만(밀린 옛 장면은 덮어써 돌려받음), 뒤 단계는 가장 새 것을 꺼냄.
//! 앞 단계는 뒤 단계를 기다리지 않는다. 장면 번호(seq)를 함께 들어 결과를 나중에 장면에 붙일 수 있다(분할 결과 → 물체).
//! 재생(결정적)은 `exact` 칸: 덮어쓰지 않고 뒤 단계가 꺼낼 때까지 넣기가 기다린다(모든 장면을 차례대로).
use std::sync::{Condvar, Mutex};

pub struct Latest<T> {
    m: Mutex<State<T>>,
    cv: Condvar,
    exact: bool,
}

struct State<T> {
    v: Option<(u64, T)>,
    closed: bool,
    dropped: u64,
}

impl<T> Latest<T> {
    /// exact = false: 덮어씀(실시간), true: 하나도 버리지 않음(재생)
    pub fn new(exact: bool) -> Latest<T> {
        Latest { m: Mutex::new(State { v: None, closed: false, dropped: 0 }), cv: Condvar::new(), exact }
    }
    /// 넣기: 실시간이면 꺼내지 않은 옛 장면을 덮어써 돌려줌(버퍼를 줄에 돌려주라고), 재생이면 빌 때까지 기다림
    pub fn put(&self, seq: u64, v: T) -> Option<(u64, T)> {
        let mut s = self.m.lock().unwrap();
        if self.exact {
            while s.v.is_some() && !s.closed {
                s = self.cv.wait(s).unwrap();
            }
        }
        let old = s.v.replace((seq, v));
        s.dropped += old.is_some() as u64;
        self.cv.notify_all();
        old
    }
    /// 꺼내기: 가장 새 장면(없으면 올 때까지), 닫혔고 비었으면 None
    pub fn take(&self) -> Option<(u64, T)> {
        let mut s = self.m.lock().unwrap();
        loop {
            if let Some(v) = s.v.take() {
                self.cv.notify_all();
                return Some(v);
            }
            if s.closed {
                return None;
            }
            s = self.cv.wait(s).unwrap();
        }
    }
    /// 기다리지 않고 꺼내기
    pub fn try_take(&self) -> Option<(u64, T)> {
        let mut s = self.m.lock().unwrap();
        let v = s.v.take();
        if v.is_some() {
            self.cv.notify_all();
        }
        v
    }
    /// 앞 단계 끝: 남은 것은 꺼낼 수 있고 그 뒤 take 는 None
    pub fn close(&self) {
        self.m.lock().unwrap().closed = true;
        self.cv.notify_all();
    }
    /// 덮어써 버린 장면 수
    pub fn dropped(&self) -> u64 {
        self.m.lock().unwrap().dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latest_overwrites_exact_waits() {
        let s = Latest::new(false);
        assert!(s.put(1, "a").is_none());
        assert_eq!(s.put(2, "b"), Some((1, "a")));
        assert_eq!(s.take(), Some((2, "b")));
        assert_eq!(s.dropped(), 1);
        let e = std::sync::Arc::new(Latest::new(true));
        let e2 = e.clone();
        let th = std::thread::spawn(move || (0..5).map(|_| e2.take().unwrap().0).collect::<Vec<_>>());
        for i in 0..5 {
            e.put(i, i);
        }
        e.close();
        assert_eq!(th.join().unwrap(), vec![0, 1, 2, 3, 4]);
    }
}
