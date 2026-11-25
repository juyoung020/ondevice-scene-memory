//! 실행기 전용 일꾼 묶음: 일 하나(번호 0..n)를 나눠 돌리고, 끝난 일꾼은 잠깐(SPIN) 돌며 다음 일을 기다린 뒤 잠든다.
//! 노드가 줄줄이 이어지는 추론에서 rayon 처럼 매번 잠든 스레드를 깨우는 비용(이 PC WSL 에서 호출당 0.1–0.8 ms)을 없앤다.
//! 스레드 수: $NN_THREADS, 없으면 rayon 전역 풀 크기. 일 안에서 다시 부르면 그 자리에서 차례로 돈다.
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

const SPIN: Duration = Duration::from_millis(2);

type Job = dyn Fn(usize) + Sync;

struct Shared {
    epoch: AtomicUsize,
    /// 지금 일(없으면 null). 가리키는 곳은 부른 쪽 스택의 &Job 하나
    job: AtomicPtr<&'static Job>,
    next: AtomicUsize,
    total: AtomicUsize,
    finished: AtomicUsize,
    active: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    caller: Mutex<()>,
}

struct Pool {
    sh: &'static Shared,
    n: usize,
}

thread_local! {
    static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn pool() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("NN_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(rayon::current_num_threads).max(1);
        let sh: &'static Shared = Box::leak(Box::new(Shared {
            epoch: AtomicUsize::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            caller: Mutex::new(()),
        }));
        for i in 1..n {
            std::thread::Builder::new().name(format!("nn-{i}")).spawn(move || worker(sh)).expect("nn 일꾼");
        }
        Pool { sh, n }
    })
}

fn work(sh: &Shared, f: &(dyn Fn(usize) + Sync + '_)) {
    loop {
        let i = sh.next.fetch_add(1, SeqCst);
        if i >= sh.total.load(SeqCst) {
            return;
        }
        f(i);
        sh.finished.fetch_add(1, SeqCst);
    }
}

fn worker(sh: &'static Shared) {
    INSIDE.with(|c| c.set(true));
    let mut seen = 0usize;
    loop {
        // 새 일 기다리기: 잠깐 돌고, 없으면 잠듦
        let t0 = Instant::now();
        let mut spins = 0u32;
        while sh.epoch.load(SeqCst) == seen {
            spins += 1;
            if spins % 64 == 0 && t0.elapsed() > SPIN {
                let g = sh.lock.lock().unwrap();
                let _g = sh.cv.wait_while(g, |_| sh.epoch.load(SeqCst) == seen).unwrap();
                break;
            }
            std::hint::spin_loop();
        }
        seen = sh.epoch.load(SeqCst);
        sh.active.fetch_add(1, SeqCst);
        let p = sh.job.load(SeqCst);
        if !p.is_null() {
            // SAFETY: 부른 쪽은 active 가 0 이 될 때까지 일을 붙들고 있음
            let f: &Job = unsafe { *p };
            work(sh, f);
        }
        sh.active.fetch_sub(1, SeqCst);
    }
}

/// f(0), f(1), … f(n-1) 을 일꾼들과 나눠 돌림(끝날 때까지 기다림)
pub fn for_each(n: usize, f: &(dyn Fn(usize) + Sync + '_)) {
    if n == 0 {
        return;
    }
    let p = pool();
    if n == 1 || p.n == 1 || INSIDE.with(|c| c.get()) {
        (0..n).for_each(f);
        return;
    }
    let sh = p.sh;
    let _c = sh.caller.lock().unwrap();
    // SAFETY: 이 함수가 돌아가기 전에 job 을 null 로 돌리고 active == 0 을 기다림
    let mut fr: &'static Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync + '_), &'static Job>(f) };
    sh.total.store(n, SeqCst);
    sh.finished.store(0, SeqCst);
    sh.next.store(0, SeqCst);
    sh.job.store(&mut fr as *mut _, SeqCst);
    {
        let _g = sh.lock.lock().unwrap();
        sh.epoch.fetch_add(1, SeqCst);
    }
    sh.cv.notify_all();
    INSIDE.with(|c| c.set(true));
    work(sh, f);
    INSIDE.with(|c| c.set(false));
    while sh.finished.load(SeqCst) < n {
        std::hint::spin_loop();
    }
    sh.job.store(std::ptr::null_mut(), SeqCst);
    while sh.active.load(SeqCst) != 0 {
        std::hint::spin_loop();
    }
}

/// 일꾼 수(부른 스레드 포함)
pub fn threads() -> usize {
    pool().n
}

/// v 를 chunk 개씩 나눠 f(덩어리 번호, 덩어리)
pub fn chunks_mut<T: Send>(v: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let chunk = chunk.max(1);
    let n = v.len().div_ceil(chunk);
    let len = v.len();
    let p = SendPtr(v.as_mut_ptr());
    for_each(n, &|i| {
        let s = i * chunk;
        let e = (s + chunk).min(len);
        // SAFETY: 덩어리끼리 겹치지 않음
        let c = unsafe { std::slice::from_raw_parts_mut(p.get().add(s), e - s) };
        f(i, c);
    });
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}
