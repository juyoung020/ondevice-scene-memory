//! 처리 루프 하나 + 작업 풀의 뼈대(docs/threads_npu.md "구조").
//!
//! - [`Waker`]: 다른 스레드(풀 일꾼 · NPU 완료 · IMU)가 루프를 깨우는 손잡이(리눅스 eventfd, 그 밖은 조건 변수)
//! - [`Poll`]: 루프가 잠드는 곳(리눅스 epoll: 깨우개 · 카메라 · 소켓 fd 를 함께 건다)
//! - [`done`]: 완료 줄(락 없는 채널 + 깨우개). 어느 스레드에서 넣어도 루프가 깬다
//! - [`Lane`] · [`SharedLane`]: 스레드 하나가 하던 "차례 있는 일"을 스레드 없이: 일이 오면 풀 일꾼 하나가 상태를 쥐고 밀린 일을 차례대로 돈다
//! - [`Recycler`] · [`Pooled`]: 놓으면 저절로 돌아가는 버퍼 손잡이
//! - [`start_pool`]: 작업 풀(rayon 전역 풀 하나, 기본 3 일꾼). 펼치기(par_iter)와 줄 일이 모두 여기서 돈다
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
mod sys {
    unsafe extern "C" {
        pub fn eventfd(init: u32, flags: i32) -> i32;
        pub fn epoll_create1(flags: i32) -> i32;
        pub fn epoll_ctl(ep: i32, op: i32, fd: i32, ev: *mut EpollEvent) -> i32;
        pub fn epoll_wait(ep: i32, evs: *mut EpollEvent, max: i32, timeout: i32) -> i32;
        pub fn read(fd: i32, buf: *mut std::ffi::c_void, n: usize) -> isize;
        pub fn write(fd: i32, buf: *const std::ffi::c_void, n: usize) -> isize;
        pub fn close(fd: i32) -> i32;
    }
    pub const EFD_CLOEXEC: i32 = 0o2000000;
    pub const EFD_NONBLOCK: i32 = 0o4000;
    pub const EPOLL_CLOEXEC: i32 = 0o2000000;
    pub const EPOLL_CTL_ADD: i32 = 1;
    pub const EPOLL_CTL_DEL: i32 = 2;
    pub const EPOLL_CTL_MOD: i32 = 3;
    pub const EPOLLIN: u32 = 1;
    pub const EPOLLOUT: u32 = 4;
    // aarch64 · x86_64 모두 커널 구조체는 묶음(packed) 12 바이트(x86_64) / 16 바이트(aarch64)
    #[cfg_attr(target_arch = "x86_64", repr(C, packed))]
    #[cfg_attr(not(target_arch = "x86_64"), repr(C))]
    #[derive(Clone, Copy, Default)]
    pub struct EpollEvent {
        pub events: u32,
        pub data: u64,
    }
}

/// 루프 깨우개. 리눅스는 eventfd(루프 epoll 에 걸림), 그 밖은 [`Poll`] 이 기다리는 조건 변수에 알린다
pub struct Waker {
    #[cfg(target_os = "linux")]
    fd: i32,
    #[cfg(not(target_os = "linux"))]
    hub: Arc<fallback::Hub>,
    /// 깨운 뒤 아직 루프가 못 본 것(쓰기 시스템 호출을 줄임: 이미 깨워 둔 동안은 다시 쓰지 않음)
    armed: AtomicBool,
    /// 깨운 횟수(계측)
    pub wakes: AtomicU64,
}

impl Waker {
    /// 다른 스레드에서 부름: 루프를 깨운다(이미 깨워 두었으면 아무것도 안 함)
    pub fn wake(&self) {
        if self.armed.swap(true, Ordering::AcqRel) {
            return;
        }
        self.wakes.fetch_add(1, Ordering::Relaxed);
        #[cfg(target_os = "linux")]
        unsafe {
            let one = 1u64;
            sys::write(self.fd, &one as *const u64 as *const std::ffi::c_void, 8);
        }
        #[cfg(not(target_os = "linux"))]
        self.hub.notify();
    }
    /// 루프가 깬 뒤 부름: 다음 wake 가 다시 깨우게.
    /// 차례가 중요하다: eventfd 를 먼저 비우고 그다음 표시를 내린다. 거꾸로(표시 먼저)면 그 사이에 온 wake 가 표시를 올리고 쓴 값을
    /// 이쪽 read 가 지워, 표시는 올라가 있는데 eventfd 는 빈 채로 남아 이후 wake 가 모두 그냥 돌아간다(루프가 영영 안 깸).
    /// 이 차례에서는 비우기와 내리기 사이에 온 wake 가 쓰지 않고 돌아가도, 루프가 이미 깨어 있어 이 뒤에 상태를 본다
    pub fn clear(&self) {
        #[cfg(target_os = "linux")]
        unsafe {
            let mut v = 0u64;
            sys::read(self.fd, &mut v as *mut u64 as *mut std::ffi::c_void, 8);
        }
        self.armed.store(false, Ordering::Release);
    }
    #[cfg(target_os = "linux")]
    pub fn fd(&self) -> i32 {
        self.fd
    }
}

#[cfg(target_os = "linux")]
impl Drop for Waker {
    fn drop(&mut self) {
        unsafe {
            sys::close(self.fd);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod fallback {
    use std::sync::{Condvar, Mutex};
    #[derive(Default)]
    pub struct Hub {
        pub m: Mutex<u64>,
        pub cv: Condvar,
    }
    impl Hub {
        pub fn notify(&self) {
            *self.m.lock().unwrap() += 1;
            self.cv.notify_all();
        }
    }
}

/// 루프가 잠드는 곳: 깨우개와 fd(카메라 · 소켓)를 함께 기다린다. 토큰은 건 쪽이 정한 번호
pub struct Poll {
    #[cfg(target_os = "linux")]
    ep: i32,
    #[cfg(not(target_os = "linux"))]
    hub: Arc<fallback::Hub>,
    #[cfg(not(target_os = "linux"))]
    seen: u64,
    wakers: Vec<(Arc<Waker>, u64)>,
    /// 잠든 횟수 · 깬 이벤트 수(계측)
    pub sleeps: u64,
}

impl Default for Poll {
    fn default() -> Self {
        Self::new()
    }
}

impl Poll {
    pub fn new() -> Poll {
        #[cfg(target_os = "linux")]
        {
            let ep = unsafe { sys::epoll_create1(sys::EPOLL_CLOEXEC) };
            assert!(ep >= 0, "epoll_create1");
            Poll { ep, wakers: vec![], sleeps: 0 }
        }
        #[cfg(not(target_os = "linux"))]
        Poll { hub: Arc::new(fallback::Hub::default()), seen: 0, wakers: vec![], sleeps: 0 }
    }

    /// 새 깨우개를 이 루프에 건다(토큰으로 어느 것이 깨웠는지 앎)
    pub fn waker(&mut self, token: u64) -> Arc<Waker> {
        #[cfg(target_os = "linux")]
        let w = {
            let fd = unsafe { sys::eventfd(0, sys::EFD_CLOEXEC | sys::EFD_NONBLOCK) };
            assert!(fd >= 0, "eventfd");
            let w = Arc::new(Waker { fd, armed: AtomicBool::new(false), wakes: AtomicU64::new(0) });
            self.add_fd(fd, token, false);
            w
        };
        #[cfg(not(target_os = "linux"))]
        let w = Arc::new(Waker { hub: self.hub.clone(), armed: AtomicBool::new(false), wakes: AtomicU64::new(0) });
        self.wakers.push((w.clone(), token));
        w
    }

    /// fd 를 건다(읽을 것이 생기면 토큰으로 깸, out = 쓸 자리가 생겨도). 리눅스만
    #[cfg(target_os = "linux")]
    pub fn add_fd(&mut self, fd: i32, token: u64, out: bool) {
        let mut ev = sys::EpollEvent { events: sys::EPOLLIN | if out { sys::EPOLLOUT } else { 0 }, data: token };
        let r = unsafe { sys::epoll_ctl(self.ep, sys::EPOLL_CTL_ADD, fd, &mut ev) };
        assert!(r == 0, "epoll_ctl add");
    }
    /// 건 fd 의 관심을 바꿈(out = 쓸 자리도 기다림)
    #[cfg(target_os = "linux")]
    pub fn mod_fd(&mut self, fd: i32, token: u64, out: bool) {
        let mut ev = sys::EpollEvent { events: sys::EPOLLIN | if out { sys::EPOLLOUT } else { 0 }, data: token };
        unsafe { sys::epoll_ctl(self.ep, sys::EPOLL_CTL_MOD, fd, &mut ev) };
    }
    #[cfg(target_os = "linux")]
    pub fn del_fd(&mut self, fd: i32) {
        let mut ev = sys::EpollEvent::default();
        unsafe { sys::epoll_ctl(self.ep, sys::EPOLL_CTL_DEL, fd, &mut ev) };
    }

    /// 준비된 것이 생기거나 timeout 이 지날 때까지 잠든다(None = 무한). 깬 토큰을 out 에(깨우개는 비워 다음 깸을 받게)
    pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Option<Duration>) {
        out.clear();
        self.sleeps += 1;
        #[cfg(target_os = "linux")]
        {
            let ms = timeout.map_or(-1, |d| d.as_micros().div_ceil(1000).min(i32::MAX as u128) as i32);
            let mut evs = [sys::EpollEvent::default(); 32];
            let n = loop {
                let n = unsafe { sys::epoll_wait(self.ep, evs.as_mut_ptr(), evs.len() as i32, ms) };
                if n >= 0 {
                    break n as usize;
                }
                // EINTR(신호) 은 다시
                if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                    break 0;
                }
            };
            for e in &evs[..n] {
                let t = e.data;
                out.push(t);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let mut g = self.hub.m.lock().unwrap();
            if *g == self.seen && timeout != Some(Duration::ZERO) {
                g = match timeout {
                    Some(d) => self.hub.cv.wait_timeout(g, d).unwrap().0,
                    None => self.hub.cv.wait(g).unwrap(),
                };
            }
            self.seen = *g;
            drop(g);
            for (w, t) in &self.wakers {
                if w.armed.load(Ordering::Acquire) {
                    out.push(*t);
                }
            }
        }
        for (w, t) in &self.wakers {
            if out.contains(t) {
                w.clear();
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for Poll {
    fn drop(&mut self) {
        unsafe {
            sys::close(self.ep);
        }
    }
}

/// 완료 줄의 넣는 쪽(복제해 여러 스레드에서): 넣고 루프를 깨운다
pub struct DoneTx<T> {
    tx: Sender<T>,
    waker: Arc<Waker>,
}
impl<T> Clone for DoneTx<T> {
    fn clone(&self) -> Self {
        DoneTx { tx: self.tx.clone(), waker: self.waker.clone() }
    }
}
impl<T> DoneTx<T> {
    /// 넣고 깨움. 받는 쪽이 없으면(끝나는 중) 버림
    pub fn send(&self, v: T) {
        if self.tx.send(v).is_ok() {
            self.waker.wake();
        }
    }
}

/// 완료 줄의 받는 쪽(루프)
pub struct DoneRx<T> {
    rx: Receiver<T>,
}
impl<T> DoneRx<T> {
    pub fn try_recv(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }
    /// 하나 올 때까지 기다림(결과를 늘 같게 하는 재생이 그 자리에서 기다릴 때)
    pub fn recv(&self) -> Option<T> {
        self.rx.recv().ok()
    }
    pub fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        self.rx.try_iter()
    }
}

/// 완료 줄 하나: 루프 깨우개에 묶는다
pub fn done<T>(waker: Arc<Waker>) -> (DoneTx<T>, DoneRx<T>) {
    let (tx, rx) = std::sync::mpsc::channel();
    (DoneTx { tx, waker }, DoneRx { rx })
}

// ------------------------------------------------------------------ 작업 풀

static POOL_N: AtomicU64 = AtomicU64::new(0);

/// 작업 풀(rayon 전역 풀) 시작: n 일꾼, 이름 pool-0 … , 우선순위 · 코어(mask 0 = 프로세스 코어 전부).
/// 펼치기(par_iter)와 줄 일([`Lane`])이 모두 여기서 돈다. 한 번만(두 번째부터는 무시)
pub fn start_pool(n: usize, nice: i32, mask: u64) {
    let n = n.max(1);
    if rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("pool-{i}"))
        .start_handler(move |_| {
            crate::threads::apply(nice, mask);
            crate::prof::track_thread();
        })
        .build_global()
        .is_ok()
    {
        POOL_N.store(n as u64, Ordering::Relaxed);
    }
}

/// 작업 풀 일꾼 수(시작 전이면 rayon 기본)
pub fn pool_size() -> usize {
    match POOL_N.load(Ordering::Relaxed) {
        0 => rayon::current_num_threads(),
        n => n as usize,
    }
}

/// 풀 일꾼 안에서 다른 줄 일이 채울 것을 기다릴 때: 기다리는 동안 풀의 다른 일을 대신 돌려(일꾼을 붙잡아 생기는 교착을 막음),
/// 할 일이 없으면 잠깐 잔다. 풀 밖(루프)에서 부르면 잠깐 자며 기다린다
pub fn wait_until(mut ready: impl FnMut() -> bool) {
    while !ready() {
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => {}
            _ => std::thread::sleep(Duration::from_micros(200)),
        }
    }
}

/// 줄 일 하나(상태를 빌려 씀)
pub type Job<S> = Box<dyn FnOnce(&mut S) + Send>;

struct LaneInner<S> {
    name: &'static str,
    tx: Sender<Job<S>>,
    rx: std::cell::UnsafeCell<Receiver<Job<S>>>,
    s: std::cell::UnsafeCell<Option<S>>,
    /// 넣었지만 아직 안 끝난 일 수: 0 → 1 로 만든 쪽이 일꾼을 띄우고, 일꾼은 1 → 0 이 될 때 내려놓는다(일꾼은 늘 하나)
    pending: AtomicU64,
    idle: (std::sync::Mutex<()>, std::sync::Condvar),
    on_idle: Option<Box<dyn Fn() + Send + Sync>>,
    nice: i32,
    busy_ns: AtomicU64,
    jobs: AtomicU64,
}
// 상태 · 받는 쪽은 pending 규칙으로 한 번에 일꾼 하나만 만진다
unsafe impl<S: Send> Sync for LaneInner<S> {}

impl<S: Send + 'static> LaneInner<S> {
    fn push(self: &Arc<Self>, job: Job<S>) {
        let _ = self.tx.send(job); // 넣은 뒤 셈(일꾼이 센 수만큼 꼭 꺼낼 수 있게)
        if self.pending.fetch_add(1, Ordering::AcqRel) == 0 {
            let me = self.clone();
            rayon::spawn(move || me.run());
        }
    }
    fn run(&self) {
        let c = Instant::now();
        if self.nice != 0 {
            crate::threads::set_nice(self.nice);
        }
        loop {
            // SAFETY: pending > 0 인 동안 이 일꾼만 rx · s 를 만진다
            let j = unsafe { (*self.rx.get()).try_recv() }.expect("줄: 센 일이 줄에 없음");
            j(unsafe { (*self.s.get()).as_mut().expect("줄 상태") });
            self.jobs.fetch_add(1, Ordering::Relaxed);
            if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                break;
            }
        }
        if self.nice != 0 {
            crate::threads::set_nice(0);
        }
        self.busy_ns.fetch_add(c.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Some(f) = &self.on_idle {
            f();
        }
        let _g = self.idle.0.lock().unwrap();
        self.idle.1.notify_all();
    }
    fn wait_idle(&self) {
        // 풀 일꾼이 기다리면(물체 기억 일이 점 줄을 기다릴 때) 일꾼을 붙잡지 않고 풀의 다른 일을 대신 돈다: 일꾼이 하나여도 막히지 않음
        if rayon::current_thread_index().is_some() {
            return wait_until(|| self.pending.load(Ordering::Acquire) == 0);
        }
        let mut g = self.idle.0.lock().unwrap();
        while self.pending.load(Ordering::Acquire) > 0 {
            g = self.idle.1.wait(g).unwrap();
        }
    }
}

/// 줄: 스레드 하나가 하던 "차례 있는 일"을 스레드 없이 한다. 상태 S 는 줄이 가지며, 일이 오면 풀 일꾼 하나가 밀린 일을
/// 차례대로 모두 돈 뒤 내려놓는다(한 번에 일꾼 하나) → 순서 · 결과가 전용 스레드와 같다. 일을 넣는 쪽은 하나(루프)이고
/// [`Lane::with`] 로 상태를 직접 만질 수 있다(밀린 일이 끝날 때까지 기다린 뒤). 여러 스레드가 넣는 줄은 [`SharedLane`]
pub struct Lane<S: Send + 'static>(Arc<LaneInner<S>>);

fn lane_inner<S>(name: &'static str, s: S, nice: i32, on_idle: Option<Box<dyn Fn() + Send + Sync>>) -> Arc<LaneInner<S>> {
    let (tx, rx) = std::sync::mpsc::channel();
    Arc::new(LaneInner {
        name,
        tx,
        rx: std::cell::UnsafeCell::new(rx),
        s: std::cell::UnsafeCell::new(Some(s)),
        pending: AtomicU64::new(0),
        idle: (std::sync::Mutex::new(()), std::sync::Condvar::new()),
        on_idle,
        nice,
        busy_ns: AtomicU64::new(0),
        jobs: AtomicU64::new(0),
    })
}

impl<S: Send + 'static> Lane<S> {
    /// nice: 일꾼이 이 줄을 도는 동안의 nice(긴 일 줄: 다시 쌓기 · 저장 = 10, 그 밖 0)
    pub fn new(name: &'static str, s: S, nice: i32) -> Lane<S> {
        Lane(lane_inner(name, s, nice, None))
    }
    /// 일꾼이 밀린 일을 다 돌고 내려놓을 때마다 부를 함수(루프 깨우기 등)
    pub fn with_idle(name: &'static str, s: S, nice: i32, on_idle: impl Fn() + Send + Sync + 'static) -> Lane<S> {
        Lane(lane_inner(name, s, nice, Some(Box::new(on_idle))))
    }
    pub fn name(&self) -> &'static str {
        self.0.name
    }
    pub fn push(&mut self, job: impl FnOnce(&mut S) + Send + 'static) {
        self.0.push(Box::new(job));
    }
    /// 아직 안 끝난 일 수
    pub fn pending(&self) -> u64 {
        self.0.pending.load(Ordering::Acquire)
    }
    pub fn idle(&self) -> bool {
        self.pending() == 0
    }
    /// 밀린 일이 모두 끝날 때까지 기다림(결과를 늘 같게 하는 재생이 그 자리에서 기다리던 곳)
    pub fn wait_idle(&self) {
        self.0.wait_idle();
    }
    /// 상태를 직접 만짐: 밀린 일이 끝날 때까지 기다린 뒤(넣는 쪽이 이 하나뿐이라 그사이 새 일이 없음)
    pub fn with<R>(&mut self, f: impl FnOnce(&mut S) -> R) -> R {
        self.0.wait_idle();
        // SAFETY: pending == 0 이고 넣는 쪽(&mut self)이 여기뿐 → 일꾼 없음
        f(unsafe { (*self.0.s.get()).as_mut().unwrap() })
    }
    /// 밀린 일을 모두 끝내고 상태를 꺼냄
    pub fn into_inner(self) -> S {
        self.0.wait_idle();
        unsafe { (*self.0.s.get()).take().unwrap() }
    }
    /// 일 수 · 일꾼이 쓴 초
    pub fn stats(&self) -> (u64, f64) {
        (self.0.jobs.load(Ordering::Relaxed), self.0.busy_ns.load(Ordering::Relaxed) as f64 * 1e-9)
    }
}

/// 여러 스레드(루프 · 다른 줄 일)가 일을 넣는 줄(기록 · 사진 벡터). 상태는 일 안에서만 만진다
pub struct SharedLane<S: Send + 'static>(Arc<LaneInner<S>>);
impl<S: Send + 'static> Clone for SharedLane<S> {
    fn clone(&self) -> Self {
        SharedLane(self.0.clone())
    }
}
impl<S: Send + 'static> SharedLane<S> {
    pub fn new(name: &'static str, s: S, nice: i32, on_idle: Option<Box<dyn Fn() + Send + Sync>>) -> SharedLane<S> {
        SharedLane(lane_inner(name, s, nice, on_idle))
    }
    pub fn push(&self, job: impl FnOnce(&mut S) + Send + 'static) {
        self.0.push(Box::new(job));
    }
    pub fn pending(&self) -> u64 {
        self.0.pending.load(Ordering::Acquire)
    }
    pub fn wait_idle(&self) {
        self.0.wait_idle();
    }
    pub fn stats(&self) -> (u64, f64) {
        (self.0.jobs.load(Ordering::Relaxed), self.0.busy_ns.load(Ordering::Relaxed) as f64 * 1e-9)
    }
}

/// 버퍼 돌려받기 줄: [`Pooled`] 손잡이를 놓으면(어느 스레드에서든) 여기로 돌아온다
pub struct Recycler<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
    /// 새로 만든 수(계측)
    pub made: u64,
}
impl<T: Send + 'static> Default for Recycler<T> {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Recycler { tx, rx, made: 0 }
    }
}
impl<T: Send + 'static> Recycler<T> {
    /// 돌아온 것 하나(없으면 make 로 새로)
    pub fn take(&mut self, make: impl FnOnce() -> T) -> Pooled<T> {
        let v = match self.rx.try_recv() {
            Ok(v) => v,
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                self.made += 1;
                make()
            }
        };
        Pooled { v: Some(v), back: self.tx.clone() }
    }
    /// 감싸기(돌려받을 곳만 붙임)
    pub fn wrap(&self, v: T) -> Pooled<T> {
        Pooled { v: Some(v), back: self.tx.clone() }
    }
}

/// 놓으면 저절로 [`Recycler`] 로 돌아가는 손잡이
pub struct Pooled<T: Send + 'static> {
    v: Option<T>,
    back: Sender<T>,
}
impl<T: Send + 'static> Pooled<T> {
    /// 돌려받지 않고 안의 값을 꺼냄
    pub fn detach(mut self) -> T {
        self.v.take().unwrap()
    }
}
impl<T: Send + 'static> std::ops::Deref for Pooled<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.v.as_ref().unwrap()
    }
}
impl<T: Send + 'static> std::ops::DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.v.as_mut().unwrap()
    }
}
impl<T: Send + 'static> Drop for Pooled<T> {
    fn drop(&mut self) {
        if let Some(v) = self.v.take() {
            let _ = self.back.send(v);
        }
    }
}

/// 마감 시각 목록(타이머): 루프가 잠들 시간을 정함
#[derive(Default)]
pub struct Deadlines(Vec<(Instant, u64)>);
impl Deadlines {
    pub fn set(&mut self, token: u64, at: Instant) {
        self.0.retain(|x| x.1 != token);
        self.0.push((at, token));
    }
    pub fn cancel(&mut self, token: u64) {
        self.0.retain(|x| x.1 != token);
    }
    /// 다음 마감까지 남은 시간(없으면 None)
    pub fn timeout(&self) -> Option<Duration> {
        self.0.iter().map(|x| x.0).min().map(|t| t.saturating_duration_since(Instant::now()))
    }
    /// 지난 마감의 토큰을 꺼냄
    pub fn fired(&mut self, out: &mut Vec<u64>) {
        let now = Instant::now();
        self.0.retain(|x| {
            if x.0 <= now {
                out.push(x.1);
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_runs_in_order_and_returns_state() {
        let mut lane = Lane::new("t", Vec::<u32>::new(), 0);
        for i in 0..100 {
            lane.push(move |v: &mut Vec<u32>| v.push(i));
        }
        let v = lane.into_inner();
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn done_wakes_poll() {
        let mut p = Poll::new();
        let w = p.waker(7);
        let (tx, rx) = done::<u32>(w);
        std::thread::spawn(move || tx.send(5));
        let mut out = vec![];
        let c = Instant::now();
        while rx.try_recv().is_none() {
            p.wait(&mut out, Some(Duration::from_secs(5)));
            assert!(c.elapsed() < Duration::from_secs(5));
        }
    }

    #[test]
    fn pooled_returns() {
        let mut r: Recycler<Vec<u8>> = Recycler::default();
        {
            let a = r.take(|| vec![1, 2, 3]);
            assert_eq!(a.len(), 3);
        }
        let b = r.take(Vec::new);
        assert_eq!(b.len(), 3);
        assert_eq!(r.made, 1);
    }
}
