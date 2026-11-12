//! 기억 폴더 쓰기: 임시 파일에 쓴 뒤 바꾸기(윈도우에서 뷰어가 읽는 중이면 잠깐 기다렸다 다시).
use std::io::Write;
use std::path::Path;

pub fn replace(tmp: &Path, dst: &Path) {
    for _ in 0..100 {
        if std::fs::rename(tmp, dst).is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::rename(tmp, dst).expect("replace");
}

/// 기록할 내용: 바로 쓸 바이트, 또는 기록 스레드에서 만들 바이트(PLY 짜기·PNG 압축 등 무거운 것)
type Job = Box<dyn FnOnce() -> Vec<u8> + Send>;

/// 비동기 기록: 켜 두면 파일 쓰기(와 PNG 압축)를 기록 스레드가 맡는다(보낸 순서대로 → view.json 은 그 앞 파일들 뒤에 바뀜).
/// 줄이 WRITE_Q 개 차면 작업 스레드(점·다시 쌓기)는 자리가 날 때까지 기다리고, 본 스레드는 기다림 없이 줄 끝에 넣는다
static WRITER: std::sync::OnceLock<WriteQ> = std::sync::OnceLock::new();
const WRITE_Q: usize = 256;

struct WriteQ {
    q: std::sync::Mutex<std::collections::VecDeque<(std::path::PathBuf, Job)>>,
    item: std::sync::Condvar,
    space: std::sync::Condvar,
}

impl WriteQ {
    fn send(&self, p: std::path::PathBuf, job: Job, wait: bool) {
        let mut q = self.q.lock().unwrap();
        while wait && q.len() >= WRITE_Q {
            q = self.space.wait(q).unwrap();
        }
        q.push_back((p, job));
        self.item.notify_one();
    }
    fn recv(&self) -> (std::path::PathBuf, Job) {
        let mut q = self.q.lock().unwrap();
        loop {
            if let Some(x) = q.pop_front() {
                self.space.notify_one();
                return x;
            }
            q = self.item.wait(q).unwrap();
        }
    }
}

pub fn start_writer() {
    let mut first = false;
    let w = WRITER.get_or_init(|| {
        first = true;
        WriteQ { q: Default::default(), item: std::sync::Condvar::new(), space: std::sync::Condvar::new() }
    });
    if first {
        std::thread::spawn(move || {
            thread_role(Role::Writer);
            loop {
                let (p, job) = w.recv();
                if p.as_os_str().is_empty() {
                    job(); // flush_writer 표시
                    continue;
                }
                run_job(&p, job);
            }
        });
    }
}

pub fn write_atomic(dst: &Path, bytes: &[u8]) {
    write_owned(dst, bytes.to_vec());
}

/// 이미 만든 바이트를 그대로 넘겨 쓴다(복사 없이)
pub fn write_owned(dst: &Path, bytes: Vec<u8>) {
    write_lazy(dst, Box::new(move || bytes));
}

/// 바이트를 기록 스레드에서 만들어 쓴다(기록 스레드가 없으면 바로)
fn write_lazy(dst: &Path, job: Job) {
    match WRITER.get() {
        Some(w) => w.send(dst.to_path_buf(), job, BG.with(|b| b.get())),
        None => run_job(dst, job),
    }
}

fn run_job(dst: &Path, job: Job) {
    let c0 = std::time::Instant::now();
    let b = job();
    write_now(dst, &b);
    slow(c0);
    stats::add(stats::kind(dst), b.len(), c0.elapsed().as_nanos() as u64);
}

fn write_now(dst: &Path, bytes: &[u8]) {
    let tmp = dst.with_extension("tmp");
    if std::fs::write(&tmp, bytes).is_err() {
        // 폴더가 지워졌으면(지도를 새로 쌓아 옛 판 폴더를 지움) 다시 만들어 씀
        let _ = std::fs::create_dir_all(dst.parent().unwrap());
        std::fs::write(&tmp, bytes).expect("write");
    }
    replace(&tmp, dst);
}

/// binary_little_endian PLY: float xyz + uchar rgb
fn ply_bytes(pts: &[([f32; 3], [u8; 3])]) -> Vec<u8> {
    let mut b = Vec::with_capacity(pts.len() * 15 + 200);
    write!(b, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", pts.len()).unwrap();
    for (p, c) in pts {
        for v in p {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(c);
    }
    b
}

/// 점 PLY: 바이트는 쓰는 쪽(실시간이면 기록 스레드)에서 짠다
pub fn ply(dst: &Path, pts: Vec<([f32; 3], [u8; 3])>) {
    write_lazy(dst, Box::new(move || ply_bytes(&pts)));
}

pub fn pgm(dst: &Path, w: usize, h: usize, px: &[u8]) {
    let mut b = format!("P5\n{w} {h}\n255\n").into_bytes();
    b.extend_from_slice(px);
    write_owned(dst, b);
}

// PNG 압축은 기록 스레드에서(물체 사진이 바뀔 때 본 스레드가 장당 수십 ms 멈추던 것)
pub fn png_rgb(dst: &Path, w: u32, h: u32, rgb: &[u8]) {
    let px = rgb.to_vec();
    write_lazy(dst, Box::new(move || {
        let mut v = vec![];
        image::RgbImage::from_raw(w, h, px).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
        v
    }));
}
pub fn png_gray(dst: &Path, w: u32, h: u32, g: &[u8]) {
    let px = g.to_vec();
    write_lazy(dst, Box::new(move || {
        let mut v = vec![];
        image::GrayImage::from_raw(w, h, px).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
        v
    }));
}
pub fn png_u16(dst: &Path, w: u32, h: u32, d: &[u16]) {
    let px = d.to_vec();
    write_lazy(dst, Box::new(move || {
        let mut v = vec![];
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w, h, px).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
        v
    }));
}

/// 기록 통계(종류별 파일 수·바이트·짜기+쓰기 시간): 끝에 한 줄로 출력해 어느 파일이 디스크·기록 스레드를 먹는지 본다
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub const KINDS: [&str; 6] = ["view.json", "pose.json", "map.pgm", "obj ply", "obj png", "기타"];
    static N: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    static B: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    static NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    pub fn kind(p: &std::path::Path) -> usize {
        let s = p.to_string_lossy();
        let f = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        if f == "view.json" { 0 } else if f == "pose.json" { 1 } else if f == "map.pgm" { 2 }
        else if s.contains("objects") && f.ends_with(".ply") { 3 } else if s.contains("objects") && f.ends_with(".png") { 4 } else { 5 }
    }
    pub fn add(k: usize, bytes: usize, ns: u64) {
        N[k].fetch_add(1, Relaxed);
        B[k].fetch_add(bytes as u64, Relaxed);
        NS[k].fetch_add(ns, Relaxed);
    }
    pub fn report() -> String {
        let mut s = String::from("기록:");
        let (mut tb, mut tt) = (0u64, 0u64);
        for k in 0..KINDS.len() {
            let (n, b, ns) = (N[k].load(Relaxed), B[k].load(Relaxed), NS[k].load(Relaxed));
            if n > 0 {
                s += &format!(" {} {}개 {:.1} MB {:.2} s |", KINDS[k], n, b as f64 / 1048576.0, ns as f64 * 1e-9);
            }
            tb += b;
            tt += ns;
        }
        s + &format!(" 합 {:.1} MB {:.2} s", tb as f64 / 1048576.0, tt as f64 * 1e-9)
    }
}


/// 기록 스레드에 밀린 파일을 모두 쓸 때까지 기다림(끝낼 때)
pub fn flush_writer() {
    if let Some(w) = WRITER.get() {
        let (dt, dr) = std::sync::mpsc::channel::<()>();
        w.send(std::path::PathBuf::new(), Box::new(move || { let _ = dt.send(()); vec![] }), false);
        let _ = dr.recv();
    }
}

/// 스레드 역할: 우선순위(nice)·코어 묶기를 역할마다 정한다(`--threads`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role { Main, Reader, Points, Rebuild, Writer, Infer }
pub const ROLE_NAMES: [&str; 6] = ["main", "reader", "pts", "rebuild", "writer", "infer"];
/// 역할별 기본 nice(묶지 않음: 본 스레드와 같은 코어를 나눠 쓰므로 작업 스레드를 크게 낮춤)
const NICE: [(i32, u64); 6] = [(0, 0), (0, 0), (10, 0), (19, 0), (10, 0), (0, 0)];

/// 역할별 (nice, 코어 묶음 비트). 묶음 0 = 프로세스가 받은 코어 전부
static LAYOUT: std::sync::OnceLock<[(i32, u64); 6]> = std::sync::OnceLock::new();
static FULL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// 프로세스가 쓸 수 있는 코어(taskset 으로 준 것)
pub fn allowed_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        }
        let mut m = [0u64; 16];
        if sched_getaffinity(0, 128, m.as_mut_ptr()) == 0 {
            return (0..1024).filter(|i| m[i / 64] >> (i % 64) & 1 == 1).collect();
        }
    }
    (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect()
}

/// `--threads` 해석. 기본(auto): 코어 4 개 이상이면 nice 는 본 스레드·읽기·추론 0, 점 +5, 기록 +5, 다시 쌓기 +10.
/// 코어 4 개 이상이면 본 스레드·읽기(분할 전·후처리)를 앞 코어 둘에, 점·다시 쌓기·기록을 나머지 코어에 묶는다. 추론은 보드(NPU)면 앞 코어 둘, ONNX(CPU)면 나머지 코어(ONNX Runtime 스레드 포함).
/// 코어 3 개 이하는 묶지 않는다(nice 로만 나눔: 점·기록 +10, 다시 쌓기 +19). 직접 주기: `main=0:0,pts=5:2-3,rebuild=10:2-3` (역할=nice[:코어 목록]), `none` = 모두 nice 0·묶지 않음
pub fn set_layout(spec: &str) {
    let cpus = allowed_cpus();
    let bits = |v: &[usize]| v.iter().fold(0u64, |m, &c| if c < 64 { m | 1 << c } else { m });
    let mut l = NICE;
    match spec {
        "none" => l = [(0, 0); 6],
        "auto" => {
            if cpus.len() >= 4 {
                let (fg, bg) = (bits(&cpus[..2]), bits(&cpus[2..]));
                // 추론: 보드는 NPU 가 하므로 부르는 스레드만 앞 코어에, ONNX(CPU) 는 모델 계산이 무거우니 나머지 코어 전부에
                let inf = if cfg!(feature = "onnx") { bg } else { fg };
                l = [(0, fg), (0, fg), (5, bg), (10, bg), (5, bg), (0, inf)];
            }
        }
        s => {
            for part in s.split(',') {
                let (name, rest) = part.split_once('=').expect("--threads: 역할=nice[:코어]");
                let i = ROLE_NAMES.iter().position(|n| *n == name).expect("--threads: 역할 이름(main reader pts rebuild writer infer)");
                let (nice, cores) = rest.split_once(':').map_or((rest, None), |(a, b)| (a, Some(b)));
                let mut cs = vec![];
                for r in cores.unwrap_or("").split('+').filter(|x| !x.is_empty()) {
                    let (a, b) = r.split_once('-').map_or((r, r), |x| x);
                    cs.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap());
                }
                l[i] = (nice.parse().unwrap(), bits(&cs));
            }
        }
    }
    let _ = LAYOUT.set(l);
    eprintln!("스레드 배치(코어 {:?}): {}", cpus, ROLE_NAMES.iter().zip(&l).map(|(n, (ni, m))| format!("{n} nice {ni}{}", if *m == 0 { String::new() } else { format!(" 코어 {m:#x}") })).collect::<Vec<_>>().join(", "));
    // 묶지 않는 역할도 프로세스 코어 전부로 다시 묶음(앞서 다른 역할로 묶인 스레드가 역할을 바꿀 때)
    let all = bits(&cpus);
    let _ = FULL.set(all);
}

/// 지금 스레드를 역할에 맞게(우선순위·코어). 리눅스만, 실패해도 그대로 진행
pub fn thread_role(r: Role) {
    let (nice, mask) = LAYOUT.get().copied().unwrap_or(NICE)[r as usize];
    BG.with(|b| b.set(matches!(r, Role::Points | Role::Rebuild | Role::Writer)));
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
            fn gettid() -> i32;
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let tid = gettid();
        cpu::enter(tid, r as usize);
        setpriority(0, tid as u32, nice); // PRIO_PROCESS + 스레드 id = 그 스레드만
        let mask = if mask == 0 { FULL.get().copied().unwrap_or(0) } else { mask };
        if mask != 0 {
            let mut m = [0u64; 16];
            m[0] = mask;
            sched_setaffinity(tid, 128, m.as_ptr());
        }
    }
    let _ = (nice, mask);
}

/// 스레드 하나의 우선순위·코어 묶음(묶음 0 = 프로세스 코어 전부). 리눅스만
fn apply(nice: i32, mask: u64) {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
            fn gettid() -> i32;
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let tid = gettid();
        setpriority(0, tid as u32, nice);
        let mask = if mask == 0 { FULL.get().copied().unwrap_or(0) } else { mask };
        if mask != 0 {
            let mut m = [0u64; 16];
            m[0] = mask;
            sched_setaffinity(tid, 128, m.as_ptr());
        }
    }
    let _ = (nice, mask);
}

thread_local! {
    /// 이 스레드가 뒤쪽 역할(점·다시 쌓기·기록)인지
    static BG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static BG_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// 나눠 돌리는 스레드 풀 둘: 기본 풀은 본 스레드 역할의 우선순위·코어(장면 넣기·면 뽑기·분할 전후처리),
/// 뒤쪽 풀은 점 역할의 우선순위·코어(점 고르기·다시 합치기). 뒤쪽 일은 뒤쪽 풀에서만 줄 선다.
/// `set_layout` 뒤에 한 번 부른다
pub fn start_pools() {
    let l = LAYOUT.get().copied().unwrap_or(NICE);
    let full = FULL.get().copied().unwrap_or(0);
    let n = |m: u64| (if m == 0 { full } else { m }).count_ones().max(1) as usize;
    let ((n0, m0), (n1, m1)) = (l[Role::Main as usize], l[Role::Points as usize]);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n(m0)).start_handler(move |_| apply(n0, m0)).build_global();
    if let Ok(p) = rayon::ThreadPoolBuilder::new().num_threads(n(m1)).start_handler(move |_| apply(n1, m1)).build() {
        let _ = BG_POOL.set(p);
    }
}

/// 나눠 돌리는 일(rayon)을 부르는 스레드의 역할에 맞는 풀에서 돌린다
pub fn par<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match BG_POOL.get() {
        Some(p) if BG.with(|b| b.get()) => p.install(f),
        _ => f(),
    }
}

/// 보드 흉내(`--slow F`): 이 구간이 F 배 걸린 것처럼 코어를 붙잡고 기다림(잠들지 않아 코어 다툼도 그대로)
static SLOW: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub fn set_slow(f: f32) {
    SLOW.store(f.to_bits(), std::sync::atomic::Ordering::Relaxed);
}
pub fn slow(c0: std::time::Instant) {
    let f = f32::from_bits(SLOW.load(std::sync::atomic::Ordering::Relaxed));
    if f > 1.0 {
        let until = c0.elapsed().mul_f32(f - 1.0);
        let c1 = std::time::Instant::now();
        while c1.elapsed() < until {
            std::hint::spin_loop();
        }
    }
}

/// 역할별 CPU 시간: 스레드가 역할을 맡은 동안 쓴 CPU 를 역할마다 더한다(끝 요약의 `cpu_s`, `--slow` 로 붙잡은 시간 포함). 리눅스만
pub mod cpu {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static DONE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6]; // 역할을 내려놓았거나 끝난 스레드 몫(ns)
    static LIVE: std::sync::Mutex<Vec<(i32, usize, u64)>> = std::sync::Mutex::new(Vec::new()); // (스레드 id, 역할, 역할을 맡을 때의 CPU ns)

    /// 시계 id: 2 = 프로세스 전체, 3 = 지금 스레드, `of_thread(tid)` = 다른 스레드
    fn clock_ns(id: i32) -> u64 {
        #[cfg(target_os = "linux")]
        unsafe {
            unsafe extern "C" {
                fn clock_gettime(id: i32, ts: *mut [i64; 2]) -> i32;
            }
            let mut ts = [0i64; 2];
            if clock_gettime(id, &mut ts) == 0 {
                return ts[0] as u64 * 1_000_000_000 + ts[1] as u64;
            }
        }
        let _ = id;
        0
    }
    fn of_thread(tid: i32) -> i32 {
        (!tid << 3) | 6
    }

    /// 스레드가 끝날 때 남은 몫을 역할에 넘김
    struct Guard(std::cell::Cell<i32>);
    impl Drop for Guard {
        fn drop(&mut self) {
            let mut l = LIVE.lock().unwrap();
            if let Some(i) = l.iter().position(|e| e.0 == self.0.get()) {
                let (_, role, base) = l.swap_remove(i);
                DONE[role].fetch_add(clock_ns(3).saturating_sub(base), Relaxed);
            }
        }
    }
    thread_local! { static GUARD: Guard = const { Guard(std::cell::Cell::new(0)) }; }

    /// 지금 스레드(tid)가 이 역할을 맡음: 앞 역할 몫은 닫는다
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn enter(tid: i32, role: usize) {
        GUARD.with(|g| g.0.set(tid));
        let now = clock_ns(3);
        let mut l = LIVE.lock().unwrap();
        match l.iter_mut().find(|e| e.0 == tid) {
            Some(e) => {
                DONE[e.1].fetch_add(now.saturating_sub(e.2), Relaxed);
                *e = (tid, role, now);
            }
            None => l.push((tid, role, now)),
        }
    }

    /// (역할별 CPU 초, 프로세스 전체 CPU 초). 전체 − 역할 합 = 역할 없는 스레드(rayon 풀·분할 추론 스레드)
    pub fn report() -> ([f64; 6], f64) {
        let mut ns: [u64; 6] = std::array::from_fn(|k| DONE[k].load(Relaxed));
        for &(tid, role, base) in LIVE.lock().unwrap().iter() {
            ns[role] += clock_ns(of_thread(tid)).saturating_sub(base);
        }
        (ns.map(|v| v as f64 * 1e-9), clock_ns(2) as f64 * 1e-9)
    }
}
