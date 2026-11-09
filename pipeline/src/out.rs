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

/// 비동기 기록: 켜 두면 파일 쓰기(와 PNG 압축)를 기록 스레드가 맡는다(보낸 순서대로 → view.json 은 그 앞 파일들 뒤에 바뀜)
static WRITER: std::sync::OnceLock<std::sync::mpsc::SyncSender<(std::path::PathBuf, Job)>> = std::sync::OnceLock::new();

pub fn start_writer() {
    WRITER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(std::path::PathBuf, Job)>(256);
        std::thread::spawn(move || {
            thread_role(Role::Writer);
            for (p, job) in rx {
                if p.as_os_str().is_empty() {
                    job(); // flush_writer 표시
                    continue;
                }
                run_job(&p, job);
            }
        });
        tx
    });
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
        Some(tx) => tx.send((dst.to_path_buf(), job)).expect("writer"),
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

/// 면이 있는 PLY: 꼭짓점(float xyz + uchar rgb) + 삼각형(list uchar int)
fn ply_mesh_bytes(verts: &[([f32; 3], [u8; 3])], tris: &[[u32; 3]]) -> Vec<u8> {
    let mut b = Vec::with_capacity(verts.len() * 15 + tris.len() * 13 + 256);
    write!(b, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n", verts.len(), tris.len()).unwrap();
    for (p, c) in verts {
        for v in p {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(c);
    }
    for t in tris {
        b.push(3);
        for v in t {
            b.extend_from_slice(&(*v as i32).to_le_bytes());
        }
    }
    b
}

/// 면 PLY: 바이트는 쓰는 쪽(실시간이면 기록 스레드)에서 짠다
pub fn ply_mesh(dst: &Path, verts: Vec<([f32; 3], [u8; 3])>, tris: Vec<[u32; 3]>) {
    write_lazy(dst, Box::new(move || ply_mesh_bytes(&verts, &tris)));
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
    pub const KINDS: [&str; 8] = ["view.json", "pose.json", "map.pgm", "mesh r_", "mesh t_", "obj ply", "obj png", "기타"];
    static N: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    static B: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    static NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    pub fn kind(p: &std::path::Path) -> usize {
        let s = p.to_string_lossy();
        let f = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        if f == "view.json" { 0 } else if f == "pose.json" { 1 } else if f == "map.pgm" { 2 }
        else if f.starts_with("r_") && s.contains("mesh") { 3 } else if f.starts_with("t_") && s.contains("mesh") { 4 }
        else if s.contains("objects") && f.ends_with(".ply") { 5 } else if s.contains("objects") && f.ends_with(".png") { 6 } else { 7 }
    }
    pub fn add(k: usize, bytes: usize, ns: u64) {
        N[k].fetch_add(1, Relaxed);
        B[k].fetch_add(bytes as u64, Relaxed);
        NS[k].fetch_add(ns, Relaxed);
    }
    pub fn report() -> String {
        let mut s = String::from("기록:");
        let (mut tb, mut tt) = (0u64, 0u64);
        for k in 0..8 {
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
    if let Some(tx) = WRITER.get() {
        let (dt, dr) = std::sync::mpsc::channel::<()>();
        let _ = tx.send((std::path::PathBuf::new(), Box::new(move || { let _ = dt.send(()); vec![] })));
        let _ = dr.recv();
    }
}

/// 스레드 역할: 우선순위(nice)·코어 묶기를 역할마다 정한다(`--threads`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role { Main, Reader, Points, Texture, Rebuild, Writer, Infer }
pub const ROLE_NAMES: [&str; 7] = ["main", "reader", "pts", "tex", "rebuild", "writer", "infer"];
/// 역할별 기본 nice(묶지 않음)
const NICE: [(i32, u64); 7] = [(0, 0), (0, 0), (5, 0), (10, 0), (10, 0), (5, 0), (0, 0)];

/// 역할별 (nice, 코어 묶음 비트). 묶음 0 = 프로세스가 받은 코어 전부
static LAYOUT: std::sync::OnceLock<[(i32, u64); 7]> = std::sync::OnceLock::new();
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

/// `--threads` 해석. 기본(auto): nice 는 본 스레드·읽기·추론 0, 점 +5, 기록 +5, 텍스처·다시 쌓기 +10.
/// 코어 4 개 이상이면 본 스레드·읽기(분할 전·후처리)·추론(모델을 부르는 스레드, PC 는 ONNX Runtime 스레드도)을 앞 코어 둘에, 점·텍스처·다시 쌓기·기록을 나머지 코어에 묶는다.
/// 코어 3 개 이하는 묶지 않는다(nice 로만 나눔). 직접 주기: `main=0:0,pts=5:2-3,tex=10:2-3` (역할=nice[:코어 목록]), `none` = 모두 nice 0·묶지 않음
pub fn set_layout(spec: &str) {
    let cpus = allowed_cpus();
    let bits = |v: &[usize]| v.iter().fold(0u64, |m, &c| if c < 64 { m | 1 << c } else { m });
    let mut l = NICE;
    match spec {
        "none" => l = [(0, 0); 7],
        "auto" => {
            if cpus.len() >= 4 {
                let (fg, bg) = (bits(&cpus[..2]), bits(&cpus[2..]));
                l = [(0, fg), (0, fg), (5, bg), (10, bg), (10, bg), (5, bg), (0, fg)];
            }
        }
        s => {
            for part in s.split(',') {
                let (name, rest) = part.split_once('=').expect("--threads: 역할=nice[:코어]");
                let i = ROLE_NAMES.iter().position(|n| *n == name).expect("--threads: 역할 이름(main reader pts tex rebuild writer infer)");
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
    static DONE: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7]; // 역할을 내려놓았거나 끝난 스레드 몫(ns)
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
    pub fn report() -> ([f64; 7], f64) {
        let mut ns: [u64; 7] = std::array::from_fn(|k| DONE[k].load(Relaxed));
        for &(tid, role, base) in LIVE.lock().unwrap().iter() {
            ns[role] += clock_ns(of_thread(tid)).saturating_sub(base);
        }
        (ns.map(|v| v as f64 * 1e-9), clock_ns(2) as f64 * 1e-9)
    }
}
