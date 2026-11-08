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
            for (p, job) in rx {
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
    stats::add(stats::kind(dst), b.len(), c0.elapsed().as_nanos() as u64);
}

fn write_now(dst: &Path, bytes: &[u8]) {
    let tmp = dst.with_extension("tmp");
    std::fs::write(&tmp, bytes).expect("write");
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

/// 지금 스레드를 뒤로(nice +10): 다시 쌓기·텍스처 같은 작업 스레드가 2 코어 보드에서 장면 처리 스레드와 코어를 다툴 때
/// 장면 처리가 먼저 돌게 한다(작업 스레드는 남는 시간에 끝남). 리눅스만, 실패해도 그대로 진행.
pub fn background_thread() {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
            fn gettid() -> i32;
        }
        setpriority(0, gettid() as u32, 10); // PRIO_PROCESS + 스레드 id = 그 스레드만
    }
}
