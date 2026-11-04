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

/// 기록할 내용: 바로 쓸 바이트, 또는 기록 스레드에서 만들 바이트(PNG 압축 등 무거운 것)
type Job = Box<dyn FnOnce() -> Vec<u8> + Send>;

/// 비동기 기록: 켜 두면 파일 쓰기(와 PNG 압축)를 기록 스레드가 맡는다(보낸 순서대로 → view.json 은 그 앞 파일들 뒤에 바뀜)
static WRITER: std::sync::OnceLock<std::sync::mpsc::SyncSender<(std::path::PathBuf, Job)>> = std::sync::OnceLock::new();

pub fn start_writer() {
    WRITER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(std::path::PathBuf, Job)>(256);
        std::thread::spawn(move || {
            for (p, job) in rx {
                write_now(&p, &job());
            }
        });
        tx
    });
}

pub fn write_atomic(dst: &Path, bytes: &[u8]) {
    let b = bytes.to_vec();
    write_lazy(dst, Box::new(move || b));
}

/// 바이트를 기록 스레드에서 만들어 쓴다(기록 스레드가 없으면 바로)
fn write_lazy(dst: &Path, job: Job) {
    match WRITER.get() {
        Some(tx) => tx.send((dst.to_path_buf(), job)).expect("writer"),
        None => write_now(dst, &job()),
    }
}

fn write_now(dst: &Path, bytes: &[u8]) {
    let tmp = dst.with_extension("tmp");
    std::fs::write(&tmp, bytes).expect("write");
    replace(&tmp, dst);
}

/// binary_little_endian PLY: float xyz + uchar rgb
pub fn ply(dst: &Path, pts: &[([f32; 3], [u8; 3])]) {
    let mut b = Vec::with_capacity(pts.len() * 15 + 200);
    write!(b, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", pts.len()).unwrap();
    for (p, c) in pts {
        for v in p {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(c);
    }
    write_atomic(dst, &b);
}

/// 면이 있는 PLY: 꼭짓점(float xyz + uchar rgb) + 삼각형(list uchar int)
pub fn ply_mesh(dst: &Path, verts: &[([f32; 3], [u8; 3])], tris: &[[u32; 3]]) {
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
    write_atomic(dst, &b);
}

pub fn pgm(dst: &Path, w: usize, h: usize, px: &[u8]) {
    let mut b = format!("P5\n{w} {h}\n255\n").into_bytes();
    b.extend_from_slice(px);
    write_atomic(dst, &b);
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
