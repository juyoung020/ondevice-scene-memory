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

/// 비동기 기록: 켜 두면 파일 쓰기를 기록 스레드가 맡는다(보낸 순서대로 → view.json 은 그 앞 파일들 뒤에 바뀜)
static WRITER: std::sync::OnceLock<std::sync::mpsc::SyncSender<(std::path::PathBuf, Vec<u8>)>> = std::sync::OnceLock::new();

pub fn start_writer() {
    WRITER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(std::path::PathBuf, Vec<u8>)>(256);
        std::thread::spawn(move || {
            for (p, b) in rx {
                write_now(&p, &b);
            }
        });
        tx
    });
}

pub fn write_atomic(dst: &Path, bytes: &[u8]) {
    match WRITER.get() {
        Some(tx) => tx.send((dst.to_path_buf(), bytes.to_vec())).expect("writer"),
        None => write_now(dst, bytes),
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

pub fn png_rgb(dst: &Path, w: u32, h: u32, rgb: &[u8]) {
    let mut v = vec![];
    image::RgbImage::from_raw(w, h, rgb.to_vec()).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
    write_atomic(dst, &v);
}
pub fn png_gray(dst: &Path, w: u32, h: u32, g: &[u8]) {
    let mut v = vec![];
    image::GrayImage::from_raw(w, h, g.to_vec()).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
    write_atomic(dst, &v);
}
pub fn png_u16(dst: &Path, w: u32, h: u32, d: &[u16]) {
    let mut v = vec![];
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w, h, d.to_vec()).unwrap().write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
    write_atomic(dst, &v);
}
