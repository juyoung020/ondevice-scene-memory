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

pub fn write_atomic(dst: &Path, bytes: &[u8]) {
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
