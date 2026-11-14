//! PNG · PLY 바이트. 기억 폴더(pipeline)와 녹화 도구(tools)가 같은 인코더·설정을 쓴다
//! (`PngEncoder::new` = 압축 Default + 거르기 Adaptive, `ImageBuffer::write_to` 와 같은 바이트).
use image::{ExtendedColorType, ImageEncoder};
use std::io::Write;

/// 화소 바이트(16 비트는 이 기계의 바이트 순서) → PNG
pub fn png(w: u32, h: u32, ct: ExtendedColorType, px: &[u8]) -> Vec<u8> {
    let mut v = vec![];
    image::codecs::png::PngEncoder::new(&mut v).write_image(px, w, h, ct).expect("png");
    v
}

pub fn png_rgb(w: u32, h: u32, px: &[u8]) -> Vec<u8> {
    png(w, h, ExtendedColorType::Rgb8, &px[..(w * h * 3) as usize])
}

pub fn png_gray(w: u32, h: u32, px: &[u8]) -> Vec<u8> {
    png(w, h, ExtendedColorType::L8, &px[..(w * h) as usize])
}

/// 16 비트 회색(값 그대로, PNG 안에서는 큰 끝 순서)
pub fn png_u16(w: u32, h: u32, px: &[u16]) -> Vec<u8> {
    let n = (w * h) as usize;
    // 빌린 u16 을 바이트로 그대로 보기(복사 없이, 이 기계의 바이트 순서)
    let b = unsafe { std::slice::from_raw_parts(px[..n].as_ptr() as *const u8, n * 2) };
    png(w, h, ExtendedColorType::L16, b)
}

/// binary_little_endian PLY: float xyz + uchar rgb
pub fn ply(pts: &[([f32; 3], [u8; 3])]) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn png_same_as_image_buffer() {
        let (w, h) = (17u32, 9u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let d: Vec<u16> = (0..w * h).map(|i| (i * 977 % 65521) as u16).collect();
        let mut a = vec![];
        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w, h, rgb.clone()).unwrap().write_to(&mut std::io::Cursor::new(&mut a), image::ImageFormat::Png).unwrap();
        assert_eq!(png_rgb(w, h, &rgb), a);
        let mut b = vec![];
        image::ImageBuffer::<image::Luma<u16>, _>::from_raw(w, h, d.clone()).unwrap().write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png).unwrap();
        assert_eq!(png_u16(w, h, &d), b);
    }
}
