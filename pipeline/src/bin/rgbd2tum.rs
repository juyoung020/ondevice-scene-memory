//! RGB-D 영상(다른 카메라)을 TUM fr3 보정값(640×480, fx 535.4 fy 539.2 cx 320.1 cy 247.6, 깊이 5000/m)으로 다시 그려
//! TUM 형식 폴더로: rgb_sync/·depth_sync/(시각.png, RTAB-Map rgbd_dataset 도구용) + rgb.txt·depth.txt(우리 memmap 용).
//!   rgbd2tum <원본 폴더(association.txt)> <출력 폴더(이름에 freiburg3 포함)> fx fy cx cy <깊이 배율(1 m 값, 예: 1000)> [정답 위치 파일]
//! 화면 위치는 보정값 비율로 옮기고(컬러 쌍선형, 깊이 가까운 화소), 원본 시야 밖은 검정·깊이 0. 흑백 컬러는 3 채널로.
use image::{ImageBuffer, Luma, Rgb, RgbImage};
use std::io::Write;
use std::path::Path;

const T: [f64; 4] = [535.4, 539.2, 320.1, 247.6];
const W: u32 = 640;
const H: u32 = 480;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        eprintln!("rgbd2tum <src> <dst> fx fy cx cy depth_per_m [groundtruth]");
        std::process::exit(1);
    }
    let (src, dst) = (Path::new(&a[1]), Path::new(&a[2]));
    let k: Vec<f64> = a[3..7].iter().map(|s| s.parse().unwrap()).collect();
    let dper: f64 = a[7].parse().unwrap();
    std::fs::create_dir_all(dst.join("rgb_sync")).unwrap();
    std::fs::create_dir_all(dst.join("depth_sync")).unwrap();
    let assoc = std::fs::read_to_string(src.join("association.txt")).or_else(|_| std::fs::read_to_string(src.join("associations.txt"))).expect("association.txt");
    let (mut frgb, mut fdep) = (String::new(), String::new());
    let (sx, sy) = (T[0] / k[0], T[1] / k[1]);
    let mut n = 0;
    for l in assoc.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let v: Vec<&str> = l.split_whitespace().collect();
        let (t, rp, dp) = (v[0].parse::<f64>().unwrap(), v[1], v[3]);
        let img = image::open(src.join(rp)).expect("rgb");
        let (rgb, w, h) = {
            let im = img.to_rgb8();
            let (w, h) = im.dimensions();
            (im.into_raw(), w as usize, h as usize)
        };
        let dep: ImageBuffer<Luma<u16>, Vec<u16>> = image::open(src.join(dp)).expect("depth").to_luma16();
        let mut orgb = RgbImage::new(W, H);
        let mut odep: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(W, H);
        for y in 0..H {
            let fy = (y as f64 - T[3]) / sy + k[3];
            for x in 0..W {
                let fx = (x as f64 - T[2]) / sx + k[2];
                if fx < 0.0 || fy < 0.0 || fx >= (w - 1) as f64 || fy >= (h - 1) as f64 {
                    continue;
                }
                let (x0, y0) = (fx as usize, fy as usize);
                let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
                let px = |xx: usize, yy: usize, c: usize| rgb[(yy * w + xx) * 3 + c] as f64;
                let c = [0, 1, 2].map(|c| {
                    let top = px(x0, y0, c) * (1.0 - ax) + px(x0 + 1, y0, c) * ax;
                    let bot = px(x0, y0 + 1, c) * (1.0 - ax) + px(x0 + 1, y0 + 1, c) * ax;
                    (top * (1.0 - ay) + bot * ay).round() as u8
                });
                orgb.put_pixel(x, y, Rgb(c));
                let d = dep.get_pixel(fx.round() as u32, fy.round() as u32)[0] as f64 / dper; // 미터
                odep.put_pixel(x, y, Luma([(d * 5000.0).round().clamp(0.0, 65535.0) as u16]));
            }
        }
        let name = format!("{t:.6}.png");
        orgb.save(dst.join("rgb_sync").join(&name)).unwrap();
        odep.save(dst.join("depth_sync").join(&name)).unwrap();
        frgb += &format!("{t:.6} rgb_sync/{name}\n");
        fdep += &format!("{t:.6} depth_sync/{name}\n");
        n += 1;
        if n % 200 == 0 {
            eprintln!("{n}");
        }
    }
    std::fs::write(dst.join("rgb.txt"), frgb).unwrap();
    std::fs::write(dst.join("depth.txt"), fdep).unwrap();
    if let Some(gt) = a.get(8) {
        std::fs::copy(gt, dst.join("groundtruth.txt")).unwrap();
    }
    let mut f = std::fs::File::create(dst.join("README.txt")).unwrap();
    writeln!(f, "rgbd2tum: {} → TUM fr3 보정값 640×480, 깊이 5000/m, {n} 장", src.display()).unwrap();
    println!("{n} frames → {}", dst.display());
}
