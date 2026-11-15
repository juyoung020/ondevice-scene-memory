//! 컬러 1280×720 + 깊이 640×480(`Config::color_width/height`): 정렬은 고른 컬러 모드의 내부 보정값을 쓴다.
//! 보정 표(tests/data c00..c05, 1280×720 컬러 보정값은 librealsense_same 이 librealsense 와 비트 비교)로 fast_align(640×480 → 1280×720)이 옮겨 온 librealsense 정렬(SIMD 경로 · 일반 경로)과 화소까지 같은지 본다.
use rsdriver::calib::*;
use tools::fast_align::{DepthAligner, Method, Options};
use tools::rsref;

fn data(n: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{n}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// 벽 · 상자 · 구멍이 섞인 깊이(mm, 단위 0.001)
fn scene(w: usize, h: usize) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if (x / 37 + y / 29) % 11 == 0 {
                0
            } else if x > w / 3 && x < w / 2 && y > h / 4 && y < h / 2 {
                700 + (x as u16 % 13)
            } else {
                1500 + ((x * 3 + y * 5) % 900) as u16
            }
        })
        .collect()
}

#[test]
fn align_640x480_depth_to_1280x720_color() {
    let (dw, dh) = (640usize, 480usize);
    let z = scene(dw, dh);
    for k in 0..6 {
        let co = data(&format!("c{k:02}_coeff.bin"));
        let rgb = data(&format!("c{k:02}_rgb.bin"));
        let nw = data(&format!("c{k:02}_new.bin"));
        let d = depth_intrinsics(&nw, &co, dw as u32, dh as u32).unwrap();
        let c = color_intrinsics(&rgb, 1280, 720).unwrap();
        let e = depth_to_color(&rgb).unwrap();
        let n = (c.width * c.height) as usize;
        // 실제 정렬 경로(Sdk = rs2::align SIMD 규칙) ↔ 옮겨 온 librealsense SIMD 경로
        let mut fa = DepthAligner::new(&d, &c, &e, 0.001, &Options::default()).unwrap();
        let mut ours = vec![0u16; n];
        fa.process(&z, &mut ours);
        let mut simd = rsref::SimdAlign::new(&d, 0.001);
        let mut refo = vec![0u16; n];
        simd.align_depth_to_other(&z, &mut refo, &c, &e);
        let bad = ours.iter().zip(&refo).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "c{k:02} Sdk ↔ librealsense SIMD: 다른 화소 {bad}");
        // 일반 경로
        let mut fg = DepthAligner::new(&d, &c, &e, 0.001, &Options { method: Method::Generic, ..Default::default() }).unwrap();
        fg.process(&z, &mut ours);
        rsref::align_z_to_other_generic(&mut refo, &z, 0.001, &d, &e, &c);
        let bad = ours.iter().zip(&refo).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "c{k:02} Generic ↔ librealsense 일반: 다른 화소 {bad}");
        assert!(ours.iter().any(|&v| v != 0), "c{k:02} 정렬 결과가 비었음");
    }
}
