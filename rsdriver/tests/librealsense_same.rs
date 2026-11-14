//! librealsense v2.59.1 의 실제 함수(WSL 빌드, get_d400_intrinsic_by_resolution · get_d400_color_stream_intrinsic ·
//! get_d400_color_stream_extrinsic + from_pose/inverse · calc_crc32 · unpack_yuy2(SSE) · image-neon.cpp(aarch64, qemu))가
//! 만든 결과를 tests/data 에 넣어 두고, 같은 입력에서 우리 결과가 비트 단위로 같은지 본다.
//! 표는 그럴듯한 D435 값에 난수를 섞은 것(계수표 576 · RGB 표 256 · 새 깊이 표 0 또는 80 바이트, CRC 맞춤).
use rsdriver::calib::*;
use rsdriver::format::*;

const RES: [(u32, u32); 14] = [(640, 480), (848, 480), (1280, 720), (640, 360), (424, 240), (848, 100), (256, 144), (1024, 768), (1920, 1080), (320, 240), (1280, 800), (640, 400), (480, 270), (960, 540)];

fn data(n: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{n}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}
fn ib(k: &Intrinsics, o: &mut Vec<u8>) {
    o.extend(k.width.to_le_bytes());
    o.extend(k.height.to_le_bytes());
    for f in [k.ppx, k.ppy, k.fx, k.fy] {
        o.extend(f.to_bits().to_le_bytes());
    }
    o.extend(k.model.to_le_bytes());
    for f in k.coeffs {
        o.extend(f.to_bits().to_le_bytes());
    }
}
fn eb(e: &Extrinsics, o: &mut Vec<u8>) {
    for f in e.rotation.iter().chain(e.translation.iter()) {
        o.extend(f.to_bits().to_le_bytes());
    }
}

#[test]
fn calibration_bit_identical() {
    let mut o = Vec::new();
    for k in 0..6 {
        let co = data(&format!("c{k:02}_coeff.bin"));
        let rgb = data(&format!("c{k:02}_rgb.bin"));
        let nw = data(&format!("c{k:02}_new.bin"));
        for &(w, h) in &RES {
            ib(&depth_intrinsics(&nw, &co, w, h).unwrap(), &mut o);
            ib(&color_intrinsics(&rgb, w, h).unwrap(), &mut o);
        }
        eb(&color_to_depth(&rgb).unwrap(), &mut o);
        eb(&depth_to_color(&rgb).unwrap(), &mut o);
    }
    let exp = data("expected_calib.bin");
    assert_eq!(o.len(), exp.len());
    let bad = o.chunks(4).zip(exp.chunks(4)).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "다른 4 바이트 값 {bad} 개");
}

#[test]
fn depth_scale_and_crc_identical() {
    let mut o = Vec::new();
    for u in [1u32, 50, 100, 250, 500, 1000, 2500, 10000, 100000, 123, 999] {
        o.extend(depth_scale_from_units(u).to_bits().to_le_bytes());
    }
    for n in [0, 1, 7, 240, 560, 1000] {
        o.extend(crc32(&data(&format!("crc{n}.bin"))).to_le_bytes());
    }
    assert_eq!(o, data("expected_scale_crc.bin"));
}

#[test]
fn calib_rejects_bad_crc() {
    let mut rgb = data("c00_rgb.bin");
    rgb[100] ^= 1;
    assert!(color_intrinsics(&rgb, 640, 480).is_err());
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// 모든 (Y, U, V) 조합: 4096×8192 화소. 기대 해시 = librealsense 출력 파일의 FNV-1a
#[test]
fn yuy2_rgb8_all_combinations() {
    let (w, h) = (4096usize, 8192usize);
    let mut s = vec![0u8; w * h * 2];
    let mut i = 0;
    for y in 0..256u32 {
        for u in 0..256u32 {
            for v in 0..256u32 {
                s[i] = y as u8;
                s[i + 1] = u as u8;
                s[i + 2] = ((y * 7 + v) & 255) as u8;
                s[i + 3] = v as u8;
                i += 4;
            }
        }
    }
    let mut d = vec![0u8; w * h * 3];
    yuyv_to_rgb8(&s, w * 2, w, &mut d, Yuy2Rule::Sse);
    assert_eq!(fnv(&d), 0x19a4d6232cb9901f, "SSE(x86 librealsense unpack_yuy2) 와 다름");
    yuyv_to_rgb8(&s, w * 2, w, &mut d, Yuy2Rule::Neon);
    assert_eq!(fnv(&d), 0x18632b5ea37acb77, "NEON(image-neon.cpp, aarch64 qemu) 과 다름");
}

#[test]
fn hwm_command_layout() {
    let b = rsdriver::hwm::build(0x15, [32, 0, 0, 0], &[]);
    assert_eq!(b.len(), 1024);
    assert_eq!(&b[..12], &[20, 0, 0xab, 0xcd, 0x15, 0, 0, 0, 32, 0, 0, 0]);
    let mut r = vec![0u8; 1024];
    r[0] = 0x15;
    r[4..8].copy_from_slice(&[1, 2, 3, 4]);
    r[1020..1024].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(rsdriver::hwm::parse(0x15, &r).unwrap(), vec![1, 2, 3, 4]);
    r[0..4].copy_from_slice(&(-7i32).to_le_bytes());
    assert!(rsdriver::hwm::parse(0x15, &r).is_err());
}

#[test]
fn sync_rule() {
    use rsdriver::sync::*;
    assert_eq!(decide(100.0, 116.0, 30.0), Decision::Pair);
    assert_eq!(decide(100.0, 117.0, 30.0), Decision::DropColor);
    assert_eq!(decide(140.0, 100.0, 30.0), Decision::DropDepth);
}

#[test]
fn metadata_color() {
    use rsdriver::meta::*;
    // uvc_meta_buf: ns, sof, 머리(길이, 정보=PTS 있음, PTS, SCR), capture_timing, capture_stats(80), rgb_control
    let mut b = vec![0u8; 10];
    b[0..8].copy_from_slice(&123u64.to_le_bytes());
    let mut h = vec![0u8; 12];
    h[1] = 0x0c;
    h[2..6].copy_from_slice(&0xfffffff0u32.to_le_bytes());
    let mut t = Vec::new();
    for v in [MD_CAPTURE_TIMING, 32, 1, 3, 77, 555] {
        t.extend((v as u32).to_le_bytes());
    }
    t.resize(40, 0);
    h.extend(&t);
    h.resize(132, 0);
    let mut c = Vec::new();
    for v in [MD_RGB_CONTROL, 64, 1, (1 << 6) | (1 << 10), 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 333] {
        c.extend((v as u32).to_le_bytes());
    }
    h.extend(&c);
    h[0] = h.len() as u8;
    b.extend(&h);
    let m = parse_meta_buf(&b);
    assert_eq!(m.host_ns, 123);
    assert_eq!(m.frame_ts_us, Some(0xfffffff0));
    assert_eq!(m.frame_counter, Some(77));
    assert_eq!(m.sensor_ts_us, Some(555));
    assert_eq!(m.gain, Some(64));
    assert_eq!(m.actual_exposure, Some(333));
    let mut u = Unwrap::default();
    assert_eq!(u.push(0xfffffff0), 0xfffffff0);
    assert_eq!(u.push(5), (1u64 << 32) + 5);
}

/// ioctl 번호·상수가 리눅스 헤더(linux/videodev2.h, linux/uvcvideo.h; x86_64·aarch64 같음)와 같음
#[cfg(target_os = "linux")]
#[test]
fn v4l2_numbers_match_kernel_headers() {
    use rsdriver::v4l2::*;
    let got = [VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_QUERYBUF, VIDIOC_QBUF, VIDIOC_DQBUF, VIDIOC_STREAMON, VIDIOC_STREAMOFF, VIDIOC_S_PARM, VIDIOC_G_CTRL, VIDIOC_S_CTRL, UVCIOC_CTRL_QUERY];
    let exp: [u64; 12] = [0x80685600, 0xc0d05605, 0xc0145608, 0xc0585609, 0xc058560f, 0xc0585611, 0x40045612, 0x40045613, 0xc0cc5616, 0xc008561b, 0xc008561c, 0xc0107521];
    for (g, e) in got.iter().zip(exp) {
        assert_eq!(*g as u64, e);
    }
    assert_eq!([CID_GAIN, CID_EXPOSURE_AUTO, CID_EXPOSURE_ABSOLUTE, CID_EXPOSURE_AUTO_PRIORITY, CAP_META_CAPTURE, FMT_META_UVC], [0x980913, 0x9a0901, 0x9a0902, 0x9a0903, 0x800000, 0x48435655]);
    assert_eq!(std::mem::offset_of!(Buffer, sequence), 56);
    assert_eq!(std::mem::offset_of!(Buffer, offset), 64);
}
