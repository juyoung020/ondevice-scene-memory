//! 카메라 정보·보정값(librealsense 없이). rs_info 의 보정값과 견줄 수 있게 JSON 으로, f32 는 비트(16진)도 같이.
//!   rsd_info [--size 640x480] [--serial S] [--dump 폴더]   (--dump: HWM 원래 바이트 gvd·coeff·rgb·new·depth_table .bin 저장)
#[cfg(target_os = "linux")]
fn main() {
    use rsdriver::camera::*;
    use rsdriver::{Extrinsics, Intrinsics};
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let (w, h) = arg("--size").and_then(|s| tools::common::parse_wh(&s)).unwrap_or((640, 480));
    let serial = arg("--serial").unwrap_or_default();
    let list = discover();
    if list.is_empty() {
        eprintln!("rsd_info: D435 없음");
        std::process::exit(2);
    }
    let hx = |v: f32| format!("{:08x}", v.to_bits());
    let ij = |k: &Intrinsics| {
        format!(
            "{{\"width\": {}, \"height\": {}, \"ppx\": {} , \"ppy\": {}, \"fx\": {}, \"fy\": {}, \"model\": {}, \"coeffs\": {:?}, \"bits\": [\"{}\", \"{}\", \"{}\", \"{}\", {}]}}",
            k.width, k.height, k.ppx, k.ppy, k.fx, k.fy, k.model, k.coeffs,
            hx(k.ppx), hx(k.ppy), hx(k.fx), hx(k.fy),
            k.coeffs.iter().map(|c| format!("\"{}\"", hx(*c))).collect::<Vec<_>>().join(", ")
        )
    };
    let ej = |e: &Extrinsics| {
        format!(
            "{{\"rotation\": {:?}, \"translation\": {:?}, \"bits\": [{}]}}",
            e.rotation, e.translation,
            e.rotation.iter().chain(e.translation.iter()).map(|c| format!("\"{}\"", hx(*c))).collect::<Vec<_>>().join(", ")
        )
    };
    for f in list {
        let n = match rsdriver::v4l2::Node::open(&f.depth) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{}: {e}", f.depth);
                continue;
            }
        };
        let raw = match read_raw_calib(&n) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: HWM 실패: {e}", f.depth);
                continue;
            }
        };
        let c = match parse_calib(&raw, w as u32, h as u32, w as u32, h as u32) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("보정값 해석 실패: {e}");
                continue;
            }
        };
        if !serial.is_empty() && c.gvd.serial != serial {
            continue;
        }
        if let Some(d) = arg("--dump") {
            let _ = std::fs::create_dir_all(&d);
            for (name, b) in [("gvd", &raw.gvd), ("coeff", &raw.coeff), ("rgb", &raw.rgb), ("new", &raw.new_table), ("depth_table", &raw.depth_table)] {
                let _ = std::fs::write(format!("{d}/{name}.bin"), b);
            }
        }
        println!(
            "{{\n  \"serial\": \"{}\", \"usb_serial\": \"{}\", \"firmware\": \"{}\", \"usb_speed\": \"{}\",\n  \"nodes\": {{\"depth\": \"{}\", \"depth_meta\": {:?}, \"color\": \"{}\", \"color_meta\": {:?}}},\n  \"advanced_mode\": {},\n  \"depth_intrinsics\": {},\n  \"color_intrinsics\": {},\n  \"depth_to_color\": {},\n  \"depth_scale\": {}, \"depth_scale_bits\": \"{}\"\n}}",
            c.gvd.serial, f.usb_serial, c.gvd.firmware, f.speed, f.depth, f.depth_meta, f.color, f.color_meta, raw.adv_mode,
            ij(&c.depth), ij(&c.color), ej(&c.depth_to_color), c.depth_scale, hx(c.depth_scale)
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rsd_info: 리눅스 전용");
    std::process::exit(2);
}
