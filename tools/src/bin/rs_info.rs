//! RealSense(D435·D455) 정보를 JSON 으로: 장치·펌웨어·USB, 센서별 스트림 목록, IMU 유무, 깊이 단위, 고른 해상도의 깊이·컬러 내부 보정값과
//! 깊이→컬러 외부 변환(스트리밍 없이 프로파일에서 읽음).
//!   rs_info [--depth 640x480] [--color 640x480] [--fps 30] [> rs_info.json]
//! 장치가 없으면 {"devices": [], "error": ...} 를 내고 끝 코드 2.
use std::collections::HashSet;
use tools::common::*;
use tools::fast_align::{fov_deg, DepthAligner, Extrinsics, Intrinsics, Options};
use tools::rs2::*;

fn info(dev: P, k: i32) -> Result<String, Error> {
    Ok(if ck(|e| unsafe { rs2_supports_device_info(dev, k, e) })? != 0 { device_info(dev, k)? } else { String::new() })
}

fn run(dw: i32, dh: i32, cw: i32, ch: i32, fps: i32) -> Result<i32, Error> {
    let ctx = ck(|e| unsafe { rs2_create_context(25901, e) })?;
    let list = ck(|e| unsafe { rs2_query_devices(ctx, e) })?;
    let count = ck(|e| unsafe { rs2_get_device_count(list, e) })?;
    let mut js = format!("{{\n  \"librealsense\": {},\n", jstr(&api_version()));
    if count == 0 {
        js += "  \"devices\": [],\n  \"error\": \"장치 없음: RealSense 를 USB3 포트에 꽂고 다시\"\n}\n";
        print!("{js}");
        eprintln!("rs_info: 장치 없음");
        return Ok(2);
    }
    js += "  \"devices\": [\n";
    for di in 0..count {
        let dev = ck(|e| unsafe { rs2_create_device(list, di, e) })?;
        js += "    {\n";
        for (k, key) in [
            (INFO_NAME, "name"),
            (INFO_SERIAL, "serial"),
            (INFO_FIRMWARE, "firmware"),
            (INFO_RECOMMENDED_FIRMWARE, "recommended_firmware"),
            (INFO_USB_TYPE, "usb"),
            (INFO_PRODUCT_LINE, "product_line"),
            (INFO_PRODUCT_ID, "product_id"),
        ] {
            js += &format!("      \"{key}\": {},\n", jstr(&info(dev, k)?));
        }
        let mut imu = false;
        let mut depth_unit = 0.0f32;
        let (mut dprof, mut cprof): (P, P) = (std::ptr::null_mut(), std::ptr::null_mut());
        js += "      \"sensors\": [\n";
        let sensors = ck(|e| unsafe { rs2_query_sensors(dev, e) })?;
        let ns = ck(|e| unsafe { rs2_get_sensors_count(sensors, e) })?;
        for si in 0..ns {
            let s = ck(|e| unsafe { rs2_create_sensor(sensors, si, e) })?;
            let name = if ck(|e| unsafe { rs2_supports_sensor_info(s, INFO_NAME, e) })? != 0 { cstr(ck(|e| unsafe { rs2_get_sensor_info(s, INFO_NAME, e) })?) } else { String::new() };
            js += &format!("        {{\"name\": {}, ", jstr(&name));
            if ck(|e| unsafe { rs2_is_sensor_extendable_to(s, EXT_DEPTH_SENSOR, e) })? != 0 {
                depth_unit = ck(|e| unsafe { rs2_get_depth_scale(s, e) })?;
                js += &format!("\"depth_unit_m\": {}, ", jnum(depth_unit as f64));
            }
            // 지금 옵션 값 몇 개
            js += "\"options\": {";
            let mut first = true;
            for o in [OPT_EMITTER_ENABLED, OPT_LASER_POWER, OPT_VISUAL_PRESET, OPT_ENABLE_AUTO_EXPOSURE, OPT_EXPOSURE, OPT_GAIN, OPT_GLOBAL_TIME_ENABLED] {
                if ck(|e| unsafe { rs2_supports_option(s, o, e) })? == 0 {
                    continue;
                }
                // 원본과 같이 이름까지 쓴 뒤 값 읽기가 실패하면 값 없이 넘어감
                js += &format!("{}{}: ", if first { "" } else { ", " }, jstr(&cstr(unsafe { rs2_option_to_string(o) })));
                if let Ok(v) = ck(|e| unsafe { rs2_get_option(s, o, e) }) {
                    js += &jnum(v as f64);
                    first = false;
                }
            }
            js += "}, \"streams\": [";
            let mut seen = HashSet::new();
            let mut firsts = true;
            let profs = ck(|e| unsafe { rs2_get_stream_profiles(s, e) })?;
            for pi in 0..ck(|e| unsafe { rs2_get_stream_profiles_count(profs, e) })? {
                let p = ck(|e| unsafe { rs2_get_stream_profile(profs, pi, e) })?;
                let (mut st, mut fm, mut ix, mut uid, mut pf) = (0, 0, 0, 0, 0);
                ck(|e| unsafe { rs2_get_stream_profile_data(p, &mut st, &mut fm, &mut ix, &mut uid, &mut pf, e) })?;
                if st == STREAM_GYRO || st == STREAM_ACCEL || st == STREAM_MOTION {
                    imu = true;
                }
                let mut one = format!("\"{} {} {}", cstr(unsafe { rs2_stream_to_string(st) }), ix, cstr(unsafe { rs2_format_to_string(fm) }));
                if ck(|e| unsafe { rs2_stream_profile_is(p, EXT_VIDEO_PROFILE, e) })? != 0 {
                    let (mut w, mut h) = (0, 0);
                    ck(|e| unsafe { rs2_get_video_stream_resolution(p, &mut w, &mut h, e) })?;
                    one += &format!(" {w}x{h}");
                    if st == STREAM_DEPTH && fm == FORMAT_Z16 && w == dw && h == dh && pf == fps {
                        dprof = p;
                    }
                    if st == STREAM_COLOR && fm == FORMAT_RGB8 && w == cw && h == ch && pf == fps {
                        cprof = p;
                    }
                }
                one += &format!(" @{pf}\"");
                if seen.insert(one.clone()) {
                    js += if firsts { "" } else { ", " };
                    js += &one;
                    firsts = false;
                }
            }
            js += if si + 1 < ns { "]},\n" } else { "]}\n" };
        }
        js += "      ],\n";
        js += &format!("      \"imu\": {},\n", imu);
        js += &format!("      \"depth_unit_m\": {},\n", jnum(depth_unit as f64));
        js += &format!("      \"chosen\": {{\"depth\": \"{dw}x{dh}\", \"color\": \"{cw}x{ch}\", \"fps\": {fps}}},\n");
        if !dprof.is_null() && !cprof.is_null() {
            let (mut dk, mut ck_) = (Intrinsics::default(), Intrinsics::default());
            let mut ex = Extrinsics::default();
            ck(|e| unsafe { rs2_get_video_stream_intrinsics(dprof, &mut dk, e) })?;
            ck(|e| unsafe { rs2_get_video_stream_intrinsics(cprof, &mut ck_, e) })?;
            ck(|e| unsafe { rs2_get_extrinsics(dprof, cprof, &mut ex, e) })?;
            js += &format!("      \"depth_intrinsics\": {},\n", tools::rs2::intrin_json(&dk));
            js += &format!("      \"color_intrinsics\": {},\n", tools::rs2::intrin_json(&ck_));
            js += &format!("      \"depth_to_color\": {},\n", extrin_json(&ex));
            let a = DepthAligner::new(&dk, &ck_, &ex, depth_unit, &Options::default())?;
            let (fd, fc) = (fov_deg(&dk), fov_deg(&ck_));
            js += &format!(
                "      \"fov_deg\": {{\"depth\": [{}, {}], \"color\": [{}, {}]}},\n",
                jnum(fd[0] as f64),
                jnum(fd[1] as f64),
                jnum(fc[0] as f64),
                jnum(fc[1] as f64)
            );
            js += &format!("      \"sdk_align_fills_rect\": {}\n", a.uses_rect());
        } else {
            js += "      \"calibration_error\": \"고른 해상도·프레임의 Z16 깊이 또는 RGB8 컬러 스트림이 없음\"\n";
        }
        js += if di + 1 < count { "    },\n" } else { "    }\n" };
    }
    js += "  ]\n}\n";
    print!("{js}");
    Ok(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut dw, mut dh, mut cw, mut ch, mut fps) = (640, 480, 640, 480, 30);
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--depth" => {
                if let Some((w, h)) = parse_wh(&args[i + 1]) {
                    (dw, dh) = (w, h);
                }
            }
            "--color" => {
                if let Some((w, h)) = parse_wh(&args[i + 1]) {
                    (cw, ch) = (w, h);
                }
            }
            "--fps" => fps = atoi(&args[i + 1]),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    match run(dw, dh, cw, ch, fps) {
        Ok(rc) => std::process::exit(rc),
        Err(Error::Rs { what, .. }) => {
            println!("{{\"error\": {}}}", jstr(&format!("librealsense: {what}")));
            std::process::exit(3);
        }
        Err(Error::Other(m)) => {
            println!("{{\"error\": {}}}", jstr(&m));
            std::process::exit(3);
        }
    }
}
