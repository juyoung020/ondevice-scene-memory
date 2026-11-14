//! 카메라 없이 SDK 의 실제 rs2::align(이 기계의 realsense2 라이브러리)과 fast_align 을 같은 합성 장면에서 비교한다.
//! software_device 에 깊이·컬러 프레임과 보정값을 넣어 rs2::align 을 돌리고, 결과 화소 차이와 시간을 잰다.
//!   align_sdk_check [--calib <녹화 폴더/calib.json>] [--frames N] [--core C] [--threads T] [--write-rec <폴더>]
//! --write-rec: 640x480 합성 장면을 rs_rec 과 같은 폴더 모양(원래 깊이·rs2::align 결과·calib.json)으로 써서
//!              카메라 없이 rs_align_check 를 시험할 수 있게 한다.
//! --calib 이 없으면 D455 비슷한 가정 보정값(scenes)으로 640x480·848x480 깊이 두 가지.
use std::fs::File;
use std::io::Write;
use tools::common::*;
use tools::fast_align::{DepthAligner, Extrinsics, Intrinsics, Options};
use tools::rs2::{self, Error};
use tools::scenes;

#[allow(clippy::too_many_arguments)]
fn run(name: &str, di: &Intrinsics, ci: &Intrinsics, e: &Extrinsics, unit: f32, frames: i32, threads: i32, core: i32, rec: &str) -> Result<i32, Error> {
    let mut lists: Option<[File; 5]> = None;
    if !rec.is_empty() {
        for sub in ["rgb", "depth", "depth_raw", "depth_rs"] {
            let _ = std::fs::create_dir_all(format!("{rec}/{sub}"));
        }
        let mk = |n: &str| File::create(format!("{rec}/{n}")).map_err(|x| Error::Other(format!("{rec}/{n}: {x}")));
        lists = Some([mk("depth_raw.txt")?, mk("depth_rs.txt")?, mk("depth.txt")?, mk("rgb.txt")?, mk("association.txt")?]);
        let cj = format!(
            "{{\n  \"device\": \"synthetic (align_sdk_check)\",\n  \"depth_intrinsics\": {},\n  \"color_intrinsics\": {},\n  \"depth_to_color\": {},\n  \"depth_unit_m\": {}\n}}\n",
            rs2::intrin_json(di),
            rs2::intrin_json(ci),
            extrin_json(e),
            jnum(unit as f64)
        );
        let _ = std::fs::write(format!("{rec}/calib.json"), cj);
    }

    println!("\n== {}: 깊이 {}x{} → 컬러 {}x{}, 컬러 왜곡 {} ==", name, di.width, di.height, ci.width, ci.height, rs2::distortion_name(ci.model));
    pin_cores(core, 1);
    let mut sdk = rs2::SdkAligner::new(di, ci, e, unit)?;
    let mut ours = DepthAligner::new(di, ci, e, unit, &Options::default())?;
    let mut ours_fast = DepthAligner::new(di, ci, e, unit, &Options { exact: false, ..Default::default() })?;
    let n = (ci.width * ci.height) as usize;
    let (mut a, mut b, mut c) = (vec![0u16; n], vec![0u16; n], vec![0u16; n]);
    let (mut t_sdk, mut t_ours, mut t_fast) = (vec![], vec![], vec![]);
    let (mut diff_exact, mut diff_fast, mut px) = (0usize, 0usize, 0usize);
    let mut bad = 0;
    for f in 0..frames {
        // 장면마다 잡음 씨앗을 바꿔 다른 깊이
        let z = scenes::render_depth(&scenes::room(), di, unit, 0.004, 0.03, (100 + f) as u32);
        let Some(ms) = sdk.process(&z, &mut a)? else {
            println!("  장면 {f}: SDK 정렬 결과를 못 받음");
            bad += 1;
            continue;
        };
        let t0 = now_ms();
        ours.process(&z, &mut b);
        let t1 = now_ms();
        ours_fast.process(&z, &mut c);
        let t2 = now_ms();
        if f > 0 {
            // 첫 장면은 SDK 가 표를 만듦
            t_sdk.push(ms);
            t_ours.push(t1 - t0);
            t_fast.push(t2 - t1);
        }
        for i in 0..n {
            diff_exact += (a[i] != b[i]) as usize;
            diff_fast += (a[i] != c[i]) as usize;
        }
        px += n;
        if let Some(l) = lists.as_mut() {
            let ts = format!("{:.6}", 1.0 + f as f64 / 30.0);
            let rgb = vec![128u8; n * 3];
            write_png16(&format!("{rec}/depth_raw/{ts}.png"), &z, di.width, di.height);
            write_png16(&format!("{rec}/depth_rs/{ts}.png"), &a, ci.width, ci.height);
            write_png16(&format!("{rec}/depth/{ts}.png"), &b, ci.width, ci.height);
            write_png_rgb(&format!("{rec}/rgb/{ts}.png"), &rgb, ci.width, ci.height);
            let _ = writeln!(l[0], "{ts} depth_raw/{ts}.png");
            let _ = writeln!(l[1], "{ts} depth_rs/{ts}.png");
            let _ = writeln!(l[2], "{ts} depth/{ts}.png");
            let _ = writeln!(l[3], "{ts} rgb/{ts}.png");
            let _ = writeln!(l[4], "{ts} rgb/{ts}.png {ts} depth/{ts}.png");
        }
    }
    let imp = rs2::align_impl_text();
    println!("  SDK 가 고른 정렬 구현: {}", if imp.is_empty() { "(로그 없음)" } else { &imp });
    println!(
        "  rs2::align 과 다른 화소: fast_align 정확한 식 {} / {}, 빠른 식 {} ({:.4} %)",
        diff_exact,
        px,
        diff_fast,
        if px > 0 { 100.0 * diff_fast as f64 / px as f64 } else { 0.0 }
    );
    println!("  시간(코어 {} 하나, 장면 {} 개 중앙값 / 90 %):", core, t_sdk.len());
    println!("    rs2::align.process         {:7.3} / {:7.3} ms", median(&t_sdk), percentile(&t_sdk, 0.9));
    println!("    fast_align Sdk 정확한 식   {:7.3} / {:7.3} ms", median(&t_ours), percentile(&t_ours, 0.9));
    println!("    fast_align Sdk 빠른 식     {:7.3} / {:7.3} ms", median(&t_fast), percentile(&t_fast, 0.9));
    if threads > 1 {
        pin_cores(core - threads + 1, threads);
        let mut mt = DepthAligner::new(di, ci, e, unit, &Options { threads, ..Default::default() })?;
        let z = scenes::render_depth(&scenes::room(), di, unit, 0.004, 0.03, 7);
        let mut t = vec![];
        for _ in 0..frames {
            let t0 = now_ms();
            mt.process(&z, &mut b);
            t.push(now_ms() - t0);
        }
        println!("    fast_align Sdk 스레드 {}    {:7.3} / {:7.3} ms (코어 {} 개)", threads, median(&t), percentile(&t, 0.9), threads);
    }
    Ok(if diff_exact == 0 && bad == 0 { 0 } else { 1 })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut calib, mut rec) = (String::new(), String::new());
    let (mut frames, mut core, mut threads) = (60, 29, 4);
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--calib" => calib = args[i + 1].clone(),
            "--frames" => frames = atoi(&args[i + 1]),
            "--core" => core = atoi(&args[i + 1]),
            "--threads" => threads = atoi(&args[i + 1]),
            "--write-rec" => rec = args[i + 1].clone(),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    println!("align_sdk_check: librealsense {}", rs2::api_version());
    rs2::align_impl_log_install();
    let r = (|| -> Result<i32, Error> {
        let mut rc = 0;
        if !calib.is_empty() {
            let js = read_file(&calib)?;
            let di = intrin_from_json(&jfind(&js, "depth_intrinsics")?)?;
            let ci = intrin_from_json(&jfind(&js, "color_intrinsics")?)?;
            let e = extrin_from_json(&jfind(&js, "depth_to_color")?)?;
            let unit = atof(&jfind(&js, "depth_unit_m")?) as f32;
            rc |= run(&calib, &di, &ci, &e, unit, frames, threads, core, "")?;
        } else {
            let e = scenes::d455_extrin();
            rc |= run("D455 가정 640x480", &scenes::d455_depth_640(), &scenes::d455_color_640(), &e, 0.001, frames, threads, core, &rec)?;
            rc |= run("D455 가정 848x480", &scenes::d455_depth_848(), &scenes::d455_color_640(), &e, 0.001, frames, threads, core, "")?;
        }
        Ok(rc)
    })();
    match r {
        Ok(rc) => {
            println!("\n{}", if rc != 0 { "rs2::align 과 다른 결과 있음" } else { "rs2::align 과 화소 단위로 같음(정확한 식)" });
            std::process::exit(rc);
        }
        Err(Error::Rs { what, func }) => {
            println!("librealsense 오류: {what} ({func})");
            std::process::exit(3);
        }
        Err(Error::Other(m)) => {
            println!("오류: {m}");
            std::process::exit(3);
        }
    }
}
