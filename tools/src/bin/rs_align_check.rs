//! 녹화 폴더(rs_rec)의 원래 깊이에 fast_align 을 다시 돌려 rs2::align 결과(depth_rs/)와 화소 차이·시간을 비교한다.
//! 같은 원래 깊이로 rs2::align 도 카메라 없이(software_device) 다시 돌려 시간을 같은 조건에서 잰다.
//!   rs_align_check <녹화 폴더> [--core 29] [--threads 4] [--max N]
use tools::common::*;
use tools::fast_align::{DepthAligner, Options};
use tools::rs2::{self, Error};

fn show(name: &str, d: &Diff) {
    let diff = d.both - d.equal + d.only_a + d.only_b;
    let ne = d.both - d.equal;
    println!(
        "  {} 일치 {:.5} (다른 화소 {} / {}), 둘 다 값 있는데 다름 {}(평균 |차| {:.1} 단위), 우리만 {}, rs 만 {}, 경계 화소 중 다름 {:.4}",
        lpad(name, 34),
        1.0 - diff as f64 / d.total as f64,
        diff,
        d.total,
        ne,
        if ne > 0 { d.sum_abs / ne as f64 } else { 0.0 },
        d.only_a,
        d.only_b,
        if d.edge > 0 { d.edge_diff as f64 / d.edge as f64 } else { 0.0 }
    );
}

fn run(dir: &str, core: i32, threads: i32, maxn: i32) -> Result<i32, Error> {
    let js = read_file(&join(dir, "calib.json"))?;
    let di = intrin_from_json(&jfind(&js, "depth_intrinsics")?)?;
    let ci = intrin_from_json(&jfind(&js, "color_intrinsics")?)?;
    let e = extrin_from_json(&jfind(&js, "depth_to_color")?)?;
    let unit = atof(&jfind(&js, "depth_unit_m")?) as f32;
    let raw = read_list(&join(dir, "depth_raw.txt"))?;
    let rsl = read_list(&join(dir, "depth_rs.txt"))?;
    if raw.len() != rsl.len() {
        return Err(Error::Other("depth_raw.txt 와 depth_rs.txt 장면 수가 다름".into()));
    }
    println!(
        "rs_align_check: {}, {} 장, 깊이 {}x{} → 컬러 {}x{}, 컬러 왜곡 {}, 깊이 단위 {} m",
        dir,
        raw.len(),
        di.width,
        di.height,
        ci.width,
        ci.height,
        rs2::distortion_name(ci.model),
        fmt_g(unit as f64, 6)
    );

    pin_cores(core, 1);
    let mut ours = DepthAligner::new(&di, &ci, &e, unit, &Options::default())?;
    let mut fast = DepthAligner::new(&di, &ci, &e, unit, &Options { exact: false, ..Default::default() })?;
    let mut cdist = DepthAligner::new(&di, &ci, &e, unit, &Options { color_distortion: true, ..Default::default() })?;
    let mut sdk = rs2::SdkAligner::new(&di, &ci, &e, unit)?;
    let n = (ci.width * ci.height) as usize;
    let mut z: Vec<u16> = vec![];
    let (mut a, mut b, mut s, mut cd) = (vec![0u16; n], vec![0u16; n], vec![0u16; n], vec![0u16; n]);
    let (mut tot, mut totf, mut tots, mut totcd) = (Diff::default(), Diff::default(), Diff::default(), Diff::default());
    let (mut t_ours, mut t_fast, mut t_sdk) = (vec![], vec![], vec![]);
    let mut frames = 0;
    let jump = (0.05f32 / unit) as i32; // 경계: 3×3 안 5 cm 넘는 차
    for k in 0..raw.len() {
        if frames >= maxn {
            break;
        }
        let Some((zz, w, h)) = read_png16(&join(dir, &raw[k].1)) else { continue };
        z = zz;
        if w != di.width || h != di.height {
            continue;
        }
        let Some((rs, w, h)) = read_png16(&join(dir, &rsl[k].1)) else { continue };
        if w != ci.width || h != ci.height {
            continue;
        }
        let t0 = now_ms();
        ours.process(&z, &mut a);
        let t1 = now_ms();
        fast.process(&z, &mut b);
        let t2 = now_ms();
        let sdk_ms = sdk.process(&z, &mut s)?;
        cdist.process(&z, &mut cd);
        if frames > 0 {
            t_ours.push(t1 - t0);
            t_fast.push(t2 - t1);
            if let Some(ms) = sdk_ms {
                t_sdk.push(ms);
            }
        }
        tot.add(&compare(&a, &rs, ci.width, ci.height, jump));
        totf.add(&compare(&b, &rs, ci.width, ci.height, jump));
        if sdk_ms.is_some() {
            tots.add(&compare(&s, &rs, ci.width, ci.height, jump));
        }
        totcd.add(&compare(&cd, &rs, ci.width, ci.height, jump));
        frames += 1;
    }
    if frames == 0 {
        return Err(Error::Other("읽은 장면 없음".into()));
    }
    println!("\n녹화된 rs2::align 결과(depth_rs/)와 비교, {frames} 장:");
    show("fast_align Sdk 정확한 식", &tot);
    show("fast_align Sdk 빠른 식", &totf);
    show("rs2::align 다시(software_device)", &tots);
    show("fast_align 컬러 왜곡 적용(SDK 와 다름)", &totcd);
    println!("\n시간(코어 {core} 하나, 중앙 / 90 %):");
    println!("  rs2::align.process(다시)  {:7.3} / {:7.3} ms", median(&t_sdk), percentile(&t_sdk, 0.9));
    println!("  fast_align 정확한 식      {:7.3} / {:7.3} ms", median(&t_ours), percentile(&t_ours, 0.9));
    println!("  fast_align 빠른 식        {:7.3} / {:7.3} ms", median(&t_fast), percentile(&t_fast, 0.9));
    if let Ok(st) = read_file(&join(dir, "rec_stats.json")) {
        if let (Ok(r), Ok(f)) = (jfind(&st, "rs_align_ms_median"), jfind(&st, "fast_align_ms_median")) {
            println!("  녹화 중 rs2::align        {:7.3} ms, fast_align {:7.3} ms(rec_stats.json, 코어 묶지 않음)", atof(&r), atof(&f));
        }
    }
    if threads > 1 {
        pin_cores(core - threads + 1, threads);
        let mut mt = DepthAligner::new(&di, &ci, &e, unit, &Options { threads, ..Default::default() })?;
        let mut t = vec![];
        for _ in 0..50 {
            let t0 = now_ms();
            mt.process(&z, &mut a);
            t.push(now_ms() - t0);
        }
        println!("  fast_align 스레드 {}       {:7.3} / {:7.3} ms(마지막 장면 50 번)", threads, median(&t), percentile(&t, 0.9));
    }
    let imp = rs2::align_impl_text();
    println!("SDK 정렬 구현(이 기계): {}", if imp.is_empty() { "(로그 없음)" } else { &imp });
    let d0 = tot.both - tot.equal + tot.only_a + tot.only_b;
    Ok(if d0 == 0 { 0 } else { 1 })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("rs_align_check <녹화 폴더> [--core C] [--threads T] [--max N]");
        std::process::exit(1);
    }
    let (mut core, mut threads, mut maxn) = (29, 4, 1 << 30);
    let mut i = 2;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--core" => core = atoi(&args[i + 1]),
            "--threads" => threads = atoi(&args[i + 1]),
            "--max" => maxn = atoi(&args[i + 1]),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    rs2::align_impl_log_install();
    match run(&args[1], core, threads, maxn) {
        Ok(rc) => std::process::exit(rc),
        Err(Error::Rs { what, func }) => {
            eprintln!("librealsense 오류: {what} ({func})");
            std::process::exit(3);
        }
        Err(Error::Other(m)) => {
            eprintln!("오류: {m}");
            std::process::exit(3);
        }
    }
}
