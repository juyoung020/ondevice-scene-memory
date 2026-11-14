//! 정렬 시간 재기(ms/장면). 프로세스를 코어 하나(또는 여러 개)에 묶고, 여러 번 돌려 중앙값.
//!   bench_align [--core N] [--reps R] [--rounds K]
//! 재는 것: fast_align Sdk(스레드 1·2·4), fast_align Generic, 옮겨 온 librealsense SIMD 경로(x86 SSE / ARM NEON:
//! rs2::align 의 계산부, 프레임 할당 빼고), 옮겨 온 librealsense 일반 경로(align_images + rsutil, API 호출 비용 빼고).
use std::time::Instant;
use tools::common::{atoi, lpad, pin_cores};
use tools::fast_align::{DepthAligner, Intrinsics, Method, Options};
use tools::{rsref, scenes};

/// fn 을 reps 번 돌린 평균(ms)을 rounds 번 → 중앙값과 최소
fn timeit(name: &str, reps: i32, rounds: i32, f: &mut dyn FnMut()) -> (f64, f64) {
    f(); // 데우기
    let mut r = vec![];
    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        r.push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (med, mn) = (r[r.len() / 2], r[0]);
    println!("    {} 중앙 {:7.3} ms  최소 {:7.3} ms", lpad(name, 40), med, mn);
    (med, mn)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut core, mut reps, mut rounds) = (30, 20, 15);
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--core" => {
                i += 1;
                core = atoi(&args[i]);
            }
            "--reps" => {
                i += 1;
                reps = atoi(&args[i]);
            }
            "--rounds" => {
                i += 1;
                rounds = atoi(&args[i]);
            }
            _ => {}
        }
        i += 1;
    }
    let arch = if cfg!(target_arch = "aarch64") { "aarch64(NEON)" } else { "x86-64(SSE)" };
    println!("bench_align: {arch}, 코어 {core} 부터, {reps} 번 평균 × {rounds} 회 중앙값");
    let cfgs: [(&str, Intrinsics, Intrinsics); 3] = [
        ("깊이 640x480 → 컬러 640x480", scenes::d455_depth_640(), scenes::d455_color_640()),
        ("깊이 848x480 → 컬러 640x480", scenes::d455_depth_848(), scenes::d455_color_640()),
        ("깊이 640x480 → 컬러 1280x720", scenes::d455_depth_640(), scenes::d455_color_1280x720()),
    ];
    let e = scenes::d455_extrin();
    let unit = 0.001f32;
    println!("| 구성 | 방식 | 중앙 ms | 최소 ms |");
    for (name, d, c) in &cfgs {
        let z = scenes::render_depth(&scenes::room(), d, unit, 0.004, 0.03, 3);
        let mut out = vec![0u16; (c.width * c.height) as usize];
        println!("  {name}");
        let mut rows: Vec<String> = vec![];
        let mut row = |m: &str, (med, mn): (f64, f64)| rows.push(format!("| {name} | {m} | {med:.3} | {mn:.3} |"));
        pin_cores(core, 1);
        {
            let mut s = rsref::SimdAlign::new(d, unit);
            row("librealsense SIMD 경로 이식", timeit("librealsense SIMD 경로 이식(1 코어)", reps, rounds, &mut || s.align_depth_to_other(&z, &mut out, c, &e)));
        }
        row(
            "librealsense 일반 경로 이식",
            timeit("librealsense 일반 경로 이식(1 코어)", (reps / 4).max(1), rounds, &mut || rsref::align_z_to_other_generic(&mut out, &z, unit, d, &e, c)),
        );
        {
            let mut a = DepthAligner::new(d, c, &e, unit, &Options { method: Method::Generic, ..Default::default() }).unwrap();
            row("fast_align Generic", timeit("fast_align Generic(1 코어)", reps, rounds, &mut || a.process(&z, &mut out)));
        }
        {
            let mut a = DepthAligner::new(d, c, &e, unit, &Options::default()).unwrap();
            row("fast_align Sdk, 스레드 1", timeit("fast_align Sdk(1 코어)", reps, rounds, &mut || a.process(&z, &mut out)));
        }
        {
            let mut a = DepthAligner::new(d, c, &e, unit, &Options { exact: false, ..Default::default() }).unwrap();
            row("fast_align Sdk 빠른 식(exact=false), 스레드 1", timeit("fast_align Sdk 빠른 식(1 코어)", reps, rounds, &mut || a.process(&z, &mut out)));
        }
        for th in [2, 4] {
            pin_cores(core - th + 1, th);
            for exact in [true, false] {
                let mut a = DepthAligner::new(d, c, &e, unit, &Options { threads: th, exact, ..Default::default() }).unwrap();
                let nm = format!("fast_align Sdk{}(스레드 {th}, 코어 {th})", if exact { "" } else { " 빠른 식" });
                let r = timeit(&nm, reps, rounds, &mut || a.process(&z, &mut out));
                if a.last_fell_back_serial() {
                    println!("      (한 스레드로 다시 돈 장면 있음)");
                }
                row(&format!("fast_align Sdk{}, 스레드 {th}(코어 {th})", if exact { "" } else { " 빠른 식" }), r);
            }
        }
        for r in &rows {
            println!("{r}");
        }
    }
}
