//! RealSense 녹화: 컬러(RGB8)·원래 깊이(정렬 전)·rs2::align 결과·fast_align 결과를 TUM 형식 폴더로.
//!   rs_rec <출력 폴더> [초=10] [--emitter on|off] [--preset default|hand|high_accuracy|high_density|medium_density]
//!          [--depth 640x480] [--color 640x480] [--fps 30] [--threads 1] [--skip 30]
//! 폴더:
//!   rgb/<시각>.png        컬러 RGB8                      rgb.txt
//!   depth/<시각>.png      fast_align 정렬 깊이(컬러 크기)  depth.txt   ← TUM 짝(association.txt)
//!   depth_raw/<시각>.png  정렬 전 원래 깊이               depth_raw.txt
//!   depth_rs/<시각>.png   rs2::align 결과                depth_rs.txt
//!   calib.json(보정값·설정·rgbd2tum 명령), rec_stats.json(장면 수·시간)
//! 깊이 PNG 값은 카메라 깊이 단위 그대로(D400 기본 1 mm = 1000/m). memmap 은 rgbd2tum 으로 TUM fr3(5000/m)로 바꿔 읽는다.
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use tools::common::*;
use tools::fast_align::{DepthAligner, Extrinsics, Intrinsics, Options};
use tools::rs2::*;

struct Job {
    tc: String,
    td: String,
    cw: i32,
    ch: i32,
    dw: i32,
    dh: i32,
    rgb: Vec<u8>,
    raw: Vec<u16>,
    rs: Vec<u16>,
    ours: Vec<u16>,
}

struct WriterShared {
    q: Mutex<(VecDeque<Job>, bool)>,
    cv: Condvar,
    failed: AtomicI32,
}

/// PNG 쓰기 스레드 n 개, 큐가 cap 이면 밀어 넣기 실패(그 장면은 버림)
struct Writer {
    sh: Arc<WriterShared>,
    cap: usize,
    th: Vec<std::thread::JoinHandle<()>>,
}

impl Writer {
    fn new(out: &str, n: usize, cap: usize) -> Writer {
        let sh = Arc::new(WriterShared { q: Mutex::new((VecDeque::new(), false)), cv: Condvar::new(), failed: AtomicI32::new(0) });
        let th = (0..n)
            .map(|_| {
                let sh = sh.clone();
                let out = out.to_string();
                std::thread::spawn(move || loop {
                    let j = {
                        let mut g = sh.q.lock().unwrap();
                        while !g.1 && g.0.is_empty() {
                            g = sh.cv.wait(g).unwrap();
                        }
                        match g.0.pop_front() {
                            Some(j) => j,
                            None => return,
                        }
                    };
                    let mut ok = write_png_rgb(&join(&out, &format!("rgb/{}.png", j.tc)), &j.rgb, j.cw, j.ch);
                    ok &= write_png16(&join(&out, &format!("depth/{}.png", j.td)), &j.ours, j.cw, j.ch);
                    ok &= write_png16(&join(&out, &format!("depth_rs/{}.png", j.td)), &j.rs, j.cw, j.ch);
                    ok &= write_png16(&join(&out, &format!("depth_raw/{}.png", j.td)), &j.raw, j.dw, j.dh);
                    if !ok {
                        sh.failed.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        Writer { sh, cap, th }
    }
    fn push(&self, j: Job) -> bool {
        let mut g = self.sh.q.lock().unwrap();
        if g.0.len() >= self.cap {
            return false;
        }
        g.0.push_back(j);
        self.sh.cv.notify_one();
        true
    }
    fn finish(&mut self) {
        self.sh.q.lock().unwrap().1 = true;
        self.sh.cv.notify_all();
        for t in self.th.drain(..) {
            let _ = t.join();
        }
    }
}

fn tstr(ms: f64) -> String {
    format!("{:.6}", ms / 1000.0)
}

struct Args {
    out: String,
    secs: f64,
    emitter: String,
    preset: String,
    dw: i32,
    dh: i32,
    cw: i32,
    ch: i32,
    fps: i32,
    threads: i32,
    skip: i32,
}

fn mk(p: &str) -> Result<File, Error> {
    File::create(p).map_err(|e| Error::Other(format!("{p}: {e}")))
}

fn run(a: &Args) -> Result<i32, Error> {
    let out = a.out.as_str();
    let ctx = ck(|e| unsafe { rs2_create_context(25901, e) })?;
    let list = ck(|e| unsafe { rs2_query_devices(ctx, e) })?;
    if ck(|e| unsafe { rs2_get_device_count(list, e) })? == 0 {
        eprintln!("rs_rec: 장치 없음(RealSense 를 USB3 포트에 꽂고 다시)");
        return Ok(2);
    }
    for sub in ["rgb", "depth", "depth_raw", "depth_rs"] {
        std::fs::create_dir_all(join(out, sub)).map_err(|e| Error::Other(format!("{out}/{sub}: {e}")))?;
    }
    let cfg = ck(|e| unsafe { rs2_create_config(e) })?;
    ck(|e| unsafe { rs2_config_enable_stream(cfg, STREAM_DEPTH, -1, a.dw, a.dh, FORMAT_Z16, a.fps, e) })?;
    ck(|e| unsafe { rs2_config_enable_stream(cfg, STREAM_COLOR, -1, a.cw, a.ch, FORMAT_RGB8, a.fps, e) })?;
    let pipe = ck(|e| unsafe { rs2_create_pipeline(ctx, e) })?;
    let prof = ck(|e| unsafe { rs2_pipeline_start_with_config(pipe, cfg, e) })?;
    let dev = ck(|e| unsafe { rs2_pipeline_profile_get_device(prof, e) })?;
    let sensors = ck(|e| unsafe { rs2_query_sensors(dev, e) })?;
    let ns = ck(|e| unsafe { rs2_get_sensors_count(sensors, e) })?;
    let mut all = vec![];
    let mut ds: P = std::ptr::null_mut();
    for i in 0..ns {
        let s = ck(|e| unsafe { rs2_create_sensor(sensors, i, e) })?;
        if ds.is_null() && ck(|e| unsafe { rs2_is_sensor_extendable_to(s, EXT_DEPTH_SENSOR, e) })? != 0 {
            ds = s;
        }
        all.push(s);
    }
    if ds.is_null() {
        return Err(Error::Other("깊이 센서 없음".into()));
    }
    if !a.preset.is_empty() {
        let mut pv = -1;
        for p in 0..RS400_VISUAL_PRESET_COUNT {
            let n = cstr(unsafe { rs2_rs400_visual_preset_to_string(p) }).to_lowercase().replace(' ', "_");
            if n == a.preset {
                pv = p;
            }
        }
        if pv < 0 {
            return Err(Error::Other(format!("모르는 프리셋: {}", a.preset)));
        }
        ck(|e| unsafe { rs2_set_option(ds, OPT_VISUAL_PRESET, pv as f32, e) })?;
    }
    if ck(|e| unsafe { rs2_supports_option(ds, OPT_EMITTER_ENABLED, e) })? != 0 {
        ck(|e| unsafe { rs2_set_option(ds, OPT_EMITTER_ENABLED, if a.emitter == "off" { 0.0 } else { 1.0 }, e) })?;
    }
    for &s in &all {
        if ck(|e| unsafe { rs2_supports_option(s, OPT_ENABLE_AUTO_EXPOSURE, e) })? != 0 {
            ck(|e| unsafe { rs2_set_option(s, OPT_ENABLE_AUTO_EXPOSURE, 1.0, e) })?;
        }
    }
    // 컬러 자동 노출이 어두울 때 프레임 수를 낮추지 않게(위치 추적은 고른 30 장/초가 낫다)
    for &s in &all {
        if ck(|e| unsafe { rs2_is_sensor_extendable_to(s, EXT_COLOR_SENSOR, e) })? != 0 && ck(|e| unsafe { rs2_supports_option(s, OPT_AUTO_EXPOSURE_PRIORITY, e) })? != 0 {
            ck(|e| unsafe { rs2_set_option(s, OPT_AUTO_EXPOSURE_PRIORITY, 0.0, e) })?;
        }
    }

    // 스트림 프로파일(종류별 첫 번째)
    let streams = ck(|e| unsafe { rs2_pipeline_profile_get_streams(prof, e) })?;
    let (mut dprof, mut cprof): (P, P) = (std::ptr::null_mut(), std::ptr::null_mut());
    for i in 0..ck(|e| unsafe { rs2_get_stream_profiles_count(streams, e) })? {
        let p = ck(|e| unsafe { rs2_get_stream_profile(streams, i, e) })?;
        let (mut st, mut fm, mut ix, mut uid, mut f) = (0, 0, 0, 0, 0);
        ck(|e| unsafe { rs2_get_stream_profile_data(p, &mut st, &mut fm, &mut ix, &mut uid, &mut f, e) })?;
        if st == STREAM_DEPTH && dprof.is_null() {
            dprof = p;
        }
        if st == STREAM_COLOR && cprof.is_null() {
            cprof = p;
        }
    }
    if dprof.is_null() || cprof.is_null() {
        return Err(Error::Other("깊이·컬러 스트림 없음".into()));
    }
    let (mut di, mut ci) = (Intrinsics::default(), Intrinsics::default());
    let mut ex = Extrinsics::default();
    ck(|e| unsafe { rs2_get_video_stream_intrinsics(dprof, &mut di, e) })?;
    ck(|e| unsafe { rs2_get_video_stream_intrinsics(cprof, &mut ci, e) })?;
    ck(|e| unsafe { rs2_get_extrinsics(dprof, cprof, &mut ex, e) })?;
    let unit = ck(|e| unsafe { rs2_get_depth_scale(ds, e) })?;

    let (align, align_q) = create_align()?;
    let mut ours = DepthAligner::new(&di, &ci, &ex, unit, &Options { threads: a.threads, ..Default::default() })?;
    let hw = std::thread::available_parallelism().map_or(0, |n| n.get());
    let mut writer = Writer::new(out, (hw / 4).max(2), 240);

    let (name, serial) = (device_info(dev, INFO_NAME)?, device_info(dev, INFO_SERIAL)?);
    println!(
        "rs_rec: {} {}, 깊이 {}x{} · 컬러 {}x{} @{}, 레이저 {}, 프리셋 {}, 깊이 단위 {} m, {} 초 → {}",
        name,
        serial,
        a.dw,
        a.dh,
        a.cw,
        a.ch,
        a.fps,
        a.emitter,
        if a.preset.is_empty() { "(그대로)" } else { &a.preset },
        fmt_g(unit as f64, 6),
        fmt_g(a.secs, 6),
        out
    );
    let ug = fmt_g(unit as f64, 6);
    let (mut frgb, mut fdep, mut fraw, mut frs, mut fas) =
        (mk(&join(out, "rgb.txt"))?, mk(&join(out, "depth.txt"))?, mk(&join(out, "depth_raw.txt"))?, mk(&join(out, "depth_rs.txt"))?, mk(&join(out, "association.txt"))?);
    let _ = write!(frgb, "# color RGB8\n# timestamp filename\n");
    let _ = write!(fdep, "# depth aligned to color by fast_align (units {ug} m)\n# timestamp filename\n");
    let _ = write!(fraw, "# raw depth (units {ug} m)\n# timestamp filename\n");
    let _ = write!(frs, "# depth aligned by rs2::align (units {ug} m)\n# timestamp filename\n");

    let (mut t_rs, mut t_ours) = (vec![], vec![]);
    let (mut n, mut dropped, mut gaps, mut skipped) = (0, 0, 0i64, 0);
    let mut last_fn: i64 = -1;
    let (mut diff_px, mut all_px) = (0usize, 0usize);

    let (dw, dh, cw, ch) = (a.dw as usize, a.dh as usize, a.cw as usize, a.ch as usize);
    let t_start = now_ms();
    while now_ms() - t_start < (a.secs + a.skip as f64 / a.fps as f64) * 1000.0 {
        let fsr = ck(|e| unsafe { rs2_pipeline_wait_for_frames(pipe, 5000, e) })?;
        let (cf, df) = split(fsr)?;
        if cf.is_null() || df.is_null() {
            release(&[cf, df, fsr]);
            continue;
        }
        if skipped < a.skip {
            // 자동 노출이 자리 잡을 때까지
            skipped += 1;
            release(&[cf, df, fsr]);
            continue;
        }
        let fnum = ck(|e| unsafe { rs2_get_frame_number(df, e) })? as i64;
        if last_fn >= 0 && fnum > last_fn + 1 {
            gaps += fnum - last_fn - 1;
        }
        last_fn = fnum;
        let tc = tstr(ck(|e| unsafe { rs2_get_frame_timestamp(cf, e) })?);
        let td = tstr(ck(|e| unsafe { rs2_get_frame_timestamp(df, e) })?);
        let zp = unsafe { std::slice::from_raw_parts(ck(|e| rs2_get_frame_data(df, e))? as *const u16, dw * dh) };
        let raw = zp.to_vec();
        // 화면 가운데 40×40 깊이 중앙값과 전체 유효 비율(최소 거리 확인용: 벽에 다가가며 값이 사라지는 거리)
        let (center_m, center_valid, valid_ratio) = {
            let mut c = vec![];
            for y in dh / 2 - 20..dh / 2 + 20 {
                for x in dw / 2 - 20..dw / 2 + 20 {
                    if zp[y * dw + x] != 0 {
                        c.push((zp[y * dw + x] as f32 * unit) as f64);
                    }
                }
            }
            let valid = zp.iter().filter(|&&v| v != 0).count();
            (if c.len() > 200 { median(&c) } else { 0.0 }, c.len() as f64 / 1600.0, valid as f64 / (dw * dh) as f64)
        };
        let rgb = unsafe { std::slice::from_raw_parts(ck(|e| rs2_get_frame_data(cf, e))?, cw * ch * 3) }.to_vec();
        let t0 = now_ms();
        let al = align_process(align, align_q, fsr);
        let t1 = now_ms();
        release(&[cf, df, fsr]);
        let al = al?;
        let (ac, ad) = split(al)?;
        if ad.is_null() {
            release(&[ac, al]);
            return Err(Error::Other("rs2::align 결과에 깊이 없음".into()));
        }
        let rs = unsafe { std::slice::from_raw_parts(ck(|e| rs2_get_frame_data(ad, e))? as *const u16, cw * ch) }.to_vec();
        release(&[ac, ad, al]);
        let mut o = vec![0u16; cw * ch];
        let t2 = now_ms();
        ours.process(&raw, &mut o);
        let t3 = now_ms();
        if n > 0 {
            t_rs.push(t1 - t0);
            t_ours.push(t3 - t2);
        }
        diff_px += o.iter().zip(&rs).filter(|(x, y)| x != y).count();
        all_px += o.len();
        n += 1;
        let job = Job { tc: tc.clone(), td: td.clone(), cw: a.cw, ch: a.ch, dw: a.dw, dh: a.dh, rgb, raw, rs, ours: o };
        if !writer.push(job) {
            // 쓰기가 밀리면 이 장면은 목록에도 넣지 않음
            dropped += 1;
            continue;
        }
        let _ = writeln!(frgb, "{tc} rgb/{tc}.png");
        let _ = writeln!(fdep, "{td} depth/{td}.png");
        let _ = writeln!(fraw, "{td} depth_raw/{td}.png");
        let _ = writeln!(frs, "{td} depth_rs/{td}.png");
        let _ = writeln!(fas, "{tc} rgb/{tc}.png {td} depth/{td}.png");
        if n % a.fps == 0 {
            println!(
                "  {} 장, rs2::align {:.2} ms · fast_align {:.2} ms(중앙), 쓰기 밀림 {} | 가운데 깊이 {:.3} m(유효 {:.0} %), 전체 유효 {:.0} %",
                n,
                median(&t_rs),
                median(&t_ours),
                dropped,
                center_m,
                100.0 * center_valid,
                100.0 * valid_ratio
            );
        }
    }
    let _ = ck(|e| unsafe { rs2_pipeline_stop(pipe, e) });
    writer.finish();

    let imp = align_impl_text();
    let cmd = format!("rgbd2tum {out} {out}_freiburg3 {:.4} {:.4} {:.4} {:.4} {:.1}", ci.fx, ci.fy, ci.ppx, ci.ppy, 1.0 / unit as f64);
    let usb = if ck(|e| unsafe { rs2_supports_device_info(dev, INFO_USB_TYPE, e) })? != 0 { device_info(dev, INFO_USB_TYPE)? } else { String::new() };
    let cj = format!(
        "{{\n  \"librealsense\": {},\n  \"device\": {},\n  \"serial\": {},\n  \"firmware\": {},\n  \"usb\": {},\n  \"depth_intrinsics\": {},\n  \"color_intrinsics\": {},\n  \"depth_to_color\": {},\n  \"depth_unit_m\": {},\n  \"depth_png_per_m\": {},\n  \"settings\": {{\"fps\": {}, \"emitter\": {}, \"preset\": {}, \"skip\": {}}},\n  \"sdk_align_impl\": {},\n  \"rgbd2tum\": {}\n}}\n",
        jstr(&api_version()),
        jstr(&name),
        jstr(&serial),
        jstr(&device_info(dev, INFO_FIRMWARE)?),
        jstr(&usb),
        tools::rs2::intrin_json(&di),
        tools::rs2::intrin_json(&ci),
        extrin_json(&ex),
        jnum(unit as f64),
        jnum(1.0 / unit as f64),
        a.fps,
        jstr(&a.emitter),
        jstr(if a.preset.is_empty() { "unchanged" } else { &a.preset }),
        a.skip,
        jstr(&imp),
        jstr(&cmd)
    );
    let _ = std::fs::write(join(out, "calib.json"), cj);
    let sj = format!(
        "{{\n  \"frames\": {},\n  \"write_dropped\": {},\n  \"write_failed\": {},\n  \"depth_frame_gaps\": {},\n  \"rs_align_ms_median\": {},\n  \"rs_align_ms_p90\": {},\n  \"fast_align_ms_median\": {},\n  \"fast_align_ms_p90\": {},\n  \"fast_align_threads\": {},\n  \"pixels_differ_from_rs\": {},\n  \"pixels_total\": {}\n}}\n",
        n,
        dropped,
        writer.sh.failed.load(Ordering::Relaxed),
        gaps,
        jnum(median(&t_rs)),
        jnum(percentile(&t_rs, 0.9)),
        jnum(median(&t_ours)),
        jnum(percentile(&t_ours, 0.9)),
        a.threads,
        diff_px,
        all_px
    );
    let _ = std::fs::write(join(out, "rec_stats.json"), sj);
    println!("끝: {} 장(쓰기 밀려 빠진 {}, 깊이 프레임 번호 건너뜀 {}), rs2::align {:.2} ms · fast_align {:.2} ms(중앙)", n, dropped, gaps, median(&t_rs), median(&t_ours));
    println!("SDK 정렬 구현: {}", if imp.is_empty() { "(로그 없음)" } else { &imp });
    println!("rs2::align 과 다른 화소: {diff_px} / {all_px}");
    println!("memmap 용 TUM fr3 변환: {cmd}");
    if dropped > 0 {
        println!("주의: 쓰기가 밀려 {dropped} 장을 버림(폴더·목록 모두에 없음)");
    }
    Ok(0)
}

fn main() {
    let v: Vec<String> = std::env::args().collect();
    if v.len() < 2 {
        eprintln!("rs_rec <출력 폴더> [초=10] [--emitter on|off] [--preset 이름] [--depth WxH] [--color WxH] [--fps N] [--threads N] [--skip N]");
        std::process::exit(1);
    }
    let mut a = Args { out: v[1].clone(), secs: 10.0, emitter: "on".into(), preset: String::new(), dw: 640, dh: 480, cw: 640, ch: 480, fps: 30, threads: 1, skip: 30 };
    let mut i = 2;
    if v.len() > 2 && !v[2].starts_with('-') {
        a.secs = atof(&v[2]);
        i = 3;
    }
    while i + 1 < v.len() {
        let x = v[i + 1].as_str();
        match v[i].as_str() {
            "--emitter" => a.emitter = x.into(),
            "--preset" => a.preset = x.into(),
            "--depth" => {
                if let Some((w, h)) = parse_wh(x) {
                    (a.dw, a.dh) = (w, h);
                }
            }
            "--color" => {
                if let Some((w, h)) = parse_wh(x) {
                    (a.cw, a.ch) = (w, h);
                }
            }
            "--fps" => a.fps = atoi(x),
            "--threads" => a.threads = atoi(x),
            "--skip" => a.skip = atoi(x),
            _ => {
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    align_impl_log_install();
    match run(&a) {
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
