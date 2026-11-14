//! rsdriver 로 녹화(D435 녹화 도구): 정렬 확인 도구 rs_align_check 가 읽는 폴더 꼴.
//!   rsd_rec <출력 폴더> [장면=60] [--size 640x480] [--fps 30] [--exposure ms | --ae-cap ms] [--camera-profile librealsense|tracking] [--gain g] [--emitter on|off] [--skip 30] [--writers 3] [--queue 150]
//! 폴더: rgb/ depth/(rsdriver 정렬, depth_rs.txt 도 이것을 가리킴) depth_raw/ yuyv/<시각>.bin(원본 YUYV) calib.json frames.tsv(시각·순번·메타·해시)
#[cfg(target_os = "linux")]
fn main() {
    use rsdriver::camera::*;
    use std::io::Write;
    use tools::common::*;
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("rsd_rec <출력 폴더> [장면=60] [--size WxH] [--fps 30] [--exposure ms | --ae-cap ms] [--gain g] [--emitter on|off] [--skip 30]");
        std::process::exit(1);
    }
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let out = a[1].clone();
    let n: usize = a.get(2).filter(|s| !s.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(60);
    let (w, h) = arg("--size").and_then(|s| parse_wh(&s)).unwrap_or((640, 480));
    let cfg = Config {
        width: w as u32,
        height: h as u32,
        fps: arg("--fps").and_then(|s| s.parse().ok()).unwrap_or(30),
        exposure_ms: arg("--exposure").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        gain: arg("--gain").and_then(|s| s.parse().ok()).unwrap_or(-1),
        ae_cap_ms: arg("--ae-cap").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        profile: arg("--camera-profile").map(|s| rsdriver::camera::Profile::parse(&s)).transpose().unwrap_or_else(|e| {
            eprintln!("rsd_rec: {e}");
            std::process::exit(2)
        }).unwrap_or(rsdriver::camera::Profile::Librealsense),
        emitter: arg("--emitter").map(|s| s != "off").unwrap_or(true),
        warmup: arg("--skip").and_then(|s| s.parse().ok()).unwrap_or(30),
        ..Default::default()
    };
    let mut cam = match Camera::open(cfg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rsd_rec: {e}");
            std::process::exit(2);
        }
    };
    println!("{}", cam.describe());
    for s in ["rgb", "depth", "depth_raw", "yuyv"] {
        let _ = std::fs::create_dir_all(join(&out, s));
    }
    let c = &cam.calib;
    let cj = format!(
        "{{\n  \"driver\": \"rsdriver\",\n  \"serial\": {},\n  \"firmware\": {},\n  \"depth_intrinsics\": {},\n  \"color_intrinsics\": {},\n  \"depth_to_color\": {},\n  \"depth_unit_m\": {},\n  \"depth_png_per_m\": {}\n}}\n",
        jstr(&c.gvd.serial),
        jstr(&c.gvd.firmware),
        intrin_json(&c.depth, "Brown Conrady"),
        intrin_json(&c.color, "Inverse Brown Conrady"),
        extrin_json(&c.depth_to_color),
        jnum(c.depth_scale as f64),
        jnum(cam.depth_per_m() as f64)
    );
    let _ = std::fs::write(join(&out, "calib.json"), cj);
    let mk = |p: &str| std::fs::File::create(join(&out, p)).unwrap();
    let (mut frgb, mut fd, mut fraw, mut frs, mut ft) = (mk("rgb.txt"), mk("depth.txt"), mk("depth_raw.txt"), mk("depth_rs.txt"), mk("frames.tsv"));
    let _ = writeln!(ft, "color_ts_us\tdepth_ts_us\tcolor_seq\tdepth_seq\tcolor_counter\tdepth_counter\texposure\tgain\tt_ms\tyuyv_fnv\tz16_fnv\trgb_fnv\taligned_fnv\twait_ms\talign_ms");
    // 받기 스레드는 카메라만: 장면 버퍼를 통째로 일감으로 넘기고(복사 없이 옮김) PNG 압축·쓰기는 쓰기 스레드들이 한다.
    // 전에는 한 스레드에서 PNG 넷을 쓰느라 Pi 5 에서 장면당 30 ms 를 넘겨 카메라 버퍼가 넘치고 1–5 초씩 빠졌다(walk1 빈틈 7 곳).
    // 다 쓴 버퍼는 돌려받아 다시 씀. 일감 줄이 꽉 차면(디스크가 못 따라감) 그 장면은 버리고 센다(카메라는 멈추지 않음).
    struct Job {
        seq: usize,
        tc: String,
        meta: String,
        rgb: Vec<u8>,
        aligned: Vec<u16>,
        raw: Vec<u16>,
        yuyv: Vec<u8>,
    }
    let fnv = |b: &[u8]| b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
    let b16 = |v: &[u16]| v.iter().fold(0xcbf29ce484222325u64, |h, x| x.to_le_bytes().iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3)));
    let writers: usize = arg("--writers").and_then(|s| s.parse().ok()).unwrap_or(3);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(arg("--queue").and_then(|s| s.parse().ok()).unwrap_or(150));
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let (back_tx, back_rx) = std::sync::mpsc::channel::<Job>();
    let lines = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::<usize, String>::new()));
    let mut hs = Vec::new();
    for _ in 0..writers.max(1) {
        let (rx, back_tx, lines, out) = (rx.clone(), back_tx.clone(), lines.clone(), out.clone());
        hs.push(std::thread::spawn(move || loop {
            let job = { rx.lock().unwrap().recv() };
            let Ok(j) = job else { break };
            let tc = &j.tc;
            let _ = write_png_rgb(&join(&out, &format!("rgb/{tc}.png")), &j.rgb, w, h);
            let _ = write_png16(&join(&out, &format!("depth/{tc}.png")), &j.aligned, w, h);
            let _ = write_png16(&join(&out, &format!("depth_raw/{tc}.png")), &j.raw, w, h);
            let _ = std::fs::write(join(&out, &format!("yuyv/{tc}.bin")), &j.yuyv);
            let line = j.meta.replace("{H}", &format!("{:016x}\t{:016x}\t{:016x}\t{:016x}", fnv(&j.yuyv), b16(&j.raw), fnv(&j.rgb), b16(&j.aligned)));
            lines.lock().unwrap().insert(j.seq, line);
            let _ = back_tx.send(j);
        }));
    }
    drop(back_tx);
    let mut o = Output { want_raw: true, want_yuyv: true, ..Default::default() };
    let mut dropped = 0usize;
    for seq in 0..n {
        // 돌려받은 버퍼가 있으면 그것에 채움
        if let Ok(j) = back_rx.try_recv() {
            (o.rgb, o.aligned, o.depth_raw, o.yuyv) = (j.rgb, j.aligned, j.raw, j.yuyv);
        }
        let fi = match cam.next_into(&mut o) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("rsd_rec: {e}");
                std::process::exit(3);
            }
        };
        let tc = format!("{:.6}", fi.t_ms * 1e-3);
        let o2 = |v: Option<u32>| v.map_or("-".to_string(), |x| x.to_string());
        let meta = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{{H}}\t{:.3}\t{:.3}",
            o2(fi.color_meta.frame_ts_us),
            o2(fi.depth_meta.frame_ts_us),
            fi.color_seq,
            fi.depth_seq,
            o2(fi.color_meta.frame_counter),
            o2(fi.depth_meta.frame_counter),
            o2(fi.color_meta.actual_exposure),
            o2(fi.color_meta.gain),
            fi.t_ms,
            fi.wait_ms,
            fi.align_ms
        );
        let job = Job { seq, tc: tc.clone(), meta, rgb: std::mem::take(&mut o.rgb), aligned: std::mem::take(&mut o.aligned), raw: std::mem::take(&mut o.depth_raw), yuyv: std::mem::take(&mut o.yuyv) };
        match tx.try_send(job) {
            Ok(()) => {
                let _ = writeln!(frgb, "{tc} rgb/{tc}.png");
                let _ = writeln!(fd, "{tc} depth/{tc}.png");
                let _ = writeln!(fraw, "{tc} depth_raw/{tc}.png");
                let _ = writeln!(frs, "{tc} depth/{tc}.png"); // depth_rs = depth(같은 rsdriver 정렬), 따로 쓰지 않음
            }
            Err(std::sync::mpsc::TrySendError::Full(j)) | Err(std::sync::mpsc::TrySendError::Disconnected(j)) => {
                dropped += 1;
                (o.rgb, o.aligned, o.depth_raw, o.yuyv) = (j.rgb, j.aligned, j.raw, j.yuyv);
            }
        }
    }
    drop(tx);
    for hd in hs {
        let _ = hd.join();
    }
    for (_, l) in lines.lock().unwrap().iter() {
        let _ = writeln!(ft, "{l}");
    }
    eprintln!("rsd_rec: 장면 {n}, 쓰기 줄이 꽉 차 버린 장면 {dropped}");
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rsd_rec: 리눅스 전용");
    std::process::exit(2);
}
