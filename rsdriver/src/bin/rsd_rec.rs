//! rsdriver 로 녹화: rs_rec 과 같은 폴더 꼴이라 tools 의 rs_align_check 를 그대로 돌릴 수 있다.
//!   rsd_rec <출력 폴더> [장면=60] [--size 640x480] [--fps 30] [--exposure ms] [--gain g] [--emitter on|off] [--skip 30]
//! 폴더: rgb/ depth/(= depth_rs/, rsdriver 정렬) depth_raw/ yuyv/<시각>.bin(원본 YUYV) calib.json frames.tsv(시각·순번·메타·해시)
#[cfg(target_os = "linux")]
fn main() {
    use rsdriver::camera::*;
    use std::io::Write;
    use tools::common::*;
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("rsd_rec <출력 폴더> [장면=60] [--size WxH] [--fps 30] [--exposure ms] [--gain g] [--emitter on|off] [--skip 30]");
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
    for s in ["rgb", "depth", "depth_raw", "depth_rs", "yuyv"] {
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
    let fnv = |b: &[u8]| b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
    let mut o = Output { want_raw: true, want_yuyv: true, ..Default::default() };
    for _ in 0..n {
        let fi = match cam.next_into(&mut o) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("rsd_rec: {e}");
                std::process::exit(3);
            }
        };
        let tc = format!("{:.6}", fi.t_ms * 1e-3);
        let _ = write_png_rgb(&join(&out, &format!("rgb/{tc}.png")), &o.rgb, w, h);
        let _ = write_png16(&join(&out, &format!("depth/{tc}.png")), &o.aligned, w, h);
        let _ = write_png16(&join(&out, &format!("depth_rs/{tc}.png")), &o.aligned, w, h);
        let _ = write_png16(&join(&out, &format!("depth_raw/{tc}.png")), &o.depth_raw, w, h);
        let _ = std::fs::write(join(&out, &format!("yuyv/{tc}.bin")), &o.yuyv);
        let _ = writeln!(frgb, "{tc} rgb/{tc}.png");
        let _ = writeln!(fd, "{tc} depth/{tc}.png");
        let _ = writeln!(fraw, "{tc} depth_raw/{tc}.png");
        let _ = writeln!(frs, "{tc} depth_rs/{tc}.png");
        let b16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let o2 = |v: Option<u32>| v.map_or("-".to_string(), |x| x.to_string());
        let _ = writeln!(
            ft,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:.3}\t{:.3}",
            o2(fi.color_meta.frame_ts_us),
            o2(fi.depth_meta.frame_ts_us),
            fi.color_seq,
            fi.depth_seq,
            o2(fi.color_meta.frame_counter),
            o2(fi.depth_meta.frame_counter),
            o2(fi.color_meta.actual_exposure),
            o2(fi.color_meta.gain),
            fi.t_ms,
            fnv(&o.yuyv),
            fnv(&b16(&o.depth_raw)),
            fnv(&o.rgb),
            fnv(&b16(&o.aligned)),
            fi.wait_ms,
            fi.align_ms
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("rsd_rec: 리눅스 전용");
    std::process::exit(2);
}
