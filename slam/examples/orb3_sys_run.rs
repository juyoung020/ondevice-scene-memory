//! 녹화(rgb.txt · depth.txt, TUM 형식)를 ORB-SLAM3 System 으로 처음부터 끝까지 재생: 원본(libORB_SLAM3) 또는 Rust 판(`RustSystem`).
//! 장면마다 지도 스레드 한가를 기다림(원본 하네스 ds_run 과 같음). 출력 TSV: 시각 · 상태 · 지도 · Twc(3×4) · 추적 ms, 끝에 키프레임 목록(.kf).
//! cargo run --release --example orb3_sys_run -- rs|cpp 단어장 설정.yaml 녹화폴더 최대장수 출력.tsv
use slam::orb3sys::Orb3System;
use std::io::Write;
use std::time::Instant;

fn list(path: &str) -> Vec<(f64, String)> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split_whitespace();
            (it.next().unwrap().parse().unwrap(), it.next().unwrap().to_string())
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mode, voc, set, dir, maxf, out) = (&a[1], &a[2], &a[3], &a[4], a[5].parse::<usize>().unwrap(), &a[6]);
    let (rgb, dep) = (list(&format!("{dir}/rgb.txt")), list(&format!("{dir}/depth.txt")));
    let first = image::open(format!("{dir}/{}", rgb[0].1)).unwrap();
    let (w, h) = (first.width(), first.height());
    let t0 = Instant::now();
    let mut sys: Box<dyn Orb3System> = if mode == "rs" {
        let mut s = slam::orb3sys::rsys::RustSystem::open(set, w as i32, h as i32).unwrap();
        s.set_vocabulary(std::sync::Arc::new(slam::orb3loop::dbow2::Vocabulary::load(voc).unwrap()));
        Box::new(s)
    } else {
        slam::orb3sys::open(voc, set).unwrap()
    };
    eprintln!("열기 {:.1} s", t0.elapsed().as_secs_f64());
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
    let (mut j, mut n) = (0usize, 0usize);
    let mut ms_all = Vec::new();
    for (t, p) in &rgb {
        if n >= maxf {
            break;
        }
        while j + 1 < dep.len() && (dep[j + 1].0 - t).abs() < (dep[j].0 - t).abs() {
            j += 1;
        }
        if (dep[j].0 - t).abs() > 0.02 {
            continue;
        }
        // cv::imread 처럼 BGR 순서
        let mut c = image::open(format!("{dir}/{p}")).unwrap().to_rgb8().into_raw();
        for px in c.chunks_exact_mut(3) {
            px.swap(0, 2);
        }
        let d = image::open(format!("{dir}/{}", dep[j].1)).unwrap().to_luma16().into_raw();
        let s = Instant::now();
        let tr = sys.track(&c, &d, w, h, *t, None);
        let ms = s.elapsed().as_secs_f64() * 1e3;
        sys.wait_idle();
        ms_all.push(ms);
        write!(f, "{t:.6}\t{}\t{}", tr.state, tr.map_id).unwrap();
        for v in tr.twc {
            write!(f, "\t{v:?}").unwrap();
        }
        writeln!(f, "\t{ms:.3}\t{}", tr.ref_kf.map_or(-1, |k| k as i64)).unwrap();
        n += 1;
        if n % 500 == 0 {
            eprintln!("{n} 장, {:.1} s", t0.elapsed().as_secs_f64());
        }
    }
    f.flush().unwrap();
    let mut kf = std::io::BufWriter::new(std::fs::File::create(format!("{out}.kf")).unwrap());
    let kfs = sys.keyframes(true);
    for (id, m, p) in &kfs {
        write!(kf, "{id}\t{m}").unwrap();
        for v in p {
            write!(kf, "\t{v:.9}").unwrap();
        }
        writeln!(kf).unwrap();
    }
    ms_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ms_all.iter().sum::<f64>() / ms_all.len().max(1) as f64;
    eprintln!(
        "장면 {n} · 키프레임 {} · 추적 ms 평균 {mean:.2} 중간 {:.2} 최대 {:.2} · 전체 {:.1} s",
        kfs.len(),
        ms_all.get(ms_all.len() / 2).copied().unwrap_or(0.0),
        ms_all.last().copied().unwrap_or(0.0),
        t0.elapsed().as_secs_f64()
    );
    drop(sys);
}
