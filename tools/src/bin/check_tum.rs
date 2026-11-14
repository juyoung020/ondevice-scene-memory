//! TUM 폴더 하나로 연결 프로그램 경로를 오프라인 경로와 견줌(docs/bridge.md "검증").
//!   check_tum <TUM 폴더> <출력 폴더> [최대 거리 m(기본 3.5)]
//! 환경 변수: BRIDGE(slam_bridge 또는 rtab_bridge), MEMMAP·ATE(pipeline 바이너리), INI(RTAB-Map 설정), SEG(분할 모델 .onnx)
//!   1) 기준: 같은 설정 + 모든 장면을 지도 노드로(Rtabmap/CreateIntermediateNodes true) 끝까지 돌린 뒤 최적화된 위치 → memmap --tum --gt
//!   2) 스트림: BRIDGE(보드와 같은 설정) | memmap --stream -
//!   3) 같은 위치 다른 입력: 스트림으로 보낸 위치 파일 → memmap --tum --gt (입력 경로만 다름)
//!   4) 위치 오차(ate, 정답 기준), 물체 수
//! 출력 폴더에 이미 결과가 있으면 그 단계는 건너뜀. 단계 하나가 실패하면 끝 코드 1 로 멈춤.
use std::fs::File;
use std::path::Path;
use std::process::{exit, Command, Stdio};

fn die(msg: &str) -> ! {
    eprintln!("check_tum: {msg}");
    exit(1);
}

fn env(k: &str) -> String {
    match std::env::var(k) {
        Ok(v) if !v.is_empty() => v,
        _ => die(&format!("{k}: parameter null or not set")),
    }
}

fn create(p: &str) -> File {
    File::create(p).unwrap_or_else(|e| die(&format!("{p}: {e}")))
}

/// 실행하고 실패면 멈춤
fn run(c: &mut Command) {
    match c.status() {
        Ok(s) if s.success() => {}
        Ok(s) => exit(s.code().unwrap_or(1)),
        Err(e) => die(&format!("{:?}: {e}", c.get_program())),
    }
}

/// $(명령): 표준 출력, 끝 줄바꿈 뺌(실패해도 계속)
fn capture(prog: &str, args: &[&str]) -> String {
    match Command::new(prog).args(args).stderr(Stdio::inherit()).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim_end_matches('\n').to_string(),
        Err(e) => {
            eprintln!("check_tum: {prog}: {e}");
            String::new()
        }
    }
}

/// grep -o 'key[chars]*' 를 줄바꿈으로 이음
fn grep_o(line: &str, key: &str, chars: &str) -> String {
    let mut out = vec![];
    let mut s = line;
    while let Some(p) = s.find(key) {
        let rest = &s[p + key.len()..];
        let n = rest.find(|c: char| !chars.contains(c)).unwrap_or(rest.len());
        out.push(format!("{key}{}", &rest[..n]));
        s = &rest[n..];
    }
    out.join("\n")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("check_tum <TUM 폴더> <출력 폴더> [최대 거리 m(기본 3.5)]");
        exit(1);
    }
    let (seq, out) = (a[1].as_str(), a[2].as_str());
    let maxd = a.get(3).map_or("3.5", |s| s.as_str());
    let (bridge, memmap, ate, ini, seg) = (env("BRIDGE"), env("MEMMAP"), env("ATE"), env("INI"), env("SEG"));
    std::fs::create_dir_all(out).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    let o = |f: &str| format!("{out}/{f}");
    let mm_args = ["--depth", "kinect", "--seg", seg.as_str(), "--every", "2", "--max-depth", maxd];

    if !Path::new(&o("offline.log")).exists() {
        run(Command::new(&bridge)
            .args(["--tum", seq, "--config", &ini, "--Rtabmap/CreateIntermediateNodes", "true", "--out", "/dev/null", "--final-poses", &o("ref_final.txt")])
            .stderr(create(&o("ref_bridge.log"))));
        let log = create(&o("offline.log"));
        let log2 = log.try_clone().unwrap();
        run(Command::new(&memmap).args(["--tum", seq, "--gt", &o("ref_final.txt")]).args(mm_args).args(["--out", &o("offline")]).stdout(log).stderr(log2));
    }
    if !Path::new(&o("stream.log")).exists() {
        let mut b = Command::new(&bridge)
            .args(["--tum", seq, "--config", &ini, "--poses", &o("sent.txt"), "--final-poses", &o("final.txt"), "--stats", &o("stats.csv")])
            .stdout(Stdio::piped())
            .stderr(create(&o("bridge.log")))
            .spawn()
            .unwrap_or_else(|e| die(&format!("{bridge}: {e}")));
        let pipe = b.stdout.take().unwrap();
        let m = Command::new(&memmap)
            .args(["--stream", "-"])
            .args(mm_args)
            .args(["--out", &o("stream")])
            .stdin(pipe)
            .stdout(create(&o("stream.log")))
            .stderr(create(&o("stream.err")))
            .status()
            .unwrap_or_else(|e| die(&format!("{memmap}: {e}")));
        let bs = b.wait().unwrap_or_else(|e| die(&format!("{bridge}: {e}")));
        // pipefail: 오른쪽부터 실패한 마지막 단계의 끝 코드
        if !m.success() {
            exit(m.code().unwrap_or(1));
        }
        if !bs.success() {
            exit(bs.code().unwrap_or(1));
        }
    }
    if !Path::new(&o("offline_sent.log")).exists() {
        let log = create(&o("offline_sent.log"));
        let log2 = log.try_clone().unwrap();
        run(Command::new(&memmap).args(["--tum", seq, "--gt", &o("sent.txt")]).args(mm_args).args(["--out", &o("offline_sent")]).stdout(log).stderr(log2));
    }

    let gt = format!("{seq}/groundtruth.txt");
    println!("== 위치 오차(정답 기준)");
    println!("기준(전 장면 노드, 끝에 최적화): {}", capture(&ate, &[&o("ref_final.txt"), &gt]));
    println!("스트림으로 보낸 위치(그때그때):   {}", capture(&ate, &[&o("sent.txt"), &gt]));
    println!("스트림 실행의 지도 노드(끝):     {}", capture(&ate, &[&o("final.txt"), &gt]));
    println!("memmap 이 받은 위치(poses_est):  {}", capture(&ate, &[&o("stream/poses_est.txt"), &gt]));
    println!("보낸 위치 대 memmap 받은 위치:    {}", capture(&ate, &[&o("stream/poses_est.txt"), &o("sent.txt"), "0.001"]));
    println!("== 물체·장면(키프레임, 물체, kf/s)");
    for k in ["offline", "offline_sent", "stream"] {
        let p = o(&format!("{k}.log"));
        let text = std::fs::read(&p).unwrap_or_else(|_| die(&format!("{p}: 읽을 수 없음")));
        let text = String::from_utf8_lossy(&text);
        let l = text.trim_end_matches('\n').rsplit('\n').next().unwrap_or("");
        println!(
            "{k}: {} {} {} {}",
            grep_o(l, "\"keyframes\":", "0123456789"),
            grep_o(l, "\"objects\":", "0123456789"),
            grep_o(l, "\"kf_per_s\":", "0123456789."),
            grep_o(l, "\"stream_lost\":", "0123456789")
        );
    }
    if let Ok(t) = std::fs::read(o("stream.err")) {
        for l in String::from_utf8_lossy(&t).lines() {
            if l.contains("수평 맞춤") || l.contains("스트림 끝") || l.contains("새 지도") {
                println!("{l}");
            }
        }
    }
    println!("== 연결 프로그램");
    let t = std::fs::read(o("bridge.log")).unwrap_or_else(|_| die(&format!("{}: 읽을 수 없음", o("bridge.log"))));
    let t = String::from_utf8_lossy(&t);
    let keep: Vec<&str> = t.lines().filter(|l| !l.contains("WARN")).collect();
    for l in &keep[keep.len().saturating_sub(9)..] {
        println!("{l}");
    }
    if keep.is_empty() {
        exit(1);
    }
}
