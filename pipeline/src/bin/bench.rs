//! 위치 추정·지도 평가 표: 같은 폴더의 memmap 을 정해진 조건들로 돌려 위치 오차·실패·속도·물체 수를 한 표로.
//!   bench [--depth <깊이 모델>] [--data data/tum] [--extra "<memmap 에 더 줄 인자>"] [--quick]
//! PC: --depth models/yolo26s-depth.onnx, 보드: --depth <...>/yolo26-depth-s_768x768.dxnn --extra "--seg <...>.dxnn"
use std::process::Command;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let depth = get("--depth").unwrap_or_else(|| "models/yolo26s-depth.onnx".into());
    let data = get("--data").unwrap_or_else(|| "data/tum".into());
    let extra: Vec<String> = get("--extra").map(|s| s.split_whitespace().map(String::from).collect()).unwrap_or_default();
    let quick = a.iter().any(|x| x == "--quick");
    let exe = std::env::current_exe().unwrap().with_file_name(if cfg!(windows) { "memmap.exe" } else { "memmap" });
    let (fr1, fr3) = ("rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg3_long_office_household");
    // (이름, 영상, 간격, 깊이, 위치 방식 인자)
    let mut cases: Vec<(&str, &str, usize, &str, Vec<&str>)> = vec![
        ("책상 · Kinect · 직접", fr1, 3, "kinect", vec!["--pose", "dvo"]),
        ("책상 · 딥러닝 · 직접", fr1, 3, "model", vec!["--pose", "dvo"]),
        ("책상 · 딥러닝 · 위치만 먼저(보드 흉내: 카메라 15 /초, 깊이 5 /초, 2 장 늦게)", fr1, 2, "model", vec!["--pose", "dvo", "--fast-pose", "--depth-every", "3", "--depth-lag", "2"]),
        ("사무실 · Kinect · 직접", fr3, 5, "kinect", vec!["--pose", "dvo"]),
        ("사무실 · 딥러닝 · 직접", fr3, 5, "model", vec!["--pose", "dvo"]),
        ("사무실 · 딥러닝 · 위치만 먼저(보드 흉내)", fr3, 2, "model", vec!["--pose", "dvo", "--fast-pose", "--depth-every", "3", "--depth-lag", "2"]),
    ];
    if !quick {
        cases.extend([
            ("책상 · 딥러닝 · 특징점", fr1, 3, "model", vec!["--pose", "vo"]),
            ("사무실 · 딥러닝 · 특징점", fr3, 5, "model", vec!["--pose", "vo"]),
            ("책상 · 딥러닝 · 직접(6 장 간격 = 깊이만 5 /초)", fr1, 6, "model", vec!["--pose", "dvo"]),
            ("사무실 · 딥러닝 · 직접(6 장 간격)", fr3, 6, "model", vec!["--pose", "dvo"]),
            ("사무실 · 딥러닝 · 직접(15 장 간격)", fr3, 15, "model", vec!["--pose", "dvo"]),
        ]);
    }
    let out = std::env::temp_dir().join("memmap_bench");
    println!("| 조건 | 위치 오차 | 실패 | 장/초 | 위치만 추적 ms | 물체 |");
    println!("|---|---|---|---|---|---|");
    for (name, seq, every, dep, pose) in cases {
        let dep = if dep == "model" { depth.as_str() } else { dep };
        let mut c = Command::new(&exe);
        c.args(["--tum", &format!("{data}/{seq}"), "--out", out.to_str().unwrap(), "--every", &every.to_string(), "--depth", dep]).args(&pose).args(&extra);
        let o = match c.output() {
            Ok(o) => o,
            Err(e) => {
                println!("| {name} | 실행 실패: {e} | | | | |");
                continue;
            }
        };
        let s = String::from_utf8_lossy(&o.stdout);
        let Some(j) = s.lines().rev().find_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()) else {
            println!("| {name} | 요약 없음(종료 코드 {:?}) | | | | |", o.status.code());
            continue;
        };
        let f = |v: &serde_json::Value| v.as_f64().unwrap_or(f64::NAN);
        println!(
            "| {name} | {:.2} m | {} | {:.1} | {:.1} | {} |",
            f(&j["vo"]["ate_rmse_m"]),
            j["vo"]["fail"],
            f(&j["kf_per_s"]),
            f(&j["pose_only_ms"]),
            j["objects"]
        );
    }
}
