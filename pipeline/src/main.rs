//! 손카메라 영상 + 포즈 → 점유 지도·3D 점 지도·물체 기억·씬그래프(방·엣지) → 기억 폴더(view.json, sgview 가 읽음).
//!
//!   memmap --tum data/tum/<seq> --out data/memory/<name> [--seg models/yolo26s-seg.onnx] [--every 5] [--conf 0.35]
//!          [--live 1] [--no-skip] [--loop] [--scene-every 10] [--snaps 150,500]
//!   rtab_bridge ... | memmap --stream - --out <dir> ...   (위치·컬러·깊이를 연결 프로그램 bridge/ 에서 받음, docs/bridge.md)
//!
//! 지도·트래커는 누적식(maps.rs): 키프레임마다 새 점만 해시 칸에 더하고, 기록은 칸을 한 번 훑기만 한다.
//! 깊이는 1 단계로 TUM Kinect 실측을 쓴다(딥러닝 깊이는 다음 단계).
#[cfg(feature = "cam")]
mod cam;
mod dvo;
mod embed;
mod geom;
mod graph;
mod kfstore;
mod maps;
mod out;
mod rec;
mod rooms;
mod scale;
mod seg;
mod segsim;
mod stream;
mod vo;
mod tum;

use geom::*;
use maps::*;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone)]
struct Args {
    tum: PathBuf,
    out: PathBuf,
    seg: String,
    every: usize,
    conf: f32,
    live: f64,
    no_skip: bool,
    looping: bool,
    scene_every: u32,
    depth: String,
    scale_win: usize,
    vo: bool,
    dvo: bool, // --pose dvo: 특징점 없이 RGB-D 직접 방식
    cam: Option<String>,
    k: Option<[f64; 4]>,
    gt: Option<String>, // --gt <파일>: TUM 영상의 위치를 이 파일(TUM 형식)로(예: RTAB-Map 이 낸 위치)
    sp: String,
    snaps: Vec<u32>,
    rec: Option<PathBuf>,
    seg_every: usize,
    npu_par: bool,
    npu_jobs: usize,    // --npu-jobs K: 스트림 분할 추론을 NPU 에 K 장까지 함께 올림(코어 여럿이 나눠 돎, 기본 4, 1 = 한 장씩)
    sp_every: usize,
    fast_pose: bool,   // 모든 장면은 위치만 먼저 추적, 깊이·분할은 따로(카메라: 모델이 쉬는 대로 최신 장면에)
    depth_every: usize, // --fast-pose 로 TUM 을 돌릴 때 깊이 장면 간격
    depth_lag: usize,   // --fast-pose 로 TUM 을 돌릴 때 깊이 장면을 위치만 추적 장면 몇 장 뒤에 넣을지(보드의 모델 지연 흉내) // SuperPoint 는 N 장마다(사이 장면은 RGB-D 직접 방식으로 위치)
    threads: String,    // --threads: 스레드 우선순위·코어 배치(auto · none · 역할=nice[:코어],…)
    slow: f32,          // --slow F: 보드 흉내, 본·점·다시 쌓기 스레드 일을 F 배로(코어를 붙잡고 기다림)
    seg_cache: Option<PathBuf>, // --seg-cache DIR: 보드 흉내, 장면별 분할 결과를 폴더에 남기고 있으면 읽어 씀(추론이 코어를 쓰지 않음)
    seg_cpu_ms: f64,    // --seg-cpu-ms M: --seg-cache 로 읽어 온 장면마다 읽기 스레드가 M ms 코어를 붙잡음(보드 분할의 CPU 전·후처리 몫)
    stream_rate: f64,   // --stream-rate R: 스트림을 찍은 시각대로 R 배속으로 내보냄(0 = 최대 속도), 밀림(지연) 기록
    seg_min_ms: f64,    // --seg-min-ms M: 분할 한 번(전처리 + 추론 + 후처리)이 적어도 M ms 걸린 것처럼, 모자란 만큼 추론 단계에서 기다림(보드 분할 97 ms 흉내)
    infer_min_ms: f64,  // --infer-min-ms N: 분할 추론 단계만 적어도 N ms 걸린 것처럼 기다림(보드 NPU 추론 시간 흉내; 전·후처리는 --slow 배)
    seg_pipe: bool,     // --seg-pipe off: 스트림 분할의 전처리 → 추론 → 후처리를 읽기 스레드 하나에서 차례로(기본: 단계마다 스레드를 두어 서로 겹침)
    seg_check: usize,   // --seg-check N: 지도는 만들지 않고, 스트림 분할 장면 N 장에 전·후처리를 기준 구현과 견줌(결과 같음 여부 + 시간)
    no_drop: bool,      // --no-drop: 스트림이 밀려도 장면을 버리지 않음(재생 결과가 늘 같게)
    seg_skip: Option<(f64, f64, usize, f64)>, // --seg-skip M,DEG[,N[,L]] (기본 0.10,10,4,0.1, off 로 끔): 스트림이 L 초(기본 0.1) 넘게 밀렸을 때만, 직전 분할 키프레임보다 M 미터·DEG 도 안쪽이면 분할 생략(N 장째는 꼭 분할)
    scene_vox: f64,
    max_depth: f64,
    scene_step: usize, // 3D 지도에 쓸 화소 간격
    pts_depth: f64,    // --pts-depth M: 점 지도에 넣을 최대 깊이(멀수록 깊이 잡음이 커 표면이 두꺼워짐)
    pts_obs: u8,       // --pts-obs N: 고운 칸이 점이 되려면 키프레임 N 장 이상에서 보여야(한두 번 맞은 잡음 점을 거름)
    stream: Option<String>, // --stream <파일|->: 연결 프로그램(bridge/)이 보내는 장면 스트림(위치·컬러·깊이)
    no_level: bool,         // --no-level: 스트림 위치의 z 를 그대로 위로(바닥 합의로 수평을 다시 맞추지 않음)
    rebuild_every: f64,     // --rebuild-every S: 지도 노드가 움직이면(루프 클로저) S 초에 한 번까지 보관한 장면을 고친 위치로 다시 쌓음(0 = 끝에만, --no-rebuild = 안 함)
    embed: Option<String>,  // --embed <vision_model.onnx>: 물체 사진마다 검색용 벡터(SigLIP2 영상 인코더)를 남김
    kf_store_mb: usize,     // --kf-store-mb: 다시 쌓기용 장면 보관 예산(넘으면 하나 걸러 버림)
}

fn args() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let get = |k: &str| v.iter().position(|a| a == k).and_then(|i| v.get(i + 1)).cloned();
    Args {
        tum: get("--tum").unwrap_or_default().into(),
        out: get("--out").or_else(|| get("--seg-check").map(|_| String::new())).expect("--out <dir>").into(), // --seg-check 는 쓰는 파일이 없음
        seg: get("--seg").unwrap_or("models/yolo26s-seg.onnx".into()),
        embed: get("--embed"),
        every: get("--every").map_or(5, |s| s.parse().unwrap()),
        conf: get("--conf").map_or(0.35, |s| s.parse().unwrap()),
        live: get("--live").map_or(0.0, |s| s.parse().unwrap()),
        no_skip: v.iter().any(|a| a == "--no-skip"),
        looping: v.iter().any(|a| a == "--loop"),
        scene_every: get("--scene-every").map_or(10, |s| s.parse().unwrap()),
        depth: get("--depth").unwrap_or("kinect".into()),
        scale_win: get("--scale-win").map_or(1, |s| s.parse().unwrap()),
        vo: get("--pose").is_some_and(|p| p == "vo" || p == "dvo"),
        dvo: get("--pose").is_some_and(|p| p == "dvo"),
        cam: get("--cam"),
        gt: get("--gt"),
        k: get("--k").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3]] }),
        sp: get("--superpoint").unwrap_or(if cfg!(feature = "dxrt") { "superpoint_480x640.dxnn" } else { "models/superpoint.onnx" }.into()),
        snaps: get("--snaps").map_or(vec![], |s| s.split(',').filter_map(|x| x.parse().ok()).collect()),
        rec: get("--rec").map(PathBuf::from),
        seg_every: get("--seg-every").map_or(1, |s| s.parse().unwrap()),
        npu_par: v.iter().any(|x| x == "--npu-par"),
        npu_jobs: get("--npu-jobs").and_then(|s| s.parse().ok()).unwrap_or(4).clamp(1, 8),
        sp_every: get("--sp-every").map_or(1, |s| s.parse().unwrap()),
        fast_pose: v.iter().any(|x| x == "--fast-pose"),
        depth_every: get("--depth-every").map_or(3, |s| s.parse().unwrap()),
        depth_lag: get("--depth-lag").map_or(0, |s| s.parse().unwrap()),
        threads: get("--threads").unwrap_or("auto".into()),
        slow: get("--slow").map_or(1.0, |s| s.parse().unwrap()),
        stream_rate: get("--stream-rate").map_or(0.0, |s| s.parse().unwrap()),
        seg_min_ms: get("--seg-min-ms").map_or(0.0, |s| s.parse().unwrap()),
        seg_cache: get("--seg-cache").map(Into::into),
        seg_cpu_ms: get("--seg-cpu-ms").map_or(0.0, |s| s.parse().unwrap()),
        infer_min_ms: get("--infer-min-ms").map_or(0.0, |s| s.parse().unwrap()),
        seg_pipe: get("--seg-pipe").is_none_or(|s| s != "off") && get("--seg-cache").is_none(), // --seg-cache 는 분할 한 번을 통째로 읽어 오므로 차례 실행
        seg_check: get("--seg-check").map_or(0, |s| s.parse().unwrap()),
        no_drop: v.iter().any(|x| x == "--no-drop"),
        // 기본 켬(0.10,10,4,0.1): 밀리지 않으면 늘 분할하므로 빠른 PC 에선 동작 그대로, "off" 로 끔
        seg_skip: match get("--seg-skip") { Some(s) if s == "off" => None, s => { let s = s.unwrap_or_else(|| "0.10,10,4,0.1".into());
            let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
            Some((v[0], v[1], v.get(2).map_or(4, |n| *n as usize), v.get(3).copied().unwrap_or(0.1))) } },
        scene_vox: get("--scene-vox").map_or(0.01, |s| s.parse().unwrap()),
        max_depth: get("--max-depth").map_or(4.0, |s| s.parse().unwrap()),
        scene_step: get("--scene-step").map_or(2, |s| s.parse().unwrap()),
        pts_depth: get("--pts-depth").map_or(3.0, |s| s.parse().unwrap()),
        pts_obs: get("--pts-obs").map_or(3, |s| s.parse().unwrap()),
        stream: get("--stream"),
        no_level: v.iter().any(|x| x == "--no-level"),
        rebuild_every: if v.iter().any(|x| x == "--no-rebuild") { -1.0 } else { get("--rebuild-every").map_or(5.0, |s| s.parse().unwrap()) },
        kf_store_mb: get("--kf-store-mb").map_or(3000, |s| s.parse().unwrap()),
    }
}

/// 마스크 침식(정사각 k×k): 가장자리 깊이 번짐 제거
fn erode(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut a = vec![false; w * h];
    // 가로 → 세로 두 번(분리 가능)
    for y in 0..h {
        let mut run = 0usize;
        for x in 0..w {
            run = if m[y * w + x] { run + 1 } else { 0 };
            if run > 2 * r {
                a[y * w + x - r] = true;
            }
        }
    }
    // 세로는 열마다 이어진 길이를 들고 줄 순서로 훑음(메모리를 차례로 읽게)
    let mut b = vec![false; w * h];
    let mut run = vec![0usize; w];
    for y in 0..h {
        for x in 0..w {
            run[x] = if a[y * w + x] { run[x] + 1 } else { 0 };
            if run[x] > 2 * r {
                b[(y - r) * w + x] = true;
            }
        }
    }
    b
}

/// 얇고 넓은 수직면(두께 < 8 cm, 폭 > 0.8 m, 법선 수평)이면 벽 — 물체에서 뺀다
fn is_wall(p: &[(V3, [u8; 3])]) -> bool {
    if p.len() < 100 {
        return false;
    }
    let n = p.len() as f64;
    let m = [0, 1, 2].map(|k| p.iter().map(|q| q.0[k]).sum::<f64>() / n);
    let mut c = [[0.0; 3]; 3];
    for q in p {
        let d = sub(&q.0, &m);
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] += d[i] * d[j] / n;
            }
        }
    }
    let (w, v) = eig_sym(&c);
    let sd = w.map(|x| x.max(0.0).sqrt());
    sd[0] * 2.0 < 0.08 && sd[1] * 4.0 > 0.8 && v[2][0].abs() < 0.35
}

#[derive(Default)]
struct Clock {
    load: f64,
    seg: f64,
    vo: f64,
    map: f64,
    obj: f64,
    write: f64,
    n: u32,
    frame: Vec<f32>, // 장면마다 본 스레드가 쓴 시간(ms): 멈춤 최대·95 % 보기
    part: [f64; NPART], // 이번 장면 단계별 시간(초)
    part_max: [f32; NPART],
    part_sum: [f64; NPART],
    worst: Vec<(f32, u32, [f32; NPART])>, // 가장 긴 장면 몇 개: (ms, 장면 번호, 단계별 ms)
    lag: Vec<f32>, // 밀림: 장면을 끝낸 시각 − 나올 시각(초)
    t_span: (f64, f64), // 지도에 넣은 장면의 첫·끝 찍은 시각
}

/// 본 스레드 장면 단계(가장 긴 장면이 어디서 오는지 보기)
const PART_NAMES: [&str; 11] = ["graph", "install", "rb_start", "vo", "map", "obj", "pts", "scene_write", "view_write", "merge", "prune"];
const NPART: usize = PART_NAMES.len();
#[derive(Clone, Copy)]
enum P { Graph, Install, RbStart, Vo, Map, Obj, Pts, Scene, View, Merge, Prune }

impl Clock {
    fn add(&mut self, p: P, c: Instant) {
        self.part[p as usize] += c.elapsed().as_secs_f64();
    }
    fn end_frame(&mut self, ms: f32, kf: u32) {
        self.frame.push(ms);
        let pm: [f32; NPART] = self.part.map(|x| (x * 1e3) as f32);
        for i in 0..NPART {
            self.part_max[i] = self.part_max[i].max(pm[i]);
            self.part_sum[i] += self.part[i];
        }
        self.part = [0.0; NPART];
        self.worst.push((ms, kf, pm));
        if self.worst.len() > 16 {
            self.worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            self.worst.truncate(8);
        }
    }
    fn parts_json(&mut self) -> serde_json::Value {
        self.worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        self.worst.truncate(8);
        let n = self.frame.len().max(1) as f64;
        let named = |v: &[f32; NPART]| -> serde_json::Map<String, serde_json::Value> {
            PART_NAMES.iter().zip(v).filter(|(_, x)| **x >= 0.5).map(|(k, x)| (k.to_string(), json!((*x * 10.0).round() / 10.0))).collect()
        };
        json!({
            "max": named(&self.part_max),
            "mean": named(&self.part_sum.map(|x| (x / n * 1e3) as f32)),
            "worst": self.worst.iter().map(|(ms, kf, p)| json!({"kf": kf, "ms": ms.round(), "parts": named(p)})).collect::<Vec<_>>(),
        })
    }
}

/// 깊이 모델: kinect(실측) 이면 없음, .dxnn 은 NPU, .onnx 는 PC
fn make_depth(path: &str) -> Option<Box<dyn seg::DepthNet>> {
    if path == "kinect" {
        return None;
    }
    #[cfg(feature = "dxrt")]
    if path.ends_with(".dxnn") {
        return Some(Box::new(seg::DxDepth::new(path)));
    }
    #[cfg(feature = "onnx")]
    if path.ends_with(".onnx") {
        return Some(Box::new(seg::OnnxDepth::new(path)));
    }
    panic!("no backend for depth {path}")
}

/// 특징점 모델(SuperPoint): .dxnn 은 NPU, .onnx 는 PC
fn make_sp(path: &str) -> Box<dyn seg::FeatNet> {
    #[cfg(feature = "dxrt")]
    if path.ends_with(".dxnn") {
        return Box::new(seg::DxSuperPoint::new(path));
    }
    #[cfg(feature = "onnx")]
    if path.ends_with(".onnx") {
        return Box::new(seg::OnnxSuperPoint::new(path));
    }
    panic!("no backend for superpoint {path}")
}

/// 모델 셋 실행: 깊이·특징점·분할(do_seg 일 때). par 면 셋을 동시에(DX-M1 NPU 코어 3 개), 아니면 차례대로.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_models(dnet: &mut Option<Box<dyn seg::DepthNet>>, seg: &mut Box<dyn seg::Segmenter + Send>, spnet: &mut Option<Box<dyn seg::FeatNet>>,
              rgb: &[u8], w: usize, h: usize, conf: f32, do_seg: bool, par: bool, do_sp: bool) -> (Option<Vec<f32>>, Option<Vec<seg::Det>>, Option<vo::Feat>) {
    if par {
        std::thread::scope(|sc| {
            let hd = sc.spawn(|| dnet.as_mut().map(|n| n.run(rgb, w, h)));
            let hf = sc.spawn(|| spnet.as_mut().filter(|_| do_sp).map(|n| n.run(rgb, w, h)));
            let dets = do_seg.then(|| seg.run(rgb, w, h, conf));
            (hd.join().unwrap(), dets, hf.join().unwrap())
        })
    } else {
        let pd = dnet.as_mut().map(|n| n.run(rgb, w, h));
        let dets = do_seg.then(|| seg.run(rgb, w, h, conf));
        (pd, dets, spnet.as_mut().filter(|_| do_sp).map(|n| n.run(rgb, w, h)))
    }
}

/// 한 장면의 '대략 수평'인 가장 큰 면(바닥·책상)을 RANSAC 으로 찾아 그 위쪽 법선을 세계 좌표로(IMU 없이 중력 방향 단서).
/// 여러 장면의 법선을 모아 다수결로 수평을 정한다(gravity 합의).
fn up_normal(depth: &[f32], w: usize, h: usize, k: [f64; 4], r: &M3) -> Option<V3> {
    let [fx, fy, cx, cy] = k;
    let mut pts: Vec<V3> = vec![];
    for v in (0..h).step_by(8) {
        for u in (0..w).step_by(8) {
            let z = depth[v * w + u] as f64;
            if z > 0.2 && z < 4.0 {
                pts.push([(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z]);
            }
        }
    }
    if pts.len() < 200 {
        return None;
    }
    let up_c = [r[2][0], r[2][1], r[2][2]]; // 지금 가정한 세계 위(z)를 카메라 좌표로(Rᵀ·ẑ)
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    let (mut best_n, mut best_c) = ([0.0; 3], 0usize);
    for _ in 0..300 {
        let (a, b, c) = (pts[rnd(pts.len())], pts[rnd(pts.len())], pts[rnd(pts.len())]);
        let mut n = cross(&sub(&b, &a), &sub(&c, &a));
        let l = norm(&n);
        if l < 1e-6 {
            continue;
        }
        n = n.map(|x| x / l);
        if dot(&n, &up_c) < 0.0 {
            n = n.map(|x| -x);
        }
        if dot(&n, &up_c) < 50f64.to_radians().cos() {
            continue; // 가정한 위에서 50° 넘게 기운 면(벽)은 제외
        }
        let d = dot(&n, &a);
        let cnt = pts.iter().filter(|q| (dot(&n, q) - d).abs() < 0.02 + 0.01 * q[2]).count();
        if cnt > best_c {
            (best_n, best_c) = (n, cnt);
        }
    }
    if best_c < pts.len() / 5 {
        return None;
    }
    Some(mv(r, &best_n))
}

/// 딥러닝 깊이 배율 맞추기: (이번 깊이 z, 랜드마크로 본 거리 zc) 짝의 zc / z 중앙값(0.5–2). 짝이 20 개 미만이면 None.
fn depth_scale(zz: &[(f64, f64)]) -> Option<f64> {
    if zz.len() < 20 {
        return None;
    }
    let mut rs: Vec<f64> = zz.iter().map(|(z, zc)| zc / z).collect();
    rs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Some(rs[rs.len() / 2].clamp(0.5, 2.0))
}
fn make_seg(path: &str) -> Box<dyn seg::Segmenter + Send> {
    #[cfg(feature = "dxrt")]
    if path.ends_with(".dxnn") {
        return Box::new(seg::DxSeg::new(path));
    }
    #[cfg(feature = "onnx")]
    {
        return Box::new(seg::OnnxSeg::new(seg::ort_env(), path));
    }
    #[allow(unreachable_code)]
    {
        panic!("no backend for {path}")
    }
}

struct Frame {
    t: f64,
    p: V3,
    r: M3,
    rgb: Vec<u8>,
    w: usize,
    h: usize,
    depth: Vec<f32>,
    dets: Option<Vec<seg::Det>>, // 이번 장면에 분할을 안 돌렸으면 None(--seg-every)
    load: f64,
    seg: f64,
    scale: f64, // 딥러닝 깊이 / Kinect 중앙값(비교용, Kinect 깊이를 쓸 땐 NaN)
    feat: Option<vo::Feat>,
    track_only: bool, // 깊이·분할 없이 위치만 추적하는 장면(--fast-pose)
    lost: bool,       // 스트림 입력에서 위치를 놓친 장면(지도·물체에 넣지 않음)
    held: bool,       // 스트림: 맡아 둘 장면(믿는 지도와 이어지면 다시 쌓기 때 지도에 들어감)
    anchor: u32,      // 스트림: 기준 지도 노드 id(0 = 없음)
    graph: Vec<stream::Node>, // 스트림: 이 장면 앞에 온 그래프 블록 노드
}

/// 스트림 분할 단계별 걸린 시간 합(초)과 분할 횟수, 분할한 구간의 벽시계 길이(초)
#[derive(Default, Clone, Copy)]
struct SegStages {
    pre: f64,
    infer: f64,
    post: f64,
    n: u32,
    wall: f64,
}

/// 스트림 장면 하나가 분할 단계(전처리 → 추론 → 후처리)를 지나며 들고 다니는 것. 분할을 안 하는 장면(위치 놓침·생략)도 순서를 지키려 같은 길로 지나간다.
struct SegJob {
    sf: stream::StreamFrame,
    depth: Vec<f32>,
    do_seg: bool, // 이 장면에 분할을 돌림(위치를 놓친 장면은 늘 false)
    load: f64,
    graph: Vec<stream::Node>,
    bufs: (seg::SegIn, seg::SegRaw), // 전처리 결과, 추론 결과(다시 쓰는 버퍼)
    dets: Option<Vec<seg::Det>>, // 실행기가 단계를 거치지 않고 바로 준 결과(--seg-cache)
    pd: Option<Vec<f32>>,
    feat: Option<vo::Feat>,
    t_front: f64, // 고른 뒤 전처리 끝까지(깊이 변환 + 전처리)
    t: [f64; 3],  // 전처리, 추론, 후처리
}

/// 분할 단계 2 — 모델 호출: 분할 추론(깊이·특징점 모델을 쓰면 그것도). 한 장씩 차례로 부른다(`--npu-jobs` 로 여러 장을 함께 올릴 때는 스트림 쪽에서 올리기·받기를 나눠 한다).
struct InferStage {
    seg: Box<dyn seg::Segmenter + Send>,
    dnet: Option<Box<dyn seg::DepthNet>>,
    spnet: Option<Box<dyn seg::FeatNet>>,
    par: bool,
    infer_min: f64, // --infer-min-ms(초)
    seg_min: f64,   // --seg-min-ms(초)
    conf: f32,
    stats: std::sync::Arc<std::sync::Mutex<SegStages>>,
}
impl InferStage {
    fn run(&mut self, j: &mut SegJob) {
        if j.sf.lost() {
            return;
        }
        let (rgb, w, h) = (&j.sf.rgb[..], j.sf.w, j.sf.h);
        let (seg, dnet, spnet) = (&mut self.seg, &mut self.dnet, &mut self.spnet);
        // 흉내: 추론만 N ms, 또는 셋 합이 M ms 가 되게(후처리는 지금까지 평균으로 어림)
        let post_est = { let s = self.stats.lock().unwrap(); if s.n > 0 { s.post / s.n as f64 } else { 0.0 } };
        let want = self.infer_min.max(self.seg_min - j.t_front - post_est);
        let (do_seg, bufs, direct, conf, c_all) = (j.do_seg, &mut j.bufs, &mut j.dets, self.conf, Instant::now());
        let mut infer = || {
            if do_seg {
                let c = Instant::now();
                *direct = seg.direct(rgb, w, h, conf);
                if direct.is_none() {
                    seg.infer(&bufs.0, &mut bufs.1);
                }
                let pad = want - c.elapsed().as_secs_f64();
                if pad > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(pad));
                }
            }
        };
        let (pd, feat) = if self.par {
            std::thread::scope(|sc| {
                let hd = sc.spawn(|| dnet.as_mut().map(|n| n.run(rgb, w, h)));
                let hf = sc.spawn(|| spnet.as_mut().map(|n| n.run(rgb, w, h)));
                infer();
                (hd.join().unwrap(), hf.join().unwrap())
            })
        } else {
            let pd = dnet.as_mut().map(|n| n.run(rgb, w, h));
            infer();
            (pd, spnet.as_mut().map(|n| n.run(rgb, w, h)))
        };
        (j.pd, j.feat, j.t[1]) = (pd, feat, c_all.elapsed().as_secs_f64());
    }
}

/// 분할 단계 3 — 후처리: 마스크 해독, 본 스레드에 넘길 장면으로 묶음. 버퍼는 돌려준다.
fn seg_stage_post(mut j: SegJob, conf: f32, stats: &std::sync::Mutex<SegStages>) -> (Frame, (seg::SegIn, seg::SegRaw)) {
    let (w, h, lost, held) = (j.sf.w, j.sf.h, j.sf.lost(), j.sf.held());
    let dets = j.do_seg.then(|| {
        let c = Instant::now();
        let d = j.dets.take().unwrap_or_else(|| seg::seg_post(&j.bufs.1, j.bufs.0.geom, w, h, conf));
        out::slow(c);
        j.t[2] = c.elapsed().as_secs_f64();
        let mut s = stats.lock().unwrap();
        (s.pre, s.infer, s.post, s.n) = (s.pre + j.t[0], s.infer + j.t[1], s.post + j.t[2], s.n + 1);
        d
    });
    let sf = j.sf;
    let f = Frame { t: sf.t, p: sf.p, r: sf.r, rgb: sf.rgb, w, h, depth: j.pd.unwrap_or(j.depth), dets, load: j.load, seg: j.t_front + j.t[1] + j.t[2], scale: f64::NAN, feat: j.feat,
                    track_only: false, lost, held, anchor: sf.anchor, graph: j.graph };
    (f, j.bufs)
}

/// 본 스레드가 받아 간 장면 수(분할 단계가 다음 장면을 언제 고를지 정하는 데 씀)
#[derive(Default)]
struct Taken {
    n: std::sync::Mutex<u64>,
    cv: std::sync::Condvar,
    dead: std::sync::atomic::AtomicBool, // 본 스레드가 끝나 더 받지 않음
}

/// --seg-check N: 스트림 앞쪽 분할 장면 N 장(--every 간격)에 전·후처리를 기준 구현과 지금 구현으로 각각 돌려, 결과가 같은지와 걸린 시간을 견준다.
/// 모델은 장면마다 한 번 돌려 같은 출력을 두 후처리에 준다. 다른 장면 수(레터박스·검출)와 다른 화소 수가 모두 0 이면 같음.
fn seg_check(a: &Args, seg: &mut dyn seg::Segmenter, n_max: usize) {
    let mut rd = stream::StreamReader::open(a.stream.as_deref().expect("--seg-check 는 --stream 과 함께 쓴다"));
    let (mut inp, mut raw) = (seg::SegIn::default(), seg::SegRaw::default());
    let mut t = [0.0f64; 5]; // 기준 전처리, 지금 전처리, 추론, 기준 후처리, 지금 후처리(초 합)
    let (mut n, mut i, mut n_det, mut px_mask, mut bad_in, mut bad_det, mut bad_px) = (0usize, 0usize, 0u64, 0u64, 0u64, 0u64, 0u64);
    let time = |f: &mut dyn FnMut()| {
        let c = Instant::now();
        f();
        c.elapsed().as_secs_f64()
    };
    while let Some(sf) = rd.next() {
        i += 1;
        if (i - 1) % a.every != 0 || sf.lost() {
            continue;
        }
        let (w, h) = (sf.w, sf.h);
        // 캐시 덕을 한쪽만 보지 않게 장면마다 순서를 바꿈
        let (mut lb0, mut d0, mut d1) = ((vec![], seg::LbGeom::default()), vec![], vec![]);
        let mut pre0 = || lb0 = seg::reference::letterbox(&sf.rgb, w, h, seg::SEG_S);
        let mut pre1 = || seg::seg_pre(&sf.rgb, w, h, &mut inp);
        if n % 2 == 0 { t[0] += time(&mut pre0); t[1] += time(&mut pre1); } else { t[1] += time(&mut pre1); t[0] += time(&mut pre0); }
        bad_in += (lb0.0 != inp.u8 || lb0.1 != inp.geom) as u64;
        t[2] += time(&mut || seg.infer(&inp, &mut raw));
        let (det, proto) = (&raw.buf[raw.det.0..raw.det.0 + raw.det.1], &raw.buf[raw.proto.0..raw.proto.0 + raw.proto.1]);
        let mut post0 = || d0 = seg::reference::decode(det, proto, raw.pc, raw.ph, raw.pw, seg::SEG_S, inp.geom, w, h, a.conf);
        let mut post1 = || d1 = seg::seg_post(&raw, inp.geom, w, h, a.conf);
        if n % 2 == 0 { t[3] += time(&mut post0); t[4] += time(&mut post1); } else { t[4] += time(&mut post1); t[3] += time(&mut post0); }
        n_det += d0.len() as u64;
        if d0.len() != d1.len() {
            bad_det += d0.len().abs_diff(d1.len()) as u64;
        }
        for (p, q) in d0.iter().zip(&d1) {
            px_mask += p.mask.len() as u64;
            if (p.cls, p.score.to_bits(), p.box_) != (q.cls, q.score.to_bits(), q.box_) || p.mask.len() != q.mask.len() {
                bad_det += 1;
            } else {
                bad_px += p.mask.iter().zip(&q.mask).filter(|(x, y)| x != y).count() as u64;
            }
        }
        n += 1;
        if n >= n_max {
            break;
        }
    }
    let m = 1e3 / n.max(1) as f64;
    let sl = a.slow.max(1.0) as f64;
    println!("분할 전·후처리 검사: 장면 {n} 장, 검출 {n_det} 개, 마스크 화소 {px_mask} 개");
    println!("  다른 것: 레터박스 {bad_in} 장, 검출(클래스·점수·상자) {bad_det} 개, 마스크 화소 {bad_px} 개 → {}", if bad_in + bad_det + bad_px == 0 { "같음" } else { "다름" });
    println!("  장면당 ms(괄호: --slow {sl} 배)   기준      지금");
    println!("  전처리(레터박스)        {:8.3} {:8.3}   ({:.2} → {:.2})", t[0] * m, t[1] * m, t[0] * m * sl, t[1] * m * sl);
    println!("  후처리(마스크 해독)     {:8.3} {:8.3}   ({:.2} → {:.2})", t[3] * m, t[4] * m, t[3] * m * sl, t[4] * m * sl);
    println!("  전처리 + 후처리         {:8.3} {:8.3}   ({:.2} → {:.2})", (t[0] + t[3]) * m, (t[1] + t[4]) * m, (t[0] + t[3]) * m * sl, (t[1] + t[4]) * m * sl);
    println!("  추론(이 실행기)         {:8.3}", t[2] * m);
    if bad_in + bad_det + bad_px != 0 {
        std::process::exit(1);
    }
}

fn run(a: &Args, seg: Box<dyn seg::Segmenter + Send>) -> Box<dyn seg::Segmenter + Send> {
    // 스트림 입력이면 첫 장면의 보정값, 카메라 입력이면 빈 시퀀스(보정값만), 아니면 TUM
    let mut stream_rd = a.stream.as_deref().map(stream::StreamReader::open);
    let stream_first = stream_rd.as_mut().map(|r| r.next().expect("--stream: 장면이 하나도 오지 않음"));
    let seq = if let Some(f) = &stream_first {
        tum::Tum::empty(f.k)
    } else if a.cam.is_some() {
        tum::Tum::empty(a.k.unwrap_or([477.0, 477.0, 320.0, 240.0]))
    } else {
        tum::Tum::open(&a.tum, a.gt.as_deref().map(Path::new))
    };
    std::fs::create_dir_all(a.out.join("objects")).unwrap();
    let [fx, fy, cx, cy] = seq.k;
    let mut grid = Grid::new(0.05);
    // 점 지도는 점 작업 스레드가(스트림은 밀리면 건너뛰고, TUM 입력은 기다려 결과가 늘 같게)
    let mut pw = Some({
        let mut s = SceneMap::new(a.scene_vox, 0.04);
        s.min_obs = a.pts_obs;
        PtsWorker::spawn(std::sync::Arc::new(a.clone()), seq.k, s, a.stream.is_none())
    });
    let mut mem = Memory::new(0.012, 0.04);
    let mut room_tr = rooms::Tracker::default(); // 방 번호를 실행 내내 잇는다
    let mut traj: Vec<(f64, [f64; 3])> = vec![];
    let mut hist_h: Vec<f64> = vec![];
    let mut z0: Option<f64> = None;
    let (mut t_start, mut w_start) = (None::<f64>, Instant::now());
    let mut clock = Clock::default();
    let mut nkf: u32 = 0;
    let mut cam_json = json!(null);
    let mut last_view = Instant::now();
    let mut last_scene = Instant::now();
    let mut last_merge = 0u32;
    let mut scene_json = json!(null);
    // 3D 지도 조각: 이번 실행 번호(뷰어가 새 실행이면 받은 조각을 비움), 조각 수, 보낸 점 수
    let mut run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let _ = std::fs::remove_dir_all(a.out.join("scene"));
    std::fs::create_dir_all(a.out.join("scene")).unwrap();
    // 분할 모델이 낼 수 있는 이름 전체: 검색이 "기억에 없는 이름"(고양이 → cat)을 가려내는 데 쓴다
    let _ = std::fs::write(a.out.join("vocab.json"), serde_json::to_string(&seg::COCO[..]).unwrap());
    let mut last_pose = [0.0; 3];
    let t0 = Instant::now();
    // 앞 단계(영상·깊이 읽기 + 분할)는 작업 스레드에서 한 장 앞서 돌린다 → NPU/GPU 분할과 CPU 지도 갱신이 겹친다.
    // 실시간 재생의 속도 맞춤(앞서면 기다리고, 밀리면 건너뛰기)도 그쪽에서 한다.
    let seq = std::sync::Arc::new(seq);
    // 장면 i 가 나올 시각 = pace0 + (t_i − t_첫) / R(R = 밀림 기준 배속). 본 스레드가 장면을 끝낸 시각과의 차이 = 밀림
    let (pace0, pace_t0, rate) = (Instant::now(), stream_first.as_ref().map_or(0.0, |f| f.t), a.stream_rate);
    let drop_late = !a.no_drop;
    // 밀림 기준 배속: --stream-rate 없이 받는 스트림(연결 프로그램에서 바로 오는 실시간 입력)은 찍은 시각 그대로(1 배속). --stream-rate 없는 --no-drop 재생은 밀림을 재지 않고 늘 분할(결과가 늘 같게)
    let lag_rate = if rate > 0.0 { rate } else { 1.0 };
    let lag_on = rate > 0.0 || (a.stream.is_some() && drop_late);
    let drop_n = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let drop_r = drop_n.clone();
    let seg_n = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seg_n_r = seg_n.clone();
    // 분할 단계별 시간 합, 본 스레드가 받아 간 장면 수(스트림 입력의 분할 단계가 씀)
    let seg_stats = std::sync::Arc::new(std::sync::Mutex::new(SegStages::default()));
    let seg_stats_r = seg_stats.clone();
    let taken = std::sync::Arc::new(Taken::default());
    let taken_r = taken.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(1);
    let worker = {
        let seq = seq.clone();
        let (every, conf, live, no_skip, seg_every, npu_par, sp_every) = (a.every, a.conf, a.live, a.no_skip, a.seg_every.max(1), a.npu_par, a.sp_every.max(1));
        let (fast_pose, depth_every, depth_lag) = (a.fast_pose, a.depth_every.max(1), a.depth_lag);
        let (seg_min, infer_min, seg_pipe, npu_jobs) = (a.seg_min_ms / 1e3, a.infer_min_ms / 1e3, a.seg_pipe, a.npu_jobs);
        let seg_skip = a.seg_skip;
        let mut dnet = make_depth(&a.depth);
        // --fast-pose 는 위치를 모두 직접 방식으로 구하니 SuperPoint 는 돌리지 않는다(NPU 작업 하나 줄임)
        let mut spnet = (a.vo && !a.dvo && !a.fast_pose).then(|| make_sp(&a.sp));
        let cam_path = a.cam.clone();
        let rec_dir = a.rec.clone();
        std::thread::spawn(move || {
            let mut seg = seg;
            out::thread_role(out::Role::Reader);
            if let (Some(mut rd), Some(first)) = (stream_rd, stream_first) {
                // 장면 스트림(bridge/): 위치·깊이는 받은 그대로, 분할만 여기서. --every N 장마다, 위치를 놓친 장면은 모델 없이 표시만 넘김.
                // 분할은 세 단계: 전처리(이 스레드) → 추론(모델 호출) → 후처리(마스크 해독). 단계마다 스레드를 두고 장면을 한 장씩 차례로 넘겨
                // (순서 그대로) 추론이 도는 동안 앞 장면의 후처리와 다음 장면의 읽기·전처리가 겹친다. --seg-pipe off 는 셋을 이 스레드에서 차례로.
                use std::sync::atomic::Ordering::Relaxed;
                let k0 = first.k;
                let stage2 = InferStage { seg, dnet, spnet, par: npu_par, infer_min, seg_min, conf, stats: seg_stats_r.clone() };
                enum Sink {
                    Seq(InferStage),
                    Pipe(std::sync::mpsc::SyncSender<SegJob>, std::thread::JoinHandle<InferStage>, std::thread::JoinHandle<()>),
                }
                let inf_ema = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)); // 추론 단계가 한 장에 걸린 시간(초, f64 비트)의 이동 평균
                let mut in_flight = 1u64; // 추론 단계에 함께 들어가 있을 수 있는 장면 수(--npu-jobs)
                let (buf_tx, buf_rx) = std::sync::mpsc::channel::<(seg::SegIn, seg::SegRaw)>();
                let mut sink = if seg_pipe {
                    let (to_inf, from_pre) = std::sync::mpsc::sync_channel::<SegJob>(0);
                    let (to_post, from_inf) = std::sync::mpsc::sync_channel::<SegJob>(0);
                    let (ema, mut stage2) = (inf_ema.clone(), stage2);
                    // --npu-jobs K(2 이상): 추론 스레드는 NPU 에 올리기만 하고 받기 스레드가 순서대로 결과를 받아 넘긴다 — K 장까지 함께 돌아 NPU 코어 여럿을 쓴다.
                    // 분할 말고 다른 모델을 같이 돌리거나 시간을 흉내 낼 때는 한 장씩.
                    let mut coll = (npu_jobs > 1 && stage2.dnet.is_none() && stage2.spnet.is_none() && infer_min <= 0.0 && seg_min <= 0.0).then(|| stage2.seg.collector()).flatten();
                    let jobs = if coll.is_some() { npu_jobs } else { 1 };
                    in_flight = jobs as u64;
                    let (to_coll, from_sub) = std::sync::mpsc::sync_channel::<(SegJob, Option<i32>, Instant)>(jobs.saturating_sub(2));
                    let h_inf = std::thread::Builder::new().name("seg-infer".into()).spawn(move || {
                        out::thread_role(out::Role::Infer);
                        let h_coll = coll.take().map(|mut wait| {
                            let (ema, to_post) = (ema.clone(), to_post.clone());
                            std::thread::Builder::new().name("seg-collect".into()).spawn(move || {
                                out::thread_role(out::Role::Infer);
                                while let Ok((mut j, id, c)) = from_sub.recv() {
                                    if let Some(id) = id {
                                        wait(id, &mut j.bufs.1);
                                        j.t[1] = c.elapsed().as_secs_f64();
                                        // 한 장이 NPU 를 차지한 몫: 올려서 받기까지 걸린 시간 / 함께 올린 수
                                        let (d, e) = (j.t[1] / jobs as f64, f64::from_bits(ema.load(Relaxed)));
                                        ema.store((if e == 0.0 { d } else { 0.8 * e + 0.2 * d }).to_bits(), Relaxed);
                                    }
                                    if to_post.send(j).is_err() {
                                        break;
                                    }
                                }
                            }).unwrap()
                        });
                        while let Ok(mut j) = from_pre.recv() {
                            let c = Instant::now();
                            if h_coll.is_some() {
                                let id = if j.do_seg && !j.sf.lost() { stage2.seg.submit(&j.bufs.0) } else { None };
                                if id.is_none() {
                                    stage2.run(&mut j);
                                }
                                if to_coll.send((j, id, c)).is_err() {
                                    break;
                                }
                                continue;
                            }
                            stage2.run(&mut j);
                            if j.do_seg {
                                let (d, e) = (c.elapsed().as_secs_f64(), f64::from_bits(ema.load(Relaxed)));
                                ema.store((if e == 0.0 { d } else { 0.8 * e + 0.2 * d }).to_bits(), Relaxed);
                            }
                            if to_post.send(j).is_err() {
                                break;
                            }
                        }
                        drop(to_coll);
                        if let Some(h) = h_coll {
                            h.join().expect("분할 받기 스레드");
                        }
                        stage2
                    }).unwrap();
                    let (tx, stats, taken) = (tx.clone(), seg_stats_r.clone(), taken_r.clone());
                    let h_post = std::thread::Builder::new().name("seg-post".into()).spawn(move || {
                        out::thread_role(out::Role::Reader);
                        while let Ok(j) = from_inf.recv() {
                            let (f, b) = seg_stage_post(j, conf, &stats);
                            let _ = buf_tx.send(b);
                            if tx.send(f).is_err() {
                                taken.dead.store(true, Relaxed);
                                taken.cv.notify_all();
                                break;
                            }
                        }
                    }).unwrap();
                    Sink::Pipe(to_inf, h_inf, h_post)
                } else {
                    Sink::Seq(stage2)
                };
                let (mut next, mut n, mut read_s, mut n_corr) = (Some(first), 0usize, 0.0f64, 0u32);
                // 건너뛴 장면 앞에 온 그래프 노드는 다음에 보내는 장면에 붙임(노드 위치를 하나도 잃지 않게)
                let mut graph: Vec<stream::Node> = vec![];
                let mut last_seg: Option<([f64; 3], [[f64; 3]; 3], usize)> = None;
                let mut n_seg = 0usize;
                let mut pool: Vec<(seg::SegIn, seg::SegRaw)> = vec![];
                // 겹쳐 돌릴 때 다음 장면을 고르는 때를 맞추는 데 쓰는 값: 넘긴 장면 수, 장면 하나를 넘기기까지 이 스레드가 쓴 시간(읽기 + 전처리, 속도 맞춤 잠 제외)
                let (mut picked, mut front_ema, mut c_front, mut slept) = (0u64, 0.0f64, Instant::now(), 0.0f64);
                let hold_on = rate > 0.0 || drop_late;
                let mut c_seg0: Option<Instant> = None;
                while let Some(mut sf) = next.take() {
                    // 지도 보정이 바뀐 장면(루프 클로저): 이후 장면은 새 보정으로 오고, 지난 장면은 그래프 블록의 노드 위치로 다시 쌓는다
                    n_corr += (sf.flags & stream::MAP_CORRECTED != 0) as u32;
                    graph.append(&mut sf.graph);
                    // 밀림: 찍은 시각(배속 R, 기본 1) 기준 DROP_LATE 초 넘게 늦은 장면은 버리고 다음(더 새) 장면으로. 그래프 노드는 위에서 이미 모아 다음 장면에 붙음
                    let late = pace0.elapsed().as_secs_f64() - (sf.t - pace_t0) / if rate > 0.0 { rate } else { 1.0 };
                    let skip = drop_late && n % every == 0 && n > 0 && late > DROP_LATE && !sf.lost();
                    if skip {
                        drop_r.fetch_add(1, Relaxed);
                    }
                    if n % every == 0 && !skip {
                        if rate > 0.0 {
                            let due = (sf.t - pace_t0) / rate - pace0.elapsed().as_secs_f64();
                            if due > 0.0 {
                                std::thread::sleep(std::time::Duration::from_secs_f64(due));
                                slept += due;
                            }
                        }
                        if sf.k != k0 {
                            eprintln!("스트림 장면 {}: 보정값이 바뀜 {:?} → {:?}(처음 값으로 계속)", sf.seq, k0, sf.k);
                        }
                        let c1 = Instant::now();
                        let lost = sf.lost();
                        if sf.flags & stream::NEW_MAP != 0 {
                            eprintln!("스트림 장면 {}: RTAB-Map 새 지도 {}", sf.seq, sf.map_id);
                        }
                        let depth = sf.depth_m();
                        // --seg-skip: 직전 분할 키프레임에서 조금만 움직였으면 화면이 거의 같으니 분할을 건너뜀(물체 기억은 그 장면을 안 본 것으로)
                        let want_seg = match (seg_skip, &last_seg) {
                            // 밀리지 않았으면(NPU 가 따라감) 늘 분할: 생략은 밀릴 때 지연을 줄이는 데만 쓴다(안 밀려도 건너뛰면 물체 관측만 줄어 물체를 잃음)
                            (Some((_, _, _, lmax)), Some(_)) if !lag_on || late <= lmax => true,
                            (Some((dm, ddeg, nmax, _)), Some((lp, lr, ns))) => {
                                let dp = ((sf.p[0] - lp[0]).powi(2) + (sf.p[1] - lp[1]).powi(2) + (sf.p[2] - lp[2]).powi(2)).sqrt();
                                let tr: f64 = (0..3).map(|i| (0..3).map(|j| lr[j][i] * sf.r[j][i]).sum::<f64>()).sum();
                                let ang = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
                                dp > dm || ang > ddeg || ns + 1 >= nmax
                            }
                            _ => true,
                        };
                        let do_seg = want_seg && !lost;
                        if !lost {
                            if do_seg {
                                last_seg = Some((sf.p, sf.r, 0));
                                n_seg += 1;
                            } else if let Some(l) = last_seg.as_mut() {
                                l.2 += 1;
                            }
                        }
                        // 단계 1 — 전처리
                        let mut bufs = pool.pop().unwrap_or_default();
                        let mut t = [0.0; 3];
                        if do_seg {
                            let c = Instant::now();
                            seg::seg_pre(&sf.rgb, sf.w, sf.h, &mut bufs.0);
                            out::slow(c);
                            t[0] = c.elapsed().as_secs_f64();
                            c_seg0.get_or_insert(c);
                        }
                        let mut job = SegJob { sf, depth, do_seg, load: read_s, graph: std::mem::take(&mut graph), bufs, dets: None, pd: None, feat: None, t_front: c1.elapsed().as_secs_f64(), t };
                        let front = c_front.elapsed().as_secs_f64() - slept;
                        let alive = match &mut sink {
                            Sink::Seq(s2) => {
                                s2.run(&mut job);
                                let (f, b) = seg_stage_post(job, conf, &seg_stats_r);
                                pool.push(b);
                                tx.send(f).is_ok()
                            }
                            Sink::Pipe(to_inf, ..) => {
                                // 추론 단계가 받을 때(앞 장면 추론이 끝날 때)까지 기다렸다 넘김
                                let ok = to_inf.send(job).is_ok();
                                let t_hand = Instant::now();
                                pool.extend(buf_rx.try_iter());
                                picked += 1;
                                // 다음 장면은 ① 앞앞 장면(여러 장을 함께 올리면 그 수만큼 더 앞)까지 본 스레드가 받아 간 뒤에 고른다: 본 스레드가 밀릴 때 끝난 장면이 차례 실행보다 더 쌓이지 않게
                                let mut g = taken_r.n.lock().unwrap();
                                while ok && *g + in_flight < picked && !taken_r.dead.load(Relaxed) {
                                    g = taken_r.cv.wait_timeout(g, std::time::Duration::from_millis(20)).unwrap().0;
                                }
                                drop(g);
                                // ② 지금 추론이 끝날 무렵 전처리가 끝나게 늦춰 고른다: 일찍 고르면 추론 한 번만큼 묵은 장면이 돼 밀림이 늘어남
                                if ok && hold_on && do_seg {
                                    front_ema = if front_ema == 0.0 { front } else { 0.8 * front_ema + 0.2 * front };
                                    let hold = f64::from_bits(inf_ema.load(Relaxed)) - 1.2 * front_ema - 0.002 - t_hand.elapsed().as_secs_f64();
                                    if hold > 0.0 {
                                        std::thread::sleep(std::time::Duration::from_secs_f64(hold));
                                    }
                                }
                                ok
                            }
                        };
                        if !alive {
                            break;
                        }
                        (c_front, slept) = (Instant::now(), 0.0);
                    }
                    n += 1;
                    let c0 = Instant::now();
                    next = rd.next();
                    read_s = c0.elapsed().as_secs_f64();
                }
                // 단계에 남은 장면을 모두 내보낸 뒤 끝
                let stage2 = match sink {
                    Sink::Seq(s) => s,
                    Sink::Pipe(to_inf, h_inf, h_post) => {
                        drop(to_inf);
                        let s = h_inf.join().expect("분할 추론 스레드");
                        h_post.join().expect("분할 후처리 스레드");
                        s
                    }
                };
                let st = {
                    let mut s = seg_stats_r.lock().unwrap();
                    s.wall = c_seg0.map_or(0.0, |c| c.elapsed().as_secs_f64());
                    *s
                };
                // 마지막 장면 뒤에 온 노드(끝난 뒤 최적화): 위치 없는 장면 하나로 넘김
                graph.append(&mut rd.take_graph());
                if !graph.is_empty() {
                    let f = Frame { t: 0.0, p: [0.0; 3], r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rgb: vec![], w: 0, h: 0, depth: vec![], dets: None, load: 0.0, seg: 0.0,
                                    scale: f64::NAN, feat: None, track_only: false, lost: true, held: false, anchor: 0, graph };
                    let _ = tx.send(f);
                }
                eprintln!("스트림 끝: 장면 {n} 개 받음, 지도 보정 바뀜 {n_corr} 번, 분할 {n_seg} 번");
                if st.n > 0 {
                    let m = 1e3 / st.n as f64;
                    eprintln!("분할 단계({}) 평균 ms: 전처리 {:.2}, 추론 {:.1}, 후처리 {:.2} · 분할 {:.2} 번/초", if seg_pipe { "겹침" } else { "차례" }, st.pre * m, st.infer * m, st.post * m,
                              st.n as f64 / st.wall.max(1e-9));
                }
                seg_n_r.store(n_seg as u32, Relaxed);
                return stage2.seg;
            }
            if let Some(cp) = cam_path {
                // 카메라(장치) 또는 녹화 폴더: 깊이·분할·특징점 모두 모델, 위치는 vo 가 만든다(첫 장면: 수평으로 앞을 봄, 높이 1.3 m)
                enum Src {
                    #[cfg(feature = "cam")]
                    Live(cam::Cam),
                    Rec(rec::Player),
                }
                let mut src = match rec::Player::open(&cp) {
                    Some(pl) => Src::Rec(pl),
                    #[cfg(feature = "cam")]
                    None => Src::Live(cam::Cam::open(&cp, 640, 480)),
                    #[cfg(not(feature = "cam"))]
                    None => panic!("--cam {cp}: 녹화 폴더(rgb.txt)가 아님 — 카메라 장치는 --features cam"),
                };
                let (w, h) = match &src {
                    #[cfg(feature = "cam")]
                    Src::Live(c) => (c.w, c.h),
                    Src::Rec(pl) => (pl.w, pl.h),
                };
                #[cfg_attr(not(feature = "cam"), allow(unused_mut, unused_variables))]
                let mut recorder = rec_dir.as_deref().map(rec::Recorder::new);
                let r0 = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
                let p0 = [0.0, 0.0, 1.3];
                let tc = Instant::now();
                let mut t_first = None::<f64>;
                let mut nf = 0usize;
                // --fast-pose: 모델(깊이·분할)은 따로 스레드에서 쉬는 대로 가장 새 장면에 돌리고, 모든 장면은 먼저 위치 추적용으로 본 루프에 보낸다
                // (모델을 기다리는 동안 들어온 카메라 장면도 위치에 쓴다 → 손카메라가 카메라 속도로 움직이고 큰 움직임에도 덜 놓친다)
                let mut seg_slot = Some(seg);
                let rec_sim = matches!(src, Src::Rec(_)) && live == 0.0;
                let mut nfp = 0usize;
                // 녹화를 최대 속도로 돌릴 땐(rec_sim) 모델을 작업 스레드에서 차례대로(TUM 흉내와 같게: 깊이가 다음 장면보다 먼저 도착)
                let model = (fast_pose && !rec_sim).then(|| {
                    let (jtx, jrx) = std::sync::mpsc::sync_channel::<(f64, Vec<u8>)>(0);
                    let tx2 = tx.clone();
                    let mut seg = seg_slot.take().unwrap();
                    let mut dnet = std::mem::take(&mut dnet);
                    let mut spnet = std::mem::take(&mut spnet);
                    let hd = std::thread::spawn(move || {
                        let mut nj = 0usize;
                        while let Ok((t, rgb)) = jrx.recv() {
                            let c1 = Instant::now();
                            nj += 1;
                            let (pd, dets, feat) = run_models(&mut dnet, &mut seg, &mut spnet, &rgb, w, h, conf, (nj - 1) % seg_every == 0, npu_par, (nj - 1) % sp_every == 0);
                            let depth = pd.expect("--cam needs --depth <.dxnn|.onnx>");
                            let f = Frame { t, p: p0, r: r0, rgb, w, h, depth, dets, load: 0.0, seg: c1.elapsed().as_secs_f64(), scale: f64::NAN, feat, track_only: false, lost: false, held: false, anchor: 0, graph: vec![] };
                            if tx2.send(f).is_err() {
                                break;
                            }
                        }
                        seg
                    });
                    (jtx, hd)
                });
                loop {
                    let c0 = Instant::now();
                    let (t, rgb) = match &mut src {
                        #[cfg(feature = "cam")]
                        Src::Live(c) => {
                            let (rgb, jpg) = c.grab();
                            let t = tc.elapsed().as_secs_f64();
                            if let Some(r) = recorder.as_mut() {
                                r.add(t, &jpg);
                            }
                            (t, rgb)
                        }
                        Src::Rec(pl) => {
                            let Some((t, rgb)) = pl.next() else { break };
                            // 재생 속도: --live 면 찍은 시각에 맞춤(밀리면 건너뜀), 아니면 최대 속도
                            if live > 0.0 {
                                let t0 = *t_first.get_or_insert(t);
                                let ahead = (t - t0) / live - tc.elapsed().as_secs_f64();
                                if !no_skip && ahead < -0.1 {
                                    continue;
                                }
                                if ahead > 0.0 {
                                    std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
                                }
                            }
                            (t, rgb)
                        }
                    };
                    let c1 = Instant::now();
                    if let Some((jtx, _)) = &model {
                        // 위치만 추적하는 장면으로 먼저 보내고(같은 장면의 깊이 결과보다 앞서 도착), 모델이 쉬고 있으면 이 장면을 맡긴다
                        let f = Frame { t, p: p0, r: r0, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: (c1 - c0).as_secs_f64(), seg: 0.0, scale: f64::NAN, feat: None, track_only: true, lost: false, held: false, anchor: 0, graph: vec![] };
                        if tx.send(f).is_err() {
                            break;
                        }
                        let _ = jtx.try_send((t, rgb));
                        continue;
                    }
                    if fast_pose {
                        // 녹화 최대 속도: 위치만 추적 장면을 먼저, depth_every 장마다 아래에서 모델까지
                        let f = Frame { t, p: p0, r: r0, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: (c1 - c0).as_secs_f64(), seg: 0.0, scale: f64::NAN, feat: None, track_only: true, lost: false, held: false, anchor: 0, graph: vec![] };
                        if tx.send(f).is_err() {
                            break;
                        }
                        nfp += 1;
                        if (nfp - 1) % depth_every != 0 {
                            continue;
                        }
                    }
                    // 모델 셋은 차례대로: 한 프로세스에서 NPU 에 동시에 던지면 DX-M1 DMA 쓰기 오류로 PCIe 링크가 내려갔다(2026-11-04)
                    // 물체 분할은 N 장마다(위치에는 깊이·특징점만 필요 → 위치를 더 자주 갱신)
                    nf += 1;
                    let do_seg = (nf - 1) % seg_every == 0;
                    let (pd, dets, feat) = run_models(&mut dnet, seg_slot.as_mut().unwrap(), &mut spnet, &rgb, w, h, conf, do_seg, npu_par, (nf - 1) % sp_every == 0);
                    let depth = pd.expect("--cam needs --depth <.dxnn|.onnx>");
                    let c2 = Instant::now();
                    let f = Frame { t, p: p0, r: r0, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale: f64::NAN, feat, track_only: false, lost: false, held: false, anchor: 0, graph: vec![] };
                    if tx.send(f).is_err() {
                        break;
                    }
                }
                if let Some((jtx, hd)) = model {
                    drop(jtx);
                    return hd.join().unwrap();
                }
                return seg_slot.unwrap();
            }
            let mut lagged: std::collections::VecDeque<(usize, Frame)> = Default::default();
            let mut n_to = 0usize;
            for (fi, (t, rel)) in seq.rgb.iter().enumerate() {
                if fi % every != 0 {
                    continue;
                }
                let t = *t;
                let Some((p, r)) = seq.pose(t) else { continue };
                if live > 0.0 {
                    match t_start {
                        None => {
                            t_start = Some(t);
                            w_start = Instant::now();
                        }
                        Some(ts) => {
                            let ahead = (t - ts) / live - w_start.elapsed().as_secs_f64();
                            if !no_skip && ahead < -0.1 {
                                continue; // 처리가 밀렸으면 건너뛰어 실제 시간에 맞춤
                            }
                            if ahead > 0.0 {
                                std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
                            }
                        }
                    }
                }
                let c0 = Instant::now();
                let img = image::open(seq.dir.join(rel)).expect("rgb").into_rgb8();
                let (w, h) = (img.width() as usize, img.height() as usize);
                let rgb = img.into_raw();
                if fast_pose {
                    // 실시간과 같게: 모든 장면은 먼저 위치만 추적, 깊이·분할은 depth_every 장마다
                    let f = Frame { t, p, r, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: 0.0, seg: 0.0, scale: f64::NAN, feat: None, track_only: true, lost: false, held: false, anchor: 0, graph: vec![] };
                    if tx.send(f).is_err() {
                        break;
                    }
                    // --depth-lag: 깊이 장면을 위치만 추적 장면 N 장 뒤에 내보냄(보드처럼 깊이가 늦게 도착)
                    n_to += 1;
                    while lagged.front().is_some_and(|(due, _)| *due <= n_to) {
                        if tx.send(lagged.pop_front().unwrap().1).is_err() {
                            break;
                        }
                    }
                    if fi / every % depth_every != 0 {
                        continue;
                    }
                }
                let Some((dw, dh, kin)) = seq.kinect_depth(t) else { continue };
                assert_eq!((dw as usize, dh as usize), (w, h));
                let c1 = Instant::now();
                // 모델 셋은 차례대로(보드 NPU 는 동시에 던지면 불안정했다)
                let (pd, dets, feat) = run_models(&mut dnet, &mut seg, &mut spnet, &rgb, w, h, conf, true, npu_par, fi / every % sp_every == 0);
                // 딥러닝 깊이 / Kinect 중앙값(비교용)
                let (depth, scale) = match pd {
                    Some(d) => {
                        let mut r: Vec<f32> = d.iter().zip(&kin).filter(|(_, k)| **k > 0.3 && **k < 5.0).step_by(37).map(|(p, k)| p / k).collect();
                        r.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        let sc = if r.is_empty() { f64::NAN } else { r[r.len() / 2] as f64 };
                        (d, sc)
                    }
                    None => (kin, f64::NAN),
                };
                let c2 = Instant::now();
                let f = Frame { t, p, r, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale, feat, track_only: false, lost: false, held: false, anchor: 0, graph: vec![] };
                if fast_pose && depth_lag > 0 {
                    lagged.push_back((n_to + depth_lag, f));
                    continue;
                }
                if tx.send(f).is_err() {
                    break;
                }
            }
            for (_, f) in lagged {
                if tx.send(f).is_err() {
                    break;
                }
            }
            seg
        })
    };
    let _ = (&mut t_start, &mut w_start);
    let mut scales: Vec<f64> = vec![];
    let mut views: std::collections::VecDeque<scale::View> = Default::default();
    let mut s_hist: Vec<f64> = vec![];
    // 위치 추정 기준 장면: 특징점마다 세계 3D 위치(랜드마크)를 들고, 이후 장면 관측으로 평균해 깊이 잡음을 줄인다.
    // 기준에서 멀어지거나 겹침이 줄면 이번 장면이 새 기준이 되고, 검증된 랜드마크는 그대로 넘겨받는다.
    struct VoRef {
        f: vo::Feat,
        lm: Vec<Option<(V3, f64)>>, // (세계 위치, 관측 수)
        r: M3,
        p: V3,
    }
    /// 기준 장면 rf 와 이번 장면 f 를 맞춰 위치를 푼다: (위치, 짝, 거부됐나)
    #[allow(clippy::too_many_arguments)]
    fn vo_try(rf: &VoRef, f: &vo::Feat, depth: &[f32], w: usize, k: [f64; 4], pred: Option<(M3, V3)>, fallback: (M3, V3), dl: bool,
              vo_pose: Option<(M3, V3)>, dt: f64) -> (Option<(M3, V3, usize)>, Vec<(usize, usize)>, bool) {
        let [fx, fy, cx, cy] = k;
        let ms = vo::match_mutual(&rf.f, f, 0.7);
        let (mut world, mut cam, mut uv) = (vec![], vec![], vec![]);
        for &(i, j) in &ms {
            let Some((xw, _)) = rf.lm[i] else { continue };
            let (u1, v1) = f.kp[j];
            let z1 = depth[v1 as usize * w + u1 as usize] as f64;
            cam.push((z1 > 0.2 && z1 < 6.0).then(|| [(u1 as f64 - cx) / fx * z1, (v1 as f64 - cy) / fy * z1, z1]));
            world.push(xw);
            uv.push((u1 as f64, v1 as f64));
        }
        // 딥러닝 깊이는 장면마다 배율이 흔들린다: 위치를 풀기 전에 예측 위치로 본 랜드마크 거리에 맞춰 이번 깊이 배율을 먼저 맞춘다
        if let (true, Some((rp, pp))) = (dl, &pred) {
            // (이번 깊이 z, 예측 위치로 본 랜드마크 거리 zc) 짝
            let zz: Vec<(f64, f64)> = world
                .iter()
                .zip(&cam)
                .filter_map(|(xw, c)| {
                    let c = (*c)?;
                    let d = sub(xw, pp);
                    let zc = rp[0][2] * d[0] + rp[1][2] * d[1] + rp[2][2] * d[2];
                    (zc > 0.2).then_some((c[2], zc))
                })
                .collect();
            if let Some(s0) = depth_scale(&zz) {
                for c in cam.iter_mut().flatten() {
                    *c = c.map(|x| x * s0);
                }
            }
        }
        let mut est = vo::estimate(&world, &cam, &uv, k, pred.unwrap_or(fallback));
        // 손으로 들고는 불가능한 움직임(2 m/s, 180°/s 초과)이면 틀린 추정
        let mut rej = false;
        if let (Some((r2, p2, _)), Some((r0, p0))) = (&est, &vo_pose) {
            let d = dt.max(0.03);
            if norm(&sub(p2, p0)) > 0.05 + 2.0 * d || rot_angle(r0, r2) > (10.0 + 180.0 * d).to_radians() {
                est = None;
                rej = true;
            }
        }
        (est, ms, rej)
    }
    let mut vo_ref: Option<VoRef> = None;    let mut vo_pose: Option<(M3, V3)> = None;
    let mut vo_last_t: Option<f64> = None;
    let mut vo_vel: V3 = [0.0; 3]; // 등속 예측용 속도(m/s)
    let mut vo_ms: Vec<(usize, usize)> = vec![];
    let mut vo_ok = false;
    let mut ate: Vec<(V3, V3)> = vec![];
    let mut est_traj: Vec<(f64, V3, M3)> = vec![]; // 추정 위치(시각, 위치, 자세) → poses_est.txt(TUM 형식)
    let mut vo_fail = 0u32;
    let mut vo_reject = 0u32;
    let mut vo_nref = 0u32;
    let mut vo_last: Option<VoRef> = None; // 바로 앞 장면(기준과 안 맞을 때 대신)
    let mut vo_fails_run = 0u32; // 연속 실패 수
    // 수평 합의(웹캠·녹화): 처음 장면들의 수평면 법선(세계 좌표)을 모아 다수결로 '위'를 정하면 세계를 한 번 바로 세운다
    let mut grav_obs: Vec<V3> = vec![];
    let mut dvo_prev: Option<(dvo::Pyr, Vec<f32>)> = None;
    let mut dvo_ref_pose: Option<(M3, V3)> = None; // dvo_prev(마지막 깊이 장면)의 자세
    // 위치만 추적한 장면들의 (시각, 자세, 성공): 깊이가 늦게 도착한 장면의 자세를 여기서 찾는다
    let mut fast_hist: std::collections::VecDeque<(f64, M3, V3, bool)> = Default::default();
    let mut last_z0v = 0.0; // 마지막 깊이 장면의 바닥 기준 높이(위치만 추적한 장면의 pose.json 용)
    let mut n_fast = 0u32;
    // --fast-pose 기준 장면은 8 cm·5° 넘게 움직였을 때만 바꿈(깊이 전파도 같은 문턱): 보드 흉내 사무실 0.80 → 0.67 m, 책상 0.33 → 0.31 m,
    // 12 cm·8° 는 빠른 움직임에서 실패가 생김(책상 0 → 10–28 회)
    let (kf_t, kf_r) = (0.08f64, 5.0f64);
    let mut dvo_ref_t = f64::MIN; // 기준 장면 시각
    let prop = true; // 깊이 전파(위치만 잡은 장면을 옮긴 깊이와 함께 새 기준으로): 깊이가 2–3 장 늦게 올 때 책상 0.70 → 0.41 m, 실패 36 → 0
    let mut fast_rel: (M3, V3) = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]); // 위치만 추적한 직전 두 장면 사이 움직임(앞 → 이번 카메라)
    let mut t_fast = 0.0f64; // 위치만 추적한 장면에 쓴 시간(본 스레드)
    let mut dvo_rel: (M3, V3) = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]); // 직전 장면 사이 움직임
    // 직접 방식: 6 단계 피라미드(20×15 까지, 큰 움직임), 시작값은 움직임 없음 → 안 되면 직전 움직임 → TUM 책상 Kinect 0.333 → 0.11 m, 실패 13 → 0
    let (dvo_dual, dvo_levels) = (true, 6usize);
    let dvo_minlv = if a.depth == "kinect" { 0 } else { 1 }; // 딥러닝 깊이는 640×480 단계 생략(정확도 같고 40 % 빠름)
    let mut vo_dvo = 0u32; // 직접 방식으로 메운 장면 수
    // 스트림: RTAB-Map 지도의 z 는 첫 장면 카메라 몸체의 위(IMU 없음)라 처음 든 각도만큼 기울 수 있다 → 바닥 합의로 한 번 바로 세운다(--no-level 이면 그대로)
    let mut leveled = if a.stream.is_some() { a.no_level } else { a.cam.is_none() || !a.vo };
    let mut stream_fix: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // 스트림 위치에 곱할 수평 맞춤 회전
    let mut stream_held = 0u32; // 맡아 둘 장면으로 받은 수
    let (mut stream_lost, mut map_traj) = (0u32, Vec::<(f64, V3, M3)>::new()); // map_traj: 지도에 넣은 장면의 카메라 위치(지도 좌표) → poses_map.txt
    // 루프 클로저 뒤 다시 쌓기: 지도에 넣은 장면을 기준 노드와 함께 보관, 노드가 움직이면 작업 스레드가 고친 위치로 지도·물체를 처음부터 다시 쌓고
    // 끝나면 본 스레드가 갈아 끼운다(그동안 실시간 장면은 계속 들어옴)
    let mut kfs = kfstore::KfStore::new(a.kf_store_mb);
    let mut last_rebuild = Instant::now();
    let mut n_rebuild = 0u32;
    let mut prune_objs = false;
    let mut view_due = false; // 다음 장면에서 view.json 쓰기
    let mut n_behind = 0u32; // 밀려서 가볍게 처리한 장면 수
    let mut lag_rb = (0u32, 0f32); // 다시 쌓기가 도는 동안 끝낸 장면 수·그중 밀림 최대
    let (mut write_due, mut scene_due) = (false, false); // 미룬 정기 기록·점 지도 쓰기
    let a_rb = std::sync::Arc::new(a.clone());
    let mut rb_job: Option<std::thread::JoinHandle<Rebuilt>> = None;
    while let Ok(Frame { t, mut p, mut r, rgb, w, h, mut depth, dets, load, seg: seg_s, scale, feat, track_only, lost, held, anchor, graph }) = rx.recv() {
        *taken.n.lock().unwrap() += 1;
        taken.cv.notify_all();
        let c_top = Instant::now();
        clock.part = [0.0; NPART];
        let view_late = std::mem::take(&mut view_due); // 앞 장면에서 미룬 view.json
        // 밀림: 장면이 나올 시각보다 LAG_DEGRADE 초 넘게 늦으면 가볍게 — 점 격자 일부만, 새 다시 쌓기 미룸, view.json 15 장마다
        let behind = lag_on && pace0.elapsed().as_secs_f64() - (t - pace_t0) / lag_rate > LAG_DEGRADE;
        n_behind += behind as u32;
        let (write_late, scene_late) = (std::mem::take(&mut write_due), std::mem::take(&mut scene_due));
        let mut busy = false; // 이 장면에서 갈아 끼우기 일을 했으면 정기 기록은 다음 장면으로
        let c_g = Instant::now();
        if !graph.is_empty() {
            kfs.apply_graph(&graph);
        }
        clock.add(P::Graph, c_g);
        let c_in = Instant::now();
        if let Some(job) = rb_job.take_if(|j| j.is_finished()) {
            let mut rb = job.join().expect("다시 쌓기 스레드");
            let newer = newer_than(&kfs, &stream_fix, z0.unwrap_or(0.0), rb.last_seq);
            if newer.len() > CATCH_UP_MAX {
                // 쌓는 동안 들어온 장면이 많으면 작업 스레드가 먼저 이어 넣고 다시 옴(그사이 들어온 몇 장만 본 스레드가)
                let (ar, k) = (a_rb.clone(), seq.k);
                rb_job = Some(std::thread::spawn(move || {
                    out::thread_role(out::Role::Rebuild);
                    let c = Instant::now();
                    catch_up(&ar, k, &mut rb, newer);
                    out::slow(c);
                    rb
                }));
            } else {
            let i1 = Instant::now();
            install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
            let i2 = Instant::now();
            (n_rebuild, last_merge) = (n_rebuild + 1, nkf);
            // view.json 은 다음 장면(한 장면에 몰리지 않게)
            scene_json = fresh_outputs(&mut run_id, a.scene_vox, pw.as_ref());
            view_due = true;
            let i3 = Instant::now();
            mem.prune(t);
            mem.merge_pass();
            busy = true;
            if std::env::var_os("MEMMAP_PARTS").is_some() {
                eprintln!("갈아 끼우기 나눔: 지도·물체 {:.0} · 출력 {:.0} · 병합 {:.0} ms", (i2 - i1).as_secs_f64() * 1e3, (i3 - i2).as_secs_f64() * 1e3, i3.elapsed().as_secs_f64() * 1e3);
            }
            prune_objs = true; // 옛 id 파일은 다음 기록 때 지움(뷰어가 지금 view.json 을 받기 전 옛 것을 읽는 동안은 남겨 둠)
            }
        }
        clock.add(P::Install, c_in);
        let c_rs = Instant::now();
        if rb_job.is_none() && !behind && kfs.dirty && a.rebuild_every > 0.0 && last_rebuild.elapsed().as_secs_f64() >= a.rebuild_every {
            let snap = snapshot(&kfs, &stream_fix, z0.unwrap_or(0.0));
            let (ar, k, next, moved) = (a_rb.clone(), seq.k, mem.next + 100_000, kfs.moved); // 물체 id: 쌓는 동안 실시간에서 생긴 id 와 겹치지 않게
            kfs.clear_dirty();
            rb_job = Some(std::thread::spawn(move || { out::thread_role(out::Role::Rebuild); let c_rb = Instant::now(); let r = rebuild_maps(&ar, k, &snap, next, moved); out::slow(c_rb); r }));
            last_rebuild = Instant::now();
        }
        clock.add(P::RbStart, c_rs);
        let raw = (r, p); // 받은 위치(수평 맞춤 전): 다시 쌓기용 보관
        if lost {
            stream_lost += (w > 0) as u32; // w = 0: 끝에 남은 그래프 노드만 담은 장면
            if w == 0 {
                continue;
            }
            if a.live > 0.0 {
                // 뷰어: 마지막 위치 그대로, 놓침 표시(손카메라 빨간색)
                out::write_atomic(&a.out.join("pose.json"), json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": false}).to_string().as_bytes());
            }
            continue;
        }
        if held {
            // 맡아 둘 장면: 믿는 지도와 이어지지 않은 새 지도의 위치라 지금 지도에는 넣지 않고 보관만 한다.
            // 기준 노드가 믿는 노드로 바뀌면(루프 클로저로 이어짐) 다시 쌓기가 고친 위치로 넣는다
            stream_held += 1;
            if leveled && a.rebuild_every >= 0.0 && kfs.holds(anchor) {
                let keep = kfs.push(t, anchor, raw.0, raw.1, cam_height(&hist_h), w, h, &rgb, &depth, &dets, true);
                match (pw.as_mut(), keep) {
                    (Some(pw), Some(kf)) => pw.hold(&depth, &rgb, w, h, kf),
                    (None, Some(kf)) => { let _ = kf.pts.set(None); }
                    _ => {}
                }
            }
            if a.live > 0.0 {
                out::write_atomic(&a.out.join("pose.json"), json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": false}).to_string().as_bytes());
            }
            continue;
        }
        if a.stream.is_some() {
            r = mm(&stream_fix, &r);
            p = mv(&stream_fix, &p);
        }
        let c_vo = Instant::now();
        let (p_gt, _r_gt) = (p, r);
        // 위치만 추적하는 장면(--fast-pose): 깊이 모델을 기다리지 않고 마지막 깊이 장면(기준)에 맞춰 위치만 갱신 → 손카메라가 카메라 속도로 움직이고,
        // 깊이 장면 사이 움직임이 잘게 나뉘어 직접 방식의 시작값이 좋아진다
        if track_only {
            if let (Some((rp, rd)), Some((rk, pk)), Some((r0, p0))) = (&dvo_prev, dvo_ref_pose, vo_pose) {
                let dt = vo_last_t.map(|t0| (t - t0).clamp(0.0, 1.0)).unwrap_or(0.0);
                let pyr = dvo::pyramid(&rgb, w, h, dvo_levels, dvo_minlv);
                let rel = |rc: &M3, pc: &V3| (mm(&tr(rc), &rk), mv(&tr(rc), &sub(&pk, pc)));
                let to_pose = |b: &(M3, V3, usize, f64, f64)| {
                    let r1 = mm(&rk, &tr(&b.0));
                    (r1, sub(&pk, &mv(&r1, &b.1)))
                };
                let ppred = add(&p0, &vo_vel.map(|x| x * dt));
                // 손으로 들고는 불가능한 움직임(2 m/s, 180°/s 초과)이면 버림
                let dd = dt.max(0.03);
                let sane = |q: &(M3, V3)| norm(&sub(&q.1, &p0)) <= 0.05 + 2.0 * dd && rot_angle(&r0, &q.0) <= (10.0 + 180.0 * dd).to_radians();
                // 시작값: 기준 장면 그대로(움직임 없음)에서 먼저. 직전 자세에서 시작하면 사이 장면 하나가 틀릴 때 그 오차가 이어져 쌓였다.
                // 실패하거나 불가능한 움직임이면 직전 자세에서 한 번 더(움직임이 커서 기준 그대로에서는 못 찾을 때)
                let i3 = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
                let b1 = dvo::track(rp, rd, &pyr, &[], seq.k, i3, dvo_minlv);
                let b2 = if b1.as_ref().is_none_or(|b| !sane(&to_pose(b))) { dvo::track(rp, rd, &pyr, &[], seq.k, rel(&r0, &p0), dvo_minlv) } else { None };
                // 둘 다 안 되면 직전 움직임을 그대로 이어 간 자세(등속)에서: 장면 간격이 길어(녹화 0.1–0.2 초) 움직임이 클 때
                let ok12 = [&b1, &b2].into_iter().flatten().any(|b| sane(&to_pose(b)));
                let b3 = if ok12 {
                    None
                } else {
                    let rp1 = mm(&r0, &tr(&fast_rel.0));
                    let pp1 = sub(&p0, &mv(&rp1, &fast_rel.1));
                    dvo::track(rp, rd, &pyr, &[], seq.k, rel(&rp1, &pp1), dvo_minlv)
                };
                let best = [b1, b2, b3].into_iter().flatten().map(|b| (to_pose(&b), b.3)).filter(|(q, _)| sane(q)).min_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
                let ok = best.is_some();
                match best {
                    Some(((r1, p1), _)) => {
                        if dt > 0.0 {
                            let v = sub(&p1, &p0).map(|x| x / dt);
                            let s = 2.0 / norm(&v).max(2.0);
                            vo_vel = [0, 1, 2].map(|k| 0.5 * vo_vel[k] + 0.5 * v[k] * s);
                        }
                        fast_rel = (mm(&tr(&r1), &r0), mv(&tr(&r1), &sub(&p0, &p1)));
                        vo_pose = Some((r1, p1));
                    }
                    None => {
                        vo_fail += 1;
                        vo_pose = Some((r0, ppred));
                        vo_vel = vo_vel.map(|x| x * 0.5);
                    }
                }
                let (r1, p1) = vo_pose.unwrap();
                // 깊이 전파: 기준에서 kf_t·kf_r 넘게 움직였으면 이 장면을 새 기준으로(기준 깊이를 이 장면으로 옮겨 붙임).
                // 깊이 모델이 늦게 와도 기준이 오래되지 않아 큰 움직임을 덜 놓친다
                if ok && prop && (norm(&sub(&p1, &pk)) > kf_t || rot_angle(&rk, &r1) > kf_r.to_radians()) {
                    let (rr, tt) = (mm(&tr(&r1), &rk), mv(&tr(&r1), &sub(&pk, &p1)));
                    let wd = dvo::warp_depth(rd, w, h, seq.k, &rr, &tt);
                    dvo_prev = Some((pyr, wd));
                    dvo_ref_pose = Some((r1, p1));
                    dvo_ref_t = t;
                }
                fast_hist.push_back((t, r1, p1, ok));
                if fast_hist.len() > 64 {
                    fast_hist.pop_front();
                }
                vo_last_t = Some(t);
                ate.push((p1, p_gt));
                est_traj.push((t, p1, r1));
                n_fast += 1;
                if a.live > 0.0 {
                    let pw = [p1[0], p1[1], p1[2] - last_z0v];
                    out::write_atomic(&a.out.join("pose.json"), json!({"t": t, "cam": {"pos": pw, "R": r1}, "pose": [pw[0], pw[1], r1[1][2].atan2(r1[0][2])], "ok": ok}).to_string().as_bytes());
                }
            }
            clock.vo += c_vo.elapsed().as_secs_f64();
            t_fast += c_vo.elapsed().as_secs_f64();
            continue;
        }
        // 깊이가 늦게 도착한 장면: 위치는 이미 추적해 둔 것을 쓴다
        let hist_pose = fast_hist.iter().find(|h| h.0 == t).map(|h| (h.1, h.2, h.3));
        // 카메라만으로 포즈(--pose vo): 이전 키프레임과 SuperPoint 짝 → 이전 장면 깊이로 3D → 재투영 최소화. 첫 장면만 정답 포즈로 시작.
        if let (true, Some((hr, hp, hok))) = (a.vo, hist_pose) {
            (r, p, vo_ok) = (hr, hp, hok);
            // 딥러닝 깊이 배율: 기준 장면 깊이를 이번 장면으로 옮긴 거리에 맞춘다
            if let (true, true, Some((_, rd)), Some((rk, pk))) = (hok, a.depth != "kinect", &dvo_prev, dvo_ref_pose) {
                let (rr, tt) = (mm(&tr(&r), &rk), mv(&tr(&r), &sub(&pk, &p)));
                let sc = dvo::depth_ratio(rd, &depth, w, h, seq.k, &rr, &tt) as f32;
                for d in depth.iter_mut() {
                    *d *= sc;
                }
            }
            // 기준 바꾸기: 기준에서 충분히 움직였거나(kf_t m·kf_r °) 위치를 놓쳤으면 이 장면을 새 기준으로. 천천히 움직이면 같은 기준을 오래 써 오차가 덜 쌓인다
            let moved = dvo_ref_pose.is_none_or(|(rk, pk)| norm(&sub(&p, &pk)) > kf_t || rot_angle(&rk, &r) > kf_r.to_radians());
            if prop && hok && t <= dvo_ref_t {
                // 기준이 이미 더 새 장면(깊이 전파): 이번 진짜 깊이를 그 기준으로 옮겨 기준 깊이를 바꿈
                if let (Some((rp, _)), Some((rk, pk))) = (dvo_prev.take(), dvo_ref_pose) {
                    let (rr, tt) = (mm(&tr(&rk), &r), mv(&tr(&rk), &sub(&p, &pk)));
                    dvo_prev = Some((rp, dvo::warp_depth(&depth, w, h, seq.k, &rr, &tt)));
                }
            } else if !hok || moved || dvo_prev.is_none() {
                dvo_ref_t = t;
                dvo_prev = Some((dvo::pyramid(&rgb, w, h, dvo_levels, dvo_minlv), depth.clone()));
                dvo_ref_pose = Some((r, p));
            }
        } else if a.vo {
            // --fast-pose 에서 위치 기록 없이 늦게 온 깊이 장면(시작 직후 등): 기준 장면 자세에서 이어 풀고, 최신 추적 상태(자세·속도·시각)는 되돌려 둔다
            let rewind = a.fast_pose && vo_last_t.is_some_and(|lt| lt > t);
            let saved = (vo_pose, vo_vel, vo_last_t, dvo_rel);
            if rewind {
                vo_pose = dvo_ref_pose.or(vo_pose);
            }
            let prev_pose = vo_pose;
            // 등속 예측: 직전 위치 + 속도 × 시간(IMU 없이 쓰는 움직임 사전 지식)
            let dt = vo_last_t.map(|t0| (t - t0).clamp(0.0, 1.0)).unwrap_or(0.0);
            let pred = vo_pose.map(|(r0, p0)| (r0, add(&p0, &vo_vel.map(|x| x * dt))));
            let mut est = None;
            vo_ms.clear();
            vo_ok = false;
            if let (Some(f), Some(rf)) = (&feat, &vo_ref) {
                let dl = a.depth != "kinect";
                let (e, ms, rej) = vo_try(rf, f, &depth, w, seq.k, pred, (r, p), dl, vo_pose, dt);
                vo_reject += rej as u32;
                (est, vo_ms) = (e, ms);
                // 기준과 안 맞으면 바로 앞 장면(움직임이 작아 짝이 많다)과 한 번 더, 맞으면 그 장면을 새 기준으로
                if est.is_none() {
                    if let Some(lf) = &vo_last {
                        let (e, ms, rej) = vo_try(lf, f, &depth, w, seq.k, pred, (r, p), dl, vo_pose, dt);
                        vo_reject += rej as u32;
                        if e.is_some() {
                            (est, vo_ms) = (e, ms);
                            vo_ref = vo_last.take();
                            vo_nref += 1;
                        }
                    }
                }
            }
            // RGB-D 직접 방식(DVO): --pose dvo 면 이것만, 아니면 특징점이 실패한 장면을 메운다(무늬 적은 장면·흐린 장면)
            let pyr = dvo::pyramid(&rgb, w, h, dvo_levels, dvo_minlv);
            let mut dvo_used = false;
            if est.is_none() {
                if let (Some((pp, pd)), Some((r0, p0))) = (&dvo_prev, vo_pose) {
                    // 시작값: '움직임 없음'으로 먼저, 밝기 오차가 크거나 실패하면 '직전 움직임(등속)'으로도 풀어 더 나은 쪽
                    let id3 = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
                    // 손으로 들고는 불가능한 움직임(2 m/s, 180°/s 초과)이면 버림
                    let dd = dt.max(0.03);
                    let i3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                    let sane = |b: &(M3, V3, usize, f64, f64)| norm(&b.1) <= 0.05 + 2.0 * dd && rot_angle(&b.0, &i3) <= (10.0 + 180.0 * dd).to_radians();
                    let b1 = dvo::track(pp, pd, &pyr, &depth, seq.k, id3, dvo_minlv);
                    // 실패했거나 불가능한 움직임으로 나왔을 때만 '직전 움직임'으로도 풀어 본다(밝기 오차로 둘을 견주면 사무실 15 장 간격 0.91 → 1.16 m 로 나빴다)
                    let b2 = if dvo_dual && b1.as_ref().is_none_or(|b| !sane(b)) { dvo::track(pp, pd, &pyr, &depth, seq.k, dvo_rel, dvo_minlv) } else { None };
                    let any = b1.is_some() || b2.is_some();
                    let best = [b1, b2].into_iter().flatten().filter(|b| sane(b)).min_by(|x, y| x.3.partial_cmp(&y.3).unwrap());
                    if let Some((rr, tt, n, _, sc)) = best {
                        let r1 = mm(&r0, &tr(&rr));
                        est = Some((r1, sub(&p0, &mv(&r1, &tt)), n));
                        dvo_used = true;
                        vo_dvo += 1;
                        if a.depth != "kinect" {
                            for d in depth.iter_mut() {
                                *d *= sc as f32;
                            }
                        }
                    } else if any {
                        vo_reject += 1;
                    }                }
            }            // 배율 이어 붙이기: 같은 랜드마크의 (평균 위치로 본 거리) / (이번 장면 예측 깊이) 중앙값으로 이번 장면 깊이를 맞춘다
            if let (Some((r2, p2, _)), true, Some(f), Some(rf), false) = (&est, a.depth != "kinect", &feat, &vo_ref, dvo_used) {
                let mut rs = vec![];
                for &(i, j) in &vo_ms {
                    let Some((xw, _)) = rf.lm[i] else { continue };
                    let (u1, v1) = f.kp[j];
                    let z1 = depth[v1 as usize * w + u1 as usize] as f64;
                    if z1 < 0.2 {
                        continue;
                    }
                    let d = sub(&xw, p2);
                    let zc = r2[0][2] * d[0] + r2[1][2] * d[1] + r2[2][2] * d[2];
                    if zc > 0.2 {
                        rs.push((z1, zc));
                    }
                }
                if let Some(s) = depth_scale(&rs) {
                    for d in depth.iter_mut() {
                        *d *= s as f32;
                    }
                }
            }
            match (est, vo_pose) {
                (Some((mut r2, mut p2, _)), prev) => {
                    // 거의 안 움직였으면(5 mm·0.3° 미만) 정지로 보고 위치를 그대로 둔다: 가만히 둔 카메라가 잡음으로 흘러가지 않게
                    if let Some((r0, p0)) = prev {
                        if norm(&sub(&p2, &p0)) < 0.005 && rot_angle(&r0, &r2) < 0.3f64.to_radians() {
                            (r2, p2) = (r0, p0);
                        }
                    }
                    if let (Some((_, p0)), true) = (prev, dt > 0.0) {
                        // 속도는 부드럽게(반반 섞기), 2 m/s 로 자름
                        let v = sub(&p2, &p0).map(|x| x / dt);
                        let s = 2.0 / norm(&v).max(2.0);
                        vo_vel = [0, 1, 2].map(|k| 0.5 * vo_vel[k] + 0.5 * v[k] * s);
                    }
                    vo_pose = Some((r2, p2));
                    vo_ok = true;
                }
                (None, None) => vo_pose = Some((r, p)),
                (None, Some(_)) => {
                    vo_fail += 1;
                    if a.dvo {
                        vo_vel = [0.0; 3];
                    }
                    if std::env::var_os("VO_DEBUG").is_some() {
                        eprintln!("vofail t {t:.2} feat {} ms {}", feat.as_ref().map_or(0, |f| f.kp.len()), vo_ms.len());
                    }
                    vo_pose = pred;
                    vo_vel = vo_vel.map(|x| x * 0.5);
                }
            }
            // 직접 방식용: 이번 장면을 다음 장면의 '앞 장면'으로(시작값은 '움직임 없음': 등속 시작값은 한 번 틀리면 계속 엉뚱하게 수렴했음)
            if let (Some((r0, p0)), Some((r1, p1)), true) = (prev_pose, vo_pose, vo_ok) {
                let r1t = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r1[j][i]));
                dvo_rel = (mm(&r1t, &r0), mv(&r1t, &sub(&p0, &p1)));
            }
            dvo_prev = Some((pyr, depth.clone()));
            dvo_ref_pose = vo_pose;
            dvo_ref_t = t;
            vo_last_t = Some(t);
            let (r2, p2) = vo_pose.unwrap();
            r = r2;
            p = p2;
            ate.push((p, p_gt));
            est_traj.push((t, p, r));
            if rewind {
                ate.pop();
                est_traj.pop();
                (vo_pose, vo_vel, vo_last_t, dvo_rel) = saved;
            }
        }        // 딥러닝 깊이 배율: 두 장면 삼각측량(포즈를 알 때)으로 실제 거리를 재서 맞춘다(scale.rs). 최근 5 개 중앙값으로 평활.
        let mut s_fix = 1.0;
        if a.depth != "kinect" && !a.vo {
            // 위치를 정답으로 받을 때만: 위치 추정(vo)과 함께 쓰면 서로의 배율을 끌어내려 무너진다
            let cur = scale::View { gray: scale::gray(&rgb), w, h, r, p };
            let prev = views.iter().rev().find(|v| norm(&sub(&v.p, &p)) >= 0.06).or(views.front());
            if let Some((sm, _n)) = prev.and_then(|pv| scale::metric_scale(&cur, pv, seq.k, &depth)) {
                s_hist.push(sm.clamp(0.3, 3.0));
            }
            if !s_hist.is_empty() {
                let mut v: Vec<f64> = s_hist.iter().rev().take(a.scale_win).copied().collect();
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                s_fix = v[v.len() / 2];
            }
            views.push_back(cur);
            if views.len() > 6 {
                views.pop_front();
            }
            for d in depth.iter_mut() {
                *d *= s_fix as f32;
            }
        }
        if scale.is_finite() {
            scales.push(scale * s_fix);
        }
        if a.vo {
            if let Some(f) = feat {
                let [fx, fy, cx, cy] = seq.k;
                let mut keep = false;
                let mut carry: Vec<(usize, V3, f64)> = vec![]; // 이번 장면 특징점 j 가 넘겨받을 검증된 랜드마크
                if let (true, Some(rf)) = (vo_ok, vo_ref.as_mut()) {
                    // 추적: 기준 랜드마크가 이번 위치로 재투영해 4 화소 안에 드는 수(기준 유지 판단).
                    // 평균: 그중 이번 깊이도 10 % 안에 맞는 것만 랜드마크 위치에 섞는다(딥러닝 깊이가 흔들려도 추적은 끊기지 않게).
                    let mut track = 0;
                    for &(i, j) in &vo_ms {
                        let (u1, v1) = f.kp[j];
                        let Some((x0, n)) = rf.lm[i].as_mut() else { continue };
                        let d = sub(x0, &p);
                        let zc = r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2];
                        if zc < 0.2 {
                            continue;
                        }
                        let uc = fx * (r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2]) / zc + cx;
                        let vc = fy * (r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2]) / zc + cy;
                        if (uc - u1 as f64).hypot(vc - v1 as f64) > 4.0 {
                            continue;
                        }
                        track += 1;
                        let z1 = depth[v1 as usize * w + u1 as usize] as f64;
                        if z1 < 0.2 || z1 > 6.0 || (zc - z1).abs() > 0.1 * zc || depth_edge(&depth, w, h, u1 as usize, v1 as usize) {
                            carry.push((j, *x0, *n));
                            continue;
                        }
                        let xw = add(&mv(&r, &[(u1 as f64 - cx) / fx * z1, (v1 as f64 - cy) / fy * z1, z1]), &p);
                        let k = 1.0 / (*n + 1.0);
                        for a in 0..3 {
                            x0[a] += (xw[a] - x0[a]) * k;
                        }
                        *n = (*n + 1.0).min(20.0);
                        carry.push((j, *x0, *n));
                    }
                    // 기준 유지: 추적되는 랜드마크가 충분하고 기준 장면에서 30 cm·20° 안
                    keep = track >= 80 && norm(&sub(&p, &rf.p)) < 0.3 && rot_angle(&rf.r, &r) < 20f64.to_radians();
                }
                // 실패하면 기준을 바로 버리지 않는다(흐린 한 장 때문에 예측 위치로 기준을 새로 잡으면 지도가 어긋남): 5 장 연속 실패면 그때 새로
                if !vo_ok && vo_ref.is_some() {
                    vo_fails_run += 1;
                    if vo_fails_run < 5 {
                        keep = true;
                    } else {
                        vo_fails_run = 0;
                    }
                } else {
                    vo_fails_run = 0;
                }
                // 이번 장면 깊이로 만든 랜드마크(새 기준 또는 '바로 앞 장면' 후보)
                let lm: Vec<Option<(V3, f64)>> = f
                    .kp
                    .iter()
                    .map(|&(u, v)| {
                        let z = depth[v as usize * w + u as usize] as f64;
                        (z > 0.2 && z < 6.0 && !depth_edge(&depth, w, h, u as usize, v as usize))
                            .then(|| (add(&mv(&r, &[(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z]), &p), 1.0))
                    })
                    .collect();
                if !keep {
                    // 새 기준 장면: 앞 기준에서 검증된 랜드마크는 평균값을 넘겨받는다
                    let mut lm2 = lm.clone();
                    for (j, x, n) in carry {
                        lm2[j] = Some((x, n));
                    }
                    vo_ref = Some(VoRef { f: f.clone(), lm: lm2, r, p });
                    vo_nref += 1;
                }
                vo_last = vo_ok.then(|| VoRef { f, lm, r, p });
            }
        }        // 수평 합의: 위치가 확인된 장면마다 수평면 법선(세계)을 모으고, 15 장 이상이 10° 안에서 합의하면 세계를 한 번 바로 세운다
        if !leveled && (vo_ok || a.stream.is_some()) {
            if let Some(n) = up_normal(&depth, w, h, seq.k, &r) {
                grav_obs.push(n);
            }
            if grav_obs.len() >= 15 {
                let mean = |v: &[V3]| {
                    let s = v.iter().fold([0.0; 3], |a, b| add(&a, b));
                    let l = norm(&s).max(1e-9);
                    s.map(|x| x / l)
                };
                let m0 = mean(&grav_obs);
                let inl: Vec<V3> = grav_obs.iter().copied().filter(|n| dot(n, &m0) > 10f64.to_radians().cos()).collect();
                if inl.len() * 10 >= grav_obs.len() * 6 {
                    let up = mean(&inl);
                    let fix = rot_between(&up, &[0.0, 0.0, 1.0]);
                    let rot_ref = |rf: &mut VoRef| {
                        rf.r = mm(&fix, &rf.r);
                        rf.p = mv(&fix, &rf.p);
                        for l in rf.lm.iter_mut().flatten() {
                            l.0 = mv(&fix, &l.0);
                        }
                    };
                    if let Some(rf) = vo_ref.as_mut() {
                        rot_ref(rf);
                    }
                    if let Some(rf) = vo_last.as_mut() {
                        rot_ref(rf);
                    }
                    r = mm(&fix, &r);
                    p = mv(&fix, &p);
                    let rot = |q: (M3, V3)| (mm(&fix, &q.0), mv(&fix, &q.1));
                    vo_pose = vo_pose.map(rot);
                    dvo_ref_pose = dvo_ref_pose.map(rot);
                    for hh in fast_hist.iter_mut() {
                        (hh.1, hh.2) = rot((hh.1, hh.2));
                    }
                    vo_vel = mv(&fix, &vo_vel);
                    stream_fix = mm(&fix, &stream_fix);
                    leveled = true;
                    eprintln!("수평 맞춤: 처음 가정이 {:.1}° 기울어져 있었음({} / {} 장 합의)", up[2].clamp(-1.0, 1.0).acos().to_degrees(), inl.len(), grav_obs.len());
                } else if grav_obs.len() > 60 {
                    grav_obs.drain(..30); // 합의가 안 되면 오래된 것부터 버리고 계속
                }
            }
        }
        if a.stream.is_some() && leveled {
            est_traj.push((t, p, r));
        }
        let c2 = Instant::now();
        clock.load += load;        clock.seg += seg_s;

        // 바닥 높이: 화면 아래쪽 점이 카메라 아래로 얼마나 떨어졌나(95 %) → 카메라 높이, 짧게 평활
        let back = |u: usize, v: usize| -> Option<V3> {
            let z = depth[v * w + u] as f64;
            (z > 0.2 && z < 6.0).then(|| [(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z])
        };
        let mut below = vec![];
        let up_c = [r[2][0], r[2][1], r[2][2]]; // 세계 z 를 카메라 좌표로(Rᵀ·ẑ)
        for v in (h / 2..h).step_by(16) {
            for u in (8..w).step_by(16) {
                if let Some(pc) = back(u, v) {
                    below.push(-dot(&pc, &up_c));
                }
            }
        }
        if below.len() > 50 {
            below.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let ch = below[(below.len() as f64 * 0.95) as usize];
            if (0.5..2.5).contains(&ch) {
                hist_h.push(ch);
            }
        }
        let h_now = cam_height(&hist_h);
        let z0v = if leveled { *z0.get_or_insert(p[2] - h_now) } else { p[2] - h_now };
        last_z0v = z0v;
        let pwc = [p[0], p[1], p[2] - z0v];
        let floor_z = pwc[2] - h_now;
        // 위치가 확인된 장면만 지도·물체에 넣는다(실패해 예측 위치뿐인 장면은 지도를 어긋나게 함)
        let map_ok = (!a.vo || vo_ok) && leveled;
        let (c3, c4) = map_frame(a, seq.k, t, &r, &pwc, floor_z, &depth, &rgb, w, h, &dets, map_ok, &mut grid, &mut mem, &mut traj, &mut map_traj);
        clock.part[P::Vo as usize] += (c2 - c_vo).as_secs_f64();
        clock.part[P::Map as usize] += (c3 - c2).as_secs_f64();
        clock.part[P::Obj as usize] += (c4 - c3).as_secs_f64();
        let c_pt = Instant::now();
        // 보관(줄인 보관이어도 점 지도는 점 작업 스레드가 붙이는 점 격자로 다시 쌓음)
        let keep = (map_ok && a.stream.is_some() && a.rebuild_every >= 0.0)
            .then(|| kfs.push(t, anchor, raw.0, raw.1, h_now, w, h, &rgb, &depth, &dets, true))
            .flatten();
        if let (Some(p), true) = (pw.as_mut(), map_ok) {
            p.frame(nkf, r, pwc, &depth, &rgb, w, h, keep, behind);
        }
        clock.add(P::Pts, c_pt);
        last_pose = [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])];
        cam_json = json!({"pos": pwc, "R": r});
        nkf += 1;
        // 실시간: 카메라 위치만 담은 작은 pose.json 은 장면마다(뷰어가 0.1 초마다 받아 손카메라를 바로 움직임),
        // 지도·물체·그래프(view.json)는 0.4 초마다
        // --fast-pose 는 위치만 추적한 장면이 이미 더 새 위치를 썼으니 늦게 온 깊이 장면은 쓰지 않음(손카메라가 뒤로 튀지 않게)
        if a.live > 0.0 && !a.fast_pose {
            out::write_atomic(&a.out.join("pose.json"), json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": !a.vo || vo_ok}).to_string().as_bytes());
        }
        let write_now = write_late || if a.live > 0.0 { last_view.elapsed().as_secs_f64() >= 0.4 || nkf == 1 } else { nkf % (if behind { 15 } else { 5 }) == 0 || nkf == 1 } || a.snaps.contains(&nkf);
        // 3D 점 지도: 실시간은 1 초마다(전에는 장면 번호가 기록 시점과 우연히 맞을 때만 나가 몇 초씩 비었다), 아니면 장면 수로
        let scene_now = write_now && (scene_late || if a.live > 0.0 { last_scene.elapsed().as_secs_f64() >= 1.0 } else { nkf % 50 == 0 || nkf % a.scene_every == 1 || a.scene_every == 1 });
        // 갈아 끼우기 일을 한 장면이면 정기 기록을 다음 장면으로(스냅샷 장면은 제때)
        let write_now = if busy && write_now && !a.snaps.contains(&nkf) {
            (write_due, scene_due) = (true, scene_now);
            false
        } else {
            write_now
        };
        if !write_now && nkf >= last_merge + 15 {
            let c_m = Instant::now();
            last_merge = nkf;
            mem.prune(t);
            mem.merge_pass();
            clock.add(P::Merge, c_m);
        }
        if write_now || view_late {
            let c_sc = Instant::now();
            let mut scene_done = false;
            if write_now {
            last_view = Instant::now();
            if scene_now || nkf == 1 || a.snaps.contains(&nkf) {
                last_scene = Instant::now();
                scene_done = true;
                // 새로 확정된 칸만 조각 파일로: 뷰어는 아직 안 받은 조각만 받는다(원격 접속에서 지도 전체를 매번 받지 않게)
                scene_json = json!({"run": run_id, "voxel": a.scene_vox});
                if let Some(p) = pw.as_ref() {
                    p.write(PTS_BUDGET);
                    scene_json["octree"] = p.json();
                }
            }
            }
            // 병합 후처리는 키프레임 15 장마다(물체 쌍을 매번 다 보지 않게, 중복이 자라기 전에 합침)
            let merge = write_now && nkf >= last_merge + 15;
            if merge {
                last_merge = nkf;
            }
            clock.add(P::Scene, c_sc);
            let c_v = Instant::now();
            // 점 노드를 쓴 장면은 view.json 을 다음 장면으로 미룸(한 장면에 몰리지 않게). 병합은 제때(결과가 같게)
            if scene_done && !a.snaps.contains(&nkf) && nkf > 1 {
                if merge {
                    mem.prune(t);
                    mem.merge_pass();
                }
                view_due = true;
            } else {
                write_view(&a.out, &grid, &mut mem, &mut room_tr, &traj, t, last_pose, &cam_json, &scene_json, merge, VIEW_REFRESH);
            }
            clock.add(if merge { P::Merge } else { P::View }, c_v);
            let c_p = Instant::now();
            if !view_late && !view_due && std::mem::take(&mut prune_objs) {
                prune_object_files(&a.out, &mem);
            }
            clock.add(P::Prune, c_p);
        }
        if a.snaps.contains(&nkf) {
            let sd = PathBuf::from(format!("{}_snap", a.out.display())).join(format!("kf{nkf:04}"));
            copy_dir(&a.out, &sd);
            out::png_rgb(&sd.join("frame.png"), w as u32, h as u32, &rgb);
        }
        let c5 = Instant::now();
        clock.vo += (c2 - c_vo).as_secs_f64();
        clock.map += (c3 - c2).as_secs_f64();
        clock.obj += (c4 - c3).as_secs_f64();
        clock.write += (c5 - c4).as_secs_f64();
        clock.n += 1;
        out::slow(c_top);
        clock.end_frame(c_top.elapsed().as_secs_f32() * 1e3, nkf);
        if clock.t_span.0 == 0.0 { clock.t_span.0 = t; }
        clock.t_span.1 = t;
        if lag_on {
            let l = (pace0.elapsed().as_secs_f64() - (t - pace_t0) / lag_rate) as f32;
            clock.lag.push(l);
            if rb_job.is_some() {
                lag_rb = (lag_rb.0 + 1, lag_rb.1.max(l));
            }
        }
        if nkf % 25 == 0 {
            let n = clock.n as f64;
            println!("kf {nkf} objects {} ({} gone) | ms/kf load {:.0} seg {:.0} vo {:.0} map {:.0} obj {:.0} write {:.0} | {:.1} kf/s | vo fail {vo_fail} dvo {vo_dvo} reject {vo_reject} ref {vo_nref} pos ({:.2},{:.2},{:.2})",
                     mem.objs.len(), mem.objs.iter().filter(|o| o.state == "gone").count(),
                     clock.load / n * 1e3, clock.seg / n * 1e3, clock.vo / n * 1e3, clock.map / n * 1e3, clock.obj / n * 1e3, clock.write / n * 1e3,
                     n / t0.elapsed().as_secs_f64(), p[0], p[1], p[2]);
        }
    }
    // 끝: 마지막 다시 쌓기 뒤에 노드가 또 움직였으면 끝난 그래프로 한 번 더(결과 지도·물체·poses_map.txt 가 최종 위치를 따름)
    let mut installed = false;
    if let Some(job) = rb_job.take() {
        let rb = job.join().expect("다시 쌓기 스레드");
        install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
        (n_rebuild, installed) = (n_rebuild + 1, true);
    }
    if kfs.dirty && a.rebuild_every >= 0.0 {
        let snap = snapshot(&kfs, &stream_fix, z0.unwrap_or(0.0));
        let rb = rebuild_maps(a, seq.k, &snap, mem.next + 100_000, kfs.moved);
        kfs.clear_dirty();
        install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
        (n_rebuild, installed) = (n_rebuild + 1, true);
    }
    if installed {
        fresh_outputs(&mut run_id, a.scene_vox, None); // scene_json 은 아래 마지막 기록에서 다시 만듦
    }
    if a.stream.is_some() {
        eprintln!("다시 쌓기 {n_rebuild} 번, 보관 장면 {} 개({:.0} MB)", kfs.len(), kfs.bytes() as f64 / 1048576.0);
    }
    eprintln!("{}", out::stats::report());
    if !est_traj.is_empty() {
        let mut s = String::from("# timestamp x y z qx qy qz qw (OpenCV 카메라 → 세계)\n");
        for (t, p, r) in &est_traj {
            let q = quat_of(r);
            s += &format!("{t:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n", p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
        }
        let _ = std::fs::write(a.out.join("poses_est.txt"), s);
    }
    // 지도에 넣은 장면의 카메라 위치(view.json 좌표, 바닥 높이 맞춘 뒤)
    if !map_traj.is_empty() {
        let mut s = String::from("# timestamp x y z qx qy qz qw (OpenCV 카메라 → 지도)\n");
        for (t, p, r) in &map_traj {
            let q = quat_of(r);
            s += &format!("{t:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n", p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
        }
        let _ = std::fs::write(a.out.join("poses_map.txt"), s);
    }
    let seg = worker.join().expect("worker");
    scene_json = json!({"run": run_id, "voxel": a.scene_vox});
    // 점 지도: 남은 노드를 모두 쓰고 점 작업 스레드를 끝냄
    let (n_pts, scene_mb, pts_dropped) = match pw.as_mut() {
        Some(p) => {
            let (s, j) = p.finish();
            scene_json["octree"] = j;
            (s.n_points(), s.bytes() as f64 / 1048576.0, (p.dropped, p.skipped))
        }
        None => (0, 0.0, (0, 0)),
    };
    let t_end = traj.last().map_or(0.0, |x| x.0);
    for o in mem.objs.iter_mut().filter(|o| o.dirty) {
        o.cells_saved = 0; // 마지막 기록은 바뀐 물체를 모두 내보냄
    }
    embed::drain(); // 마지막 병합이 모든 물체의 사진 벡터를 보게
    write_view(&a.out, &grid, &mut mem, &mut room_tr, &traj, t_end, last_pose, &cam_json, &scene_json, true, usize::MAX);
    if n_rebuild > 0 {
        prune_object_files(&a.out, &mem);
    }
    out::flush_writer();
    embed::finish();
    let n = clock.n.max(1) as f64;
    let mut by: std::collections::BTreeMap<&str, u32> = Default::default();
    for o in &mem.objs {
        *by.entry(o.name.as_str()).or_default() += 1;
    }
    let mut summary = json!({"keyframes": nkf, "objects": mem.objs.len(), "gone": mem.objs.iter().filter(|o| o.state == "gone").count(),
        "scene_points": n_pts, "scene_mb": scene_mb, "pts_dropped": pts_dropped.0, "pts_near_skipped": pts_dropped.1, "ms_per_kf": {"load": clock.load / n * 1e3, "seg": clock.seg / n * 1e3, "vo": clock.vo / n * 1e3, "map": clock.map / n * 1e3,
        "obj": clock.obj / n * 1e3, "write": clock.write / n * 1e3}, "kf_per_s": n / t0.elapsed().as_secs_f64(), "by_name": by, "depth_scale_vs_kinect": scale_stats(&scales), "vo": vo_report(&ate, vo_fail), "vo_reject": vo_reject, "vo_keyframes": vo_nref, "vo_dvo": vo_dvo, "pose_only_frames": n_fast, "pose_only_ms": t_fast / (n_fast.max(1) as f64) * 1e3, "merged": mem.merged});
    if !clock.frame.is_empty() {
        let mut f = clock.frame.clone();
        f.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |p: f64| f[((f.len() - 1) as f64 * p) as usize];
        summary["frame_ms"] = json!({"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": q(1.0)});
    }
    if a.stream.is_some() {
        summary["stream_lost"] = json!(stream_lost);
        let (joined, waiting) = kfs.held_counts();
        summary["stream_held"] = json!({"frames": stream_held, "joined": joined, "waiting": waiting});
    }
    summary["frame_parts"] = clock.parts_json();
    summary["degraded_frames"] = json!(n_behind);
    summary["dropped_late"] = json!(drop_n.load(std::sync::atomic::Ordering::Relaxed));
    summary["seg_runs"] = json!(seg_n.load(std::sync::atomic::Ordering::Relaxed));
    {
        let s = *seg_stats.lock().unwrap();
        if s.n > 0 {
            let m = 1e3 / s.n as f64;
            summary["seg_stage_ms"] = json!({ "pre": s.pre * m, "infer": s.infer * m, "post": s.post * m });
            summary["seg_per_s"] = json!(s.n as f64 / s.wall.max(1e-9));
            summary["seg_pipe"] = json!(a.seg_pipe);
        }
    }
    let span = (clock.t_span.1 - clock.t_span.0).max(1e-6);
    summary["stream_kf_per_s"] = json!(clock.n as f64 / span);
    if !clock.lag.is_empty() {
        let l = &clock.lag;
        let mut s = l.clone();
        s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        summary["lag_s"] = json!({"p50": s[s.len() / 2], "p95": s[(s.len() - 1) * 95 / 100], "max": s[s.len() - 1], "end": l[l.len() - 1], "rebuild_frames": lag_rb.0, "rebuild_max": lag_rb.1,
            "every_10pct": (1..=10).map(|i| l[(l.len() - 1) * i / 10]).collect::<Vec<_>>()});
    }
    let (cpu, cpu_all) = out::cpu::report();
    if cpu_all > 0.0 {
        summary["cpu_s"] = json!({"main": cpu[0], "reader": cpu[1], "pts": cpu[2], "rebuild": cpu[3], "writer": cpu[4], "infer": cpu[5], "other": cpu_all - cpu.iter().sum::<f64>(), "total": cpu_all});
    }
    println!("{summary}");
    seg
}



/// 다시 쌓은 지도 한 벌(작업 스레드에서 만들어 본 스레드가 갈아 끼움)
struct Rebuilt {
    pts: Option<PtsFiles>, // 점 지도 노드 파일 전부
    grid: Grid,
    scene: SceneMap,
    mem: Memory,
    traj: Vec<(f64, [f64; 3])>,
    map_traj: Vec<(f64, V3, M3)>,
    last_seq: u64, // 이 장면까지 넣음(그 뒤 보관한 장면은 갈아 끼울 때 이어서 넣음)
    extra: Snap, // 작업 스레드에서 이미 이어 넣은 장면(쌓는 동안 새로 보관한 것): 점 지도에는 갈아 끼울 때 보냄
    n: usize,
    secs: f64,
    moved: (f64, f64),
}

type Snap = Vec<(std::sync::Arc<kfstore::Kf>, M3, V3)>;

/// 보관한 장면과 지금 노드 위치로 만든 위치(수평 맞춤·바닥 높이 원점 적용, 지도 좌표)
/// 카메라 높이(바닥 위): 최근 15 번 잰 값의 가운데 값
fn cam_height(hist: &[f64]) -> f64 {
    if hist.is_empty() {
        return 1.4;
    }
    let mut s: Vec<f64> = hist.iter().rev().take(15).copied().collect();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[s.len() / 2]
}

fn snapshot(kfs: &kfstore::KfStore, fix: &M3, z0: f64) -> Snap {
    kfs.kfs()
        .iter()
        .filter(|kf| kfs.usable(kf))
        .map(|kf| {
            let (r0, p0) = kfs.pose(kf);
            let p = mv(fix, &p0);
            (kf.clone(), mm(fix, &r0), [p[0], p[1], p[2] - z0])
        })
        .collect()
}

/// 보관한 장면을 주어진 위치로 빈 지도·물체·지나온 길에 처음부터 쌓는다. 물체 id 는 next_id 부터.
fn rebuild_maps(a: &Args, k: [f64; 4], snap: &Snap, next_id: u32, moved: (f64, f64)) -> Rebuilt {
    let c0 = Instant::now();
    let mut grid = Grid::new(0.05);
    let mut scene = SceneMap::new(a.scene_vox, 0.04);
    scene.min_obs = a.pts_obs;
    let mut mem = Memory::new(0.012, 0.04);
    mem.next = next_id;
    let (mut traj, mut map_traj) = (vec![], vec![]);
    let (mut t_dec, mut t_map, mut t_obj, mut t_pts) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, (kf, r, pwc)) in snap.iter().enumerate() {
        let cd = Instant::now();
        let (kd, kr) = (kf.depth(), kf.rgb());
        t_dec += cd.elapsed().as_secs_f64();
        let (c3, c4) = map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &kd, &kr, kf.w, kf.h, &kf.dets, true, &mut grid, &mut mem, &mut traj, &mut map_traj);
        {
            // 점 지도: 점 작업 스레드가 보관해 둔 점 격자(온 장면을 풀지 않음)
            let cp = Instant::now();
            let zz = kf.pts.wait();
            if let Some(g) = zz.as_ref().map(PtsGrid::from_z) {
                scene.add(i as u32, &grid_points(k, &g, r, pwc));
            }
            t_pts += cp.elapsed().as_secs_f64();
        }
        t_map += (c3 - cd).as_secs_f64();
        t_obj += (c4 - c3).as_secs_f64();
        if (i + 1) % 15 == 0 {
            mem.prune(kf.t);
            mem.merge_pass();
        }
    }
    let ce = Instant::now();
    let pts = Some(export_points(&mut scene)); // 점 지도 옥트리 노드 파일도 작업 스레드에서
    let t_exp = ce.elapsed().as_secs_f64();
    let last_seq = snap.last().map_or(0, |s| s.0.seq);
    let secs = c0.elapsed().as_secs_f64();
    eprintln!("다시 쌓기 나눔: 보관 장면 풀기 {t_dec:.2} 초 + 점유 격자 {:.2} 초 + 물체 {t_obj:.2} 초 + 점 지도 {t_pts:.2} 초 + 노드 바이트 {t_exp:.2} 초, 합 {secs:.2} 초(병합 포함)", t_map);
    Rebuilt { pts, grid, scene, mem, traj, map_traj, last_seq, extra: vec![], n: snap.len(), secs, moved }
}

/// 다시 쌓은 지도로 갈아 끼우고, 쌓는 동안 새로 보관한 장면을 지금 위치로 이어서 넣는다
#[allow(clippy::too_many_arguments)]
fn install(a: &Args, k: [f64; 4], rb: Rebuilt, kfs: &kfstore::KfStore, fix: &M3, z0: f64, grid: &mut Grid, pw: Option<&mut PtsWorker>,
           mem: &mut Memory, traj: &mut Vec<(f64, [f64; 3])>, map_traj: &mut Vec<(f64, V3, M3)>) {
    let next = mem.next.max(rb.mem.next);
    let mut rmem = rb.mem;
    let kept = rmem.adopt_ids(mem);
    (*grid, *mem, *traj, *map_traj) = (rb.grid, rmem, rb.traj, rb.map_traj);
    mem.next = next;
    let newer = newer_than(kfs, fix, z0, rb.last_seq);
    let n_extra = rb.extra.len();
    if let Some(pw) = pw {
        pw.replace(rb.scene, rb.pts.unwrap_or_default());
        for (i, (kf, r, pwc)) in rb.extra.iter().chain(newer.iter()).enumerate() {
            pw.stored((rb.n + i) as u32, *r, *pwc, kf.clone());
        }
    }
    for (kf, r, pwc) in newer.iter() {
        map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &kf.depth(), &kf.rgb(), kf.w, kf.h, &kf.dets, true, grid, mem, traj, map_traj);
    }
    eprintln!("다시 쌓기: 노드 최대 {:.1} cm·{:.1}° 움직임 → 장면 {} 개 {:.2} 초(작업 스레드), 그동안 들어온 장면 {} 개 이어 넣음(작업 스레드 + 본 스레드), 물체 {} 개(id 이어받음 {kept})",
              rb.moved.0 * 100.0, rb.moved.1, rb.n, rb.secs, format!("{n_extra} + {}", newer.len()), mem.objs.len());
}

/// 스트림이 이만큼(초) 밀리면 가볍게 처리
const LAG_DEGRADE: f64 = 0.5;
/// 스트림 장면이 찍은 시각보다 이만큼(초) 늦으면 버림(--no-drop 이 아니면)
const DROP_LATE: f64 = 0.2;
/// view.json 한 번에 점 파일을 다시 만들 물체 수 상한(다시 쌓은 뒤 물체가 모두 바뀌어도 한 장면에 몰리지 않게, 나머지는 다음 기록에)
const VIEW_REFRESH: usize = 8;
/// 점 지도: 한 번에 다시 쓸 노드의 점 수 상한(점 작업 스레드가 한 번에 오래 붙잡지 않게)
const PTS_BUDGET: usize = 300_000;
/// 점 작업 스레드에 밀린 장면이 이만큼이면 스트림 장면은 점 지도를 건너뜀(본 스레드는 기다리지 않음)
const PTS_QUEUE: usize = 6;
/// 이만큼(거리 m, 각도 °) 안 움직인 장면은 점 격자 일부(1/PTS_SUB²)만 넣음
const PTS_GAP_M: f64 = 0.04;
const PTS_GAP_DEG: f64 = 4.0;
/// 가까운 자리 장면은 점 격자 간격을 이만큼 곱해(점 1/PTS_SUB²) 어긋남을 돌려 가며 넣음
const PTS_SUB: usize = 2;

/// 노드 파일 바이트: (수준, 노드 키, 바이트)
type PtsFiles = Vec<(u32, u64, Vec<u8>)>;

/// 점 격자: 장면 화소 (off + step 간격), 거른 뒤 남은 점의 깊이(m, 0 = 점 없음)·색
struct PtsGrid {
    step: usize,
    off: [usize; 2],
    gw: usize,
    gh: usize,
    d: Vec<f32>,
    rgb: Vec<u8>,
}

impl PtsGrid {
    fn from_z(z: &kfstore::PtsZ) -> PtsGrid {
        let (d, rgb) = z.unpack();
        PtsGrid { step: z.step, off: z.off, gw: z.gw, gh: z.gh, d, rgb }
    }
    fn pack(&self) -> kfstore::PtsZ {
        kfstore::PtsZ::pack(self.step, self.off, self.gw, self.gh, &self.d, &self.rgb)
    }
}

/// 장면에서 점 지도에 넣을 점을 고른다(--scene-step 간격): 깊이 경계·비스듬히 본 면을 빼고 --pts-depth 안만.
/// 깊이는 5×5 이웃 중 같은 면(Kinect 잡음 폭 안)의 평균으로 — 평면에서는 가운데 값과 같고
/// 잡음은 이웃 수의 제곱근만큼 줄어 면이 얇아짐.
/// part: Some(p) = 간격 PTS_SUB 배 격자의 p 번째 어긋남(점 1/PTS_SUB²) — 가까운 자리 장면이 돌아가며 모든 칸을 채움
fn pts_grid(a: &Args, k: [f64; 4], depth: &[f32], rgb: &[u8], w: usize, h: usize, part: Option<usize>) -> PtsGrid {
    use rayon::prelude::*;
    let [fx, fy, cx, cy] = k;
    let maxd = a.max_depth.min(a.pts_depth);
    let (step, off) = match part {
        Some(p) => (a.scene_step * PTS_SUB, [(p % PTS_SUB) * a.scene_step, (p / PTS_SUB % PTS_SUB) * a.scene_step]),
        None => (a.scene_step, [0, 0]),
    };
    let (gw, gh) = ((w - off[0]).div_ceil(step), (h - off[1]).div_ceil(step));
    let mut d = vec![0f32; gw * gh];
    let mut c = vec![0u8; gw * gh * 3];
    out::par(|| d.par_chunks_mut(gw).zip(c.par_chunks_mut(gw * 3)).enumerate().for_each(|(gy, (dr, cr))| {
        let v = off[1] + gy * step;
        for gx in 0..gw {
            let u = off[0] + gx * step;
            let z = depth[v * w + u] as f64;
            if !(z > 0.2 && z < 6.0) {
                continue;
            }
            let pc = [(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z];
            if pc[2] < maxd && !depth_edge(depth, w, h, u, v) && facing(depth, w, h, k, u, v, &pc) {
                dr[gx] = smooth_depth(depth, w, u, v);
                let o = (v * w + u) * 3;
                cr[gx * 3..gx * 3 + 3].copy_from_slice(&rgb[o..o + 3]);
            }
        }
    }));
    PtsGrid { step, off, gw, gh, d, rgb: c }
}

/// 5×5 이웃 중 가운데와 Kinect 잡음 폭(3 mm + 0.4 %·z²) 안인 깊이의 평균(가장자리 2 화소 안쪽만 불림: facing 이 보장)
fn smooth_depth(d: &[f32], w: usize, u: usize, v: usize) -> f32 {
    let z0 = d[v * w + u];
    let tol = 0.003 + 0.004 * z0 * z0;
    let (mut s, mut n) = (0f32, 0f32);
    for y in v - 2..=v + 2 {
        for &z in &d[y * w + u - 2..=y * w + u + 2] {
            if (z - z0).abs() < tol {
                s += z;
                n += 1.0;
            }
        }
    }
    s / n
}

/// 점 격자 → (세계 점, 색, 카메라 거리)
fn grid_points(k: [f64; 4], g: &PtsGrid, r: &M3, pwc: &V3) -> Vec<(V3, [u8; 3], f32)> {
    let [fx, fy, cx, cy] = k;
    let mut out = Vec::with_capacity(g.d.len() / 2);
    for gy in 0..g.gh {
        let v = (g.off[1] + gy * g.step) as f64;
        for gx in 0..g.gw {
            let i = gy * g.gw + gx;
            let z = g.d[i] as f64;
            if z <= 0.0 {
                continue;
            }
            let pc = [((g.off[0] + gx * g.step) as f64 - cx) / fx * z, (v - cy) / fy * z, z];
            out.push((add(&mv(r, &pc), pwc), [g.rgb[i * 3], g.rgb[i * 3 + 1], g.rgb[i * 3 + 2]], z as f32));
        }
    }
    out
}

enum PtsMsg {
    /// 실시간 장면: 점을 고르고(보관 장면이면 점 격자를 압축해 붙임) 지도에 넣음
    Frame { kf: u32, r: M3, pwc: V3, depth: Vec<f32>, rgb: Vec<u8>, w: usize, h: usize, keep: Option<std::sync::Arc<kfstore::Kf>>, part: Option<usize> },
    /// 맡아 둘 장면: 점 격자만 만들어 붙임(지도에는 이어진 뒤 다시 쌓기가 넣음)
    Hold { depth: Vec<f32>, rgb: Vec<u8>, w: usize, h: usize, keep: std::sync::Arc<kfstore::Kf> },
    /// 보관 장면(다시 쌓은 뒤 이어 넣기): 붙여 둔 점 격자로
    Stored { kf: u32, r: M3, pwc: V3, store: std::sync::Arc<kfstore::Kf> },
    /// 노드 파일 쓰기(점 수 상한, 0 = 모두), 다 쓰면 알림(기다리는 모드)
    Write(usize, Option<std::sync::mpsc::Sender<()>>),
    /// 다시 쌓은 지도로 갈아 끼우고 노드 파일 전부를 새 판 폴더에
    Replace(Box<SceneMap>, PtsFiles),
}

/// 점 작업 스레드: 점 고르기·지도 넣기·옥트리 노드 쓰기를 모두 맡아 본 스레드는 장면을 넘기기만 한다.
/// 노드 목록 JSON 은 쓰기가 끝난 뒤에야 바뀌므로 view.json 은 늘 다 쓴 판만 가리킨다
struct PtsWorker {
    tx: Option<std::sync::mpsc::Sender<PtsMsg>>,
    json: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
    published: std::sync::Arc<std::sync::atomic::AtomicU64>, // view.json 에 실은 판(이 판 폴더는 지우지 않음)
    queued: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    handle: Option<std::thread::JoinHandle<SceneMap>>,
    wait: bool, // 밀려도 기다림(TUM 입력: 결과가 늘 같게)
    out: PathBuf,
    gap: (f64, f64), // 이만큼(m, 라디안) 안 움직인 장면은 점 격자 일부만
    last: Option<(M3, V3)>, // 마지막으로 다 넣은 장면 위치
    skipped: u32, // 점 격자 일부만 넣은 가까운 자리 장면 수
    dropped: u32,
}

impl PtsWorker {
    fn spawn(a: std::sync::Arc<Args>, k: [f64; 4], mut scene: SceneMap, wait: bool) -> PtsWorker {
        use std::sync::atomic::Ordering::Relaxed;
        let (tx, rx) = std::sync::mpsc::channel::<PtsMsg>();
        let json = std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null));
        let published = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(scene.run));
        let queued = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let out = a.out.clone();
        let gap = (PTS_GAP_M, PTS_GAP_DEG.to_radians());
        let (j2, p2, q2) = (json.clone(), published.clone(), queued.clone());
        let handle = std::thread::spawn(move || {
            out::thread_role(out::Role::Points);
            let (mut t_grid, mut t_add, mut t_write, mut nf) = (0f64, 0f64, 0f64, 0u32);
            for m in rx {
                match m {
                    PtsMsg::Frame { kf, r, pwc, depth, rgb, w, h, keep, part } => {
                        let c0 = Instant::now();
                        let g = pts_grid(&a, k, &depth, &rgb, w, h, part);
                        if let Some(s) = keep {
                            let _ = s.pts.set(Some(g.pack()));
                        }
                        let c1 = Instant::now();
                        scene.add(kf, &grid_points(k, &g, &r, &pwc));
                        (t_grid, t_add, nf) = (t_grid + (c1 - c0).as_secs_f64(), t_add + c1.elapsed().as_secs_f64(), nf + 1);
                        out::slow(c0);
                        q2.fetch_sub(1, Relaxed);
                    }
                    PtsMsg::Hold { depth, rgb, w, h, keep } => {
                        let c0 = Instant::now();
                        let _ = keep.pts.set(Some(pts_grid(&a, k, &depth, &rgb, w, h, None).pack()));
                        out::slow(c0);
                        q2.fetch_sub(1, Relaxed);
                    }
                    PtsMsg::Stored { kf, r, pwc, store } => {
                        if let Some(g) = store.pts.wait().as_ref().map(PtsGrid::from_z) {
                            scene.add(kf, &grid_points(k, &g, &r, &pwc));
                        }
                    }
                    PtsMsg::Write(budget, ack) => {
                        let c0 = Instant::now();
                        let j = write_points(&a.out, &mut scene, budget, None, p2.load(Relaxed));
                        out::slow(c0);
                        t_write += c0.elapsed().as_secs_f64();
                        *j2.lock().unwrap() = j;
                        if let Some(t) = ack {
                            let _ = t.send(());
                        }
                    }
                    PtsMsg::Replace(s, files) => {
                        scene = *s;
                        let j = write_points(&a.out, &mut scene, PTS_BUDGET, Some(files), p2.load(Relaxed));
                        *j2.lock().unwrap() = j;
                    }
                }
            }
            if std::env::var("MEMMAP_PTS_LOG").is_ok() {
                let n = nf.max(1) as f64;
                eprintln!("점 작업 스레드: 장면 {nf} 개, 점 고르기 {:.1} ms/장 + 지도 넣기 {:.1} ms/장, 노드 쓰기 합 {t_write:.2} 초", t_grid / n * 1e3, t_add / n * 1e3);
            }
            scene
        });
        PtsWorker { tx: Some(tx), json, published, queued, handle: Some(handle), wait, out, dropped: 0, gap, last: None, skipped: 0 }
    }
    fn send(&self, m: PtsMsg) {
        self.tx.as_ref().unwrap().send(m).expect("점 작업 스레드");
    }
    /// 실시간 장면 넘기기: 밀렸으면(스트림) 건너뜀 — 보관 장면이면 빈 점 격자로 표시해 다시 쌓기가 기다리지 않게.
    /// 직전에 다 넣은 장면과 거의 같은 자리에서 본 장면은 점 격자 일부만(어긋남을 돌려 가며) 넣음: 칸마다 본 횟수는 쌓이고 넣기는 1/PTS_SUB²
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, kf: u32, r: M3, pwc: V3, depth: &[f32], rgb: &[u8], w: usize, h: usize, keep: Option<std::sync::Arc<kfstore::Kf>>, thin: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.wait && self.queued.load(Relaxed) >= PTS_QUEUE {
            self.dropped += 1;
            if let Some(s) = keep {
                let _ = s.pts.set(None);
            }
            return;
        }
        self.queued.fetch_add(1, Relaxed);
        let near = thin || self.last.as_ref().is_some_and(|(lr, lp)| norm(&sub(&pwc, lp)) < self.gap.0 && rot_angle(&r, lr) < self.gap.1);
        let part = if near {
            self.skipped += 1;
            Some(self.skipped as usize % (PTS_SUB * PTS_SUB))
        } else {
            self.last = Some((r, pwc));
            None
        };
        self.send(PtsMsg::Frame { kf, r, pwc, depth: depth.to_vec(), rgb: rgb.to_vec(), w, h, keep, part });
    }
    /// 맡아 둘 장면: 점 격자만 만들어 붙이게 넘김(밀렸으면 빈 점 격자로 표시)
    fn hold(&mut self, depth: &[f32], rgb: &[u8], w: usize, h: usize, keep: std::sync::Arc<kfstore::Kf>) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.wait && self.queued.load(Relaxed) >= PTS_QUEUE {
            self.dropped += 1;
            let _ = keep.pts.set(None);
            return;
        }
        self.queued.fetch_add(1, Relaxed);
        self.send(PtsMsg::Hold { depth: depth.to_vec(), rgb: rgb.to_vec(), w, h, keep });
    }
    fn stored(&self, kf: u32, r: M3, pwc: V3, store: std::sync::Arc<kfstore::Kf>) {
        self.send(PtsMsg::Stored { kf, r, pwc, store });
    }
    /// 노드 쓰기 요청(기다리는 모드면 다 쓸 때까지 기다려 바로 뒤 json() 이 이번 판)
    fn write(&self, budget: usize) {
        if self.wait {
            let (t, r) = std::sync::mpsc::channel();
            self.send(PtsMsg::Write(budget, Some(t)));
            let _ = r.recv();
        } else {
            self.send(PtsMsg::Write(budget, None));
        }
    }
    fn replace(&self, scene: SceneMap, files: PtsFiles) {
        self.send(PtsMsg::Replace(Box::new(scene), files));
    }
    /// 다 쓴 마지막 노드 목록(view.json 용) — 이 판을 실었다고 표시
    fn json(&self) -> serde_json::Value {
        let j = self.json.lock().unwrap().clone();
        if let Some(run) = j["run"].as_u64() {
            self.published.store(run, std::sync::atomic::Ordering::Relaxed);
        }
        j
    }
    /// 남은 노드를 모두 쓰고 끝냄: (지도, 노드 목록)
    fn finish(&mut self) -> (SceneMap, serde_json::Value) {
        self.write(0);
        self.tx.take();
        let scene = self.handle.take().unwrap().join().expect("점 작업 스레드");
        // 지금 판이 아닌 판 폴더(갈아 끼우기 전 판)를 지움
        let a_out = self.out.clone();
        if let Ok(rd) = std::fs::read_dir(a_out.join("scene")) {
            for e in rd.flatten() {
                if e.path().is_dir() && e.file_name().to_string_lossy() != scene.run.to_string() {
                    let _ = std::fs::remove_dir_all(e.path());
                }
            }
        }
        (scene, self.json())
    }
}

/// 다시 쌓은 점 지도의 노드를 모두 파일 바이트로(작업 스레드에서)
fn export_points(scene: &mut SceneMap) -> PtsFiles {
    scene.place();
    scene.mark_all();
    scene.take_dirty(0).into_iter().map(|(l, k)| (l, k, scene.node_bytes(l, k))).collect()
}

/// 점 지도 옥트리 출력: 새로 확정된 점을 노드에 넣고 점이 늘어난 노드 파일(scene/<판>/n<수준>_<x>_<y>_<z>.bin)과 노드 목록
/// (scene/<판>/octree.json)을 쓴다. budget: 한 번에 다시 쓸 점 수 상한(0 = 모두), pre: 다시 쌓은 지도의 노드 전부(작업 스레드가 만든 바이트).
/// 지금 판과 view.json 에 실린 판(published)이 아닌 판 폴더는 지움
fn write_points(dir: &Path, scene: &mut SceneMap, budget: usize, pre: Option<PtsFiles>, published: u64) -> serde_json::Value {
    let c0 = Instant::now();
    let added = scene.place();
    let sub = format!("scene/{}", scene.run);
    let all = pre.is_some();
    if let Ok(rd) = std::fs::read_dir(dir.join("scene")) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() && name != scene.run.to_string() && name != published.to_string() {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
    std::fs::create_dir_all(dir.join(&sub)).unwrap();
    let mut dirty: PtsFiles = pre.unwrap_or_default();
    dirty.extend(scene.take_dirty(budget).into_iter().map(|(l, k)| (l, k, scene.node_bytes(l, k))));
    for (l, k, b) in dirty.iter_mut() {
        let c = maps::node_coord(*k);
        out::write_owned(&dir.join(format!("{sub}/n{l}_{}_{}_{}.bin", c[0], c[1], c[2])), std::mem::take(b));
    }
    let idx = scene.index();
    let ver: i64 = idx.iter().map(|n| n[5]).sum();
    let j = json!({"run": scene.run, "ver": ver, "path": format!("{sub}/octree.json"), "dir": sub, "voxel": scene.vox, "g": maps::OCT_G, "depth": maps::OCT_DEPTH, "n": scene.n_points()});
    if !dirty.is_empty() || all {
        out::write_owned(&dir.join(format!("{sub}/octree.json")), serde_json::to_vec(&json!({"run": scene.run, "voxel": scene.vox, "g": maps::OCT_G, "depth": maps::OCT_DEPTH, "nodes": idx})).unwrap());
    }
    if std::env::var("MEMMAP_PTS_LOG").is_ok() {
        eprintln!("점 지도: 새 점 {added}, 노드 {} 개 씀, 전체 {} 점 / 노드 {} 개, {:.1} ms, {:.0} MB", dirty.len(), scene.n_points(), idx.len(), c0.elapsed().as_secs_f64() * 1e3, scene.bytes() as f64 / 1048576.0);
    }
    j
}

/// 다시 쌓은 뒤 출력 새로: 실행 번호를 바꿔 뷰어가 받은 지도를 비우고 지금 지도를 처음부터 받게 함
fn fresh_outputs(run_id: &mut u64, vox: f64, pw: Option<&PtsWorker>) -> serde_json::Value {
    *run_id += 1;
    let mut sj = json!({"run": *run_id, "voxel": vox});
    if let Some(pw) = pw {
        sj["octree"] = pw.json();
    }
    sj
}

/// 한 장면을 지도(점유 격자)·물체·지나온 길에 넣는다. 실시간 장면과 루프 클로저 뒤 다시 쌓기가 같이 쓴다.
/// r, pwc: 카메라 → 지도(바닥 높이 맞춘 뒤), floor_z: 이 장면의 바닥 높이. 돌려줌: (지도 끝, 물체 끝) 시각
#[allow(clippy::too_many_arguments)]
fn map_frame(a: &Args, k: [f64; 4], t: f64, r_: &M3, pwc_: &V3, floor_z: f64, depth: &[f32], rgb: &[u8], w: usize, h: usize, dets: &Option<Vec<seg::Det>>,
             map_ok: bool, grid: &mut Grid, mem: &mut Memory,
             traj: &mut Vec<(f64, [f64; 3])>, map_traj: &mut Vec<(f64, V3, M3)>) -> (Instant, Instant) {
    let (r, pwc) = (*r_, *pwc_);
    let [fx, fy, cx, cy] = k;
    let back = |u: usize, v: usize| -> Option<V3> {
        let z = depth[v * w + u] as f64;
        (z > 0.2 && z < 6.0).then(|| [(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z])
    };
    let to_w = |pc: &V3| add(&mv(&r, pc), &pwc);

    // 점유 지도(16 화소 간격) + 3D 점 지도(3 화소 간격)
    let (mut occ, mut flo) = (vec![], vec![]);
    for v in (8..h).step_by(16) {
        for u in (8..w).step_by(16) {
            if let Some(pc) = back(u, v) {
                let q = to_w(&pc);
                let hg = q[2] - floor_z;
                if hg > 0.15 && hg < 1.8 {
                    occ.push((q[0], q[1], hg > WALL_H));
                } else if hg < 0.08 {
                    flo.push((q[0], q[1]));
                }
            }
        }
    }
    // 위치가 확인된 장면만 지도·물체에 넣는다(실패해 예측 위치뿐인 장면은 지도를 어긋나게 함)
    if map_ok {
        grid.add((pwc[0], pwc[1]), &occ, &flo);
        map_traj.push((t, pwc, r));
    }
    // 점 지도는 점 작업 스레드가 맡음(PtsWorker) — 여기서는 점유 격자·물체만
    let c3 = Instant::now();

    // 물체
    let mut seen = HashSet::new();
    if dets.is_some() && map_ok {
        mem.n_seg += 1;
    }
    for d in dets.iter().flatten().filter(|_| map_ok) {
        let name = seg::COCO.get(d.cls).copied().unwrap_or("object");
        if name == "person" {
            continue;
        }
        let [x1, y1, x2, y2] = d.box_.map(|v| v as usize);
        let (bw, bh) = (x2 - x1, y2 - y1);
        let m = erode(&d.mask, bw, bh, 3);
        // 마스크 안 깊이 있는 화소: 먼저 세어 간격을 정하고, 그 간격으로 최대 약 2,500 개만 모은다
        let valid = |yy: usize, xx: usize| m[yy * bw + xx] && depth[(y1 + yy) * w + x1 + xx] as f64 > 0.2;
        let cnt = (0..bh).map(|yy| (0..bw).filter(|&xx| valid(yy, xx)).count()).sum::<usize>();
        if cnt < 80 {
            continue;
        }
        let step = (cnt / 2500).max(1);
        let mut pix: Vec<(usize, usize, f64)> = Vec::with_capacity(cnt / step + 1);
        let mut k = 0usize;
        for yy in 0..bh {
            for xx in 0..bw {
                if valid(yy, xx) {
                    if k % step == 0 {
                        let (u, v) = (x1 + xx, y1 + yy);
                        pix.push((u, v, depth[v * w + u] as f64));
                    }
                    k += 1;
                }
            }
        }
        let mut zs: Vec<f64> = pix.iter().map(|q| q.2).collect();
        let mid = zs.len() / 2;
        let med = *zs.select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap()).1; // 정렬 대신 선택(같은 값)
        if med > 5.0 {
            continue;
        }
        let pts: Vec<(V3, [u8; 3])> = pix
            .iter()
            .filter(|q| (q.2 - med).abs() < 0.25 * med + 0.1)
            .map(|&(u, v, z)| {
                let o = (v * w + u) * 3;
                (to_w(&[(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z]), [rgb[o], rgb[o + 1], rgb[o + 2]])
            })
            .collect();
        if pts.len() < 50 || is_wall(&pts) {
            continue;
        }
        // 사진은 그 물체의 지금 사진보다 클 때만 만든다(장면마다 모든 검출을 복사하지 않게)
        let crop = || {
            let mut crgb = Vec::with_capacity(bw * bh * 3);
            let mut cdep = Vec::with_capacity(bw * bh);
            for yy in 0..bh {
                let o = ((y1 + yy) * w + x1) * 3;
                crgb.extend_from_slice(&rgb[o..o + bw * 3]);
                for xx in 0..bw {
                    cdep.push((depth[(y1 + yy) * w + x1 + xx] * 1000.0).clamp(0.0, 65535.0) as u16);
                }
            }
            Crop { w: bw as u32, h: bh as u32, rgb: crgb, mask: m.iter().map(|&b| if b { 255 } else { 0 }).collect(), depth: cdep }
        };
        let id = mem.update(name, d.score, &pts, med as f32, t, (bw * bh) as u32, crop, (a.depth != "kinect").then_some(pwc));
        seen.insert(id);
    }
    if dets.is_some() && map_ok {
        mem.check_missing(&seen, k, &r, &pwc, &depth, w, h);
    }
    let c4 = Instant::now();
    // 지나온 길: 5 cm 이상 움직였을 때만 점 하나(가만히 있어도 장면마다 쌓이면 view.json 이 커져 원격 뷰어가 느려짐)
    if traj.last().is_none_or(|(_, q)| (q[0] - pwc[0]).hypot(q[1] - pwc[1]) >= 0.05) {
        traj.push((t, [pwc[0], pwc[1], floor_z]));
    }
    (c3, c4)
}
#[allow(clippy::too_many_arguments)]
/// 지금 기억에 없는 물체의 파일(objects/O<id>_*)을 지운다: 다시 쌓기·병합으로 사라진 id 의 점·사진이 쌓여
/// 사무실 스트림 끝에 objects/ 가 1,292 개 파일·64 MB 가 되던 것
fn prune_object_files(dir: &Path, mem: &Memory) {
    let live: HashSet<u32> = mem.objs.iter().map(|o| o.id).collect();
    let Ok(rd) = std::fs::read_dir(dir.join("objects")) else { return };
    for e in rd.flatten() {
        let name = e.file_name();
        let Some(id) = name.to_str().and_then(|s| s.strip_prefix('O')).and_then(|s| s.split('_').next()).and_then(|s| s.parse::<u32>().ok()) else { continue };
        if !live.contains(&id) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// 이 높이(m) 넘는 막힘은 가구가 아니라 벽으로 본다(책상 0.75, 의자 등받이·모니터 1.2 안팎) — 방 나누기에 씀
const WALL_H: f64 = 1.3;

fn write_view(dir: &Path, grid: &Grid, mem: &mut Memory, room_tr: &mut rooms::Tracker, traj: &[(f64, [f64; 3])], stamp: f64, pose: [f64; 3],
              cam: &serde_json::Value, scene: &serde_json::Value, merge: bool, refresh: usize) {
    let mut left = refresh;
    if merge {
        mem.prune(stamp);
        mem.merge_pass();
    }
    let gi = grid.image();
    let mut objs = vec![];
    for i in 0..mem.objs.len() {
        // 점 파일은 점(칸)이 5 % 넘게 늘었을 때만 다시 쓴다(관측마다 통째로 다시 쓰지 않게). 그 사이 바뀐 것은 다음 기록 때 함께.
        let grown = {
            let o = &mem.objs[i];
            o.dirty && (o.cells_saved == 0 || o.cells.len() * 20 > o.cells_saved * 21) && left > 0
        };
        left -= grown as usize;
        let pts = if grown { Some(mem.refresh(i)) } else { None };
        let o = &mut mem.objs[i];
        let base = format!("objects/O{}", o.id);
        if let Some(pts) = pts {
            o.n_pts = pts.len();
            out::ply(&dir.join(format!("{base}_points.ply")), pts);
            o.cells_saved = o.cells.len();
            o.dirty = false;
            o.stamp += 1;
        }
        // 사진은 더 크게 본 장면으로 바뀌었을 때만
        if o.crop_ver != o.crop_saved {
            if let Some(c) = &o.crop {
                out::png_rgb(&dir.join(format!("{base}_rgb.png")), c.w, c.h, &c.rgb);
                out::png_gray(&dir.join(format!("{base}_mask.png")), c.w, c.h, &c.mask);
                out::png_u16(&dir.join(format!("{base}_depth.png")), c.w, c.h, &c.depth);
                embed::submit(dir, o.id, c.w, c.h, &c.rgb);
            }
            o.crop_saved = o.crop_ver;
        }
        let mut e = json!({"id": o.id, "name": o.name, "state": o.state, "pos": o.pos, "first_pos": o.first_pos, "extent": o.extent,
            "n_obs": o.n_obs, "score": o.score, "last_seen": o.last_seen, "structural": false,
            "points": {"path": format!("{base}_points.ply"), "stamp": o.stamp, "voxel": mem.vox, "n": o.n_pts}});
        if o.crop.is_some() {
            e["rgbd"] = json!({"rgb": format!("{base}_rgb.png"), "mask": format!("{base}_mask.png"), "depth": format!("{base}_depth.png"), "v": o.crop_ver});
            if embed::on() {
                e["emb"] = json!(embed::rel_path(o.id));
            }
        }
        objs.push(e);
    }
    let (grid_j, rooms, doors, edges) = match &gi {
        Some(g) => {
            out::pgm(&dir.join("map.pgm"), g.w, g.h, &g.px);
            let ol: Vec<_> = mem.objs.iter().map(|o| (o.id, o.pos, o.extent, o.state == "gone", o.name.as_str())).collect();
            let (rj, dj, ej) = graph::build(room_tr, g, grid.res, &ol);
            (json!({"width": g.w, "height": g.h, "resolution": grid.res, "origin": [g.x0 as f64 * grid.res, g.y0 as f64 * grid.res]}), rj, dj, ej)
        }
        None => (json!(null), json!([]), json!([]), json!([])),
    };
    let nodes: Vec<_> = traj.iter().enumerate().map(|(i, (t, p))| json!({"id": format!("A{i}"), "kind": "agent", "pos": p, "t": t})).collect();
    let mut view = json!({"stamp": stamp, "pose": pose, "grid": grid_j, "objects": objs, "rooms": rooms, "doors": doors,
        "graph": {"nodes": nodes, "edges": edges}, "robot": "handheld camera", "cam": cam});
    if !scene.is_null() {
        view["scene_points"] = scene.clone();
    }
    out::write_owned(&dir.join("view.json"), serde_json::to_vec(&view).unwrap());
}

/// 표면이 시선과 이루는 각: 이웃 화소(±2)로 만든 법선과 시선의 코사인이 0.3(약 73°) 이상인 점만 — 비스듬히 본 면은 깊이 잡음이
/// 시선 방향으로 길게 번져 줄무늬 껍질이 됨
fn facing(d: &[f32], w: usize, h: usize, k: [f64; 4], u: usize, v: usize, pc: &V3) -> bool {
    if u < 2 || v < 2 || u + 2 >= w || v + 2 >= h {
        return false;
    }
    let [fx, fy, cx, cy] = k;
    let b = |x: usize, y: usize| {
        let z = d[y * w + x] as f64;
        [(x as f64 - cx) / fx * z, (y as f64 - cy) / fy * z, z]
    };
    let (du, dv) = (sub(&b(u + 2, v), &b(u - 2, v)), sub(&b(u, v + 2), &b(u, v - 2)));
    let n = cross(&du, &dv);
    let nn = norm(&n) * norm(pc);
    nn > 0.0 && dot(&n, pc).abs() / nn >= 0.3
}

/// 점 주변 5×5 깊이가 8 % 넘게 들쭉날쭉하면 경계
fn depth_edge(d: &[f32], w: usize, h: usize, u: usize, v: usize) -> bool {
    let (mut lo, mut hi) = (f32::MAX, 0f32);
    for y in v.saturating_sub(2)..(v + 3).min(h) {
        for x in u.saturating_sub(2)..(u + 3).min(w) {
            let z = d[y * w + x];
            lo = lo.min(z);
            hi = hi.max(z);
        }
    }
    lo <= 0.0 || hi / lo > 1.08
}

/// 시각 오도메트리 오차: 정답 위치와의 거리 RMSE(첫 장면을 정답으로 시작했으니 따로 정렬하지 않음)
/// 두 회전 사이 각(aᵀb 의 회전각, 라디안)
fn rot_angle(a: &M3, b: &M3) -> f64 {
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a[k][i] * b[k][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

fn vo_report(ate: &[(V3, V3)], fail: u32) -> serde_json::Value {
    if ate.is_empty() {
        return json!(null);
    }
    let n = ate.len() as f64;
    let rmse = (ate.iter().map(|(a, b)| { let d = sub(a, b); dot(&d, &d) }).sum::<f64>() / n).sqrt();
    let path: f64 = ate.windows(2).map(|w| norm(&sub(&w[1].1, &w[0].1))).sum();
    let end = norm(&sub(&ate.last().unwrap().0, &ate.last().unwrap().1));
    // 배율만 맞춘 오차: 시작점 기준 크기 한 개(단안+딥러닝 깊이는 전체 배율이 틀릴 수 있다)
    let o = ate[0].1;
    let (mut num, mut den) = (0.0, 0.0);
    for (e, g) in ate {
        let (e, g) = (sub(e, &o), sub(g, &o));
        num += dot(&e, &g);
        den += dot(&e, &e);
    }
    let s = if den > 0.0 { num / den } else { 1.0 };
    let rs = (ate.iter().map(|(e, g)| { let d = sub(&add(&sub(e, &o).map(|x| x * s), &o), g); dot(&d, &d) }).sum::<f64>() / n).sqrt();
    json!({"ate_rmse_m": rmse, "ate_scaled_m": rs, "scale": s, "end_err_m": end, "path_m": path, "fail": fail})
}

/// 깊이 배율(딥러닝 / Kinect) 분포: 중앙값·5 %·95 %
fn scale_stats(s: &[f64]) -> serde_json::Value {
    if s.is_empty() {
        return json!(null);
    }
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    json!({"median": q(0.5), "p05": q(0.05), "p95": q(0.95), "n": v.len()})
}

fn copy_dir(src: &Path, dst: &Path) {
    let _ = std::fs::remove_dir_all(dst);
    std::fs::create_dir_all(dst.join("objects")).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
        }
    }
    for e in std::fs::read_dir(src.join("objects")).unwrap().flatten() {
        std::fs::copy(e.path(), dst.join("objects").join(e.file_name())).unwrap();
    }
}

fn tr(m: &M3) -> M3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i]))
}

/// glibc 는 스레드마다 malloc 구역(arena)을 새로 만들어(최대 코어 수 × 8) 작업 스레드(다시 쌓기·기록·rayon)가
/// 잠깐 쓴 큰 버퍼가 구역마다 남는다 → 구역 2 개로 묶는다(사무실 스트림 최대 메모리 1.54 → 1.35 GB, 속도 같음).
/// MALLOC_ARENA_MAX 를 직접 주면 그 값을 따른다.
fn limit_malloc_arenas() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        unsafe extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        const M_ARENA_MAX: i32 = -8;
        unsafe {
            mallopt(M_ARENA_MAX, 2);
        }
    }
}

fn main() {
    limit_malloc_arenas();
    let a = args();
    if let Some(m) = &a.embed {
        embed::start(m);
    }
    out::set_layout(&a.threads);
    out::start_pools();
    out::set_slow(a.slow);
    out::thread_role(out::Role::Main);
    if a.live > 0.0 || a.stream.is_some() {
        out::start_writer(); // 실시간·스트림: 파일 쓰기(PLY·PNG 짜기 포함)는 기록 스레드가
    }
    // 분할 모델(ORT 스레드)은 추론 스레드 역할의 코어에서 만든다(ORT 스레드가 만든 스레드의 코어 묶음을 물려받음)
    out::thread_role(out::Role::Infer);
    let mut seg = match &a.seg_cache {
        Some(dir) => { let path = a.seg.clone(); Box::new(segsim::CachedSeg::new(dir, a.seg_cpu_ms, Box::new(move || make_seg(&path)))) as Box<dyn seg::Segmenter + Send> }
        None => make_seg(&a.seg),
    };
    out::thread_role(out::Role::Main);
    if a.seg_check > 0 {
        return seg_check(&a, seg.as_mut(), a.seg_check);
    }
    loop {
        seg = run(&a, seg);
        // 리셋 블록으로 끝났으면 같은 스트림을 이어 읽어 빈 지도에서 다시 시작
        if stream::reset_pending() {
            eprintln!("리셋: 지도를 비우고 다시 시작");
            continue;
        }
        if !a.looping {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}

/// 보관 장면 중 seq 뒤의 것(지금 노드 위치로)
fn newer_than(kfs: &kfstore::KfStore, fix: &M3, z0: f64, seq: u64) -> Snap {
    snapshot(kfs, fix, z0).into_iter().filter(|s| s.0.seq > seq).collect()
}

/// 다시 쌓기가 끝난 뒤 그동안 들어온 장면이 이만큼 넘으면 작업 스레드에서 먼저 이어 넣음(갈아 끼우는 장면에서 본 스레드가 멈추지 않게)
const CATCH_UP_MAX: usize = 3;

/// 다시 쌓은 지도에 쌓는 동안 새로 보관한 장면을 이어 넣음(작업 스레드). 본 스레드에서 넣는 것과 같은 순서·같은 함수
fn catch_up(a: &Args, k: [f64; 4], rb: &mut Rebuilt, newer: Snap) {
    let c0 = Instant::now();
    for (kf, r, pwc) in newer.iter() {
        map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &kf.depth(), &kf.rgb(), kf.w, kf.h, &kf.dets, true, &mut rb.grid, &mut rb.mem, &mut rb.traj, &mut rb.map_traj);
        rb.last_seq = kf.seq;
    }
    rb.secs += c0.elapsed().as_secs_f64();
    rb.extra.extend(newer);
}
