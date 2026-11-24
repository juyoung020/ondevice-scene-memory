//! 손카메라 영상 + 포즈 → 점유 지도·3D 점 지도·물체 기억·씬그래프(방·엣지) → 기억 폴더(view.json, sgview 가 읽음).
//!
//!   ondevice --tum data/tum/<seq> --out data/memory/<name> [--seg <분할 모델>(기본 ObjectSAM)] [--embed <SigLIP2>] [--vocab <이름 표>] [--every 5] [--conf 0.35] [--live 1] [--no-skip] [--loop]
//!   ondevice --slam "<위치 인자>" --out <dir> [--serve "<포트> [뷰어 인자]" [--save-every S]] ...   (위치 프로그램을 같은 프로세스에서, docs/bridge.md)
//!   ondevice --stream <파일|-> --out <dir> ...   (녹화한 장면 스트림)
//!   ondevice slam <위치 인자> · ondevice view <기억 폴더> [뷰어 인자]
//!
//! 지도·트래커는 누적식(maps.rs): 키프레임마다 새 점만 해시 칸에 더하고, 기록은 칸을 한 번 훑기만 한다.
//! 깊이는 센서 실측(스트림 · TUM Kinect)을 쓴다.
mod depth;
mod embed;
mod geom;
mod graph;
mod chunks;
mod kfstore;
mod maps;
#[cfg(feature = "dxrt")]
mod npu_test;
mod out;
mod rooms;
mod seg;
mod segsim;
mod stream;
mod thermal;
mod tum;

use depth::Depth;
use geom::*;
use maps::*;
use serde_json::json;
use std::collections::{HashMap, HashSet};
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
    gt: Option<String>, // --gt <파일>: TUM 영상의 위치를 이 파일(TUM 형식)로(예: RTAB-Map 이 낸 위치)
    npu_jobs: usize,    // --npu-jobs K: 스트림 분할 추론을 NPU 에 K 장까지 함께 올림(코어 여럿이 나눠 돎, 기본 4, 1 = 한 장씩)
    threads: String,    // --threads: 스레드 우선순위·코어 배치(auto · none · 역할=nice[:코어],…)
    slow: f32,          // --slow F: 보드 흉내, 본·점·다시 쌓기 스레드 일을 F 배로(코어를 붙잡고 기다림)
    seg_cache: Option<PathBuf>, // --seg-cache DIR: 보드 흉내, 장면별 분할 결과를 폴더에 남기고 있으면 읽어 씀(추론이 코어를 쓰지 않음)
    seg_cpu_ms: f64,    // --seg-cpu-ms M: --seg-cache 로 읽어 온 장면마다 읽기 스레드가 M ms 코어를 붙잡음(보드 분할의 CPU 전·후처리 몫)
    stream_rate: f64,   // --stream-rate R: 스트림을 찍은 시각대로 R 배속으로 내보냄(0 = 최대 속도), 밀림(지연) 기록
    seg_check: usize,   // --seg-check N: 지도는 만들지 않고, 스트림 분할 장면 N 장에 전·후처리를 기준 구현과 견줌(결과 같음 여부 + 시간)
    serve: Option<sgview::Config>, // --serve "<포트> [뷰어 인자...]": 뷰어 서버를 같은 프로세스에서(기억 폴더에 쓴 파일은 메모리로)
    save_every: f64,       // --save-every S: --serve 의 메모리 기억 폴더를 S 초마다 디스크에(0 = 끝에만, 뷰어 저장 버튼은 언제나)
    serve_exit: bool,      // --serve-exit: 입력이 끝나면 뷰어도 끝냄(측정·검증용)
    no_drop: bool,      // --no-drop: 스트림이 밀려도 장면을 버리지 않음(재생 결과가 늘 같게)
    scene_vox: f64,
    max_depth: f64,
    scene_step: usize, // 3D 지도에 쓸 화소 간격
    pts_depth: f64,    // --pts-depth M: 점 지도에 넣을 최대 깊이(멀수록 깊이 잡음이 커 표면이 두꺼워짐)
    pts_obs: u8,       // --pts-obs N: 고운 칸이 점이 되려면 키프레임 N 장 이상에서 보여야(한두 번 맞은 잡음 점을 거름)
    stream: Option<String>, // --stream <파일|->: 녹화·파이프로 받는 장면 스트림(위치·컬러·깊이). --slam 이면 "--slam"
    slam: Option<slam::app::Args>, // --slam "<위치 인자>": 같은 프로세스에서 위치 프로그램을 돌림(장면은 메모리 큐로)
    rebuild_every: f64,     // --rebuild-every S: 지도 노드가 움직이면(루프 클로저) S 초에 한 번까지 보관한 장면을 고친 위치로 다시 쌓음(0 = 끝에만)
    embed: Option<String>,  // --embed <vision_model.onnx | .dxnn>: 물체 사진마다 검색용 벡터(SigLIP2 영상 인코더)를 남김
    vocab: Option<String>,  // --vocab <글 벡터 파일>: 물체 사진이 가장 닮은 이름을 물체 이름으로 씀(--embed 와 함께, 파일은 sgview --encode-vocab 이 만듦)
    structural: Option<String>, // --structural <글 벡터 파일>: 벽 · 바닥 · 천장 판정 표(기본 vocab/structural.voc, 사진 벡터가 있어야)
    kf_store_mb: usize,     // --kf-store-mb: 다시 쌓기용 장면 보관 예산(넘으면 하나 걸러 버림)
    pool: usize,            // --pool N: 작업 풀 일꾼 수(기본 3, 펼치기 · 줄 일이 모두 여기서)
}

/// 명령줄을 처음에 한 번 읽는다: 물체 기억 설정 + 묶음 인자(--slam · --serve)는 각 프로그램의 설정으로 바로 읽어 둔다(잘못된 인자는 스레드를 띄우기 전에 멈춤)
fn args(v: &common::cli::Argv) -> Args {
    let get = |k: &str| v.get(k);
    Args {
        tum: get("--tum").unwrap_or_default().into(),
        out: get("--out").or_else(|| get("--seg-check").map(|_| String::new())).expect("--out <dir>").into(), // --seg-check 는 쓰는 파일이 없음
        seg: get("--seg").unwrap_or_else(|| models::seg()),
        embed: models::opt(get("--embed"), models::embed),
        vocab: models::opt(get("--vocab"), models::vocab).filter(|_| get("--embed").is_none_or(|e| e != "off" && e != "none")), // 이름은 사진 벡터가 있어야
        structural: models::opt(get("--structural"), models::structural).filter(|_| get("--embed").is_none_or(|e| e != "off" && e != "none")),
        every: get("--every").map_or(5, |s| s.parse().unwrap()),
        conf: get("--conf").map_or(0.35, |s| s.parse().unwrap()),
        live: get("--live").map_or(0.0, |s| s.parse().unwrap()),
        no_skip: v.flag("--no-skip"),
        looping: v.flag("--loop"),
        scene_every: get("--scene-every").map_or(10, |s| s.parse().unwrap()),
        gt: get("--gt"),
        npu_jobs: get("--npu-jobs").and_then(|s| s.parse().ok()).unwrap_or(4).clamp(1, 8),
        threads: get("--threads").unwrap_or("auto".into()),
        slow: get("--slow").map_or(1.0, |s| s.parse().unwrap()),
        stream_rate: get("--stream-rate").map_or(0.0, |s| s.parse().unwrap()),
        seg_cache: get("--seg-cache").map(Into::into),
        seg_cpu_ms: get("--seg-cpu-ms").map_or(0.0, |s| s.parse().unwrap()),
        seg_check: get("--seg-check").map_or(0, |s| s.parse().unwrap()),
        serve: v.group("--serve").map(|g| {
            let mut it = g.into_iter();
            let port = it.next().expect("--serve <포트>");
            let out = get("--out").unwrap_or_default();
            sgview::Config::parse([out, "--port".into(), port].into_iter().chain(it).collect())
        }),
        save_every: get("--save-every").map_or(0.0, |s| s.parse().unwrap()),
        serve_exit: v.flag("--serve-exit"),
        no_drop: v.flag("--no-drop"),
        scene_vox: get("--scene-vox").map_or(0.01, |s| s.parse().unwrap()),
        max_depth: get("--max-depth").map_or(4.0, |s| s.parse().unwrap()),
        scene_step: get("--scene-step").map_or(2, |s| s.parse().unwrap()),
        pts_depth: get("--pts-depth").map_or(4.0, |s| s.parse().unwrap()),
        pts_obs: get("--pts-obs").map_or(1, |s| s.parse().unwrap()),
        // --slam "<slam_bridge 인자>": 위치 프로그램을 이 프로세스 안에서 돌려 장면을 메모리로 받음(스트림 입력과 같은 경로, 이름만 "--slam")
        stream: get("--stream").or_else(|| get("--slam").map(|_| "--slam".into())),
        slam: v.group("--slam").map(|g| {
            let argv: Vec<String> = std::iter::once("slam".to_string()).chain(g).collect();
            slam::app::parse(&argv).unwrap_or_else(|c| std::process::exit(c))
        }),
        rebuild_every: get("--rebuild-every").map_or(5.0, |s| s.parse::<f64>().unwrap().max(0.0)),
        kf_store_mb: get("--kf-store-mb").map_or_else(kf_store_default_mb, |s| s.parse().unwrap()),
        pool: get("--pool").map_or(3, |s| s.parse().unwrap()),
    }
}

/// 마스크 침식(정사각 k×k): 가장자리 깊이 번짐 제거
#[cfg(test)]
fn erode(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut s = ErodeBuf::default();
    erode_into(m, w, h, r, &mut s);
    s.b
}

/// 침식 작업 칸(장면·검출마다 다시 씀: 할당 없음)
#[derive(Default)]
struct ErodeBuf {
    a: Vec<bool>,
    b: Vec<bool>,
    run: Vec<u32>,
}

/// 마스크 침식 결과를 s.b 에(가로 → 세로 두 번, 분리 가능)
fn erode_into(m: &[bool], w: usize, h: usize, r: usize, s: &mut ErodeBuf) {
    let n = w * h;
    s.a.clear();
    s.a.resize(n, false);
    s.b.clear();
    s.b.resize(n, false);
    let (a, b) = (&mut s.a, &mut s.b);
    let lim = 2 * r;
    for y in 0..h {
        let (mr, ar) = (&m[y * w..(y + 1) * w], &mut a[y * w..(y + 1) * w]);
        let mut run = 0usize;
        for x in 0..w {
            run = if mr[x] { run + 1 } else { 0 };
            if run > lim {
                ar[x - r] = true;
            }
        }
    }
    // 세로는 열마다 이어진 길이를 들고 줄 순서로 훑음(메모리를 차례로 읽게)
    s.run.clear();
    s.run.resize(w, 0);
    let run = &mut s.run;
    for y in 0..h {
        let ar = &a[y * w..(y + 1) * w];
        for x in 0..w {
            // 분기 없이: 칸이 켜졌으면 +1, 아니면 0
            run[x] = (run[x] + 1) * ar[x] as u32;
        }
        if y >= lim {
            let br = &mut b[(y - r) * w..(y - r + 1) * w];
            for x in 0..w {
                br[x] = run[x] as usize > lim;
            }
        }
    }
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

/// 물체가 아닌 조각(이름을 붙여도 틀리는 검출): 바닥·천장 면(넓고 평평한 수평 면이 바닥 근처 또는 2 m 위),
/// 선처럼 가는 조각(책상 모서리 · 문틀: 한 축만 길고 나머지 두 축이 3 cm 안), 아주 작은 조각(가장 긴 축 6 cm 안),
/// 화면 가장자리 셋 이상에 잘린 마스크(장면 대부분을 덮는 배경 조각). 그러면 Some(까닭)
fn fragment(p: &[(V3, [u8; 3])], floor_z: f64, cam_z: f64, border_sides: usize) -> Option<&'static str> {
    if border_sides >= 3 {
        return Some("border");
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
    // 표준편차 × 4 ≈ 고르게 퍼진 점의 폭
    let span = w.map(|x| x.max(0.0).sqrt() * 4.0);
    if span[2] < 0.06 {
        return Some("tiny");
    }
    if span[1] < 0.03 && span[2] > 0.3 {
        return Some("sliver");
    }
    let flat = span[0] < 0.04 && span[1] > 1.0;
    let horizontal = v[2][0].abs() > 0.9;
    if flat && horizontal && (m[2] < floor_z + 0.15 || m[2] > floor_z + 2.0) {
        return Some("floor/ceiling");
    }
    // 천장 조각: 카메라보다 30 cm 넘게 위의 평평한 수평 면(넓이 50 cm 넘음)
    if span[0] < 0.05 && span[1] > 0.5 && horizontal && m[2] > cam_z + 0.3 {
        return Some("ceiling");
    }
    None
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

/// 한 장면의 '대략 수평'인 가장 큰 면(바닥·책상)을 RANSAC 으로 찾아 그 위쪽 법선을 세계 좌표로(IMU 없이 중력 방향 단서).
/// 여러 장면의 법선을 모아 다수결로 수평을 정한다(gravity 합의).
/// floor_only: 카메라보다 0.3 m 넘게 아래인 면만(실시간: 천장 · 책상 위를 바닥으로 잡지 않게)
fn up_normal(depth: &Depth, w: usize, h: usize, k: [f64; 4], r: &M3, floor_only: bool) -> Option<V3> {
    let [fx, fy, cx, cy] = k;
    let mut pts: Vec<V3> = vec![];
    for v in (0..h).step_by(8) {
        for u in (0..w).step_by(8) {
            let z = depth.at(v * w + u) as f64;
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
        if floor_only && d > -0.3 {
            continue; // 위쪽(가정한 위 기준) 면 = 천장 · 높은 면
        }
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

/// 기본 모델: ObjectSAM 분할(물체만, 이름 없음) + SigLIP2 사진 벡터 + 실내 이름 표(vocab/indoor.voc).
/// 모델 폴더 = common::models::dir(). `--embed off` · `--vocab off` 로 끔.
/// 분할은 클래스 없는 ObjectSAM 만(클래스를 내는 모델은 seg::seg_post 가 멈춤).
mod models {
    use std::path::PathBuf;
    fn dir() -> PathBuf {
        common::models::dir()
    }
    /// 처음 있는 파일(없으면 첫 후보: 여는 곳에서 경로째 알림)
    fn first(c: Vec<PathBuf>) -> String {
        c.iter().find(|p| p.exists()).unwrap_or(&c[0]).to_string_lossy().into_owned()
    }
    pub fn seg() -> String {
        let d = dir();
        if cfg!(feature = "dxrt") { first(vec![d.join("objectsam-416_640.dxnn"), d.join("ObjectSAM-416_640.onnx")]) } else { first(vec![d.join("ObjectSAM-416_640.onnx")]) }
    }
    pub fn embed() -> String {
        let d = dir();
        let onnx = d.join("siglip2-base-patch32-256").join("vision_model.onnx");
        if cfg!(feature = "dxrt") { first(vec![d.join("siglip2-vision_npu.dxnn"), onnx]) } else { first(vec![onnx]) }
    }
    /// 저장소 vocab/ 의 글 벡터 파일(작업 폴더 · 모델 폴더 · 실행 파일 옆 순)
    fn voc(file: &str) -> String {
        let mut c = vec![PathBuf::from("vocab").join(file), dir().join(file)];
        if let Some(e) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
            c.push(e.join("vocab").join(file));
            c.push(e.join("../../vocab").join(file));
        }
        first(c)
    }
    pub fn vocab() -> String {
        voc("indoor.voc")
    }
    pub fn structural() -> String {
        voc("structural.voc")
    }
    /// 인자를 안 주면 기본, "off" 면 끔
    pub fn opt(v: Option<String>, def: fn() -> String) -> Option<String> {
        match v {
            Some(s) if s == "off" || s == "none" => None,
            Some(s) => Some(s),
            None => Some(def()),
        }
    }
}

fn make_seg(path: &str) -> Box<dyn seg::Segmenter + Send> {
    #[cfg(feature = "dxrt")]
    if path.ends_with(".dxnn") {
        return Box::new(seg::DxSeg::new(path));
    }
    #[cfg(feature = "onnx")]
    {
        return Box::new(seg::OnnxSeg::new(common::ort::env(), path));
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
    depth: Depth,
    dets: Option<Vec<seg::Det>>, // 이번 장면에 분할을 안 돌렸으면 None(--every 사이 · 밀릴 때 가까운 장면 건너뜀)
    load: f64,
    seg: f64,
    lost: bool,       // 스트림 입력에서 위치를 놓친 장면(지도·물체에 넣지 않음)
    held: bool,       // 스트림: 맡아 둘 장면(믿는 지도와 이어지면 다시 쌓기 때 지도에 들어감)
    held_live: bool,  // 맡아 둘 장면이지만 위치는 화면에 바로(IMU 모드 임시 위치)
    gravity: bool,    // 스트림: 지도 z 가 중력 위(IMU)
    no_map: bool,     // 스트림: 위치만(IMU 로 이어 붙인 구간 · 영상이 IMU 와 어긋난 뒤), 지도 · 물체 · 보관에 넣지 않음
    anchor: u32,      // 스트림: 기준 지도 노드 id(0 = 없음)
    graph: Vec<stream::Node>, // 스트림: 이 장면 앞에 온 그래프 블록 노드
    t_in: Option<Instant>,    // 스트림: 루프가 받은 때
    t_ready: Option<Instant>, // 스트림: 분할 단계를 마친 때
}

/// 스트림 분할 단계별 걸린 시간 합(초)과 분할 횟수, 분할한 구간의 벽시계 길이(초).
/// infer = 올린 때부터 결과까지(비동기면 NPU 에 있던 시간), sub = 올리기 호출, npu = 끝 알림을 받고 결과를 맞추는 데 쓴 시간
#[derive(Default, Clone, Copy)]
struct SegStages {
    pre: f64,
    infer: f64,
    post: f64,
    sub: f64,
    npu: f64,
    n: u32,
    wall: f64,
}

/// 스트림 장면 하나가 분할 단계(전처리 → 추론 → 후처리)를 지나며 들고 다니는 것. 분할을 안 하는 장면(위치 놓침·생략)도 순서를 지키려 같은 길로 지나간다.
struct SegJob {
    sf: stream::StreamFrame,
    depth: Depth, // 받은 u16 그대로(쓸 때 미터로)
    do_seg: bool, // 이 장면에 분할을 돌림(위치를 놓친 장면은 늘 false)
    load: f64,
    graph: Vec<stream::Node>,
    bufs: (seg::SegIn, seg::SegRaw), // 전처리 결과, 추론 결과(다시 쓰는 버퍼)
    dets: Option<Vec<seg::Det>>, // 실행기가 단계를 거치지 않고 바로 준 결과(--seg-cache)
    t_front: f64, // 고른 뒤 전처리 끝까지(깊이 변환 + 전처리)
    t: [f64; 3],  // 전처리, 추론, 후처리
    t_sub: f64,   // 올리기 호출
    t_npu: f64,   // 끝 알림 뒤 결과 맞추기
    t_in: Instant, // 루프가 장면을 받은 때(장면 지연 재기)
}

/// 분할 단계 3 — 후처리: 마스크 해독, 본 스레드에 넘길 장면으로 묶음. 버퍼는 돌려준다.
fn seg_stage_post(mut j: SegJob, conf: f32, s: &mut SegStages) -> (Frame, (seg::SegIn, seg::SegRaw)) {
    let (w, h, lost, held, gravity, no_map) = (j.sf.w, j.sf.h, j.sf.lost(), j.sf.held(), j.sf.gravity(), j.sf.no_map());
    let held_live = j.sf.held_live();
    let dets = j.do_seg.then(|| {
        let c = Instant::now();
        let d = j.dets.take().unwrap_or_else(|| seg::seg_post(&j.bufs.1, j.bufs.0.geom, w, h, conf));
        out::slow(c);
        j.t[2] = c.elapsed().as_secs_f64();
        (s.pre, s.infer, s.post, s.sub, s.npu, s.n) = (s.pre + j.t[0], s.infer + j.t[1], s.post + j.t[2], s.sub + j.t_sub, s.npu + j.t_npu, s.n + 1);
        d
    });
    let mut sf = j.sf;
    let f = Frame { t: sf.t, p: sf.p, r: sf.r, rgb: std::mem::take(&mut sf.rgb), w, h, depth: j.depth, dets, load: j.load, seg: j.t_front + j.t[1] + j.t[2], lost, held, held_live, gravity, no_map, anchor: sf.anchor, graph: j.graph,
        t_in: Some(j.t_in), t_ready: Some(Instant::now()) };
    (f, j.bufs)
}

/// 처리 루프가 잠드는 곳 · 깨우개 · NPU 끝 줄 · 저장 시각(실행마다 같은 것: 리셋으로 run 을 다시 불러도 그대로)
struct LoopIo {
    poll: common::rt::Poll,
    /// 장면 입력(같은 프로세스 위치 프로그램의 큐)이 깨움
    w_in: std::sync::Arc<common::rt::Waker>,
    /// 뷰어 버튼 · 뷰 줄이 깨움
    w_misc: std::sync::Arc<common::rt::Waker>,
    /// IMU 표본이 깨움(뷰어가 있는 IMU 모드 실시간)
    w_imu: std::sync::Arc<common::rt::Waker>,
    /// 장면 사이 위치를 낼 pose.json
    fast_out: Option<PathBuf>,
    /// 뷰어 서버(--serve): 듣는 소켓 · 연결이 이 루프 epoll 에
    server: Option<sgview::Server>,
    /// 비상 열 보호(선택 일만 줄임)
    thermal: thermal::Thermal,
    npu_tx: common::rt::DoneTx<u64>,
    npu_rx: common::rt::DoneRx<u64>,
    /// 분할 실행기가 비동기(올리고 끝 알림을 받음)인지
    seg_async: bool,
    toks: Vec<u64>,
    out: PathBuf,
    save_every: f64,
    last_save: Instant,
    /// 계측: 루프 한 바퀴(깬 뒤 다시 잠들 때까지) 시간
    turns: u64,
    turn_max: f32,
    turn_sum: f64,
    turn_t0: Option<Instant>,
    /// 루프 단계별 시간 합(초): 입력 · 위치 추적, 분할 올리기, 분할 받기 · 후처리, 뷰어 · 기타, 물체 기억(풀)
    stage: [f64; 5],
}

const STAGE_NAMES: [&str; 5] = ["track", "submit", "post", "misc", "memory_pool"];

const T_IN: u64 = 1;
const T_NPU: u64 = 2;
const T_MISC: u64 = 3;
const T_IMU: u64 = 4;

impl LoopIo {
    fn new(a: &Args, argv: &common::cli::Argv) -> LoopIo {
        let mut poll = common::rt::Poll::new();
        let w_in = poll.waker(T_IN);
        let w_npu = poll.waker(T_NPU);
        let w_misc = poll.waker(T_MISC);
        let w_imu = poll.waker(T_IMU);
        let (npu_tx, npu_rx) = common::rt::done(w_npu);
        let wm = w_misc.clone();
        common::control::on_signal(move || wm.wake());
        LoopIo { poll, w_in, w_misc, w_imu, fast_out: None, server: None, thermal: thermal::Thermal::from_args(argv), npu_tx, npu_rx, seg_async: false, toks: vec![], out: a.out.clone(), save_every: if a.serve.is_some() { a.save_every } else { 0.0 },
                 last_save: Instant::now(), turns: 0, turn_max: 0.0, turn_sum: 0.0, turn_t0: None, stage: [0.0; 5] }
    }
    fn waker(&self) -> std::sync::Arc<common::rt::Waker> {
        self.w_misc.clone()
    }
    /// 할 일이 생길 때까지 잠듦(timeout: 다음 마감). 저장 마감도 함께
    fn sleep(&mut self, timeout: Option<std::time::Duration>) {
        if let Some(c) = self.turn_t0.take() {
            let ms = c.elapsed().as_secs_f32() * 1e3;
            (self.turns, self.turn_sum, self.turn_max) = (self.turns + 1, self.turn_sum + ms as f64, self.turn_max.max(ms));
        }
        let save = (self.save_every > 0.0).then(|| std::time::Duration::from_secs_f64((self.save_every - self.last_save.elapsed().as_secs_f64()).max(0.0)));
        let t = match (timeout, save) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let min = |a: Option<std::time::Duration>, b: Option<std::time::Duration>| match (a, b) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let t = min(min(t, self.server.as_ref().and_then(|s| s.timeout())), self.thermal.timeout());
        #[cfg(target_os = "linux")]
        for fd in std::mem::take(&mut *stream::NEW_FDS.lock().unwrap()) {
            self.poll.add_fd(fd, T_IN, false);
        }
        let mut toks = std::mem::take(&mut self.toks);
        self.poll.wait(&mut toks, t);
        if let Some(sv) = self.server.as_mut() {
            for &k in &toks {
                sv.event(k, &mut self.poll);
            }
        }
        self.toks = toks;
        self.turn_t0 = Some(Instant::now());
    }
    /// 잠들지 않는 바퀴(받은 장면이 있어 바로 다음 바퀴로)에서도 뷰어 연결 · 요청은 받는다: 기다리지 않고 한 번 봄
    fn serve_now(&mut self) {
        let Some(sv) = self.server.as_mut() else { return };
        let mut toks = std::mem::take(&mut self.toks);
        self.poll.wait(&mut toks, Some(std::time::Duration::ZERO));
        for &k in &toks {
            sv.event(k, &mut self.poll);
        }
        self.toks = toks;
    }
    /// 뷰어 저장 버튼 · --save-every: 메모리 기억 폴더를 디스크에(기록 줄에서, 루프는 기다리지 않음)
    fn misc(&mut self) {
        if let Some(sv) = self.server.as_mut() {
            sv.tick(&mut self.poll);
        }
        // 비상 열 보호: 1 단 = 신경망 루프(MegaLoc)를 노드 3 개에 하나로, 2 단 = 사진 벡터 묶음 늦춤. 분할 · 위치 추적은 그대로
        if let Some(l) = self.thermal.check() {
            slam::lcnet::throttle(l >= 1);
            embed::throttle(l >= 2);
            common::info!("열 보호 {l} 단: {}", self.thermal.json());
        }
        let due = self.save_every > 0.0 && self.last_save.elapsed().as_secs_f64() >= self.save_every;
        if common::control::take_save() || due {
            self.last_save = Instant::now();
            let dir = self.out.clone();
            let job = move || {
                let c = Instant::now();
                let n = out::save(&dir);
                common::info!("기억 폴더 저장: 파일 {n} 개, {:.2} 초", c.elapsed().as_secs_f64());
            };
            match out::io_lane() {
                Some(l) => l.push(move |_| job()),
                None => job(),
            }
        }
    }
    fn json(&self) -> serde_json::Value {
        json!({"turns": self.turns, "turn_ms_mean": self.turn_sum / self.turns.max(1) as f64, "turn_ms_max": self.turn_max, "sleeps": self.poll.sleeps, "thermal": self.thermal.json(),
               "stage_s": STAGE_NAMES.iter().zip(self.stage).map(|(n, v)| (n.to_string(), json!((v * 1e3).round() / 1e3))).collect::<serde_json::Map<_, _>>()})
    }
}

/// 줄에 맡긴 일 하나의 결과(끝났는지 보기 · 기다려 받기)
struct Pending<T> {
    rx: std::sync::mpsc::Receiver<T>,
    got: Option<T>,
}
impl<T: Send + 'static> Pending<T> {
    fn on(lane: &mut common::rt::Lane<()>, f: impl FnOnce() -> T + Send + 'static) -> Pending<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        lane.push(move |_| {
            let _ = tx.send(f());
        });
        Pending { rx, got: None }
    }
    fn is_finished(&mut self) -> bool {
        if self.got.is_none() {
            self.got = self.rx.try_recv().ok();
        }
        self.got.is_some()
    }
    fn join(mut self) -> T {
        if self.got.is_none() && rayon::current_thread_index().is_some() {
            // 풀 일꾼에서 기다림: 그 일(다시 쌓기)을 이 일꾼이 대신 돌 수도 있게
            common::rt::wait_until(|| self.is_finished());
        }
        self.got.take().unwrap_or_else(|| self.rx.recv().expect("줄 일"))
    }
}

/// 분할 길에 들어간 장면의 자리
enum FlSt {
    /// 분할 결과까지 다 있음(분할 안 함 · 동기 실행기)
    Ready,
    /// NPU 자리를 기다림
    Wait,
    /// NPU 에 올림: (작업 번호, 표, 올린 때)
    Npu(i32, u64, Instant),
    /// NPU 끝 알림을 받음
    Done(i32, Instant),
}

/// 장면 입력 · 분할 올리기 · 받기를 처리 루프 안에서: 장면마다 (분할 차례면) 전처리해 NPU 자리가 나면 올리고, 끝 알림이 오면
/// 올린 차례대로 후처리해 본 처리(물체 기억)에 넘긴다. NPU 를 기다리는 동안 루프는 잠들고(끝 알림 · 새 장면이 깨움) 다른 장면 일을 한다
struct Driver {
    seg: Box<dyn seg::Segmenter + Send>,
    seg_async: bool,
    rd: Option<stream::StreamReader>,
    first: Option<stream::StreamFrame>,
    /// --stream-rate: 아직 때가 안 된 장면
    held: Option<stream::StreamFrame>,
    ended: bool,
    pace0: Instant,
    pace_t0: f64,
    rate: f64,
    lag_on: bool,
    drop_late: bool,
    k0: [f64; 4],
    pose_out: Option<PathBuf>,
    n: usize,
    n_corr: u32,
    n_seg: usize,
    dropped: u32,
    graph: Vec<stream::Node>,
    last_seg: Option<([f64; 3], [[f64; 3]; 3], usize)>,
    bufs: Vec<(seg::SegIn, seg::SegRaw)>,
    q: std::collections::VecDeque<(SegJob, FlSt)>,
    jobs: usize,
    busy: usize,
    tag: u64,
    stats: SegStages,
    c_seg0: Option<Instant>,
    t_got: Instant,
    want: Option<Instant>,
    read_s: f64,
}

impl Driver {
    #[allow(clippy::too_many_arguments)]
    fn new(a: &Args, seg: Box<dyn seg::Segmenter + Send>, rd: Option<stream::StreamReader>, first: Option<stream::StreamFrame>, pace0: Instant, pace_t0: f64, lag_on: bool, io: &LoopIo) -> Driver {
        // 함께 올려 둘 수: 실시간은 실행기 상한까지(PC ONNX 1: 더 올리면 장면 지연만 늚). 결과를 늘 같게 하는 재생(--no-drop)은 모든 분할 장면을 어차피
        // 차례대로 하므로 다음 장면을 미리 올려 둬(상한 2 이상) 앞 장면이 끝나는 즉시 이어 돌게 한다(루프가 위치 추적 중이어도 실행기가 쉬지 않음)
        let cap = if a.no_drop { seg.max_jobs().max(2) } else { seg.max_jobs() };
        let jobs = if io.seg_async { a.npu_jobs.min(cap).max(1) } else { 1 };
        Driver {
            seg_async: io.seg_async,
            seg,
            k0: first.as_ref().map_or([0.0; 4], |f| f.k),
            rd,
            first,
            held: None,
            ended: false,
            pace0,
            pace_t0,
            rate: a.stream_rate,
            lag_on,
            drop_late: !a.no_drop,
            pose_out: (a.live > 0.0).then(|| a.out.join("pose.json")),
            n: 0,
            n_corr: 0,
            n_seg: 0,
            dropped: 0,
            graph: vec![],
            last_seg: None,
            bufs: vec![],
            q: Default::default(),
            jobs,
            busy: 0,
            tag: 0,
            stats: SegStages::default(),
            c_seg0: None,
            t_got: Instant::now(),
            want: None,
            read_s: 0.0,
        }
    }

    /// 루프: 입력이 끝나고 분할 길이 빌 때까지. 돌려줌: 분할 실행기(다음 실행 · 리셋이 이어 씀)
    fn run(mut self, a: &Args, seq: &tum::Tum, io: &mut LoopIo, body: &mut (dyn FnMut(Option<Frame>) + Send)) -> Box<dyn seg::Segmenter + Send> {
        if self.rd.is_none() {
            self.run_tum(a, seq, body);
            return self.seg;
        }
        // 물체 기억(마스크 → 점 · 병합 · 지도)은 물체 기억 스레드 하나에서 한 장씩 차례로: 루프는 그동안 다음 장면 위치 추적 · 분할 올리기를 겹쳐 한다.
        // 상태는 물체 기억 일이 쥐고(뮤텍스, 한 번에 하나라 다툼 없음) 장면 순서는 그대로 → 결과 같음.
        // 작업 풀 일로 두면 안 된다: 물체 기억은 점 줄(압축 · 점 격자)을 기다리는데, 점 줄 일꾼이 펼치기(par_iter) 중에 이 일을 훔쳐 돌면
        // 점 줄이 그 밑에 깔린 채 서로 기다려 멈춘다(fr3 long_office --every 2 에서 475 장째 멈춤, gdb: pts_grid → join → settle → wait_ready)
        let mem = std::sync::Mutex::new(body);
        let busy = std::sync::atomic::AtomicBool::new(false);
        let mem_ns = std::sync::atomic::AtomicU64::new(0);
        let mut mq: std::collections::VecDeque<Frame> = Default::default();
        let wk = io.waker();
        let seg = std::thread::scope(|sc| {
        let (mtx, mrx) = std::sync::mpsc::channel::<Frame>();
        {
            let (mem, busy, wk, mem_ns) = (&mem, &busy, wk.clone(), &mem_ns);
            std::thread::Builder::new().name("memory".into()).spawn_scoped(sc, move || {
                let (_, mask) = out::layout(out::Role::Points);
                common::threads::apply(0, mask);
                common::prof::track_thread();
                for f in mrx {
                    let c = Instant::now();
                    (mem.lock().unwrap())(Some(f));
                    mem_ns.fetch_add(c.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    busy.store(false, std::sync::atomic::Ordering::Release);
                    wk.wake();
                }
            }).expect("물체 기억 스레드");
        }
        loop {
            // 물체 기억 일이 비었으면 다음 장면을 넘김
            if !busy.load(std::sync::atomic::Ordering::Acquire) {
                if let Some(f) = mq.pop_front() {
                    busy.store(true, std::sync::atomic::Ordering::Release);
                    let _ = mtx.send(f);
                }
            }
            let mut progress = false;
            // NPU 끝 알림
            for tag in io.npu_rx.try_iter() {
                if let Some(f) = self.q.iter_mut().find(|f| matches!(f.1, FlSt::Npu(_, t, _) if t == tag)) {
                    if let FlSt::Npu(id, _, c) = f.1 {
                        f.1 = FlSt::Done(id, c);
                    }
                }
            }
            // 앞 장면부터 차례로 본 처리(물체 기억)
            let c_st = Instant::now();
            while mq.len() < 2 && self.q.front().is_some_and(|f| matches!(f.1, FlSt::Ready | FlSt::Done(..))) {
                let (job, st) = self.q.pop_front().unwrap();
                let f = self.finish(job, st, a.conf);
                mq.push_back(f);
                progress = true;
            }
            io.stage[2] += c_st.elapsed().as_secs_f64();
            if !mq.is_empty() && !busy.load(std::sync::atomic::Ordering::Acquire) {
                continue;
            }
            let c_st = Instant::now();
            self.submit_waiting(a.conf);
            io.stage[1] += c_st.elapsed().as_secs_f64();
            let c_st = Instant::now();
            if let Some(rd) = self.rd.as_mut() {
                rd.service(); // 실시간: 준비된 카메라 장면은 모두 위치 추적에
            }
            if !self.ended {
                progress |= self.pull(a);
            }
            io.stage[0] += c_st.elapsed().as_secs_f64();
            let c_st = Instant::now();
            io.misc();
            if let (Some(pp), Some(rd)) = (io.fast_out.as_ref(), self.rd.as_ref()) {
                if let Some((t, ts, twc)) = rd.fast_pose() {
                    fast_pose(pp, t, ts, &twc);
                }
            }
            if !busy.load(std::sync::atomic::Ordering::Acquire) {
                if let Ok(mut b) = mem.try_lock() {
                    (b)(None);
                }
            }
            io.stage[3] += c_st.elapsed().as_secs_f64();
            if self.ended && self.q.is_empty() && mq.is_empty() && !busy.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            if progress {
                io.serve_now();
                continue;
            }
            let timeout = self.held.as_ref().map(|sf| std::time::Duration::from_secs_f64(((sf.t - self.pace_t0) / self.rate - self.pace0.elapsed().as_secs_f64()).max(0.0)))
                .or_else(|| self.rd.as_ref().and_then(|r| r.due()));
            io.sleep(timeout);
        }
        });
        let _ = seg;
        io.stage[4] += mem_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9;
        let body = mem.into_inner().unwrap();
        // 마지막 장면 뒤에 온 노드(끝난 뒤 최적화): 위치 없는 장면 하나로 넘김
        if let Some(rd) = self.rd.as_mut() {
            self.graph.append(&mut rd.take_graph());
        }
        if !self.graph.is_empty() {
            let graph = std::mem::take(&mut self.graph);
            body(Some(Frame { t: 0.0, p: [0.0; 3], r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rgb: vec![], w: 0, h: 0, depth: Depth::default(), dets: None, load: 0.0, seg: 0.0, lost: true, held: false, held_live: false, gravity: false, no_map: false, anchor: 0, graph, t_in: None, t_ready: None }));
        }
        let (n, n_corr, n_seg) = (self.n, self.n_corr, self.n_seg);
        common::info!("스트림 끝: 장면 {n} 개 받음, 지도 보정 바뀜 {n_corr} 번, 분할 {n_seg} 번");
        let st = &mut self.stats;
        st.wall = self.c_seg0.map_or(0.0, |c| c.elapsed().as_secs_f64());
        if st.n > 0 {
            let m = 1e3 / st.n as f64;
            common::info!("분할 단계({}) 평균 ms: 전처리 {:.2}, 추론 {:.1}(올리기 {:.2} · 결과 맞추기 {:.2}), 후처리 {:.2} · 분할 {:.2} 번/초", if self.seg_async { "비동기" } else { "동기" }, st.pre * m, st.infer * m, st.sub * m, st.npu * m, st.post * m,
                          st.n as f64 / st.wall.max(1e-9));
        }
        DRV_STATS.with(|d| d.set((self.dropped, self.n_seg as u32, self.stats)));
        self.seg
    }

    /// 분할 차례가 끝난 장면: 결과를 맞추고 후처리해 본 처리에 넘길 장면으로
    fn finish(&mut self, mut job: SegJob, st: FlSt, conf: f32) -> Frame {
        if let FlSt::Done(id, c) = st {
            let c2 = Instant::now();
            self.seg.take(id, &mut job.bufs.1);
            job.t_npu = c2.elapsed().as_secs_f64();
            job.t[1] = c.elapsed().as_secs_f64();
            self.busy -= 1;
        }
        let (f, b) = seg_stage_post(job, conf, &mut self.stats);
        self.bufs.push(b);
        f
    }

    /// NPU 자리가 나는 대로 기다리던 장면을 차례대로 올림
    fn submit_waiting(&mut self, conf: f32) {
        for i in 0..self.q.len() {
            if self.busy >= self.jobs {
                return;
            }
            if !matches!(self.q[i].1, FlSt::Wait) {
                continue;
            }
            self.tag += 1;
            let tag = self.tag;
            let job = &mut self.q[i].0;
            let c = Instant::now();
            match self.seg.submit(&job.bufs.0, &mut job.bufs.1, tag) {
                Some(id) => {
                    job.t_sub = c.elapsed().as_secs_f64();
                    self.q[i].1 = FlSt::Npu(id, tag, c);
                    self.busy += 1;
                }
                None => {
                    Self::run_sync(self.seg.as_mut(), job, conf);
                    self.q[i].1 = FlSt::Ready;
                }
            }
        }
    }

    /// 동기 실행기(--seg-cache 분할 대역 · 올리기가 없는 실행기): 그 자리에서
    fn run_sync(seg: &mut (dyn seg::Segmenter + Send), j: &mut SegJob, conf: f32) {
        let c = Instant::now();
        j.dets = seg.direct(&j.sf.rgb, j.sf.w, j.sf.h, conf);
        if j.dets.is_none() {
            seg.infer(&j.bufs.0, &mut j.bufs.1);
        }
        j.t[1] = c.elapsed().as_secs_f64();
    }

    /// 입력 장면 받기: 결과를 늘 같게(--no-drop)면 NPU 자리를 기다리는 장면이 있는 동안 다음 장면을 읽지 않고(위치 쪽 큐가 막음),
    /// 실시간이면 늘 받되 자리를 기다리던 분할 장면은 새 장면이 이긴다(가장 새로 추적된 장면을 올림). 돌려줌: 받은 장면이 있었는지
    fn pull(&mut self, a: &Args) -> bool {
        let mut got = false;
        loop {
            let waiting = self.q.iter().any(|f| matches!(f.1, FlSt::Wait));
            if (!self.drop_late && waiting) || self.q.len() >= self.jobs + if self.drop_late { 4 } else { 2 } {
                return got;
            }
            let sf = match self.held.take().or_else(|| self.first.take()) {
                Some(sf) => sf,
                None => match self.rd.as_mut().unwrap().try_next() {
                    stream::Next::Frame(sf) => {
                        self.t_got = Instant::now();
                        self.read_s = self.want.take().map_or(0.0, |c| c.elapsed().as_secs_f64());
                        sf
                    }
                    stream::Next::Wait => {
                        self.want.get_or_insert_with(Instant::now);
                        return got;
                    }
                    stream::Next::End => {
                        self.ended = true;
                        return got;
                    }
                },
            };
            // --stream-rate: 찍은 시각(배속 R)보다 이르면 그때까지 둠(루프는 다른 일을 하거나 잠듦)
            if self.rate > 0.0 && self.n % a.every == 0 {
                let due = (sf.t - self.pace_t0) / self.rate - self.pace0.elapsed().as_secs_f64();
                if due > 0.0 {
                    self.held = Some(sf);
                    return got;
                }
            }
            got = true;
            if let Some(job) = self.take(a, sf) {
                self.push(job, a.conf);
            }
        }
    }

    /// 분할 길에 넣기: 분할 장면은 NPU 자리를 기다림(실시간이면 기다리던 장면을 이 장면이 대신함), 나머지는 바로 준비됨
    fn push(&mut self, mut job: SegJob, conf: f32) {
        if !(job.do_seg && !job.sf.lost()) {
            self.q.push_back((job, FlSt::Ready));
            return;
        }
        if !self.seg_async {
            Self::run_sync(self.seg.as_mut(), &mut job, conf);
            self.q.push_back((job, FlSt::Ready));
            return;
        }
        if self.drop_late && self.q.back().is_some_and(|f| matches!(f.1, FlSt::Wait)) {
            let (mut old, _) = self.q.pop_back().unwrap();
            old.graph.append(&mut job.graph); // 버린 장면 앞 노드는 이 장면에 붙임(차례 그대로)
            job.graph = std::mem::take(&mut old.graph);
            self.bufs.push(std::mem::take(&mut old.bufs));
            self.dropped += 1;
        }
        self.q.push_back((job, FlSt::Wait));
    }

    /// 장면 하나를 받아 분할 길 일감으로(실시간이면 위치를 바로 뷰어로). --every 차례가 아니거나 늦어 버리면 None
    fn take(&mut self, a: &Args, mut sf: stream::StreamFrame) -> Option<SegJob> {
        let every = a.every;
        // 실시간: 받은 장면의 카메라 위치를 분할 · 지도 일을 기다리지 않고 바로 뷰어로(건너뛰는 장면도)
        if let Some(pp) = &self.pose_out {
            early_pose(pp, &sf);
        }
        // 지도 보정이 바뀐 장면(루프 클로저): 이후 장면은 새 보정으로 오고, 지난 장면은 그래프 블록의 노드 위치로 다시 쌓는다
        self.n_corr += (sf.flags & stream::MAP_CORRECTED != 0) as u32;
        self.graph.append(&mut sf.graph);
        // 밀림: 찍은 시각(배속 R, 기본 1) 기준 DROP_LATE 초 넘게 늦은 장면은 버리고 다음(더 새) 장면으로. 그래프 노드는 위에서 이미 모아 다음 장면에 붙음
        let late = self.pace0.elapsed().as_secs_f64() - (sf.t - self.pace_t0) / if self.rate > 0.0 { self.rate } else { 1.0 };
        let n = self.n;
        self.n += 1;
        let skip = self.drop_late && n % every == 0 && n > 0 && late > DROP_LATE && !sf.lost();
        if skip {
            self.dropped += 1;
        }
        if n % every != 0 || skip {
            return None;
        }
        if sf.k != self.k0 {
            common::info!("스트림 장면 {}: 보정값이 바뀜 {:?} → {:?}(처음 값으로 계속)", sf.seq, self.k0, sf.k);
        }
        let c1 = Instant::now();
        let lost = sf.lost();
        if sf.flags & stream::NEW_MAP != 0 {
            common::info!("스트림 장면 {}: RTAB-Map 새 지도 {}", sf.seq, sf.map_id);
        }
        // 깊이는 u16 그대로 넘김(미터 값은 쓰는 곳에서 같은 식으로)
        let depth = Depth::Raw { d: std::mem::take(&mut sf.depth), per_m: sf.depth_per_m };
        // 스트림이 SEG_SKIP_LAG 초 넘게 밀렸을 때만: 직전 분할 키프레임에서 조금만 움직였으면 화면이 거의 같으니 분할을 건너뜀(물체 기억은 그 장면을 안 본 것으로).
        // 밀리지 않았으면(NPU 가 따라감) 늘 분할(안 밀려도 건너뛰면 물체 관측만 줄어 물체를 잃음)
        let want_seg = match &self.last_seg {
            Some(_) if !self.lag_on || late <= SEG_SKIP_LAG => true,
            Some((lp, lr, ns)) => {
                let dp = ((sf.p[0] - lp[0]).powi(2) + (sf.p[1] - lp[1]).powi(2) + (sf.p[2] - lp[2]).powi(2)).sqrt();
                let tr: f64 = (0..3).map(|i| (0..3).map(|j| lr[j][i] * sf.r[j][i]).sum::<f64>()).sum();
                let ang = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
                dp > SEG_SKIP_M || ang > SEG_SKIP_DEG || ns + 1 >= SEG_SKIP_N
            }
            None => true,
        };
        let do_seg = want_seg && !lost && !sf.no_map();
        if !lost {
            if do_seg {
                self.last_seg = Some((sf.p, sf.r, 0));
                self.n_seg += 1;
            } else if let Some(l) = self.last_seg.as_mut() {
                l.2 += 1;
            }
        }
        // 분할 전처리
        let mut bufs = self.bufs.pop().unwrap_or_default();
        let mut t = [0.0; 3];
        if do_seg {
            let c = Instant::now();
            seg::seg_pre(&sf.rgb, sf.w, sf.h, &mut bufs.0);
            out::slow(c);
            t[0] = c.elapsed().as_secs_f64();
            self.c_seg0.get_or_insert(c);
        }
        Some(SegJob { sf, depth, do_seg, load: self.read_s, graph: std::mem::take(&mut self.graph), bufs, dets: None, t_front: c1.elapsed().as_secs_f64(), t, t_sub: 0.0, t_npu: 0.0, t_in: self.t_got })
    }

    /// TUM 입력(--tum, 위치 = 정답 · --gt): 영상 · 깊이 읽기와 분할을 루프에서 차례로(--live 면 찍은 속도에 맞춤)
    fn run_tum(&mut self, a: &Args, seq: &tum::Tum, body: &mut (dyn FnMut(Option<Frame>) + Send)) {
        let (mut t_start, mut w_start) = (None::<f64>, Instant::now());
        for (fi, (t, rel)) in seq.rgb.iter().enumerate() {
            if fi % a.every != 0 {
                continue;
            }
            let t = *t;
            let Some((p, r)) = seq.pose(t) else { continue };
            if a.live > 0.0 {
                match t_start {
                    None => {
                        t_start = Some(t);
                        w_start = Instant::now();
                    }
                    Some(ts) => {
                        let ahead = (t - ts) / a.live - w_start.elapsed().as_secs_f64();
                        if !a.no_skip && ahead < -0.1 {
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
            let Some((dw, dh, kin)) = seq.kinect_depth(t) else { continue };
            assert_eq!((dw as usize, dh as usize), (w, h));
            let c1 = Instant::now();
            let dets = Some(self.seg.run(&rgb, w, h, a.conf));
            let c2 = Instant::now();
            body(Some(Frame { t, p, r, rgb, w, h, depth: Depth::M(kin), dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), lost: false, held: false, held_live: false, gravity: false, no_map: false, anchor: 0, graph: vec![], t_in: None, t_ready: None }));
        }
    }
}

thread_local! {
    /// 끝난 실행의 입력 통계(버린 장면, 분할 수, 분할 단계 시간): 끝 요약용
    static DRV_STATS: std::cell::Cell<(u32, u32, SegStages)> = std::cell::Cell::new((0, 0, SegStages { pre: 0.0, infer: 0.0, post: 0.0, sub: 0.0, npu: 0.0, n: 0, wall: 0.0 }));
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

fn run(a: &Args, seg: Box<dyn seg::Segmenter + Send>, io: &mut LoopIo) -> Box<dyn seg::Segmenter + Send> {
    // 새 실행(리셋): 받는 스레드가 위치를 바로 낼 회전 · 원점과 마지막 위치를 비움(본 스레드가 첫 장면에서 다시 정함)
    (*POSE_FIX.lock().unwrap(), *POSE_LAST.lock().unwrap()) = (None, None);
    // 스트림 입력이면 첫 장면의 보정값, 카메라 입력이면 빈 시퀀스(보정값만), 아니면 TUM
    let mut stream_rd = a.stream.as_deref().map(stream::StreamReader::open);
    // 리셋을 연달아 누르면 첫 장면보다 리셋 블록이 먼저 올 수 있음 → 이어 읽기로 다시 열어 첫 장면을 기다림
    let stream_first = stream_rd.as_mut().map(|r| loop {
        match r.next() {
            Some(f) => break f,
            None if stream::reset_pending() => *r = stream::StreamReader::open(a.stream.as_deref().unwrap()),
            None => panic!("--stream: 장면이 하나도 오지 않음"),
        }
    });
    let seq = if let Some(f) = &stream_first {
        tum::Tum::empty(f.k)
    } else {
        tum::Tum::open(&a.tum, a.gt.as_deref().map(Path::new))
    };
    std::fs::create_dir_all(a.out.join("objects")).unwrap();
    let [fx, fy, cx, cy] = seq.k;
    let mut grid = Grid::new(0.05);
    // 점 지도는 점 줄이(스트림은 밀리면 건너뛰고, TUM 입력·--no-drop 은 기다려 결과가 늘 같게)
    let mut pw = Some(PtsWorker::new(std::sync::Arc::new(a.clone()), seq.k, a.stream.is_none() || a.no_drop)); // --no-drop: 스트림도 기다림
    let mut mem = Memory::new(0.012, 0.04);
    let mut view = ViewWorker::new(a.out.clone(), a.live > 0.0, io.waker()); // 방 · 엣지 · view.json(방 번호를 실행 내내 잇는다)
    let mut traj: Vec<(f64, [f64; 3])> = vec![];
    let mut hist_h: Vec<f64> = vec![];
    let mut z0: Option<f64> = None;
    let mut clock = Clock::default();
    let mut nkf: u32 = 0;
    let mut cam_json = json!(null);
    let mut last_view = Instant::now();
    let (view_gap, scene_gap) = if out::memory_only() { (0.0, 0.0) } else { (0.4, 1.0) };
    let mut last_scene = Instant::now();
    let mut last_merge = 0u32;
    let mut scene_json = json!(null);
    // 3D 지도 조각: 이번 실행 번호(뷰어가 새 실행이면 받은 조각을 비움), 조각 수, 보낸 점 수
    let mut run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    out::remove_dir_all(&a.out.join("scene")); // 예전 판(옥트리)이 남긴 폴더
    // 물체가 가질 수 있는 이름 전체(object + --vocab 의 이름): 검색이 "기억에 없는 이름"(고양이 → cat)을 가려내는 데 쓴다
    let mut names: Vec<String> = vec!["object".into()];
    if let Some(f) = &a.vocab {
        names.extend(embed::load_vocab(f));
        names.sort();
        names.dedup();
    }
    match &a.structural {
        Some(f) if Path::new(f).exists() => embed::load_structural(f),
        Some(f) => common::info!("--structural {f} 없음: 벽 · 바닥 · 천장 판정 없이"),
        None => {}
    }
    out::write_direct(&a.out.join("vocab.json"), serde_json::to_string(&names).unwrap().into_bytes());
    let mut last_pose = [0.0; 3];
    let t0 = Instant::now();
    let seq = std::sync::Arc::new(seq);
    // 장면 i 가 나올 시각 = pace0 + (t_i − t_첫) / R(R = 밀림 기준 배속). 장면을 끝낸 시각과의 차이 = 밀림
    let (pace0, pace_t0, rate) = (Instant::now(), stream_first.as_ref().map_or(0.0, |f| f.t), a.stream_rate);
    let drop_late = !a.no_drop;
    // 밀림 기준 배속: --stream-rate 없이 받는 스트림(연결 프로그램에서 바로 오는 실시간 입력)은 찍은 시각 그대로(1 배속). --stream-rate 없는 --no-drop 재생은 밀림을 재지 않고 늘 분할(결과가 늘 같게)
    let lag_rate = if rate > 0.0 { rate } else { 1.0 };
    let lag_on = rate > 0.0 || (a.stream.is_some() && drop_late);
    let mut est_traj: Vec<(f64, V3, M3)> = vec![]; // 추정 위치(시각, 위치, 자세) → poses_est.txt(TUM 형식)
    // 수평 합의(웹캠·녹화): 처음 장면들의 수평면 법선(세계 좌표)을 모아 다수결로 '위'를 정하면 세계를 한 번 바로 세운다
    let mut grav_obs: Vec<V3> = vec![];
    // 바닥으로 기울기 고침(장면마다): 위치 그래프의 좌표가 지도 조각마다 기울어 있어도(새 지도 · 임시 위치 · 루프 클로저) 바닥이 수평이 되게
    let mut tilt_c: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // 스트림: RTAB-Map 지도의 z 는 첫 장면 카메라 몸체의 위(IMU 없음)라 처음 든 각도만큼 기울 수 있다 → 바닥 합의로 한 번 바로 세운다
    let mut leveled = a.stream.is_none();
    // 실시간 스트림: 위치는 받는 스레드가 바로 냄(early_pose)
    let early_pose_on = a.stream.is_some() && a.live > 0.0;
    // 실시간: 첫 장면 카메라가 똑바로 서 있다고 보고(위 = 첫 장면 카메라의 −y) 처음부터 수평으로 보여 주고, 바닥 합의가 나오면 한 번 고쳐 다시 쌓는다
    let (mut upright, mut refine) = (a.stream.is_some() && a.live > 0.0, false);
    let mut force_rb = false; // 수평 버튼: 기다리지 않고 바로 다시 쌓기
    let mut stream_fix: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // 스트림 위치에 곱할 수평 맞춤 회전
    let mut stream_held = 0u32; // 맡아 둘 장면으로 받은 수
    let (mut stream_lost, mut map_traj) = (0u32, Vec::<(f64, V3, M3)>::new()); // map_traj: 지도에 넣은 장면의 카메라 위치(지도 좌표) → poses_map.txt
    // 루프 클로저 뒤 다시 쌓기: 지도에 넣은 장면을 기준 노드와 함께 보관, 노드가 움직이면 작업 스레드가 고친 위치로 지도·물체를 처음부터 다시 쌓고
    // 끝나면 본 스레드가 갈아 끼운다(그동안 실시간 장면은 계속 들어옴)
    let mut kfs = kfstore::KfStore::new(a.kf_store_mb);
    let mut last_rebuild = Instant::now();
    // 재생(밀림을 재지 않음: TUM, --stream-rate 없는 --no-drop 스트림)은 다시 쌓기 시작과 갈아 끼우기를 장면 시각으로 정한다:
    // 시작 = 앞 시작에서 장면 시각 --rebuild-every 초 뒤, 갈아 끼우기 = 시작한 장면 시각 + --rebuild-every 초가 된 장면(덜 끝났으면 기다림).
    // 벽시계로 정하면 시작 때의 물체 id · 갈아 끼우는 장면이 실행마다 달라 같은 입력에서 결과가 달랐다. 실시간 입력은 지연이 늘지 않게 벽시계 그대로
    let det_rb = !lag_on;
    let (mut rb_t0, mut rb_due) = (f64::NAN, f64::INFINITY);
    let mut n_rebuild = 0u32;
    let mut prune_objs = false;
    let mut view_due = false; // 다음 장면에서 view.json 쓰기
    let mut n_behind = 0u32; // 밀려서 가볍게 처리한 장면 수
    let mut lag_rb = (0u32, 0f32); // 다시 쌓기가 도는 동안 끝낸 장면 수·그중 밀림 최대
    let (mut write_due, mut scene_due) = (false, false); // 미룬 정기 기록·점 지도 쓰기
    let a_rb = std::sync::Arc::new(a.clone());
    // 다시 쌓기 · 이어 넣기(긴 일: 도는 동안 nice 10). 점 줄을 기다리는 일이라 풀이 아닌 전용 스레드에서(Lane::thread 설명)
    let mut rb_lane = common::rt::Lane::thread("rebuild", (), 10, || {
        let (_, mask) = out::layout(out::Role::Points);
        common::threads::apply(0, mask);
        common::prof::track_thread();
    });
    let mut rb_job: Option<Pending<Rebuilt>> = None;
    let mut rb_cache: Option<std::sync::Arc<RbCache>> = None; // 지난 다시 쌓기의 중간 상태
    let mut lat: Vec<[f32; 4]> = vec![]; // 장면 지연(ms): 받음→분할 끝, 분할 끝→본 스레드가 받음, 본 스레드 처리, 받음→본 스레드 끝
    // 장면 하나를 지도 · 물체 기억에(None = 장면 없이 뒤처리만: 뷰 판 밀어 넣기)
    let mut body = |f: Option<Frame>| {
            let Some(Frame { t, mut p, mut r, rgb, w, h, depth, dets, load, seg: seg_s, lost, held, held_live, gravity, no_map, anchor, graph, t_in, t_ready }) = f else {
                view.tick();
                return;
            };
            // 이 장면의 컬러·깊이: 어느 길로 끝나든(놓침·맡김·지도 밖) 다 쓰면 카메라 입력에 돌려줌
            let (mut rgb, mut depth) = (Recycle(rgb), Recycle(depth));
            let c_top = Instant::now();
            clock.part = [0.0; NPART];
            let view_late = std::mem::take(&mut view_due); // 앞 장면에서 미룬 view.json
            // 밀림: 장면이 나올 시각보다 LAG_DEGRADE 초 넘게 늦으면 가볍게 — 점 격자 일부만, 새 다시 쌓기 미룸, view.json 15 장마다
            let behind = lag_on && pace0.elapsed().as_secs_f64() - (t - pace_t0) / lag_rate > LAG_DEGRADE;
            n_behind += behind as u32;
            let (write_late, scene_late) = (std::mem::take(&mut write_due), std::mem::take(&mut scene_due));
            let mut busy = false; // 이 장면에서 갈아 끼우기 일을 했으면 정기 기록은 다음 장면으로
            kfs.settle(false); // 앞 장면 압축이 끝났으면 예산 맞추기(덜 끝났어도 예산 안이면 기다리지 않음)
            let c_g = Instant::now();
            if !graph.is_empty() {
                kfs.apply_graph(&graph);
            }
            clock.add(P::Graph, c_g);
            let c_in = Instant::now();
            if rb_t0.is_nan() && w > 0 {
                rb_t0 = t;
            }
            if let Some(job) = rb_job.take_if(|j| if det_rb { w > 0 && t >= rb_due } else { j.is_finished() }) {
                let mut rb = job.join();
                let newer = newer_than(&kfs, &stream_fix, z0.unwrap_or(0.0), rb.last_seq);
                if newer.len() > CATCH_UP_MAX {
                    // 쌓는 동안 들어온 장면이 많으면 작업 스레드가 먼저 이어 넣고 다시 옴(그사이 들어온 몇 장만 본 스레드가)
                    rb_due = t + a.rebuild_every;
                    let (ar, k) = (a_rb.clone(), seq.k);
                    rb_job = Some(Pending::on(&mut rb_lane, move || {
                        let c = Instant::now();
                        catch_up(&ar, k, &mut rb, newer);
                        out::slow(c);
                        rb
                    }));
                } else {
                let i1 = Instant::now();
                rb_cache = Some(rb.cache.clone());
                install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
                view.traj_replaced();
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
                    common::info!("갈아 끼우기 나눔: 지도·물체 {:.0} · 출력 {:.0} · 병합 {:.0} ms", (i2 - i1).as_secs_f64() * 1e3, (i3 - i2).as_secs_f64() * 1e3, i3.elapsed().as_secs_f64() * 1e3);
                }
                prune_objs = true; // 옛 id 파일은 다음 기록 때 지움(뷰어가 지금 view.json 을 받기 전 옛 것을 읽는 동안은 남겨 둠)
                }
            }
            clock.add(P::Install, c_in);
            let c_rs = Instant::now();
            if rb_job.is_none() && !behind && kfs.dirty && a.rebuild_every > 0.0 && (force_rb || if det_rb { t - rb_t0 >= a.rebuild_every } else { last_rebuild.elapsed().as_secs_f64() >= a.rebuild_every }) {
                force_rb = false;
                UPS.lock().unwrap().retain(|s, _| kfs.kfs().iter().any(|k| k.seq == *s));
                let snap = snapshot(&kfs, &stream_fix, z0.unwrap_or(0.0));
                // 점 지도는 다시 쌓기를 기다리지 않고 지금 바로 청크 자세만 고침(점유 격자 · 물체만 작업 스레드가 다시 쌓음)
                if let Some(p) = pw.as_mut() {
                    p.repose(&snap);
                }
                let (ar, k, next, moved) = (a_rb.clone(), seq.k, mem.next + 100_000, kfs.moved); // 물체 id: 쌓는 동안 실시간에서 생긴 id 와 겹치지 않게
                kfs.clear_dirty();
                let cache = rb_cache.clone();
                rb_job = Some(Pending::on(&mut rb_lane, move || { let c_rb = Instant::now(); let r = rebuild_maps(&ar, k, &snap, next, moved, cache); out::slow(c_rb); r }));
                last_rebuild = Instant::now();
                (rb_t0, rb_due) = (t, t + a.rebuild_every);
            }
            clock.add(P::RbStart, c_rs);
            let raw = (r, p); // 받은 위치(수평 맞춤 전): 다시 쌓기용 보관
            if lost {
                stream_lost += (w > 0) as u32; // w = 0: 끝에 남은 그래프 노드만 담은 장면
                if w == 0 {
                    return;
                }
                if a.live > 0.0 && POSE_FIX.lock().unwrap().is_none() {
                    // 뷰어: 마지막 위치 그대로, 놓침 표시(손카메라 빨간색)
                    out::write_direct(&a.out.join("pose.json"), out::json_bytes(&json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": false, "wall": wall_ms()})));
                }
                return;
            }
            if held && no_map {
                return; // 위치만 장면은 맡아 두지도 않음
            }
            if held {
                // 맡아 둘 장면: 믿는 지도와 이어지지 않은 새 지도의 위치라 지금 지도에는 넣지 않고 보관만 한다.
                // 기준 노드가 믿는 노드로 바뀌면(루프 클로저로 이어짐) 다시 쌓기가 고친 위치로 넣는다
                stream_held += 1;
                if leveled && kfs.holds(anchor) {
                    let keep = kfs.push_lazy(t, anchor, raw.0, raw.1, cam_height(&hist_h), w, h, dets, true);
                    match (pw.as_mut(), keep) {
                        (Some(pw), Some(kf)) => pw.hold(std::mem::take(&mut *depth), std::mem::take(&mut *rgb), w, h, kf), // 이 장면은 여기서 끝: 복사 없이 넘김(압축은 점 작업 스레드)
                        (None, Some(kf)) => { kf.fill(&rgb, &depth); let _ = kf.pts.set(None); }
                        _ => {}
                    }
                }
                if a.live > 0.0 && !held_live && POSE_FIX.lock().unwrap().is_none() {
                    out::write_direct(&a.out.join("pose.json"), out::json_bytes(&json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": false, "wall": wall_ms()})));
                }
                return;
            }
            // 깊이는 받은 u16 그대로: 읽는 자리마다 같은 식(v × (1 / per_m))으로 미터(장면마다 1.2 MB 미터 칸을 만들지 않음, 같은 비트)
            // 뷰어의 수평 버튼(한 프로그램이면 공유 신호, 따로 띄운 뷰어면 level.req): 지금 카메라 자세를 수평으로 삼는다. 받은 위치에 곱하는 회전을 '카메라의 위 = 세계의 위'가 되게 바꾸고,
            // 보관한 장면을 그 회전으로 다시 쌓아 이미 만든 지도·물체도 함께 바로 세운다(쌓는 중이면 끝난 뒤에)
            if a.stream.is_some() && a.live > 0.0 && rb_job.is_none() && level_requested(a) {
                let up_raw = [-raw.0[0][1], -raw.0[1][1], -raw.0[2][1]]; // 카메라 좌표의 위(−y)를 받은 좌표로
                let fix = rot_between(&up_raw, &[0.0, 0.0, 1.0]);
                let old_t: M3 = [[stream_fix[0][0], stream_fix[1][0], stream_fix[2][0]], [stream_fix[0][1], stream_fix[1][1], stream_fix[2][1]], [stream_fix[0][2], stream_fix[1][2], stream_fix[2][2]]];
                let delta = mm(&fix, &old_t); // 지금까지의 좌표 → 새 좌표
                for e in est_traj.iter_mut() {
                    (e.1, e.2) = (mv(&delta, &e.1), mm(&delta, &e.2));
                }
                let tilt = delta[2][2].clamp(-1.0, 1.0).acos().to_degrees();
                stream_fix = fix;
                leveled = true;
                z0 = Some(mv(&stream_fix, &raw.1)[2] - cam_height(&hist_h)); // 바닥 높이 원점도 새 좌표로
                kfs.dirty = true;
                force_rb = true;
                common::info!("수평 버튼: 지금 카메라 자세를 수평으로 삼음(지도를 {tilt:.1}° 돌려 다시 쌓음)");
            }
            // 위치 쪽이 IMU 로 지도를 중력 수평으로 맞춰 보냄: 받은 그대로 수평(첫 장면 가정 · 바닥 합의 없음, 수평 버튼은 그대로 됨)
            if gravity && !leveled && a.stream.is_some() {
                (leveled, upright, refine) = (true, false, false);
                common::info!("수평: 위치 추적이 IMU 중력으로 맞춘 지도를 그대로 씀");
            }
            if std::mem::take(&mut upright) && !leveled {
                stream_fix = rot_between(&[-raw.0[0][1], -raw.0[1][1], -raw.0[2][1]], &[0.0, 0.0, 1.0]);
                (leveled, refine) = (true, true);
                common::info!("수평: 첫 장면 카메라를 똑바로 선 것으로 보고 시작(바닥 합의가 나오면 고침)");
            }
            if a.stream.is_some() {
                r = mm(&stream_fix, &r);
                p = mv(&stream_fix, &p);
            }
            let c_vo = Instant::now();
            // 수평 합의: 위치가 확인된 장면마다 수평면 법선(세계)을 모으고, 15 장 이상이 10° 안에서 합의하면 세계를 한 번 바로 세운다
            if (!leveled || refine) && a.stream.is_some() {
                if let Some(n) = up_normal(&depth, w, h, seq.k, &r, refine) {
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
                        'lv: {
                        let up = mean(&inl);
                        let fix = rot_between(&up, &[0.0, 0.0, 1.0]);
                        let tilt = up[2].clamp(-1.0, 1.0).acos().to_degrees();
                        if refine && tilt > 70.0 {
                            // 첫 장면이 똑바르다는 가정과 70° 넘게 다르면 믿지 않음(세계를 뒤집지 않음, 천장은 floor_only 로 이미 빠짐 · 녹화 첫 장면은 46° 숙여 있기도 함): 모은 것을 버리고 계속
                            common::info!("수평 맞춤: 바닥 합의가 처음 가정과 {tilt:.1}° 달라 버림");
                            grav_obs.clear();
                            break 'lv;
                        }
                        if std::mem::take(&mut refine) {
                            // 처음 가정(첫 장면이 똑바름)을 바닥 합의로 고침: 지난 위치 · 바닥 높이 원점도 새 좌표로, 지도는 다시 쌓음
                            for e in est_traj.iter_mut() {
                                (e.1, e.2) = (mv(&fix, &e.1), mm(&fix, &e.2));
                            }
                            // 바닥 높이는 기울어진 채 잰 값이라 버리고, 이 장면에서 새 좌표로 다시 잰다(바닥 = z 0)
                            hist_h.clear();
                            z0 = None;
                            (kfs.dirty, force_rb) = (true, true);
                        }
                        r = mm(&fix, &r);
                        p = mv(&fix, &p);
                        stream_fix = mm(&fix, &stream_fix);
                        leveled = true;
                        common::info!("수평 맞춤: 처음 가정이 {:.1}° 기울어져 있었음({} / {} 장 합의)", up[2].clamp(-1.0, 1.0).acos().to_degrees(), inl.len(), grav_obs.len());
                        }
                    } else if grav_obs.len() > 60 {
                        grav_obs.drain(..30); // 합의가 안 되면 오래된 것부터 버리고 계속
                    }
                }
            }
            if a.stream.is_some() && leveled {
                est_traj.push((t, p, r));
                // 바닥으로 기울기 고침: 세 장마다 바닥 면 법선을 재 고침 회전을 반씩 다가가게(잡음은 누름), 자세에 곱함
                r = mm(&tilt_c, &r);
                // IMU 로 중력에 맞춘 지도(GRAVITY)는 위치 쪽이 중력을 지키므로 바닥으로 고치지 않음: 비스듬한 면을 바닥으로 잡으면 고칠수록 더 숙여져
                // 가만히 있어도 지도가 계단처럼 기울며 겹겹이 쌓였다(위치 쪽 지도 기울기 0.0° 인데 받은 쪽 −30°)
                if !gravity && nkf % 3 == 0 && !no_map {
                    if let Some(n) = up_normal(&depth, w, h, seq.k, &r, true) {
                        let n = { let l = norm(&n).max(1e-9); n.map(|x| x / l) };
                        if n[2] > 25f64.to_radians().cos() {
                            let half = { let m = add(&n, &[0.0, 0.0, 1.0]); let l = norm(&m).max(1e-9); m.map(|x| x / l) };
                            let step = rot_between(&n, &half);
                            tilt_c = orthonormal(&mm(&step, &tilt_c)); // 곱이 쌓여도 회전으로 남게
                            r = mm(&step, &r);
                        }
                    }
                }
            }
            let c2 = Instant::now();
            clock.load += load;        clock.seg += seg_s;

            // 바닥 높이: 화면 아래쪽 점이 카메라 아래로 얼마나 떨어졌나(95 %) → 카메라 높이, 짧게 평활
            let back = |u: usize, v: usize| -> Option<V3> {
                let z = depth.at(v * w + u) as f64;
                (z > 0.2 && z < MAX_DEPTH_M).then(|| [(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z])
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
                let k95 = (below.len() as f64 * 0.95) as usize;
                let ch = *below.select_nth_unstable_by(k95, |x, y| x.partial_cmp(y).unwrap()).1; // 정렬 대신 선택(같은 값)
                if (0.5..2.5).contains(&ch) && !no_map {
                    hist_h.push(ch);
                }
            }
            let h_now = cam_height(&hist_h);
            let z0v = if leveled { *z0.get_or_insert(p[2] - h_now) } else { p[2] - h_now };
            // 실시간 스트림: 받는 스레드가 다음 장면부터 이 회전 · 원점으로 위치를 바로 냄(이 장면은 아직 아무도 안 냈으면 아래에서)
            let pose_first = early_pose_on && POSE_FIX.lock().unwrap().replace((stream_fix, z0v)).is_none();
            let pwc = [p[0], p[1], p[2] - z0v];
            let floor_z = pwc[2] - h_now;
            // 위치가 확인된 장면만 지도·물체에 넣는다(실패해 예측 위치뿐인 장면은 지도를 어긋나게 함)
            let map_ok = leveled && !no_map;
            let (c3, c4) = map_frame(a, seq.k, t, &r, &pwc, floor_z, &depth, &rgb, w, h, &dets, map_ok, &mut grid, &mut mem, &mut traj, &mut map_traj);
            clock.part[P::Vo as usize] += (c2 - c_vo).as_secs_f64();
            clock.part[P::Map as usize] += (c3 - c2).as_secs_f64();
            clock.part[P::Obj as usize] += (c4 - c3).as_secs_f64();
            let c_pt = Instant::now();
            // 보관(줄인 보관이어도 점 지도는 점 작업 스레드가 붙이는 점 격자로 다시 쌓음)
            let keep = (map_ok && a.stream.is_some())
                .then(|| kfs.push_lazy(t, anchor, raw.0, raw.1, h_now, w, h, dets, true))
                .flatten();
            // 보관 장면에 이때 고친 카메라 자세의 위(카메라 좌표)를 붙여 둠: 다시 쌓을 때 노드가 움직여도 같은 기울기 고침을 다시 함
            if let Some(kf) = &keep {
                UPS.lock().unwrap().insert(kf.seq, [r[2][0], r[2][1], r[2][2]]); // Rᵀ·ẑ
            }
            if let (Some(p), true) = (pw.as_mut(), map_ok) {
                // 깊이·사진은 이 뒤 본 스레드가 쓰지 않으므로 복사 없이 넘김
                let rgb_pts = std::mem::take(&mut *rgb);
                p.frame(nkf, r, pwc, std::mem::take(&mut *depth), rgb_pts, w, h, keep, behind);
            } else if let Some(kf) = keep {
                kf.fill(&rgb, &depth);
            }
            clock.add(P::Pts, c_pt);
            last_pose = [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])];
            cam_json = json!({"pos": pwc, "R": r});
            nkf += 1;
            // 실시간: 카메라 위치만 담은 작은 pose.json 은 장면마다(스트림은 받는 스레드가 받자마자 냄, 여기서는 첫 장면과 TUM · 카메라 입력만),
            // 지도·물체·그래프(view.json)는 0.4 초마다
            if a.live > 0.0 && (!early_pose_on || pose_first) {
                out::write_direct(&a.out.join("pose.json"), out::json_bytes(&json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": true, "wall": wall_ms()})));
            }
            // 메모리 기억 폴더(--serve): 받은 장면마다 바로 뷰어에(view 판은 뷰 스레드가 짜고 밀리면 마지막 것만, 점 노드는 점 작업 스레드가 한가할 때마다). 디스크 기록이면 0.4 · 1 초 간격
            let write_now = write_late || if a.live > 0.0 { last_view.elapsed().as_secs_f64() >= view_gap || nkf == 1 } else { nkf % (if behind { 15 } else { 5 }) == 0 || nkf == 1 };
            // 3D 점 지도: 실시간은 1 초마다(전에는 장면 번호가 기록 시점과 우연히 맞을 때만 나가 몇 초씩 비었다), 아니면 장면 수로
            let scene_now = write_now && (scene_late || if a.live > 0.0 { last_scene.elapsed().as_secs_f64() >= scene_gap && pw.as_ref().is_none_or(|p| p.idle()) } else { nkf % 50 == 0 || nkf % a.scene_every == 1 || a.scene_every == 1 });
            // 갈아 끼우기 일을 한 장면이면 정기 기록을 다음 장면으로
            let write_now = if busy && write_now {
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
                if scene_now || nkf == 1 {
                    last_scene = Instant::now();
                    scene_done = true;
                    // 새로 확정된 칸만 조각 파일로: 뷰어는 아직 안 받은 조각만 받는다(원격 접속에서 지도 전체를 매번 받지 않게)
                    scene_json = json!({"run": run_id, "voxel": a.scene_vox});
                    if let Some(p) = pw.as_mut() {
                        p.write();
                        scene_json["submap"] = p.json();
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
                if scene_done && nkf > 1 && !(out::memory_only() && a.live > 0.0) { // 메모리 기억 폴더는 같은 장면에서(미루면 실시간에선 장면마다 다시 미뤄져 판이 안 나감)
                    if merge {
                        mem.prune(t);
                        mem.merge_pass();
                    }
                    view_due = true;
                } else {
                    write_view(&a.out, &grid, &mut mem, &mut view, &traj, t, last_pose, &cam_json, &scene_json, merge, VIEW_REFRESH);
                }
                clock.add(if merge { P::Merge } else { P::View }, c_v);
                let c_p = Instant::now();
                if !view_late && !view_due && std::mem::take(&mut prune_objs) {
                    prune_object_files(&a.out, &mem);
                }
                clock.add(P::Prune, c_p);
            }
            let c5 = Instant::now();
            clock.vo += (c2 - c_vo).as_secs_f64();
            clock.map += (c3 - c2).as_secs_f64();
            clock.obj += (c4 - c3).as_secs_f64();
            clock.write += (c5 - c4).as_secs_f64();
            clock.n += 1;
            out::slow(c_top);
            clock.end_frame(c_top.elapsed().as_secs_f32() * 1e3, nkf);
            if let (Some(ti), Some(tr)) = (t_in, t_ready) {
                let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f32() * 1e3;
                let now = Instant::now();
                lat.push([ms(ti, tr), ms(tr, c_top), ms(c_top, now), ms(ti, now)]);
            }
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
                println!("kf {nkf} objects {} ({} gone) | ms/kf wait {:.0} seg {:.0} vo {:.0} map {:.0} obj {:.0} write {:.0} | {:.1} kf/s | pos ({:.2},{:.2},{:.2})",
                         mem.objs.len(), mem.objs.iter().filter(|o| o.state == "gone").count(),
                         clock.load / n * 1e3, clock.seg / n * 1e3, clock.vo / n * 1e3, clock.map / n * 1e3, clock.obj / n * 1e3, clock.write / n * 1e3,
                         n / t0.elapsed().as_secs_f64(), p[0], p[1], p[2]);
                // 메모리: 프로세스 RSS · 메모리 기억 폴더(파일 수 · MB) · 보관 장면(MB)
                let rss = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmRSS:")).and_then(|v| v.trim().trim_end_matches(" kB").trim().parse::<f64>().ok())).unwrap_or(0.0) / 1024.0;
                let (mf, mb) = sgview::memfs::usage();
                println!("mem kf {nkf} t {:.1} rss_mb {rss:.0} memfs_files {mf} memfs_mb {:.1} kfstore_mb {:.1}", t, mb as f64 / 1048576.0, kfs.bytes() as f64 / 1048576.0);
            }
        };
    // ---------------------------------------------------------------- 처리 루프
    // 입력(스트림 · TUM) → 분할(NPU 비동기 · 동기) → 장면 차례대로 물체 기억. 스레드 넘김 없이 한 루프에서, 무거운 뒤 일은 줄(점 · 뷰 · 기록 · 사진 벡터 · 다시 쌓기)이 작업 풀에서
    let drv = Driver::new(a, seg, stream_rd, stream_first, pace0, pace_t0, lag_on, io);
    let seg = drv.run(a, &seq, io, &mut body);
    drop(body);
    let (drv_dropped, drv_seg, drv_stats) = DRV_STATS.with(|d| d.get());
    // 끝: 마지막 다시 쌓기 뒤에 노드가 또 움직였으면 끝난 그래프로 한 번 더(결과 지도·물체·poses_map.txt 가 최종 위치를 따름)
    let mut installed = false;
    if let Some(job) = rb_job.take() {
        let rb = job.join();
        rb_cache = Some(rb.cache.clone());
        install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
        view.traj_replaced();
        (n_rebuild, installed) = (n_rebuild + 1, true);
    }
    kfs.settle(true);
    if kfs.dirty {
        let snap = snapshot(&kfs, &stream_fix, z0.unwrap_or(0.0));
        let rb = rebuild_maps(a, seq.k, &snap, mem.next + 100_000, kfs.moved, rb_cache.clone());
        kfs.clear_dirty();
        install(a, seq.k, rb, &kfs, &stream_fix, z0.unwrap_or(0.0), &mut grid, pw.as_mut(), &mut mem, &mut traj, &mut map_traj);
        view.traj_replaced();
        (n_rebuild, installed) = (n_rebuild + 1, true);
    }
    if installed {
        fresh_outputs(&mut run_id, a.scene_vox, None); // scene_json 은 아래 마지막 기록에서 다시 만듦
    }
    if a.stream.is_some() {
        common::info!("다시 쌓기 {n_rebuild} 번, 보관 장면 {} 개({:.0} MB)", kfs.len(), kfs.bytes() as f64 / 1048576.0);
    }
    common::info!("{}", out::stats::report());
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
    scene_json = json!({"run": run_id, "voxel": a.scene_vox});
    // 점 지도: 남은 노드를 모두 쓰고 점 작업 스레드를 끝냄
    let (n_pts, scene_mb, pts_dropped) = match pw.as_mut() {
        Some(p) => {
            let (n, mb, j) = p.finish();
            scene_json["submap"] = j;
            (n, mb, (p.dropped, p.skipped))
        }
        None => (0, 0.0, (0, 0)),
    };
    let t_end = traj.last().map_or(0.0, |x| x.0);
    for o in mem.objs.iter_mut().filter(|o| o.dirty) {
        o.cells_saved = 0; // 마지막 기록은 바뀐 물체를 모두 내보냄
    }
    embed::drain(); // 마지막 병합이 모든 물체의 사진 벡터를 보게
    write_view(&a.out, &grid, &mut mem, &mut view, &traj, t_end, last_pose, &cam_json, &scene_json, true, usize::MAX);
    if a.vocab.is_some() {
        // 방금 기록이 맡긴 사진의 벡터가 끝난 뒤 그 이름을 view.json 에 넣음
        embed::drain();
        write_view(&a.out, &grid, &mut mem, &mut view, &traj, t_end, last_pose, &cam_json, &scene_json, true, usize::MAX);
    }
    if n_rebuild > 0 {
        prune_object_files(&a.out, &mem);
    }
    view.flush();
    out::flush_writer();
    if std::env::var("ONDEVICE_NAME_DEBUG").is_ok() {
        for o in mem.objs.iter().filter(|o| o.state != "gone") {
            embed::debug_name(o.id);
        }
    }
    // 리셋으로 끝난 실행이면 사진 벡터 작업 스레드는 다음 실행이 이어 씀(끄면 리셋 뒤 물체 이름이 안 붙음), id 별 이름·벡터 합만 비움
    if stream::reset_pending() {
        embed::drain();
        embed::clear();
    } else {
        embed::finish();
    }
    let n = clock.n.max(1) as f64;
    // 물체 이름별 수: 화면에 쓰는 이름(사진으로 붙인 이름, 없으면 분할 이름 object)
    let mut by: std::collections::BTreeMap<String, u32> = Default::default();
    for o in &mem.objs {
        let n = embed::name(o.id, o.name != "object").map_or_else(|| o.name.clone(), |l| l.0);
        if embed::structure(o.id) != Some("ceiling") {
            *by.entry(n).or_default() += 1;
        }
    }
    let mut summary = json!({"keyframes": nkf, "objects": mem.objs.len(), "gone": mem.objs.iter().filter(|o| o.state == "gone").count(),
        "scene_points": n_pts, "scene_mb": scene_mb, "pts_dropped": pts_dropped.0, "pts_near_skipped": pts_dropped.1, "ms_per_kf": {"load": clock.load / n * 1e3, "seg": clock.seg / n * 1e3, "vo": clock.vo / n * 1e3, "map": clock.map / n * 1e3,
        "obj": clock.obj / n * 1e3, "write": clock.write / n * 1e3}, "kf_per_s": n / t0.elapsed().as_secs_f64(), "by_name": by, "merged": mem.merged});
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
    summary["dropped_late"] = json!(drv_dropped);
    summary["seg_runs"] = json!(drv_seg);
    {
        let s = drv_stats;
        if s.n > 0 {
            let m = 1e3 / s.n as f64;
            summary["seg_stage_ms"] = json!({ "pre": s.pre * m, "infer": s.infer * m, "post": s.post * m, "submit": s.sub * m, "npu_take": s.npu * m });
            summary["seg_per_s"] = json!(s.n as f64 / s.wall.max(1e-9));
        }
    }
    summary["loop"] = io.json();
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
    if !lat.is_empty() {
        let col = |k: usize| { let mut v: Vec<f32> = lat.iter().map(|x| x[k]).collect(); common::prof::pct(&mut v) };
        summary["scene_ms"] = json!({"n": lat.len(), "seg": col(0), "queue": col(1), "main": col(2), "total": col(3)});
    }
    summary["copies"] = common::prof::copies().into_iter().filter(|c| c.1 > 0).map(|(n, k, b)| (n.to_string(), json!({"n": k, "mb": b as f64 / 1048576.0}))).collect::<serde_json::Map<_, _>>().into();
    summary["work"] = common::prof::times().into_iter().filter(|c| c.1 > 0).map(|(n, k, s)| (n.to_string(), json!({"n": k, "s": s}))).collect::<serde_json::Map<_, _>>().into();
    summary["waits"] = common::prof::waits().into_iter().filter(|c| c.1 > 0).map(|(n, k, s)| (n.to_string(), json!({"n": k, "s": s}))).collect::<serde_json::Map<_, _>>().into();
    summary["viewer_wakes"] = json!(sgview::memfs::bumps()); // 뷰어 /stream 을 깨운 횟수(파일 걸기 · view 판)
    let (ths, live) = common::prof::threads();
    summary["threads"] = json!({"live": live, "by_name": ths.iter().map(|t| json!({"name": t.name, "n": t.n, "cpu_s": t.cpu_s, "wakeups": t.wakeups, "preempted": t.preempted})).collect::<Vec<_>>()});
    println!("{summary}");
    seg
}



/// 다시 쌓은 지도 한 벌(작업 스레드에서 만들어 본 스레드가 갈아 끼움)
struct Rebuilt {
    grid: Grid,
    mem: Memory,
    traj: Vec<(f64, [f64; 3])>,
    map_traj: Vec<(f64, V3, M3)>,
    last_seq: u64, // 이 장면까지 넣음(그 뒤 보관한 장면은 갈아 끼울 때 이어서 넣음)
    extra: Snap, // 작업 스레드에서 이미 이어 넣은 장면(쌓는 동안 새로 보관한 것): 점 지도에는 갈아 끼울 때 보냄
    n: usize,
    secs: f64,
    moved: (f64, f64),
    /// 이번에 쌓을 때 쓴 장면·위치와 중간 상태: 다음 다시 쌓기가 바뀌지 않은 앞부분을 건너뜀
    cache: std::sync::Arc<RbCache>,
    /// 이번에 처음부터 다시 넣은 첫 장면 번호(0 = 처음부터)
    start: usize,
}

/// 다시 쌓기 중간 상태: i 번째 장면을 넣기 직전의 지도·물체·길(장면 순서·위치가 같으면 그대로 이어 쌓아도 처음부터 쌓은 것과 같다)
struct Ckpt {
    i: usize,
    grid: Grid,
    mem: Memory,
    traj: Vec<(f64, [f64; 3])>,
    map_traj: Vec<(f64, V3, M3)>,
}

/// 지난 다시 쌓기: 장면마다 (순번, 회전, 위치)와 중간 상태(CKPT_EVERY 장마다, 최근 CKPT_KEEP 개)
#[derive(Default)]
struct RbCache {
    keys: Vec<(u64, M3, V3)>,
    ckpts: Vec<std::sync::Arc<Ckpt>>,
}
/// 중간 상태 간격(물체 정리 주기 15 의 배수라 그 자리에서 이어도 같은 결과)
const CKPT_EVERY: usize = 120;
const CKPT_KEEP: usize = 6;

type Snap = Vec<(std::sync::Arc<kfstore::Kf>, M3, V3)>;

/// 보관한 장면과 지금 노드 위치로 만든 위치(수평 맞춤·바닥 높이 원점 적용, 지도 좌표)
/// 카메라 높이(바닥 위): 최근 15 번 잰 값의 가운데 값
fn cam_height(hist: &[f64]) -> f64 {
    if hist.is_empty() {
        return 1.4;
    }
    let mut s: Vec<f64> = hist.iter().rev().take(15).copied().collect();
    let mid = s.len() / 2;
    *s.select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap()).1
}

/// 거의 회전인 행렬을 회전으로(그람-슈미트: 행 셋을 서로 수직인 길이 1 로)
fn orthonormal(m: &M3) -> M3 {
    let unit = |v: V3| { let l = norm(&v).max(1e-12); v.map(|x| x / l) };
    let a = unit(m[0]);
    let b = unit(sub(&m[1], &a.map(|x| x * dot(&a, &m[1]))));
    [a, b, cross(&a, &b)]
}

/// 보관 장면별 '그때 고친 카메라 자세의 위'(카메라 좌표, Rᵀ·ẑ): 키 = 보관 순번
/// 실시간 위치 바로 내기(스트림): 본 스레드가 정한 받은 좌표 → 화면 좌표(수평 회전 · 바닥 높이 원점). 정해지기 전(첫 장면)과 TUM · 카메라 입력은 본 스레드가 씀
static POSE_FIX: std::sync::Mutex<Option<(M3, f64)>> = std::sync::Mutex::new(None);
/// 받는 스레드가 마지막으로 낸 위치(놓친 장면 · 맡아 둔 장면은 그 자리에 놓침 표시로)
static POSE_LAST: std::sync::Mutex<Option<(serde_json::Value, [f64; 3])>> = std::sync::Mutex::new(None);

/// 받은 장면의 카메라 위치를 pose.json 으로 바로(분할 · 점 지도 · 물체 일을 기다리지 않음, 기록 스레드 줄도 거치지 않음).
/// 본 스레드가 쓰던 값과 같은 식: 위치 = 수평 회전 · 받은 위치 − 바닥 높이 원점, 방향 = 수평 회전 · 받은 회전
fn early_pose(pp: &Path, sf: &stream::StreamFrame) {
    let Some((fix, z0v)) = *POSE_FIX.lock().unwrap() else { return };
    let mut last = POSE_LAST.lock().unwrap();
    let ok = !sf.lost() && (!sf.held() || sf.held_live());
    if ok {
        let (r, p) = (mm(&fix, &sf.r), mv(&fix, &sf.p));
        let pwc = [p[0], p[1], p[2] - z0v];
        *last = Some((json!({"pos": pwc, "R": r}), [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])]));
    }
    let Some((cam, pose)) = last.as_ref() else { return };
    out::write_direct(pp, out::json_bytes(&json!({"t": sf.t, "cam": cam, "pose": pose, "ok": ok, "wall": wall_ms()})));
}

/// 장면 사이 위치(IMU 회전, 200 Hz)를 pose.json 으로: early_pose 와 같은 좌표 바꾸기. 같은 프로세스 뷰어가 메모리에서 읽으므로 디스크에는 쓰지 않음
fn fast_pose(pp: &Path, t: f64, ts: f64, twc: &slam::register::Tf) {
    let Some((fix, z0v)) = *POSE_FIX.lock().unwrap() else { return };
    if !sgview::memfs::enabled() {
        return;
    }
    let a = twc.0.map(|v| v as f64);
    let r0: M3 = [[a[0], a[1], a[2]], [a[4], a[5], a[6]], [a[8], a[9], a[10]]];
    let (r, p) = (mm(&fix, &r0), mv(&fix, &[a[3], a[7], a[11]]));
    let pwc = [p[0], p[1], p[2] - z0v];
    let pose = [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])];
    sgview::memfs::publish(pp, std::sync::Arc::new(out::json_bytes(&json!({"t": t, "cam": {"pos": pwc, "R": r}, "pose": pose, "ok": true, "ts": ts, "wall": wall_ms()}))));
}

static UPS: std::sync::Mutex<std::collections::BTreeMap<u64, V3>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn snapshot(kfs: &kfstore::KfStore, fix: &M3, z0: f64) -> Snap {
    let ups = UPS.lock().unwrap();
    kfs.kfs()
        .iter()
        .filter(|kf| kfs.usable(kf))
        .map(|kf| {
            let (r0, p0) = kfs.pose(kf);
            let p = mv(fix, &p0);
            let r = mm(fix, &r0);
            // 그 장면에서 바닥으로 고친 기울기를 지금 노드 자세에 다시: 카메라의 '위'가 세계 위가 되게
            match ups.get(&kf.seq) {
                Some(uc) => {
                    let u = mv(&r, uc);
                    let u = { let l = norm(&u).max(1e-9); u.map(|x| x / l) };
                    let c = if u[2] > 25f64.to_radians().cos() { rot_between(&u, &[0.0, 0.0, 1.0]) } else { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
                    (kf.clone(), mm(&c, &r), [p[0], p[1], p[2] - z0])
                }
                None => (kf.clone(), r, [p[0], p[1], p[2] - z0]),
            }
        })
        .collect()
}

/// 보관한 장면을 주어진 위치로 빈 지도·물체·지나온 길에 처음부터 쌓는다. 물체 id 는 next_id 부터.
/// 지난번(cache)과 앞부분 장면·위치가 같으면 그 끝 가까운 중간 상태에서 이어 쌓는다(루프 클로저로 뒤쪽 새 지도만 움직였을 때 그 부분만)
fn rebuild_maps(a: &Args, k: [f64; 4], snap: &Snap, next_id: u32, moved: (f64, f64), cache: Option<std::sync::Arc<RbCache>>) -> Rebuilt {
    let c0 = Instant::now();
    let mut grid = Grid::new(0.05);
    let mut mem = Memory::new(0.012, 0.04);
    mem.next = next_id;
    let (mut traj, mut map_traj) = (vec![], vec![]);
    let mut ckpts: Vec<std::sync::Arc<Ckpt>> = vec![];
    let mut start = 0usize;
    let mut old_keys: Vec<(u64, M3, V3)> = vec![];
    if let Some(c) = cache {
        // 처음 달라진 장면: 순번이나 위치가 다름(같은 위치만 이어 쌓음: 처음부터 쌓은 것과 같다)
        let d = snap
            .iter()
            .zip(c.keys.iter())
            .position(|((kf, r, p), (s, r0, p0))| kf.seq != *s || r != r0 || p != p0)
            .unwrap_or(snap.len().min(c.keys.len()));
        if std::env::var_os("MEMMAP_PARTS").is_some() {
            common::info!("다시 쌓기: 지난번과 처음 달라진 장면 {d} / {}", snap.len());
        }
        if let Some(ck) = c.ckpts.iter().rev().find(|ck| ck.i <= d && ck.i > 0) {
            (grid, mem, traj, map_traj) = (ck.grid.clone(), ck.mem.clone(), ck.traj.clone(), ck.map_traj.clone());
            start = ck.i;
            ckpts = c.ckpts.iter().filter(|x| x.i <= start).cloned().collect();
            old_keys = c.keys[..start].to_vec(); // 이어 받은 앞부분은 그때 위치로 쌓은 것
        }
    }
    let (mut t_dec, mut t_map, mut t_obj) = (0.0f64, 0.0f64, 0.0f64);
    // 보관 장면 풀기(깊이·컬러·검출)는 장면끼리 상관없어 몇 장씩 여러 코어로 미리 풀고, 쌓기는 차례대로(결과 같음).
    // 한 번에 푸는 장 수 = 풀의 스레드 수(최대 8): 푼 장면 하나가 약 2.1 MB(깊이 f32 + 컬러)라 32 장씩이면 최대 메모리가 67 MB 늘었다.
    // 쌓기가 차례대로라 8 장 넘게 미리 풀어도 빨라지지 않는다
    let dec_chunk = rayon::current_num_threads().clamp(1, 8);
    let mut dec: std::collections::VecDeque<(Depth, Vec<u8>, Option<Vec<seg::Det>>)> = Default::default();
    // 시험용 MEMMAP_CKPT_EVERY(15 의 배수)
    let ck_every = std::env::var("MEMMAP_CKPT_EVERY").ok().and_then(|v| v.parse::<usize>().ok()).filter(|v| *v > 0 && v % 15 == 0).unwrap_or(CKPT_EVERY);
    // 점 지도는 다시 쌓지 않는다(점 줄이 청크 자세만 고침): 점유 격자 · 물체만
    for (i, (kf, r, pwc)) in snap.iter().enumerate().skip(start) {
        if i > start && i % ck_every == 0 {
            ckpts.push(std::sync::Arc::new(Ckpt { i, grid: grid.clone(), mem: mem.clone(), traj: traj.clone(), map_traj: map_traj.clone() }));
        }
        let cd = Instant::now();
        if dec.is_empty() {
            use rayon::prelude::*;
            let end = (i + dec_chunk).min(snap.len());
            // 압축이 다 채워진 뒤 풀에 들어감(점 작업 스레드와 풀을 서로 기다리는 교착 막기, unpack_all 과 같음)
            for (kf, _, _) in &snap[i..end] {
                kf.wait_ready();
            }
            dec = snap[i..end].par_iter().map(|(kf, _, _)| (Depth::M(kf.depth()), kf.rgb(), kf.dets())).collect::<Vec<_>>().into();
        }
        let (kd, kr, kdets) = dec.pop_front().unwrap();
        t_dec += cd.elapsed().as_secs_f64();
        let (c3, c4) = map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &kd, &kr, kf.w, kf.h, &kdets, true, &mut grid, &mut mem, &mut traj, &mut map_traj);
        t_map += (c3 - cd).as_secs_f64();
        t_obj += (c4 - c3).as_secs_f64();
        if (i + 1) % 15 == 0 {
            mem.prune(kf.t);
            mem.merge_pass();
        }
    }
    while ckpts.len() > CKPT_KEEP {
        ckpts.remove(0);
    }
    let last_seq = snap.last().map_or(0, |s| s.0.seq);
    let secs = c0.elapsed().as_secs_f64();
    common::info!("다시 쌓기 나눔: 장면 {start} 번부터 {} 개(앞은 저장해 둔 상태) · 보관 장면 풀기 {t_dec:.2} 초 + 점유 격자 {:.2} 초 + 물체 {t_obj:.2} 초, 합 {secs:.2} 초(병합 포함)", snap.len() - start, t_map);
    old_keys.extend(snap[start..].iter().map(|(kf, r, p)| (kf.seq, *r, *p)));
    let cache = std::sync::Arc::new(RbCache { keys: old_keys, ckpts });
    Rebuilt { grid, mem, traj, map_traj, last_seq, extra: vec![], n: snap.len(), secs, moved, cache, start }
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
        pw.repose(&snapshot(kfs, fix, z0)); // 지금 보관 장면 위치로 청크 자세만(다시 쌓기 시작 때 한 번 고쳤고 그 뒤 움직임까지)
    }
    for ((kf, r, pwc), (d, rgb, dets)) in newer.iter().zip(unpack_all(&newer)) {
        map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &d, &rgb, kf.w, kf.h, &dets, true, grid, mem, traj, map_traj);
    }
    common::info!("다시 쌓기: 노드 최대 {:.1} cm·{:.1}° 움직임 → 장면 {} 개(그중 {} 번부터 다시) {:.2} 초(작업 스레드), 그동안 들어온 장면 {} 개 이어 넣음(작업 스레드 + 본 스레드), 물체 {} 개(id 이어받음 {kept})",
              rb.moved.0 * 100.0, rb.moved.1, rb.n, rb.start, rb.secs, format!("{n_extra} + {}", newer.len()), mem.objs.len());
}

/// 밀릴 때 분할 생략: 스트림이 SEG_SKIP_LAG 초 넘게 밀렸고 직전 분할 키프레임에서 SEG_SKIP_M 미터 · SEG_SKIP_DEG 도 안쪽이면 생략(SEG_SKIP_N 장째는 꼭 분할)
const SEG_SKIP_M: f64 = 0.10;
const SEG_SKIP_DEG: f64 = 10.0;
const SEG_SKIP_N: usize = 4;
const SEG_SKIP_LAG: f64 = 0.1;
/// 스트림이 이만큼(초) 밀리면 가볍게 처리
const LAG_DEGRADE: f64 = 0.5;
/// 스트림 장면이 찍은 시각보다 이만큼(초) 늦으면 버림(--no-drop 이 아니면)
const DROP_LATE: f64 = 0.2;
/// view.json 한 번에 점 파일을 다시 만들 물체 수 상한(다시 쌓은 뒤 물체가 모두 바뀌어도 한 장면에 몰리지 않게, 나머지는 다음 기록에)
const VIEW_REFRESH: usize = 8;
/// 점 작업 스레드에 밀린 장면이 이만큼이면 스트림 장면은 점 지도를 건너뜀(본 스레드는 기다리지 않음)
const PTS_QUEUE: usize = 6;
/// 이만큼(거리 m, 각도 °) 안 움직인 장면은 점 격자 일부(1/PTS_SUB²)만 넣음
const PTS_GAP_M: f64 = 0.04;
const PTS_GAP_DEG: f64 = 4.0;
/// 가까운 자리 장면은 점 격자 간격을 이만큼 곱해(점 1/PTS_SUB²) 어긋남을 돌려 가며 넣음
const PTS_SUB: usize = 2;


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
fn pts_grid(a: &Args, k: [f64; 4], depth: &Depth, rgb: &[u8], w: usize, h: usize, part: Option<usize>) -> PtsGrid {
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
    let ax: Vec<f64> = (0..w).map(|x| (x as f64 - cx) / fx).collect();
    let ay: Vec<f64> = (0..h).map(|y| (y as f64 - cy) / fy).collect();
    // 격자 줄 G 개씩 한 일감. 깊이는 쓰는 줄만 미터로: 스트림 u16 은 일감마다 필요한 줄을 한 번 바꿔 담고(창이 겹치는 간격 < 5 면 줄 묶음 통째,
    // 아니면 격자 줄마다 v−2 … v+2 다섯 줄), f32 면 그 자리 조각(복사 없음). 창 안에서 가운데 줄 = 2
    const G: usize = 8;
    out::par(|| d.par_chunks_mut(gw * G).zip(c.par_chunks_mut(gw * 3 * G)).enumerate().for_each_init(|| (Vec::new(), Vec::new()), |(b1, b2), (ci, (dc, cc))| {
        // 가장자리 2 화소 안은 facing 이 늘 거짓 → 창(5×5)이 온전한 줄(2 ≤ v < h−2)만, 이웃 창을 범위 검사 없이 읽는 안쪽 판만 부름
        let (mut lo, mut hi) = (ci * G, ci * G + dc.len() / gw);
        while lo < hi && off[1] + lo * step < 2 {
            lo += 1;
        }
        while hi > lo && off[1] + (hi - 1) * step + 2 >= h {
            hi -= 1;
        }
        if lo == hi {
            return;
        }
        let span = step < 5;
        let ylo = off[1] + lo * step - 2;
        let blk: &[f32] = if span { depth.rows(ylo, off[1] + (hi - 1) * step + 3 - ylo, w, b1) } else { &[] };
        for gy in lo..hi {
            let v = off[1] + gy * step;
            let win: &[f32] = if span { &blk[(v - 2 - ylo) * w..(v + 3 - ylo) * w] } else { depth.rows(v - 2, 5, w, b2) };
            let (dr, cr) = (&mut dc[(gy - ci * G) * gw..][..gw], &mut cc[(gy - ci * G) * gw * 3..][..gw * 3]);
            let ayw = &ay[v - 2..v + 3];
            for gx in 0..gw {
                let u = off[0] + gx * step;
                let z = win[2 * w + u] as f64;
                if !(z > 0.2 && z < MAX_DEPTH_M) {
                    continue;
                }
                if u < 2 || u + 2 >= w {
                    continue;
                }
                let pc = [ax[u] * z, ay[v] * z, z];
                if pc[2] < maxd && !depth_edge_in(win, w, u, 2) && facing(win, w, 5, &ax, ayw, u, 2, &pc) {
                    dr[gx] = smooth_depth(win, w, u, 2);
                    let o = (v * w + u) * 3;
                    cr[gx * 3..gx * 3 + 3].copy_from_slice(&rgb[o..o + 3]);
                }
            }
        }
    }));
    PtsGrid { step, off, gw, gh, d, rgb: c }
}

/// 5×5 이웃 중 가운데와 Kinect 잡음 폭(3 mm + 0.4 %·z²) 안인 깊이의 평균(가장자리 2 화소 안쪽만 불림: facing 이 보장)
fn smooth_depth(d: &[f32], w: usize, u: usize, v: usize) -> f32 {
    let z0 = d[v * w + u];
    let tol = 0.003 + 0.004 * z0 * z0;
    // 분기 없이: 빠지는 이웃은 +0 을 더함(s 는 늘 +0 이상이라 s + 0 = s, 더하는 차례 그대로 → 같은 값)
    let (mut s, mut n) = (0f32, 0u32);
    for y in v - 2..=v + 2 {
        let r: &[f32; 5] = d[y * w + u - 2..y * w + u + 3].try_into().unwrap();
        for &z in r {
            let ok = (z - z0).abs() < tol;
            s += if ok { z } else { 0.0 };
            n += ok as u32;
        }
    }
    s / n as f32
}

/// 천장 높이: 카메라보다 CEIL_MIN_M 넘게 위의 높이 칸(CEIL_BIN_M) 하나에 장면 점이 몰리면(넓은 수평 면) 그 높이를 후보로,
/// CEIL_HITS 장면 연속 같은 높이(± CEIL_BIN_M)면 천장으로 정한다. 정한 뒤로는 천장 − CEIL_CUT_M 위 점은 지도에 넣지 않는다.
/// 카메라가 정한 때보다 CEIL_FLOOR_M 넘게 오르내리면(층이 바뀜) 다시 찾는다. 지도 z 는 위쪽(바닥 맞춤 · IMU 중력)
/// 카메라 깊이를 이 거리(m)까지만 씀(D435 깊이 오차 약 6.7 cm @ 4 m)
const MAX_DEPTH_M: f64 = 4.0;
/// 정한 천장 자르는 높이(f64 비트, 못 정했으면 무한): 점 줄이 쓰고 물체 쪽이 읽음
static CEIL_TOP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x7FF0_0000_0000_0000);
fn ceil_top() -> f64 {
    f64::from_bits(CEIL_TOP.load(std::sync::atomic::Ordering::Relaxed))
}
const CEIL_MIN_M: f64 = 0.4;
const CEIL_BIN_M: f64 = 0.05;
const CEIL_HITS: u32 = 3;
const CEIL_CUT_M: f64 = 0.1;
const CEIL_FLOOR_M: f64 = 1.5;

#[derive(Default)]
struct Ceil {
    z: Option<(f64, f64)>,   // (천장 높이, 정한 때 카메라 높이)
    cand: Option<(f64, u32)>, // (후보 높이, 연속 장면 수)
}

impl Ceil {
    /// 이 장면 점(지도 좌표 높이)으로 천장을 찾거나 확인한다
    fn observe(&mut self, cam_z: f64, zs: &[f64]) -> bool {
        if let Some((_, cz)) = self.z {
            if (cam_z - cz).abs() > CEIL_FLOOR_M {
                common::info!("천장: 카메라 높이가 {:.2} m 바뀜 → 다시 찾음", cam_z - cz);
                *self = Ceil::default();
                CEIL_TOP.store(f64::INFINITY.to_bits(), std::sync::atomic::Ordering::Relaxed);
            } else {
                return false;
            }
        }
        let lo = cam_z + CEIL_MIN_M;
        let mut bins: HashMap<i64, u32> = HashMap::new();
        for &z in zs.iter().filter(|&&z| z > lo) {
            *bins.entry((z / CEIL_BIN_M).floor() as i64).or_default() += 1;
        }
        // 가장 낮은 '몰린' 칸(이웃 칸 합이 장면 점의 4 % 넘고 200 개 넘음)
        let need = ((zs.len() as f64 * 0.04) as u32).max(200);
        let mut keys: Vec<i64> = bins.keys().copied().collect();
        keys.sort_unstable();
        let hit = keys.into_iter().find(|b| bins[b] + bins.get(&(b + 1)).copied().unwrap_or(0) >= need).map(|b| b as f64 * CEIL_BIN_M);
        self.cand = match (hit, self.cand) {
            (Some(h), Some((c, n))) if (h - c).abs() <= CEIL_BIN_M => Some((c.min(h), n + 1)),
            (Some(h), _) => Some((h, 1)),
            (None, c) => c,
        };
        if let Some((c, n)) = self.cand {
            if n >= CEIL_HITS {
                common::info!("천장 정함: 지도 높이 {:.2} m (카메라 + {:.2} m), 위 {:.2} m 부터 자름", c, c - cam_z, c - CEIL_CUT_M);
                self.z = Some((c, cam_z));
                CEIL_TOP.store((c - CEIL_CUT_M).to_bits(), std::sync::atomic::Ordering::Relaxed);
                return true;
            }
        }
        false
    }
    /// 이 높이보다 위면 버림(천장을 아직 못 정했으면 무한)
    fn cut(&self) -> f64 {
        self.z.map_or(f64::INFINITY, |(c, _)| c - CEIL_CUT_M)
    }
}

/// 점 격자 → 청크 점 지도(카메라 좌표 그대로 넘기면 청크가 자기 기준 자세 좌표로 바꿔 담음). 천장 위 점은 빼고 넣는다
#[allow(clippy::too_many_arguments)]
fn chunk_points(k: [f64; 4], g: &PtsGrid, r: &M3, pwc: &V3, cm: &mut chunks::ChunkMap, kf: u32, seq: Option<u64>, ceil: &mut Ceil) {
    let [fx, fy, cx, cy] = k;
    let ax: Vec<f64> = (0..g.gw).map(|gx| ((g.off[0] + gx * g.step) as f64 - cx) / fx).collect();
    // 지도 높이 = r 셋째 줄 · 카메라 점 + 카메라 높이
    let wz = |pc: &V3| r[2][0] * pc[0] + r[2][1] * pc[1] + r[2][2] * pc[2] + pwc[2];
    let pts: Vec<(V3, [u8; 3], f32)> = (0..g.gh).flat_map(|gy| {
        let ay = ((g.off[1] + gy * g.step) as f64 - cy) / fy;
        let ax = &ax;
        (0..g.gw).filter_map(move |gx| {
            let i = gy * g.gw + gx;
            let z = g.d[i] as f64;
            (z > 0.0).then(|| ([ax[gx] * z, ay * z, z], [g.rgb[i * 3], g.rgb[i * 3 + 1], g.rgb[i * 3 + 2]], z as f32))
        })
    }).collect();
    let zs: Vec<f64> = pts.iter().map(|p| wz(&p.0)).collect();
    let locked = ceil.observe(pwc[2], &zs);
    let top = ceil.cut();
    if locked {
        let n = cm.cut_above(top);
        common::info!("천장: 이미 쌓인 점 {n} 개 지움");
    }
    cm.add(kf, seq, r, pwc, pts.into_iter().zip(zs).filter(|(_, z)| *z <= top).map(|(p, _)| p));
}

/// 장면 버퍼를 쥔 채 쓰다가 버려질 때 카메라 입력에 돌려줌(Deref 로 안의 값을 그대로 씀)
struct Recycle<T: Give + Default>(T);
trait Give {
    fn give(self);
}
impl Give for Vec<u8> {
    fn give(self) {
        if slam::bufpool::enabled() {
            slam::bufpool::give_rgb(self);
        }
    }
}
impl Give for Depth {
    fn give(self) {
        // 미터 칸은 늘 버퍼 줄로, 원래 값은 카메라 입력이면 돌려줌
        if let (Some(d), true) = (self.into_raw(), slam::bufpool::enabled()) {
            slam::bufpool::give_depth_buf(d);
        }
    }
}
impl<T: Give + Default> Drop for Recycle<T> {
    fn drop(&mut self) {
        std::mem::take(&mut self.0).give();
    }
}
impl<T: Give + Default> std::ops::Deref for Recycle<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T: Give + Default> std::ops::DerefMut for Recycle<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// 다 쓴 장면 컬러·깊이 버퍼를 카메라 입력(rsdriver)에 돌려줌(카메라 입력이 아니면 그냥 버림)
fn give_back(rgb: Vec<u8>, depth: Depth) {
    let d = depth.into_raw(); // 미터 칸은 버퍼 줄로
    if slam::bufpool::enabled() {
        slam::bufpool::give_rgb(rgb);
        if let Some(d) = d {
            slam::bufpool::give_depth_buf(d);
        }
    }
}

/// 점 줄의 상태: 점 지도와 그 시간 계측
struct PtsSt {
    a: std::sync::Arc<Args>,
    k: [f64; 4],
    chunks: chunks::ChunkMap,
    json_tx: std::sync::mpsc::Sender<serde_json::Value>,
    t_grid: f64,
    t_add: f64,
    t_write: f64,
    nf: u32,
    ceil: Ceil,
}

/// 점 줄: 점 고르기·청크 넣기·청크 파일 쓰기를 작업 풀에서 차례대로(스레드 없음). 루프는 장면을 넘기기만 한다.
/// 청크 목록 JSON 은 쓰기가 끝난 뒤에야 바뀌므로(완료 줄로 돌아옴) view.json 은 늘 다 쓴 판만 가리킨다
struct PtsWorker {
    lane: common::rt::Lane<PtsSt>,
    json_rx: std::sync::mpsc::Receiver<serde_json::Value>,
    json: std::cell::RefCell<serde_json::Value>,
    wait: bool, // 밀려도 기다림(TUM 입력: 결과가 늘 같게)
    gap: (f64, f64), // 이만큼(m, 라디안) 안 움직인 장면은 점 격자 일부만
    last: Option<(M3, V3)>, // 마지막으로 다 넣은 장면 위치
    skipped: u32, // 점 격자 일부만 넣은 가까운 자리 장면 수
    dropped: u32,
}

impl PtsWorker {
    fn new(a: std::sync::Arc<Args>, k: [f64; 4], wait: bool) -> PtsWorker {
        let (json_tx, json_rx) = std::sync::mpsc::channel();
        let chunks = chunks::ChunkMap::new(a.scene_vox, a.pts_obs);
        let st = PtsSt { a, k, chunks, json_tx, t_grid: 0.0, t_add: 0.0, t_write: 0.0, nf: 0, ceil: Ceil::default() };
        PtsWorker { lane: common::rt::Lane::new("pts", st, 0), json_rx, json: std::cell::RefCell::new(serde_json::Value::Null), wait,
                    dropped: 0, gap: (PTS_GAP_M, PTS_GAP_DEG.to_radians()), last: None, skipped: 0 }
    }
    /// 실시간 장면 넘기기: 지도에 넣는 장면은 버리지 않는다(물체가 들어간 장면이면 점도 들어가게). 줄이 밀리면(스트림) 점 격자 일부만 넣고,
    /// 많이 밀리면 줄이 줄 때까지 기다린다. 직전에 다 넣은 장면과 거의 같은 자리에서 본 장면도 점 격자 일부만(어긋남을 돌려 가며):
    /// 칸마다 본 횟수는 쌓이고 넣기는 1/PTS_SUB²
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, kf: u32, r: M3, pwc: V3, depth: Depth, rgb: Vec<u8>, w: usize, h: usize, keep: Option<std::sync::Arc<kfstore::Kf>>, thin: bool) {
        let late = !self.wait && self.lane.pending() as usize >= PTS_QUEUE;
        if late {
            self.dropped += 1; // 밀려 일부만 넣은 장면 수
            if self.lane.pending() as usize >= PTS_QUEUE * 3 {
                common::rt::wait_until(|| (self.lane.pending() as usize) < PTS_QUEUE * 2);
            }
        }
        let near = thin || late || self.last.as_ref().is_some_and(|(lr, lp)| norm(&sub(&pwc, lp)) < self.gap.0 && rot_angle(&r, lr) < self.gap.1);
        let part = if near {
            self.skipped += 1;
            Some(self.skipped as usize % (PTS_SUB * PTS_SUB))
        } else {
            self.last = Some((r, pwc));
            None
        };
        self.lane.push(move |s: &mut PtsSt| {
            let c0 = Instant::now();
            if let Some(z) = &keep {
                z.fill(&rgb, &depth); // 보관 장면 압축(루프 대신)
            }
            let g = pts_grid(&s.a, s.k, &depth, &rgb, w, h, part);
            give_back(rgb, depth);
            let seq = keep.as_ref().map(|z| z.seq);
            if let Some(z) = keep {
                let _ = z.pts.set(Some(g.pack()));
            }
            let c1 = Instant::now();
            chunk_points(s.k, &g, &r, &pwc, &mut s.chunks, kf, seq, &mut s.ceil);
            (s.t_grid, s.t_add, s.nf) = (s.t_grid + (c1 - c0).as_secs_f64(), s.t_add + c1.elapsed().as_secs_f64(), s.nf + 1);
            out::slow(c0);
        });
    }
    /// 맡아 둘 장면: 점 격자만 만들어 붙이게 넘김
    fn hold(&mut self, depth: Depth, rgb: Vec<u8>, w: usize, h: usize, keep: std::sync::Arc<kfstore::Kf>) {
        // 맡아 둔 장면도 이어지면 점이 들어가야 하니 버리지 않음(많이 밀렸으면 줄이 줄 때까지 기다림)
        if !self.wait && self.lane.pending() as usize >= PTS_QUEUE * 3 {
            common::rt::wait_until(|| (self.lane.pending() as usize) < PTS_QUEUE * 2);
        }
        self.lane.push(move |s: &mut PtsSt| {
            let c0 = Instant::now();
            keep.fill(&rgb, &depth);
            let _ = keep.pts.set(Some(pts_grid(&s.a, s.k, &depth, &rgb, w, h, None).pack()));
            give_back(rgb, depth);
            out::slow(c0);
        });
    }
    /// 고친 보관 장면 위치로 청크 기준 자세만 고치고, 이제 지도에 들어갈 보관 장면(맡아 둔 것이 이어짐)을 넣는다
    fn repose(&mut self, snap: &Snap) {
        let poses: HashMap<u64, (M3, V3)> = snap.iter().map(|(kf, r, p)| (kf.seq, (*r, *p))).collect();
        let snap: Vec<_> = snap.iter().map(|(kf, r, p)| (kf.clone(), *r, *p)).collect();
        self.lane.push(move |s: &mut PtsSt| {
            let (cm, ceil) = (&mut s.chunks, &mut s.ceil);
            let n = cm.repose(&poses);
            if std::env::var("MEMMAP_PTS_LOG").is_ok() {
                common::info!("청크 자세 고침: {n} 개 / 지금까지 {} 개 {:.1} ms", cm.n_repose, cm.t_repose * 1e3);
            }
            for (kf, r, p) in &snap {
                if cm.has(kf.seq) {
                    continue;
                }
                if let Some(g) = kf.pts_wait().as_ref().map(PtsGrid::from_z) {
                    chunk_points(s.k, &g, r, p, cm, 0x8000_0000 | kf.seq as u32, Some(kf.seq), ceil);
                }
            }
        });
    }
    /// 밀린 장면 없음(실시간 점 노드 쓰기를 장면마다 해도 줄이 늘지 않을 때)
    fn idle(&self) -> bool {
        self.lane.pending() <= 1
    }
    /// 청크 파일 쓰기 요청(기다리는 모드면 다 쓸 때까지 기다려 바로 뒤 json() 이 이번 판)
    fn write(&mut self) {
        self.lane.push(move |s: &mut PtsSt| {
            let c0 = Instant::now();
            let j = s.chunks.write(&s.a.out);
            out::slow(c0);
            s.t_write += c0.elapsed().as_secs_f64();
            let _ = s.json_tx.send(j);
        });
        if self.wait {
            self.lane.wait_idle();
        }
    }
    /// 다 쓴 마지막 청크 목록(view.json 용)
    fn json(&self) -> serde_json::Value {
        let mut cur = self.json.borrow_mut();
        while let Ok(j) = self.json_rx.try_recv() {
            *cur = j;
        }
        cur.clone()
    }
    /// 남은 청크를 모두 쓰고 끝냄: (점 수, 지도 MB, 청크 목록)
    fn finish(&mut self) -> (usize, f64, serde_json::Value) {
        self.write();
        let (n, mb) = self.lane.with(|s| {
            if std::env::var("MEMMAP_PTS_LOG").is_ok() {
                let n = s.nf.max(1) as f64;
                common::info!("점 줄: 장면 {} 개, 점 고르기 {:.1} ms/장 + 지도 넣기 {:.1} ms/장, 청크 쓰기 합 {:.2} 초", s.nf, s.t_grid / n * 1e3, s.t_add / n * 1e3, s.t_write);
            }
            let cm = &s.chunks;
            common::info!("점 지도: 청크 {} 개 · 점 {} · {:.1} MB, 이미 점이 있어 건너뛴 새 점 {} · 자세 고친 뒤 겹쳐 뺀 점 {}, 자세 고침 {} 번 합 {:.1} ms", cm.write_count(), cm.n_points(), cm.bytes() as f64 / 1048576.0, cm.n_dup, cm.n_dedup, cm.n_repose, cm.t_repose * 1e3);
            (cm.n_points(), cm.bytes() as f64 / 1048576.0)
        });
        (n, mb, self.json())
    }
}

/// 다시 쌓은 뒤 출력 새로: 실행 번호를 바꿔 뷰어가 받은 지도를 비우고 지금 지도를 처음부터 받게 함
fn fresh_outputs(run_id: &mut u64, vox: f64, pw: Option<&PtsWorker>) -> serde_json::Value {
    *run_id += 1;
    let mut sj = json!({"run": *run_id, "voxel": vox});
    if let Some(pw) = pw {
        sj["submap"] = pw.json();
    }
    sj
}

/// 한 장면을 지도(점유 격자)·물체·지나온 길에 넣는다. 실시간 장면과 루프 클로저 뒤 다시 쌓기가 같이 쓴다.
/// r, pwc: 카메라 → 지도(바닥 높이 맞춘 뒤), floor_z: 이 장면의 바닥 높이. 돌려줌: (지도 끝, 물체 끝) 시각
#[allow(clippy::too_many_arguments)]
fn map_frame(_a: &Args, k: [f64; 4], t: f64, r_: &M3, pwc_: &V3, floor_z: f64, depth: &Depth, rgb: &[u8], w: usize, h: usize, dets: &Option<Vec<seg::Det>>,
             map_ok: bool, grid: &mut Grid, mem: &mut Memory,
             traj: &mut Vec<(f64, [f64; 3])>, map_traj: &mut Vec<(f64, V3, M3)>) -> (Instant, Instant) {
    let (r, pwc) = (*r_, *pwc_);
    let [fx, fy, cx, cy] = k;
    let back = |u: usize, v: usize| -> Option<V3> {
        let z = depth.at(v * w + u) as f64;
        (z > 0.2 && z < MAX_DEPTH_M).then(|| [(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z])
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
    // 검출마다 쓰는 작업 칸(유효 화소 · 표본 · 깊이 · 세계 점)은 스레드에 하나 두고 다시 씀(장면마다 할당 없음)
    thread_local! { static OS: std::cell::Cell<(Vec<u32>, Vec<(usize, usize, f64)>, Vec<f64>, Vec<(V3, [u8; 3])>)> = const { std::cell::Cell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) }; }
    let (mut valid, mut pix, mut zs, mut pts) = OS.with(|c| c.take());
    // 침식 작업 칸은 스레드에 하나(장면·검출마다 다시 씀)
    thread_local! { static EB: std::cell::Cell<ErodeBuf> = std::cell::Cell::new(ErodeBuf { a: vec![], b: vec![], run: vec![] }); }
    let mut eb = EB.with(|c| c.take());
    // 깊이 > 0.2 m 판정: 원래 값이면 문턱 값 하나와 비교(나눗셈 없이 같은 판정)
    let raw_th = depth.raw().zip(depth.raw_above(0.2));
    for d in dets.iter().flatten().filter(|_| map_ok) {
        let c_obj = Instant::now();
        let name = "object";
        let [x1, y1, x2, y2] = d.box_.map(|v| v as usize);
        let (bw, bh) = (x2 - x1, y2 - y1);
        erode_into(&d.mask, bw, bh, 3, &mut eb);
        let m = &eb.b;
        // 마스크 안 깊이 있는 화소: 먼저 세어 간격을 정하고, 그 간격으로 최대 약 2,500 개만 모은다
        // 한 번 훑어 깊이 있는 화소 자리를 모아 두고(마스크·깊이를 두 번 읽지 않게) 그 간격으로 고름
        valid.clear();
        for yy in 0..bh {
            let (mr, o0) = (&m[yy * bw..(yy + 1) * bw], (y1 + yy) * w + x1);
            match raw_th {
                Some((dr, th)) => {
                    let dr = &dr[o0..o0 + bw];
                    for xx in 0..bw {
                        if mr[xx] && dr[xx] as u32 >= th {
                            valid.push((o0 + xx) as u32);
                        }
                    }
                }
                None => {
                    for xx in 0..bw {
                        if mr[xx] && depth.at(o0 + xx) as f64 > 0.2 {
                            valid.push((o0 + xx) as u32);
                        }
                    }
                }
            }
        }
        let cnt = valid.len();
        if cnt < 80 {
            continue;
        }
        let step = (cnt / 2500).max(1);
        pix.clear();
        pix.extend(valid.iter().step_by(step).map(|&o| {
            let o = o as usize;
            (o % w, o / w, depth.at(o) as f64)
        }));
        zs.clear();
        zs.extend(pix.iter().map(|q| q.2));
        let mid = zs.len() / 2;
        let med = *zs.select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap()).1; // 정렬 대신 선택(같은 값)
        if med > MAX_DEPTH_M {
            continue;
        }
        pts.clear();
        pts.extend(pix
            .iter()
            .filter(|q| (q.2 - med).abs() < 0.25 * med + 0.1)
            .map(|&(u, v, z)| {
                let o = (v * w + u) * 3;
                (to_w(&[(u as f64 - cx) / fx * z, (v as f64 - cy) / fy * z, z]), [rgb[o], rgb[o + 1], rgb[o + 2]])
            }));
        if pts.len() < 50 || is_wall(&pts) {
            continue;
        }
        let border = [x1 <= 1, y1 <= 1, x2 + 1 >= w, y2 + 1 >= h].iter().filter(|b| **b).count();
        // 천장(정한 높이 위)에 걸친 검출은 물체로 넣지 않음(천장 조명·에어컨 등이 천장 면을 끌고 옴)
        if pts.iter().map(|q| q.0[2]).sum::<f64>() / pts.len().max(1) as f64 > ceil_top() {
            continue;
        }
        if let Some(why) = fragment(&pts, floor_z, pwc[2], border) {
            if std::env::var("ONDEVICE_NAME_DEBUG").is_ok() {
                println!("fragdbg {why}");
            }
            continue;
        }
        // 사진은 그 물체의 지금 사진보다 클 때만 만든다(장면마다 모든 검출을 복사하지 않게)
        let crop = || {
            let mut crgb = Vec::with_capacity(bw * bh * 3);
            let mut cdep = Vec::with_capacity(bw * bh);
            for yy in 0..bh {
                let o = ((y1 + yy) * w + x1) * 3;
                crgb.extend_from_slice(&rgb[o..o + bw * 3]);
                depth.extend_mm_trunc((y1 + yy) * w + x1, bw, &mut cdep);
            }
            common::prof::copy(common::prof::C::Crop, bw * bh * (3 + 2 + 1));
            Crop { w: bw as u32, h: bh as u32, rgb: crgb.into(), mask: std::sync::Arc::new(m.iter().map(|&b| if b { 255 } else { 0 }).collect()), depth: cdep.into() }
        };
        let on_main = std::thread::current().name() == Some("main"); // 다시 쌓기 스레드가 같은 함수를 부를 때는 빼고
        if on_main {
            common::prof::time(common::prof::T::ObjPoints, c_obj);
        }
        let c_up = Instant::now();
        let id = mem.update(name, d.score, &pts, med as f32, t, crop_score(rgb, w, h, x1, y1, bw, bh, m), crop, None);
        if on_main {
            common::prof::time(common::prof::T::ObjUpdate, c_up);
        }
        seen.insert(id);
    }
    EB.with(|c| c.set(eb));
    OS.with(|c| c.set((valid, pix, zs, pts)));
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
    for (name, _) in out::list_dir(&dir.join("objects")) {
        let Some(id) = Some(name.as_str()).and_then(|s| s.strip_prefix('O')).and_then(|s| s.split('_').next()).and_then(|s| s.parse::<u32>().ok()) else { continue };
        if !live.contains(&id) {
            out::remove_file(&dir.join("objects").join(&name));
        }
    }
}

/// 이 높이(m) 넘는 막힘은 가구가 아니라 벽으로 본다(책상 0.75, 의자 등받이·모니터 1.2 안팎) — 방 나누기에 씀
const WALL_H: f64 = 1.3;

fn write_view(dir: &Path, grid: &Grid, mem: &mut Memory, view: &mut ViewWorker, traj: &[(f64, [f64; 3])], stamp: f64, pose: [f64; 3],
              cam: &serde_json::Value, scene: &serde_json::Value, merge: bool, refresh: usize) {
    // 재생(결과를 늘 같게): 앞 회차에 맡긴 사진 벡터를 이 회차 앞에서 차례대로 반영(실시간은 끝나는 대로 반영됨)
    embed::sync_round();
    let mut left = refresh;
    if merge {
        mem.prune(stamp);
        mem.merge_pass();
    }
    let gi = grid.image();
    let mut to_embed = vec![]; // 이번 회차에 바뀐 물체 사진: 끝에 한 묶음으로 사진 벡터에
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
                out::png_rgb(&dir.join(format!("{base}_rgb.png")), c.w, c.h, c.rgb.clone());
                out::png_gray(&dir.join(format!("{base}_mask.png")), c.w, c.h, c.mask.clone());
                out::png_u16(&dir.join(format!("{base}_depth.png")), c.w, c.h, c.depth.clone());
                to_embed.push((o.id, c.w, c.h, c.rgb.clone()));
                common::prof::copy(common::prof::C::CropShare, 0);
            }
            o.crop_saved = o.crop_ver;
        }
        // 이름: 사진이 닮은 이름(--vocab)이 있으면 그것, 없으면 분할 모델이 준 이름
        let look = embed::name(o.id, o.name != "object");
        // 천장 면은 기억에 넣지 않음(위에서 볼 때 지도를 가림): 구조물 판정 표(--structural)로
        if embed::structure(o.id) == Some("ceiling") {
            continue;
        }
        // 이름 없는 분할 모델이 자른 벽 · 바닥 · 천장 면: 구조물 판정 표가 구조물로 본 물체(뷰어가 따로 끄고 켬)
        let structural = o.name == "object" && embed::structure(o.id).is_some();
        let mut e = json!({"id": o.id, "name": look.as_ref().map_or(&o.name, |l| &l.0), "seg_name": o.name, "state": o.state, "pos": o.pos, "first_pos": o.first_pos, "extent": o.extent,
            "n_obs": o.n_obs, "score": o.score, "last_seen": o.last_seen, "structural": structural,
            "points": {"path": format!("{base}_points.ply"), "stamp": o.stamp, "voxel": mem.vox, "n": o.n_pts}});
        if let Some((_, p)) = &look {
            e["name_p"] = json!((*p as f64 * 100.0).round() / 100.0);
        }
        if o.crop.is_some() {
            e["rgbd"] = json!({"rgb": format!("{base}_rgb.png"), "mask": format!("{base}_mask.png"), "depth": format!("{base}_depth.png"), "v": o.crop_ver});
            if embed::on() {
                e["emb"] = json!(embed::rel_path(o.id));
            }
        }
        objs.push(e);
    }
    embed::submit(dir, to_embed);
    // 방 · 엣지 · map.pgm · view.json 짜기는 뷰 스레드가(본 스레드는 물체 JSON 과 격자 그림만 넘김)
    let ol: Vec<_> = mem.objs.iter().map(|o| (o.id, o.pos, o.extent, o.state == "gone", embed::name(o.id, o.name != "object").map_or_else(|| o.name.clone(), |l| l.0))).filter(|o| embed::structure(o.0) != Some("ceiling")).collect();
    let traj = view.traj_update(traj);
    let wall = if view.live { wall_ms() } else { 0.0 };
    view.send(ViewJob { gi, res: grid.res, objs, ol, traj, stamp, pose, cam: cam.clone(), scene: scene.clone(), wall });
}

/// 지나온 길 바뀐 몫: 덧붙인 점, 또는 다시 쌓기로 새로 만든 길 전체
enum TrajUpd {
    Append(Vec<(f64, [f64; 3])>),
    Reset(Vec<(f64, [f64; 3])>),
}

/// 뷰 한 판의 재료(본 스레드가 만든 것)
struct ViewJob {
    gi: Option<GridImg>,
    res: f64,
    objs: Vec<serde_json::Value>,
    ol: Vec<(u32, [f64; 3], [f64; 3], bool, String)>, // 방 이름 근거: (id, 위치, 크기, 사라짐, 보이는 이름)
    traj: TrajUpd,
    stamp: f64,
    pose: [f64; 3],
    cam: serde_json::Value,
    scene: serde_json::Value,
    wall: f64, // 실시간: 이 판을 낸 장면의 벽시계(ms) — 뷰어(?metrics=1)가 지도 → 화면 지연을 잼. 재생은 0(view.json 에 안 실음)
}

/// 뷰 줄의 상태: 방 번호 잇기 · 지나온 길 기록
struct ViewSt {
    dir: PathBuf,
    tr: rooms::Tracker,
    log: std::sync::Arc<sgview::live::TrajLog>,
    n: u64,
}

/// 뷰 한 판 짜기: 방 나누기 · 엣지(graph::build), map.pgm, view.json(디스크) 또는 같은 프로세스 뷰어에 구조 그대로 건 판(--serve)
fn view_job(st: &mut ViewSt, j: ViewJob) {
    match &j.traj {
        TrajUpd::Append(v) => st.log.extend(v),
        TrajUpd::Reset(v) => {
            st.log = sgview::live::TrajLog::new();
            st.log.extend(v);
        }
    }
    let dir = &st.dir;
    let (grid_j, rooms, doors, edges) = match j.gi {
        Some(mut g) => {
            let ol: Vec<_> = j.ol.iter().map(|o| (o.0, o.1, o.2, o.3, o.4.as_str())).collect();
            let (rj, dj, ej) = graph::build(&mut st.tr, &g, j.res, &ol);
            let pgm = std::mem::take(&mut g.pgm); // 머리까지 짜 둔 바이트 그대로(복사 없음)
            if out::memory_only() {
                out::write_direct(&dir.join("map.pgm"), pgm); // 메모리: 판보다 먼저 걸림
            } else {
                out::write_owned(&dir.join("map.pgm"), pgm);
            }
            (json!({"width": g.w, "height": g.h, "resolution": j.res, "origin": [g.x0 as f64 * j.res, g.y0 as f64 * j.res]}), rj, dj, ej)
        }
        None => (json!(null), json!([]), json!([]), json!([])),
    };
    // view.json 에서 graph.nodes(지나온 길)만 뺀 값: 길은 로그에서 짤 때 끼움(같은 바이트)
    let mut head = json!({"stamp": j.stamp, "pose": j.pose, "grid": grid_j, "objects": j.objs, "rooms": rooms, "doors": doors,
        "graph": {"edges": edges}, "robot": "handheld camera", "cam": j.cam});
    if !j.scene.is_null() {
        head["scene_points"] = j.scene;
    }
    if j.wall > 0.0 {
        head["wall"] = json!(j.wall);
    }
    st.n += 1;
    let len = st.log.len();
    if out::memory_only() {
        sgview::live::publish(sgview::live::View::live(st.n, head, st.log.clone(), len));
    } else {
        out::write_owned(&dir.join("view.json"), st.log.view_bytes(&head, len));
    }
}

/// 뷰 줄: 받은 차례대로 하므로 방 번호 잇기 · 파일 차례는 루프에서 하던 것과 같다.
/// 실시간(메모리 기억 폴더)은 줄이 바쁘면 가장 새 판 하나만 맡아 두었다가(덮어쓴 판의 길 점은 새 판 앞에 이어 붙여 하나도 잃지 않음)
/// 줄이 비면 넘긴다(루프는 기다리지 않음). 디스크 기록 · 재생은 하나도 건너뛰지 않음(결과가 늘 같게)
struct ViewWorker {
    lane: common::rt::Lane<ViewSt>,
    stash: Option<ViewJob>,
    sent: usize,   // 넘긴 길 점 수
    fresh: bool,   // 길이 새로 만들어짐(다시 쌓기 갈아 끼우기)
    live: bool,    // 실시간 입력(판에 벽시계를 실음)
}

impl ViewWorker {
    /// waker: 줄이 비면 깨울 루프(맡아 둔 판을 넘기게)
    fn new(dir: PathBuf, live: bool, waker: std::sync::Arc<common::rt::Waker>) -> ViewWorker {
        let st = ViewSt { dir, tr: rooms::Tracker::default(), log: sgview::live::TrajLog::new(), n: 0 };
        ViewWorker { lane: common::rt::Lane::with_idle("view", st, 0, move || waker.wake()), stash: None, sent: 0, fresh: true, live }
    }
    fn traj_update(&mut self, traj: &[(f64, [f64; 3])]) -> TrajUpd {
        let u = if std::mem::take(&mut self.fresh) || traj.len() < self.sent { TrajUpd::Reset(traj.to_vec()) } else { TrajUpd::Append(traj[self.sent..].to_vec()) };
        common::prof::copy(common::prof::C::ViewTraj, match &u { TrajUpd::Reset(v) | TrajUpd::Append(v) => v.len() * 32 });
        self.sent = traj.len();
        u
    }
    /// 길이 새로 만들어짐(갈아 끼운 뒤): 다음 판은 길 전체를 넘김
    fn traj_replaced(&mut self) {
        self.fresh = true;
    }
    fn send(&mut self, mut j: ViewJob) {
        if !out::memory_only() {
            self.lane.push(move |s: &mut ViewSt| view_job(s, j));
            return;
        }
        // 실시간: 아직 안 넘긴 판은 이 판으로 바꾸되 그 판의 길 점을 앞에 붙임
        if let Some(old) = self.stash.take() {
            j.traj = match (old.traj, j.traj) {
                (_, TrajUpd::Reset(v)) => TrajUpd::Reset(v),
                (TrajUpd::Reset(mut a), TrajUpd::Append(b)) => {
                    a.extend(b);
                    TrajUpd::Reset(a)
                }
                (TrajUpd::Append(mut a), TrajUpd::Append(b)) => {
                    a.extend(b);
                    TrajUpd::Append(a)
                }
            };
        }
        self.stash = Some(j);
        self.tick();
    }
    /// 줄이 비었으면 맡아 둔 판을 넘김(루프가 깰 때마다)
    fn tick(&mut self) {
        if self.lane.idle() {
            if let Some(j) = self.stash.take() {
                self.lane.push(move |s: &mut ViewSt| view_job(s, j));
            }
        }
    }
    /// 넘긴 판을 모두 짤 때까지 기다림
    fn flush(&mut self) {
        if let Some(j) = self.stash.take() {
            self.lane.push(move |s: &mut ViewSt| view_job(s, j));
        }
        self.lane.wait_idle();
    }
}

/// 표면이 시선과 이루는 각: 이웃 화소(±2)로 만든 법선과 시선의 코사인이 0.3(약 73°) 이상인 점만 — 비스듬히 본 면은 깊이 잡음이
/// 시선 방향으로 길게 번져 줄무늬 껍질이 됨
/// 물체 사진 점수(클수록 이름 붙이기에 좋은 사진): 넓이 × 밝기 × 선명함 × 잘림. 마스크 안 화소를 4 칸마다 본다.
/// 어두운 사진(밝기 평균 60 아래), 흐린 사진(이웃 밝기 차 평균이 작음), 영상 가장자리에 닿아 잘린 사진은 깎는다
#[allow(clippy::too_many_arguments)]
fn crop_score(rgb: &[u8], w: usize, h: usize, x1: usize, y1: usize, bw: usize, bh: usize, m: &[bool]) -> u32 {
    let luma = |o: usize| (rgb[o * 3] as u32 * 77 + rgb[o * 3 + 1] as u32 * 150 + rgb[o * 3 + 2] as u32 * 29) >> 8;
    let (mut n, mut sum, mut grad) = (0u32, 0u32, 0u32);
    for yy in (0..bh).step_by(4) {
        for xx in (0..bw.saturating_sub(1)).step_by(4) {
            if !m[yy * bw + xx] {
                continue;
            }
            let o = (y1 + yy) * w + x1 + xx;
            let (a, b) = (luma(o), luma(o + 1));
            n += 1;
            sum += a;
            grad += a.abs_diff(b);
        }
    }
    if n == 0 {
        return 0;
    }
    let (mean, sharp) = (sum as f32 / n as f32, grad as f32 / n as f32);
    let bright = (mean / 60.0).clamp(0.2, 1.0);
    let crisp = (sharp / 6.0).clamp(0.3, 1.0);
    let edge = if x1 == 0 || y1 == 0 || x1 + bw >= w || y1 + bh >= h { 0.6 } else { 1.0 };
    ((bw * bh) as f32 * bright * crisp * edge) as u32
}

/// 장면 보관 예산 기본값: 시작할 때 쓸 수 있는 메모리(MemAvailable)의 15 %(리눅스), 그 밖은 3000 MB.
/// 8 GB 보드에서 3000 MB 는 다른 실행과 겹치면 OOM 이 났다. 예산 안이면 보관·다시 쌓기 결과는 같다
fn kf_store_default_mb() -> usize {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map_or(3000, |kb| (kb / 1024 * 15 / 100).max(256))
}

/// ax[x] = (x − cx)/fx, ay[y] = (y − cy)/fy: 화소마다 나누지 않게 미리(곱하는 차례는 그대로 → 같은 값)
fn facing(d: &[f32], w: usize, h: usize, ax: &[f64], ay: &[f64], u: usize, v: usize, pc: &V3) -> bool {
    if u < 2 || v < 2 || u + 2 >= w || v + 2 >= h {
        return false;
    }
    let b = |x: usize, y: usize| {
        let z = d[y * w + x] as f64;
        [ax[x] * z, ay[y] * z, z]
    };
    let (du, dv) = (sub(&b(u + 2, v), &b(u - 2, v)), sub(&b(u, v + 2), &b(u, v - 2)));
    let n = cross(&du, &dv);
    let nn = norm(&n) * norm(pc);
    nn > 0.0 && dot(&n, pc).abs() / nn >= 0.3
}

/// 점 주변 5×5 깊이가 8 % 넘게 들쭉날쭉하면 경계(안쪽 판: 2 ≤ u < w−2, 2 ≤ v < h−2)(2 ≤ u < w−2, 2 ≤ v < h−2): 줄마다 5 칸 조각을 한 번에(같은 최솟값·최댓값)
#[inline]
fn depth_edge_in(d: &[f32], w: usize, u: usize, v: usize) -> bool {
    let (mut lo, mut hi) = (f32::MAX, 0f32);
    for y in v - 2..=v + 2 {
        let r: &[f32; 5] = d[y * w + u - 2..y * w + u + 3].try_into().unwrap();
        for &z in r {
            // f32::min/max 와 같은 결과(NaN 은 건너뜀)를 비교 한 번으로
            lo = if z < lo { z } else { lo };
            hi = if z > hi { z } else { hi };
        }
    }
    lo <= 0.0 || hi / lo > 1.08
}

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

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // 한 프로그램(ondevice): 첫 인자가 slam · view 면 위치 프로그램만 · 뷰어 서버만, 그 밖(run 또는 없음)은 물체 기억(+ --slam · --serve)
    let argv = common::cli::Argv::env();
    // --rt: 역할별 SCHED_FIFO(카메라 받기·추론 등). 권한이 없으면 조용히 그대로
    common::threads::set_rt(&argv.get("--rt").unwrap_or_default());
    common::log::set_quiet(argv.flag("--quiet"));
    // DXRT 설정(NPU 코어 묶기 · 버퍼 · NFH 스레드 · 프로파일러, common::dxrt::Config): 하위 명령(view)에도
    #[cfg(feature = "dxrt")]
    common::dxrt::Config::from_args(|k| argv.get(k), |k| argv.flag(k)).apply();
    // NPU 끝 알림: thread(기본, 모델마다 기다림 스레드) | callback(DXRT 콜백 → 루프 깨움, 보드 `ondevice npu-test` 통과 뒤)
    #[cfg(feature = "dxrt")]
    if argv.get("--npu-wake").as_deref() == Some("callback") {
        common::dxrt::set_wake(common::dxrt::Wake::Callback);
    }
    match argv.0.get(1).map(String::as_str) {
        Some("slam") => {
            let code = slam::app::run(&argv.0[1..]).unwrap_or_else(|e| {
                eprintln!("slam 오류: {e}");
                1
            });
            std::process::exit(code);
        }
        Some("view") => return sgview::run(argv.0[2..].to_vec()),
        // NPU 끝 알림 시험(보드): 콜백 · 기다림 스레드 길의 비트 같음 · 지연 · 콜백 스레드
        #[cfg(feature = "dxrt")]
        Some("npu-test") => return npu_test::run(&argv),
        _ => {}
    }
    limit_malloc_arenas();
    // 사진 벡터 확인: ondevice --embed-check <모델> <사진 폴더> <벡터 파일>
    let v = &argv.0;
    if let Some(i) = v.iter().position(|x| x == "--embed-check") {
        let g = |k: usize| v.get(i + k).unwrap_or_else(|| panic!("--embed-check <모델> <사진 폴더> <벡터 파일>")).as_str();
        return embed::check(g(1), g(2), g(3));
    }
    let a = args(&argv);
    // 결과를 늘 같게 하는 입력(밀림을 재지 않음: TUM, --stream-rate 없는 --no-drop 스트림): 사진 벡터는 기록 회차 앞에서 차례대로 반영
    let det = !(a.stream_rate > 0.0 || (a.stream.is_some() && !a.no_drop));
    out::set_layout(&a.threads);
    out::start_pools(a.pool);
    if let Some(m) = &a.embed {
        if let Some(k) = argv.get("--embed-jobs").and_then(|s| s.parse().ok()) {
            embed::set_jobs(k);
        }
        embed::start(m, det);
    }
    out::set_slow(a.slow);
    out::thread_role(out::Role::Main);
    if a.live > 0.0 || a.stream.is_some() {
        out::start_writer(); // 실시간·스트림: 파일 쓰기(PLY·PNG 짜기 포함)는 기록 줄이
    }
    let mut io = LoopIo::new(&a, &argv);
    // 분할 모델(ORT 스레드)은 추론 스레드 역할의 코어에서 만든다(ORT 스레드가 만든 스레드의 코어 묶음을 물려받음)
    out::thread_role(out::Role::Infer);
    let mut seg = match &a.seg_cache {
        Some(dir) => { let path = a.seg.clone(); Box::new(segsim::CachedSeg::new(dir, a.seg_cpu_ms, Box::new(move || make_seg(&path)))) as Box<dyn seg::Segmenter + Send> }
        None => make_seg(&a.seg),
    };
    // NPU(PC 는 ONNX 실행 스레드) 작업 끝 → 루프 깨움
    let ntx = io.npu_tx.clone();
    io.seg_async = seg.set_notify(std::sync::Arc::new(move |tag| ntx.send(tag)));
    out::thread_role(out::Role::Main);
    // --serve: 뷰어 서버(sgview)를 이 처리 루프에서(듣는 소켓 · 연결이 루프 epoll 에, 스레드 없음). 기억 폴더에 쓰는 파일은 쓴 바이트를 메모리로 넘겨받는다(디스크에서 다시 읽지 않음)
    let serving = a.serve.clone().map(|cfg| {
        // 기억 폴더는 메모리에만(끝에 · --save-every 초마다 · 뷰어 저장 버튼으로 디스크에, 저장은 루프가 기록 줄에 맡김), 뷰어 버튼은 공유 신호로
        sgview::memfs::enable();
        out::set_memory_only();
        common::control::wire();
        if let Some(sv) = sgview::Server::new(cfg, io.waker()) {
            sv.register(&mut io.poll);
            io.server = Some(sv);
        }
    });
    // --slam: 위치 프로그램을 같은 처리 루프에서(장면은 줄로 넘어와 --stream 과 같은 경로를 탄다, 스레드 없음).
    // 카메라 장치 fd 는 루프 epoll 에, 녹화 재생은 작업 풀이 한 장 앞서 풂
    if let Some(sa) = a.slam.clone() {
        let side = stream::SlamSide::new(sa, io.w_in.clone()).unwrap_or_else(|e| panic!("--slam: {e}"));
        // IMU 모드 실시간 + 뷰어: 표본이 올 때마다(200 Hz) 루프가 장면 사이 위치를 pose.json(메모리)으로
        if a.live > 0.0 && a.serve.is_some() && side.set_imu_waker(io.w_imu.clone()) {
            io.fast_out = Some(a.out.join("pose.json"));
        }
        #[cfg(target_os = "linux")]
        for fd in side.fds() {
            io.poll.add_fd(fd, T_IN, false);
        }
        stream::hand_slam(side);
    }
    // --mlock: 모델·스레드를 다 띄운 뒤 그때 있는 메모리만 묶는다(뒤에 늘어나는 장면 보관·지도는 묶지 않음)
    if argv.flag("--mlock") && !common::threads::mlock_all() {
        common::info!("--mlock: 권한이 없어 메모리를 묶지 못함(그대로 진행)");
    }
    if a.seg_check > 0 {
        return seg_check(&a, seg.as_mut(), a.seg_check);
    }
    loop {
        seg = run(&a, seg, &mut io);
        // 리셋 블록으로 끝났으면 같은 스트림을 이어 읽어 빈 지도에서 다시 시작
        if stream::reset_pending() {
            common::info!("리셋: 지도를 비우고 다시 시작");
            continue;
        }
        if !a.looping {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
    // 입력이 끝나도 뷰어는 계속 띄워 둠(Ctrl-C 로 끝냄, 저장 버튼 · --save-every 는 루프가 받음). --serve-exit 이면 바로 끝냄(측정·검증용)
    if serving.is_some() {
        out::flush_writer();
        let c = Instant::now();
        let n = out::save(&a.out);
        common::info!("기억 폴더 저장: 파일 {n} 개, {:.2} 초", c.elapsed().as_secs_f64());
        if !a.serve_exit {
            common::info!("입력 끝: 뷰어는 계속 띄워 둠(Ctrl-C 로 끝냄)");
            loop {
                io.sleep(None);
                io.misc();
            }
        }
    }
}

/// 뷰어 수평 버튼: 한 프로그램(--serve)이면 공유 신호, 따로 띄운 뷰어면 기억 폴더의 level.req
fn level_requested(a: &Args) -> bool {
    common::control::take_level() || (a.serve.is_none() && std::fs::remove_file(a.out.join("level.req")).is_ok())
}

/// 보관 장면 중 seq 뒤의 것(지금 노드 위치로)
fn newer_than(kfs: &kfstore::KfStore, fix: &M3, z0: f64, seq: u64) -> Snap {
    snapshot(kfs, fix, z0).into_iter().filter(|s| s.0.seq > seq).collect()
}

/// 다시 쌓기가 끝난 뒤 그동안 들어온 장면이 이만큼 넘으면 작업 스레드에서 먼저 이어 넣음(갈아 끼우는 장면에서 본 스레드가 멈추지 않게)
const CATCH_UP_MAX: usize = 3;

/// 보관 장면 여럿을 코어마다 나눠 풂(깊이 · 컬러 · 검출). 쌓기는 부른 쪽이 차례대로(같은 결과)
fn unpack_all(snap: &[(std::sync::Arc<kfstore::Kf>, M3, V3)]) -> Vec<(Depth, Vec<u8>, Option<Vec<seg::Det>>)> {
    use rayon::prelude::*;
    // 압축은 점 작업 스레드가 채우는데 그 스레드도 같은 뒤쪽 풀을 쓴다: 풀에 들어가기 전에 여기서 다 기다려야
    // 풀 일꾼이 아직 안 채운 장면을 기다리며 풀을 막는 교착이 없다(--block 재생에서 멈춤)
    for (kf, _, _) in snap {
        kf.wait_ready();
    }
    out::par(|| snap.par_iter().map(|(kf, _, _)| (Depth::M(kf.depth()), kf.rgb(), kf.dets())).collect())
}

/// 다시 쌓은 지도에 쌓는 동안 새로 보관한 장면을 이어 넣음(작업 스레드). 본 스레드에서 넣는 것과 같은 순서·같은 함수
fn catch_up(a: &Args, k: [f64; 4], rb: &mut Rebuilt, newer: Snap) {
    let c0 = Instant::now();
    for ((kf, r, pwc), (d, rgb, dets)) in newer.iter().zip(unpack_all(&newer)) {
        map_frame(a, k, kf.t, r, pwc, pwc[2] - kf.h_cam, &d, &rgb, kf.w, kf.h, &dets, true, &mut rb.grid, &mut rb.mem, &mut rb.traj, &mut rb.map_traj);
        rb.last_seq = kf.seq;
    }
    rb.secs += c0.elapsed().as_secs_f64();
    rb.extra.extend(newer);
}

#[cfg(test)]
mod erode_tests {
    /// 바꾸기 전 식(새 버퍼 둘)
    fn erode_ref(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
        let mut a = vec![false; w * h];
        for y in 0..h {
            let mut run = 0usize;
            for x in 0..w {
                run = if m[y * w + x] { run + 1 } else { 0 };
                if run > 2 * r {
                    a[y * w + x - r] = true;
                }
            }
        }
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

    #[test]
    fn erode_same_as_before() {
        let mut s = 99u64;
        let mut eb = super::ErodeBuf::default();
        for (w, h) in [(1, 1), (7, 3), (40, 31), (64, 64), (5, 90)] {
            for r in 0..4 {
                let m: Vec<bool> = (0..w * h).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 61) != 0 }).collect();
                super::erode_into(&m, w, h, r, &mut eb);
                assert_eq!(eb.b, erode_ref(&m, w, h, r), "{w}x{h} r{r}");
                assert_eq!(super::erode(&m, w, h, r), eb.b);
            }
        }
    }
}

/// 지금 벽시계(유닉스 ms): 실시간 pose.json 의 wall — 뷰어(?metrics=1)가 화면에 그린 시각과 견줘 위치 → 화면 지연을 잰다
fn wall_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1e3)
}
