//! 손카메라 영상 + 포즈 → 점유 지도·3D 점 지도·물체 기억·씬그래프(방·엣지) → 기억 폴더(view.json, sgview 가 읽음).
//!
//!   memmap --tum data/tum/<seq> --out data/memory/<name> [--seg models/yolo26s-seg.onnx] [--every 5] [--conf 0.35]
//!          [--live 1] [--no-skip] [--loop] [--scene-every 10] [--snaps 150,500]
//!
//! 지도·트래커는 누적식(maps.rs): 키프레임마다 새 점만 해시 칸에 더하고, 기록은 칸을 한 번 훑기만 한다.
//! 깊이는 1 단계로 TUM Kinect 실측을 쓴다(딥러닝 깊이는 다음 단계).
#[cfg(feature = "cam")]
mod cam;
mod dvo;
mod geom;
mod graph;
mod maps;
mod out;
mod rec;
mod scale;
mod seg;
mod tsdf;
mod vo;
mod tum;

use geom::*;
use maps::*;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    sp_every: usize,
    fast_pose: bool,   // 모든 장면은 위치만 먼저 추적, 깊이·분할은 따로(카메라: 모델이 쉬는 대로 최신 장면에)
    depth_every: usize, // --fast-pose 로 TUM 을 돌릴 때 깊이 장면 간격
    depth_lag: usize,   // --fast-pose 로 TUM 을 돌릴 때 깊이 장면을 위치만 추적 장면 몇 장 뒤에 넣을지(보드의 모델 지연 흉내) // SuperPoint 는 N 장마다(사이 장면은 RGB-D 직접 방식으로 위치)
    scene_vox: f64,
    max_depth: f64,
    scene_step: usize, // 3D 지도에 쓸 화소 간격
    surface: bool,     // --surface: 점 대신 표면 지도(TSDF + Surface Nets)
    tsdf_vox: f64,
}

fn args() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let get = |k: &str| v.iter().position(|a| a == k).and_then(|i| v.get(i + 1)).cloned();
    Args {
        tum: get("--tum").unwrap_or_default().into(),
        out: get("--out").expect("--out <dir>").into(),
        seg: get("--seg").unwrap_or("models/yolo26s-seg.onnx".into()),
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
        sp_every: get("--sp-every").map_or(1, |s| s.parse().unwrap()),
        fast_pose: v.iter().any(|x| x == "--fast-pose"),
        depth_every: get("--depth-every").map_or(3, |s| s.parse().unwrap()),
        depth_lag: get("--depth-lag").map_or(0, |s| s.parse().unwrap()),
        scene_vox: get("--scene-vox").map_or(0.01, |s| s.parse().unwrap()),
        max_depth: get("--max-depth").map_or(4.0, |s| s.parse().unwrap()),
        scene_step: get("--scene-step").map_or(2, |s| s.parse().unwrap()),
        surface: v.iter().any(|x| x == "--surface"),
        // 표면 칸 3 cm: 칸마다 측정이 더 모여 D435 깊이 잡음에도 면이 이어짐(2 cm 는 바닥·벽이 얼룩), 작은 물체(키보드·마우스)는 그대로
        tsdf_vox: get("--tsdf-vox").map_or(0.03, |s| s.parse().unwrap()),
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
    let mut b = vec![false; w * h];
    for x in 0..w {
        let mut run = 0usize;
        for y in 0..h {
            run = if a[y * w + x] { run + 1 } else { 0 };
            if run > 2 * r {
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
/// 바뀐 구역의 면 파일 쓰기(mesh/r_x_y_z.ply)
fn write_surface(dir: &Path, ts: &mut tsdf::Tsdf) {
    for (reg, verts, tris) in ts.take_dirty_meshes() {
        out::ply_mesh(&dir.join(format!("mesh/r_{}_{}_{}.ply", reg.0, reg.1, reg.2)), &verts, &tris);
    }
}

/// 뷰어용 표면 목록: 구역마다 [x, y, z, 판 번호](판이 바뀐 구역만 다시 받음)
fn surface_json(ts: &tsdf::Tsdf, run: u64) -> serde_json::Value {
    let regs: Vec<[i64; 4]> = ts.ver.iter().map(|(k, v)| [k.0 as i64, k.1 as i64, k.2 as i64, *v as i64]).collect();
    json!({"run": run, "vox": ts.vox, "regions": regs})
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
}

fn run(a: &Args, seg: Box<dyn seg::Segmenter + Send>) -> Box<dyn seg::Segmenter + Send> {
    // 카메라 입력이면 빈 시퀀스(보정값만), 아니면 TUM
    let seq = if a.cam.is_some() { tum::Tum::empty(a.k.unwrap_or([477.0, 477.0, 320.0, 240.0])) } else { tum::Tum::open(&a.tum, a.gt.as_deref().map(Path::new)) };
    std::fs::create_dir_all(a.out.join("objects")).unwrap();
    let [fx, fy, cx, cy] = seq.k;
    let mut grid = Grid::new(0.05);
    let mut scene = SceneMap::new(a.scene_vox, 0.04);
    let mut tsdf = a.surface.then(|| tsdf::Tsdf::new(a.tsdf_vox));
    let mut mem = Memory::new(0.012, 0.04);
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
    let run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let _ = std::fs::remove_dir_all(a.out.join("scene"));
    let _ = std::fs::remove_dir_all(a.out.join("mesh"));
    std::fs::create_dir_all(a.out.join("mesh")).unwrap();
    std::fs::create_dir_all(a.out.join("scene")).unwrap();
    let (mut n_chunk, mut n_scene) = (0usize, 0usize);
    let mut last_pose = [0.0; 3];
    let t0 = Instant::now();
    // 앞 단계(영상·깊이 읽기 + 분할)는 작업 스레드에서 한 장 앞서 돌린다 → NPU/GPU 분할과 CPU 지도 갱신이 겹친다.
    // 실시간 재생의 속도 맞춤(앞서면 기다리고, 밀리면 건너뛰기)도 그쪽에서 한다.
    let seq = std::sync::Arc::new(seq);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(1);
    let worker = {
        let seq = seq.clone();
        let (every, conf, live, no_skip, seg_every, npu_par, sp_every) = (a.every, a.conf, a.live, a.no_skip, a.seg_every.max(1), a.npu_par, a.sp_every.max(1));
        let (fast_pose, depth_every, depth_lag) = (a.fast_pose, a.depth_every.max(1), a.depth_lag);
        let mut dnet = make_depth(&a.depth);
        // --fast-pose 는 위치를 모두 직접 방식으로 구하니 SuperPoint 는 돌리지 않는다(NPU 작업 하나 줄임)
        let mut spnet = (a.vo && !a.dvo && !a.fast_pose).then(|| make_sp(&a.sp));
        let cam_path = a.cam.clone();
        let rec_dir = a.rec.clone();
        std::thread::spawn(move || {
            let mut seg = seg;
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
                            let f = Frame { t, p: p0, r: r0, rgb, w, h, depth, dets, load: 0.0, seg: c1.elapsed().as_secs_f64(), scale: f64::NAN, feat, track_only: false };
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
                        let f = Frame { t, p: p0, r: r0, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: (c1 - c0).as_secs_f64(), seg: 0.0, scale: f64::NAN, feat: None, track_only: true };
                        if tx.send(f).is_err() {
                            break;
                        }
                        let _ = jtx.try_send((t, rgb));
                        continue;
                    }
                    if fast_pose {
                        // 녹화 최대 속도: 위치만 추적 장면을 먼저, depth_every 장마다 아래에서 모델까지
                        let f = Frame { t, p: p0, r: r0, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: (c1 - c0).as_secs_f64(), seg: 0.0, scale: f64::NAN, feat: None, track_only: true };
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
                    let f = Frame { t, p: p0, r: r0, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale: f64::NAN, feat, track_only: false };
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
                    let f = Frame { t, p, r, rgb: rgb.clone(), w, h, depth: vec![], dets: None, load: 0.0, seg: 0.0, scale: f64::NAN, feat: None, track_only: true };
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
                let f = Frame { t, p, r, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale, feat, track_only: false };
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
    let mut leveled = a.cam.is_none() || !a.vo;
    while let Ok(Frame { t, mut p, mut r, rgb, w, h, mut depth, dets, load, seg: seg_s, scale, feat, track_only }) = rx.recv() {
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
        if !leveled && vo_ok {
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
                    leveled = true;
                    eprintln!("수평 맞춤: 처음 가정이 {:.1}° 기울어져 있었음({} / {} 장 합의)", up[2].clamp(-1.0, 1.0).acos().to_degrees(), inl.len(), grav_obs.len());
                } else if grav_obs.len() > 60 {
                    grav_obs.drain(..30); // 합의가 안 되면 오래된 것부터 버리고 계속
                }
            }
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
        let h_now = if hist_h.is_empty() { 1.4 } else {
            let mut s: Vec<f64> = hist_h.iter().rev().take(15).copied().collect();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
            s[s.len() / 2]
        };
        let z0v = if leveled { *z0.get_or_insert(p[2] - h_now) } else { p[2] - h_now };
        last_z0v = z0v;
        let pwc = [p[0], p[1], p[2] - z0v];
        let floor_z = pwc[2] - h_now;
        let to_w = |pc: &V3| add(&mv(&r, pc), &pwc);

        // 점유 지도(16 화소 간격) + 3D 점 지도(3 화소 간격)
        let (mut occ, mut flo) = (vec![], vec![]);
        for v in (8..h).step_by(16) {
            for u in (8..w).step_by(16) {
                if let Some(pc) = back(u, v) {
                    let q = to_w(&pc);
                    let hg = q[2] - floor_z;
                    if hg > 0.15 && hg < 1.8 {
                        occ.push((q[0], q[1]));
                    } else if hg < 0.08 {
                        flo.push((q[0], q[1]));
                    }
                }
            }
        }
        // 위치가 확인된 장면만 지도·물체에 넣는다(실패해 예측 위치뿐인 장면은 지도를 어긋나게 함)
        let map_ok = (!a.vo || vo_ok) && leveled;
        if map_ok {
            grid.add((pwc[0], pwc[1]), &occ, &flo);
        }
        // 점 지도용 3D 점: 줄마다 나눠 상주 스레드 풀에서(표면 지도면 TSDF 가 깊이를 직접 쓰니 건너뜀)
        let sp: Vec<(V3, [u8; 3], f32)> = if a.surface || !map_ok {
            vec![]
        } else {
            use rayon::prelude::*;
            let (step, maxd) = (a.scene_step, a.max_depth);
            let rows: Vec<usize> = (0..h).step_by(step).collect();
            rows.par_iter()
                .flat_map_iter(|&v| {
                    let (depth, rgb) = (&depth, &rgb);
                    (0..w).step_by(step).filter_map(move |u| {
                        let pc = back(u, v)?;
                        (pc[2] < maxd && !depth_edge(depth, w, h, u, v)).then(|| {
                            let o = (v * w + u) * 3;
                            (to_w(&pc), [rgb[o], rgb[o + 1], rgb[o + 2]], pc[2] as f32)
                        })
                    })
                })
                .collect()
        };        if let (Some(ts), true) = (tsdf.as_mut(), map_ok) {
            ts.integrate(&depth, &rgb, w, h, seq.k, &r, &pwc, a.max_depth);
        } else if !a.surface {
            scene.add(nkf, &sp);
        }
        let c3 = Instant::now();

        // 물체
        let mut seen = HashSet::new();
        for d in dets.iter().flatten().filter(|_| map_ok) {
            let name = seg::COCO.get(d.cls).copied().unwrap_or("object");
            if name == "person" {
                continue;
            }
            let [x1, y1, x2, y2] = d.box_.map(|v| v as usize);
            let (bw, bh) = (x2 - x1, y2 - y1);
            let m = erode(&d.mask, bw, bh, 3);
            let mut pix: Vec<(usize, usize, f64)> = vec![];
            for yy in 0..bh {
                for xx in 0..bw {
                    if m[yy * bw + xx] {
                        let (u, v) = (x1 + xx, y1 + yy);
                        let z = depth[v * w + u] as f64;
                        if z > 0.2 {
                            pix.push((u, v, z));
                        }
                    }
                }
            }
            if pix.len() < 80 {
                continue;
            }
            let step = (pix.len() / 2500).max(1);
            let pix: Vec<_> = pix.into_iter().step_by(step).collect();
            let mut zs: Vec<f64> = pix.iter().map(|q| q.2).collect();
            zs.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let med = zs[zs.len() / 2];
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
            let mut crgb = Vec::with_capacity(bw * bh * 3);
            let mut cdep = Vec::with_capacity(bw * bh);
            for yy in 0..bh {
                let o = ((y1 + yy) * w + x1) * 3;
                crgb.extend_from_slice(&rgb[o..o + bw * 3]);
                for xx in 0..bw {
                    cdep.push((depth[(y1 + yy) * w + x1 + xx] * 1000.0).clamp(0.0, 65535.0) as u16);
                }
            }
            let crop = Crop { w: bw as u32, h: bh as u32, rgb: crgb, mask: m.iter().map(|&b| if b { 255 } else { 0 }).collect(), depth: cdep };
            let id = mem.update(name, d.score, &pts, med as f32, t, Some(crop), (a.depth != "kinect").then_some(pwc));
            seen.insert(id);
        }
        if dets.is_some() && map_ok {
            mem.check_missing(&seen, seq.k, &r, &pwc, &depth, w, h);
        }
        let c4 = Instant::now();

        // 지나온 길: 5 cm 이상 움직였을 때만 점 하나(가만히 있어도 장면마다 쌓이면 view.json 이 커져 원격 뷰어가 느려짐)
        if traj.last().is_none_or(|(_, q)| (q[0] - pwc[0]).hypot(q[1] - pwc[1]) >= 0.05) {
            traj.push((t, [pwc[0], pwc[1], floor_z]));
        }
        last_pose = [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])];
        cam_json = json!({"pos": pwc, "R": r});
        nkf += 1;
        // 실시간: 카메라 위치만 담은 작은 pose.json 은 장면마다(뷰어가 0.1 초마다 받아 손카메라를 바로 움직임),
        // 지도·물체·그래프(view.json)는 0.4 초마다
        // --fast-pose 는 위치만 추적한 장면이 이미 더 새 위치를 썼으니 늦게 온 깊이 장면은 쓰지 않음(손카메라가 뒤로 튀지 않게)
        if a.live > 0.0 && !a.fast_pose {
            out::write_atomic(&a.out.join("pose.json"), json!({"t": t, "cam": cam_json, "pose": last_pose, "ok": !a.vo || vo_ok}).to_string().as_bytes());
        }
        let write_now = if a.live > 0.0 { last_view.elapsed().as_secs_f64() >= 0.4 || nkf == 1 } else { nkf % 5 == 0 || nkf == 1 } || a.snaps.contains(&nkf);
        if !write_now && nkf >= last_merge + 15 {
            last_merge = nkf;
            mem.prune(t);
            mem.merge_pass();
        }
        if write_now {
            last_view = Instant::now();
            // 3D 지도 조각·표면: 실시간은 1 초마다(전에는 장면 번호가 기록 시점과 우연히 맞을 때만 나가 몇 초씩 비었다), 아니면 장면 수로
            let scene_now = if a.live > 0.0 { last_scene.elapsed().as_secs_f64() >= 1.0 } else { nkf % 50 == 0 || nkf % a.scene_every == 1 || a.scene_every == 1 };
            if scene_now || nkf == 1 || a.snaps.contains(&nkf) {
                last_scene = Instant::now();
                // 새로 확정된 칸만 조각 파일로: 뷰어는 아직 안 받은 조각만 받는다(원격 접속에서 지도 전체를 매번 받지 않게)
                let pts = scene.new_points();
                if !pts.is_empty() {
                    out::ply(&a.out.join(format!("scene/c{n_chunk}.ply")), &pts);
                    n_chunk += 1;
                    n_scene += pts.len();
                }
                scene_json = json!({"run": run_id, "chunks": n_chunk, "prefix": "scene/c", "voxel": scene.vox, "n": n_scene});
                if let Some(ts) = tsdf.as_mut() {
                    write_surface(&a.out, ts);
                    scene_json["surface"] = surface_json(ts, run_id);
                }
            }
            // 병합 후처리는 키프레임 15 장마다(물체 쌍을 매번 다 보지 않게, 중복이 자라기 전에 합침)
            let merge = nkf >= last_merge + 15;
            if merge {
                last_merge = nkf;
            }
            write_view(&a.out, &grid, &mut mem, &traj, t, last_pose, &cam_json, &scene_json, merge);
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
        if nkf % 25 == 0 {
            let n = clock.n as f64;
            println!("kf {nkf} objects {} ({} gone) | ms/kf load {:.0} seg {:.0} vo {:.0} map {:.0} obj {:.0} write {:.0} | {:.1} kf/s | vo fail {vo_fail} dvo {vo_dvo} reject {vo_reject} ref {vo_nref} pos ({:.2},{:.2},{:.2})",
                     mem.objs.len(), mem.objs.iter().filter(|o| o.state == "gone").count(),
                     clock.load / n * 1e3, clock.seg / n * 1e3, clock.vo / n * 1e3, clock.map / n * 1e3, clock.obj / n * 1e3, clock.write / n * 1e3,
                     n / t0.elapsed().as_secs_f64(), p[0], p[1], p[2]);
        }
    }
    if !est_traj.is_empty() {
        let mut s = String::from("# timestamp x y z qx qy qz qw (OpenCV 카메라 → 세계)\n");
        for (t, p, r) in &est_traj {
            let q = quat_of(r);
            s += &format!("{t:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n", p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
        }
        let _ = std::fs::write(a.out.join("poses_est.txt"), s);
    }
    let seg = worker.join().expect("worker");
    let pts = scene.new_points();
    if !pts.is_empty() {
        out::ply(&a.out.join(format!("scene/c{n_chunk}.ply")), &pts);
        n_chunk += 1;
        n_scene += pts.len();
    }
    scene_json = json!({"run": run_id, "chunks": n_chunk, "prefix": "scene/c", "voxel": scene.vox, "n": n_scene});
    if let Some(ts) = tsdf.as_mut() {
        write_surface(&a.out, ts);
        scene_json["surface"] = surface_json(ts, run_id);
    }
    let pts = scene.points();
    let t_end = traj.last().map_or(0.0, |x| x.0);
    for o in mem.objs.iter_mut().filter(|o| o.dirty) {
        o.cells_saved = 0; // 마지막 기록은 바뀐 물체를 모두 내보냄
    }
    write_view(&a.out, &grid, &mut mem, &traj, t_end, last_pose, &cam_json, &scene_json, true);
    let n = clock.n.max(1) as f64;
    let mut by: std::collections::BTreeMap<&str, u32> = Default::default();
    for o in &mem.objs {
        *by.entry(o.name.as_str()).or_default() += 1;
    }
    println!("{}", json!({"keyframes": nkf, "objects": mem.objs.len(), "gone": mem.objs.iter().filter(|o| o.state == "gone").count(),
        "scene_points": pts.len(), "ms_per_kf": {"load": clock.load / n * 1e3, "seg": clock.seg / n * 1e3, "vo": clock.vo / n * 1e3, "map": clock.map / n * 1e3,
        "obj": clock.obj / n * 1e3, "write": clock.write / n * 1e3}, "kf_per_s": n / t0.elapsed().as_secs_f64(), "by_name": by, "depth_scale_vs_kinect": scale_stats(&scales), "vo": vo_report(&ate, vo_fail), "vo_reject": vo_reject, "vo_keyframes": vo_nref, "vo_dvo": vo_dvo, "pose_only_frames": n_fast, "pose_only_ms": t_fast / (n_fast.max(1) as f64) * 1e3, "merged": mem.merged}));
    seg
}

#[allow(clippy::too_many_arguments)]
fn write_view(dir: &Path, grid: &Grid, mem: &mut Memory, traj: &[(f64, [f64; 3])], stamp: f64, pose: [f64; 3],
              cam: &serde_json::Value, scene: &serde_json::Value, merge: bool) {
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
            o.dirty && (o.cells_saved == 0 || o.cells.len() * 20 > o.cells_saved * 21)
        };
        let pts = if grown { Some(mem.refresh(i)) } else { None };
        let o = &mut mem.objs[i];
        let base = format!("objects/O{}", o.id);
        if let Some(pts) = pts {
            out::ply(&dir.join(format!("{base}_points.ply")), &pts);
            o.n_pts = pts.len();
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
            }
            o.crop_saved = o.crop_ver;
        }
        let mut e = json!({"id": o.id, "name": o.name, "state": o.state, "pos": o.pos, "first_pos": o.first_pos, "extent": o.extent,
            "n_obs": o.n_obs, "score": o.score, "last_seen": o.last_seen, "structural": false,
            "points": {"path": format!("{base}_points.ply"), "stamp": o.stamp, "voxel": mem.vox, "n": o.n_pts}});
        if o.crop.is_some() {
            e["rgbd"] = json!({"rgb": format!("{base}_rgb.png"), "mask": format!("{base}_mask.png"), "depth": format!("{base}_depth.png"), "v": o.crop_ver});
        }
        objs.push(e);
    }
    let (grid_j, rooms, edges) = match &gi {
        Some(g) => {
            out::pgm(&dir.join("map.pgm"), g.w, g.h, &g.px);
            let ol: Vec<_> = mem.objs.iter().map(|o| (o.id, o.pos, o.extent, o.state == "gone", o.name.as_str())).collect();
            let (rj, ej) = graph::build(g, grid.res, &ol);
            (json!({"width": g.w, "height": g.h, "resolution": grid.res, "origin": [g.x0 as f64 * grid.res, g.y0 as f64 * grid.res]}), rj, ej)
        }
        None => (json!(null), json!([]), json!([])),
    };
    let nodes: Vec<_> = traj.iter().enumerate().map(|(i, (t, p))| json!({"id": format!("A{i}"), "kind": "agent", "pos": p, "t": t})).collect();
    let mut view = json!({"stamp": stamp, "pose": pose, "grid": grid_j, "objects": objs, "rooms": rooms,
        "graph": {"nodes": nodes, "edges": edges}, "robot": "handheld camera", "cam": cam});
    if !scene.is_null() {
        view["scene_points"] = scene.clone();
    }
    out::write_atomic(&dir.join("view.json"), serde_json::to_string(&view).unwrap().as_bytes());
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

fn main() {
    let a = args();
    if a.live > 0.0 {
        out::start_writer(); // 실시간: 파일 쓰기는 기록 스레드가
    }
    let mut seg = make_seg(&a.seg);
    loop {
        seg = run(&a, seg);
        if !a.looping {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
