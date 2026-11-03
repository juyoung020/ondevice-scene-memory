//! 손카메라 영상 + 포즈 → 점유 지도·3D 점 지도·물체 기억·씬그래프(방·엣지) → 기억 폴더(view.json, sgview 가 읽음).
//!
//!   memmap --tum data/tum/<seq> --out data/memory/<name> [--seg models/yolo26s-seg.onnx] [--every 5] [--conf 0.35]
//!          [--live 1] [--no-skip] [--loop] [--scene-every 10] [--snaps 150,500]
//!
//! 지도·트래커는 누적식(maps.rs): 키프레임마다 새 점만 해시 칸에 더하고, 기록은 칸을 한 번 훑기만 한다.
//! 깊이는 1 단계로 TUM Kinect 실측을 쓴다(딥러닝 깊이는 다음 단계).
#[cfg(feature = "cam")]
mod cam;
mod geom;
mod graph;
mod maps;
mod out;
mod scale;
mod seg;
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
    vo_ess: bool,
    cam: Option<String>,
    k: Option<[f64; 4]>,
    sp: String,
    snaps: Vec<u32>,
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
        vo: get("--pose").is_some_and(|p| p == "vo"),
        vo_ess: v.iter().any(|x| x == "--vo-ess"),
        cam: get("--cam"),
        k: get("--k").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2], v[3]] }),
        sp: get("--superpoint").unwrap_or("superpoint_480x640.dxnn".into()),
        snaps: get("--snaps").map_or(vec![], |s| s.split(',').filter_map(|x| x.parse().ok()).collect()),
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
    map: f64,
    obj: f64,
    write: f64,
    n: u32,
}

fn make_seg(path: &str) -> Box<dyn seg::Segmenter + Send> {
    #[cfg(feature = "dxrt")]
    if path.ends_with(".dxnn") {
        return Box::new(seg::DxSeg::new(path));
    }
    #[cfg(feature = "onnx")]
    {
        let env = Box::leak(Box::new(ort::init().build().expect("onnxruntime")));
        return Box::new(seg::OnnxSeg::new(env, path));
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
    dets: Vec<seg::Det>,
    load: f64,
    seg: f64,
    scale: f64, // 딥러닝 깊이 / Kinect 중앙값(비교용, Kinect 깊이를 쓸 땐 NaN)
    feat: Option<vo::Feat>,
}

fn run(a: &Args, seg: Box<dyn seg::Segmenter + Send>) -> Box<dyn seg::Segmenter + Send> {
    // 카메라 입력이면 빈 시퀀스(보정값만), 아니면 TUM
    let seq = if a.cam.is_some() { tum::Tum::empty(a.k.unwrap_or([450.0, 450.0, 320.0, 240.0])) } else { tum::Tum::open(&a.tum) };
    std::fs::create_dir_all(a.out.join("objects")).unwrap();
    let [fx, fy, cx, cy] = seq.k;
    let mut grid = Grid::new(0.05);
    let mut scene = SceneMap::new(0.01, 0.04);
    let mut mem = Memory::new(0.012, 0.04);
    let mut traj: Vec<(f64, [f64; 3])> = vec![];
    let mut hist_h: Vec<f64> = vec![];
    let mut z0: Option<f64> = None;
    let (mut t_start, mut w_start) = (None::<f64>, Instant::now());
    let mut clock = Clock::default();
    let mut nkf: u32 = 0;
    let mut cam_json = json!(null);
    let mut scene_json = json!(null);
    let mut last_pose = [0.0; 3];
    let t0 = Instant::now();
    // 앞 단계(영상·깊이 읽기 + 분할)는 작업 스레드에서 한 장 앞서 돌린다 → NPU/GPU 분할과 CPU 지도 갱신이 겹친다.
    // 실시간 재생의 속도 맞춤(앞서면 기다리고, 밀리면 건너뛰기)도 그쪽에서 한다.
    let seq = std::sync::Arc::new(seq);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(1);
    let worker = {
        let seq = seq.clone();
        let (every, conf, live, no_skip) = (a.every, a.conf, a.live, a.no_skip);
        #[cfg(feature = "dxrt")]
        let mut dnet = a.depth.ends_with(".dxnn").then(|| seg::DxDepth::new(&a.depth));
        #[cfg(not(feature = "dxrt"))]
        assert!(a.depth == "kinect" && !a.vo, "NPU depth / --pose vo need --features dxrt");
        #[cfg(feature = "dxrt")]
        let mut spnet = a.vo.then(|| seg::DxSuperPoint::new(&a.sp));
        let cam_path = a.cam.clone();
        std::thread::spawn(move || {
            let mut seg = seg;
            #[cfg(all(feature = "cam", feature = "dxrt"))]
            if let Some(cp) = cam_path {
                // 실시간 카메라: 깊이·분할·특징점 모두 NPU, 포즈는 vo 가 만든다(첫 장면: 수평으로 앞을 봄, 높이 1.3 m)
                let mut cam = cam::Cam::open(&cp, 640, 480);
                let (w, h) = (cam.w, cam.h);
                let r0 = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
                let p0 = [0.0, 0.0, 1.3];
                let tc = Instant::now();
                loop {
                    let c0 = Instant::now();
                    let rgb = cam.grab();
                    let t = tc.elapsed().as_secs_f64();
                    let c1 = Instant::now();
                    let depth = dnet.as_mut().expect("--cam needs --depth <.dxnn>").run(&rgb, w, h);
                    let dets = seg.run(&rgb, w, h, conf);
                    let feat = spnet.as_mut().map(|n| n.run(&rgb, w, h));
                    let c2 = Instant::now();
                    let f = Frame { t, p: p0, r: r0, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale: f64::NAN, feat };
                    if tx.send(f).is_err() {
                        break;
                    }
                }
                return seg;
            }
            #[cfg(not(all(feature = "cam", feature = "dxrt")))]
            assert!(cam_path.is_none(), "--cam needs --features dxrt,cam");
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
                let Some((dw, dh, kin)) = seq.kinect_depth(t) else { continue };
                assert_eq!((dw as usize, dh as usize), (w, h));
                let (depth, scale) = {
                    #[cfg(feature = "dxrt")]
                    {
                        match dnet.as_mut() {
                            Some(n) => {
                                let d = n.run(&rgb, w, h);
                                let mut r: Vec<f32> = d.iter().zip(&kin).filter(|(_, k)| **k > 0.3 && **k < 5.0).step_by(37).map(|(p, k)| p / k).collect();
                                r.sort_by(|x, y| x.partial_cmp(y).unwrap());
                                let sc = if r.is_empty() { f64::NAN } else { r[r.len() / 2] as f64 };
                                (d, sc)
                            }
                            None => (kin, f64::NAN),
                        }
                    }
                    #[cfg(not(feature = "dxrt"))]
                    {
                        (kin, f64::NAN)
                    }
                };
                let c1 = Instant::now();
                let dets = seg.run(&rgb, w, h, conf);
                #[cfg(feature = "dxrt")]
                let feat = spnet.as_mut().map(|n| n.run(&rgb, w, h));
                #[cfg(not(feature = "dxrt"))]
                let feat = None;
                let c2 = Instant::now();
                let f = Frame { t, p, r, rgb, w, h, depth, dets, load: (c1 - c0).as_secs_f64(), seg: (c2 - c1).as_secs_f64(), scale, feat };
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
    let mut vo_prev: Option<(vo::Feat, Vec<f32>, M3, V3)> = None;
    let mut vo_pose: Option<(M3, V3)> = None;
    let mut ate: Vec<(V3, V3)> = vec![];
    let mut vo_fail = 0u32;
    while let Ok(Frame { t, mut p, mut r, rgb, w, h, mut depth, dets, load, seg: seg_s, scale, feat }) = rx.recv() {
        let (p_gt, _r_gt) = (p, r);
        // 카메라만으로 포즈(--pose vo): 이전 키프레임과 SuperPoint 짝 → 이전 장면 깊이로 3D → 재투영 최소화. 첫 장면만 정답 포즈로 시작.
        if a.vo {
            let [fx, fy, cx, cy] = seq.k;
            let mut est = None;
            if let (Some(f), Some((pf, pdep, pr, pp))) = (&feat, &vo_prev) {
                let ms = vo::match_mutual(pf, f, 0.7);
                let (mut world, mut cam, mut uv) = (vec![], vec![], vec![]);
                for (i, j) in ms {
                    let (u0, v0) = pf.kp[i];
                    let z0 = pdep[v0 as usize * w + u0 as usize] as f64;
                    if z0 < 0.2 || z0 > 6.0 || depth_edge(pdep, w, h, u0 as usize, v0 as usize) {
                        continue; // 깊이 경계(물체 테두리)의 점은 딥러닝 깊이가 크게 틀린다
                    }
                    let xw = add(&mv(pr, &[(u0 as f64 - cx) / fx * z0, (v0 as f64 - cy) / fy * z0, z0]), pp);
                    let (u1, v1) = f.kp[j];
                    let z1 = depth[v1 as usize * w + u1 as usize] as f64;
                    cam.push((z1 > 0.2 && z1 < 6.0).then(|| [(u1 as f64 - cx) / fx * z1, (v1 as f64 - cy) / fy * z1, z1]));
                    world.push(xw);
                    uv.push((u1 as f64, v1 as f64));
                }
                // 기본: 두 장면 기하로 회전·이동 방향(깊이 안 씀) + 이동 거리만 이전 장면 깊이로. 실패하면 3D-2D(PnP)로.
                let ms2 = vo::match_mutual(pf, f, 0.7);
                let n0: Vec<(f64, f64)> = ms2.iter().map(|&(i, _)| ((pf.kp[i].0 as f64 - cx) / fx, (pf.kp[i].1 as f64 - cy) / fy)).collect();
                let n1: Vec<(f64, f64)> = ms2.iter().map(|&(_, j)| ((f.kp[j].0 as f64 - cx) / fx, (f.kp[j].1 as f64 - cy) / fy)).collect();
                if a.vo_ess {
                    if let Some((rr, tt, ids, zt)) = vo::relative_pose(&n0, &n1) {
                        let mut rs: Vec<f64> = ids.iter().zip(&zt).filter_map(|(&k, &z)| {
                            let (u0, v0) = pf.kp[ms2[k].0];
                            let zd = pdep[v0 as usize * w + u0 as usize] as f64;
                            (zd > 0.2 && zd < 6.0 && z > 1e-6).then(|| zd / z)
                        }).collect();
                        if rs.len() >= 15 {
                            rs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                            let s = rs[rs.len() / 2];
                            let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| rr[j][i])); // Rrelᵀ
                            let rc = mm(pr, &rt);
                            let tc = mv(&rc, &tt.map(|x| x * s));
                            est = Some((rc, sub(pp, &tc), ids.len()));
                        }
                    }
                }
                if est.is_none() {
                    est = vo::estimate(&world, &cam, &uv, seq.k, vo_pose.unwrap_or((r, p)));
                }
            }
            // 배율 이어 붙이기: 짝지은 같은 점의 (이전 장면 깊이로 본 거리) / (이번 장면 예측 깊이) 중앙값으로 이번 장면 깊이를 맞춘다
            if let (Some((r2, p2, _)), true) = (&est, a.depth != "kinect") {
                if let (Some(f), Some((pf, pdep, pr, pp))) = (&feat, &vo_prev) {
                    let mut rs = vec![];
                    for (i, j) in vo::match_mutual(pf, f, 0.7) {
                        let (u0, v0) = pf.kp[i];
                        let z0 = pdep[v0 as usize * w + u0 as usize] as f64;
                        let (u1, v1) = f.kp[j];
                        let z1 = depth[v1 as usize * w + u1 as usize] as f64;
                        if z0 < 0.2 || z1 < 0.2 {
                            continue;
                        }
                        let xw = add(&mv(pr, &[(u0 as f64 - cx) / fx * z0, (v0 as f64 - cy) / fy * z0, z0]), pp);
                        let d = sub(&xw, p2);
                        let zc = r2[0][2] * d[0] + r2[1][2] * d[1] + r2[2][2] * d[2];
                        if zc > 0.2 {
                            rs.push(zc / z1);
                        }
                    }
                    if rs.len() >= 20 {
                        rs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                        let s = rs[rs.len() / 2].clamp(0.5, 2.0) as f32;
                        for d in depth.iter_mut() {
                            *d *= s;
                        }
                    }
                }
            }
            match (est, vo_pose) {
                (Some((r2, p2, _)), _) => vo_pose = Some((r2, p2)),
                (None, None) => vo_pose = Some((r, p)),
                (None, Some(_)) => vo_fail += 1,
            }
            let (r2, p2) = vo_pose.unwrap();
            r = r2;
            p = p2;
            ate.push((p, p_gt));
        }
        // 딥러닝 깊이 배율: 두 장면 삼각측량(포즈를 알 때)으로 실제 거리를 재서 맞춘다(scale.rs). 최근 5 개 중앙값으로 평활.
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
                vo_prev = Some((f, depth.clone(), r, p));
            }
        }
        let c2 = Instant::now();
        clock.load += load;
        clock.seg += seg_s;

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
        let z0v = *z0.get_or_insert(p[2] - h_now);
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
        grid.add((pwc[0], pwc[1]), &occ, &flo);
        let mut sp = Vec::with_capacity(w * h / 9);
        for v in (1..h).step_by(3) {
            for u in (1..w).step_by(3) {
                if let Some(pc) = back(u, v) {
                    if pc[2] < 4.0 {
                        let o = (v * w + u) * 3;
                        sp.push((to_w(&pc), [rgb[o], rgb[o + 1], rgb[o + 2]], pc[2] as f32));
                    }
                }
            }
        }
        scene.add(nkf, &sp);
        let c3 = Instant::now();

        // 물체
        let mut seen = HashSet::new();
        for d in &dets {
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
            let id = mem.update(name, d.score, &pts, med as f32, t, Some(crop));
            seen.insert(id);
        }
        mem.check_missing(&seen, seq.k, &r, &pwc, &depth, w, h);
        let c4 = Instant::now();

        traj.push((t, [pwc[0], pwc[1], floor_z]));
        last_pose = [pwc[0], pwc[1], (r[1][2]).atan2(r[0][2])];
        cam_json = json!({"pos": pwc, "R": r});
        nkf += 1;
        let write_now = a.live > 0.0 || nkf % 5 == 0 || nkf == 1 || a.snaps.contains(&nkf);
        if write_now {
            if nkf % a.scene_every == 1 || a.scene_every == 1 || a.snaps.contains(&nkf) || a.live == 0.0 && nkf % 50 == 0 {
                let pts = scene.points();
                out::ply(&a.out.join("scene_points.ply"), &pts);
                scene_json = json!({"path": "scene_points.ply", "stamp": nkf, "voxel": scene.vox, "n": pts.len()});
            }
            write_view(&a.out, &grid, &mut mem, &traj, t, last_pose, &cam_json, &scene_json);
        }
        if a.snaps.contains(&nkf) {
            let sd = PathBuf::from(format!("{}_snap", a.out.display())).join(format!("kf{nkf:04}"));
            copy_dir(&a.out, &sd);
            out::png_rgb(&sd.join("frame.png"), w as u32, h as u32, &rgb);
        }
        let c5 = Instant::now();
        clock.map += (c3 - c2).as_secs_f64();
        clock.obj += (c4 - c3).as_secs_f64();
        clock.write += (c5 - c4).as_secs_f64();
        clock.n += 1;
        if nkf % 25 == 0 {
            let n = clock.n as f64;
            println!("kf {nkf} objects {} ({} gone) | ms/kf load {:.0} seg {:.0} map {:.0} obj {:.0} write {:.0} | {:.1} kf/s",
                     mem.objs.len(), mem.objs.iter().filter(|o| o.state == "gone").count(),
                     clock.load / n * 1e3, clock.seg / n * 1e3, clock.map / n * 1e3, clock.obj / n * 1e3, clock.write / n * 1e3,
                     n / t0.elapsed().as_secs_f64());
        }
    }
    let seg = worker.join().expect("worker");
    let pts = scene.points();
    out::ply(&a.out.join("scene_points.ply"), &pts);
    scene_json = json!({"path": "scene_points.ply", "stamp": nkf + 1, "voxel": scene.vox, "n": pts.len()});
    let t_end = traj.last().map_or(0.0, |x| x.0);
    write_view(&a.out, &grid, &mut mem, &traj, t_end, last_pose, &cam_json, &scene_json);
    let n = clock.n.max(1) as f64;
    let mut by: std::collections::BTreeMap<&str, u32> = Default::default();
    for o in &mem.objs {
        *by.entry(o.name.as_str()).or_default() += 1;
    }
    println!("{}", json!({"keyframes": nkf, "objects": mem.objs.len(), "gone": mem.objs.iter().filter(|o| o.state == "gone").count(),
        "scene_points": pts.len(), "ms_per_kf": {"load": clock.load / n * 1e3, "seg": clock.seg / n * 1e3, "map": clock.map / n * 1e3,
        "obj": clock.obj / n * 1e3, "write": clock.write / n * 1e3}, "kf_per_s": n / t0.elapsed().as_secs_f64(), "by_name": by, "depth_scale_vs_kinect": scale_stats(&scales), "vo": vo_report(&ate, vo_fail)}));
    seg
}

#[allow(clippy::too_many_arguments)]
fn write_view(dir: &Path, grid: &Grid, mem: &mut Memory, traj: &[(f64, [f64; 3])], stamp: f64, pose: [f64; 3],
              cam: &serde_json::Value, scene: &serde_json::Value) {
    let gi = grid.image();
    let mut objs = vec![];
    for i in 0..mem.objs.len() {
        let pts = if mem.objs[i].dirty { Some(mem.refresh(i)) } else { None };
        let o = &mut mem.objs[i];
        let base = format!("objects/O{}", o.id);
        if let Some(pts) = pts {
            out::ply(&dir.join(format!("{base}_points.ply")), &pts);
            if let Some(c) = &o.crop {
                out::png_rgb(&dir.join(format!("{base}_rgb.png")), c.w, c.h, &c.rgb);
                out::png_gray(&dir.join(format!("{base}_mask.png")), c.w, c.h, &c.mask);
                out::png_u16(&dir.join(format!("{base}_depth.png")), c.w, c.h, &c.depth);
            }
            o.dirty = false;
            o.stamp += 1;
        }
        let mut e = json!({"id": o.id, "name": o.name, "state": o.state, "pos": o.pos, "first_pos": o.first_pos, "extent": o.extent,
            "n_obs": o.n_obs, "score": o.score, "last_seen": o.last_seen, "structural": false,
            "points": {"path": format!("{base}_points.ply"), "stamp": o.stamp, "voxel": mem.vox}});
        if o.crop.is_some() {
            e["rgbd"] = json!({"rgb": format!("{base}_rgb.png"), "mask": format!("{base}_mask.png"), "depth": format!("{base}_depth.png")});
        }
        objs.push(e);
    }
    let (grid_j, rooms, edges) = match &gi {
        Some(g) => {
            out::pgm(&dir.join("map.pgm"), g.w, g.h, &g.px);
            let ol: Vec<_> = mem.objs.iter().map(|o| (o.id, o.pos, o.extent, o.state == "gone")).collect();
            let (rj, ej) = graph::build(g, grid.res, &ol, &|r| (format!("room {r}"), 0.0));
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

fn main() {
    let a = args();
    let mut seg = make_seg(&a.seg);
    loop {
        seg = run(&a, seg);
        if !a.looping {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
}
