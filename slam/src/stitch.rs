//! 지도 잇기(ORB-SLAM3 백엔드): 빈틈 · 놓침 뒤 새로 시작한 지도와 예전 지도를 신경망 장소 인식으로 이어 붙일 변환을 찾는다.
//! ORB 특징의 재위치는 다른 때 다른 거리 · 방향에서 본 같은 자리를 거의 못 맞춘다(강의실 녹화: 다른 지도 키프레임과 ORB 짝 15 개 미만이 99 %).
//!
//! 한 스레드("stitch")가 장면을 받는다(위치가 나온 장면 0.5 초마다): MegaLoc 기술자 · XFeat 점 · 깊이 3D · IMU 위쪽을 남기고,
//! 지금 다른 지도에 든 예전 장면 가운데 MegaLoc 상위 후보를 XFeat 짝 + 깊이 + 중력 고정 4 자유도 검증(lcverify)으로 확인한다.
//! 맞으면 두 장면의 키프레임 · 상대 위치와 광학 좌표 변환(질의 ← 후보)을 돌려준다. 지도 좌표로 바꾸는 것은 부르는 쪽(키프레임 위치를 앎).
use crate::lcnet::LcConfig;
use crate::lcverify::{self, Intr, V3};
use crate::nnrun::{self, Net};
use crate::place::{self, Head, PlaceDb};
use crate::xfeat::{self, XFeatures};
use crate::DepthBuf;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

/// 3×4 행 우선 강체 변환
pub type T34 = [f64; 12];

/// 장면 하나
pub struct Node {
    pub seq: u64,
    pub t: f64,
    /// 기준 키프레임과 그 키프레임에서 본 장면 위치(ORB 키프레임 카메라 ← 장면 카메라)
    pub kf: u64,
    pub rel: T34,
    /// 640×480 컬러, bgr = B·G·R 순서
    pub color: Vec<u8>,
    pub bgr: bool,
    /// 컬러 격자 깊이(mm)
    pub depth: DepthBuf,
    pub k: Intr,
    pub w: usize,
    pub h: usize,
    /// IMU 위쪽(카메라 광학 축, 단위 벡터)
    pub up: V3,
    /// 지금 키프레임 → 지도 id(후보를 다른 지도로 거름)
    pub maps: Arc<HashMap<u64, u64>>,
}

/// 맞은 짝: 질의 장면(kf, rel) 와 후보 장면(kf, rel), 광학 좌표 변환 질의 ← 후보
#[derive(Clone, Debug)]
pub struct Hit {
    pub t: f64,
    pub q_kf: u64,
    pub q_rel: T34,
    pub c_kf: u64,
    pub c_rel: T34,
    pub c_t: f64,
    pub t_qc: T34,
    pub inliers: usize,
    pub score: f32,
}

struct Stored {
    t: f64,
    kf: u64,
    rel: T34,
    up: V3,
    kps: Vec<[f32; 2]>,
    desc: Vec<i8>,
    x3: Vec<Option<V3>>,
    /// 깊이 작은 격자(DW × DH, mm)와 그 격자의 내부값
    small_d: Vec<u16>,
    k: Intr,
}

/// 깊이 맞춤 검사용 작은 격자(8 화소마다)
const DS: usize = 8;

/// 두 장면 깊이가 변환 t_qc(질의 ← 후보)로 겹치는지: 후보 격자 점을 질의 영상에 투영해 질의 깊이와 비교.
/// 돌려줌: (겹친 칸 비율, 겹친 칸 가운데 깊이가 맞는 비율). 맞음 = |차| < 5 cm + 깊이의 3 %
fn depth_agree(qd: &[u16], cd: &[u16], w: usize, h: usize, kq: &Intr, kc: &Intr, t: &T34) -> (f64, f64) {
    depth_agree_tol(qd, cd, w, h, kq, kc, t, 0.05, 0.03)
}

fn depth_agree_tol(qd: &[u16], cd: &[u16], w: usize, h: usize, kq: &Intr, kc: &Intr, t: &T34, a0: f64, a1: f64) -> (f64, f64) {
    let (gw, gh) = (w / DS, h / DS);
    let (mut n, mut over, mut good) = (0usize, 0usize, 0usize);
    for v in 0..gh {
        for u in 0..gw {
            let z = cd[v * gw + u] as f64 / 1000.0;
            if !(z > 0.2 && z < 6.0) {
                continue;
            }
            n += 1;
            let (px, py) = ((u * DS + DS / 2) as f64, (v * DS + DS / 2) as f64);
            let x = [(px - kc.cx) * z / kc.fx, (py - kc.cy) * z / kc.fy, z];
            let y = [0, 1, 2].map(|i| t[i * 4] * x[0] + t[i * 4 + 1] * x[1] + t[i * 4 + 2] * x[2] + t[i * 4 + 3]);
            if y[2] < 0.2 {
                continue;
            }
            let (qu, qv) = (kq.fx * y[0] / y[2] + kq.cx, kq.fy * y[1] / y[2] + kq.cy);
            if qu < 0.0 || qv < 0.0 {
                continue;
            }
            let (gu, gv) = ((qu as usize) / DS, (qv as usize) / DS);
            if gu >= gw || gv >= gh {
                continue;
            }
            let zq = qd[gv * gw + gu] as f64 / 1000.0;
            if !(zq > 0.2 && zq < 6.0) {
                continue;
            }
            over += 1;
            if (zq - y[2]).abs() < a0 + a1 * y[2] {
                good += 1;
            }
        }
    }
    (over as f64 / n.max(1) as f64, good as f64 / over.max(1) as f64)
}

/// 깊이 ICP(투영 짝 · 점 대 점, Kabsch): 후보 격자 점을 질의 영상에 투영해 같은 칸 질의 점과 짝짓고 강체 변환을 다시 구함.
/// 짝 거리 문턱 0.3 → 0.15 → 0.08 m, 각 4 번. 짝이 30 개 미만이면 그대로
fn icp(qd: &[u16], cd: &[u16], w: usize, h: usize, kq: &Intr, kc: &Intr, t0: &T34) -> T34 {
    use nalgebra::{Matrix3, Vector3};
    let (gw, gh) = (w / DS, h / DS);
    let pt = |d: &[u16], k: &Intr, u: usize, v: usize| -> Option<Vector3<f64>> {
        let z = d[v * gw + u] as f64 / 1000.0;
        (z > 0.2 && z < 6.0).then(|| Vector3::new(((u * DS + DS / 2) as f64 - k.cx) * z / k.fx, ((v * DS + DS / 2) as f64 - k.cy) * z / k.fy, z))
    };
    let mut t = *t0;
    for &th in &[0.3, 0.3, 0.3, 0.3, 0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for v in 0..gh {
            for u in 0..gw {
                let Some(x) = pt(cd, kc, u, v) else { continue };
                let y = Vector3::new(
                    t[0] * x[0] + t[1] * x[1] + t[2] * x[2] + t[3],
                    t[4] * x[0] + t[5] * x[1] + t[6] * x[2] + t[7],
                    t[8] * x[0] + t[9] * x[1] + t[10] * x[2] + t[11],
                );
                if y[2] < 0.2 {
                    continue;
                }
                let (qu, qv) = (kq.fx * y[0] / y[2] + kq.cx, kq.fy * y[1] / y[2] + kq.cy);
                if qu < 0.0 || qv < 0.0 || (qu as usize) / DS >= gw || (qv as usize) / DS >= gh {
                    continue;
                }
                let Some(q) = pt(qd, kq, (qu as usize) / DS, (qv as usize) / DS) else { continue };
                if (q - y).norm() < th {
                    a.push(x);
                    b.push(q);
                }
            }
        }
        if a.len() < 30 {
            return t;
        }
        let n = a.len() as f64;
        let ca = a.iter().sum::<Vector3<f64>>() / n;
        let cb = b.iter().sum::<Vector3<f64>>() / n;
        let mut hm = Matrix3::zeros();
        for (x, y) in a.iter().zip(&b) {
            hm += (x - ca) * (y - cb).transpose();
        }
        let svd = hm.svd(true, true);
        let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
        let mut r = vt.transpose() * u.transpose();
        if r.determinant() < 0.0 {
            let mut v = vt.transpose();
            v.column_mut(2).neg_mut();
            r = v * u.transpose();
        }
        let tr = cb - r * ca;
        t = [r[(0, 0)], r[(0, 1)], r[(0, 2)], tr[0], r[(1, 0)], r[(1, 1)], r[(1, 2)], tr[1], r[(2, 0)], r[(2, 1)], r[(2, 2)], tr[2]];
    }
    t
}

fn small_depth(d: &[u16], w: usize, h: usize) -> Vec<u16> {
    let (gw, gh) = (w / DS, h / DS);
    (0..gh).flat_map(|v| (0..gw).map(move |u| d[(v * DS + DS / 2) * w + u * DS + DS / 2])).collect()
}

pub struct Stitch {
    tx: Option<Sender<Node>>,
    rx: Receiver<Option<Hit>>,
    th: Option<std::thread::JoinHandle<()>>,
    pub pending: usize,
    pub sent: u64,
    pub hits: u64,
}

const Q8: f32 = 127.0;
/// 후보 수(다른 지도만), MegaLoc 내적 최소
const TOPK: usize = 12;
const MIN_SCORE: f32 = 0.25;
/// 받을 맞은 점 수(틀린 짝은 부르는 쪽이 두 번 같은 변환으로 거름)
const MIN_INLIERS: usize = 20;
/// 깊이 맞춤: 후보 깊이 점 가운데 질의 영상 안에 들어 깊이가 있는 비율 · 그 가운데 깊이가 맞는 비율
const DEPTH_OVERLAP: f64 = 0.2;
const DEPTH_AGREE: f64 = 0.6;
// (ICP 뒤 강의실 녹화에서 영상으로 같은 곳인 짝 0.67–0.99, 다른 모니터 줄 짝 0.29–0.59)

impl Stitch {
    pub fn start(models: &str) -> Result<Stitch, String> {
        let cfg = LcConfig::from_dir(models);
        let (tx, rx) = channel::<Node>();
        let (otx, orx) = channel::<Option<Hit>>();
        let (rtx, rrx) = channel::<Result<(), String>>();
        let th = std::thread::Builder::new()
            .name("stitch".into())
            .spawn(move || {
                let mut st = match State::open(&cfg) {
                    Ok(s) => {
                        let _ = rtx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = rtx.send(Err(e));
                        return;
                    }
                };
                while let Ok(n) = rx.recv() {
                    let o = st.step(n);
                    if otx.send(o).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        rrx.recv().map_err(|e| e.to_string())??;
        Ok(Stitch { tx: Some(tx), rx: orx, th: Some(th), pending: 0, sent: 0, hits: 0 })
    }

    pub fn send(&mut self, n: Node) {
        if let Some(tx) = &self.tx
            && tx.send(n).is_ok()
        {
            self.pending += 1;
            self.sent += 1;
        }
    }

    /// 끝난 결과(맞은 짝이 있으면 Some). block = 하나 끝날 때까지 기다림
    pub fn recv(&mut self, block: bool) -> Option<Option<Hit>> {
        if self.pending == 0 {
            return None;
        }
        let r = if block { self.rx.recv().ok() } else { self.rx.try_recv().ok() };
        if let Some(h) = &r {
            self.pending -= 1;
            self.hits += h.is_some() as u64;
        }
        r
    }
}

impl Drop for Stitch {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

struct State {
    cfg: LcConfig,
    ml: Net,
    xf: Net,
    head: Head,
    db: PlaceDb,
    nodes: Vec<Stored>,
    small: Vec<u8>,
    patches: Vec<f32>,
    rgb: Vec<u8>,
    gray: Vec<f32>,
    q8: Vec<u8>,
    out_ml: Vec<f32>,
    out_xf: Vec<f32>,
    desc: Vec<f32>,
    xsc: xfeat::Scratch,
    sim: Vec<f32>,
    pairs: Vec<(u32, u32)>,
    /// 시험용 기록(환경 변수 STITCH_LOG = 파일): 검증해 본 후보마다
    log: Option<std::fs::File>,
}

impl State {
    fn open(cfg: &LcConfig) -> Result<State, String> {
        let ml = Net::open(&cfg.megaloc, cfg.threads, 0)?;
        let xf = Net::open(&cfg.xfeat, cfg.threads, 0)?;
        let head = Head::load(&cfg.head)?;
        Ok(State {
            db: PlaceDb::new(head.dim),
            cfg: cfg.clone(),
            ml,
            xf,
            head,
            nodes: Vec::new(),
            small: Vec::new(),
            patches: Vec::new(),
            rgb: Vec::new(),
            gray: Vec::new(),
            q8: Vec::new(),
            out_ml: Vec::new(),
            out_xf: Vec::new(),
            desc: Vec::new(),
            xsc: xfeat::Scratch::default(),
            sim: Vec::new(),
            pairs: Vec::new(),
            log: std::env::var("STITCH_LOG").ok().and_then(|p| std::fs::File::create(p).ok()),
        })
    }

    fn step(&mut self, n: Node) -> Option<Hit> {
        let (w, h) = (n.w, n.h);
        if w != xfeat::W || h != xfeat::H || n.color.len() != w * h * 3 || n.depth.len() != w * h {
            return None;
        }
        let rgb: &[u8] = if n.bgr {
            self.rgb.clear();
            self.rgb.extend(n.color.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]));
            &self.rgb
        } else {
            &n.color
        };
        place::prepare(rgb, w, h, &mut self.small, &mut self.patches);
        let tk = self.ml.submit(nnrun::f32_bytes(&self.patches), &mut self.out_ml);
        xfeat::normalize_rgb(rgb, &mut self.gray);
        let xin: &[u8] = if self.xf.wants_u8() {
            xfeat::quantize_input(&self.gray, &mut self.q8);
            &self.q8
        } else {
            nnrun::f32_bytes(&self.gray)
        };
        let tx = self.xf.submit(xin, &mut self.out_xf);
        self.ml.wait(tk);
        {
            let (f, s, t) = (self.ml.out(&self.out_ml, "f"), self.ml.out(&self.out_ml, "s"), self.ml.out(&self.out_ml, "t"));
            self.head.describe(f, s, t, &mut self.desc);
        }
        self.xf.wait(tx);
        let mut q = XFeatures::default();
        {
            let maps = xfeat::Maps { feats: self.xf.out(&self.out_xf, "feats"), kpts: self.xf.out(&self.out_xf, "kpts"), heat: self.xf.out(&self.out_xf, "heat") };
            xfeat::detect(&maps, self.cfg.kps, 0.05, &mut self.xsc, &mut q);
        }
        // 교실은 4 m 넘는 벽 · 문이 많아 8 m 까지(먼 점은 4 자유도 가설 · 재투영 오차로 걸러짐)
        let vp = lcverify::Params { min_inliers: MIN_INLIERS, zmax: 8.0, px: 4.0, px_hyp: 10.0, refine6: false, ..self.cfg.verify.clone() };
        let x3: Vec<Option<V3>> = q.kps.iter().map(|&p| lcverify::backproject(&n.depth, w, h, 1000.0, &n.k, p, vp.zmin, vp.zmax)).collect();
        // 후보: 지금 다른 지도에 든 예전 장면
        let my_map = n.maps.get(&n.kf).copied();
        let nodes = &self.nodes;
        let cands = match my_map {
            None => vec![],
            Some(m) => self.db.topk(&self.desc, TOPK, |id| nodes.get(id as usize).and_then(|s| n.maps.get(&s.kf)).is_some_and(|&cm| cm != m)),
        };
        if let Some(l) = self.log.as_mut() {
            use std::io::Write as _;
            let _ = writeln!(l, "{:.3} node map {:?} cands {:?}", n.t, my_map, cands.iter().map(|c| (c.0, (c.1 * 100.0).round() / 100.0)).collect::<Vec<_>>());
        }
        let qsmall = small_depth(&n.depth, w, h);
        let mut hit = None;
        let mut b = XFeatures::default();
        for &(cid, score) in &cands {
            if score < MIN_SCORE {
                break;
            }
            let c = &self.nodes[cid as usize];
            b.kps.clone_from(&c.kps);
            b.desc.clear();
            b.desc.extend(c.desc.iter().map(|&v| v as f32 / Q8));
            xfeat::match_mnn(&q, &b, -1.0, &mut self.sim, &mut self.pairs);
            let kq: Vec<[f32; 2]> = self.pairs.iter().map(|p| q.kps[p.0 as usize]).collect();
            let xq: Vec<Option<V3>> = self.pairs.iter().map(|p| x3[p.0 as usize]).collect();
            let xd: Vec<Option<V3>> = self.pairs.iter().map(|p| c.x3[p.1 as usize]).collect();
            let v = lcverify::verify(&kq, &xq, &xd, n.up, c.up, &n.k, &vp);
            if let Some(l) = self.log.as_mut() {
                use std::io::Write as _;
                let _ = writeln!(l, "{:.3} {:.3} map {:?} score {score:.3} pairs {} inl {:?} hyp {:?} ok {}", n.t, c.t, my_map, self.pairs.len(), v.as_ref().map(|v| v.inliers), v.as_ref().map(|v| v.hyp_inliers), v.as_ref().is_some_and(|v| v.accepted));
            }
            if let Some(v) = v
                && v.accepted
            {
                let mut t = [0f64; 12];
                for i in 0..3 {
                    for j in 0..3 {
                        t[i * 4 + j] = v.r[i][j];
                    }
                    t[i * 4 + 3] = v.t[i];
                }
                // 깊이 ICP 로 다듬고, 깊이 전체가 같은 변환으로 겹치는지(반복되는 책상 · 모니터 줄에서 한 줄 어긋난 짝은 벽 · 바닥 깊이가 안 맞음)
                let t0 = t;
                t = icp(&qsmall, &c.small_d, w, h, &n.k, &c.k, &t0);
                let (ov, ag) = depth_agree(&qsmall, &c.small_d, w, h, &n.k, &c.k, &t);
                let ag_tight = depth_agree_tol(&qsmall, &c.small_d, w, h, &n.k, &c.k, &t, 0.02, 0.01).1;
                if let Some(l) = self.log.as_mut() {
                    use std::io::Write as _;
                    let _ = writeln!(l, "{:.3} {:.3} depth overlap {ov:.2} agree {ag:.2} tight {ag_tight:.2}", n.t, c.t);
                }
                if ov < DEPTH_OVERLAP || ag < DEPTH_AGREE {
                    continue;
                }
                hit = Some(Hit { t: n.t, q_kf: n.kf, q_rel: n.rel, c_kf: c.kf, c_rel: c.rel, c_t: c.t, t_qc: t, inliers: v.inliers, score });
                break;
            }
        }
        self.db.add(self.nodes.len() as i64, &self.desc);
        let desc: Vec<i8> = q.desc.iter().map(|&v| (v * Q8).round().clamp(-127.0, 127.0) as i8).collect();
        self.nodes.push(Stored { t: n.t, kf: n.kf, rel: n.rel, up: n.up, kps: q.kps, desc, x3, small_d: qsmall, k: n.k });
        hit
    }
}
