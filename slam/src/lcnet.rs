//! 장소 인식 · 루프 클로저: 지도 노드마다 MegaLoc 기술자로 예전 노드 후보를 찾고, XFeat 짝 + 깊이 +
//! 중력 고정 4 자유도 검증(lcverify)으로 맞으면 그 상대 위치를 루프 링크로 지도에 넘긴다.
//!
//! 한 스레드("lc")가 노드를 차례로 받는다: 신경망 둘(보드 NPU `.dxnn`, PC ONNX) → 후보 상위 k → 검증 → 결과 줄.
//! 노드마다 남기는 것: MegaLoc 기술자(주성분 1,024 차원 f32) · XFeat 점 · 기술자(i8) · 점의 3D(카메라 좌표) · 위쪽 방향.
//! 링크 변환 = 질의 노드 → 후보 노드(몸체 좌표, from⁻¹·to). 최적화 · 모순 검사는 지도 쪽(app::Mapper)이 한다.
use crate::lcverify::{self, Intr, Params as VParams, V3};
use crate::nnrun::{self, Net};
use crate::place::{self, Head, PlaceDb};
use crate::geom::Tf;
use crate::xfeat::{self, XFeatures};
use crate::{Camera, DepthBuf, Pose};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// 비상 열 보호 1 단(물체 기억 `thermal`): 켜지면 앱이 노드 3 개 가운데 하나만 루프 스레드로 보낸다
pub static THROTTLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn throttle(on: bool) {
    THROTTLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// 모델 · 문턱
#[derive(Clone, Debug)]
pub struct LcConfig {
    pub megaloc: String,
    pub head: String,
    pub xfeat: String,
    /// ONNX Runtime 스레드(PC)
    pub threads: usize,
    /// 후보 수(MegaLoc 내적 상위)
    pub topk: usize,
    /// 후보로 볼 MegaLoc 내적 최소
    pub min_score: f32,
    /// 질의와 이 시간(초) 안의 노드는 후보에서 뺌(바로 앞 노드는 위치 추적이 이미 잇는다)
    pub skip_s: f64,
    /// XFeat 점 수
    pub kps: usize,
    pub verify: VParams,
}

impl LcConfig {
    /// 모델 폴더의 기본 파일: 보드 빌드(dxrt)는 `.dxnn`, PC 는 같은 계산의 ONNX
    pub fn from_dir(dir: &str) -> LcConfig {
        let ext = if cfg!(feature = "dxrt") { "dxnn" } else { "onnx" };
        LcConfig {
            megaloc: format!("{dir}/megaloc_168x224.{ext}"),
            head: format!("{dir}/megaloc_head_pca1024.bin"),
            xfeat: format!("{dir}/xfeat_480x640.{ext}"),
            threads: 4,
            topk: 3,
            min_score: 0.3,
            skip_s: 20.0,
            kps: 1000,
            // 맞은 점 20 – 30 의 검증은 흐린 장면에서 30 – 55 cm 틀린 것이 있어(fr1_room 흐림 0.02) 40 개부터 받음
            verify: VParams { min_inliers: 40, ..VParams::default() },
        }
    }
}

/// 지도 노드 하나(노드 id 가 정해진 뒤 넘김)
pub struct LcNode {
    pub id: i32,
    pub stamp: f64,
    /// 640×480 컬러(줄 순서), bgr = true 면 B·G·R 순서
    pub color: Vec<u8>,
    pub bgr: bool,
    /// 컬러 격자에 맞춘 깊이(mm)
    pub depth: DepthBuf,
    /// 위치 추적 자세(몸체, 위치 추적 좌표). 위쪽 = 이 좌표의 z
    pub pose: Pose,
    pub cam: Camera,
    /// 0 이 아니면 이 id 앞 노드는 시간 제한(skip_s) 없이 후보로: 위치 추적이 다시 시작한 직후 빈틈 앞 지도에서 제자리 찾기
    pub reloc_before: i32,
}

/// 맞은 루프: from(질의) → to(예전 노드) 몸체 변환
#[derive(Clone, Debug)]
pub struct LcHit {
    pub from: i32,
    pub to: i32,
    pub t: Tf,
    pub score: f32,
    pub inliers: usize,
}

/// 노드 하나 처리 결과(맞은 루프가 없어도 옴: 최대 속도 재생은 이걸 기다려 결정적으로)
#[derive(Clone, Debug, Default)]
pub struct LcOut {
    pub id: i32,
    pub hit: Option<LcHit>,
    /// 후보 수 · 검증해 본 수
    pub cands: usize,
    pub tried: usize,
    /// 검증은 맞았지만 위치 추적과 같은 말이라(보정 없음) 버린 수 · 받아들이는 간격 안이라 건너뛴 노드인지
    pub redundant: usize,
    pub gap: bool,
    /// ms: MegaLoc(전처리 + 신경망 + 집계) · XFeat(전처리 + 신경망 + 점) · 짝 + 검증
    pub ms: [f64; 3],
}

struct Stored {
    id: i32,
    stamp: f64,
    /// 위치 추적 자세(몸체): 맞은 루프가 위치 추적과 이미 같은 말을 하는지(보탤 것 없음) 보는 데
    pose: Pose,
    up: V3,
    kps: Vec<[f32; 2]>,
    desc: Vec<i8>,
    x3: Vec<Option<V3>>,
}

enum Msg {
    Node(Box<LcNode>),
}

/// 루프 스레드 손잡이
pub struct LcWorker {
    tx: Option<Sender<Msg>>,
    rx: Receiver<LcOut>,
    th: Option<std::thread::JoinHandle<()>>,
    /// 보냈지만 결과를 아직 받지 않은 노드 수
    pub pending: usize,
}

impl LcWorker {
    /// 모델을 읽고 스레드를 띄움(읽기 실패는 여기서 오류)
    pub fn start(cfg: LcConfig) -> Result<LcWorker, String> {
        let (tx, rx) = channel::<Msg>();
        let (otx, orx) = channel::<LcOut>();
        let (rtx, rrx) = channel::<Result<(), String>>();
        let th = std::thread::Builder::new()
            .name("lc".into())
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
                while let Ok(Msg::Node(n)) = rx.recv() {
                    let o = st.step(&n);
                    if otx.send(o).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        rrx.recv().map_err(|e| e.to_string())??;
        Ok(LcWorker { tx: Some(tx), rx: orx, th: Some(th), pending: 0 })
    }

    pub fn send(&mut self, n: LcNode) {
        if let Some(tx) = &self.tx {
            if tx.send(Msg::Node(Box::new(n))).is_ok() {
                self.pending += 1;
            }
        }
    }

    /// 끝난 결과 하나(block = 끝날 때까지 기다림)
    pub fn recv(&mut self, block: bool) -> Option<LcOut> {
        if self.pending == 0 {
            return None;
        }
        let r = if block { self.rx.recv().ok() } else { self.rx.try_recv().ok() };
        if r.is_some() {
            self.pending -= 1;
        }
        r
    }
}

impl Drop for LcWorker {
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
    // 작업 버퍼
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
    /// 마지막으로 받아들인 루프의 노드 시각 · 최근에 이은 예전 노드(노드 시각, id)
    last_hit: f64,
    recent_to: std::collections::VecDeque<(f64, i32)>,
    /// 시험용 기록(환경 변수 LC_LOG = 파일): 맞은 루프마다 시각 · id · 점수 · 맞은 점 · 광학 좌표 r · t
    log: Option<std::fs::File>,
}

const Q8: f32 = 127.0;
/// 맞은 루프가 위치 추적 상대 자세와 이만큼 안으로 같으면 보탤 보정이 없어 버림(가만히 있거나 표류 없이 같은 자리를 다시 볼 때):
/// 낱말 루프가 움직임이 작은 노드에서 루프를 찾지 않는 것과 같은 몫. 받아들이면 그때마다 지도 최적화 · 다시 쌓기가 돈다
const SAME_M: f32 = 0.05;
const SAME_DEG: f32 = 2.0;
/// 받아들인 루프 사이 최소 간격(초, 노드 시각): 같은 자리를 연달아 볼 때 노드마다 최적화하지 않게
const MIN_GAP_S: f64 = 2.0;
/// 이 시간(초) 안에 이미 루프로 이은 예전 노드는 다시 잇지 않음
const SAME_TO_S: f64 = 30.0;

impl State {
    fn open(cfg: &LcConfig) -> Result<State, String> {
        let ml = Net::open(&cfg.megaloc, cfg.threads, 0)?;
        let xf = Net::open(&cfg.xfeat, cfg.threads, 0)?;
        let head = Head::load(&cfg.head)?;
        for n in ["f", "s", "t"] {
            if !ml.has(n) {
                return Err(format!("{}: 출력 {n} 없음", cfg.megaloc));
            }
        }
        for n in ["feats", "kpts", "heat"] {
            if !xf.has(n) {
                return Err(format!("{}: 출력 {n} 없음", cfg.xfeat));
            }
        }
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
            last_hit: f64::NEG_INFINITY,
            recent_to: Default::default(),
            log: std::env::var("LC_LOG").ok().and_then(|p| std::fs::File::create(p).ok()),
        })
    }

    fn step(&mut self, n: &LcNode) -> LcOut {
        let mut o = LcOut { id: n.id, ..Default::default() };
        let (w, h) = (n.cam.width as usize, n.cam.height as usize);
        if w != xfeat::W || h != xfeat::H || n.color.len() != w * h * 3 || n.depth.len() != w * h {
            return o;
        }
        let rgb: &[u8] = if n.bgr {
            self.rgb.clear();
            self.rgb.extend(n.color.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]));
            &self.rgb
        } else {
            &n.color
        };
        // MegaLoc: 신경망을 올려 두고 그동안 XFeat 전처리
        let t0 = Instant::now();
        place::prepare(rgb, w, h, &mut self.small, &mut self.patches);
        let tk = self.ml.submit(nnrun::f32_bytes(&self.patches), &mut self.out_ml);
        let t1 = Instant::now();
        xfeat::normalize_rgb(rgb, &mut self.gray);
        let xin: &[u8] = if self.xf.wants_u8() {
            xfeat::quantize_input(&self.gray, &mut self.q8);
            &self.q8
        } else {
            nnrun::f32_bytes(&self.gray)
        };
        let tx = self.xf.submit(xin, &mut self.out_xf);
        let t2 = Instant::now();
        self.ml.wait(tk);
        {
            let (f, s, t) = (self.ml.out(&self.out_ml, "f"), self.ml.out(&self.out_ml, "s"), self.ml.out(&self.out_ml, "t"));
            self.head.describe(f, s, t, &mut self.desc);
        }
        let t3 = Instant::now();
        self.xf.wait(tx);
        let mut q = XFeatures::default();
        {
            let maps = xfeat::Maps { feats: self.xf.out(&self.out_xf, "feats"), kpts: self.xf.out(&self.out_xf, "kpts"), heat: self.xf.out(&self.out_xf, "heat") };
            xfeat::detect(&maps, self.cfg.kps, 0.05, &mut self.xsc, &mut q);
        }
        let k = Intr { fx: n.cam.fx, fy: n.cam.fy, cx: n.cam.cx, cy: n.cam.cy };
        let vp = &self.cfg.verify;
        let x3: Vec<Option<V3>> = q.kps.iter().map(|&p| lcverify::backproject(&n.depth, w, h, 1000.0, &k, p, vp.zmin, vp.zmax)).collect();
        let t4 = Instant::now();
        o.ms[0] = (t1 - t0).as_secs_f64() * 1e3 + (t3 - t2).as_secs_f64() * 1e3;
        o.ms[1] = (t2 - t1).as_secs_f64() * 1e3 + (t4 - t3).as_secs_f64() * 1e3;
        // 위쪽 방향(카메라 광학 좌표): (R_몸체 · R_광학)ᵀ · z
        let r_opt = (n.pose * n.cam.local).rotation.to_rotation_matrix();
        let m = r_opt.matrix();
        let up: V3 = [m[(2, 0)], m[(2, 1)], m[(2, 2)]];
        // 후보 → 검증
        let me = n.stamp;
        let skip = self.cfg.skip_s;
        let stamps: std::collections::HashMap<i64, f64> = self.nodes.iter().map(|s| (s.id as i64, s.stamp)).collect();
        while self.recent_to.front().is_some_and(|r| me - r.0 > SAME_TO_S) {
            self.recent_to.pop_front();
        }
        let recent: std::collections::HashSet<i64> = self.recent_to.iter().map(|r| r.1 as i64).collect();
        // 받아들인 루프 바로 뒤(MIN_GAP_S 안)는 후보를 찾지 않음
        o.gap = (me - self.last_hit).abs() < MIN_GAP_S;
        let cands = if o.gap {
            vec![]
        } else {
            self.db.topk(&self.desc, self.cfg.topk, |id| !recent.contains(&id) && (id < n.reloc_before as i64 || stamps.get(&id).is_some_and(|&t| (me - t).abs() > skip)))
        };
        o.cands = cands.iter().filter(|c| c.1 >= self.cfg.min_score).count();
        let mut b = XFeatures::default();
        for &(cid, score) in &cands {
            if score < self.cfg.min_score {
                break;
            }
            let Some(c) = self.nodes.iter().find(|s| s.id as i64 == cid) else { continue };
            o.tried += 1;
            b.kps.clone_from(&c.kps);
            b.desc.clear();
            b.desc.extend(c.desc.iter().map(|&v| v as f32 / Q8));
            xfeat::match_mnn(&q, &b, -1.0, &mut self.sim, &mut self.pairs);
            let kq: Vec<[f32; 2]> = self.pairs.iter().map(|p| q.kps[p.0 as usize]).collect();
            let xq: Vec<Option<V3>> = self.pairs.iter().map(|p| x3[p.0 as usize]).collect();
            let xd: Vec<Option<V3>> = self.pairs.iter().map(|p| c.x3[p.1 as usize]).collect();
            if let Some(v) = lcverify::verify(&kq, &xq, &xd, up, c.up, &k, vp)
                && v.accepted
            {
                // 광학 좌표 X_q = r X_d + t → 몸체 링크 = L · T · L⁻¹
                let mut a = [0f32; 12];
                for i in 0..3 {
                    for j in 0..3 {
                        a[i * 4 + j] = v.r[i][j] as f32;
                    }
                    a[i * 4 + 3] = v.t[i] as f32;
                }
                if let Some(f) = self.log.as_mut() {
                    use std::io::Write as _;
                    let _ = writeln!(f, "{:.6} {:.6} {} {} {:.4} {} {:?} {:?}", n.stamp, c.stamp, n.id, c.id, score, v.inliers, v.r, v.t);
                }
                let l = Tf::from_pose(&n.cam.local);
                let t = l.mul(&Tf(a)).mul(&l.inverse());
                // 위치 추적의 상대 자세(from⁻¹ · to)와 같은 말이면 지도에 보탤 보정이 없음 → 버리고 이 노드는 끝
                let odo = Tf::from_pose(&(n.pose.inverse() * c.pose));
                let err = odo.inverse().mul(&t);
                if err.norm_sq().sqrt() < SAME_M && err.angle_to(&Tf::IDENTITY).to_degrees() < SAME_DEG {
                    o.redundant += 1;
                    break;
                }
                self.last_hit = me;
                self.recent_to.push_back((me, c.id));
                o.hit = Some(LcHit { from: n.id, to: c.id, t, score, inliers: v.inliers });
                break;
            }
        }
        o.ms[2] = t4.elapsed().as_secs_f64() * 1e3;
        // 표에 넣기
        self.db.add(n.id as i64, &self.desc);
        let desc: Vec<i8> = q.desc.iter().map(|&v| (v * Q8).round().clamp(-127.0, 127.0) as i8).collect();
        self.nodes.push(Stored { id: n.id, stamp: n.stamp, pose: n.pose, up, kps: q.kps, desc, x3 });
        o
    }
}
