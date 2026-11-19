//! 위치 프로그램: RGB-D(+ IMU) 입력 → 위치 추적(장면마다) + 지도 노드(1 초에 1 번) → 장면(같은 프로세스의 메모리 큐 또는 장면 스트림).
//!
//!   ondevice --slam "--tum <폴더> --rate 0" ...   (한 프로세스: 처리 루프가 장면마다 [`Runner::step`] 을 부르고 장면을 메모리 큐로 넘김)
//!   ondevice slam --tum <폴더> [--rate 1] --out <스트림 파일>   (위치만: 장면 스트림을 파일 · 표준출력으로)
//!
//! 구조(docs/slam.md):
//!  - 앞단(장면마다, 이 스레드): 광류 + PnP + IMU 회전 예측(klt · imu). 놓쳐도 지도는 하나 — 마지막 위치(IMU 자세)에서 다시 시작하고,
//!    그 사이 연속 간선을 약하게 둬 뒷단이 예전 자리를 찾으면 그 구간이 통째로 옮겨진다.
//!  - 뒷단(따로 도는 스레드 둘, 처리 루프는 기다리지 않음): 지도 노드마다 MegaLoc 후보 → XFeat · 깊이 · 중력 4 자유도 검증(lcnet) →
//!    루프 간선 → 자세 그래프 가우스-뉴턴(graph) → 노드 자세 갱신(그래프 블록). 받는 쪽은 노드에 묶인 장면 · 청크를 그 자세로 옮긴다.
use crate::geom::rtf::{self, Tf};
use crate::graph::{self, link_type, Link, OptimizerParams};
use crate::io::{FrameSource, PrefetchSource, TumSource};
use crate::klt::OdometryKlt;
use crate::lcnet::{LcConfig, LcNode, LcOut, LcWorker};
use crate::stream::{self, GraphNode, OutFrame, Writer};
use crate::Frame;
use std::collections::{BTreeMap, VecDeque};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// 지도 노드: 이 시간(초)마다, 또는 NODE_MIN_S 가 지났고 앞 노드에서 NODE_MOVE_M · NODE_TURN_DEG 넘게 움직였으면
const NODE_EVERY_S: f64 = 1.0;
const NODE_MIN_S: f64 = 0.3;
const NODE_MOVE_M: f32 = 0.25;
const NODE_TURN_DEG: f32 = 15.0;
/// IMU: 이어 붙인 장면 · 영상이 IMU 와 어긋난 장면 뒤 이만큼 장면까지는 위치만(지도에 안 쌓음)
const IMU_SETTLE: u32 = 5;
/// 연속 간선 표준편차: 움직임에 비례(TUM 정답 대조 1 초 상대 오차: 이동의 7 – 16 %, 회전의 8 – 9 %, fr1 은 3 – 5 cm · 2 – 3°).
/// 고정 1 cm · 0.3° 는 실제보다 3 – 10 배 자신해 맞는 루프도 그래프와 "모순"으로 버려졌다
const SD_ODOM_M: [f64; 2] = [0.01, 0.08];
const SD_ODOM_RAD: [f64; 2] = [0.005, 0.08];
/// 위치 추적이 다시 시작한 구간(IMU 로 이어 감)
const SD_GAP: [f64; 2] = [0.5, 0.1];
/// 다시 시작한 뒤 이 시간(초) 안의 노드는 빈틈 앞 노드를 시간 제한 없이 후보로(제자리 찾기)
const RELOC_WINDOW_S: f64 = 15.0;
/// 루프 받기 전 확인: 지금 지도가 말하는 상대 자세와 루프의 차가 두 노드 사이 연속 간선 불확실성의 이 배수 + 여유를 넘으면 버림
const GATE_SIGMA: f64 = 3.0;
const GATE_M: f64 = 0.05;
const GATE_RAD: f64 = 0.035;
/// 그래프 풀이: 루프 간선을 Cauchy 가중(바꿀 수 있는 제약과 같은 몫)으로 다시 풀고, 가중이 이보다 작아진 루프는 버림
const ROBUST_C2: f64 = 9.0;
const ROBUST_DROP_W: f64 = 0.1;
const ROBUST_ITERS: usize = 4;

/// 위치를 TUM 한 줄로: 시각 x y z qx qy qz qw
fn write_tum(o: &mut impl std::io::Write, t: f64, m: &Tf) {
    let a = &m.0;
    let q = rtf::quat_normalized_f(rtf::quat_from_mat3f(&[a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10]]));
    let _ = writeln!(o, "{t:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}", a[3], a[7], a[11], q[0], q[1], q[2], q[3]);
}

#[derive(Default)]
struct Summary(Vec<f64>);

impl Summary {
    fn add(&mut self, x: f64) {
        self.0.push(x);
    }
    fn str(&self) -> String {
        if self.0.is_empty() {
            return "-".into();
        }
        let mut s = self.0.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        format!("평균 {mean:.2} 중앙 {:.2} 95% {:.2} 최대 {:.2} ({})", s[s.len() / 2], s[((s.len() - 1) as f64 * 0.95) as usize], s[s.len() - 1], s.len())
    }
    fn max(&self) -> f64 {
        self.0.iter().cloned().fold(0.0, f64::max)
    }
}

fn ms_since(a: Instant) -> f64 {
    a.elapsed().as_secs_f64() * 1e3
}

/// 표준편차(이동 m, 회전 rad) → 정보 행렬(x y z roll pitch yaw)
fn info_of(sd: [f64; 2]) -> [f64; 36] {
    let mut m = [0.0; 36];
    for i in 0..6 {
        let s = if i < 3 { sd[0] } else { sd[1] };
        m[i * 7] = 1.0 / (s * s);
    }
    m
}

/// 두 변환 차이의 (이동 m, 회전 °)
fn tf_diff(a: &Tf, b: &Tf) -> (f32, f32) {
    let e = a.inverse().mul(b);
    (e.norm_sq().sqrt(), e.angle_to(&Tf::IDENTITY).to_degrees())
}

// ---------------- 뒷단: 지도 노드 · 루프 · 자세 그래프 ----------------

/// 지도 노드
#[derive(Clone)]
struct Node {
    t: f64,
    /// 위치 추적 자세(몸체)
    odom: Tf,
    /// 지도 자세(최적화 뒤, 몸체)
    opt: Tf,
    /// 앞 노드에서 이 노드까지 연속 간선의 분산(이동 m², 회전 rad²)
    var: [f64; 2],
}

/// 최적화 스레드에 넘기는 일
struct OptJob {
    era: u64,
    root: i32,
    poses: BTreeMap<i32, [f32; 12]>,
    links: Vec<Link>,
}

/// 최적화 결과: 노드 자세 · 어긋나 버린 루프(from, to)
struct OptDone {
    era: u64,
    poses: BTreeMap<i32, Tf>,
    dropped: Vec<(i32, i32)>,
    ms: f64,
}

/// 간선이 최적화 결과와 어긋난 정도를 그 간선의 표준편차로 나눈 제곱 합(이동 + 회전)
fn resid_chi(l: &Link, p: &BTreeMap<i32, [f32; 12]>) -> f64 {
    match (p.get(&l.from), p.get(&l.to)) {
        (Some(a), Some(b)) => {
            let (m, d) = tf_diff(&Tf(*a).inverse().mul(&Tf(*b)), &Tf(l.transform));
            let r = (d as f64).to_radians();
            (m as f64).powi(2) * l.inf[0] + r * r * l.inf[21]
        }
        _ => 0.0,
    }
}

/// 움직임(이동 m, 회전 rad)에 비례한 연속 간선 표준편차
fn sd_odom(m: f32, rad: f32) -> [f64; 2] {
    [SD_ODOM_M[0] + SD_ODOM_M[1] * m as f64, SD_ODOM_RAD[0] + SD_ODOM_RAD[1] * rad as f64]
}

/// 자세 그래프 풀이 스레드: 일이 쌓이면 가장 새 것만. 모순되는 루프는 빼고 한 번 더
fn spawn_optimizer() -> (Sender<OptJob>, Receiver<OptDone>) {
    let (tx, rx) = channel::<OptJob>();
    let (dtx, drx) = channel::<OptDone>();
    std::thread::Builder::new()
        .name("graph".into())
        .spawn(move || {
            let p = OptimizerParams { fast_solver: true, ..OptimizerParams::default() };
            while let Ok(mut job) = rx.recv() {
                while let Ok(j) = rx.try_recv() {
                    job = j;
                }
                let c = Instant::now();
                // 루프 간선은 Cauchy 가중으로 몇 번 다시 풀어(IRLS) 다른 간선과 모순되는 루프의 힘을 빼고, 가중이 작아진 루프는 버림
                let base: Vec<Link> = job.links.clone();
                let mut w: Vec<f64> = vec![1.0; base.len()];
                let mut out = BTreeMap::new();
                for _ in 0..ROBUST_ITERS {
                    let links: Vec<Link> = base
                        .iter()
                        .zip(&w)
                        .map(|(l, &wi)| {
                            let mut l = l.clone();
                            if wi < 1.0 {
                                for v in l.inf.iter_mut() {
                                    *v *= wi;
                                }
                            }
                            l
                        })
                        .collect();
                    out = graph::optimize(job.root, &job.poses, &links, &p).map(|o| o.poses).unwrap_or_default();
                    if out.is_empty() {
                        break;
                    }
                    let mut changed = false;
                    for (i, l) in base.iter().enumerate() {
                        if l.kind != link_type::GLOBAL_CLOSURE {
                            continue;
                        }
                        let nw = 1.0 / (1.0 + resid_chi(l, &out) / ROBUST_C2);
                        if (nw - w[i]).abs() > 0.05 {
                            changed = true;
                        }
                        w[i] = nw;
                    }
                    if !changed {
                        break;
                    }
                }
                let dropped: Vec<(i32, i32)> = base.iter().zip(&w).filter(|(l, wi)| l.kind == link_type::GLOBAL_CLOSURE && **wi < ROBUST_DROP_W).map(|(l, _)| (l.from, l.to)).collect();
                if !dropped.is_empty() {
                    let links: Vec<Link> = base.iter().filter(|l| !dropped.contains(&(l.from, l.to))).cloned().collect();
                    out = graph::optimize(job.root, &job.poses, &links, &p).map(|o| o.poses).unwrap_or_default();
                }
                let poses = out.into_iter().map(|(k, v)| (k, Tf(v))).collect();
                if dtx.send(OptDone { era: job.era, poses, dropped, ms: ms_since(c) }).is_err() {
                    break;
                }
            }
        })
        .expect("graph 스레드");
    (tx, drx)
}

/// 루프 스레드 통계
#[derive(Default)]
struct LcStats {
    sent: u64,
    done: u64,
    tried: u64,
    hits: u64,
    redundant: u64,
    gap: u64,
    accepted: u64,
    dropped: u64,
    gated: u64,
    ms: [f64; 3],
    ms_max: [f64; 3],
}

/// 지도: 노드 · 간선 · 지도 보정(지도 = 보정 × 위치 추적). 처리 루프는 노드를 넣고 결과를 거둬 갈 뿐 기다리지 않는다(최대 속도 재생만 결정적이게 기다림)
struct Mapper {
    nodes: BTreeMap<i32, Node>,
    links: Vec<Link>,
    next_id: i32,
    /// 리셋 뒤 첫 노드 id(그 앞 id 로 오는 루프는 버림)
    first_id: i32,
    corr: Tf,
    lc: LcWorker,
    lc_q: VecDeque<LcOut>,
    lc_block: bool,
    stats: LcStats,
    opt_tx: Sender<OptJob>,
    opt_rx: Receiver<OptDone>,
    era: u64,
    /// 보낸 최적화 중 아직 안 온 것 · 결과를 받은 뒤 다시 풀 일이 생겼는지
    opt_pending: bool,
    opt_dirty: bool,
    s_opt: Summary,
    /// 다음 연속 간선을 약하게(위치 추적이 다시 시작함)
    weak_next: bool,
    /// 위치 추적이 다시 시작한 뒤 첫 노드 id 와 그 시각: 그 뒤 RELOC_WINDOW_S 동안은 그 앞 노드를 시간 제한 없이 후보로
    gap: Option<(i32, f64)>,
    n_gap_edges: u64,
}

impl Mapper {
    fn new(cfg: LcConfig, lc_block: bool) -> Result<Mapper, String> {
        let lc = LcWorker::start(cfg)?;
        let (opt_tx, opt_rx) = spawn_optimizer();
        Ok(Mapper {
            nodes: BTreeMap::new(),
            links: vec![],
            next_id: 1,
            first_id: 1,
            corr: Tf::IDENTITY,
            lc,
            lc_q: VecDeque::new(),
            lc_block,
            stats: LcStats::default(),
            opt_tx,
            opt_rx,
            era: 0,
            opt_pending: false,
            opt_dirty: false,
            s_opt: Summary::default(),
            weak_next: false,
            gap: None,
            n_gap_edges: 0,
        })
    }

    /// 지도를 비움(리셋). 루프 스레드의 예전 노드는 first_id 앞이라 무시된다
    fn reset(&mut self) {
        self.nodes.clear();
        self.links.clear();
        self.first_id = self.next_id;
        self.corr = Tf::IDENTITY;
        self.era += 1; // 날아오는 최적화 결과는 버림
        self.opt_dirty = false;
        self.weak_next = false;
        self.gap = None;
    }

    fn last_id(&self) -> i32 {
        self.nodes.keys().next_back().copied().unwrap_or(0)
    }

    /// 노드 하나: 연속 간선 · 루프 스레드로. 돌려줌 = 새 노드 id
    fn add_node(&mut self, fr: &Frame, odom: Tf) -> i32 {
        let id = self.next_id;
        self.next_id += 1;
        let mut var = [0.0; 2];
        if let Some((&pid, prev)) = self.nodes.iter().next_back() {
            let rel = prev.odom.inverse().mul(&odom);
            let sd = if self.weak_next {
                self.n_gap_edges += 1;
                self.gap = Some((id, fr.t));
                SD_GAP
            } else {
                sd_odom(rel.norm_sq().sqrt(), rel.angle_to(&Tf::IDENTITY))
            };
            var = [sd[0] * sd[0], sd[1] * sd[1]];
            self.links.push(Link { from: pid, to: id, kind: link_type::NEIGHBOR, transform: rel.0, inf: info_of(sd) });
        }
        self.weak_next = false;
        let opt = self.corr.mul(&odom);
        self.nodes.insert(id, Node { t: fr.t, odom, opt, var });
        let reloc_before = match self.gap {
            Some((g, t0)) if fr.t - t0 < RELOC_WINDOW_S => g,
            _ => 0,
        };
        self.lc.send(LcNode { id, stamp: fr.t, color: fr.color.clone(), bgr: !fr.color_is_rgb, depth: fr.depth_mm.clone(), pose: odom.to_pose(), map_pose: opt.to_pose(), cam: fr.cam.clone(), reloc_before });
        self.stats.sent += 1;
        id
    }

    /// 루프 받기 전 확인: 지금 지도의 상대 자세와 루프의 차가 두 노드 사이 연속 간선 불확실성(분산 합) 안인지
    fn gate(&self, from: i32, to: i32, t: &Tf) -> bool {
        let (Some(a), Some(b)) = (self.nodes.get(&from), self.nodes.get(&to)) else { return false };
        let (lo, hi) = if to < from { (to, from) } else { (from, to) };
        let mut var = [0.0f64; 2];
        for (_, n) in self.nodes.range(lo + 1..=hi) {
            var[0] += n.var[0];
            var[1] += n.var[1];
        }
        let (m, d) = tf_diff(&a.opt.inverse().mul(&b.opt), t);
        let ok = (m as f64) <= GATE_SIGMA * var[0].sqrt() + GATE_M && (d as f64).to_radians() <= GATE_SIGMA * var[1].sqrt() + GATE_RAD;
        if !ok {
            common::info!("루프 {from} → {to} 버림: 지도와 {m:.2} m · {d:.1}° 차(허용 {:.2} m · {:.1}°)", GATE_SIGMA * var[0].sqrt() + GATE_M, (GATE_SIGMA * var[1].sqrt() + GATE_RAD).to_degrees());
        }
        ok
    }

    /// 끝난 루프 결과를 간선으로. 최대 속도 재생(lc_block)은 한 노드 늦게 고정: 바로 앞에 보낸 노드 것만 빼고 모두 기다림
    fn poll_lc(&mut self) -> usize {
        while let Some(o) = self.lc.recv(false) {
            self.lc_q.push_back(o);
        }
        let n = if self.lc_block {
            let need = self.stats.sent.saturating_sub(1);
            while self.stats.done + (self.lc_q.len() as u64) < need {
                match self.lc.recv(true) {
                    Some(o) => self.lc_q.push_back(o),
                    None => break,
                }
            }
            need.saturating_sub(self.stats.done) as usize
        } else {
            self.lc_q.len()
        };
        let mut added = 0;
        for o in self.lc_q.drain(..n.min(self.lc_q.len())).collect::<Vec<_>>() {
            let s = &mut self.stats;
            s.done += 1;
            s.tried += o.tried as u64;
            s.redundant += o.redundant as u64;
            s.gap += o.gap as u64;
            for i in 0..3 {
                s.ms[i] += o.ms[i];
                s.ms_max[i] = s.ms_max[i].max(o.ms[i]);
            }
            let Some(h) = o.hit else { continue };
            s.hits += 1;
            if h.to < self.first_id || h.from < self.first_id || !self.nodes.contains_key(&h.from) || !self.nodes.contains_key(&h.to) {
                continue;
            }
            if !self.gate(h.from, h.to, &h.t) {
                self.stats.gated += 1;
                continue;
            }
            common::info!("루프: 노드 {} → {} ({}, 점수 {:.3}, 맞은 점 {}, 이동 {:.2} m)", h.from, h.to, if h.near { "근처" } else { "MegaLoc" }, h.score, h.inliers, h.t.norm_sq().sqrt());
            // 불확실성: 맞은 점이 많을수록 작게(긴 사무실 정답 대조: 맞은 점 20 – 50 은 5 – 11 cm, 100 개 넘으면 1 – 3 cm)
            let k = 1.0 / (h.inliers.max(1) as f64).sqrt();
            let sd = [(0.3 * k).clamp(0.01, 0.1), (10f64 * k).clamp(0.3, 3.0).to_radians()];
            self.links.push(Link { from: h.from, to: h.to, kind: link_type::GLOBAL_CLOSURE, transform: h.t.0, inf: info_of(sd) });
            self.stats.accepted += 1;
            added += 1;
        }
        if added > 0 {
            self.opt_dirty = true;
        }
        added
    }

    /// 풀 일이 있으면 최적화 스레드로(이미 하나 돌고 있으면 끝난 뒤)
    fn kick_opt(&mut self) {
        if !self.opt_dirty || self.opt_pending || self.nodes.is_empty() {
            return;
        }
        self.era += 1;
        let poses = self.nodes.iter().map(|(&k, n)| (k, n.opt.0)).collect();
        let root = *self.nodes.keys().next().unwrap();
        if self.opt_tx.send(OptJob { era: self.era, root, poses, links: self.links.clone() }).is_ok() {
            self.opt_pending = true;
            self.opt_dirty = false;
        }
    }

    /// 최적화 결과를 받아 노드 자세 · 지도 보정을 고침. 돌려줌 = 자세가 바뀐 노드
    fn poll_opt(&mut self, block: bool) -> Vec<i32> {
        if !self.opt_pending {
            return vec![];
        }
        let r = if block { self.opt_rx.recv().ok() } else { self.opt_rx.try_recv().ok() };
        let Some(d) = r else { return vec![] };
        self.opt_pending = false;
        if d.era != self.era || d.poses.is_empty() {
            return vec![];
        }
        self.s_opt.add(d.ms);
        if !d.dropped.is_empty() {
            self.links.retain(|l| !d.dropped.contains(&(l.from, l.to)));
            self.stats.dropped += d.dropped.len() as u64;
            common::info!("루프 {} 개가 다른 간선과 맞지 않아 버림: {:?}", d.dropped.len(), d.dropped);
        }
        // 풀 때 있던 마지막 노드로 새 보정을 정하고, 그 뒤에 생긴 노드는 새 보정으로 옮김
        let last_solved = *d.poses.keys().next_back().unwrap();
        let mut moved = vec![];
        for (id, n) in self.nodes.iter_mut() {
            let p = match d.poses.get(id) {
                Some(p) => *p,
                None => continue,
            };
            let (m, a) = tf_diff(&n.opt, &p);
            n.opt = p;
            if m > 1e-3 || a > 0.05 {
                moved.push(*id);
            }
        }
        if let Some(n) = self.nodes.get(&last_solved) {
            self.corr = n.opt.mul(&n.odom.inverse());
        }
        for (id, n) in self.nodes.range_mut(last_solved + 1..) {
            n.opt = self.corr.mul(&n.odom);
            moved.push(*id);
        }
        moved
    }
}

// ---------------- 명령줄 ----------------

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
mod sys {
    unsafe extern "C" {
        fn dup(fd: i32) -> i32;
        fn dup2(a: i32, b: i32) -> i32;
        fn signal(sig: i32, h: usize) -> usize;
    }
    extern "C" fn on_signal(_: i32) {
        super::STOP.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// 표준출력을 떼어 내 스트림용 파일로 돌려주고, 표준출력은 표준오류로 돌린다(라이브러리 출력이 장면 바이트에 섞이지 않게)
    pub fn take_stdout() -> std::fs::File {
        use std::os::fd::FromRawFd;
        let _ = std::io::Write::flush(&mut std::io::stdout());
        unsafe {
            let fd = dup(1);
            dup2(2, 1);
            std::fs::File::from_raw_fd(fd)
        }
    }
    pub fn on_stop_signals() {
        unsafe {
            signal(2, on_signal as extern "C" fn(i32) as usize); // SIGINT
            signal(15, on_signal as extern "C" fn(i32) as usize); // SIGTERM
        }
    }
}

fn usage() {
    common::info!(
        "ondevice slam (--tum|--rsdrec <폴더> [--rate R] [--stride N] | --realsense [--serial S] [--size 640x480] [--color-size 1280x720] [--fps 30] [--camera-profile librealsense|tracking] [--exposure ms | --ae-cap ms] [--gain 0..128])\n            [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop]\n            [--poses <파일>] [--final-poses <파일>] [--stats <파일>] [--max-frames N] [--reset-file <파일>] [--reset-at N] [--imu /dev/i2c-1 [--imu-calib <yaml>] [--imu-bridge 초] [--imu-max-blur 화소] [--imu-exposure 초] [--imu-min-inliers N] [--imu-min-ratio R]] [--imu-csv <imu0.csv> (재생)] [--stress-gap 주기:길이 [--stress-imu-gap]] [--stress-blur 노출초 [--stress-gyro <imu0.csv>]]"
    );
}

/// 위치 프로그램 설정(명령줄을 `parse` 로 한 번 읽은 것)
#[derive(Clone)]
pub struct Args {
    tum: String,
    out: String,
    poses: String,
    final_poses: String,
    stats: String,
    serial: String,
    reset_file: String,
    realsense: bool,
    reset_at: i64,
    stride: usize,
    block: i32,
    rate: f64,
    exposure_ms: f64,
    ae_cap_ms: f64,
    camera_profile: String,
    gain: i32,
    send_every: i64,
    queue: usize,
    w: u32,
    h: u32,
    /// --color-size: 컬러 크기(0 = --size 와 같음)
    color: (u32, u32),
    fps: u32,
    pub max_frames: i64,
    /// IMU 융합 모드: I2C 장치(빈 값 = 카메라 전용 모드)
    imu: String,
    /// IMU-카메라 보정 yaml(빈 값 = 저장소 docs/imu_cam_calib.yaml 을 빌드에 넣은 것)
    imu_calib: String,
    /// 영상이 다 실패한 구간을 IMU 회전으로 이어 붙이는 최대 시간(초)
    imu_bridge: f64,
    /// IMU 모드: 번짐(자이로 각속도 × 노출 × 초점 거리, 화소)이 이보다 큰 장면은 위치만(지도에 안 넣음)
    imu_max_blur: f64,
    /// 번짐 계산에 쓰는 노출(초)
    imu_exposure: f64,
    /// IMU 모드: 광류 맞은 점이 이보다 적거나 맞은 비율이 이보다 낮은 장면은 위치만
    imu_min_inliers: f64,
    imu_min_ratio: f64,
    /// 녹화 IMU 재생(Kalibr imu0.csv): IMU 융합 모드를 --imu 대신 이 파일로(--tum · --rsdrec 재생과 함께)
    imu_csv: String,
    /// 끊김 시험: 주기:길이(초). 첫 장면부터 주기마다 길이만큼 장면을 빼고, --stress-imu-gap 이면 IMU 표본도 뺌
    stress_gap: (f64, f64),
    stress_imu_gap: bool,
    /// 흐림 시험: 노출(초) 동안 각속도만큼 번진 영상(각속도는 --stress-gyro 또는 --imu-csv 파일)
    stress_blur: f64,
    stress_gyro: String,
}

impl Args {
    /// 받는 쪽을 기다리는지(--block | --drop, 기본: 최대 속도 재생이면 기다림)
    pub fn block_mode(&self) -> bool {
        let replay_max = !self.tum.is_empty() && self.rate == 0.0;
        if self.block < 0 { replay_max } else { self.block == 1 }
    }
}

/// 명령줄(0 번은 프로그램 이름) → 설정. 도움말 · 잘못된 인자면 끝 코드
pub fn parse(argv: &[String]) -> Result<Args, i32> {
    let mut a = Args {
        tum: String::new(),
        out: "-".into(),
        poses: String::new(),
        final_poses: String::new(),
        stats: String::new(),
        serial: String::new(),
        reset_file: String::new(),
        realsense: false,
        reset_at: -1,
        stride: 1,
        block: -1, // -1: 입력에 따라(재생 = 기다림, 카메라 = 버림)
        rate: 0.0,
        exposure_ms: 0.0,
        camera_profile: "librealsense".into(), // 컬러 제어: librealsense 와 같음(기본) | tracking(60 Hz · 노출 상한 16.6 ms)
        ae_cap_ms: 0.0, // 노출 상한(ms). 0 = 카메라 자동 노출
        gain: -1,
        send_every: 1,
        queue: 4,
        color: (0, 0),
        w: 640,
        h: 480,
        fps: 30,
        max_frames: -1,
        imu: String::new(),
        imu_calib: String::new(),
        imu_bridge: 1.0,
        imu_max_blur: 15.0,
        imu_exposure: 0.03,
        imu_min_inliers: 60.0,
        imu_min_ratio: 0.6,
        imu_csv: String::new(),
        stress_gap: (0.0, 0.0),
        stress_imu_gap: false,
        stress_blur: 0.0,
        stress_gyro: String::new(),
    };
    let mut i = 1;
    let num = |s: &str| -> Result<f64, i32> {
        s.parse::<f64>().map_err(|_| {
            common::info!("숫자가 아님: {s}");
            2
        })
    };
    while i < argv.len() {
        let k = argv[i].as_str();
        let mut val = || -> Result<String, i32> {
            if i + 1 >= argv.len() {
                usage();
                return Err(2);
            }
            i += 1;
            Ok(argv[i].clone())
        };
        match k {
            "--tum" | "--rsdrec" => a.tum = val()?,
            "--imu-csv" => a.imu_csv = val()?,
            "--stress-gap" => {
                let v = val()?;
                let (p, l) = v.split_once(':').ok_or(2)?;
                a.stress_gap = (num(p)?, num(l)?);
            }
            "--stress-imu-gap" => a.stress_imu_gap = true,
            "--stress-blur" => a.stress_blur = num(&val()?)?,
            "--stress-gyro" => a.stress_gyro = val()?,
            "--realsense" => a.realsense = true,
            "--rate" => a.rate = num(&val()?)?,
            "--stride" => a.stride = num(&val()?)? as usize,
            "--serial" => a.serial = val()?,
            "--exposure" => a.exposure_ms = num(&val()?)?,
            "--ae-cap" => a.ae_cap_ms = num(&val()?)?,
            "--camera-profile" => a.camera_profile = val()?,
            "--gain" => a.gain = num(&val()?)? as i32,
            "--size" => {
                let s = val()?;
                if let Some((w, h)) = s.split_once('x') {
                    a.w = w.parse().unwrap_or(a.w);
                    a.h = h.parse().unwrap_or(a.h);
                }
            }
            "--color-size" => {
                let s = val()?;
                if let Some((w, h)) = s.split_once('x') {
                    a.color = (w.parse().unwrap_or(0), h.parse().unwrap_or(0));
                }
            }
            "--fps" => a.fps = num(&val()?)? as u32,
            "--out" => a.out = val()?,
            "--send-every" => a.send_every = (num(&val()?)? as i64).max(1),
            "--queue" => a.queue = (num(&val()?)? as i64).max(1) as usize,
            "--block" => a.block = 1,
            "--drop" => a.block = 0,
            "--poses" => a.poses = val()?,
            "--final-poses" => a.final_poses = val()?,
            "--stats" => a.stats = val()?,
            "--max-frames" => a.max_frames = num(&val()?)? as i64,
            "--imu" => a.imu = val()?,
            "--imu-calib" => a.imu_calib = val()?,
            "--imu-bridge" => a.imu_bridge = num(&val()?)?,
            "--imu-max-blur" => a.imu_max_blur = num(&val()?)?,
            "--imu-exposure" => a.imu_exposure = num(&val()?)?,
            "--imu-min-inliers" => a.imu_min_inliers = num(&val()?)?,
            "--imu-min-ratio" => a.imu_min_ratio = num(&val()?)?,
            "--reset-file" => a.reset_file = val()?,
            "--reset-at" => a.reset_at = num(&val()?)? as i64, // 시험용: 이 장면 번호에서 리셋 파일이 생긴 것처럼
            // 프로세스 설정(ondevice 가 처음에 읽음): 실시간 배치 · 메모리 묶기 · 조용히
            "--rt" => {
                let _ = val()?;
            }
            "--mlock" | "--quiet" => {}
            "-h" | "--help" => {
                usage();
                return Err(0);
            }
            _ => {
                common::info!("모르는 인자: {k}");
                usage();
                return Err(2);
            }
        }
        i += 1;
    }
    if a.tum.is_empty() == !a.realsense {
        usage();
        return Err(2);
    }
    Ok(a)
}

/// 명령줄 그대로 실행(`ondevice slam …`): 스트림을 파일 · 표준출력으로 쓴다
pub fn run(argv: &[String]) -> Result<i32, String> {
    match parse(argv) {
        Ok(a) => run_in(a),
        Err(c) => Ok(c),
    }
}

/// 같은 프로세스 루프로 넘기는 장면 줄: 길이 · 기다림/버림 · 그래프 블록을 버리지 않는 규칙은 파이프와 같다(기다림 = 루프가 다음 장면을 넣지 않음)
pub struct LocalQ {
    pub q: VecDeque<OutFrame>,
    pub cap: usize,
    pub block: bool,
    pub written: usize,
    pub dropped: usize,
}

/// 장면 스트림을 받는 쪽
pub enum Sink {
    /// 파일 · 파이프(쓰기 스레드)
    Pipe(Writer),
    /// 같은 프로세스 처리 루프(줄)
    Local(LocalQ),
}

impl Sink {
    /// 받는 쪽이 닫혔으면 false
    fn push(&mut self, f: OutFrame) -> bool {
        match self {
            Sink::Pipe(w) => w.push(f),
            Sink::Local(l) => {
                // 실시간: 줄이 차면 가장 오래된 장면을 버림(그래프 블록은 남김). 재생(block)은 루프가 줄이 빌 때까지 다음 장면을 넣지 않음
                if !f.is_graph() && !l.block && l.q.iter().filter(|x| !x.is_graph()).count() >= l.cap {
                    if let Some(i) = l.q.iter().position(|x| !x.is_graph()) {
                        if let Some(mut old) = l.q.remove(i) {
                            crate::bufpool::give_rgb(std::mem::take(&mut old.color));
                            crate::bufpool::give_depth_buf(std::mem::take(&mut old.depth));
                        }
                        l.dropped += 1;
                    }
                }
                if !f.is_graph() {
                    l.written += 1;
                }
                l.q.push_back(f);
                true
            }
        }
    }
    fn close(&mut self) {
        if let Sink::Pipe(w) = self {
            w.close();
        }
    }
    fn written(&self) -> usize {
        match self {
            Sink::Pipe(w) => w.written(),
            Sink::Local(l) => l.written,
        }
    }
    fn dropped(&self) -> usize {
        match self {
            Sink::Pipe(w) => w.dropped(),
            Sink::Local(l) => l.dropped,
        }
    }
    fn write_ms(&self) -> Vec<f64> {
        match self {
            Sink::Pipe(w) => w.write_ms(),
            Sink::Local(_) => vec![],
        }
    }
    /// 같은 프로세스 줄(받는 쪽 루프가 꺼냄)
    pub fn local(&mut self) -> Option<&mut LocalQ> {
        match self {
            Sink::Local(l) => Some(l),
            Sink::Pipe(_) => None,
        }
    }
}

/// 위치 프로그램 한 판의 상태: 장면마다 [`Runner::step`]. 따로 도는 프로그램(`ondevice slam`)은 [`run_in`] 이 입력을 돌리고,
/// 같은 프로세스(`ondevice --slam`)는 처리 루프가 장면이 올 때마다 부른다
pub struct Runner {
    t_run: Instant,
    calib: crate::imu::Calib,
    imu_replay: Option<crate::imu::CsvReplay>,
    pub imu_ring: Option<std::sync::Arc<crate::imu::Ring>>,
    imu_on: bool,
    skipped: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    stress_counts: Option<(std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<std::sync::atomic::AtomicUsize>)>,
    replay_max: bool,
    odom: OdometryKlt,
    map: Mapper,
    pub writer: Sink,
    poses_f: Option<std::io::BufWriter<std::fs::File>>,
    stats_f: Option<std::io::BufWriter<std::fs::File>>,
    /// 처음이거나 리셋 직후: 위치 추적이 새로 시작하는 것이 정상
    fresh: bool,
    /// 마지막 노드 시각
    node_t: f64,
    /// 받는 쪽에 보낸 노드 자세(광학 축 아님, 몸체)
    sent_nodes: BTreeMap<i32, Tf>,
    since_imu: u32,
    prev_corr: Tf,
    fast_prev: Option<(f64, [f32; 3])>,
    fast_vel: [f32; 3],
    tilt_map: f32,
    tilt_map_max: f32,
    s_read: Summary,
    s_odom: Summary,
    s_back: Summary,
    s_send: Summary,
    s_total: Summary,
    pub n: i64,
    n_lost: i64,
    n_restart: i64,
    n_corr: i64,
    n_graph: i64,
    n_graph_nodes: i64,
    n_imu_hold: i64,
    n_imu_reject: i64,
    n_fast_rot: i64,
    n_weak: i64,
    nq: [i64; 3],
    pub a: Args,
}

impl Runner {
    /// 준비: IMU · 위치 추적 · 지도(루프 · 그래프 스레드) · 출력. 장면 입력은 돌려주는 (원래 입력, 미리 할 일)로 부르는 쪽이 돌린다.
    /// local: 장면 스트림을 같은 프로세스 처리 루프의 줄로(아니면 파일 · 표준출력 쓰기 스레드)
    pub fn new(a: Args, local: bool, _pool: Option<std::sync::Arc<rayon::ThreadPool>>) -> Result<(Runner, Box<dyn FrameSource>, Option<crate::io::Prep>), String> {
        // 실시간이면 위치 추적 스레드를 분할 · memmap · 뷰어보다 먼저
        if a.realsense || a.rate > 0.0 {
            common::info!("위치 추적 스레드 우선: {}", common::threads::boost_tracking());
        }
        let calib = crate::imu::Calib::parse(&if a.imu_calib.is_empty() { crate::imu::DEFAULT_CALIB.to_string() } else { std::fs::read_to_string(&a.imu_calib).map_err(|e| format!("{}: {e}", a.imu_calib))? })?;
        if !a.imu_csv.is_empty() && a.tum.is_empty() {
            return Err("--imu-csv 는 --tum · --rsdrec 재생과 함께".into());
        }
        let mut imu_replay = if a.imu_csv.is_empty() { None } else { Some(crate::imu::CsvReplay::load(&a.imu_csv, calib.shift)?) };
        let mut imu_ring = None;
        let imu_fusion = if a.imu.is_empty() && imu_replay.is_none() {
            None
        } else {
            let ring = crate::imu::Ring::new();
            match &imu_replay {
                Some(r) => {
                    r.count_into(&ring);
                    let (t0, t1) = r.span();
                    common::info!("IMU 재생: {} 표본 {} 개({t0:.3} – {t1:.3})", a.imu_csv, r.len());
                }
                None => {
                    crate::imu::spawn_reader(&a.imu, ring.clone())?;
                }
            }
            imu_ring = Some(ring.clone());
            common::info!("모드: IMU 융합(시각 보정 {:+.4} 초, 이어 붙이기 최대 {} 초)", calib.shift, a.imu_bridge);
            Some(crate::imu::Fusion::new(ring, calib.clone(), a.imu_bridge))
        };
        let imu_on = imu_fusion.is_some();
        #[cfg(unix)]
        let stream_fd = (!local && a.out == "-").then(sys::take_stdout);
        #[cfg(not(unix))]
        let stream_fd: Option<std::fs::File> = None;

        #[allow(unused_assignments)]
        let mut skipped = None;
        let mut stress_counts = None;
        let src: Box<dyn FrameSource> = if !a.tum.is_empty() {
            let mut ts = TumSource::with_stride(&a.tum, a.rate, a.stride)?;
            skipped = Some(ts.skipped_counter());
            if a.stress_gap.0 > 0.0 && a.stress_gap.1 > 0.0 {
                let t0 = ts.t_first();
                ts.drop = (1..10_000).map(|k| (t0 + k as f64 * a.stress_gap.0, t0 + k as f64 * a.stress_gap.0 + a.stress_gap.1)).collect();
                if let (Some(r), true) = (imu_replay.as_mut(), a.stress_imu_gap) {
                    r.drop = ts.drop.clone();
                }
                common::info!("끊김 시험: {} 초마다 {} 초 장면 뺌{}", a.stress_gap.0, a.stress_gap.1, if a.stress_imu_gap { "(IMU 표본도)" } else { "" });
            }
            if a.stress_blur > 0.0 {
                let gp = if a.stress_gyro.is_empty() { &a.imu_csv } else { &a.stress_gyro };
                if gp.is_empty() {
                    return Err("--stress-blur 에는 --stress-gyro 또는 --imu-csv 가 필요".into());
                }
                let s = crate::imu::CsvReplay::load(gp, calib.shift)?.samples();
                let (r_ci, shift) = (calib.r_ci, calib.shift);
                ts.blur = Some((a.stress_blur, Box::new(move |t| crate::imu::CsvReplay::gyro_at(&s, &r_ci, shift, t))));
                common::info!("흐림 시험: 노출 {} ms 동안 각속도만큼 번짐({gp})", a.stress_blur * 1e3);
            }
            stress_counts = Some((ts.n_drop.clone(), ts.n_blur.clone()));
            // 실시간 속도 재생: IMU 표본을 장면 시계에 맞춰 링에 넣는 스레드(최대 속도 재생은 step 이 장면 시각까지 넣음)
            if let (Some(r), Some(ring)) = (imu_replay.take_if(|_| a.rate > 0.0), imu_ring.as_ref()) {
                r.spawn_timed(ring.clone(), ts.clock(), a.rate);
            }
            Box::new(ts)
        } else {
            open_camera(&a)?
        };
        let replay_max = !a.tum.is_empty() && a.rate == 0.0;
        let block = if a.block < 0 { replay_max } else { a.block == 1 };
        common::info!("입력: {}", src.describe());
        common::info!("출력: {} ({}, 큐 {}, {} 장마다)", if local { "물체 기억 메모리 큐" } else { &a.out }, if block { "받는 쪽을 기다림" } else { "밀리면 오래된 장면 버림" }, a.queue, a.send_every);

        let mut odom = OdometryKlt::new();
        if let Some(f) = imu_fusion {
            odom.imu = Some(f);
        }
        if a.realsense || a.rate > 0.0 {
            odom.set_async_desc();
        }
        let cfg = LcConfig::from_dir(&common::models::dir().to_string_lossy());
        for f in [&cfg.megaloc, &cfg.head, &cfg.xfeat] {
            if !std::path::Path::new(f).exists() {
                return Err(format!("장소 인식 모델이 없음: {f} (모델 폴더 = $ONDEVICE_MODELS · 보드 ~/models · PC models/)"));
            }
        }
        common::info!("slam: 광류 + PnP{} → 지도 노드 {NODE_EVERY_S} 초마다(또는 {NODE_MIN_S} 초 뒤 {NODE_MOVE_M} m · {NODE_TURN_DEG}° 움직이면) → 근처 · MegaLoc {} · XFeat {}(후보 {} · 내적 ≥ {} · {} 초 안 노드 뺌) → 자세 그래프", if imu_on { " + IMU" } else { "" }, cfg.megaloc, cfg.xfeat, cfg.topk, cfg.min_score, cfg.skip_s);
        let map = Mapper::new(cfg, replay_max).map_err(|e| format!("장소 인식 신경망: {e}"))?;

        let writer = if local { Sink::Local(LocalQ { q: VecDeque::new(), cap: a.queue.max(1), block, written: 0, dropped: 0 }) } else { Sink::Pipe(Writer::new(&a.out, stream_fd, a.queue, block)?) };
        let mut poses_f = (!a.poses.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.poses).expect("--poses")));
        let mut stats_f = (!a.stats.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.stats).expect("--stats")));
        if let Some(f) = poses_f.as_mut() {
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → 지도, 보낸 그대로)");
        }
        if let Some(f) = stats_f.as_mut() {
            let _ = writeln!(f, "seq,t,read_ms,odom_ms,slam_ms,send_ms,lost,new_map,map_corrected,loop,features,inliers,held,no_pose,quality,method,klt_inliers,klt_ratio,reproj_rms,spread,jump{}", if imu_on { ",tilt_deg,no_map,tilt_map" } else { "" });
        }
        #[cfg(unix)]
        if !local {
            sys::on_stop_signals();
        }
        Ok((
            Runner {
                t_run: Instant::now(),
                calib,
                imu_replay,
                imu_ring,
                imu_on,
                skipped,
                stress_counts,
                replay_max,
                odom,
                map,
                writer,
                poses_f,
                stats_f,
                fresh: true,
                node_t: f64::NEG_INFINITY,
                sent_nodes: BTreeMap::new(),
                since_imu: u32::MAX,
                prev_corr: Tf::IDENTITY,
                fast_prev: None,
                fast_vel: [0.0; 3],
                tilt_map: -1.0,
                tilt_map_max: 0.0,
                s_read: Summary::default(),
                s_odom: Summary::default(),
                s_back: Summary::default(),
                s_send: Summary::default(),
                s_total: Summary::default(),
                n: 0,
                n_lost: 0,
                n_restart: 0,
                n_corr: 0,
                n_graph: 0,
                n_graph_nodes: 0,
                n_imu_hold: 0,
                n_imu_reject: 0,
                n_fast_rot: 0,
                n_weak: 0,
                nq: [0; 3],
                a,
            },
            src,
            None,
        ))
    }

    /// 바뀐 노드 자세를 그래프 블록으로(받는 쪽이 노드에 묶인 장면 · 청크를 옮김). 받는 쪽이 닫혔으면 false
    fn send_nodes(&mut self, ids: &[i32]) -> bool {
        let mut upd = vec![];
        for &id in ids {
            let Some(n) = self.map.nodes.get(&id) else { continue };
            if let Some(s) = self.sent_nodes.get(&id) {
                let (m, a) = tf_diff(s, &n.opt);
                if m < 1e-3 && a < 0.05 {
                    continue;
                }
            }
            self.sent_nodes.insert(id, n.opt);
            let tn = n.opt.mul(&rtf::OPTICAL);
            upd.push(GraphNode { id: id as u32, held: 0, t: tn.0.map(|v| v as f64) });
        }
        if upd.is_empty() {
            return true;
        }
        self.n_graph_nodes += upd.len() as i64;
        self.n_graph += 1;
        self.writer.push(OutFrame { graph: upd, ..Default::default() })
    }

    pub fn step(&mut self, mut fr: Frame, read_ms: f64, _prepped: Option<crate::Features>) -> bool {
        let n = self.n;
        let c0 = Instant::now();
        // 최대 속도 IMU 재생: 이 장면 시각(+ 앞서 읽는 몫)까지 표본을 링에 넣음
        if let (Some(r), Some(ring)) = (self.imu_replay.as_mut(), self.imu_ring.as_ref()) {
            r.feed(ring, fr.t + self.calib.shift + crate::imu::REPLAY_LEAD);
        }
        fr.cam.oriented = self.imu_on; // IMU 모드: 방향 BRIEF(굴려도 같은 설명자)
        fr.id = self.n + 1;
        let reset_req = self.n == self.a.reset_at || common::control::take_reset() || (!self.a.reset_file.is_empty() && std::path::Path::new(&self.a.reset_file).exists());
        if reset_req {
            let _ = std::fs::remove_file(&self.a.reset_file);
            common::info!("장면 {n}: 리셋 → 위치 추적 · 지도를 처음부터");
            self.odom.reset_tf(Tf::IDENTITY);
            self.map.reset();
            self.sent_nodes.clear();
            self.prev_corr = Tf::IDENTITY;
            self.node_t = f64::NEG_INFINITY;
            self.fresh = true;
            if !self.writer.push(OutFrame { reset: true, ..Default::default() }) {
                return false;
            }
        }

        // 앞단: 위치 추적(광류 + PnP + IMU)
        let c1 = Instant::now();
        let r = self.odom.process(&fr);
        let blurred = self.imu_on && (self.odom.imu_rate as f64).to_radians() * self.a.imu_exposure * fr.cam.fx > self.a.imu_max_blur;
        let pose = if r.lost { None } else { r.pose.as_ref().map(Tf::from_pose) };
        if pose.is_none() {
            // 놓침: 마지막 위치(IMU 모드면 그 뒤 자이로 회전까지)에서 다음 장면이 다시 시작. 지도는 그대로 하나
            let p = self.odom.pose_tf();
            self.odom.reset_tf(p);
        }
        let restarted = pose.is_some() && !self.fresh && r.covariance[0][0] >= 9999.0;
        if restarted {
            common::info!("장면 {n}: 위치 추적 다시 시작 → 같은 지도에 이어 감(그 구간은 약한 간선, 예전 자리를 찾으면 옮겨짐)");
            self.map.weak_next = true;
            self.n_restart += 1;
        }
        if pose.is_some() {
            self.fresh = false;
        }
        let dg = self.odom.diag;
        let imu_rej = self.imu_on && self.odom.imu_rejected;
        self.n_imu_reject += imu_rej as i64;
        self.since_imu = if self.imu_on && pose.is_some() && (dg.method == 5 || imu_rej) { 0 } else { self.since_imu.saturating_add(1) };
        let fast_rot = self.imu_on && pose.is_some() && blurred;
        let weak = self.imu_on && pose.is_some() && dg.method == 0 && ((dg.q.inliers as f64) < self.a.imu_min_inliers || (dg.q.ratio as f64) < self.a.imu_min_ratio);
        self.n_fast_rot += fast_rot as i64;
        self.n_weak += (weak && !fast_rot) as i64;
        let imu_hold = self.imu_on && pose.is_some() && (self.since_imu <= IMU_SETTLE || fast_rot || weak);
        self.n_imu_hold += imu_hold as i64;
        let quality = if pose.is_none() { 2u8 } else if dg.method != 0 || imu_hold { 1 } else { 0 };
        self.nq[quality as usize] += 1;
        let odom_ms = ms_since(c1);

        // 뒷단: 노드 넣기 · 끝난 루프 · 그래프 결과 거두기(기다리지 않음. 최대 속도 재생만 결정적이게 기다림)
        let c2 = Instant::now();
        let mut new_node = None;
        let node_due = |p: &Tf, m: &Mapper, node_t: f64| {
            let dt = fr.t - node_t;
            dt >= NODE_EVERY_S
                || (dt >= NODE_MIN_S
                    && m.nodes.values().next_back().is_some_and(|n| {
                        let (dm, dd) = tf_diff(&n.odom, p);
                        dm > NODE_MOVE_M || dd > NODE_TURN_DEG
                    }))
        };
        if let Some(p) = pose
            && !imu_hold
            && node_due(&p, &self.map, self.node_t)
        {
            self.node_t = fr.t;
            new_node = Some(self.map.add_node(&fr, p));
        }
        let hits = self.map.poll_lc();
        self.map.kick_opt();
        let moved = self.map.poll_opt(self.replay_max);
        let back_ms = ms_since(c2);
        let mut ids = moved;
        if let Some(id) = new_node {
            ids.push(id);
        }
        ids.sort_unstable();
        ids.dedup();
        if !ids.is_empty() && !self.send_nodes(&ids) {
            common::info!("받는 쪽이 닫힘: 끝냄");
            return false;
        }
        let corr = self.map.corr;
        let corrected = (0..12).any(|i| (corr.0[i] - self.prev_corr.0[i]).abs() > 1e-6);
        if corrected {
            self.prev_corr = corr;
            self.n_corr += 1;
        }
        let lost = pose.is_none();
        self.n_lost += lost as i64;
        let gravity = self.odom.gravity_aligned;
        let mut of = OutFrame {
            flags: if lost { stream::LOST } else { 0 } | if corrected { stream::MAP_CORRECTED } else { 0 } | if gravity { stream::GRAVITY } else { 0 } | if imu_hold { stream::NO_MAP } else { 0 } | if quality == 1 { stream::UNCERTAIN } else { 0 },
            map_id: 0,
            seq: self.n as u64,
            t: fr.t,
            ..Default::default()
        };
        if let Some(p) = pose {
            let body = corr.mul(&p);
            if self.imu_on {
                // 장면 사이 위치(IMU 적분)의 기준: 보낸 몸체 위치 · 속도
                let pp = [body.0[3], body.0[7], body.0[11]];
                match self.fast_prev {
                    Some((t0, p0)) if fr.t - t0 > 0.0 && fr.t - t0 < 0.2 => {
                        let dt = (fr.t - t0) as f32;
                        for i in 0..3 {
                            self.fast_vel[i] = (0.5 * self.fast_vel[i] + 0.5 * (pp[i] - p0[i]) / dt).clamp(-2.0, 2.0);
                        }
                    }
                    _ => self.fast_vel = [0.0; 3],
                }
                self.fast_prev = Some((fr.t, pp));
                if let Some(u) = self.odom.imu.as_ref()
                    && let Some(an) = u.anchor(fr.t, body, rtf::OPTICAL, self.fast_vel)
                {
                    u.ring().set_fast_anchor(an);
                }
                self.tilt_map = self.odom.imu.as_ref().map(|u| u.tilt_deg(&body)).unwrap_or(-1.0);
                self.tilt_map_max = self.tilt_map_max.max(self.tilt_map);
            }
            let twc = body.mul(&rtf::OPTICAL);
            of.pose = twc.0.map(|v| v as f64);
            if let Some(f) = self.poses_f.as_mut() {
                write_tum(f, fr.t, &twc);
            }
            let last = self.map.last_id();
            of.anchor = if self.sent_nodes.contains_key(&last) { last as u32 } else { 0 };
        }
        of.fx = fr.cam.fx;
        of.fy = fr.cam.fy;
        of.cx = fr.cam.cx;
        of.cy = fr.cam.cy;
        of.depth_per_m = fr.depth_per_m;
        let mut send_ms = 0.0;
        if self.n % self.a.send_every == 0 {
            let c3 = Instant::now();
            of.width = fr.cam.width;
            of.height = fr.cam.height;
            of.color = std::mem::take(&mut fr.color);
            of.color_is_rgb = fr.color_is_rgb;
            of.depth = fr.take_stream_depth();
            if !self.writer.push(of) {
                common::info!("받는 쪽이 닫힘: 끝냄");
                return false;
            }
            send_ms = ms_since(c3);
        }
        let total_ms = ms_since(c0);
        self.s_read.add(read_ms);
        if self.n > 0 {
            self.s_odom.add(odom_ms);
        }
        self.s_back.add(back_ms);
        self.s_send.add(send_ms);
        self.s_total.add(total_ms);
        if let Some(f) = self.stats_f.as_mut() {
            let _ = write!(
                f,
                "{n},{:.6},{read_ms:.3},{odom_ms:.3},{back_ms:.3},{send_ms:.3},{},0,{},{},{},{},0,{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
                fr.t, lost as i32, corrected as i32, (hits > 0) as i32, self.odom.tracks(), r.inliers, lost as i32, quality, dg.method, dg.q.inliers, dg.q.ratio, dg.q.rms, dg.q.spread, dg.jump
            );
            if let Some(u) = self.odom.imu.as_ref() {
                let _ = write!(f, ",{:.2},{},{:.2}", u.tilt_deg(&self.odom.pose_tf()), imu_hold as i32, self.tilt_map);
            }
            let _ = writeln!(f);
        }
        self.n += 1;
        true
    }

    pub fn finish(mut self, src_dropped: usize) -> Result<i32, String> {
        let wall = self.t_run.elapsed().as_secs_f64();
        // 남은 루프 · 최적화를 끝까지 반영(끝 지도)
        while self.map.lc.pending > 0 {
            self.map.lc_block = true;
            let before = self.map.stats.done;
            self.map.stats.sent += 1; // poll_lc 의 "바로 앞 하나 빼고"를 없앰
            self.map.poll_lc();
            self.map.stats.sent -= 1;
            if self.map.stats.done == before {
                break;
            }
        }
        self.map.kick_opt();
        let moved = self.map.poll_opt(true);
        let _ = self.send_nodes(&moved);
        if let Some(r) = &self.imu_ring {
            r.closed.store(true, Ordering::Relaxed); // 읽기 · 재생 · 위치 내기 스레드 끝냄
        }
        self.writer.close();
        let mut s_write = Summary::default();
        for x in self.writer.write_ms() {
            s_write.add(x);
        }
        if !self.a.final_poses.is_empty() {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&self.a.final_poses).map_err(|e| format!("{}: {e}", self.a.final_poses))?);
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → 지도, 끝에 최적화된 지도 노드)");
            for n in self.map.nodes.values() {
                write_tum(&mut f, n.t, &n.opt.mul(&rtf::OPTICAL));
            }
            common::info!("최적화된 지도 노드 {} 개 → {}", self.map.nodes.len(), self.a.final_poses);
        }
        drop(self.poses_f.take());
        drop(self.stats_f.take());
        let n = self.n;
        let l = &self.map.stats;
        let d = l.done.max(1) as f64;
        common::info!(
            "루프: 노드 {} 개 보냄 · {} 개 끝남, 검증 {} 번, 맞음 {} · 위치 추적과 같아 버림 {} · 간격 안이라 건너뜀 {} · 지도와 너무 달라 버림 {} · 간선 {} · 그래프와 맞지 않아 버림 {}, ms/노드 MegaLoc {:.1}(최대 {:.1}) · XFeat {:.1}(최대 {:.1}) · 짝 + 검증 {:.1}(최대 {:.1})",
            l.sent, l.done, l.tried, l.hits, l.redundant, l.gap, l.gated, l.accepted, l.dropped, l.ms[0] / d, l.ms_max[0], l.ms[1] / d, l.ms_max[1], l.ms[2] / d, l.ms_max[2]
        );
        common::info!("자세 그래프: 노드 {} · 간선 {}(다시 시작 구간 약한 간선 {}), 풀이 ms {}", self.map.nodes.len(), self.map.links.len(), self.map.n_gap_edges, self.map.s_opt.str());
        common::info!("장면 {n} 개, {wall:.1} 초({:.1} 장/초), 놓침 {}, 위치 추적 다시 시작 {}, 지도 보정 바뀜 {}", n as f64 / wall.max(1e-9), self.n_lost, self.n_restart, self.n_corr);
        common::info!("위치 질: 좋음 {} · 불확실 {} · 놓침 {}", self.nq[0], self.nq[1], self.nq[2]);
        if let Some(u) = self.odom.imu.as_ref() {
            let (ns, dr, fl) = u.counts();
            common::info!(
                "IMU: 표본 {ns} 개(링 가득 차 버림 {dr}, 읽기 실패 {fl}), IMU 회전으로 이어 붙인 장면 {}, 자이로 치우침 {:.4} {:.4} {:.4} rad/s(영상으로 고친 수 {}, 되돌린 수 {}), 지금 기울기 {:.1}°",
                self.odom.by_imu, u.bias[0], u.bias[1], u.bias[2], u.n_bias_upd, u.n_bias_reset, u.tilt_deg(&self.odom.pose_tf())
            );
            common::info!("IMU: 영상이 IMU 회전과 어긋나 버린 장면 {}, 위치만 보내고 지도에 안 쌓은 장면 {}(번짐 {} · 영상 근거 약함 {}), 가만히 있을 때 기울기를 중력 쪽으로 고친 장면 {}(합 {:.1}°)", self.n_imu_reject, self.n_imu_hold, self.n_fast_rot, self.n_weak, self.odom.n_grav, self.odom.grav_deg);
            common::info!("IMU: 보낸 위치 기준 지도 기울기 마지막 {:.1}° · 최대 {:.1}°, 장면 사이 위치 {} 개", self.tilt_map, self.tilt_map_max, crate::imu::FAST_N.load(Ordering::Relaxed));
        }
        if let Some(s) = &self.skipped {
            common::info!("재생에서 밀려 건너뛴 장면 {}", s.load(Ordering::Relaxed));
        }
        if let Some((d, b)) = &self.stress_counts {
            common::info!("시험 조건: 뺀 장면 {}, 흐리게 한 장면 {}", d.load(Ordering::Relaxed), b.load(Ordering::Relaxed));
        }
        if src_dropped > 0 {
            common::info!("위치 추적이 밀려 버린 장면 {src_dropped}");
        }
        common::info!("보냄 {}, 밀려 버림 {}, 그래프 블록 {} 개(노드 {} 개)", self.writer.written(), self.writer.dropped(), self.n_graph, self.n_graph_nodes);
        common::info!("ms/장  읽기    {}", self.s_read.str());
        common::info!("ms/장  위치추적 {}", self.s_odom.str());
        common::info!("ms/장  뒷단 거두기 {} (최대 {:.1})", self.s_back.str(), self.s_back.max());
        common::info!("ms/장  큐넣기  {}", self.s_send.str());
        common::info!("ms/장  파이프쓰기(쓰기 스레드) {}", s_write.str());
        common::info!("ms/장  합(본 스레드) {}", self.s_total.str());
        Ok(0)
    }
}

/// 따로 도는 위치 프로그램(`ondevice slam`): 입력을 읽기 스레드(PrefetchSource)로 돌리며 장면마다 step
fn run_in(a: Args) -> Result<i32, String> {
    let (max, replay_max) = (a.max_frames, !a.tum.is_empty() && a.rate == 0.0);
    let (mut r, src, prep) = Runner::new(a, false, None)?;
    let mut src = PrefetchSource::with_prep(src, !replay_max, prep);
    while !STOP.load(Ordering::Relaxed) && (max < 0 || r.n < max) {
        let Some(fr) = src.next()? else { break };
        let (ms, pf) = (src.last_read_ms(), src.take_prepped());
        if !r.step(fr, ms, pf) {
            break;
        }
    }
    r.finish(src.dropped())
}

/// D435 입력(rsdriver: V4L2 직접). 빌드에 든 것만 고를 수 있다
fn open_camera(a: &Args) -> Result<Box<dyn FrameSource>, String> {
    #[cfg(feature = "rsdriver")]
    {
        Ok(Box::new(crate::io::RsDriverSource::new(a.w, a.h, a.color, a.fps, &a.serial, a.exposure_ms, a.gain, a.ae_cap_ms, &a.camera_profile)?))
    }
    #[cfg(not(feature = "rsdriver"))]
    {
        let _ = (&a.serial, a.exposure_ms, a.gain, a.ae_cap_ms, a.w, a.h, a.color, a.fps, &a.camera_profile);
        Err("D435 입력은 리눅스 빌드(rsdriver 기능)에만 있음".into())
    }
}
