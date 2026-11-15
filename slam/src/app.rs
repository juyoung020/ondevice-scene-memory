//! 위치 프로그램: RGB-D 입력 → 위치 추적(매 장) + 지도 갱신(DetectionRate) → 장면(같은 프로세스의 메모리 큐 또는 장면 스트림).
//! 원본: RTAB-Map 연결 프로그램 rtab_bridge(C++). 명령줄·장면마다의 판단(다시 잡기, 새 지도, 맡아 둠, 그래프 블록)·끝 요약을 그대로 옮겼다.
//!
//!   ondevice --slam "--tum <폴더> --rate 0" ...   (한 프로세스: `spawn` 이 이 루프를 스레드로 돌리고 장면을 메모리 큐로 넘김)
//!   ondevice slam --tum <폴더> [--rate 1] --out <스트림 파일>   (위치만: 장면 스트림을 파일·표준출력으로)
//!
//!
//! 위치 추적·지도는 `engine::Engine` 하나로 부른다(특징점 → OdometryF2F, 다시 잡기 RegistrationVis, 지도 Slam, 난수 상태 하나를 함께 씀).
use crate::io::{FrameSource, PrefetchSource, TumSource};
use crate::params::Params;
use crate::stream::{self, GraphNode, OutFrame, Writer};
use crate::register::{rtf, Tf};
use std::collections::{BTreeMap, VecDeque};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// 위치를 TUM 한 줄로: 시각 x y z qx qy qz qw
fn write_tum(o: &mut impl std::io::Write, t: f64, m: &Tf) {
    // Transform::getQuaternionf = Quaternionf(linear).normalized() (float)
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
}

fn ms_since(a: Instant) -> f64 {
    a.elapsed().as_secs_f64() * 1e3
}

mod engine {
    //! 위치 추적 · 다시 잡기 · 지도(RTAB-Map Odometry · RegistrationVis · Rtabmap 자리)를 한 데 묶은 연결부.
    //! 원본 한 프로세스와 같은 결과가 나오도록 두 가지를 맞춘다.
    //!  - 위치는 모두 RTAB-Map Transform 과 같은 float 3x4(Tf) 로 다룸(곱·역행렬 비트가 원본과 같아야 함)
    //!  - 무작위 상태(glibc rand() · std::random_device 호출 수)는 프로세스에 하나: 모든 모듈이 같은 SharedRng 를 원본과 같은 순서로 씀
    use crate::features::Extractor;
    use crate::loopclose::flann::KdTreeIndex;
    use crate::loopclose::mem::{Kp, NodeFeatures, Signature};
    use crate::loopclose::{GtsamOptimizer, LoopRegistration, NodeInput, RegOut, Slam, WordIndex};
    use crate::odom::OdometryF2F;
    use crate::klt::OdometryKlt;

    /// 위치 추적 방식: Odom/Strategy=9 → 광류(klt.rs, 기본), 그 밖 → RTAB-Map F2F(비교용)
    enum Odom {
        F2f(OdometryF2F),
        Klt(OdometryKlt),
    }
    use crate::params::Params;
    use crate::register::{new_shared_rng, RegistrationVis, SharedRng, Tf};
    use crate::{Camera, Features, Frame};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    /// 위치 추적 한 장의 결과(OdometryInfo 에서 쓰는 칸)
    #[derive(Clone)]
    pub struct OdomOut {
        /// None = 위치 없음(Transform 이 null)
        pub pose: Option<Tf>,
        pub lost: bool,
        /// 공분산 6×6(행 우선). [0] >= 9999 = 위치 추적이 다시 시작함
        pub cov: [f64; 36],
        pub features: i32,
        pub inliers: i32,
        /// OdometryInfo::transform · interval(OdometryEvent::velocity 에 씀)
        pub transform: Tf,
        pub interval: f64,
    }

    impl OdomOut {
        /// OdometryEvent::velocity(): interval > 0 이면 transform 의 xyz·rpy / interval(float / double → float)
        pub fn velocity(&self) -> Vec<f32> {
            if self.interval > 0.0 {
                self.transform.xyz_rpy().iter().map(|&v| (v as f64 / self.interval) as f32).collect()
            } else {
                Vec::new()
            }
        }
    }

    /// 다시 잡기 결과: 기준 장면 → 지금 장면 변환, 맞은 점, 공분산
    pub struct RelocOut {
        pub t: Tf,
        pub inliers: i32,
        pub cov: [f64; 36],
    }

    /// 기준 장면(원본 Signature(SensorData(영상, 깊이))): 특징은 처음 다시 잡기에 쓸 때 뽑음
    pub struct RefSig {
        frame: Frame,
        /// 지금 Signature 의 단어 특징(원본 단어 순서). None = 아직 안 뽑음
        feats: Option<Features>,
        /// 특징마다의 단어 id. None = 처음(뽑은 순서대로 1, 2, … 와 같은 뜻)
        ids: Option<Vec<i32>>,
    }

    fn cov36(c: &[[f64; 6]; 6]) -> [f64; 36] {
        let mut o = [0.0; 36];
        for r in 0..6 {
            o[r * 6..r * 6 + 6].copy_from_slice(&c[r]);
        }
        o
    }

    /// 지도 사전 색인(rtflann KD-tree): 짓기·더하기·지우기 때 공유 난수 상태를 빌려 쓰고 돌려줌
    struct SharedIndex {
        inner: KdTreeIndex,
        rng: SharedRng,
    }

    impl SharedIndex {
        fn with<R>(&mut self, f: impl FnOnce(&mut KdTreeIndex) -> R) -> R {
            let used = Arc::new(AtomicU32::new(0));
            {
                let st = self.rng.lock().unwrap();
                self.inner.rand = st.rand.clone();
                let start = st.rd_calls;
                let u = used.clone();
                self.inner.set_seed_source(move || {
                    let k = start.wrapping_add(u.fetch_add(1, Ordering::Relaxed));
                    k.wrapping_add(1).wrapping_mul(2654435761)
                });
            }
            let r = f(&mut self.inner);
            let mut st = self.rng.lock().unwrap();
            st.rand = self.inner.rand.clone();
            st.rd_calls = st.rd_calls.wrapping_add(used.load(Ordering::Relaxed));
            r
        }
    }

    impl WordIndex for SharedIndex {
        fn is_built(&self) -> bool {
            self.inner.is_built()
        }
        fn build(&mut self, rows: &[[u8; 32]]) {
            self.with(|i| i.build(rows))
        }
        fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32> {
            self.with(|i| i.add_points(rows))
        }
        fn remove_point(&mut self, idx: u32) {
            self.with(|i| i.remove_point(idx))
        }
        fn knn_search(&mut self, q: &[[u8; 32]], k: usize, checks: i32) -> (Vec<i32>, Vec<f32>) {
            self.inner.knn_search(q, k, checks)
        }
        fn release(&mut self) {
            self.inner.release()
        }
    }

    /// Memory::computeTransform(fromS, toS, guess) → RegistrationVis. 노드의 단어 특징을 순번 그대로 넘김
    /// (노드 단어 id 는 겹치는 것이 있어 원본도 id 를 새로 매김 = 순번 순서. 겹침이 없는 노드는 세어 둠)
    struct LoopReg {
        reg: RegistrationVis,
        cam: Arc<Mutex<Option<Camera>>>,
        unique_ids: Arc<AtomicU32>,
    }

    fn sig_features(s: &Signature) -> Features {
        Features {
            kps: s.kpts.iter().map(|k| crate::KeyPoint { x: k.x, y: k.y, size: k.size, angle: k.angle, response: k.response, octave: k.octave }).collect(),
            pts3d: s.pts3d.iter().map(|p| nalgebra::Point3::new(p[0], p[1], p[2])).collect(),
            desc: s.desc.clone(),
        }
    }

    impl LoopRegistration for LoopReg {
        fn compute(&mut self, from: &Signature, to: &Signature, guess: Tf) -> RegOut {
            if from.words.values().all(|v| v.len() == 1) {
                self.unique_ids.fetch_add(1, Ordering::Relaxed);
            }
            let cam = self.cam.lock().unwrap().clone().expect("카메라 없음");
            let info = self.reg.compute_tf(&sig_features(from), &sig_features(to), &cam, guess);
            RegOut { t: info.tf, cov: cov36(&info.covariance), inliers: info.inliers as i32, matches: info.matches as i32 }
        }
    }

    /// 지도 단계 결과(동기면 매번, 지도 스레드면 끝난 단계마다 갱신)
    #[derive(Clone, Default)]
    pub struct MapSt {
        loop_id: i32,
        corr: ((f32, f32), u32),
        lp: BTreeMap<i32, Tf>,
        last_id: i32,
        map_corr: Tf,
        jobs: u64,
    }

    fn snap_of(s: &Slam) -> MapSt {
        MapSt {
            loop_id: s.loop_closure_id(),
            corr: (s.info.max_correction, s.info.rejected_by_correction),
            lp: s.local_optimized_tf().clone(),
            last_id: s.last_location_id(),
            map_corr: s.map_correction(),
            jobs: 0,
        }
    }

    /// 지도 스레드에서 노드 특징을 뽑을 장면
    pub struct RawNode {
        gray: Vec<u8>,
        depth: crate::DepthBuf,
        cam: Camera,
        pose: Option<Tf>,
        trusted: bool,
    }

    type KfOut = (Features, Tf, bool, i32);

    pub enum Cmd {
        Proc(Box<NodeInput>, i32, u64, bool, Option<Box<RawNode>>),
        NewMap(std::sync::mpsc::SyncSender<i32>),
        Reset(std::sync::mpsc::SyncSender<()>),
    }

    /// 지도(Slam)를 이 스레드에서 바로 부르거나, 지도 스레드에 일을 넘김
    pub enum MapHost {
        Inline(Box<Slam>, MapSt),
        Thread {
            tx: Option<std::sync::mpsc::Sender<Cmd>>,
            h: Option<std::thread::JoinHandle<Box<Slam>>>,
            /// 끝난 단계 결과 + 그 단계에서 정해진 (일 번호, 노드 id)
            done: Arc<Mutex<(Vec<MapSt>, Vec<(u64, i32)>, Vec<KfOut>)>>,
            st: MapSt,
            jobs: u64,
        },
        Empty,
    }

    impl MapHost {
        fn is_thread(&self) -> bool {
            matches!(self, MapHost::Thread { .. })
        }
        fn jobs(&self) -> u64 {
            match self {
                MapHost::Thread { jobs, .. } => *jobs,
                MapHost::Inline(_, st) => st.jobs,
                MapHost::Empty => 0,
            }
        }
        /// 지금 볼 결과
        fn st(&self) -> &MapSt {
            match self {
                MapHost::Thread { st, .. } => st,
                MapHost::Inline(_, st) => st,
                MapHost::Empty => unreachable!(),
            }
        }
        /// 노드 넣기. 돌려줌: 이번에 id 가 정해진 (일 번호, 노드 id)
        fn process(&mut self, inp: NodeInput, forced: i32, node: bool, job: u64, raw: Option<Box<RawNode>>) -> (Vec<(u64, i32)>, Vec<KfOut>) {
            match self {
                MapHost::Inline(s, st) => {
                    s.forced_hyp = forced;
                    s.process(&inp);
                    *st = snap_of(s);
                    st.jobs = job;
                    (if node { vec![(job, s.last_location_id())] } else { vec![] }, vec![])
                }
                MapHost::Thread { tx, done, st, jobs, .. } => {
                    *jobs = job;
                    let _ = tx.as_ref().unwrap().send(Cmd::Proc(Box::new(inp), forced, job, node, raw));
                    // 끝난 단계가 있으면 가장 최근 결과로(루프 클로저는 그 단계들 가운데 있었으면 알림)
                    let (sts, ids, kfs) = std::mem::take(&mut *done.lock().unwrap());
                    if let Some(last) = sts.last() {
                        let lp_loop = sts.iter().rev().find(|x| x.loop_id > 0).map(|x| x.loop_id).unwrap_or(0);
                        let rej: u32 = sts.iter().map(|x| x.corr.1).sum();
                        let mc = sts.iter().fold((0f32, 0f32), |m, x| (m.0.max(x.corr.0 .0), m.1.max(x.corr.0 .1)));
                        *st = last.clone();
                        st.loop_id = lp_loop;
                        st.corr = (mc, rej);
                    } else {
                        st.loop_id = 0;
                        st.corr = ((0.0, 0.0), 0);
                    }
                    (ids, kfs)
                }
                MapHost::Empty => unreachable!(),
            }
        }
        /// 지도 스레드에 쌓인 일이 끝나길 기다림(새 지도·리셋·끝 그래프처럼 순서가 중요한 일 앞에서)
        fn trigger_new_map(&mut self) -> i32 {
            match self {
                MapHost::Inline(s, _) => s.trigger_new_map(),
                MapHost::Thread { tx, .. } => {
                    let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
                    let _ = tx.as_ref().unwrap().send(Cmd::NewMap(rtx));
                    rrx.recv().unwrap_or(-1)
                }
                MapHost::Empty => -1,
            }
        }
        fn reset(&mut self) {
            match self {
                MapHost::Inline(s, st) => {
                    s.reset();
                    *st = snap_of(s);
                }
                MapHost::Thread { tx, st, .. } => {
                    let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
                    let _ = tx.as_ref().unwrap().send(Cmd::Reset(rtx));
                    let _ = rrx.recv();
                    *st = MapSt::default();
                }
                MapHost::Empty => {}
            }
        }
        fn start_thread(&mut self, ext: Extractor) {
            let MapHost::Inline(..) = self else { return };
            let MapHost::Inline(mut slam, st0) = std::mem::replace(self, MapHost::Empty) else { unreachable!() };
            let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
            let done: Arc<Mutex<(Vec<MapSt>, Vec<(u64, i32)>, Vec<KfOut>)>> = Default::default();
            let d2 = done.clone();
            let h = std::thread::spawn(move || {
                common::threads::apply(5, 0); // 지도 스레드는 위치 추적보다 뒤(nice +5)
                while let Ok(c) = rx.recv() {
                    match c {
                        Cmd::Proc(mut inp, forced, job, node, raw) => {
                            let mut kf = None;
                            if let Some(r) = raw {
                                let fr = Frame { id: 0, t: inp.stamp, gray: r.gray, color: Vec::new(), color_is_rgb: true, depth_mm: r.depth, depth_raw: Default::default(), depth_per_m: 1000.0, cam: r.cam };
                                let feats = ext.extract(&fr);
                                inp.features = NodeFeatures {
                                    kps: feats.kps.iter().map(|k| Kp { x: k.x, y: k.y, size: k.size, angle: k.angle, response: k.response, octave: k.octave, class_id: -1 }).collect(),
                                    pts3d: feats.pts3d.iter().map(|p| [p.x, p.y, p.z]).collect(),
                                    desc: feats.desc.clone(),
                                };
                                if let Some(p) = r.pose {
                                    kf = Some((feats, p, r.trusted));
                                }
                            }
                            slam.forced_hyp = forced;
                            slam.process(&inp);
                            let st = snap_of(&slam);
                            let mut d = d2.lock().unwrap();
                            d.0.push(st);
                            if node {
                                d.1.push((job, slam.last_location_id()));
                            }
                            if let Some((f, p, t)) = kf {
                                d.2.push((f, p, t, slam.last_location_id()));
                            }
                        }
                        Cmd::NewMap(r) => {
                            let _ = r.send(slam.trigger_new_map());
                        }
                        Cmd::Reset(r) => {
                            slam.reset();
                            let _ = r.send(());
                        }
                    }
                }
                slam
            });
            let jobs = st0.jobs;
            *self = MapHost::Thread { tx: Some(tx), h: Some(h), done, st: st0, jobs };
        }
        /// 지도 스레드를 끝내고 Slam 을 돌려받음(끝 그래프)
        fn inline(&mut self) -> &mut Slam {
            if let MapHost::Thread { tx, h, .. } = self {
                drop(tx.take());
                let s = h.take().unwrap().join().expect("지도 스레드");
                let st = snap_of(&s);
                *self = MapHost::Inline(s, st);
            }
            match self {
                MapHost::Inline(s, _) => s,
                _ => unreachable!(),
            }
        }
    }

    pub struct Engine {
        ext: Extractor,
        odom: Odom,
        reloc: RegistrationVis,
        slam: MapHost,
        cam: Arc<Mutex<Option<Camera>>>,
        /// 지금 장면 SensorData 의 특징(위치 추적이 data.setFeatures 로 채운 것)
        cur: Features,
        /// Odometry 의 previous_stamp · framesProcessed 그림자(OdometryInfo::interval 계산용)
        prev_stamp: f64,
        frames: i32,
        pub loop_unique_ids: Arc<AtomicU32>,
        pub ref_reuse: u32,
        pub ref_unknown: u32,
        /// 지도 노드마다 특징(기술자·3D)과 그때 위치 추적 위치: 놓쳤을 때 지도 전체에서 다시 잡기(global_reloc)
        kf_db: Vec<(Features, Tf)>,
        /// kf_db 와 같은 순서: (지도 노드 id, 믿는 지도의 노드인지)
        kf_ids: Vec<(i32, bool, u64)>,
        /// 다음 지도 단계에 넘길 루프 후보
        forced_hyp: i32,
        /// 이어 붙인 구간 시작 노드 id(0 = 없음): 이 앞 노드들에서만 제자리 후보를 찾고, 루프가 이 구간을 옮기는 것은 허용
        pub prov_from: i32,
        /// 지금 지도가 믿는 지도인지(app 이 slam_process 앞에 알려 줌)
        pub trusted: bool,
    }

    impl Engine {
        /// params = Rtabmap 설정, odom_params = 위치 추적 설정, reloc_min = 다시 잡기 Vis/MinInliers
        pub fn new(params: &Params, odom_params: &Params, reloc_min: i32) -> Engine {
            let rng = new_shared_rng();
            let odom = if odom_params.get("Odom/Strategy", 0i32) == 9 {
                Odom::Klt(OdometryKlt::new(odom_params))
            } else {
                let mut o = OdometryF2F::new(odom_params);
                o.set_rng(rng.clone());
                Odom::F2f(o)
            };
            let mut rp = params.clone();
            rp.set("Vis/MinInliers", &reloc_min.to_string());
            let mut reloc = RegistrationVis::new(&rp);
            reloc.set_rng(rng.clone());
            let mut lreg = RegistrationVis::new(params);
            lreg.set_rng(rng.clone());
            let cam = Arc::new(Mutex::new(None));
            let unique = Arc::new(AtomicU32::new(0));
            let index = SharedIndex { inner: KdTreeIndex::from_params(params), rng: rng.clone() };
            let opt = GtsamOptimizer(crate::graph::OptimizerParams::from_params(params));
            let slam = Slam::new(params, Box::new(index), Box::new(opt), Box::new(LoopReg { reg: lreg, cam: cam.clone(), unique_ids: unique.clone() }));
            Engine {
                ext: Extractor::new(odom_params),
                odom,
                reloc,
                slam: { let st = snap_of(&slam); MapHost::Inline(Box::new(slam), st) },
                cam,
                cur: Features::default(),
                prev_stamp: 0.0,
                frames: 0,
                loop_unique_ids: unique,
                ref_reuse: 0,
                ref_unknown: 0,
                kf_db: Vec::new(),
                kf_ids: Vec::new(),
                forced_hyp: 0,
                prov_from: 0,
                trusted: true,
            }
        }

        /// Odometry::process. feats = None 이면 새 SensorData(특징 없음 → 뽑음), Some 이면 이미 특징이 붙은 data
        pub fn odom_process(&mut self, f: &Frame, feats: Option<Features>) -> OdomOut {
            {
                let mut c = self.cam.lock().unwrap();
                if c.is_none() {
                    *c = Some(f.cam.clone());
                }
            }
            let stamp = f.t;
            let dt = if self.prev_stamp > 0.0 || (self.prev_stamp == 0.0 && self.frames == 1) { stamp - self.prev_stamp } else { 0.0 };
            let r = match &mut self.odom {
                Odom::F2f(o) => {
                    let feats = feats.unwrap_or_else(|| self.ext.extract(f));
                    o.process(f, feats)
                }
                Odom::Klt(o) => o.process(f),
            };
            let n_feat = match &self.odom {
                Odom::Klt(o) => o.tracks() as i32,
                _ => r.features.kps.len() as i32,
            };
            if r.pose.is_some() {
                self.prev_stamp = stamp;
                self.frames += 1;
            }
            // 놓쳐도 이전 시각은 그대로(Odom/ResetCountdown=0 의 원본 동작)
            let out = OdomOut {
                pose: r.pose.map(|_| r.tf),
                lost: r.lost,
                cov: cov36(&r.covariance),
                features: n_feat,
                inliers: r.inliers as i32,
                transform: r.transform,
                interval: if dt > 0.0 { dt } else { 0.0 },
            };
            self.cur = r.features;
            out
        }

        /// 지금 장면에 붙은 특징(다시 잡은 뒤 같은 data 로 위치 추적을 한 번 더 할 때)
        /// 광류 경로는 특징(BRIEF)을 장면마다 뽑지 않는다: 지도 노드·다시 잡기에 필요할 때만 이 장면에서 뽑음
        pub fn ensure_cur(&mut self, f: &Frame) {
            if matches!(self.odom, Odom::Klt(_)) && self.cur.kps.is_empty() {
                self.cur = self.ext.extract(f);
            }
        }


        /// 지도 전체에서 다시 잡기: 모든 지도 노드와 기술자 닮음(해밍 < 40 인 짝 수, 이번 특징 100 개 표본)으로 순위를 매기고
        /// 위 3 개만 PnP 로 맞춰 본다(최근 기준 장면만 보는 reloc 와 달리 오래전 노드도). 성공하면 (위치, 맞은 점, 공분산)
        /// 광류 경로: 방금 다시 시작한 장면이 마지막 성공 장면 깊이와 ICP 로 맞았는지
        pub fn bridge_verified(&self) -> bool {
            matches!(&self.odom, Odom::Klt(o) if o.bridge_verified)
        }

        pub fn global_reloc(&mut self, f: &Frame, min_inliers: usize) -> Option<(Tf, i32, [f64; 36])> {
            self.ensure_cur(f);
            if self.cur.desc.is_empty() || self.kf_db.is_empty() {
                return None;
            }
            let words = |d: &[u8; 32]| -> [u64; 4] { std::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap())) };
            let step = (self.cur.desc.len() / 100).max(1);
            let q: Vec<[u64; 4]> = self.cur.desc.iter().step_by(step).map(words).collect();
            let from = self.kf_db.len().saturating_sub(40);
            let mut scores: Vec<(usize, usize)> = self
                .kf_db
                .iter()
                .enumerate()
                .skip(from)
                .map(|(i, (kf, _))| {
                    let kw: Vec<[u64; 4]> = kf.desc.iter().map(words).collect();
                    let hits = q
                        .iter()
                        .filter(|a| kw.iter().any(|b| (0..4).map(|j| (a[j] ^ b[j]).count_ones()).sum::<u32>() < 40))
                        .count();
                    (hits, i)
                })
                .collect();
            scores.sort_by(|a, b| b.cmp(a));
            let prof = std::env::var("KLT_PROF").is_ok();
            if prof {
                eprintln!("global db {} q {} top {:?}", self.kf_db.len(), q.len(), &scores[..scores.len().min(3)]);
            }
            let cam = f.cam.clone();
            for &(hits, i) in scores.iter().take(1) {
                if hits < 10 {
                    break;
                }
                let (kf, kp) = &self.kf_db[i];
                let info = self.reloc.compute_tf(kf, &self.cur, &cam, Tf::NULL);
                if prof {
                    eprintln!("global try {i} inliers {} null {}", info.inliers, info.tf.is_null());
                }
                if !info.tf.is_null() && info.inliers >= min_inliers {
                    return Some((kp.mul(&info.tf), info.inliers as i32, cov36(&info.covariance)));
                }
            }
            None
        }

        pub fn cur_features(&self) -> Features {
            self.cur.clone()
        }

        /// Odometry::reset(p)
        pub fn odom_reset(&mut self, p: Tf) {
            match &mut self.odom {
                Odom::F2f(o) => o.reset_tf(p),
                Odom::Klt(o) => o.reset_tf(p),
            }
            self.prev_stamp = 0.0;
            self.frames = 0;
        }

        pub fn odom_pose(&self) -> Tf {
            match &self.odom {
                Odom::F2f(o) => o.pose_tf(),
                Odom::Klt(o) => o.pose_tf(),
            }
        }

        /// 회색조는 이 뒤 쓰지 않아 옮김. mm 깊이는 스트림으로 보낼 원본이 따로 있으면 옮기고, 같은 버퍼면(1000/m) 스트림과 나눠 듦(Arc)
        pub fn make_ref(&self, f: &mut Frame) -> RefSig {
            let depth_mm = if f.depth_raw.is_empty() { f.depth_mm.clone() } else { std::mem::take(&mut f.depth_mm) }; // 같은 버퍼면 나눠 듦(복사 없음)
            let frame = Frame { id: f.id, t: f.t, gray: std::mem::take(&mut f.gray), color: Vec::new(), color_is_rgb: f.color_is_rgb, depth_mm, depth_raw: Default::default(), depth_per_m: f.depth_per_m, cam: f.cam.clone() };
            RefSig { frame, feats: None, ids: None }
        }

        /// reloc.computeTransformationMod(기준 Signature, Signature(data), 추측 없음).
        /// 원본은 기준 Signature 를 고쳐 둔다: 지금 장면에 특징이 없으면 단어 id 를 0, 1, … 로 다시 매기고(id 0 은 뒤의 kNN 에서 빠짐),
        /// 두 번째 판(추측 있음)까지 가면 단어 순서를 그 판의 순서로 바꾼다. 다음 시도는 그 상태에서 시작
        pub fn reloc(&mut self, r: &mut RefSig) -> Option<RelocOut> {
            if r.feats.is_none() {
                r.feats = Some(self.ext.extract(&r.frame));
            }
            let n = r.feats.as_ref().unwrap().kps.len();
            if self.cur.kps.is_empty() {
                // RegistrationVis "just create fake words": id = 순번
                if n > 0 {
                    r.ids = Some((0..n as i32).collect());
                }
                return None;
            }
            if r.ids.is_some() {
                self.ref_reuse += 1;
            }
            let info = self.reloc.compute_tf_ids(r.feats.as_ref().unwrap(), r.ids.as_deref(), &self.cur, &r.frame.cam, Tf::NULL);
            if info.to_kept.is_some() {
                match &info.from_order {
                    Some(ord) => {
                        let f = r.feats.as_ref().unwrap();
                        let ids: Vec<i32> = r.ids.clone().unwrap_or_else(|| (1..=n as i32).collect());
                        r.feats = Some(Features { kps: ord.iter().map(|&i| f.kps[i]).collect(), pts3d: ord.iter().map(|&i| f.pts3d[i]).collect(), desc: ord.iter().map(|&i| f.desc[i]).collect() });
                        r.ids = Some(ord.iter().map(|&i| ids[i]).collect());
                    }
                    None => self.ref_unknown += 1,
                }
            }
            if info.tf.is_null() {
                return None;
            }
            Some(RelocOut { t: info.tf, inliers: info.inliers as i32, cov: cov36(&info.covariance) })
        }

        /// Rtabmap::process(data, odomPose, covariance, velocity). node = false 면 중간 장면(특징 없음)
        pub fn slam_process(&mut self, f: &Frame, node: bool, pose: Option<Tf>, cov: &[f64; 36], velocity: Vec<f32>) {
            // 지도 스레드 + 광류 경로: 노드 특징(GFTT·BRIEF)도 지도 스레드에서 뽑음(회색조 사본·깊이 나눠 듦만 넘김)
            let raw = if node && self.cur.kps.is_empty() && matches!(self.odom, Odom::Klt(_)) && self.slam.is_thread() {
                Some(Box::new(RawNode { gray: f.gray.clone(), depth: f.depth_mm.clone(), cam: f.cam.clone(), pose, trusted: self.trusted }))
            } else {
                None
            };
            if node && raw.is_none() {
                self.ensure_cur(f);
                if let Some(p) = pose {
                    if !self.cur.desc.is_empty() {
                        self.kf_db.push((self.cur.clone(), p));
                        self.kf_ids.push((0, self.trusted, 0)); // id 는 process 뒤에 채움
                    }
                }
            }
            let features = if node {
                // 이 장면 특징은 노드로 넘기면 더 쓰지 않음: 설명자는 복사 없이 옮김
                NodeFeatures {
                    kps: self.cur.kps.iter().map(|k| Kp { x: k.x, y: k.y, size: k.size, angle: k.angle, response: k.response, octave: k.octave, class_id: -1 }).collect(),
                    pts3d: self.cur.pts3d.iter().map(|p| [p.x, p.y, p.z]).collect(),
                    desc: std::mem::take(&mut self.cur.desc),
                }
            } else {
                NodeFeatures::default()
            };
            let inp = NodeInput { features, stamp: f.t, pose: pose.unwrap_or(Tf::NULL), cov: *cov, velocity, image_size: (f.cam.width, f.cam.height), prov_from: self.prov_from };
            let had = node && self.kf_ids.last().map(|x| x.0 == 0).unwrap_or(false);
            let job = self.slam.jobs() + 1;
            if had {
                if let Some(l) = self.kf_ids.last_mut() {
                    l.2 = job;
                }
            }
            let forced = std::mem::take(&mut self.forced_hyp);
            let (ids, kfs) = self.slam.process(inp, forced, had, job, raw);
            for (feats, p, trusted, id) in kfs {
                if !feats.desc.is_empty() {
                    self.kf_db.push((feats, p));
                    self.kf_ids.push((id, trusted, 0));
                }
            }
            for (k, id) in ids {
                // 지도 노드 id 가 정해진 kf_db 칸을 채움(지도 스레드면 한 단계 늦게 옴)
                if let Some(e) = self.kf_ids.iter_mut().find(|e| e.0 == 0 && e.2 == k) {
                    e.0 = id;
                }
            }
        }

        /// 이어지지 않은 새 지도에서: 믿는 지도의 노드 가운데 이번 장면과 기술자로 맞는 노드를 찾아(PnP 확인)
        /// 다음 slam 단계의 루프 후보로 넘긴다. 실제 연결은 slam 이 원래 루프처럼 정합·최적화 검사를 거쳐 정한다
        /// 지금 지도의 마지막 노드 id(지도 스레드면 한 단계 늦음)
        pub fn last_node(&self) -> i32 {
            self.slam.st().last_id
        }

        pub fn propose_merge(&mut self, f: &Frame, min_inliers: usize) -> bool {
            self.ensure_cur(f);
            if self.cur.desc.is_empty() {
                return false;
            }
            let words = |d: &[u8; 32]| -> [u64; 4] { std::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap())) };
            let step = (self.cur.desc.len() / 60).max(1);
            let q: Vec<[u64; 4]> = self.cur.desc.iter().step_by(step).map(words).collect();
            let mut best: Vec<(usize, usize)> = Vec::new();
            // 믿는 지도의 노드 가운데 최근 40 개(새 지도 노드는 빼고 셈: 전에는 최근 40 칸이 모두 새 지도 노드라 후보가 0 이었음)
            let below = if self.prov_from > 0 { self.prov_from } else { i32::MAX }; // 이어 붙인 구간이면 그 앞 노드만
            let cand: Vec<usize> = (0..self.kf_db.len()).rev().filter(|&i| self.kf_ids[i].1 && self.kf_ids[i].0 > 0 && self.kf_ids[i].0 < below).take(40).collect();
            for &i in &cand {
                let kf = &self.kf_db[i].0;
                let id = self.kf_ids[i].0;
                // 짝 = 가장 가까운 것이 두 번째보다 충분히 가까움(비율 0.8) 이고 거리 < 50
                let kw: Vec<[u64; 4]> = kf.desc.iter().map(words).collect();
                let hits = q
                    .iter()
                    .filter(|a| {
                        let (mut d0, mut d1) = (u32::MAX, u32::MAX);
                        for b in &kw {
                            let d = (0..4).map(|j| (a[j] ^ b[j]).count_ones()).sum::<u32>();
                            if d < d0 {
                                d1 = d0;
                                d0 = d;
                            } else if d < d1 {
                                d1 = d;
                            }
                        }
                        d0 < 50 && (d0 as f32) < 0.8 * d1 as f32
                    })
                    .count();
                best.push((hits, i));
            }
            best.sort_by(|a, b| b.cmp(a));
            let cam = f.cam.clone();
            for &(hits, i) in best.iter().take(1) {
                if hits < 8 {
                    break;
                }
                let info = self.reloc.compute_tf(&self.kf_db[i].0, &self.cur, &cam, Tf::NULL);
                if !info.tf.is_null() && info.inliers >= min_inliers {
                    self.forced_hyp = self.kf_ids[i].0;
                    return true;
                }
            }
            false
        }

        /// 이번 단계 루프 보정 크기(m, rad)와 크기 때문에 버린 루프 수
        pub fn correction_info(&self) -> ((f32, f32), u32) {
            self.slam.st().corr
        }

        pub fn loop_closure_id(&self) -> i32 {
            self.slam.st().loop_id
        }
        pub fn local_optimized(&self) -> &BTreeMap<i32, Tf> {
            &self.slam.st().lp
        }
        pub fn last_location_id(&self) -> i32 {
            self.slam.st().last_id
        }
        /// None = null Transform
        pub fn map_correction(&self) -> Option<Tf> {
            let c = self.slam.st().map_corr;
            (!c.is_null()).then_some(c)
        }
        pub fn trigger_new_map(&mut self) -> i32 {
            self.slam.trigger_new_map()
        }
        pub fn reset(&mut self) {
            self.kf_db.clear();
            self.kf_ids.clear();
            self.slam.reset();
        }

        /// 지도 갱신(노드 넣기·루프 클로저·그래프 최적화)을 따로 도는 스레드로: 위치 추적 스레드는 기다리지 않고
        /// 지난 단계 결과(지도 보정·최적화 위치)를 쓴다(한 노드 늦음). 실시간 입력에서만 켠다(재생 결과는 그대로 결정적)
        pub fn start_map_thread(&mut self) {
            self.slam.start_thread(self.ext.clone());
            if let Odom::Klt(o) = &mut self.odom {
                o.set_async_desc();
            }
        }

        /// Rtabmap::getGraph(optimized=true, global=true): (id, 시각, 위치). 노드가 없으면 빔
        pub fn final_graph(&mut self) -> Vec<(i32, f64, Tf)> {
            let slam = self.slam.inline();
            let (poses, _) = slam.get_graph(true, true);
            poses.iter().filter_map(|(id, t)| slam.mem.sig(*id).map(|s| (*id, s.stamp, *t))).collect()
        }
    }
}

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
        "ondevice slam (--tum <폴더> [--rate R] [--stride N] | --realsense [--serial S] [--size 640x480] [--color-size 1280x720] [--fps 30] [--camera-profile librealsense|tracking] [--exposure ms | --ae-cap ms] [--gain 0..128])\n            [--config <ini>] [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop]\n            [--poses <파일>] [--final-poses <파일>] [--stats <파일>] [--max-frames N] [--reset-file <파일>] [--reloc-wait 초] [--no-bridge] [--no-provisional] [--reset-at N] [--Group/Name 값 ...]"
    );
}

/// 위치 프로그램 설정(명령줄을 `parse` 로 한 번 읽은 것)
#[derive(Clone)]
pub struct Args {
    tum: String,
    config: String,
    out: String,
    poses: String,
    final_poses: String,
    stats: String,
    align: String,
    serial: String,
    reset_file: String,
    realsense: bool,
    driver: String,
    reloc_wait: f64,
    /// 놓친 뒤 다시 시작할 때 새 지도를 열지 않고 마지막 위치에서 같은 지도로 이어 붙임(광류 경로, --no-bridge 로 끔)
    bridge: bool,
    /// 이어지지 않은 새 지도의 장면을 맡아 두지 않고 임시 위치로 바로 보냄(--no-provisional 로 끔)
    provisional: bool,
    /// 이 시간(초) 이하의 빈틈만 그 자리에서 이어 붙임(그보다 길면 새 지도, 깊이 ICP 로 확인되면 예외)
    bridge_gap: f64,
    /// 지도 갱신을 위치 추적 스레드에서 바로(시험·비교용)
    map_inline: bool,
    no_boost: bool,
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
    max_frames: i64,
    overrides: Vec<(String, String)>,
}

/// 명령줄(0 번은 프로그램 이름) → 설정. 도움말·잘못된 인자면 끝 코드
pub fn parse(argv: &[String]) -> Result<Args, i32> {
    let mut a = Args {
        tum: String::new(),
        config: String::new(),
        out: "-".into(),
        poses: String::new(),
        final_poses: String::new(),
        stats: String::new(),
        align: "rs2".into(),
        serial: String::new(),
        reset_file: String::new(),
        realsense: false,
        driver: String::new(),
        reset_at: -1,
        stride: 1,
        reloc_wait: -1.0, // 놓친 뒤 이 시간(초) 동안은 같은 지도에서 다시 잡기만 시도. 음수 = 광류 경로 0(바로 다시 시작), F2F 1 초
        bridge: true,
        provisional: true,
        bridge_gap: 5.0, // 빈틈 뒤 먼저 지도 전체에서 제자리를 찾고, 못 찾으면 5 초 이하 빈틈은 이어 붙임(한 지도 유지, 거짓 루프는 보정 크기 검사가 막음)
        map_inline: false,
        no_boost: false,
        block: -1,       // -1: 입력에 따라(재생 = 기다림, 카메라 = 버림)
        rate: 0.0,
        exposure_ms: 0.0,
        camera_profile: "librealsense".into(), // 컬러 제어: librealsense 와 같음(기본) | tracking(60 Hz · 노출 상한 16.6 ms)
        ae_cap_ms: 0.0, // 노출 상한(ms). 0 = 카메라 자동 노출. 밝은 방에서 번짐을 줄이려면 8.3(60 Hz 조명 한 주기)
        gain: -1,
        send_every: 1,
        queue: 4,
        color: (0, 0),
        w: 640,
        h: 480,
        fps: 30,
        max_frames: -1,
        overrides: vec![],
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
            "--tum" => a.tum = val()?,
            "--realsense" => a.realsense = true,
            "--driver" => a.driver = val()?,
            "--rate" => a.rate = num(&val()?)?,
            "--stride" => a.stride = num(&val()?)? as usize,
            "--align" => a.align = val()?,
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
            "--config" => a.config = val()?,
            "--out" => a.out = val()?,
            "--send-every" => a.send_every = (num(&val()?)? as i64).max(1),
            "--queue" => a.queue = (num(&val()?)? as i64).max(1) as usize,
            "--block" => a.block = 1,
            "--drop" => a.block = 0,
            "--poses" => a.poses = val()?,
            "--final-poses" => a.final_poses = val()?,
            "--stats" => a.stats = val()?,
            "--db" => {
                let _ = val()?;
                common::info!("--db: 이 판에는 데이터베이스 저장이 없어 넘어감");
            }
            "--max-frames" => a.max_frames = num(&val()?)? as i64,
            "--reset-file" => a.reset_file = val()?,
            "--reloc-wait" => a.reloc_wait = num(&val()?)?,
            "--no-bridge" => a.bridge = false,
            "--no-provisional" => a.provisional = false,
            "--bridge-gap" => a.bridge_gap = num(&val()?)?,
            "--map-inline" => a.map_inline = true,
            "--no-boost" => a.no_boost = true,
            "--reset-at" => a.reset_at = num(&val()?)? as i64, // 시험용: 이 장면 번호에서 리셋 파일이 생긴 것처럼
            // 프로세스 설정(ondevice 가 처음에 읽음): 실시간 배치·메모리 묶기·조용히
            "--rt" => {
                let _ = val()?;
            }
            "--mlock" | "--quiet" => {}
            "-h" | "--help" => {
                usage();
                return Err(0);
            }
            _ if k.starts_with("--") && k.contains('/') => {
                let name = k[2..].to_string();
                a.overrides.push((name, val()?));
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

/// 기준 장면: 믿는 위치의 장면(다시 잡기용)
struct Ref {
    sig: engine::RefSig,
    pose: Tf,
    t: f64,
}

/// 같은 프로세스에서 돌리기: 읽어 둔 설정으로 위치 루프를 스레드에서 시작하고, 스트림 대신 메모리 큐의 받는 쪽을 돌려준다.
/// 큐 길이·기다림/버림은 명령줄(--queue, --block|--drop) 그대로. 받는 쪽은 파이프 스트림과 같은 차례·같은 내용의 장면·그래프·리셋 블록을 받는다
/// pool: 위치 루프와 특징점 추출(읽기 스레드)의 나눠 돌리기가 모두 이 풀에서 돈다(부른 쪽이 코어·우선순위를 정해 만든 풀)
pub fn spawn(a: Args, pool: rayon::ThreadPool) -> Result<(stream::Receiver, std::thread::JoinHandle<i32>), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let th = std::thread::Builder::new()
        .name("slam".into())
        .spawn(move || {
            let pool = std::sync::Arc::new(pool);
            let p2 = pool.clone();
            match pool.install(|| run_in(a, Some(tx), Some(p2))) {
                Ok(c) => c,
                Err(e) => {
                    common::info!("slam 오류: {e}");
                    1
                }
            }
        })
        .map_err(|e| e.to_string())?;
    match rx.recv() {
        Ok(r) => Ok((r, th)),
        Err(_) => Err(format!("slam 이 시작하지 못함(끝 코드 {:?})", th.join())),
    }
}

/// 명령줄 그대로 실행(`ondevice slam …`): 스트림을 파일·표준출력으로 쓴다
pub fn run(argv: &[String]) -> Result<i32, String> {
    match parse(argv) {
        Ok(a) => run_in(a, None, None),
        Err(c) => Ok(c),
    }
}

fn run_in(a: Args, to_mem: Option<std::sync::mpsc::Sender<stream::Receiver>>, pool: Option<std::sync::Arc<rayon::ThreadPool>>) -> Result<i32, String> {
    let in_proc = to_mem.is_some();
    // 실시간이면 위치 추적 스레드를 분할·memmap·뷰어보다 먼저(--no-boost 로 끔)
    if (a.realsense || a.rate > 0.0) && !a.no_boost {
        common::info!("위치 추적 스레드 우선: {}", common::threads::boost_tracking());
    }
    // 스트림이 표준출력이면 그 파일 기술자를 따로 떼어 두고 표준출력은 표준오류로 돌린다
    #[cfg(unix)]
    let stream_fd = (!in_proc && a.out == "-").then(sys::take_stdout);
    #[cfg(not(unix))]
    let stream_fd: Option<std::fs::File> = None;

    #[allow(unused_assignments)]
    let mut skipped = None;
    let src: Box<dyn FrameSource> = if !a.tum.is_empty() {
        let ts = TumSource::with_stride(&a.tum, a.rate, a.stride)?;
        skipped = Some(ts.skipped_counter());
        Box::new(ts)
    } else {
        open_camera(&a)?
    };
    let replay_max = !a.tum.is_empty() && a.rate == 0.0;
    let block = if a.block < 0 { replay_max } else { a.block == 1 };
    // 장면 읽기는 따로 도는 스레드에서(최대 속도 재생만 장면을 버리지 않음)
    // 파라미터: 기본값 ← ini([Core]) ← 명령줄
    let mut params = if a.config.is_empty() { Params::default() } else { Params::read_ini(&a.config)? };
    for (k, v) in &a.overrides {
        params.set(k, v);
    }
    // 위치 추적 특징 추출(GFTT/BRIEF·3D 점)도 읽기 스레드에서: 난수를 안 쓰고 장면만 보므로 어디서 뽑아도 결과가 같다
    let ext = crate::features::Extractor::new(&params);
    let klt = params.get("Odom/Strategy", 0i32) == 9;
    let prep: Option<crate::io::Prep> = if klt {
        None // 광류 경로: 장면마다 특징을 뽑지 않음
    } else {
        Some(Box::new(move |f: &crate::Frame| match &pool {
            Some(p) => p.install(|| ext.extract(f)),
            None => ext.extract(f),
        }))
    };
    let mut src = PrefetchSource::with_prep(src, !replay_max, prep);
    params.0.remove("Rtabmap/PublishRAMUsage");
    let intermediate = params.get_bool("Rtabmap/CreateIntermediateNodes", false);
    let detection_rate: f32 = params.get("Rtabmap/DetectionRate", 1.0);

    common::info!("입력: {}", src.describe());
    let mut line = format!("slam (RTAB-Map 0.24.2 경로), 설정 {}, 바꾼 파라미터 {} 개:", if a.config.is_empty() { "(기본값)" } else { &a.config }, params.0.len());
    for (k, v) in &params.0 {
        line += &format!(" {k}={v}");
    }
    common::info!("{line}");
    common::info!("출력: {} ({}, 큐 {}, {} 장마다)", if in_proc { "물체 기억 메모리 큐" } else { &a.out }, if block { "받는 쪽을 기다림" } else { "밀리면 오래된 장면 버림" }, a.queue, a.send_every);

    // 놓쳤을 때 다시 잡기는 아래에서 직접 한다: 위치 추적이 혼자 '마지막 위치'에서 다시 시작하지 않게 함
    let mut odom_params = params.clone();
    odom_params.set("Odom/ResetCountdown", "0");
    // 다시 잡기: 놓치기 전의 장면(기준 장면)과 지금 장면의 특징점을 직접 맞춰 위치를 구함
    let reloc_min = params.get("Vis/MinInliers", 20i32).max(25);
    let mut eng = engine::Engine::new(&params, &odom_params, reloc_min);
    let klt_path = params.get("Odom/Strategy", 0i32) == 9;
    // 실시간(카메라·찍은 속도 재생)이면 지도 갱신을 따로 도는 스레드로(--map-inline 으로 끔). 최대 속도 재생은 결정적이게 그대로
    if (a.realsense || a.rate > 0.0) && !a.map_inline {
        eng.start_map_thread();
    }
    let reloc_wait = if a.reloc_wait >= 0.0 { a.reloc_wait } else if klt_path { 0.0 } else { 1.0 };
    let bridge = a.bridge && klt_path;
    let mut n_bridge = 0i64;
    let mut n_global = 0i64;
    let mut global_tried = false;
    let mut n_loop_rejected = 0i64;
    let mut n_merge_try = 0i64;
    let mut bridge_t = f64::NEG_INFINITY; // 확인 안 된 이어 붙이기 시각
    let mut n_bridge_fix = 0i64;
    let mut last_merge_try = f64::MIN;
    let mut max_corr = (0f32, 0f32);
    let mut last_good_t = f64::MAX; // 마지막으로 위치를 낸 장면 시각
    let mut n_bridge_ok = 0i64;

    let mut refs: VecDeque<Ref> = VecDeque::new();
    let mut ref_next = 0usize;
    let mut lost_since = -1.0f64;
    let (mut good_t0, mut good_t1) = (-1.0f64, -1.0f64); // 믿는 지도의 첫·마지막 장면 시각
    let mut restart = false;
    let mut n_reloc = 0i64;
    let mut s_reloc = Summary::default();

    let mut writer = match to_mem {
        Some(tx) => {
            let (w, r) = Writer::channel(a.queue, block);
            tx.send(r).map_err(|_| "받는 쪽이 없음".to_string())?;
            w
        }
        None => Writer::new(&a.out, stream_fd, a.queue, block)?,
    };
    let mut poses_f = (!a.poses.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.poses).expect("--poses")));
    let mut stats_f = (!a.stats.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.stats).expect("--stats")));
    if let Some(f) = poses_f.as_mut() {
        let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 보낸 그대로)");
    }
    if let Some(f) = stats_f.as_mut() {
        let _ = writeln!(f, "seq,t,read_ms,odom_ms,slam_ms,send_ms,lost,new_map,map_corrected,loop,features,inliers,held,no_pose");
    }
    #[cfg(unix)]
    if !in_proc {
        sys::on_stop_signals();
    }

    // 카메라 광학 축 → 몸체 축
    let optical = rtf::OPTICAL;
    let (mut s_read, mut s_odom, mut s_slam, mut s_send, mut s_total) = (Summary::default(), Summary::default(), Summary::default(), Summary::default(), Summary::default());
    let (mut n, mut n_lost, mut n_newmap, mut n_loop, mut n_corr, mut n_graph, mut n_graph_nodes) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
    let mut sent_nodes: BTreeMap<i32, Tf> = BTreeMap::new(); // memmap 에 보낸 노드 위치(지도 몸체 축)
    // 믿는 지도와 아직 안 이어진 새 지도: 노드를 "맡아 둠" 표시로 보내고 장면도 위치·기준 노드와 함께 맡겨 둔다
    let mut held_nodes: BTreeMap<i32, i64> = BTreeMap::new(); // 맡아 둔 노드 id → 그 노드를 기준으로 맡긴 장면 수
    let mut held_last = -1i32; // 지금 새 지도의 마지막 노드(없으면 -1)
    let (mut n_held, mut n_joined) = (0i64, 0i64);
    let mut map_id = 0u32;
    let mut fresh = true; // 처음이거나 리셋 직후의 장면: 위치 추적이 새로 시작하는 것이 정상
    // 다시 시작한 뒤의 위치는 "마지막 위치에서 이어 붙인" 값이라, 믿던 지도의 노드(good_id)와 이어질 때까지 놓침으로 보낸다
    let mut anchored = true;
    let mut good_id = -1i32;
    let mut prev_stamp = 0.0f64;
    let mut prev_corr = Tf::IDENTITY;
    // 임시 위치(--no-provisional 로 끔): 이어지지 않은 새 지도의 장면·노드를 맡아 두지 않고, 마지막으로 보낸 위치에서 이어 붙인
    // 표시 변환(prov)으로 바로 보낸다(화면이 멈추지 않음). slam 그래프에는 아무 제약도 넣지 않는다(표시 전용).
    // 루프 클로저로 이어지면 노드가 참 위치로 다시 가고(움직임 → memmap 다시 쌓기) 그 장면들도 고친 위치로 다시 놓인다.
    let provisional = a.provisional;
    let mut prov = Tf::IDENTITY;
    let mut prov_nodes: std::collections::BTreeSet<i32> = Default::default();
    let mut last_sent_world: Option<Tf> = None; // 마지막으로 보낸 장면의 몸체 위치(표시 좌표)
    let mut n_prov = 0i64;
    let t_run = Instant::now();
    while !STOP.load(Ordering::Relaxed) && (a.max_frames < 0 || n < a.max_frames) {
        let c0 = Instant::now();
        let Some(mut fr) = src.next()? else { break };
        fr.id = n + 1;
        // 리셋 요청(뷰어의 Reset 버튼: 한 프로그램이면 common::control 신호, 따로 띄운 뷰어면 --reset-file 파일): 위치 추적과 지도를 비우고 받는 쪽에 리셋 블록을 보낸다
        // 지도가 아직 작을 때(3 초 몫 미만) 놓쳐 다시 못 잡은 경우에도 같은 식으로 처음부터 다시 한다
        let reset_req = n == a.reset_at || common::control::take_reset() || (!a.reset_file.is_empty() && std::path::Path::new(&a.reset_file).exists());
        if reset_req {
            let _ = std::fs::remove_file(&a.reset_file);
        }
        if reset_req || restart {
            common::info!("장면 {n}: {} → 위치 추적·지도를 처음부터", if reset_req { "리셋" } else { "지도가 작을 때 놓침" });
            restart = false;
            good_t0 = -1.0;
            good_t1 = -1.0;
            eng.odom_reset(Tf::IDENTITY);
            eng.reset();
            sent_nodes.clear();
            held_nodes.clear();
            held_last = -1;
            map_id = 0;
            prev_stamp = 0.0;
            prev_corr = Tf::IDENTITY;
            anchored = true;
            good_id = -1;
            refs.clear();
            lost_since = -1.0;
            fresh = true;
            if !writer.push(OutFrame { reset: true, ..Default::default() }) {
                break;
            }
        }
        let read_ms = src.last_read_ms();

        let c1 = Instant::now();
        let mut info = eng.odom_process(&fr, src.take_prepped());
        let mut pose = info.pose;
        let mut relocalized = false;
        if pose.is_none() || info.lost {
            pose = None;
            if lost_since < 0.0 {
                lost_since = fr.t;
            }
            let cr = Instant::now();
            eng.ensure_cur(&fr);
            // 기준 장면을 최근 것부터 한 장면에 둘씩 돌아가며 맞춰 봄
            let mut k = 0;
            // 광류 경로는 위치 추적 안에서 키프레임 기술자 맞추기를 이미 함: 최근 기준 장면 맞추기(장면당 수십 ms)는 F2F 에서만
            while anchored && !klt_path && k < 2 && k < refs.len() {
                k += 1;
                let idx = refs.len() - 1 - (ref_next % refs.len());
                ref_next += 1;
                let r = &mut refs[idx];
                let Some(ri) = eng.reloc(&mut r.sig) else { continue };
                let p = r.pose.mul(&ri.t);
                let rt = r.t;
                eng.odom_reset(p);
                let cf = eng.cur_features();
                let info2 = eng.odom_process(&fr, Some(cf));
                if info2.pose.is_none() {
                    continue;
                }
                pose = info2.pose;
                info = info2;
                info.cov = ri.cov;
                info.inliers = ri.inliers;
                relocalized = true;
                n_reloc += 1;
                common::info!("장면 {n}: 같은 지도에서 다시 잡음({:.2} 초 전 장면, 맞은 점 {})", fr.t - rt, ri.inliers);
                break;
            }
            // 최근 기준 장면으로 못 잡으면 지도 전체에서(광류 경로, 놓친 동안 장면마다)
            // 장면당 예산: 이 장면에서 위치 추적이 이미 20 ms 넘게 썼으면 다음 장면으로 미룸(추적 스레드가 40 ms 넘게 막히지 않게)
            // 빈틈 바로 뒤 첫 장면은 예산과 상관없이 한 번 꼭 해 봄(이어 붙이기 전에 제자리 찾기)
            if !relocalized && klt_path && fr.t - last_good_t > 0.5 && (ms_since(c1) < 20.0 || !global_tried) {
                global_tried = true;
                if let Some((p, inl, cov)) = eng.global_reloc(&fr, reloc_min as usize) {
                    eng.odom_reset(p);
                    let info2 = eng.odom_process(&fr, None);
                    if info2.pose.is_some() {
                        pose = info2.pose;
                        info = info2;
                        info.cov = cov;
                        info.inliers = inl;
                        relocalized = true;
                        n_reloc += 1;
                        n_global += 1;
                        common::info!("장면 {n}: 지도 전체에서 다시 잡음(맞은 점 {inl})");
                    }
                }
            }
            s_reloc.add(ms_since(cr));
            // 기다려도 못 잡으면 마지막 위치에서 다시 시작 → 새 지도(이전 지도와 이어질 때까지 보내지 않음)
            if !relocalized && (!anchored || fr.t - lost_since > reloc_wait) {
                // 믿는 지도가 3 초 몫도 안 되면 붙들 것이 없다: 처음부터 다시
                if anchored && good_t1 - good_t0 < 3.0 {
                    restart = true;
                } else {
                    let p = eng.odom_pose();
                    eng.odom_reset(p);
                }
                lost_since = -1.0;
            }
        }
        let gap_since_good = fr.t - last_good_t; // 마지막으로 위치를 낸 뒤 흐른 시간(녹화·USB 빈틈 포함)
        if pose.is_some() {
            lost_since = -1.0;
            last_good_t = fr.t;
            global_tried = false;
        }
        // 위치 추적 자체가 위치를 못 낸 장면(지도 잇기 여부와 별개)
        let no_pose = pose.is_none();
        let odom_ms = ms_since(c1);

        // 새 지도는 위치 추적이 실제로 다시 시작한 장면(위치가 채워져 옴)에서만 연다
        let mut new_map = false;
        // 이어 붙이기는 그사이 움직임을 아는 경우만: 깊이 ICP 로 자리를 확인했거나 빈틈이 1 초 이하.
        // 그보다 길면(사람이 그사이 걸었을 수 있음) 새 지도를 열고 루프 클로저가 옛 지도와 잇게 둔다(틀린 이어 붙이기를 거대한 보정으로 끌어오지 않게)
        if bridge && !fresh && !relocalized && pose.is_some() && info.cov[0] >= 9999.0 && (eng.bridge_verified() || gap_since_good <= a.bridge_gap) {
            // 이어 붙이기: 위치 추적은 마지막 위치에서 다시 시작했다. 같은 지도에 큰 불확실성(0.5 m², 0.5 rad²)으로 이어
            // 지도가 끊기지 않게 하고, 실제로 어긋났으면 뒤의 루프 클로저가 그래프 최적화로 바로잡는다
            info.cov = [0.0; 36];
            for i in 0..6 {
                info.cov[i * 7] = 0.5;
            }
            n_bridge += 1;
            if !eng.bridge_verified() && gap_since_good > 0.5 {
                // 확인 안 된 이어 붙이기: 다음 10 초 동안 옛 노드에서 제자리를 찾아 루프로 이 구간을 바로잡음
                eng.prov_from = eng.last_node() + 1;
                bridge_t = fr.t;
            }
            if eng.bridge_verified() {
                // 깊이 ICP 로 그사이 움직임을 잡았다: 보통 위치 추적처럼 작은 불확실성
                for i in 0..6 {
                    info.cov[i * 7] = 1e-3;
                }
                n_bridge_ok += 1;
            }
        }
        if !fresh && !relocalized && pose.is_some() && info.cov[0] >= 9999.0 {
            common::info!("장면 {n}: 위치 추적 다시 시작 → 새 지도");
            let id = eng.trigger_new_map();
            map_id = if id >= 0 { id as u32 } else { map_id + 1 };
            new_map = true;
            held_last = -1;
            if good_id > 0 {
                anchored = false;
            }
            n_newmap += 1;
        }
        fresh = false;
        let mut process = true;
        // 확인 안 된 이어 붙이기 뒤 10 초: 장면마다(0.2 초 간격, 장면 예산 안) 옛 노드에서 제자리를 찾고, 찾으면 이 장면을 바로 지도 노드로 넣어
        // 그 옛 노드와 루프를 건다(그래프 최적화가 이어 붙인 구간 전체를 끌어옴)
        let fixing = anchored && eng.prov_from > 0 && fr.t - bridge_t < 10.0;
        if anchored && eng.prov_from > 0 && !fixing {
            eng.prov_from = 0; // 10 초 동안 못 찾음: 이어 붙인 그대로 둠
        }
        let mut force_node = false;
        if fixing && klt_path && pose.is_some() && fr.t - last_merge_try >= 0.2 && ms_since(c1) < 25.0 {
            last_merge_try = fr.t;
            if eng.propose_merge(&fr, reloc_min as usize) {
                n_merge_try += 1;
                force_node = true;
            }
        }
        let det_rate = if anchored || klt_path { detection_rate } else { detection_rate.max(4.0) }; // 이어지기를 기다리는 동안은 더 자주 찾아봄
        if !force_node && det_rate > 0.0 && prev_stamp > 0.0 && fr.t > prev_stamp && fr.t - prev_stamp < 1.0 / det_rate as f64 {
            process = false;
        }
        if process {
            prev_stamp = fr.t;
        }
        let mut data_id = (n + 1) as i32;
        if !process {
            data_id = -1; // 중간 장면(지도 노드 아님)
            process = intermediate;
        }
        let c2 = Instant::now();
        let mut slam_ms = -1.0;
        let mut lp = BTreeMap::new();
        let mut loop_ = false;
        if process {
            eng.trusted = anchored;
            // 이어지지 않은 새 지도: 노드를 넣을 때마다 믿는 지도 노드 전체에서 이어 줄 후보를 찾아 루프 후보로 넘김
            // 잇기 후보 찾기는 1.5 초에 한 번, 이 장면 위치 추적이 가벼웠을 때만
            if !anchored && klt_path && data_id > 0 && pose.is_some() && fr.t - last_merge_try >= 1.5 && ms_since(c1) < 30.0 && { last_merge_try = fr.t; eng.propose_merge(&fr, reloc_min as usize) } {
                n_merge_try += 1;
            }
            eng.slam_process(&fr, data_id > 0, pose, &info.cov, info.velocity());
            slam_ms = ms_since(c2);
            loop_ = eng.loop_closure_id() > 0;
            n_loop += loop_ as i64;
            if loop_ && eng.prov_from > 0 && anchored {
                n_bridge_fix += 1;
                common::info!("장면 {n}: 이어 붙인 구간을 옛 지도 루프로 바로잡음");
                eng.prov_from = 0;
            }
            let (mc, rj) = eng.correction_info();
            n_loop_rejected += rj as i64;
            if loop_ || rj > 0 {
                max_corr.0 = max_corr.0.max(mc.0);
                max_corr.1 = max_corr.1.max(mc.1);
                common::info!("장면 {n}: 루프 클로저 {} (기존 노드 최대 보정 {:.2} m · {:.1}°)", if rj > 0 { "버림" } else { "받음" }, mc.0, mc.1.to_degrees());
            }
            lp = eng.local_optimized().clone();
            let last = eng.last_location_id();
            if !anchored && lp.contains_key(&good_id) {
                anchored = true;
                common::info!("장면 {n}: 이전 지도와 다시 이어짐");
            }
            if anchored && lp.contains_key(&last) {
                good_id = last;
            }
            if !anchored && data_id > 0 && lp.contains_key(&last) {
                held_last = last;
            }
        }
        // 기준 장면: 믿는 위치의 장면을 0.4 초마다 하나씩, 최근 6 개
        if anchored && pose.is_some() && !new_map {
            if good_t0 < 0.0 {
                good_t0 = fr.t;
            }
            good_t1 = fr.t;
        }
        if anchored && !new_map {
            if let Some(p) = pose {
                if refs.back().is_none_or(|r| fr.t - r.t >= 0.4) {
                    refs.push_back(Ref { sig: eng.make_ref(&mut fr), pose: p, t: fr.t });
                    if refs.len() > 6 {
                        refs.pop_front();
                    }
                }
            }
        }
        // 이번 장면의 지도 위치 = 지도 보정 × 오도메트리 위치 × 광학 축
        let corr = eng.map_correction();
        let mut corrected = false;
        if let Some(c) = &corr {
            corrected = (0..12).any(|i| (c.0[i] - prev_corr.0[i]).abs() > 1e-6);
            if corrected {
                prev_corr = *c;
                n_corr += 1;
            }
        }
        // 놓친 장면: 위치를 못 구함. 다시 시작한 새 지도의 장면도 믿던 지도와 이어지기 전에는 놓침으로 표시
        // 맡아 둠: 이어지지 않은 새 지도(임시 위치를 쓰면 맡아 두지 않음)
        let hold = !anchored && !provisional;
        if !anchored && provisional && new_map {
            // 새 지도 첫 장면 = 마지막으로 보낸 위치(그사이 움직임 모름): 표시 변환을 그에 맞춤
            if let (Some(w), Some(p)) = (last_sent_world, pose.as_ref()) {
                prov = w.mul(&corr.unwrap_or(Tf::IDENTITY).mul(p).inverse());
            }
        }
        if anchored {
            prov = Tf::IDENTITY;
        }
        let lost = pose.is_none() || info.lost || (new_map && !provisional) || hold;
        n_lost += lost as i64;

        // 그래프 블록: 지도 갱신 뒤 새로 생기거나 움직인(행렬 원소 1e-3 넘게) 노드를 이 장면보다 먼저 보낸다
        // 이어지지 않은 동안의 노드는 맡아 둠 표시로, 이어진 뒤에는 믿는 노드로(표시가 바뀐 노드는 안 움직였어도 다시 보냄)
        if process {
            let mut upd = vec![];
            for (&id, p) in &lp {
                if id <= 0 {
                    continue;
                }
                if p.is_null() {
                    continue;
                }
                let pm = prov.mul(p);
                let was_held = held_nodes.contains_key(&id);
                let sent = sent_nodes.get(&id);
                if !anchored && sent.is_some() && !was_held && !prov_nodes.contains(&id) {
                    continue; // 믿는 노드는 이어지지 않은 지도의 값으로 덮지 않음
                }
                if !anchored && provisional {
                    prov_nodes.insert(id);
                } else if anchored {
                    prov_nodes.remove(&id);
                }
                if let Some(s) = sent {
                    if was_held == hold {
                        let dmax = (0..12).map(|i| (s.0[i] - pm.0[i]).abs()).fold(0f32, f32::max);
                        if dmax < 1e-3 {
                            continue;
                        }
                    }
                }
                sent_nodes.insert(id, pm);
                if hold && !was_held {
                    held_nodes.insert(id, 0);
                }
                if anchored && was_held {
                    n_joined += held_nodes.remove(&id).unwrap();
                }
                let tn = pm.mul(&optical);
                upd.push(GraphNode { id: id as u32, held: hold as u32, t: tn.0.map(|v| v as f64) });
            }
            if !upd.is_empty() {
                n_graph_nodes += upd.len() as i64;
                n_graph += 1;
                if !writer.push(OutFrame { graph: upd, ..Default::default() }) {
                    common::info!("받는 쪽이 닫힘: 끝냄");
                    break;
                }
            }
        }

        // 맡길 장면: 새 지도에서 위치를 구했고 그 지도의 노드를 이미 보냈을 때
        let held = hold && pose.is_some() && !info.lost && !new_map && held_nodes.contains_key(&held_last);
        let mut of = OutFrame {
            flags: if lost { stream::LOST } else { 0 } | if new_map { stream::NEW_MAP } else { 0 } | if corrected { stream::MAP_CORRECTED } else { 0 } | if held { stream::HELD } else { 0 },
            map_id,
            seq: n as u64,
            t: fr.t,
            ..Default::default()
        };
        if !lost || held {
            let body = prov.mul(&corr.unwrap_or(Tf::IDENTITY).mul(pose.as_ref().unwrap()));
            if !lost {
                last_sent_world = Some(body);
                if !anchored {
                    of.flags |= stream::PROVISIONAL;
                    n_prov += 1;
                }
            }
            let twc = body.mul(&optical);
            of.pose = twc.0.map(|v| v as f64);
            if !lost {
                // 위치 파일엔 믿는 위치만(임시 위치는 화면·지도용)
                if let (Some(f), true) = (poses_f.as_mut(), anchored) {
                    write_tum(f, fr.t, &twc);
                }
                // 기준 노드: 마지막 지도 노드(이 장면 위치 = 그 노드 위치 × 오도메트리 상대 위치)
                let last_id = eng.last_location_id();
                of.anchor = if sent_nodes.contains_key(&last_id) { last_id as u32 } else { 0 };
            } else {
                of.anchor = held_last as u32;
                if n % a.send_every == 0 {
                    *held_nodes.get_mut(&held_last).unwrap() += 1;
                }
                n_held += 1;
            }
        }
        of.fx = fr.cam.fx;
        of.fy = fr.cam.fy;
        of.cx = fr.cam.cx;
        of.cy = fr.cam.cy;
        of.depth_per_m = fr.depth_per_m;
        let mut send_ms = 0.0;
        if n % a.send_every == 0 {
            let c3 = Instant::now();
            of.width = fr.cam.width;
            of.height = fr.cam.height;
            // 장면 버퍼를 복사 없이 넘김(기준 장면은 make_ref 가 필요한 것을 이미 가져감)
            of.color = std::mem::take(&mut fr.color);
            of.color_is_rgb = fr.color_is_rgb;
            of.depth = fr.take_stream_depth();
            if !writer.push(of) {
                common::info!("받는 쪽이 닫힘: 끝냄");
                break;
            }
            send_ms = ms_since(c3); // 큐에 넣기(기다림 포함)
        }
        let total_ms = ms_since(c0);
        s_read.add(read_ms);
        if n > 0 {
            s_odom.add(odom_ms); // 첫 장면(초기화)은 뺌
        }
        if slam_ms >= 0.0 && n > 0 {
            s_slam.add(slam_ms);
        }
        s_send.add(send_ms);
        s_total.add(total_ms);
        if let Some(f) = stats_f.as_mut() {
            let _ = writeln!(
                f,
                "{n},{:.6},{read_ms:.3},{odom_ms:.3},{slam_ms:.3},{send_ms:.3},{},{},{},{},{},{},{},{}",
                fr.t, lost as i32, new_map as i32, corrected as i32, loop_ as i32, info.features, info.inliers, held as i32, no_pose as i32
            );
        }
        n += 1;
    }
    let wall = t_run.elapsed().as_secs_f64();
    writer.close();
    let mut s_write = Summary::default();
    for x in writer.write_ms() {
        s_write.add(x);
    }

    if !a.final_poses.is_empty() {
        let g = eng.final_graph();
        let mut f = std::io::BufWriter::new(std::fs::File::create(&a.final_poses).map_err(|e| format!("{}: {e}", a.final_poses))?);
        let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 끝에 최적화된 지도 노드)");
        for (_, t, p) in &g {
                write_tum(&mut f, *t, &p.mul(&optical));
        }
        common::info!("최적화된 지도 노드 {} 개 → {}", g.len(), a.final_poses);
    }
    drop(poses_f);
    drop(stats_f);

    if bridge {
        common::info!("놓친 뒤 같은 지도로 이어 붙임 {n_bridge} 번(깊이 ICP 로 자리 확인 {n_bridge_ok} 번, 루프로 바로잡음 {n_bridge_fix} 번), 지도 전체에서 다시 잡음 {n_global} 번");
    }
    {
        common::info!("루프 클로저 받음 {n_loop} · 보정이 커서 버림 {n_loop_rejected}, 살펴본 보정 최대 {:.2} m · {:.1}°", max_corr.0, max_corr.1.to_degrees());
    }
    common::info!("장면 {n} 개, {wall:.1} 초({:.1} 장/초), 놓침 {n_lost}, 새 지도 {n_newmap}, 루프 클로저 {n_loop}, 지도 보정 바뀜 {n_corr}", n as f64 / wall.max(1e-9));
    if provisional {
        common::info!("이어지지 않은 새 지도에서 임시 위치로 바로 보낸 장면 {n_prov}, 지도 전체에서 찾은 잇기 후보 {n_merge_try} 번");
    }
    common::info!("놓침 가운데 새 지도에서 맡긴 장면 {n_held}(보낸 것 가운데 나중에 이어져 지도에 들어간 장면 {n_joined})");
    common::info!("같은 지도에서 다시 잡음 {n_reloc} 번, ms/회 {}", s_reloc.str());
    common::info!("루프 정합에서 단어 id 가 겹치지 않는 노드(원본은 id 순서를 씀) {} 번, 다시 잡기에서 원본이 고쳐 둔 기준 장면을 다시 씀 {} 번(두 번째 판 순서를 모름 {} 번)", eng.loop_unique_ids.load(Ordering::Relaxed), eng.ref_reuse, eng.ref_unknown);
    if let Some(s) = skipped {
        common::info!("재생에서 밀려 건너뛴 장면 {}", s.load(Ordering::Relaxed));
    }
    common::info!("위치 추적이 밀려 버린 장면 {}", src.dropped());
    common::info!("보냄 {}, 밀려 버림 {}, 그래프 블록 {n_graph} 개(노드 {n_graph_nodes} 개)", writer.written(), writer.dropped());
    common::info!("ms/장  읽기    {}", s_read.str());
    common::info!("ms/장  위치추적 {}", s_odom.str());
    common::info!("ms/회  지도갱신 {}", s_slam.str());
    common::info!("ms/장  큐넣기  {}", s_send.str());
    common::info!("ms/장  파이프쓰기(쓰기 스레드) {}", s_write.str());
    common::info!("ms/장  합(본 스레드) {}", s_total.str());
    Ok(0)
}

/// D435 입력: --driver rsdriver(V4L2 직접, 기본) | librealsense(--align 로 정렬 방식). 빌드에 든 것만 고를 수 있다
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
