//! slam_bridge: RGB-D 입력 → 위치 추적(매 장) + 지도 갱신(DetectionRate) → 장면 스트림(memmap --stream).
//! 원본: bridge/src/main.cpp(rtab_bridge). 명령줄·장면마다의 판단(다시 잡기, 새 지도, 맡아 둠, 그래프 블록)·끝 요약을 그대로 옮겼다.
//!
//!   slam_bridge --tum <폴더> [--rate 1] | memmap --stream - ...
//!   slam_bridge --realsense [--align rs2] | memmap --stream - ...
//!
//! 위치 추적·지도는 `engine::Engine` 하나로 부른다(특징점 → OdometryF2F, 다시 잡기 RegistrationVis, 지도 Slam, 난수 상태 하나를 함께 씀).
use slam::io::{FrameSource, PrefetchSource, TumSource};
use slam::params::Params;
use slam::stream::{self, GraphNode, OutFrame, Writer};
use slam::register::{rtf, Tf};
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
    use slam::features::Extractor;
    use slam::loopclose::flann::KdTreeIndex;
    use slam::loopclose::mem::{Kp, NodeFeatures, Signature};
    use slam::loopclose::{GtsamOptimizer, LoopRegistration, NodeInput, RegOut, Slam, WordIndex};
    use slam::odom::OdometryF2F;
    use slam::params::Params;
    use slam::register::{new_shared_rng, RegistrationVis, SharedRng, Tf};
    use slam::{Camera, Features, Frame};
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
            kps: s.kpts.iter().map(|k| slam::KeyPoint { x: k.x, y: k.y, size: k.size, angle: k.angle, response: k.response, octave: k.octave }).collect(),
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

    pub struct Engine {
        ext: Extractor,
        odom: OdometryF2F,
        reloc: RegistrationVis,
        slam: Slam,
        cam: Arc<Mutex<Option<Camera>>>,
        /// 지금 장면 SensorData 의 특징(위치 추적이 data.setFeatures 로 채운 것)
        cur: Features,
        /// Odometry 의 previous_stamp · framesProcessed 그림자(OdometryInfo::interval 계산용)
        prev_stamp: f64,
        frames: i32,
        pub loop_unique_ids: Arc<AtomicU32>,
        pub ref_reuse: u32,
        pub ref_unknown: u32,
    }

    impl Engine {
        /// params = Rtabmap 설정, odom_params = 위치 추적 설정, reloc_min = 다시 잡기 Vis/MinInliers
        pub fn new(params: &Params, odom_params: &Params, reloc_min: i32) -> Engine {
            let rng = new_shared_rng();
            let mut odom = OdometryF2F::new(odom_params);
            odom.set_rng(rng.clone());
            let mut rp = params.clone();
            rp.set("Vis/MinInliers", &reloc_min.to_string());
            let mut reloc = RegistrationVis::new(&rp);
            reloc.set_rng(rng.clone());
            let mut lreg = RegistrationVis::new(params);
            lreg.set_rng(rng.clone());
            let cam = Arc::new(Mutex::new(None));
            let unique = Arc::new(AtomicU32::new(0));
            let index = SharedIndex { inner: KdTreeIndex::from_params(params), rng: rng.clone() };
            let opt = GtsamOptimizer(slam::graph::OptimizerParams::from_params(params));
            let slam = Slam::new(params, Box::new(index), Box::new(opt), Box::new(LoopReg { reg: lreg, cam: cam.clone(), unique_ids: unique.clone() }));
            Engine {
                ext: Extractor::new(odom_params),
                odom,
                reloc,
                slam,
                cam,
                cur: Features::default(),
                prev_stamp: 0.0,
                frames: 0,
                loop_unique_ids: unique,
                ref_reuse: 0,
                ref_unknown: 0,
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
            let feats = feats.unwrap_or_else(|| self.ext.extract(f));
            let stamp = f.t;
            let dt = if self.prev_stamp > 0.0 || (self.prev_stamp == 0.0 && self.frames == 1) { stamp - self.prev_stamp } else { 0.0 };
            let r = self.odom.process(f, feats);
            if r.pose.is_some() {
                self.prev_stamp = stamp;
                self.frames += 1;
            }
            // 놓쳐도 이전 시각은 그대로(Odom/ResetCountdown=0 의 원본 동작)
            let out = OdomOut {
                pose: r.pose.map(|_| r.tf),
                lost: r.lost,
                cov: cov36(&r.covariance),
                features: r.features.kps.len() as i32,
                inliers: r.inliers as i32,
                transform: r.transform,
                interval: if dt > 0.0 { dt } else { 0.0 },
            };
            self.cur = r.features;
            out
        }

        /// 지금 장면에 붙은 특징(다시 잡은 뒤 같은 data 로 위치 추적을 한 번 더 할 때)
        pub fn cur_features(&self) -> Features {
            self.cur.clone()
        }

        /// Odometry::reset(p)
        pub fn odom_reset(&mut self, p: Tf) {
            self.odom.reset_tf(p);
            self.prev_stamp = 0.0;
            self.frames = 0;
        }

        pub fn odom_pose(&self) -> Tf {
            self.odom.pose_tf()
        }

        pub fn make_ref(&self, f: &Frame) -> RefSig {
            let frame = Frame { id: f.id, t: f.t, gray: f.gray.clone(), color: Vec::new(), color_is_rgb: f.color_is_rgb, depth_mm: f.depth_mm.clone(), depth_raw: Vec::new(), depth_per_m: f.depth_per_m, cam: f.cam.clone() };
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
            let features = if node {
                NodeFeatures {
                    kps: self.cur.kps.iter().map(|k| Kp { x: k.x, y: k.y, size: k.size, angle: k.angle, response: k.response, octave: k.octave, class_id: -1 }).collect(),
                    pts3d: self.cur.pts3d.iter().map(|p| [p.x, p.y, p.z]).collect(),
                    desc: self.cur.desc.clone(),
                }
            } else {
                NodeFeatures::default()
            };
            let inp = NodeInput { features, stamp: f.t, pose: pose.unwrap_or(Tf::NULL), cov: *cov, velocity, image_size: (f.cam.width, f.cam.height) };
            self.slam.process(&inp);
        }

        pub fn loop_closure_id(&self) -> i32 {
            self.slam.loop_closure_id()
        }
        pub fn local_optimized(&self) -> &BTreeMap<i32, Tf> {
            self.slam.local_optimized_tf()
        }
        pub fn last_location_id(&self) -> i32 {
            self.slam.last_location_id()
        }
        /// None = null Transform
        pub fn map_correction(&self) -> Option<Tf> {
            let c = self.slam.map_correction();
            (!c.is_null()).then_some(c)
        }
        pub fn trigger_new_map(&mut self) -> i32 {
            self.slam.trigger_new_map()
        }
        pub fn reset(&mut self) {
            self.slam.reset();
        }

        /// Rtabmap::getGraph(optimized=true, global=true): (id, 시각, 위치). 노드가 없으면 빔
        pub fn final_graph(&mut self) -> Vec<(i32, f64, Tf)> {
            let (poses, _) = self.slam.get_graph(true, true);
            poses.iter().filter_map(|(id, t)| self.slam.mem.sig(*id).map(|s| (*id, s.stamp, *t))).collect()
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
    eprintln!(
        "slam_bridge (--tum <폴더> [--rate R] | --realsense [--align rs2] [--serial S] [--size 640x480] [--fps 30] [--exposure ms] [--gain 0..128])\n            [--config <ini>] [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop]\n            [--poses <파일>] [--final-poses <파일>] [--stats <파일>] [--max-frames N] [--reset-file <파일>] [--reloc-wait 초] [--reset-at N] [--Group/Name 값 ...]"
    );
}

struct Args {
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
    reloc_wait: f64,
    reset_at: i64,
    block: i32,
    rate: f64,
    exposure_ms: f64,
    gain: i32,
    send_every: i64,
    queue: usize,
    w: u32,
    h: u32,
    fps: u32,
    max_frames: i64,
    overrides: Vec<(String, String)>,
}

fn parse_args() -> Result<Args, i32> {
    let argv: Vec<String> = std::env::args().collect();
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
        reset_at: -1,
        reloc_wait: 1.0, // 놓친 뒤 이 시간(초) 동안은 같은 지도에서 다시 잡기만 시도
        block: -1,       // -1: 입력에 따라(재생 = 기다림, 카메라 = 버림)
        rate: 0.0,
        exposure_ms: 0.0,
        gain: -1,
        send_every: 1,
        queue: 4,
        w: 640,
        h: 480,
        fps: 30,
        max_frames: -1,
        overrides: vec![],
    };
    let mut i = 1;
    let num = |s: &str| -> Result<f64, i32> {
        s.parse::<f64>().map_err(|_| {
            eprintln!("숫자가 아님: {s}");
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
            "--rate" => a.rate = num(&val()?)?,
            "--align" => a.align = val()?,
            "--serial" => a.serial = val()?,
            "--exposure" => a.exposure_ms = num(&val()?)?,
            "--gain" => a.gain = num(&val()?)? as i32,
            "--size" => {
                let s = val()?;
                if let Some((w, h)) = s.split_once('x') {
                    a.w = w.parse().unwrap_or(a.w);
                    a.h = h.parse().unwrap_or(a.h);
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
                eprintln!("--db: 이 판에는 데이터베이스 저장이 없어 넘어감");
            }
            "--max-frames" => a.max_frames = num(&val()?)? as i64,
            "--reset-file" => a.reset_file = val()?,
            "--reloc-wait" => a.reloc_wait = num(&val()?)?,
            "--reset-at" => a.reset_at = num(&val()?)? as i64, // 시험용: 이 장면 번호에서 리셋 파일이 생긴 것처럼
            "-h" | "--help" => {
                usage();
                return Err(0);
            }
            _ if k.starts_with("--") && k.contains('/') => {
                let name = k[2..].to_string();
                a.overrides.push((name, val()?));
            }
            _ => {
                eprintln!("모르는 인자: {k}");
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

fn run() -> Result<i32, String> {
    let a = match parse_args() {
        Ok(a) => a,
        Err(c) => return Ok(c),
    };
    // 스트림이 표준출력이면 그 파일 기술자를 따로 떼어 두고 표준출력은 표준오류로 돌린다
    #[cfg(unix)]
    let stream_fd = (a.out == "-").then(sys::take_stdout);
    #[cfg(not(unix))]
    let stream_fd: Option<std::fs::File> = None;

    #[allow(unused_assignments)]
    let mut skipped = None;
    let src: Box<dyn FrameSource> = if !a.tum.is_empty() {
        let ts = TumSource::new(&a.tum, a.rate)?;
        skipped = Some(ts.skipped_counter());
        Box::new(ts)
    } else {
        #[cfg(feature = "realsense")]
        {
            Box::new(slam::io::RealSenseSource::new(a.w, a.h, a.fps, &a.align, &a.serial, a.exposure_ms, a.gain)?)
        }
        #[cfg(not(feature = "realsense"))]
        {
            let _ = (&a.align, &a.serial, a.exposure_ms, a.gain, a.w, a.h, a.fps);
            eprintln!("이 빌드에는 RealSense 입력이 없다(--features realsense 로 빌드)");
            return Ok(2);
        }
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
    let ext = slam::features::Extractor::new(&params);
    let mut src = PrefetchSource::with_prep(src, !replay_max, Some(Box::new(move |f: &slam::Frame| ext.extract(f))));
    params.0.remove("Rtabmap/PublishRAMUsage");
    let intermediate = params.get_bool("Rtabmap/CreateIntermediateNodes", false);
    let detection_rate: f32 = params.get("Rtabmap/DetectionRate", 1.0);

    eprintln!("입력: {}", src.describe());
    let mut line = format!("slam (RTAB-Map 0.24.2 경로), 설정 {}, 바꾼 파라미터 {} 개:", if a.config.is_empty() { "(기본값)" } else { &a.config }, params.0.len());
    for (k, v) in &params.0 {
        line += &format!(" {k}={v}");
    }
    eprintln!("{line}");
    eprintln!("출력: {} ({}, 큐 {}, {} 장마다)", a.out, if block { "받는 쪽을 기다림" } else { "밀리면 오래된 장면 버림" }, a.queue, a.send_every);

    // 놓쳤을 때 다시 잡기는 아래에서 직접 한다: 위치 추적이 혼자 '마지막 위치'에서 다시 시작하지 않게 함
    let mut odom_params = params.clone();
    odom_params.set("Odom/ResetCountdown", "0");
    // 다시 잡기: 놓치기 전의 장면(기준 장면)과 지금 장면의 특징점을 직접 맞춰 위치를 구함
    let reloc_min = params.get("Vis/MinInliers", 20i32).max(25);
    let mut eng = engine::Engine::new(&params, &odom_params, reloc_min);

    let mut refs: VecDeque<Ref> = VecDeque::new();
    let mut ref_next = 0usize;
    let mut lost_since = -1.0f64;
    let (mut good_t0, mut good_t1) = (-1.0f64, -1.0f64); // 믿는 지도의 첫·마지막 장면 시각
    let mut restart = false;
    let mut n_reloc = 0i64;
    let mut s_reloc = Summary::default();

    let mut writer = Writer::new(&a.out, stream_fd, a.queue, block)?;
    let mut poses_f = (!a.poses.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.poses).expect("--poses")));
    let mut stats_f = (!a.stats.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.stats).expect("--stats")));
    if let Some(f) = poses_f.as_mut() {
        let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 보낸 그대로)");
    }
    if let Some(f) = stats_f.as_mut() {
        let _ = writeln!(f, "seq,t,read_ms,odom_ms,slam_ms,send_ms,lost,new_map,map_corrected,loop,features,inliers,held");
    }
    #[cfg(unix)]
    sys::on_stop_signals();

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
    let t_run = Instant::now();
    while !STOP.load(Ordering::Relaxed) && (a.max_frames < 0 || n < a.max_frames) {
        let c0 = Instant::now();
        let Some(mut fr) = src.next()? else { break };
        fr.id = n + 1;
        // 리셋 요청(뷰어의 Reset 버튼이 이 파일을 만듦): 위치 추적과 지도를 비우고 받는 쪽에 리셋 블록을 보낸다
        // 지도가 아직 작을 때(3 초 몫 미만) 놓쳐 다시 못 잡은 경우에도 같은 식으로 처음부터 다시 한다
        let reset_req = n == a.reset_at || (!a.reset_file.is_empty() && std::path::Path::new(&a.reset_file).exists());
        if reset_req {
            let _ = std::fs::remove_file(&a.reset_file);
        }
        if reset_req || restart {
            eprintln!("장면 {n}: {} → 위치 추적·지도를 처음부터", if reset_req { "리셋" } else { "지도가 작을 때 놓침" });
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
            // 기준 장면을 최근 것부터 한 장면에 둘씩 돌아가며 맞춰 봄
            let mut k = 0;
            while anchored && k < 2 && k < refs.len() {
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
                eprintln!("장면 {n}: 같은 지도에서 다시 잡음({:.2} 초 전 장면, 맞은 점 {})", fr.t - rt, ri.inliers);
                break;
            }
            s_reloc.add(ms_since(cr));
            // 기다려도 못 잡으면 마지막 위치에서 다시 시작 → 새 지도(이전 지도와 이어질 때까지 보내지 않음)
            if !relocalized && (!anchored || fr.t - lost_since > a.reloc_wait) {
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
        if pose.is_some() {
            lost_since = -1.0;
        }
        let odom_ms = ms_since(c1);

        // 새 지도는 위치 추적이 실제로 다시 시작한 장면(위치가 채워져 옴)에서만 연다
        let mut new_map = false;
        if !fresh && !relocalized && pose.is_some() && info.cov[0] >= 9999.0 {
            eprintln!("장면 {n}: 위치 추적 다시 시작 → 새 지도");
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
        let det_rate = if anchored { detection_rate } else { detection_rate.max(4.0) }; // 이어지기를 기다리는 동안은 더 자주 찾아봄
        if det_rate > 0.0 && prev_stamp > 0.0 && fr.t > prev_stamp && fr.t - prev_stamp < 1.0 / det_rate as f64 {
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
            eng.slam_process(&fr, data_id > 0, pose, &info.cov, info.velocity());
            slam_ms = ms_since(c2);
            loop_ = eng.loop_closure_id() > 0;
            n_loop += loop_ as i64;
            lp = eng.local_optimized().clone();
            let last = eng.last_location_id();
            if !anchored && lp.contains_key(&good_id) {
                anchored = true;
                eprintln!("장면 {n}: 이전 지도와 다시 이어짐");
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
                    refs.push_back(Ref { sig: eng.make_ref(&fr), pose: p, t: fr.t });
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
        let lost = pose.is_none() || info.lost || new_map || !anchored;
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
                let pm = *p;
                let was_held = held_nodes.contains_key(&id);
                let sent = sent_nodes.get(&id);
                if !anchored && sent.is_some() && !was_held {
                    continue; // 믿는 노드는 이어지지 않은 지도의 값으로 덮지 않음
                }
                if let Some(s) = sent {
                    if was_held == !anchored {
                        let dmax = (0..12).map(|i| (s.0[i] - pm.0[i]).abs()).fold(0f32, f32::max);
                        if dmax < 1e-3 {
                            continue;
                        }
                    }
                }
                sent_nodes.insert(id, pm);
                if !anchored && !was_held {
                    held_nodes.insert(id, 0);
                }
                if anchored && was_held {
                    n_joined += held_nodes.remove(&id).unwrap();
                }
                let tn = pm.mul(&optical);
                upd.push(GraphNode { id: id as u32, held: (!anchored) as u32, t: tn.0.map(|v| v as f64) });
            }
            if !upd.is_empty() {
                n_graph_nodes += upd.len() as i64;
                n_graph += 1;
                if !writer.push(OutFrame { graph: upd, ..Default::default() }) {
                    eprintln!("받는 쪽이 닫힘: 끝냄");
                    break;
                }
            }
        }

        // 맡길 장면: 새 지도에서 위치를 구했고 그 지도의 노드를 이미 보냈을 때
        let held = !anchored && pose.is_some() && !info.lost && !new_map && held_nodes.contains_key(&held_last);
        let mut of = OutFrame {
            flags: if lost { stream::LOST } else { 0 } | if new_map { stream::NEW_MAP } else { 0 } | if corrected { stream::MAP_CORRECTED } else { 0 } | if held { stream::HELD } else { 0 },
            map_id,
            seq: n as u64,
            t: fr.t,
            ..Default::default()
        };
        if !lost || held {
            let twc = corr.unwrap_or(Tf::IDENTITY).mul(pose.as_ref().unwrap()).mul(&optical);
            of.pose = twc.0.map(|v| v as f64);
            if !lost {
                if let Some(f) = poses_f.as_mut() {
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
            of.depth = std::mem::take(&mut fr.depth_raw);
            if !writer.push(of) {
                eprintln!("받는 쪽이 닫힘: 끝냄");
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
                "{n},{:.6},{read_ms:.3},{odom_ms:.3},{slam_ms:.3},{send_ms:.3},{},{},{},{},{},{},{}",
                fr.t, lost as i32, new_map as i32, corrected as i32, loop_ as i32, info.features, info.inliers, held as i32
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
        eprintln!("최적화된 지도 노드 {} 개 → {}", g.len(), a.final_poses);
    }
    drop(poses_f);
    drop(stats_f);

    eprintln!("장면 {n} 개, {wall:.1} 초({:.1} 장/초), 놓침 {n_lost}, 새 지도 {n_newmap}, 루프 클로저 {n_loop}, 지도 보정 바뀜 {n_corr}", n as f64 / wall.max(1e-9));
    eprintln!("놓침 가운데 새 지도에서 맡긴 장면 {n_held}(보낸 것 가운데 나중에 이어져 지도에 들어간 장면 {n_joined})");
    eprintln!("같은 지도에서 다시 잡음 {n_reloc} 번, ms/회 {}", s_reloc.str());
    eprintln!("루프 정합에서 단어 id 가 겹치지 않는 노드(원본은 id 순서를 씀) {} 번, 다시 잡기에서 원본이 고쳐 둔 기준 장면을 다시 씀 {} 번(두 번째 판 순서를 모름 {} 번)", eng.loop_unique_ids.load(Ordering::Relaxed), eng.ref_reuse, eng.ref_unknown);
    if let Some(s) = skipped {
        eprintln!("재생에서 밀려 건너뛴 장면 {}", s.load(Ordering::Relaxed));
    }
    eprintln!("위치 추적이 밀려 버린 장면 {}", src.dropped());
    eprintln!("보냄 {}, 밀려 버림 {}, 그래프 블록 {n_graph} 개(노드 {n_graph_nodes} 개)", writer.written(), writer.dropped());
    eprintln!("ms/장  읽기    {}", s_read.str());
    eprintln!("ms/장  위치추적 {}", s_odom.str());
    eprintln!("ms/회  지도갱신 {}", s_slam.str());
    eprintln!("ms/장  큐넣기  {}", s_send.str());
    eprintln!("ms/장  파이프쓰기(쓰기 스레드) {}", s_write.str());
    eprintln!("ms/장  합(본 스레드) {}", s_total.str());
    Ok(0)
}

fn main() {
    let code = match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("slam_bridge 오류: {e}");
            1
        }
    };
    std::process::exit(code);
}
