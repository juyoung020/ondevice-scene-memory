//! 위치 프로그램: RGB-D 입력 → 위치 추적(매 장) + 지도 갱신(DetectionRate) → 장면(같은 프로세스의 메모리 큐 또는 장면 스트림).
//! 원본: RTAB-Map 연결 프로그램 rtab_bridge(C++). 명령줄·장면마다의 판단(다시 잡기, 새 지도, 맡아 둠, 그래프 블록)·끝 요약을 그대로 옮겼다.
//!
//!   ondevice --slam "--tum <폴더> --rate 0" ...   (한 프로세스: `spawn` 이 이 루프를 스레드로 돌리고 장면을 메모리 큐로 넘김)
//!   ondevice slam --tum <폴더> [--rate 1] --out <스트림 파일>   (위치만: 장면 스트림을 파일·표준출력으로)
//!
//!
//! 위치 추적·지도는 `engine::Engine` 하나로 부른다(특징점 → OdometryF2F, 다시 잡기 RegistrationVis, 지도 Slam, 난수 상태 하나를 함께 씀).
use crate::io::{FrameSource, PrefetchSource, TumSource};
use crate::Frame;
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
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// 기한이 지났는가
    fn past(d: Instant) -> bool {
        Instant::now() >= d
    }

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
        corr: ((f32, f32), u32, f32),
        lp: BTreeMap<i32, Tf>,
        last_id: i32,
        map_corr: Tf,
        jobs: u64,
        /// 이 단계(지도 스레드면 끝난 단계들)에 들어간 신경망 루프 링크 수
        lc: u32,
    }

    fn snap_of(s: &Slam) -> MapSt {
        MapSt {
            loop_id: s.loop_closure_id(),
            corr: (s.info.max_correction, s.info.rejected_by_correction, s.info.max_moved_all),
            lp: s.local_optimized_tf().clone(),
            last_id: s.last_location_id(),
            map_corr: s.map_correction(),
            jobs: 0,
            lc: s.info.lc_links,
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

    /// 지도 줄의 노드 한 번: (광류 경로면) 노드 특징을 뽑고 지도 갱신, 결과를 done 에
    #[allow(clippy::too_many_arguments)]
    fn map_proc(slam: &mut Slam, ext: &Extractor, mut inp: Box<NodeInput>, forced: i32, job: u64, node: bool, raw: Option<Box<RawNode>>, d2: &Mutex<(Vec<MapSt>, Vec<(u64, i32)>, Vec<KfOut>)>) {
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
        let st = snap_of(slam);
        let mut d = d2.lock().unwrap();
        d.0.push(st);
        if node {
            d.1.push((job, slam.last_location_id()));
        }
        if let Some((f, p, t)) = kf {
            d.2.push((f, p, t, slam.last_location_id()));
        }
    }

    /// 지도(Slam)를 이 스레드에서 바로 부르거나, 지도 스레드에 일을 넘김
    pub enum MapHost {
        Inline(Box<Slam>, MapSt),
        /// 지도 줄(작업 풀에서 차례대로, 스레드 없음): 지도와 노드 특징 뽑기를 함께 쥠
        Thread {
            lane: Option<common::rt::Lane<(Box<Slam>, Extractor)>>,
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
                MapHost::Thread { lane, done, st, jobs, .. } => {
                    *jobs = job;
                    let d2 = done.clone();
                    lane.as_mut().unwrap().push(move |(slam, ext)| map_proc(slam, ext, Box::new(inp), forced, job, node, raw, &d2));
                    // 끝난 단계가 있으면 가장 최근 결과로(루프 클로저는 그 단계들 가운데 있었으면 알림)
                    let (sts, ids, kfs) = std::mem::take(&mut *done.lock().unwrap());
                    if let Some(last) = sts.last() {
                        let lp_loop = sts.iter().rev().find(|x| x.loop_id > 0).map(|x| x.loop_id).unwrap_or(0);
                        let rej: u32 = sts.iter().map(|x| x.corr.1).sum();
                        let mc = sts.iter().fold((0f32, 0f32), |m, x| (m.0.max(x.corr.0 .0), m.1.max(x.corr.0 .1)));
                        *st = last.clone();
                        st.loop_id = lp_loop;
                        st.corr = (mc, rej, sts.iter().fold(0f32, |m, x| m.max(x.corr.2)));
                        st.lc = sts.iter().map(|x| x.lc).sum();
                    } else {
                        st.loop_id = 0;
                        st.corr = ((0.0, 0.0), 0, 0.0);
                        st.lc = 0;
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
                MapHost::Thread { lane, .. } => {
                    let l = lane.as_mut().unwrap();
                    let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
                    l.push(move |(slam, _)| {
                        let _ = rtx.send(slam.trigger_new_map());
                    });
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
                MapHost::Thread { lane, st, .. } => {
                    lane.as_mut().unwrap().push(|(slam, _)| slam.reset());
                    lane.as_ref().unwrap().wait_idle();
                    *st = MapSt::default();
                }
                MapHost::Empty => {}
            }
        }
        fn start_thread(&mut self, ext: Extractor) {
            let MapHost::Inline(..) = self else { return };
            let MapHost::Inline(slam, st0) = std::mem::replace(self, MapHost::Empty) else { unreachable!() };
            let done: Arc<Mutex<(Vec<MapSt>, Vec<(u64, i32)>, Vec<KfOut>)>> = Default::default();
            let jobs = st0.jobs;
            // 지도 줄은 위치 추적보다 뒤(도는 동안 nice +5)
            *self = MapHost::Thread { lane: Some(common::rt::Lane::new("map", (slam, ext), 5)), done, st: st0, jobs };
        }
        /// 지도 스레드를 끝내고 Slam 을 돌려받음(끝 그래프)
        fn inline(&mut self) -> &mut Slam {
            if let MapHost::Thread { lane, .. } = self {
                let (s, _) = lane.take().unwrap().into_inner();
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
        /// 지도 잇기 후보 가운데 PnP 로 맞춰 볼 수(IMU 모드 5, 그 밖 1). 지도 전체 다시 잡기는 이것과 2 중 큰 수
        pub merge_top: usize,
        /// 다시 잡기 · 지도 잇기 후보를 지금 위치 추적 자세에서 가까운 노드부터(IMU 모드: 놓친 사이 자세를 자이로로 이어 가 위치 추적 좌표가 이어짐)
        pub near_first: bool,
        /// 지금(이어지지 않은) 새 지도의 첫 노드 id(0 = 믿는 지도 안)
        pub map_start: i32,
        rr: usize,
        /// 지도 전체에서 다시 잡기: 최근 40 개 앞 노드를 40 개씩 돌아가며 보는 자리
        grr: usize,
        /// 지금 지도가 믿는 지도인지(app 이 slam_process 앞에 알려 줌)
        pub trusted: bool,
        /// 신경망 루프 스레드와 노드 id 를 기다리는 노드(일 번호), 다음 단계들에 넘길 루프 링크
        lc: Option<crate::lcnet::LcWorker>,
        lc_wait: Vec<(u64, crate::lcnet::LcNode)>,
        lc_links: VecDeque<(i32, i32, Tf, [f64; 2])>,
        lc_recv: VecDeque<crate::lcnet::LcOut>,
        lc_last: i32,
        /// 신경망 루프에 알리는 지도 번호(새 지도마다 +1)
        lc_map: u32,
        /// 최대 속도 재생: 노드마다 루프 결과를 기다려 결정적으로
        pub lc_block: bool,
        /// 통계: 보낸 노드 · 결과 · 맞은 루프 · 지도에 들어간 루프, ms 합(MegaLoc · XFeat · 짝 + 검증), 후보 · 검증 수
        pub lc_stats: LcStats,
    }

    #[derive(Default, Clone, Debug)]
    pub struct LcStats {
        pub sent: u64,
        pub done: u64,
        pub hits: u64,
        pub accepted: u64,
        pub ms: [f64; 3],
        pub ms_max: [f64; 3],
        pub tried: u64,
        /// 열 보호로 건너뛴 노드 수를 세는 자리
        pub skipped: u64,
        /// 위치 추적과 같은 말이라 버린 수 · 받아들인 루프 바로 뒤라 찾지 않은 노드 수
        pub redundant: u64,
        pub gap: u64,
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
                merge_top: 1,
                near_first: false,
                map_start: 0,
                rr: 0,
                grr: 0,
                trusted: true,
                lc: None,
                lc_wait: Vec::new(),
                lc_links: VecDeque::new(),
                lc_recv: VecDeque::new(),
                lc_last: 0,
                lc_map: 0,
                lc_block: false,
                lc_stats: LcStats::default(),
            }
        }

        /// 신경망 루프 켜기(낱말 루프와 함께)
        pub fn start_lc(&mut self, cfg: crate::lcnet::LcConfig) -> Result<(), String> {
            self.lc = Some(crate::lcnet::LcWorker::start(cfg)?);
            Ok(())
        }

        /// 끝난 루프 결과를 다음 노드에 넘길 것으로. 최대 속도 재생(lc_block)은 한 노드 늦게 고정: 바로 앞에 보낸 노드 것만 빼고
        /// 모두 끝날 때까지 기다린다(실행마다 같은 노드에 같은 링크, 신경망은 그 사이 장면들과 겹쳐 돎)
        fn lc_poll(&mut self) {
            let Some(lc) = self.lc.as_mut() else { return };
            while let Some(o) = lc.recv(false) {
                self.lc_recv.push_back(o);
            }
            let n = if self.lc_block {
                let need = self.lc_stats.sent.saturating_sub(1);
                while self.lc_stats.done + (self.lc_recv.len() as u64) < need {
                    match lc.recv(true) {
                        Some(o) => self.lc_recv.push_back(o),
                        None => break,
                    }
                }
                need.saturating_sub(self.lc_stats.done) as usize
            } else {
                self.lc_recv.len()
            };
            for o in self.lc_recv.drain(..n.min(self.lc_recv.len())) {
                let s = &mut self.lc_stats;
                s.done += 1;
                s.tried += o.tried as u64;
                s.redundant += o.redundant as u64;
                s.gap += o.gap as u64;
                for i in 0..3 {
                    s.ms[i] += o.ms[i];
                    s.ms_max[i] = s.ms_max[i].max(o.ms[i]);
                }
                if let Some(h) = o.hit {
                    s.hits += 1;
                    common::info!("신경망 루프: 노드 {} → {} (MegaLoc {:.3}, 맞은 점 {}, 이동 {:.2} m)", h.from, h.to, h.score, h.inliers, h.t.norm_sq().sqrt());
                    // 불확실성: 맞은 점이 많을수록 작게(TUM 긴 사무실 정답 대조: 맞은 점 20 – 50 은 5 – 11 cm, 100 개 넘으면 1 – 3 cm)
                    let k = 1.0 / (h.inliers.max(1) as f64).sqrt();
                    let sd = [(0.3 * k).clamp(0.01, 0.1), (10f64 * k).clamp(0.3, 3.0).to_radians()];
                    self.lc_links.push_back((h.from, h.to, h.t, sd));
                }
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
        /// IMU 융합 켜기(광류 경로). F2F 면 Err
        pub fn set_imu(&mut self, f: crate::imu::Fusion) -> Result<(), String> {
            match &mut self.odom {
                Odom::Klt(o) => {
                    o.imu = Some(f);
                    Ok(())
                }
                _ => Err("IMU 융합은 광류 경로(Odom/Strategy=9)에서만".into()),
            }
        }

        /// 지도 z 가 중력 위로 맞춰져 있는지(IMU)
        pub fn gravity_aligned(&self) -> bool {
            matches!(&self.odom, Odom::Klt(o) if o.gravity_aligned)
        }

        /// IMU 진단: (지도 위와 IMU 위 사이각 °, IMU 로 이어 붙인 장면 수, 치우침, 치우침 고친 수)
        /// IMU 끝 요약 한 줄
        pub fn imu_summary(&self) -> Option<String> {
            match &self.odom {
                Odom::Klt(o) => o.imu.as_ref().map(|u| {
                    let (n, d, f) = u.counts();
                    let b: Vec<u64> = u.ring().bad.iter().map(|x| x.load(Ordering::Relaxed)).collect();
                    format!("IMU: 표본 {n} 개(링 가득 차 버림 {d}, 읽기 실패 {f}, 거른 표본 무효 {} · 포화 {} · 튐 {} · 같은 값 {} · 느린 읽기 {}, 간격이 벌어져 적분 안 함 {}), IMU 회전으로 이어 붙인 장면 {}, 자이로 치우침 {:.4} {:.4} {:.4} rad/s(영상으로 고친 수 {}, 영상과 계속 어긋나 시작 값으로 되돌린 수 {}), 지금 기울기 {:.1}°", b[0], b[1], b[2], b[3], b[4], u.n_gap.get(), o.by_imu, u.bias[0], u.bias[1], u.bias[2], u.n_bias_upd, u.n_bias_reset, u.tilt_deg(&o.pose_tf()))
                }),
                _ => None,
            }
        }

        pub fn imu_diag(&self) -> Option<(f32, usize, [f32; 4], u64)> {
            match &self.odom {
                Odom::Klt(o) => o.imu.as_ref().map(|u| (u.tilt_deg(&o.pose_tf()), o.by_imu, u.bias, u.n_bias_upd)),
                _ => None,
            }
        }

        /// IMU: 이번 장면에서 영상 결과를 IMU 회전과 어긋나 버렸는지
        /// 장면 사이 위치 기준을 이 장면으로(IMU 모드). body = 보낸 몸체 → 지도
        pub fn imu_anchor(&self, t: f64, body: Tf, optical: Tf, vel: [f32; 3]) {
            if let Odom::Klt(o) = &self.odom {
                if let Some(u) = o.imu.as_ref() {
                    if let Some(an) = u.anchor(t, body, optical, vel) {
                        u.ring().set_fast_anchor(an);
                    }
                }
            }
        }

        /// IMU: 다시 잡은 위치(위치 추적 좌표, 몸체 → 지도)가 중력과 맞는지. 가만히 있어 IMU 위 방향을 바로 잴 수 있을 때 15° 넘게 기울면 false
        pub fn imu_pose_plausible(&self, p: &Tf) -> bool {
            match &self.odom {
                Odom::Klt(o) if o.gravity_aligned => match o.imu.as_ref().and_then(|u| u.up_static()) {
                    Some(u) => crate::simd::m34_vec(&crate::imu::m34_of_tf(p), &u)[2] >= 15f32.to_radians().cos(),
                    None => true,
                },
                _ => true,
            }
        }

        /// IMU: 중력 쪽으로 기울기를 고친 장면 수 · 각 합(°) · 다시 시작한 자리에서 한 번에 맞춘 수
        pub fn imu_grav(&self) -> Option<(usize, f64, usize)> {
            match &self.odom {
                Odom::Klt(o) if o.imu.is_some() => Some((o.n_grav, o.grav_deg, o.n_relevel)),
                _ => None,
            }
        }

        /// IMU: 보낸 위치(지도 보정 · 이어 붙임 포함, 몸체 → 지도) 기준 지도 위와 IMU 위 사이각(°)
        pub fn imu_tilt(&self, body: &Tf) -> Option<f32> {
            match &self.odom {
                Odom::Klt(o) => o.imu.as_ref().map(|u| u.tilt_deg(body)),
                _ => None,
            }
        }

        /// IMU: 이번 장면 앞 구간 자이로 회전 속도(°/s)
        pub fn imu_rate(&self) -> f32 {
            match &self.odom {
                Odom::Klt(o) => o.imu_rate,
                _ => 0.0,
            }
        }

        pub fn imu_rejected(&self) -> bool {
            matches!(&self.odom, Odom::Klt(o) if o.imu_rejected)
        }

        /// 광류 경로의 마지막 장면 진단
        pub fn klt_diag(&self) -> Option<crate::klt::Diag> {
            match &self.odom {
                Odom::Klt(o) => Some(o.diag),
                _ => None,
            }
        }

        pub fn bridge_verified(&self) -> bool {
            matches!(&self.odom, Odom::Klt(o) if o.bridge_verified)
        }

        /// from 가운데 지금 위치 추적 자세와 가까운(키프레임 위치 거리, 같으면 번호 순) n 개
        fn nearest_kf(&self, from: &[usize], n: usize) -> Vec<usize> {
            let p = self.odom_pose();
            let d = |i: usize| {
                let q = &self.kf_db[i].1;
                (q.x() - p.x()).powi(2) + (q.y() - p.y()).powi(2) + (q.z() - p.z()).powi(2)
            };
            let mut v: Vec<(f32, usize)> = from.iter().map(|&i| (d(i), i)).collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            v.into_iter().take(n).map(|x| x.1).collect()
        }

        /// deadline: 이 시각을 넘으면 남은 후보(PnP RANSAC 한 번에 보드 수 ms – 수십 ms)는 다음 장면으로 미룸(첫 후보는 늘 맞춰 봄)
        pub fn global_reloc(&mut self, f: &Frame, min_inliers: usize, deadline: Instant) -> Option<(Tf, i32, [f64; 36])> {
            self.ensure_cur(f);
            if self.cur.desc.is_empty() || self.kf_db.is_empty() {
                return None;
            }
            let words = |d: &[u8; 32]| -> [u64; 4] { std::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap())) };
            let step = (self.cur.desc.len() / 100).max(1);
            let q: Vec<[u64; 4]> = self.cur.desc.iter().step_by(step).map(words).collect();
            // 후보: 최근 40 개 + 그 앞 노드 40 개(장면마다 다음 40 개, 놓친 동안 몇 장면이면 지도 전체를 한 바퀴)
            let from = self.kf_db.len().saturating_sub(40);
            let mut cand: Vec<usize> = (from..self.kf_db.len()).collect();
            if from > 0 && self.near_first && from > 40 {
                // IMU 모드: 앞 노드 가운데 가까운 20 개 + 돌아가며 20 개
                let older: Vec<usize> = (0..from).collect();
                let near = self.nearest_kf(&older, 20);
                self.grr = (self.grr + 20) % from;
                let rr: Vec<usize> = (0..20).map(|j| (self.grr + j) % from).filter(|i| !near.contains(i)).collect();
                cand.extend(near);
                cand.extend(rr);
            } else if from > 0 {
                let k = from.min(40);
                self.grr = (self.grr + k) % from;
                cand.extend((0..k).map(|j| (self.grr + j) % from));
            }
            let mut scores: Vec<(usize, usize)> = cand
                .iter()
                .map(|&i| (i, &self.kf_db[i]))
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
            for (k, &(hits, i)) in scores.iter().take(self.merge_top.max(2)).enumerate() {
                if hits < 10 || (k > 0 && past(deadline)) {
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
            if node {
                self.lc_poll();
            }
            let lc_link = if node { self.lc_links.pop_front() } else { None };
            let inp = NodeInput { features, stamp: f.t, pose: pose.unwrap_or(Tf::NULL), cov: *cov, velocity, image_size: (f.cam.width, f.cam.height), prov_from: self.prov_from, lc_link };
            let had = node && self.kf_ids.last().map(|x| x.0 == 0).unwrap_or(false);
            let job = self.slam.jobs() + 1;
            let lc_node = node && self.lc.is_some() && !f.color.is_empty();
            if let (true, Some(p)) = (lc_node, pose) {
                let n = crate::lcnet::LcNode { id: 0, stamp: f.t, color: f.color.clone(), bgr: !f.color_is_rgb, depth: f.depth_mm.clone(), pose: p.to_pose(), cam: f.cam.clone(), map: self.lc_map };
                self.lc_wait.push((job, n));
            }
            if had {
                if let Some(l) = self.kf_ids.last_mut() {
                    l.2 = job;
                }
            }
            let forced = std::mem::take(&mut self.forced_hyp);
            let (ids, kfs) = self.slam.process(inp, forced, had || lc_node, job, raw);
            self.lc_stats.accepted += self.slam.st().lc as u64;
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
                // 신경망 루프: id 가 정해진 노드를 루프 스레드로(새 노드가 안 생겨 id 가 그대로면 넘기지 않음)
                if let Some(i) = self.lc_wait.iter().position(|e| e.0 == k) {
                    let (_, mut n) = self.lc_wait.swap_remove(i);
                    let hot = crate::lcnet::THROTTLE.load(Ordering::Relaxed) && {
                        self.lc_stats.skipped += 1;
                        self.lc_stats.skipped % 3 != 0
                    };
                    if id > 0 && id != self.lc_last && !hot {
                        self.lc_last = id;
                        n.id = id;
                        self.lc_stats.sent += 1;
                        self.lc.as_mut().unwrap().send(n);
                    }
                }
            }
            self.lc_wait.retain(|e| e.0 + 8 > job);
        }

        /// 이어지지 않은 새 지도에서: 믿는 지도의 노드 가운데 이번 장면과 기술자로 맞는 노드를 찾아(PnP 확인)
        /// 다음 slam 단계의 루프 후보로 넘긴다. 실제 연결은 slam 이 원래 루프처럼 정합·최적화 검사를 거쳐 정한다
        /// 지금 지도의 마지막 노드 id(지도 스레드면 한 단계 늦음)
        pub fn last_node(&self) -> i32 {
            self.slam.st().last_id
        }

        /// deadline: global_reloc 와 같음
        pub fn propose_merge(&mut self, f: &Frame, min_inliers: usize, deadline: Instant) -> bool {
            self.ensure_cur(f);
            if self.cur.desc.is_empty() {
                return false;
            }
            let words = |d: &[u8; 32]| -> [u64; 4] { std::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap())) };
            let step = (self.cur.desc.len() / 60).max(1);
            let q: Vec<[u64; 4]> = self.cur.desc.iter().step_by(step).map(words).collect();
            let mut best: Vec<(usize, usize)> = Vec::new();
            // 믿는 지도의 노드 가운데 최근 40 개(새 지도 노드는 빼고 셈: 전에는 최근 40 칸이 모두 새 지도 노드라 후보가 0 이었음)
            // 후보: 이어 붙인 구간이면 그 앞 노드, 새 지도면 지금 지도보다 앞의 모든 노드(믿는 지도 · 다른 새 지도 모두).
            // 노드 전체를 40 개씩 돌아가며 봄(장면당 시간 상한, 몇 장면이면 지도 전체를 한 바퀴)
            let below = if self.prov_from > 0 { self.prov_from } else if self.map_start > 0 { self.map_start } else { i32::MAX };
            let all: Vec<usize> = (0..self.kf_db.len()).filter(|&i| self.kf_ids[i].0 > 0 && self.kf_ids[i].0 < below && (self.map_start > 0 || self.kf_ids[i].1)).collect();
            let cand: Vec<usize> = if all.len() <= 40 {
                all
            } else if self.near_first {
                // 가까운 20 개 + 돌아가며 20 개
                let mut c = self.nearest_kf(&all, 20);
                self.rr = (self.rr + 20) % all.len();
                c.extend((0..20).map(|k| all[(self.rr + k) % all.len()]).filter(|i| !c.contains(i)).collect::<Vec<_>>());
                c
            } else {
                self.rr = (self.rr + 40) % all.len();
                (0..40).map(|k| all[(self.rr + k) % all.len()]).collect()
            };
            for &i in &cand {
                let kf = &self.kf_db[i].0;
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
            for (k, &(hits, i)) in best.iter().take(self.merge_top).enumerate() {
                if hits < 8 || (k > 0 && past(deadline)) {
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
        pub fn correction_info(&self) -> ((f32, f32), u32, f32) {
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
            self.lc_map += 1;
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
        "ondevice slam (--tum|--rsdrec <폴더> [--rate R] [--stride N] | --realsense [--serial S] [--size 640x480] [--color-size 1280x720] [--fps 30] [--camera-profile librealsense|tracking] [--exposure ms | --ae-cap ms] [--gain 0..128])\n            [--config <ini>] [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop]\n            [--poses <파일>] [--final-poses <파일>] [--stats <파일>] [--max-frames N] [--reset-file <파일>] [--reloc-wait 초] [--reset-at N] [--imu /dev/i2c-1 [--imu-calib <yaml>] [--imu-bridge 초] [--imu-max-blur 화소] [--imu-exposure 초] [--imu-min-inliers N] [--imu-min-ratio R] [--imu-hold-wait 초] [--imu-hold-min-nodes N]] [--imu-csv <imu0.csv> (재생)] [--stress-gap 주기:길이 [--stress-imu-gap]] [--stress-blur 노출초 [--stress-gyro <imu0.csv>]] [--Group/Name 값 ...]"
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
    /// 이 시간(초) 이하의 빈틈만 그 자리에서 이어 붙임(그보다 길면 새 지도, 깊이 ICP 로 확인되면 예외)
    bridge_gap: f64,
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
    overrides: Vec<(String, String)>,
    /// IMU 융합 모드: I2C 장치(빈 값 = 카메라 전용 모드)
    imu: String,
    /// IMU-카메라 보정 yaml(빈 값 = 저장소 docs/imu_cam_calib.yaml 을 빌드에 넣은 것)
    imu_calib: String,
    /// 영상이 다 실패한 구간을 IMU 회전으로 이어 붙이는 최대 시간(초)
    imu_bridge: f64,
    /// IMU 모드: 번짐(자이로 각속도 × 노출 × 초점 거리, 화소)이 이보다 큰 장면은 위치만(지도에 안 넣음) · 다시 잡기 · 지도 잇기도 건너뜀
    imu_max_blur: f64,
    /// 번짐 계산에 쓰는 노출(초)
    imu_exposure: f64,
    /// IMU 모드: 광류 맞은 점이 이보다 적거나 맞은 비율이 이보다 낮은 장면은 위치만
    imu_min_inliers: f64,
    imu_min_ratio: f64,
    /// IMU 모드: 이어지지 않은 새 지도를 맡겨 두고 기다리는 최대 시간(초, 기본 끝없음 = 이어질 때까지). 지나면 새 지도를 받아들여 바로 쌓음
    imu_hold_wait: f64,
    /// IMU 모드: 앞의 믿는 지도 노드가 이보다 적으면(시작 직후) 기다리지 않고 바로 새 지도로
    imu_hold_min_nodes: f64,
    /// 녹화 IMU 재생(Kalibr imu0.csv): IMU 융합 모드를 --imu 대신 이 파일로(--tum · --rsdrec 재생과 함께)
    imu_csv: String,
    /// 끊김 시험: 주기:길이(초). 첫 장면부터 주기마다 길이만큼 장면을 빼고, --stress-imu-gap 이면 IMU 표본도 뺌
    stress_gap: (f64, f64),
    stress_imu_gap: bool,
    /// 흐림 시험: 노출(초) 동안 각속도만큼 번진 영상(각속도는 --stress-gyro 또는 --imu-csv 파일)
    stress_blur: f64,
    stress_gyro: String,
}

/// 명령줄(0 번은 프로그램 이름) → 설정. 도움말·잘못된 인자면 끝 코드
impl Args {
    /// 우리 깊이 카메라 녹화(rsd_rec · rsdrec 폴더, calib.json 있음) 재생인지: 손에 든 카메라를 똑바로 세우고 시작한 것이라 실시간과 같게 다룬다
    pub fn is_own_recording(&self) -> bool {
        !self.tum.is_empty() && std::path::Path::new(&self.tum).join("calib.json").exists()
    }
    /// 받는 쪽을 기다리는지(--block | --drop, 기본: 최대 속도 재생이면 기다림)
    pub fn block_mode(&self) -> bool {
        let replay_max = !self.tum.is_empty() && self.rate == 0.0;
        if self.block < 0 { replay_max } else { self.block == 1 }
    }
}

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
        reloc_wait: -1.0, // 놓친 뒤 이 시간(초) 동안은 같은 지도에서 다시 잡기만 시도. 음수 = 광류 경로 RELOC_WAIT_KLT, F2F 1 초
        bridge_gap: 0.0, // 이어 붙이기는 깊이 ICP 로 자리가 확인될 때만(확인 안 된 이어 붙이기는 어긋난 위치를 감춤). 못 찾으면 새 지도(임시 위치, 불확실 표시) → 루프로 합침
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
        imu: String::new(),
        imu_calib: String::new(),
        imu_bridge: 1.0,
        imu_max_blur: 15.0,
        imu_exposure: 0.03,
        imu_min_inliers: 60.0,
        imu_min_ratio: 0.6,
        imu_hold_wait: f64::INFINITY, // 이어지지 않은 새 지도를 임시 위치로 쌓으면 끝내 안 이어질 때 같은 책상 · 물체가 어긋난 자리에 한 번 더 남는다
        imu_hold_min_nodes: 30.0,
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
            "--tum" => a.tum = val()?,
            "--rsdrec" => a.tum = val()?,
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
            "--imu" => a.imu = val()?,
            "--imu-calib" => a.imu_calib = val()?,
            "--imu-bridge" => a.imu_bridge = num(&val()?)?,
            "--imu-max-blur" => a.imu_max_blur = num(&val()?)?,
            "--imu-exposure" => a.imu_exposure = num(&val()?)?,
            "--imu-min-inliers" => a.imu_min_inliers = num(&val()?)?,
            "--imu-min-ratio" => a.imu_min_ratio = num(&val()?)?,
            "--imu-hold-wait" => a.imu_hold_wait = num(&val()?)?,
            "--imu-hold-min-nodes" => a.imu_hold_min_nodes = num(&val()?)?,
            "--reset-file" => a.reset_file = val()?,
            "--reloc-wait" => a.reloc_wait = num(&val()?)?,
            "--bridge-gap" => a.bridge_gap = num(&val()?)?,
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

/// 명령줄 그대로 실행(`ondevice slam …`): 스트림을 파일·표준출력으로 쓴다
pub fn run(argv: &[String]) -> Result<i32, String> {
    match parse(argv) {
        Ok(a) => run_in(a, None),
        Err(c) => Ok(c),
    }
}

/// IMU: 이어 붙인 장면 · 영상이 IMU 와 어긋난 장면 뒤 이만큼 장면까지는 위치만(지도에 안 쌓음)
const IMU_SETTLE: u32 = 5;
/// 광류 경로: 놓친 뒤 새 지도를 열기 전에 기존 지도(최근 · 전체 노드)에 다시 잡아 보는 시간(초)
const RELOC_WAIT_KLT: f64 = 0.0;
/// 새 지도를 연 뒤 이 시간(초) 동안은 앞 지도 노드에 잇기를 장면마다(그 뒤는 0.2 초 간격)
const MERGE_FAST: f64 = 3.0;

/// 위치 프로그램 한 판의 상태: 장면마다 [`Runner::step`]. 따로 도는 프로그램(`ondevice slam`)은 [`run_in`] 이 입력을 돌리고,
/// 같은 프로세스(`ondevice --slam`)는 처리 루프가 장면이 올 때마다 부른다(스레드 없음)
pub struct Runner {
    t_run: Instant,
    in_proc: bool,
    calib: crate::imu::Calib,
    imu_replay: Option<crate::imu::CsvReplay>,
    pub imu_ring: Option<std::sync::Arc<crate::imu::Ring>>,
    imu_on: bool,
    skipped: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    stress_counts: Option<(std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<std::sync::atomic::AtomicUsize>)>,
    replay_max: bool,
    intermediate: bool,
    detection_rate: f32,
    reloc_min: i32,
    eng: engine::Engine,
    klt_path: bool,
    reloc_wait: f64,
    bridge: bool,
    n_bridge: i64,
    n_global: i64,
    n_reloc_bridge: i64,
    global_tried: bool,
    n_reloc_tilt: i64,
    n_loop_rejected: i64,
    n_merge_try: i64,
    nq: [i64; 3],
    bridge_t: f64,
    n_bridge_fix: i64,
    last_merge_try: f64,
    newmap_t: f64,
    prev_map_small: bool,
    n_accept: i64,
    max_corr: (f32, f32),
    last_good_t: f64,
    n_bridge_ok: i64,
    since_imu: u32,
    n_imu_hold: i64,
    n_imu_reject: i64,
    n_fast_rot: i64,
    n_weak: i64,
    n_map_held: i64,
    refs: VecDeque<Ref>,
    ref_next: usize,
    lost_since: f64,
    good_t0: f64,
    good_t1: f64,
    restart: bool,
    n_reloc: i64,
    s_reloc: Summary,
    pub writer: Sink,
    poses_f: Option<std::io::BufWriter<std::fs::File>>,
    stats_f: Option<std::io::BufWriter<std::fs::File>>,
    optical: Tf,
    s_read: Summary,
    s_odom: Summary,
    s_slam: Summary,
    s_send: Summary,
    s_total: Summary,
    pub n: i64,
    n_lost: i64,
    n_newmap: i64,
    n_loop: i64,
    n_corr: i64,
    n_graph: i64,
    n_graph_nodes: i64,
    sent_nodes: BTreeMap<i32, Tf>,
    held_nodes: BTreeMap<i32, i64>,
    held_last: i32,
    n_held: i64,
    n_joined: i64,
    map_id: u32,
    fresh: bool,
    anchored: bool,
    good_id: i32,
    prev_stamp: f64,
    prev_corr: Tf,
    prov: Tf,
    /// 믿는 지도들의 (첫 노드 id, 표시 변환): 처음 지도 (0, 그대로) + 받아들인 새 지도마다 하나
    bases: Vec<(i32, Tf)>,
    prov_nodes: std::collections::BTreeSet<i32>,
    last_sent_world: Option<Tf>,
    /// 그 장면의 위치 추적 위치(몸체 → 위치 추적 좌표)
    last_sent_odom: Option<Tf>,
    fast_prev: Option<(f64, [f32; 3])>,
    fast_vel: [f32; 3],
    tilt_map: f32,
    tilt_map_max: f32,
    n_prov: i64,
    pub a: Args,
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

impl Runner {
    /// 준비: 설정 · IMU · 위치 추적 엔진 · 출력. 장면 입력은 돌려주는 (원래 입력, 미리 할 일)로 부르는 쪽이 돌린다.
    /// local: 장면 스트림을 같은 프로세스 처리 루프의 줄로(아니면 파일 · 표준출력 쓰기 스레드)
    pub fn new(a: Args, local: bool, pool: Option<std::sync::Arc<rayon::ThreadPool>>) -> Result<(Runner, Box<dyn FrameSource>, Option<crate::io::Prep>), String> {
        let in_proc = local;
        // 실시간이면 위치 추적 스레드를 분할·memmap·뷰어보다 먼저
        if a.realsense || a.rate > 0.0 {
            common::info!("위치 추적 스레드 우선: {}", common::threads::boost_tracking());
        }
        // IMU 융합 모드: 읽기 스레드를 카메라보다 먼저 띄워 첫 장면 전에 중력·치우침을 잡는다
        let calib = crate::imu::Calib::parse(&if a.imu_calib.is_empty() { crate::imu::DEFAULT_CALIB.to_string() } else { std::fs::read_to_string(&a.imu_calib).map_err(|e| format!("{}: {e}", a.imu_calib))? })?;
        if !a.imu_csv.is_empty() && a.tum.is_empty() {
            return Err("--imu-csv 는 --tum · --rsdrec 재생과 함께".into());
        }
        // 녹화 IMU 재생: 실시간 읽기 스레드 자리에 파일 표본을 같은 링으로(최대 속도 재생은 장면마다 그 시각까지, 찍은 속도 재생은 재생 시계로)
        let mut imu_replay = if a.imu_csv.is_empty() { None } else { Some(crate::imu::CsvReplay::load(&a.imu_csv, calib.shift)?) };
        let mut imu_ring = None;
        let imu_fusion = if a.imu.is_empty() && imu_replay.is_none() {
            None
        } else {
            let ring = crate::imu::Ring::new();
            let ok = match &imu_replay {
                Some(r) => {
                    r.count_into(&ring);
                    let (t0, t1) = r.span();
                    common::info!("IMU 재생: {} 표본 {} 개({t0:.3} – {t1:.3})", a.imu_csv, r.len());
                    true
                }
                // 실시간 IMU 가 응답하지 않으면(배선 빠짐 등) 실행을 멈추지 않고 카메라 전용으로
                None => match crate::imu::spawn_reader(&a.imu, ring.clone()) {
                    Ok(_) => true,
                    Err(e) => {
                        common::info!("IMU 없음: {e} → 카메라 전용으로 계속(IMU 배선 · I2C 를 확인)");
                        false
                    }
                },
            };
            ok.then(|| {
                imu_ring = Some(ring.clone());
                common::info!("모드: IMU 융합(시각 보정 {:+.4} 초, 이어 붙이기 최대 {} 초)", calib.shift, a.imu_bridge);
                crate::imu::Fusion::new(ring, calib.clone(), a.imu_bridge)
            })
        };
        let imu_on = imu_fusion.is_some();
        // 스트림이 표준출력이면 그 파일 기술자를 따로 떼어 두고 표준출력은 표준오류로 돌린다
        #[cfg(unix)]
        let stream_fd = (!in_proc && a.out == "-").then(sys::take_stdout);
        #[cfg(not(unix))]
        let stream_fd: Option<std::fs::File> = None;

        #[allow(unused_assignments)]
        let mut skipped = None;
        let mut stress_counts = None;
        let src: Box<dyn FrameSource> = if !a.tum.is_empty() {
            let mut ts = TumSource::with_stride(&a.tum, a.rate, a.stride)?;
            skipped = Some(ts.skipped_counter());
            // 끊김 시험: 첫 장면 + 주기 k 번째마다 길이만큼 장면(--stress-imu-gap 이면 IMU 표본도) 뺌
            if a.stress_gap.0 > 0.0 && a.stress_gap.1 > 0.0 {
                let t0 = ts.t_first();
                ts.drop = (1..10_000).map(|k| (t0 + k as f64 * a.stress_gap.0, t0 + k as f64 * a.stress_gap.0 + a.stress_gap.1)).collect();
                if let (Some(r), true) = (imu_replay.as_mut(), a.stress_imu_gap) {
                    r.drop = ts.drop.clone();
                }
                common::info!("끊김 시험: {} 초마다 {} 초 장면 뺌{}", a.stress_gap.0, a.stress_gap.1, if a.stress_imu_gap { "(IMU 표본도)" } else { "" });
            }
            // 흐림 시험: 녹화 IMU 의 각속도로 번진 영상(카메라 전용 · IMU 모드가 같은 영상을 받게 같은 파일에서)
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
            if let (Some(r), Some(ring)) = (imu_replay.take_if(|_| a.rate > 0.0), imu_ring.as_ref()) {
                r.spawn_timed(ring.clone(), ts.clock(), a.rate);
            }
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
        // 루프 클로저: RTAB-Map 낱말 루프 + MegaLoc 후보 · XFeat 검증(모델 폴더 common::models::dir(), 없으면 시작하지 않음)
        {
            let cfg = crate::lcnet::LcConfig::from_dir(&common::models::dir().to_string_lossy());
            for f in [&cfg.megaloc, &cfg.head, &cfg.xfeat] {
                if !std::path::Path::new(f).exists() {
                    return Err(format!("루프 클로저 모델이 없음: {f} (모델 폴더 = $ONDEVICE_MODELS · 보드 ~/models · PC models/)"));
                }
            }
            common::info!("루프 클로저: 낱말 + 신경망(MegaLoc {} · XFeat {} · 후보 {} · 내적 ≥ {} · {} 초 안 노드 뺌)", cfg.megaloc, cfg.xfeat, cfg.topk, cfg.min_score, cfg.skip_s);
            eng.start_lc(cfg).map_err(|e| format!("루프 클로저 신경망: {e}"))?;
            eng.lc_block = !a.tum.is_empty() && a.rate == 0.0;
        }
        if let Some(f) = imu_fusion {
            eng.set_imu(f)?;
            eng.merge_top = 5;
            eng.near_first = true;
        }
        // 실시간(카메라·찍은 속도 재생)이면 지도 갱신을 따로 도는 스레드로. 최대 속도 재생은 결정적이게 그대로
        if a.realsense || a.rate > 0.0 {
            eng.start_map_thread();
        }
        let reloc_wait = if a.reloc_wait >= 0.0 { a.reloc_wait } else if klt_path { RELOC_WAIT_KLT } else { 1.0 };
        let bridge = klt_path;
        let n_bridge = 0i64;
        let n_global = 0i64;
        let n_reloc_bridge = 0i64; // IMU 로 이어 붙이던 중 지도 전체에서 다시 잡은 수
        let global_tried = false;
        let n_reloc_tilt = 0i64; // IMU: 중력과 15° 넘게 어긋나 버린 다시 잡기
        let n_loop_rejected = 0i64;
        let n_merge_try = 0i64;
        let nq = [0i64; 3]; // 위치 질별 장면 수
        let bridge_t = f64::NEG_INFINITY; // 확인 안 된 이어 붙이기 시각
        let n_bridge_fix = 0i64;
        let last_merge_try = f64::MIN;
        let newmap_t = f64::MIN; // 마지막으로 새 지도를 연 장면 시각
        let (prev_map_small, n_accept) = (false, 0i64);
        let max_corr = (0f32, 0f32);
        let last_good_t = f64::MAX; // 마지막으로 위치를 낸 장면 시각
        let n_bridge_ok = 0i64;
        // IMU: 이어 붙인 장면 · 영상이 IMU 와 어긋난 장면 뒤 IMU_SETTLE 장까지는 위치만(지도에 안 쌓음)
        let (since_imu, n_imu_hold, n_imu_reject) = (u32::MAX, 0i64, 0i64);
        let (n_fast_rot, n_weak, n_map_held) = (0i64, 0i64, 0i64);

        let refs: VecDeque<Ref> = VecDeque::new();
        let ref_next = 0usize;
        let lost_since = -1.0f64;
        let (good_t0, good_t1) = (-1.0f64, -1.0f64); // 믿는 지도의 첫·마지막 장면 시각
        let restart = false;
        let n_reloc = 0i64;
        let s_reloc = Summary::default();

        let writer = if local { Sink::Local(LocalQ { q: VecDeque::new(), cap: a.queue.max(1), block, written: 0, dropped: 0 }) } else { Sink::Pipe(Writer::new(&a.out, stream_fd, a.queue, block)?) };
        let mut poses_f = (!a.poses.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.poses).expect("--poses")));
        let mut stats_f = (!a.stats.is_empty()).then(|| std::io::BufWriter::new(std::fs::File::create(&a.stats).expect("--stats")));
        if let Some(f) = poses_f.as_mut() {
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 보낸 그대로)");
        }
        if let Some(f) = stats_f.as_mut() {
            let _ = writeln!(f, "seq,t,read_ms,odom_ms,slam_ms,send_ms,lost,new_map,map_corrected,loop,features,inliers,held,no_pose,quality,method,klt_inliers,klt_ratio,reproj_rms,spread,jump{}", if imu_on { ",tilt_deg,no_map,tilt_map" } else { "" });
        }
        #[cfg(unix)]
        if !in_proc {
            sys::on_stop_signals();
        }

        // 카메라 광학 축 → 몸체 축
        let optical = rtf::OPTICAL;
        let (s_read, s_odom, s_slam, s_send, s_total) = (Summary::default(), Summary::default(), Summary::default(), Summary::default(), Summary::default());
        let (n, n_lost, n_newmap, n_loop, n_corr, n_graph, n_graph_nodes) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        let sent_nodes: BTreeMap<i32, Tf> = BTreeMap::new(); // memmap 에 보낸 노드 위치(지도 몸체 축)
        // 믿는 지도와 아직 안 이어진 새 지도: 노드를 "맡아 둠" 표시로 보내고 장면도 위치·기준 노드와 함께 맡겨 둔다
        let held_nodes: BTreeMap<i32, i64> = BTreeMap::new(); // 맡아 둔 노드 id → 그 노드를 기준으로 맡긴 장면 수
        let held_last = -1i32; // 지금 새 지도의 마지막 노드(없으면 -1)
        let (n_held, n_joined) = (0i64, 0i64);
        let map_id = 0u32;
        let fresh = true; // 처음이거나 리셋 직후의 장면: 위치 추적이 새로 시작하는 것이 정상
        // 다시 시작한 뒤의 위치는 "마지막 위치에서 이어 붙인" 값이라, 믿던 지도의 노드(good_id)와 이어질 때까지 놓침으로 보낸다
        let anchored = true;
        let good_id = -1i32;
        let prev_stamp = 0.0f64;
        let prev_corr = Tf::IDENTITY;
        // 임시 위치: 이어지지 않은 새 지도의 장면·노드를 맡아 두지 않고, 마지막으로 보낸 위치에서 이어 붙인
        // 표시 변환(prov)으로 바로 보낸다(화면이 멈추지 않음). slam 그래프에는 아무 제약도 넣지 않는다(표시 전용).
        // 루프 클로저로 이어지면 노드가 참 위치로 다시 가고(움직임 → memmap 다시 쌓기) 그 장면들도 고친 위치로 다시 놓인다.
        let prov = Tf::IDENTITY;
        let bases = vec![(0, Tf::IDENTITY)];
        let prov_nodes: std::collections::BTreeSet<i32> = Default::default();
        let last_sent_world: Option<Tf> = None; // 마지막으로 보낸 장면의 몸체 위치(표시 좌표)
        let last_sent_odom: Option<Tf> = None;
        // IMU: 장면 사이 위치 기준의 지도 좌표 속도(앞 장면 위치 · 시각, 이동 평균)
        let (fast_prev, fast_vel): (Option<(f64, [f32; 3])>, [f32; 3]) = (None, [0.0; 3]);
        let tilt_map = -1f32;
        let tilt_map_max = 0f32;
        let n_prov = 0i64;
        let t_run = Instant::now();
        Ok((Runner {
            t_run,
            in_proc,
            calib,
            imu_replay,
            imu_ring,
            imu_on,
            skipped,
            stress_counts,
            replay_max,
            intermediate,
            detection_rate,
            reloc_min,
            eng,
            klt_path,
            reloc_wait,
            bridge,
            n_bridge,
            n_global,
            n_reloc_bridge,
            global_tried,
            n_reloc_tilt,
            n_loop_rejected,
            n_merge_try,
            nq,
            bridge_t,
            n_bridge_fix,
            last_merge_try,
            newmap_t,
            prev_map_small,
            n_accept,
            max_corr,
            last_good_t,
            n_bridge_ok,
            since_imu,
            n_imu_hold,
            n_imu_reject,
            n_fast_rot,
            n_weak,
            n_map_held,
            refs,
            ref_next,
            lost_since,
            good_t0,
            good_t1,
            restart,
            n_reloc,
            s_reloc,
            writer,
            poses_f,
            stats_f,
            optical,
            s_read,
            s_odom,
            s_slam,
            s_send,
            s_total,
            n,
            n_lost,
            n_newmap,
            n_loop,
            n_corr,
            n_graph,
            n_graph_nodes,
            sent_nodes,
            held_nodes,
            held_last,
            n_held,
            n_joined,
            map_id,
            fresh,
            anchored,
            good_id,
            prev_stamp,
            prev_corr,
            prov,
            bases,
            prov_nodes,
            last_sent_world,
            last_sent_odom,
            fast_prev,
            fast_vel,
            tilt_map,
            tilt_map_max,
            n_prov,
            a,
        }, src, prep))
    }

    /// 장면 하나(위치 추적 · 다시 잡기 · 지도 갱신 · 장면 스트림 내보내기). read_ms: 읽기 시간, prepped: 미리 뽑아 둔 특징(F2F 경로).
    /// 돌려줌: false = 받는 쪽이 닫혀 끝냄
    pub fn step(&mut self, mut fr: Frame, read_ms: f64, prepped: Option<crate::Features>) -> bool {
        let n = self.n;
        let c0 = Instant::now();
        // 녹화 IMU 재생(최대 속도): 이 장면을 처리하는 순간 실시간이면 링에 있을 표본까지
        if let (Some(r), Some(ring)) = (self.imu_replay.as_mut(), self.imu_ring.as_ref()) {
            r.feed(ring, fr.t + self.calib.shift + crate::imu::REPLAY_LEAD);
        }
        // IMU 모드: BRIEF 를 점마다 방향으로 돌려 뽑음(ORB 식) → 굴려도 같은 설명자, 다시 잡기 · 지도 잇기 · 루프 검증이 시점 변화에 버팀
        fr.cam.oriented = self.imu_on && self.klt_path;
        fr.id = self.n + 1;
        // 리셋 요청(뷰어의 Reset 버튼: 한 프로그램이면 common::control 신호, 따로 띄운 뷰어면 --reset-file 파일): 위치 추적과 지도를 비우고 받는 쪽에 리셋 블록을 보낸다
        // 지도가 아직 작을 때(3 초 몫 미만) 놓쳐 다시 못 잡은 경우에도 같은 식으로 처음부터 다시 한다
        let reset_req = self.n == self.a.reset_at || common::control::take_reset() || (!self.a.reset_file.is_empty() && std::path::Path::new(&self.a.reset_file).exists());
        if reset_req {
            let _ = std::fs::remove_file(&self.a.reset_file);
        }
        if reset_req || self.restart {
            common::info!("장면 {n}: {} → 위치 추적·지도를 처음부터", if reset_req { "리셋" } else { "지도가 작을 때 놓침" });
            self.restart = false;
            self.good_t0 = -1.0;
            self.good_t1 = -1.0;
            self.eng.odom_reset(Tf::IDENTITY);
            self.eng.reset();
            self.sent_nodes.clear();
            self.held_nodes.clear();
            self.held_last = -1;
            self.map_id = 0;
            self.prev_stamp = 0.0;
            self.prev_corr = Tf::IDENTITY;
            self.anchored = true;
            self.good_id = -1;
            self.prov = Tf::IDENTITY;
            self.bases = vec![(0, Tf::IDENTITY)];
            self.refs.clear();
            self.lost_since = -1.0;
            self.fresh = true;
            if !self.writer.push(OutFrame { reset: true, ..Default::default() }) {
                return false;
            }
        }

        let c1 = Instant::now();
        let mut info = self.eng.odom_process(&fr, prepped);
        // IMU 모드 번짐(화소): 이 장면 앞 구간 자이로 각속도 × 노출 × 초점 거리. 크면 다시 잡기 · 잇기 · 지도 넣기를 건너뜀(번진 사진은 어떤 특징으로도 검증되지 않음)
        let blurred = self.imu_on && (self.eng.imu_rate() as f64).to_radians() * self.a.imu_exposure * fr.cam.fx > self.a.imu_max_blur;
        let mut pose = info.pose;
        let mut relocalized = false;
        if pose.is_none() || info.lost {
            pose = None;
            if self.lost_since < 0.0 {
                self.lost_since = fr.t;
            }
            let cr = Instant::now();
            self.eng.ensure_cur(&fr);
            // 기준 장면을 최근 것부터 한 장면에 둘씩 돌아가며 맞춰 봄
            let mut k = 0;
            // 광류 경로는 위치 추적 안에서 키프레임 기술자 맞추기를 이미 함: 최근 기준 장면 맞추기(장면당 수십 ms)는 F2F 에서만
            while self.anchored && !self.klt_path && k < 2 && k < self.refs.len() {
                k += 1;
                let idx = self.refs.len() - 1 - (self.ref_next % self.refs.len());
                self.ref_next += 1;
                let r = &mut self.refs[idx];
                let Some(ri) = self.eng.reloc(&mut r.sig) else { continue };
                let p = r.pose.mul(&ri.t);
                let rt = r.t;
                self.eng.odom_reset(p);
                let cf = self.eng.cur_features();
                let info2 = self.eng.odom_process(&fr, Some(cf));
                if info2.pose.is_none() {
                    continue;
                }
                pose = info2.pose;
                info = info2;
                info.cov = ri.cov;
                info.inliers = ri.inliers;
                relocalized = true;
                self.n_reloc += 1;
                common::info!("장면 {n}: 같은 지도에서 다시 잡음({:.2} 초 전 장면, 맞은 점 {})", fr.t - rt, ri.inliers);
                break;
            }
            // 최근 기준 장면으로 못 잡으면 지도 전체에서(광류 경로, 놓친 동안 장면마다)
            // 장면당 예산: 이 장면에서 위치 추적이 이미 20 ms 넘게 썼으면 다음 장면으로 미룸(추적 스레드가 40 ms 넘게 막히지 않게)
            // 빈틈 바로 뒤 첫 장면은 예산과 상관없이 한 번 꼭 해 봄(이어 붙이기 전에 제자리 찾기)
            if !relocalized && self.klt_path && (budget_ok(c1, 20.0) || !self.global_tried) && !blurred {
                self.global_tried = true;
                if let Some((p, inl, cov)) = self.eng.global_reloc(&fr, self.reloc_min as usize, deadline(c1, 20.0)).filter(|r| {
                    let ok = self.eng.imu_pose_plausible(&r.0);
                    self.n_reloc_tilt += !ok as i64;
                    ok
                }) {
                    self.eng.odom_reset(p);
                    let info2 = self.eng.odom_process(&fr, None);
                    if info2.pose.is_some() {
                        pose = info2.pose;
                        info = info2;
                        info.cov = cov;
                        info.inliers = inl;
                        relocalized = true;
                        self.n_reloc += 1;
                        self.n_global += 1;
                        common::info!("장면 {n}: 지도 전체에서 다시 잡음(맞은 점 {inl})");
                    }
                }
            }
            self.s_reloc.add(ms_since(cr));
            // 기다려도 못 잡으면 마지막 위치에서 다시 시작 → 새 지도(이전 지도와 이어질 때까지 보내지 않음)
            if !relocalized && (!self.anchored || fr.t - self.lost_since > self.reloc_wait) {
                // 믿는 지도가 3 초 몫도 안 되면 붙들 것이 없다: 처음부터 다시
                if self.anchored && self.good_t1 - self.good_t0 < 3.0 {
                    self.restart = true;
                } else {
                    let p = self.eng.odom_pose();
                    self.eng.odom_reset(p);
                }
                self.lost_since = -1.0;
            }
        } else if self.imu_on && self.klt_path && self.anchored && !blurred && self.eng.klt_diag().is_some_and(|d| d.method == 5) && budget_ok(c1, 20.0) {
            // IMU 로 이어 붙이는 동안(영상이 다 실패, 위치 = 마지막 성공 × 자이로 회전 · 이동 0)도 지도 전체에서 다시 잡아 본다:
            // 잡히면 이어 붙인 자세 대신 영상 위치로(이어 붙이기가 1 초 한도를 다 쓰고 새 지도를 여는 일을 줄임)
            let cr = Instant::now();
            if let Some((p, inl, cov)) = self.eng.global_reloc(&fr, self.reloc_min as usize, deadline(c1, 20.0)).filter(|r| self.eng.imu_pose_plausible(&r.0)) {
                self.eng.odom_reset(p);
                let info2 = self.eng.odom_process(&fr, None);
                if info2.pose.is_some() {
                    pose = info2.pose;
                    info = info2;
                    info.cov = cov;
                    info.inliers = inl;
                    relocalized = true;
                    self.n_reloc += 1;
                    self.n_global += 1;
                    self.n_reloc_bridge += 1;
                    common::info!("장면 {n}: IMU 로 이어 붙이던 중 지도 전체에서 다시 잡음(맞은 점 {inl})");
                }
            }
            self.s_reloc.add(ms_since(cr));
        }
        let gap_since_good = fr.t - self.last_good_t; // 마지막으로 위치를 낸 뒤 흐른 시간(녹화·USB 빈틈 포함)
        if pose.is_some() {
            self.lost_since = -1.0;
            self.last_good_t = fr.t;
            self.global_tried = false;
        }
        // 위치 추적 자체가 위치를 못 낸 장면(지도 잇기 여부와 별개)
        let no_pose = pose.is_none();
        // 위치 질: 0 좋음(광류 · 다시 잡기), 1 불확실(ICP · 기술자 되찾기 · 첫 장면 · 이어 붙인 구간 · 이어지지 않은 새 지도), 2 놓침
        let dg = self.eng.klt_diag().unwrap_or_default();
        let imu_rej = self.imu_on && self.eng.imu_rejected();
        self.n_imu_reject += imu_rej as i64;
        self.since_imu = if self.imu_on && pose.is_some() && (dg.method == 5 || imu_rej) { 0 } else { self.since_imu.saturating_add(1) };
        // 지도에 넣지 않을 장면(IMU 모드): 번진 장면(각속도 × 노출 × 초점 거리 > --imu-max-blur 화소) · 영상 근거가 약한 광류 결과
        let fast_rot = self.imu_on && pose.is_some() && blurred;
        let weak = self.imu_on && pose.is_some() && dg.method == 0 && !relocalized && ((dg.q.inliers as f64) < self.a.imu_min_inliers || (dg.q.ratio as f64) < self.a.imu_min_ratio);
        self.n_fast_rot += fast_rot as i64;
        self.n_weak += (weak && !fast_rot) as i64;
        let imu_hold = self.imu_on && pose.is_some() && (self.since_imu <= IMU_SETTLE || fast_rot || weak);
        self.n_imu_hold += imu_hold as i64;
        let quality = if pose.is_none() {
            2u8
        } else if relocalized {
            0
        } else if dg.method != 0 || !self.anchored || self.eng.prov_from > 0 || imu_hold {
            1
        } else {
            0
        };
        self.nq[quality as usize] += 1;
        let odom_ms = ms_since(c1);

        // 새 지도는 위치 추적이 실제로 다시 시작한 장면(위치가 채워져 옴)에서만 연다
        let mut new_map = false;
        // 이어 붙이기는 그사이 움직임을 아는 경우만: 깊이 ICP 로 자리를 확인했거나 빈틈이 1 초 이하.
        // 그보다 길면(사람이 그사이 걸었을 수 있음) 새 지도를 열고 루프 클로저가 옛 지도와 잇게 둔다(틀린 이어 붙이기를 거대한 보정으로 끌어오지 않게)
        if self.bridge && !self.fresh && !relocalized && pose.is_some() && info.cov[0] >= 9999.0 && (self.eng.bridge_verified() || gap_since_good <= self.a.bridge_gap) {
            // 이어 붙이기: 위치 추적은 마지막 위치에서 다시 시작했다. 같은 지도에 큰 불확실성(0.5 m², 0.5 rad²)으로 이어
            // 지도가 끊기지 않게 하고, 실제로 어긋났으면 뒤의 루프 클로저가 그래프 최적화로 바로잡는다
            info.cov = [0.0; 36];
            for i in 0..6 {
                info.cov[i * 7] = 0.5;
            }
            self.n_bridge += 1;
            if !self.eng.bridge_verified() && gap_since_good > 0.5 {
                // 확인 안 된 이어 붙이기: 다음 10 초 동안 옛 노드에서 제자리를 찾아 루프로 이 구간을 바로잡음
                self.eng.prov_from = self.eng.last_node() + 1;
                self.bridge_t = fr.t;
            }
            if self.eng.bridge_verified() {
                // 깊이 ICP 로 그사이 움직임을 잡았다: 보통 위치 추적처럼 작은 불확실성
                for i in 0..6 {
                    info.cov[i * 7] = 1e-3;
                }
                self.n_bridge_ok += 1;
            }
        }
        if !self.fresh && !relocalized && pose.is_some() && info.cov[0] >= 9999.0 {
            common::info!("장면 {n}: 위치 추적 다시 시작 → 새 지도");
            self.eng.map_start = self.eng.last_node() + 1;
            let id = self.eng.trigger_new_map();
            self.map_id = if id >= 0 { id as u32 } else { self.map_id + 1 };
            new_map = true;
            self.held_last = -1;
            if self.good_id > 0 {
                self.anchored = false;
            }
            self.n_newmap += 1;
            self.newmap_t = fr.t;
            // 앞의 믿는 지도가 작으면(시작 직후) 맡겨 두지 않고 바로 새 지도로
            self.prev_map_small = ((self.sent_nodes.len() - self.held_nodes.len()) as f64) < self.a.imu_hold_min_nodes;
        }
        self.fresh = false;
        let mut process = true;
        // 확인 안 된 이어 붙이기 뒤 10 초: 장면마다(0.2 초 간격, 장면 예산 안) 옛 노드에서 제자리를 찾고, 찾으면 이 장면을 바로 지도 노드로 넣어
        // 그 옛 노드와 루프를 건다(그래프 최적화가 이어 붙인 구간 전체를 끌어옴)
        let fixing = self.anchored && self.eng.prov_from > 0 && fr.t - self.bridge_t < 10.0;
        if self.anchored && self.eng.prov_from > 0 && !fixing {
            self.eng.prov_from = 0; // 10 초 동안 못 찾음: 이어 붙인 그대로 둠
        }
        if self.anchored {
            self.eng.map_start = 0;
        }
        let mut force_node = false;
        // 이어지지 않은 새 지도 · 이어 붙인 구간: 장면마다(0.2 초 간격, 장면 예산 안) 앞 지도들의 노드에서 이을 후보를 찾고, 찾으면 이 장면을 바로 노드로
        if (fixing || !self.anchored) && !imu_hold && self.klt_path && pose.is_some() && fr.t - self.last_merge_try >= if self.imu_on || fr.t - self.newmap_t < MERGE_FAST { 0.0 } else { 0.2 } && budget_ok(c1, 25.0) {
            self.last_merge_try = fr.t;
            if self.eng.propose_merge(&fr, self.reloc_min as usize, deadline(c1, 25.0)) {
                self.n_merge_try += 1;
                force_node = true;
            }
        }
        let det_rate = if self.anchored || self.klt_path { self.detection_rate } else { self.detection_rate.max(4.0) }; // 이어지기를 기다리는 동안은 더 자주 찾아봄
        if !force_node && det_rate > 0.0 && self.prev_stamp > 0.0 && fr.t > self.prev_stamp && fr.t - self.prev_stamp < 1.0 / det_rate as f64 {
            process = false;
        }
        if imu_hold {
            process = false; // 위치만: 지도 노드로 넣지 않음
        }
        if process {
            self.prev_stamp = fr.t;
        }
        let mut data_id = (self.n + 1) as i32;
        if !process {
            data_id = -1; // 중간 장면(지도 노드 아님)
            process = self.intermediate;
        }
        let c2 = Instant::now();
        let mut slam_ms = -1.0;
        let mut lp = BTreeMap::new();
        let mut loop_ = false;
        if process {
            self.eng.trusted = self.anchored;
            // 이어지지 않은 새 지도: 노드를 넣을 때마다 믿는 지도 노드 전체에서 이어 줄 후보를 찾아 루프 후보로 넘김
            // 잇기 후보 찾기는 1.5 초에 한 번, 이 장면 위치 추적이 가벼웠을 때만

            self.eng.slam_process(&fr, data_id > 0, pose, &info.cov, info.velocity());
            slam_ms = ms_since(c2);
            loop_ = self.eng.loop_closure_id() > 0;
            self.n_loop += loop_ as i64;
            if loop_ && self.eng.prov_from > 0 && self.anchored {
                self.n_bridge_fix += 1;
                common::info!("장면 {n}: 이어 붙인 구간을 옛 지도 루프로 바로잡음");
                self.eng.prov_from = 0;
            }
            let (mc, rj, mall) = self.eng.correction_info();
            self.n_loop_rejected += rj as i64;
            if loop_ || rj > 0 {
                self.max_corr.0 = self.max_corr.0.max(mc.0);
                self.max_corr.1 = self.max_corr.1.max(mc.1);
                common::info!("장면 {n}: 루프 클로저 {} (상대 지도 노드 최대 보정 {:.2} m · {:.1}°, 옮겨진 노드 최대 {:.2} m)", if rj > 0 { "버림" } else { "받음" }, mc.0, mc.1.to_degrees(), mall);
            }
            lp = self.eng.local_optimized().clone();
            let last = self.eng.last_location_id();
            if !self.anchored && lp.contains_key(&self.good_id) {
                self.anchored = true;
                common::info!("장면 {n}: 이전 지도와 다시 이어짐");
            }
            if self.anchored && lp.contains_key(&last) {
                self.good_id = last;
            }
            if !self.anchored && data_id > 0 && lp.contains_key(&last) {
                self.held_last = last;
            }
        }
        // 기준 장면: 믿는 위치의 장면을 0.4 초마다 하나씩, 최근 6 개
        if self.anchored && pose.is_some() && !new_map {
            if self.good_t0 < 0.0 {
                self.good_t0 = fr.t;
            }
            self.good_t1 = fr.t;
        }
        if self.anchored && !new_map {
            if let Some(p) = pose {
                if self.refs.back().is_none_or(|r| fr.t - r.t >= 0.4) {
                    self.refs.push_back(Ref { sig: self.eng.make_ref(&mut fr), pose: p, t: fr.t });
                    if self.refs.len() > 6 {
                        self.refs.pop_front();
                    }
                }
            }
        }
        // 이번 장면의 지도 위치 = 지도 보정 × 오도메트리 위치 × 광학 축
        let corr = self.eng.map_correction();
        let mut corrected = false;
        if let Some(c) = &corr {
            corrected = (0..12).any(|i| (c.0[i] - self.prev_corr.0[i]).abs() > 1e-6);
            if corrected {
                self.prev_corr = *c;
                self.n_corr += 1;
            }
        }
        // IMU 모드: 이어지지 않은 새 지도는 위치는 임시 위치로 바로 보내되 지도 · 물체에는 맡겨 두기만(이어지면 다시 쌓기가 고친 위치로 넣음).
        // 놓친 사이 움직임을 모르는 임시 위치로 바로 쌓으면 같은 표면이 어긋난 채 두 겹으로 남는다
        // 맡겨 두기: 기본은 이어질 때까지(--imu-hold-wait 초를 주면 그만큼 기다린 뒤, 앞 지도가 작으면 바로) 새 지도를 받아들여 쌓는다(맡겼던 장면도 그 지도 위치로).
        // 나중에 이어지면 노드가 움직여 다시 쌓기가 고친 위치로 옮긴다
        let accept = self.prev_map_small || fr.t - self.newmap_t > self.a.imu_hold_wait;
        if !self.anchored && new_map {
            // 새 지도 첫 장면 = 마지막으로 보낸 위치(그사이 움직임 모름): 표시 변환을 그에 맞춤
            if let (Some((w, po)), Some(p)) = (self.last_sent_world.zip(self.last_sent_odom), pose.as_ref()) {
                let a = corr.unwrap_or(Tf::IDENTITY).mul(p);
                self.prov = if self.eng.gravity_aligned() {
                    // 중력으로 맞춘 지도(IMU): 자리는 마지막으로 보낸 자리, 자세는 위치 추적이 놓친 사이 이어 온 그대로(자이로 회전 · 중력 맞춤).
                    // 표시 변환은 수직축 회전(방위) + 이동만: 새 지도가 놓친 사이 카메라가 돈 만큼 기운 채 쌓이지 않는다
                    let m = w.mul(&po.inverse()).mul(&corr.unwrap_or(Tf::IDENTITY).inverse());
                    let yaw = (m.0[4] - m.0[1]).atan2(m.0[0] + m.0[5]);
                    let (sn, cs) = yaw.sin_cos();
                    let rz = Tf([cs, -sn, 0.0, 0.0, sn, cs, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                    let ra = rz.mul(&a);
                    Tf([cs, -sn, 0.0, w.x() - ra.x(), sn, cs, 0.0, w.y() - ra.y(), 0.0, 0.0, 1.0, w.z() - ra.z()])
                } else {
                    w.mul(&a.inverse())
                };
            }
        }
        // 받아들임: 이 새 지도가 이제 믿는 지도다(앞 지도가 작았거나 기다릴 만큼 기다림). 표시 변환은 이어 붙인 그대로 두고 바로 쌓으며,
        // 뒤에 놓치면 이 지도에 다시 이어져야 한다. 믿는 지도 노드(good_id)는 이 지도의 첫 노드부터 다시 잡는다
        if self.imu_on && !self.anchored && accept {
            self.n_accept += 1;
            common::info!("장면 {n}: 이어지지 않은 새 지도를 받아들임({:.1} 초 기다림) → 믿는 지도로 바로 쌓기", fr.t - self.newmap_t);
            self.bases.push((self.eng.map_start, self.prov));
            self.anchored = true;
            self.good_id = -1;
        }
        if self.anchored {
            // 믿는 지도의 표시 변환 = 이어진 노드 가운데 가장 오래된 노드가 속한 지도의 것(그래프 좌표가 그 노드에 고정됨)
            if let Some(&lo) = lp.keys().find(|&&k| k > 0) {
                self.prov = self.bases.iter().rev().find(|b| b.0 <= lo).map_or(Tf::IDENTITY, |b| b.1);
            }
        }
        let map_hold = self.imu_on && !self.anchored;
        let hold_g = map_hold;
        let lost = pose.is_none() || info.lost;
        self.n_lost += lost as i64;

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
                let pm = self.prov.mul(p);
                let was_held = self.held_nodes.contains_key(&id);
                let sent = self.sent_nodes.get(&id);
                if !self.anchored && sent.is_some() && !was_held && !self.prov_nodes.contains(&id) {
                    continue; // 믿는 노드는 이어지지 않은 지도의 값으로 덮지 않음
                }
                if !self.anchored {
                    self.prov_nodes.insert(id);
                } else if self.anchored {
                    self.prov_nodes.remove(&id);
                }
                if let Some(s) = sent {
                    if was_held == hold_g {
                        let dmax = (0..12).map(|i| (s.0[i] - pm.0[i]).abs()).fold(0f32, f32::max);
                        if dmax < 1e-3 {
                            continue;
                        }
                    }
                }
                self.sent_nodes.insert(id, pm);
                if hold_g && !was_held {
                    self.held_nodes.insert(id, 0);
                }
                if !hold_g && was_held {
                    self.n_joined += self.held_nodes.remove(&id).unwrap();
                }
                let tn = pm.mul(&self.optical);
                upd.push(GraphNode { id: id as u32, held: hold_g as u32, t: tn.0.map(|v| v as f64) });
            }
            if !upd.is_empty() {
                self.n_graph_nodes += upd.len() as i64;
                self.n_graph += 1;
                if !self.writer.push(OutFrame { graph: upd, ..Default::default() }) {
                    common::info!("받는 쪽이 닫힘: 끝냄");
                    return false;
                }
            }
        }

        // 맡길 장면: 새 지도에서 위치를 구했고 그 지도의 노드를 이미 보냈을 때
        let held = hold_g && pose.is_some() && !info.lost && !new_map && self.held_nodes.contains_key(&self.held_last);
        let mut of = OutFrame {
            flags: if lost { stream::LOST } else { 0 } | if new_map { stream::NEW_MAP } else { 0 } | if corrected { stream::MAP_CORRECTED } else { 0 } | if held { stream::HELD } else { 0 } | if self.eng.gravity_aligned() { stream::GRAVITY } else { 0 } | if imu_hold || (map_hold && !held) { stream::NO_MAP } else { 0 },
            map_id: self.map_id,
            seq: self.n as u64,
            t: fr.t,
            ..Default::default()
        };
        if !lost || held {
            let body = self.prov.mul(&corr.unwrap_or(Tf::IDENTITY).mul(pose.as_ref().unwrap()));
            if !lost {
                self.last_sent_world = Some(body);
                self.last_sent_odom = pose;
                if self.imu_on {
                    let p = [body.0[3], body.0[7], body.0[11]];
                    match self.fast_prev {
                        Some((t0, p0)) if fr.t - t0 > 0.0 && fr.t - t0 < 0.2 && !new_map => {
                            let dt = (fr.t - t0) as f32;
                            for i in 0..3 {
                                self.fast_vel[i] = (0.5 * self.fast_vel[i] + 0.5 * (p[i] - p0[i]) / dt).clamp(-2.0, 2.0);
                            }
                        }
                        _ => self.fast_vel = [0.0; 3],
                    }
                    self.fast_prev = Some((fr.t, p));
                    self.eng.imu_anchor(fr.t, body, self.optical, self.fast_vel);
                    self.tilt_map = self.eng.imu_tilt(&body).unwrap_or(-1.0);
                    self.tilt_map_max = self.tilt_map_max.max(self.tilt_map);
                }
                if !self.anchored {
                    of.flags |= stream::PROVISIONAL;
                    self.n_prov += 1;
                }
                if quality == 1 {
                    of.flags |= stream::UNCERTAIN;
                }
            }
            let twc = body.mul(&self.optical);
            of.pose = twc.0.map(|v| v as f64);
            if !lost {
                // 위치 파일엔 믿는 위치만(임시 위치는 화면·지도용)
                if let (Some(f), true) = (self.poses_f.as_mut(), self.anchored) {
                    write_tum(f, fr.t, &twc);
                }
                // 기준 노드: 마지막 지도 노드(이 장면 위치 = 그 노드 위치 × 오도메트리 상대 위치)
                let last_id = self.eng.last_location_id();
                of.anchor = if self.sent_nodes.contains_key(&last_id) { last_id as u32 } else { 0 };
            }
            if held {
                self.n_map_held += map_hold as i64;
                of.anchor = self.held_last as u32;
                if self.n % self.a.send_every == 0 {
                    *self.held_nodes.get_mut(&self.held_last).unwrap() += 1;
                }
                self.n_held += 1;
            }
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
            // 장면 버퍼를 복사 없이 넘김(기준 장면은 make_ref 가 필요한 것을 이미 가져감)
            of.color = std::mem::take(&mut fr.color);
            of.color_is_rgb = fr.color_is_rgb;
            of.depth = fr.take_stream_depth();
            if !self.writer.push(of) {
                common::info!("받는 쪽이 닫힘: 끝냄");
                return false;
            }
            send_ms = ms_since(c3); // 큐에 넣기(기다림 포함)
        }
        let total_ms = ms_since(c0);
        self.s_read.add(read_ms);
        if self.n > 0 {
            self.s_odom.add(odom_ms); // 첫 장면(초기화)은 뺌
        }
        if slam_ms >= 0.0 && self.n > 0 {
            self.s_slam.add(slam_ms);
        }
        self.s_send.add(send_ms);
        self.s_total.add(total_ms);
        if let Some(f) = self.stats_f.as_mut() {
            let _ = write!(
                f,
                "{n},{:.6},{read_ms:.3},{odom_ms:.3},{slam_ms:.3},{send_ms:.3},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
                fr.t, lost as i32, new_map as i32, corrected as i32, loop_ as i32, info.features, info.inliers, held as i32, no_pose as i32, quality, dg.method, dg.q.inliers, dg.q.ratio, dg.q.rms, dg.q.spread, dg.jump
            );
            // IMU 모드만 끝 칸(지도 위와 IMU 위 사이각)
            if let Some((tilt, ..)) = self.eng.imu_diag() {
                let _ = write!(f, ",{tilt:.2},{},{:.2}", imu_hold as i32, self.tilt_map);
            }
            let _ = writeln!(f);
        }
        self.n += 1;
            true
    }

    /// 끝: 출력 닫기 · 끝 그래프 · 요약 줄. src_dropped: 입력에서 밀려 버린 장면 수
    pub fn finish(self, src_dropped: usize) -> Result<i32, String> {
        let Runner { t_run, in_proc, calib, imu_replay, imu_ring, imu_on, skipped, stress_counts, replay_max, intermediate, detection_rate, reloc_min, mut eng, klt_path, reloc_wait, bridge, n_bridge, n_global, n_reloc_bridge, global_tried, n_reloc_tilt, n_loop_rejected, n_merge_try, nq, bridge_t, n_bridge_fix, last_merge_try, newmap_t, prev_map_small, n_accept, max_corr, last_good_t, n_bridge_ok, since_imu, n_imu_hold, n_imu_reject, n_fast_rot, n_weak, n_map_held, refs, ref_next, lost_since, good_t0, good_t1, restart, n_reloc, s_reloc, mut writer, poses_f, stats_f, optical, s_read, s_odom, s_slam, s_send, s_total, n, n_lost, n_newmap, n_loop, n_corr, n_graph, n_graph_nodes, sent_nodes, held_nodes, held_last, n_held, n_joined, map_id, fresh, anchored, good_id, prev_stamp, prev_corr, prov, bases, prov_nodes, last_sent_world, last_sent_odom, fast_prev, fast_vel, tilt_map, tilt_map_max, n_prov, a } = self;
        let _ = (&in_proc, &calib, &imu_replay, &imu_on, &replay_max, &intermediate, &detection_rate, &reloc_min, &klt_path, &reloc_wait, &n_bridge, &global_tried, &bridge_t,
                 &last_merge_try, &newmap_t, &prev_map_small, &last_good_t, &since_imu,
                 &refs, &ref_next, &lost_since, &good_t0, &good_t1, &restart, &sent_nodes, &held_nodes, &held_last, &map_id, &fresh, &anchored, &good_id, &prev_stamp, &prev_corr,
                 &prov, &bases, &prov_nodes, &last_sent_world, &last_sent_odom, &fast_prev, &fast_vel, &a);
        let wall = t_run.elapsed().as_secs_f64();
        if let Some(r) = &imu_ring {
            r.closed.store(true, Ordering::Relaxed); // 읽기 · 재생 · 위치 내기 스레드 끝냄
        }
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
            common::info!("놓친 뒤 같은 지도로 이어 붙임 {n_bridge} 번(깊이 ICP 로 자리 확인 {n_bridge_ok} 번, 루프로 바로잡음 {n_bridge_fix} 번), 지도 전체에서 다시 잡음 {n_global} 번(IMU 로 이어 붙이던 중 {n_reloc_bridge} 번)");
        }
        {
            common::info!("루프 클로저 받음 {n_loop} · 보정이 커서 버림 {n_loop_rejected}, 살펴본 보정 최대 {:.2} m · {:.1}°", max_corr.0, max_corr.1.to_degrees());
            let l = &eng.lc_stats;
            if l.sent > 0 {
                let d = l.done.max(1) as f64;
                common::info!(
                    "신경망 루프: 노드 {} 개 보냄 · {} 개 끝남, 검증 {} 번, 맞음 {} · 위치 추적과 같아 버림 {} · 간격 안이라 건너뜀 {} · 지도에 들어감 {}, ms/노드 MegaLoc {:.1}(최대 {:.1}) · XFeat {:.1}(최대 {:.1}) · 짝 + 검증 {:.1}(최대 {:.1})",
                    l.sent, l.done, l.tried, l.hits, l.redundant, l.gap, l.accepted, l.ms[0] / d, l.ms_max[0], l.ms[1] / d, l.ms_max[1], l.ms[2] / d, l.ms_max[2]
                );
            }
        }
        common::info!("장면 {n} 개, {wall:.1} 초({:.1} 장/초), 놓침 {n_lost}, 새 지도 {n_newmap}, 루프 클로저 {n_loop}, 지도 보정 바뀜 {n_corr}", n as f64 / wall.max(1e-9));
        common::info!("위치 질: 좋음 {} · 불확실 {} · 놓침 {}", nq[0], nq[1], nq[2]);
        if let Some(s) = eng.imu_summary() {
            common::info!("{s}");
            common::info!("IMU: 영상이 IMU 회전과 어긋나 버린 장면 {n_imu_reject}, 위치만 보내고 지도에 안 쌓은 장면 {n_imu_hold}(이어 붙임 · 어긋남 뒤 {IMU_SETTLE} 장 포함, 번짐 {n_fast_rot} · 영상 근거 약함 {n_weak}), 이어지지 않은 새 지도라 맡겨 둔 장면 {n_map_held}, 기다리다 받아들인 새 지도 {n_accept}");
            common::info!("IMU: 보낸 위치 기준 지도 기울기 마지막 {tilt_map:.1}° · 최대 {tilt_map_max:.1}°, 장면 사이 위치 {} 개({:.0} 개/초)", crate::imu::FAST_N.load(Ordering::Relaxed), crate::imu::FAST_N.load(Ordering::Relaxed) as f64 / wall.max(1e-9));
            if let Some((ng, gd, nr)) = eng.imu_grav() {
                common::info!("IMU: 기울기를 중력 쪽으로 고친 장면 {ng}(합 {gd:.1}°, 다시 시작한 자리에서 한 번에 맞춤 {nr}), 중력과 어긋나 버린 다시 잡기 {n_reloc_tilt}");
            }
        }
        {
            common::info!("이어지지 않은 새 지도에서 임시 위치로 바로 보낸 장면 {n_prov}, 지도 전체에서 찾은 잇기 후보 {n_merge_try} 번");
        }
        common::info!("놓침 가운데 새 지도에서 맡긴 장면 {n_held}(보낸 것 가운데 나중에 이어져 지도에 들어간 장면 {n_joined})");
        common::info!("같은 지도에서 다시 잡음 {n_reloc} 번, ms/회 {}", s_reloc.str());
        common::info!("루프 정합에서 단어 id 가 겹치지 않는 노드(원본은 id 순서를 씀) {} 번, 다시 잡기에서 원본이 고쳐 둔 기준 장면을 다시 씀 {} 번(두 번째 판 순서를 모름 {} 번)", eng.loop_unique_ids.load(Ordering::Relaxed), eng.ref_reuse, eng.ref_unknown);
        if let Some(s) = skipped {
            common::info!("재생에서 밀려 건너뛴 장면 {}", s.load(Ordering::Relaxed));
        }
        if let Some((d, b)) = &stress_counts {
            common::info!("시험 조건: 뺀 장면 {}, 흐리게 한 장면 {}", d.load(Ordering::Relaxed), b.load(Ordering::Relaxed));
        }
        if src_dropped > 0 {
            common::info!("위치 추적이 밀려 버린 장면 {src_dropped}");
        }
        common::info!("보냄 {}, 밀려 버림 {}, 그래프 블록 {n_graph} 개(노드 {n_graph_nodes} 개)", writer.written(), writer.dropped());
        common::info!("ms/장  읽기    {}", s_read.str());
        common::info!("ms/장  위치추적 {}", s_odom.str());
        common::info!("ms/회  지도갱신 {}", s_slam.str());
        common::info!("ms/장  큐넣기  {}", s_send.str());
        common::info!("ms/장  파이프쓰기(쓰기 스레드) {}", s_write.str());
        common::info!("ms/장  합(본 스레드) {}", s_total.str());
        Ok(0)
    }
}

/// 따로 도는 위치 프로그램(`ondevice slam`): 입력을 읽기 스레드(PrefetchSource)로 돌리며 장면마다 step
fn run_in(a: Args, pool: Option<std::sync::Arc<rayon::ThreadPool>>) -> Result<i32, String> {
    let (max, replay_max) = (a.max_frames, !a.tum.is_empty() && a.rate == 0.0);
    let (mut r, src, prep) = Runner::new(a, false, pool)?;
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

/// 장면당 시간 예산(다시 잡기 20 ms · 잇기 25 ms, 벽시계) 안인가. 시험용 ONDEVICE_NO_BUDGET=1 이면 늘 참:
/// 최대 속도 재생의 결과가 PC 부하와 상관없이 같게(바이트 비교)
/// 장면 시작 c 에서 ms 뒤 시각(시험용 ONDEVICE_NO_BUDGET=1 이면 끝없음)
fn deadline(c: Instant, ms: f64) -> Instant {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var_os("ONDEVICE_NO_BUDGET").is_some()) {
        c + std::time::Duration::from_secs(3600)
    } else {
        c + std::time::Duration::from_secs_f64(ms * 1e-3)
    }
}

fn budget_ok(c: Instant, ms: f64) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ONDEVICE_NO_BUDGET").is_some()) || ms_since(c) < ms
}
