//! Rust System 뼈대: 원본 `System::TrackRGBD` 흐름을 옮겨진 부품으로 엮는다.
//! 설정 · 입력 변환(`orb3cv::system`) → ORB 추출(`orb3rs::extract`) → Frame(`orb3rs::frame`) → 추적 흐름(`orb3rs::track::track_flow`)
//! → 키프레임은 지도 스레드(`orb3sys::mapping`, Rust 지역 BA) → 루프 대기열(`orb3sys::loopflow` 자리).
//!
//! 아직 Rust 판이 없는 단계는 [`Hole`] 로 이름을 남기고, 지나가면 그 이름으로 멈춘다(원본 FFI 로 대신 부르지 않음 — 지도 자료가 따로라
//! 바꿔 끼우는 단위가 System 통째이기 때문, `mod.rs` 머리글). 구멍 목록은 [`HOLES`]. 이 판이 [`Orb3System`] 를 구현하므로
//! 구멍이 다 메워지면 `orb3_backend` 가 `RustSystem` 을 그대로 연다.
use super::mapping::{self, KfSteps, MapState, MappingStats};
use super::graft::{self, GraftState};
use super::kfgraph::AtlasGraph;
use super::detectops::RsLoop;
use super::gbaops::RsGba;
use super::loopflow::{LoopFlow, LoopStats};
use super::threads::{LoopQueue, MapUpdate, Mapper};
use crate::orb3loop::detect::RustSim3;
use super::{Orb3System, Track};
use crate::orb3cv::system::{self as cvsys, SystemInit};
use crate::orb3rs::extract::Extractor;
use crate::orb3rs::track::{self, state, MapperStatus, NeedKfCtx, Tracker, TrackOps};
use crate::orb3loop::detect::best_covisibles;
use crate::orb3rs::localmap::{self, NewMp, TriParams};
use crate::orb3rs::matcher::Matcher;
use crate::orb3loop::crand::GlibcRand;
use crate::orb3loop::dbow2::{BowVector, Vocabulary};
use crate::orb3loop::kfdb::{KeyFrameDatabase, KfId as DbKf, MapId as DbMap};
use crate::orb3loop::reloc::{relocalize, RustPnp};
use crate::orb3rs::stdsort;
use crate::orb3rs::types::{quat_from_mat, Atlas, Frame, KeyFrame, KfId, Map, MapId, MapPoint, MpId, Se3};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Rust 판이 아직 없는 단계(이름 = 원본 함수)
pub const HOLES: &[&str] = &[];

/// 옮겨지지 않은 단계를 지나면 그 이름으로 멈춤
#[track_caller]
pub fn hole(name: &str) -> ! {
    panic!("Rust System: 아직 옮기지 않은 단계 {name}")
}

/// 지도 스레드 키프레임 단계: 포터 TRACK 판(MapPointCulling · CreateNewMapPoints · KeyFrameCulling, 원본 덤프와 비트 동일)을 엮음.
/// ProcessNewKeyFrame · SearchInNeighbors 도 포터 TRACK 판(UpdateNormalAndDepth · UpdateConnections 는 덤프와 비트 동일, 순회는 id 순).
pub struct RsSteps {
    pub mapper: Arc<Mapper<KfId>>,
    /// mlpRecentAddedMapPoints
    pub recent: Vec<MpId>,
    pub tri: TriParams,
    pub th_depth: f32,
    /// 마지막으로 본 지도 스레드 비우기 수(바뀌면 mlpRecentAddedMapPoints 비움 = ResetIfRequested)
    pub reset_seen: u64,
    /// 단어장(ComputeBoW). set_vocabulary 가 채움
    pub voc: Arc<std::sync::Mutex<Option<Arc<Vocabulary>>>>,
    /// KeyFrame::SetBadFlag 의 mpKeyFrameDB->erase(this) 자리
    pub kfdb: Arc<std::sync::Mutex<Option<KeyFrameDatabase>>>,
}

impl RsSteps {
    fn reset_if_requested(&mut self) {
        let g = self.mapper.reset_gen();
        if g != self.reset_seen {
            self.reset_seen = g;
            self.recent.clear();
        }
    }
}

impl KfSteps for RsSteps {
    fn process_new_keyframe(&mut self, st: &mut MapState, kf: KfId) {
        // KeyFrame::ComputeBoW
        if st.atlas.kfs[&kf].bow.is_empty() {
            let voc = self.voc.lock().unwrap().clone().unwrap_or_else(|| hole("단어장(set_vocabulary) 없이 ComputeBoW"));
            let (bv, fv) = voc.transform(&st.atlas.kfs[&kf].desc, 4);
            let k = st.atlas.kfs.get_mut(&kf).unwrap();
            k.bow = bv.into_iter().collect();
            k.feat = fv.into_iter().collect();
        }
        localmap::process_new_keyframe(&mut st.atlas, kf, &mut self.recent);
    }
    fn map_point_culling(&mut self, st: &mut MapState, kf: KfId) {
        self.reset_if_requested();
        localmap::map_point_culling(&mut self.recent, kf, false, &mut st.atlas, &mut st.ext);
    }
    fn create_new_map_points(&mut self, st: &mut MapState, kf: KfId) {
        let mapper = Arc::clone(&self.mapper);
        create_new_map_points_into(&mut st.atlas, kf, &self.tri, &mut self.recent, &mut || mapper.check_new_keyframes());
    }
    /// LocalMapping::SearchInNeighbors(단안 · 관성 아님). mnFuseTargetForKF · mnFuseCandidateForKF 표시는 이 호출 안의 집합
    fn search_in_neighbors(&mut self, st: &mut MapState, kf: KfId) {
        let abort = || self.mapper.abort_ba().load(std::sync::atomic::Ordering::SeqCst);
        localmap::search_in_neighbors(&mut st.atlas, kf, &abort);
    }
    fn keyframe_culling(&mut self, st: &mut MapState, kf: KfId) {
        // 원본 KeyFrameCulling 첫 줄: mpCurrentKeyFrame->UpdateBestCovisibles()(15 미만 연결까지 순서 목록에 들어감)
        crate::orb3rs::mapdata::kf_update_best_covisibles(&mut st.atlas, kf);
        let local: Vec<KfId> = st.atlas.kfs[&kf].ordered_conn.iter().map(|c| c.0).collect();
        let th = self.th_depth;
        let abort = &self.mapper;
        localmap::keyframe_culling(&local, false, &|_| th, &|| abort.abort_ba().load(std::sync::atomic::Ordering::SeqCst), &mut st.atlas, &mut st.ext);
        // SetBadFlag 안의 KeyFrameDatabase::erase(지운 키프레임을 역색인에서 뺌)
        let gone: Vec<KfId> = local.iter().copied().filter(|k| st.atlas.kfs.get(k).is_some_and(|x| x.bad)).collect();
        if !gone.is_empty() {
            if let Some(db) = self.kfdb.lock().unwrap().as_mut() {
                for k in gone {
                    let bow: BowVector = st.atlas.kfs[&k].bow.iter().map(|(w, v)| (*w, *v)).collect();
                    db.erase(k as DbKf, &bow);
                }
            }
        }
    }
}

/// LocalMapping::CreateNewMapPoints(RGB-D) 전체: 짝 찾기 · 삼각측량(포터 TRACK) + 새 점 넣기(MapPoint(x3D, KF, 지도) ·
/// AddObservation 둘 · AddMapPoint 둘 · ComputeDistinctiveDescriptors · UpdateNormalAndDepth · Atlas::AddMapPoint · mlpRecentAddedMapPoints).
/// `check_new` = CheckNewKeyFrames. 돌려줌: 새 점들(순서대로)
pub fn create_new_map_points_into(atlas: &mut Atlas, kf: KfId, tri: &TriParams, recent: &mut Vec<MpId>, check_new: &mut dyn FnMut() -> bool) -> Vec<MpId> {
    let cur = atlas.kfs[&kf].clone();
    let nb: Vec<KeyFrame> = best_covisibles(&cur, 10).iter().map(|k| atlas.kfs[k].clone()).collect();
    let refs: Vec<&KeyFrame> = nb.iter().collect();
    let m = Matcher::new(0.6, false);
    let mut new = Vec::new();
    // 원본은 이웃마다 짝을 찾은 뒤 새 점을 바로 지도에 넣는다(다음 이웃 짝 찾기가 그 점을 봄)
    let atlas = std::cell::RefCell::new(atlas);
    localmap::create_new_map_points(
        &cur,
        &refs,
        tri,
        &mut |k2| {
            let a = atlas.borrow();
            m.search_for_triangulation(&a.kfs[&kf], &a.kfs[&k2], false, false)
        },
        check_new,
        &mut |n: NewMp| {
            let mut a = atlas.borrow_mut();
            let id = a.next_mp;
            a.next_mp += 1;
            let (map, frame) = (a.kfs[&kf].map, a.kfs[&kf].frame_id);
            let mut mp = MapPoint { id, map, pos: n.pos, first_frame: frame, first_kf: kf, ref_kf: kf, visible: 1, found: 1, ..Default::default() };
            mp.add_observation(&a.kfs[&kf], n.idx1);
            mp.add_observation(&a.kfs[&n.kf2], n.idx2);
            a.kfs.get_mut(&kf).unwrap().map_points[n.idx1] = Some(id);
            a.kfs.get_mut(&n.kf2).unwrap().map_points[n.idx2] = Some(id);
            mp.compute_distinctive_descriptors(&a.kfs);
            a.mps.insert(id, mp);
            crate::orb3rs::mapdata::mp_update_normal_and_depth(&mut a, id);
            if let Some(mi) = a.maps.iter_mut().find(|x| x.id == map) {
                mi.add_map_point(id);
            }
            recent.push(id);
            new.push(id);
        },
    );
    new
}

impl MapperStatus for Mapper<KfId> {
    fn is_stopped(&self) -> bool {
        Mapper::is_stopped(self)
    }
    fn stop_requested(&self) -> bool {
        Mapper::stop_requested(self)
    }
    fn accept_keyframes(&self) -> bool {
        Mapper::accept_keyframes(self)
    }
    fn is_initializing(&self) -> bool {
        false
    }
    fn keyframes_in_queue(&self) -> i32 {
        self.queue_len() as i32
    }
    fn interrupt_ba(&self) {
        self.abort_ba().store(true, std::sync::atomic::Ordering::SeqCst)
    }
}

pub struct RustSystem {
    init: SystemInit,
    extractor: Extractor,
    tracker: Tracker,
    map: Arc<MapUpdate<MapState>>,
    mapper: Arc<Mapper<KfId>>,
    loopq: Arc<LoopQueue<KfId>>,
    mapping: Option<JoinHandle<MappingStats>>,
    next_frame: u64,
    cur: Option<Frame>,
    last: Option<Frame>,
    changed: bool,
    gray: Vec<u8>,
    depth: Vec<f32>,
    /// Atlas::mnLastInitKFidMap
    last_init_kf: KfId,
    /// Tracking::mnLastInitFrameId · mnInitialFrameId
    last_init_frame_id: u64,
    initial_frame_id: u64,
    /// 단어장(재위치 · BoW). 없으면 재위치 단계에서 멈춤
    voc: Option<Arc<Vocabulary>>,
    voc_steps: Arc<std::sync::Mutex<Option<Arc<Vocabulary>>>>,
    /// KeyFrameDatabase(키프레임 넣기는 루프 스레드 몫)
    kfdb: Arc<std::sync::Mutex<Option<KeyFrameDatabase>>>,
    /// 원본 glibc rand(씨앗 1): Sim3Solver · MLPnP 가 한 수열을 나눠 씀
    /// 루프 스레드(`RsLoop`)와 같은 하나를 나눠 씀(원본 rand() 는 프로세스 하나의 수열)
    rand: Arc<std::sync::Mutex<GlibcRand>>,
    /// LoopClosing 스레드 · 전역 BA
    flow: Arc<LoopFlow<MapState, RsGba>>,
    looping: Option<JoinHandle<LoopStats>>,
    /// 이 장면에서 만든 키프레임(장면 끝에 지도 스레드로)
    pending_kf: Option<KfId>,
    /// System::mbResetActiveMap
    reset_pending: bool,
    /// 패치 옮겨 붙이기 칸
    graft: GraftState,
    /// Tracking::mVelocity · mlRelativeFramePoses.back()
    velocity: Se3,
    rel: Option<Se3>,
}

impl RustSystem {
    /// System(설정 파일, RGBD). 영상 크기는 원본처럼 첫 장면 경계에 쓰임
    pub fn open(settings: &str, w: i32, h: i32) -> Result<RustSystem, String> {
        let init = SystemInit::open(settings, w, h)?;
        let extractor = Extractor::new(init.ctx.pyr.clone());
        let tracker = Tracker { th_depth: init.tracking.th_depth, max_frames: init.tracking.max_frames, ..Default::default() };
        let next_frame = init.next_frame;
        let map = Arc::new(MapUpdate::new(MapState { atlas: init.atlas.clone(), ..Default::default() }));
        let mapper = Mapper::<KfId>::new();
        let pyr = &init.ctx.pyr;
        let tri = TriParams {
            cam: init.ctx.cam.clone(),
            scale: pyr.scale.clone(),
            sigma2: pyr.sigma2.clone(),
            scale_factor: pyr.scale_factor,
            monocular: false,
            inertial: false,
            far_points: init.settings.th_far_points != 0.0,
            th_far_points: init.settings.th_far_points,
        };
        let voc_steps = Arc::new(std::sync::Mutex::new(None));
        let kfdb = Arc::new(std::sync::Mutex::new(None));
        let mut steps = RsSteps {
            kfdb: Arc::clone(&kfdb), mapper: Arc::clone(&mapper), recent: Vec::new(), tri, th_depth: init.tracking.th_depth, reset_seen: 0, voc: Arc::clone(&voc_steps) };
        let loopq = LoopQueue::<KfId>::new();
        let (m2, map2, q2) = (Arc::clone(&mapper), Arc::clone(&map), Arc::clone(&loopq));
        let mapping = Some(std::thread::spawn(move || mapping::run(&m2, &map2, &q2, &mut steps)));
        let rand = Arc::new(std::sync::Mutex::new(GlibcRand::new(1)));
        let flow = Arc::new(LoopFlow::new(Arc::clone(&map), Arc::clone(&mapper), Arc::clone(&loopq), RsGba));
        let ops = RsLoop::new(Arc::clone(&kfdb), Arc::clone(&rand), RustSim3);
        let f2 = Arc::clone(&flow);
        let looping = Some(std::thread::Builder::new().name("orb3-loop".into()).spawn(move || f2.run(ops)).expect("루프 스레드"));
        Ok(RustSystem {
            init,
            extractor,
            tracker,
            map,
            mapper,
            loopq,
            mapping,
            next_frame,
            cur: None,
            last: None,
            changed: false,
            gray: Vec::new(),
            depth: Vec::new(),
            last_init_kf: 0,
            last_init_frame_id: 0,
            initial_frame_id: 0,
            voc: None,
            voc_steps,
            kfdb,
            rand,
            flow,
            looping,
            reset_pending: false,
            pending_kf: None,
            graft: GraftState::default(),
            velocity: Se3::default(),
            rel: None,
        })
    }

    /// 단어장을 붙임(재위치 · BoW 에 필요)
    pub fn set_vocabulary(&mut self, voc: Arc<Vocabulary>) {
        *self.kfdb.lock().unwrap() = Some(KeyFrameDatabase::new(&voc));
        *self.voc_steps.lock().unwrap() = Some(Arc::clone(&voc));
        self.voc = Some(voc);
    }

    /// 끝내기: 지도 스레드 · 루프 대기열을 닫고 통계를 돌려줌
    pub fn shutdown(&mut self) -> Option<MappingStats> {
        self.mapper.finish();
        let st = self.mapping.take().and_then(|h| h.join().ok());
        self.loopq.finish();
        if let Some(h) = self.looping.take() {
            let _ = h.join();
        }
        st
    }
}

impl Drop for RustSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// 흐름 도중 하위 단계가 바꾸는 Tracking 칸. 흐름이 `&mut Tracker` 를 쥐고 있어 끝난 뒤 넣고, NeedNewKeyFrame 은 넣은 판으로 판단
#[derive(Default)]
struct Edits {
    last_reloc_frame_id: Option<u64>,
    reference_kf: Option<Option<KfId>>,
    matches_inliers: Option<i32>,
    local: Option<(Vec<KfId>, Vec<MpId>)>,
    /// StereoInitialization: (마지막 키프레임, 그 장면 번호)
    init: Option<(KfId, u64)>,
}

impl Edits {
    fn apply(&self, t: &mut Tracker) {
        if let Some(id) = self.last_reloc_frame_id {
            t.last_reloc_frame_id = id;
        }
        if let Some(r) = self.reference_kf {
            t.reference_kf = r;
        }
        if let Some(n) = self.matches_inliers {
            t.matches_inliers = n;
        }
        if let Some((k, m)) = &self.local {
            t.local_kfs = k.clone();
            t.local_mps = m.clone();
        }
        if let Some((k, f)) = self.init {
            t.last_kf = Some(k);
            t.last_kf_frame_id = f;
        }
    }
}

/// Atlas::CreateNewMap: 지금 지도의 최대 키프레임 번호 다음을 새 지도의 시작 번호로, 새 지도(Map::nNextId)를 지금 지도로
fn atlas_create_new_map(a: &mut Atlas, last_init_kf: &mut KfId) {
    if let Some(mx) = a.maps.get(a.current).and_then(|m| m.kfs.iter().copied().max()) {
        if *last_init_kf < mx {
            *last_init_kf = mx + 1;
        }
    }
    let id = a.maps.iter().map(|m| m.id + 1).max().unwrap_or(0).max(a.next_map);
    a.next_map = id + 1;
    a.maps.push(Map { id, ..Default::default() });
    a.current = a.maps.len() - 1;
}

/// Track() 하위 단계: 옮겨진 것은 Rust 로, 나머지는 [`hole`]
struct Ops<'a> {
    s: &'a mut RustSystem,
    ed: Edits,
    /// 흐름 시작 때 Tracker(흐름이 &mut 로 쥐고 있어 하위 단계는 이 판 + ed 를 봄)
    tk: Tracker,
}

impl Ops<'_> {
    fn tracker_now(&self) -> Tracker {
        let mut t = self.tk.clone();
        self.ed.apply(&mut t);
        t
    }
    fn keep_local(&mut self, t: &Tracker) {
        self.ed.reference_kf = Some(t.reference_kf);
        self.ed.matches_inliers = Some(t.matches_inliers);
        self.ed.local = Some((t.local_kfs.clone(), t.local_mps.clone()));
    }
    /// Frame::ComputeBoW
    fn compute_bow(&mut self) {
        let _t = Tm::new("compute_bow");
        let voc = self.s.voc.clone().unwrap_or_else(|| hole("단어장(set_vocabulary) 없이 ComputeBoW"));
        let f = self.s.cur.as_mut().expect("지금 장면");
        if f.bow.is_empty() {
            let (bv, fv) = voc.transform(&f.desc, 4);
            f.bow = bv.into_iter().collect();
            f.feat = fv.into_iter().collect();
        }
    }
}

/// 추적 단계 자세 대조(ORBSYS_TRK=시작-끝 장면 번호, 원본 쪽 steps_dump.cc 의 T 줄과 같은 꼴)
fn trk_on(id: u64) -> bool {
    static R: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    let (a, b) = *R.get_or_init(|| std::env::var("ORBSYS_TRK").ok().and_then(|e| e.split_once('-').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).unwrap_or((1, 0)));
    id >= a && id <= b
}
fn trk_pose(w: &str, id: u64, p: &Se3, n: i64) {
    if trk_on(id) {
        crate::orb3sys::mapping::step_text(&format!("T {id} {w} {} {n}\n", crate::orb3sys::mapping::se3_hex(p)));
    }
}
fn nmp(f: &Frame) -> i64 {
    (0..f.keys.len()).filter(|&i| f.map_points[i].is_some() && !f.outlier[i]).count() as i64
}

/// 추적 단계 계측(ORBSYS_TIME=파일): 장면마다 단계별 ms · 지도 갱신 잠금 기다린 ms
struct Tm(&'static str, std::time::Instant);
thread_local! {
    static TIMES: std::cell::RefCell<Vec<(&'static str, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
}
impl Tm {
    fn new(n: &'static str) -> Tm {
        Tm(n, std::time::Instant::now())
    }
}
impl Drop for Tm {
    fn drop(&mut self) {
        let ns = self.1.elapsed().as_nanos() as u64;
        TIMES.with(|t| {
            let mut t = t.borrow_mut();
            match t.iter_mut().find(|e| e.0 == self.0) {
                Some(e) => e.1 += ns,
                None => t.push((self.0, ns)),
            }
        });
    }
}
fn time_line(frame: u64, total_ns: u64) {
    use std::io::Write;
    static OUT: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let wait = crate::orb3sys::threads::take_map_wait_ns();
    let v = TIMES.with(|t| std::mem::take(&mut *t.borrow_mut()));
    let Some(f) = OUT.get_or_init(|| std::env::var("ORBSYS_TIME").ok().map(|p| std::sync::Mutex::new(std::fs::File::create(p).expect("ORBSYS_TIME")))) else { return };
    let mut l = format!("{frame}\t{:.3}\twait={:.3}", total_ns as f64 * 1e-6, wait as f64 * 1e-6);
    for (n, ns) in v {
        l += &format!("\t{n}={:.3}", ns as f64 * 1e-6);
    }
    l.push('\n');
    let _ = f.lock().unwrap().write_all(l.as_bytes());
}

impl TrackOps for Ops<'_> {
    fn stereo_initialization(&mut self) -> i32 {
        let _t = Tm::new("stereo_initialization");
        let mut t = self.tracker_now();
        t.state = state::NOT_INITIALIZED;
        let s = &mut *self.s;
        let cur = s.cur.as_mut().expect("지금 장면");
        let kf = {
            let mut g = s.map.lock();
            track::stereo_initialization(&mut t, cur, &mut g.atlas)
        };
        if let Some(kf) = kf {
            s.mapper.insert(kf);
            s.changed = true;
            self.ed.init = Some((kf, t.last_kf_frame_id));
            self.keep_local(&t);
        }
        t.state
    }
    fn check_replaced_in_last_frame(&mut self) {
        let _t = Tm::new("check_replaced_in_last_frame");
        let s = &mut *self.s;
        if let Some(l) = s.last.as_mut() {
            let g = s.map.lock();
            track::check_replaced_in_last_frame(l, &g.atlas.mps);
        }
    }
    fn track_reference_keyframe(&mut self) -> bool {
        let _t = Tm::new("track_reference_keyframe");
        self.compute_bow();
        let t = self.tracker_now();
        let s = &mut *self.s;
        let last_pose = s.last.as_ref().and_then(|l| l.tcw).unwrap_or_default();
        let cur = s.cur.as_mut().expect("지금 장면");
        let mut g = s.map.lock();
        let Some(rkf) = t.reference_kf.and_then(|k| g.atlas.kfs.get(&k)).cloned() else { return false };
        track::track_reference_keyframe(cur, last_pose, &rkf, &mut g.atlas.mps)
    }
    fn track_with_motion_model(&mut self) -> bool {
        let _t = Tm::new("track_with_motion_model");
        let t = self.tracker_now();
        let s = &mut *self.s;
        let (Some(last), Some(tlr)) = (s.last.as_mut(), s.rel) else { return false };
        let cur = s.cur.as_mut().expect("지금 장면");
        let mut g = s.map.lock();
        let Some(ref_pose) = last.ref_kf.and_then(|k| g.atlas.kfs.get(&k)).map(|k| k.tcw) else { return false };
        if let Some(k) = last.ref_kf {
            trk_pose("tlr", cur.id, &tlr, k as i64);
            trk_pose("ref", cur.id, &ref_pose, g.atlas.kfs[&k].map as i64);
        }
        let r = track::track_with_motion_model(cur, last, ref_pose, tlr, s.velocity, t.sensor, &mut g.atlas.mps);
        if let Some(p) = last.tcw {
            trk_pose("ulf", cur.id, &p, nmp(last));
        }
        r
    }
    /// Tracking::Relocalization(pMap): ComputeBoW → DetectRelocalizationCandidates(F, pMap) → 재위치 흐름(`orb3loop::reloc`)
    fn relocalization(&mut self, map: Option<MapId>) -> bool {
        let _t = Tm::new("relocalization");
        let voc = self.s.voc.clone().unwrap_or_else(|| hole("단어장(set_vocabulary) 없이 Relocalization"));
        let s = &mut *self.s;
        let f = s.cur.as_mut().expect("지금 장면");
        if f.bow.is_empty() {
            // Frame::ComputeBoW: transform(기술자, 4)
            let (bv, fv) = voc.transform(&f.desc, 4);
            f.bow = bv.into_iter().collect();
            f.feat = fv.into_iter().collect();
        }
        let g = s.map.lock();
        let pmap = map.unwrap_or(g.atlas.maps[g.atlas.current].id);
        let bow: BowVector = f.bow.iter().map(|(w, v)| (*w, *v)).collect();
        let cands = {
            let mut db = s.kfdb.lock().unwrap();
            let db = db.as_mut().expect("KeyFrameDatabase");
            let _t = Tm::new("reloc_db");
            db.detect_reloc(&AtlasGraph::new(&g.atlas), &bow, pmap as DbMap).into_iter().map(|k| k as KfId).collect::<Vec<_>>()
        };
        // 루프 스레드와 같은 수열: 잠금 안에서 잠깐 빌려 씀
        let mut rg = s.rand.lock().unwrap();
        let cell = std::cell::RefCell::new(std::mem::replace(&mut *rg, GlibcRand::new(1)));
        let ok = {
            let _t = Tm::new("reloc_pnp");
            relocalize(f, &cands, &g.atlas.kfs, &g.atlas.mps, &mut RustPnp { rand: &cell })
        };
        *rg = cell.into_inner();
        trk_pose(if ok { "rel1" } else { "rel0" }, f.id, &f.tcw.unwrap_or_default(), pmap as i64);
        if ok {
            self.ed.last_reloc_frame_id = Some(f.id);
        }
        ok
    }
    fn track_local_map(&mut self) -> bool {
        let _t = Tm::new("track_local_map");
        let mut t = self.tracker_now();
        let far = (self.s.init.settings.th_far_points != 0.0, self.s.init.settings.th_far_points);
        let ok = {
            let s = &mut *self.s;
            let cur = s.cur.as_mut().expect("지금 장면");
            let mut g = s.map.lock();
            let st = &mut *g;
            trk_pose("tlm0", cur.id, &cur.tcw.unwrap_or_default(), nmp(cur));
            let r = track::track_local_map(&mut t, cur, &mut st.atlas, &mut st.ext, &|k| k, far);
            trk_pose("tlm1", cur.id, &cur.tcw.unwrap_or_default(), if r { t.matches_inliers as i64 } else { -1 });
            r
        };
        self.keep_local(&t);
        ok
    }
    fn pose_finite(&mut self) -> bool {
        self.s.cur.as_ref().and_then(|f| f.tcw).is_some_and(|p| p.q.iter().chain(&p.t).all(|v| v.is_finite()))
    }
    /// Tracking::CreateMapInAtlas 의 지도 · 장면 부분(추적 상태 칸은 `track::create_map_in_atlas`)
    fn create_map_in_atlas(&mut self) {
        let _t = Tm::new("create_map_in_atlas");
        let s = &mut *self.s;
        s.last_init_frame_id = s.next_frame;
        {
            let mut g = s.map.lock();
            atlas_create_new_map(&mut g.atlas, &mut s.last_init_kf);
        }
        // mLastFrame = Frame()
        s.last = None;
        s.changed = true;
    }
    /// Track() 안의 mpSystem->ResetActiveMap(): 표지만 세움. 실제 Tracking::ResetActiveMap 은 다음 TrackRGBD 시작(장면 만들기 전)
    fn reset_active_map(&mut self) {
        self.s.reset_pending = true;
    }
    /// Atlas::ChangeMap
    fn change_map(&mut self, map: MapId) {
        let mut g = self.s.map.lock();
        let i = g.atlas.maps.iter().position(|m| m.id == map).expect("바꿀 지도");
        g.atlas.current = i;
        drop(g);
        self.s.changed = true;
    }
    fn need_new_keyframe(&mut self, t: &Tracker) -> bool {
        let _t = Tm::new("need_new_keyframe");
        let mut t = t.clone();
        self.ed.apply(&mut t);
        let g = self.s.map.lock();
        let a = &g.atlas;
        let n_kfs = a.maps.get(a.current).map(|m| m.kfs.len() as i32).unwrap_or(0);
        let ref_kf = t.reference_kf;
        let tracked: Vec<i32> = (0..=3).map(|k| ref_kf.map(|r| track::tracked_map_points(a, &g.ext, r, k)).unwrap_or(0)).collect();
        let ctx = NeedKfCtx {
            n_kfs,
            ref_tracked: Box::new(move |k| tracked[k.clamp(0, 3) as usize]),
            imu_initialized: false,
            last_kf_ts: t.last_kf.and_then(|k| a.kfs.get(&k)).map(|k| k.timestamp),
        };
        let f = self.s.cur.as_ref().expect("지금 장면");
        track::need_new_keyframe(&t, f, &*self.s.mapper, &ctx)
    }
    fn create_new_keyframe(&mut self) -> Option<KfId> {
        let _t = Tm::new("create_new_keyframe");
        let th = self.s.init.tracking.th_depth;
        let s = &mut *self.s;
        if !s.mapper.set_not_stop(true) {
            // 원본은 만들지 않고 돌아감(mpReferenceKF · mpLastKeyFrame 그대로)
            return None;
        }
        let cur = s.cur.as_mut().expect("지금 장면");
        let kf = {
            let mut g = s.map.lock();
            track::create_new_keyframe(cur, th, &mut g.atlas)
        };
        // 지도 스레드에 넘기기는 장면 끝(track_tail: mlRelativeFramePoses 의 Tcr 를 잰 뒤)으로 미룸 — 원본은 넘긴 뒤 바로 Tcr 를 재므로
        // 지도 스레드가 아직 이 키프레임 자세를 안 바꾼 값을 읽는다. Rust 지도 스레드는 키프레임 처리 전체를 지도 잠금 안에서 하므로
        // 먼저 넘기면 track_tail 이 지역 BA 뒤 자세를 읽을 수도 있어(경합) 결과가 실행마다 달라졌음.
        s.pending_kf = Some(kf);
        s.changed = true;
        self.ed.reference_kf = Some(Some(kf));
        Some(kf)
    }
    /// 패치 Tracking::TryGraft: 지금 지도를 다른 지도(키프레임 많은 순)에 재위치해 보고, 같은 변환이 두 번이면 합치기 예약
    fn try_graft(&mut self) {
        let _t = Tm::new("try_graft");
        let frame_id = self.s.cur.as_ref().expect("지금 장면").id;
        let (cur, mut maps) = {
            let g = self.s.map.lock();
            if g.atlas.maps.len() < 2 || frame_id < self.s.graft.graft_try + 3 {
                return;
            }
            let a = &g.atlas;
            (a.maps[a.current].id, a.maps.iter().map(|m| (m.id, m.kfs.len(), m.bad)).collect::<Vec<_>>())
        };
        self.s.graft.graft_try = frame_id;
        let tcw_cur = self.s.cur.as_ref().unwrap().tcw.expect("지금 자세");
        let saved = self.s.cur.clone();
        let saved_reloc = self.ed.last_reloc_frame_id;
        // std::sort(KeyFramesInMap 큰 순) — 원본 순서(GetAllMaps = 포인터 순)는 id 순으로
        stdsort::sort_by(&mut maps, &mut |a: &(MapId, usize, bool), b: &(MapId, usize, bool)| a.1 > b.1);
        for (m, n, bad) in maps {
            if m == cur || bad || n == 0 {
                continue;
            }
            let ok = self.relocalization(Some(m));
            let tcw_m = self.s.cur.as_ref().unwrap().tcw;
            self.s.cur = saved.clone();
            self.ed.last_reloc_frame_id = saved_reloc;
            if !ok {
                continue;
            }
            let s = tcw_m.expect("재위치 자세").inverse().mul(&tcw_cur);
            graft::try_graft_step(&mut self.s.graft, frame_id, cur, m, s);
            return;
        }
    }
    fn all_maps(&mut self) -> Vec<(MapId, i32)> {
        let g = self.s.map.lock();
        g.atlas.maps.iter().map(|m| (m.id, m.kfs.len() as i32)).collect()
    }
    fn current_map(&mut self) -> MapId {
        let g = self.s.map.lock();
        g.atlas.maps[g.atlas.current].id
    }
    fn kfs_in_current_map(&mut self) -> i32 {
        let g = self.s.map.lock();
        g.atlas.maps.get(g.atlas.current).map(|m| m.kfs.len() as i32).unwrap_or(0)
    }
    fn imu_initialized(&mut self) -> bool {
        false
    }
    fn last_frame_set(&mut self) -> bool {
        self.s.last.as_ref().is_some_and(|f| f.tcw.is_some())
    }
    fn cur_frame_set(&mut self) -> bool {
        self.s.cur.as_ref().is_some_and(|f| f.tcw.is_some())
    }
    fn post_track_cleanup(&mut self, velocity_from_poses: bool) {
        let s = &mut *self.s;
        let cur = s.cur.as_mut().expect("지금 장면");
        let g = s.map.lock();
        let v = track::update_motion_and_clean(cur, s.last.as_ref(), &g.atlas.mps);
        if velocity_from_poses {
            s.velocity = v.expect("두 장면 자세");
        }
    }
}

impl Orb3System for RustSystem {
    fn track(&mut self, rgb: &[u8], depth: &[u16], w: u32, h: u32, t: f64, _up: Option<[f64; 3]>) -> Track {
        let t_all = std::time::Instant::now();
        if std::mem::take(&mut self.reset_pending) {
            let _t = Tm::new("reset");
            self.reset_active_map_now();
        }
        let (wu, hu) = (w as usize, h as usize);
        let t_ex = Tm::new("extract");
        let mut gray = std::mem::take(&mut self.gray);
        let mut dep = std::mem::take(&mut self.depth);
        self.init.gray(rgb, wu, hu, 3, &mut gray);
        self.init.depth(depth, &mut dep);
        let (keys, desc) = self.extractor.extract(&gray, wu, hu);
        let id = self.next_frame;
        self.next_frame += 1;
        let f = Frame::new_rgbd(id, t, keys, desc, &dep, wu, Arc::clone(&self.init.ctx));
        self.gray = gray;
        self.depth = dep;
        self.last = self.cur.replace(f);
        drop(t_ex);
        {
            let _t = Tm::new("graft_apply");
            self.do_pending_graft();
        }
        let mut tr = std::mem::take(&mut self.tracker);
        let tk = tr.clone();
        let mut ops = Ops { s: self, ed: Edits::default(), tk };
        {
            let _t = Tm::new("flow");
            track::track_flow(&mut tr, id, t, &mut ops);
        }
        let ed = ops.ed;
        ed.apply(&mut tr);
        self.tracker = tr;
        self.track_tail();
        // 지도 번호는 키프레임을 넘기기 전에 읽음(넘긴 뒤면 지도 스레드가 잡은 지도 잠금을 기다리게 됨)
        let map_id = self.current_map_id();
        let twc = match self.cur.as_ref().and_then(|f| f.tcw) {
            // orb3.cc: put34(tcw.inverse()) — Sophus 역(사원수 켤레)의 행렬
            Some(p) => crate::orb3cv::system::put34(&p.inverse()),
            // 원본 TrackRGBD 는 자세가 없으면 Sophus::SE3f() (단위)
            None => [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        if let Some(k) = self.pending_kf.take() {
            self.mapper.insert(k);
            self.mapper.set_not_stop(false);
        }
        time_line(id, t_all.elapsed().as_nanos() as u64);
        Track { state: if self.tracker.state == state::OK { state::OK } else { self.tracker.state }, twc, map_id, ref_kf: self.tracker.reference_kf }
    }
    fn map_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }
    fn keyframes(&mut self, all: bool) -> Vec<(u64, u64, [f64; 12])> {
        cvsys::keyframes(&self.map.lock().atlas, all)
    }
    /// 패치 RequestGraft(바깥): s = into ← from, 3×4 행 우선
    fn graft(&mut self, from: u64, into: u64, s: &[f64; 12]) -> bool {
        let r = [[s[0] as f32, s[1] as f32, s[2] as f32], [s[4] as f32, s[5] as f32, s[6] as f32], [s[8] as f32, s[9] as f32, s[10] as f32]];
        let se = Se3 { q: quat_from_mat(&r), t: [s[3] as f32, s[7] as f32, s[11] as f32] };
        let g = self.map.lock();
        graft::request_graft(&mut self.graft, &g.atlas, from as MapId, into as MapId, se)
    }
    fn wait_idle(&mut self) {
        self.mapper.wait_idle();
        // 재생 대조용: ORBSYS_WAIT_LOOP 가 있으면 루프 스레드 · 전역 BA 까지 기다림
        if std::env::var_os("ORBSYS_WAIT_LOOP").is_some() {
            while self.loopq.pending() > 0 || self.flow.busy() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.mapper.wait_idle();
        }
    }
    fn kf_now(&mut self, id: u64) -> Option<(u64, [f64; 12])> {
        cvsys::kf_now(&self.map.lock().atlas, id)
    }
}

impl RustSystem {
    /// Track() 끝: 새 키프레임 판단 뒤 바깥점 칸 비움, mCurrentFrame.mpReferenceKF, mlRelativeFramePoses
    fn track_tail(&mut self) {
        let st = self.tracker.state;
        if st != state::OK && st != state::RECENTLY_LOST {
            return;
        }
        let Some(cur) = self.cur.as_mut() else { return };
        for i in 0..cur.keys.len() {
            if cur.map_points[i].is_some() && cur.outlier[i] {
                cur.map_points[i] = None;
            }
        }
        if cur.ref_kf.is_none() {
            cur.ref_kf = self.tracker.reference_kf;
        }
        if let (Some(p), Some(r)) = (cur.tcw, cur.ref_kf) {
            let g = self.map.lock();
            if let Some(k) = g.atlas.kfs.get(&r) {
                self.rel = Some(p.mul(&k.twc));
            }
        }
    }

    /// Tracking::ResetActiveMap(false): 지도 스레드 비우기(지금 키프레임을 마친 뒤 대기열 · 최근 점 버림) → BoW DB 에서 지금 지도 지움
    /// → Atlas::clearMap(지도 id 는 남김) → 추적 칸. 루프 스레드 비우기 요청은 루프 스레드가 엮이면 같은 자리에.
    fn reset_active_map_now(&mut self) {
        self.mapper.reset_active_map();
        {
            let mut g = self.map.lock();
            let st = &mut *g;
            let cur = st.atlas.current;
            let cur_id = st.atlas.maps[cur].id;
            if let Some(db) = self.kfdb.lock().unwrap().as_mut() {
                db.clear_map(&AtlasGraph::new(&st.atlas), cur_id as DbMap);
            }
            let kfs = std::mem::take(&mut st.atlas.maps[cur].kfs);
            let mps = std::mem::take(&mut st.atlas.maps[cur].mps);
            for k in kfs {
                st.atlas.kfs.remove(&k);
                st.ext.kf_track_ref.remove(&k);
            }
            for m in mps {
                st.atlas.mps.remove(&m);
                st.ext.mp_track_ref.remove(&m);
            }
            st.atlas.maps[cur].origin_kf = None;
        }
        self.last_init_frame_id = self.next_frame;
        let cur_frame = self.cur.as_ref().map(|f| f.id).unwrap_or(0);
        self.initial_frame_id = cur_frame;
        let t = &mut self.tracker;
        t.state = state::NO_IMAGES_YET;
        t.last_reloc_frame_id = cur_frame;
        t.reference_kf = None;
        t.last_kf = None;
        t.velocity = false;
        self.cur = None;
        self.last = None;
        self.changed = true;
    }

    /// 패치 DoPendingGraft(Track() 시작): 예약된 합치기를 지도 스레드를 멈추고 함. 루프 · 전체 최적화 중이면 다음 장면에
    fn do_pending_graft(&mut self) {
        let Some((from, into, s)) = self.graft.pending.take() else { return };
        {
            let g = self.map.lock();
            let bad = |id: MapId| g.atlas.maps.iter().find(|m| m.id == id).is_none_or(|m| m.bad);
            if bad(from) || bad(into) {
                return;
            }
        }
        if self.loop_busy() {
            self.graft.pending = Some((from, into, s));
            return;
        }
        let move_cam = self.current_map_id() == from as u64;
        let s = graft::yaw_only(&s);
        // GraftMap: 쌓인 키프레임 처리 끝 → 지도 스레드 멈춤 → 지도 잠금 안 옮기기 → 놓기
        self.mapper.wait_idle();
        let tok = self.mapper.request_stop();
        tok.wait();
        {
            let mut g = self.map.lock();
            crate::orb3sys::mapping::step_text(&format!("G {from} {into} {}\n",crate::orb3sys::mapping::se3_hex(&s)));
            let fm = g.atlas.maps.iter().find(|m| m.id == from).map(|m| (m.kfs.iter().copied().min(), m.mps.iter().copied().min()));
            let pk = |g: &MapState, w: &str| {
                if let Some((Some(k), Some(p))) = fm {
                    let (kf, mp) = (&g.atlas.kfs[&k], &g.atlas.mps[&p]);
                    let h = |v: &[f32]| v.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>().join(" ");
                    crate::orb3sys::mapping::step_text(&format!("{w} K{k} {} P{p} {} N {}\n", crate::orb3sys::mapping::se3_hex(&kf.tcw), h(&mp.pos), h(&mp.normal)));
                }
            };
            pk(&g, "GB");
            graft::graft_map(&mut g.atlas, from, into, &s);
            pk(&g, "GA");
            if let Some((&km, kf)) = g.atlas.kfs.iter().next_back() {
                crate::orb3sys::mapping::step_text(&format!("T {km} GK {} {}\n",crate::orb3sys::mapping::se3_hex(&kf.tcw), kf.map));
            }
            let k0 = g.atlas.maps.iter().find(|m| m.id == into).and_then(|m| m.kfs.iter().copied().filter(|k| !g.atlas.kfs[k].bad).min());
            if let Some(k0) = k0 {
                crate::orb3sys::mapping::step_digest(&g, k0, "graft");
            }
            if let Some(p) = self.last.as_ref().and_then(|l| l.tcw) {
                crate::orb3sys::mapping::step_text(&format!("GL {}\n",crate::orb3sys::mapping::se3_hex(&p)));
            }
        }
        drop(tok);
        if move_cam {
            if let Some(l) = self.last.as_mut() {
                if let Some(p) = l.tcw {
                    l.set_pose(p.mul(&s.inverse()));
                }
            }
            let mut g = self.map.lock();
            g.atlas.current = g.atlas.maps.iter().position(|m| m.id == into).expect("into 지도");
        }
        self.changed = true;
    }

    /// 루프 닫기 · 합치기 · 전체 최적화가 돌고 있나(mMutexBusy · isRunningGBA)
    fn loop_busy(&self) -> bool {
        self.flow.busy()
    }

    fn current_map_id(&self) -> u64 {
        let g = self.map.lock();
        g.atlas.maps.get(g.atlas.current).map(|m| m.id as u64).unwrap_or(0)
    }
}
