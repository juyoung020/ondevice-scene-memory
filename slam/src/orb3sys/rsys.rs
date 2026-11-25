//! Rust System 뼈대: 원본 `System::TrackRGBD` 흐름을 옮겨진 부품으로 엮는다.
//! 설정 · 입력 변환(`orb3cv::system`) → ORB 추출(`orb3rs::extract`) → Frame(`orb3rs::frame`) → 추적 흐름(`orb3rs::track::track_flow`)
//! → 키프레임은 지도 스레드(`orb3sys::mapping`, Rust 지역 BA) → 루프 대기열(`orb3sys::loopflow` 자리).
//!
//! 아직 Rust 판이 없는 단계는 [`Hole`] 로 이름을 남기고, 지나가면 그 이름으로 멈춘다(원본 FFI 로 대신 부르지 않음 — 지도 자료가 따로라
//! 바꿔 끼우는 단위가 System 통째이기 때문, `mod.rs` 머리글). 구멍 목록은 [`HOLES`]. 이 판이 [`Orb3System`] 를 구현하므로
//! 구멍이 다 메워지면 `orb3_backend` 가 `RustSystem` 을 그대로 연다.
use super::mapping::{self, KfSteps, MapState, MappingStats};
use super::threads::{LoopQueue, MapUpdate, Mapper};
use super::{Orb3System, Track};
use crate::orb3cv::system::{self as cvsys, SystemInit};
use crate::orb3rs::extract::Extractor;
use crate::orb3rs::track::{self, state, MapperStatus, NeedKfCtx, Tracker, TrackOps};
use crate::orb3rs::types::{Frame, KfId, MapId};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Rust 판이 아직 없는 단계(이름 = 원본 함수)
pub const HOLES: &[&str] = &[
    "Tracking::StereoInitialization",
    "Tracking::CheckReplacedInLastFrame",
    "Tracking::TrackReferenceKeyFrame",
    "Tracking::TrackWithMotionModel",
    "Tracking::Relocalization(KeyFrameDatabase 연결)",
    "Tracking::TrackLocalMap(SearchLocalPoints · PoseOptimization 엮기)",
    "Tracking::CreateNewKeyFrame",
    "Tracking::ResetActiveMap",
    "Atlas::CreateNewMap · ChangeMap",
    "패치 DoPendingGraft",
    "LocalMapping::ProcessNewKeyFrame",
    "LocalMapping::MapPointCulling",
    "LocalMapping::CreateNewMapPoints",
    "LocalMapping::SearchInNeighbors",
    "LocalMapping::KeyFrameCulling",
];

/// 옮겨지지 않은 단계를 지나면 그 이름으로 멈춤
#[track_caller]
pub fn hole(name: &str) -> ! {
    panic!("Rust System: 아직 옮기지 않은 단계 {name}")
}

/// 지도 스레드 키프레임 단계의 빈자리(포터 TRACK 커밋을 기다림)
pub struct Hole;
impl KfSteps for Hole {
    fn process_new_keyframe(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::ProcessNewKeyFrame")
    }
    fn map_point_culling(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::MapPointCulling")
    }
    fn create_new_map_points(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::CreateNewMapPoints")
    }
    fn search_in_neighbors(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::SearchInNeighbors")
    }
    fn keyframe_culling(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::KeyFrameCulling")
    }
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
}

impl RustSystem {
    /// System(설정 파일, RGBD). 영상 크기는 원본처럼 첫 장면 경계에 쓰임
    pub fn open(settings: &str, w: i32, h: i32) -> Result<RustSystem, String> {
        RustSystem::with_steps(settings, w, h, Hole)
    }

    pub fn with_steps<S: KfSteps + 'static>(settings: &str, w: i32, h: i32, mut steps: S) -> Result<RustSystem, String> {
        let init = SystemInit::open(settings, w, h)?;
        let extractor = Extractor::new(init.ctx.pyr.clone());
        let tracker = Tracker { th_depth: init.tracking.th_depth, max_frames: init.tracking.max_frames, ..Default::default() };
        let next_frame = init.next_frame;
        let map = Arc::new(MapUpdate::new(MapState { atlas: init.atlas.clone(), ..Default::default() }));
        let mapper = Mapper::<KfId>::new();
        let loopq = LoopQueue::<KfId>::new();
        let (m2, map2, q2) = (Arc::clone(&mapper), Arc::clone(&map), Arc::clone(&loopq));
        let mapping = Some(std::thread::spawn(move || mapping::run(&m2, &map2, &q2, &mut steps)));
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
        })
    }

    /// 끝내기: 지도 스레드 · 루프 대기열을 닫고 통계를 돌려줌
    pub fn shutdown(&mut self) -> Option<MappingStats> {
        self.mapper.finish();
        self.loopq.finish();
        self.mapping.take().and_then(|h| h.join().ok())
    }
}

impl Drop for RustSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Track() 하위 단계: 옮겨진 것은 Rust 로, 나머지는 [`hole`]
struct Ops<'a> {
    s: &'a mut RustSystem,
}

impl TrackOps for Ops<'_> {
    fn stereo_initialization(&mut self) -> i32 {
        hole("Tracking::StereoInitialization")
    }
    fn check_replaced_in_last_frame(&mut self) {
        hole("Tracking::CheckReplacedInLastFrame")
    }
    fn track_reference_keyframe(&mut self) -> bool {
        hole("Tracking::TrackReferenceKeyFrame")
    }
    fn track_with_motion_model(&mut self) -> bool {
        hole("Tracking::TrackWithMotionModel")
    }
    fn relocalization(&mut self, _map: Option<MapId>) -> bool {
        hole("Tracking::Relocalization(KeyFrameDatabase 연결)")
    }
    fn track_local_map(&mut self) -> bool {
        hole("Tracking::TrackLocalMap")
    }
    fn pose_finite(&mut self) -> bool {
        self.s.cur.as_ref().and_then(|f| f.tcw).is_some_and(|p| p.q.iter().chain(&p.t).all(|v| v.is_finite()))
    }
    fn create_map_in_atlas(&mut self) {
        hole("Atlas::CreateNewMap")
    }
    fn reset_active_map(&mut self) {
        hole("Tracking::ResetActiveMap")
    }
    fn change_map(&mut self, _map: MapId) {
        hole("Atlas::ChangeMap")
    }
    fn need_new_keyframe(&mut self) -> bool {
        let g = self.s.map.lock();
        let a = &g.atlas;
        let n_kfs = a.maps.get(a.current).map(|m| m.kfs.len() as i32).unwrap_or(0);
        let ref_kf = self.s.tracker.reference_kf;
        let tracked: Vec<i32> = (0..=3).map(|k| ref_kf.map(|r| track::tracked_map_points(a, &g.ext, r, k)).unwrap_or(0)).collect();
        let ctx = NeedKfCtx {
            n_kfs,
            ref_tracked: Box::new(move |k| tracked[k.clamp(0, 3) as usize]),
            imu_initialized: false,
            last_kf_ts: self.s.tracker.last_kf.and_then(|k| a.kfs.get(&k)).map(|k| k.timestamp),
        };
        let f = self.s.cur.as_ref().expect("지금 장면");
        track::need_new_keyframe(&self.s.tracker, f, &*self.s.mapper, &ctx)
    }
    fn create_new_keyframe(&mut self) {
        hole("Tracking::CreateNewKeyFrame")
    }
    fn try_graft(&mut self) {
        // 옮겨 붙이기 요청이 없으면 할 일 없음(원본 패치도 대기 중인 요청이 있을 때만)
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
    fn post_track_cleanup(&mut self, _velocity_from_poses: bool) {
        hole("Tracking::Track 뒷정리(속도 · 바깥점)")
    }
}

impl Orb3System for RustSystem {
    fn track(&mut self, rgb: &[u8], depth: &[u16], w: u32, h: u32, t: f64, _up: Option<[f64; 3]>) -> Track {
        let (wu, hu) = (w as usize, h as usize);
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
        let mut tr = std::mem::take(&mut self.tracker);
        track::track_flow(&mut tr, id, t, &mut Ops { s: self });
        self.tracker = tr;
        let (twc, map_id) = match self.cur.as_ref().and_then(|f| f.tcw) {
            Some(p) => {
                let (r, o) = track::pose_inverse_matrices(&p);
                let mut m = [0.0; 12];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i * 4 + j] = r[i][j] as f64;
                    }
                    m[i * 4 + 3] = o[i] as f64;
                }
                (m, self.current_map_id())
            }
            None => ([0.0; 12], self.current_map_id()),
        };
        Track { state: if self.tracker.state == state::OK { state::OK } else { self.tracker.state }, twc, map_id, ref_kf: self.tracker.reference_kf }
    }
    fn map_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }
    fn keyframes(&mut self, all: bool) -> Vec<(u64, u64, [f64; 12])> {
        cvsys::keyframes(&self.map.lock().atlas, all)
    }
    fn graft(&mut self, _from: u64, _into: u64, _s: &[f64; 12]) -> bool {
        hole("패치 DoPendingGraft")
    }
    fn wait_idle(&mut self) {
        self.mapper.wait_idle()
    }
    fn kf_now(&mut self, id: u64) -> Option<(u64, [f64; 12])> {
        cvsys::kf_now(&self.map.lock().atlas, id)
    }
}

impl RustSystem {
    fn current_map_id(&self) -> u64 {
        let g = self.map.lock();
        g.atlas.maps.get(g.atlas.current).map(|m| m.id as u64).unwrap_or(0)
    }
}
