//! 포터 TRACK — ORB-SLAM3 Tracking.cc(+ cpp/ORB_SLAM3.patch)를 옮긴 것. RGB-D(IMU 는 위쪽만) 경로.
//!
//! 옮긴 함수(원본 이름): Track() 의 상태 흐름(놓침 · 재위치 · 새 지도), NeedNewKeyFrame, CreateNewKeyFrame 의 새 지도 점,
//! UpdateLocalKeyFrames, UpdateLocalPoints, KeyFrame::TrackedMapPoints, Frame::UpdatePoseMatrices · UnprojectStereo.
//! 하위 단계(ORB 짝짓기 · 자세 최적화 · 재위치 본체 · 지도 만들기)는 다른 몫이라 [`TrackOps`] 로 부른다 —
//! 지금은 원본 C++ 를 부르고(FFI), 그 몫이 옮겨지면 Rust 로 바꿔 끼운다.
//!
//! 원본 포인터 순서(std::map<KeyFrame*,int> · std::set<KeyFrame*>)는 `order(kf)` 열쇠로 받는다:
//! 실제 실행에서는 id(결정적), 원본과 비트 비교할 때는 덤프한 주소.
//!
//! 공통 타입(types.rs)에 없는 원본 칸은 [`TrackExt`] 옆 표에 둔다(통합 때 타입으로 옮길 후보):
//! MapPoint::nObs · mnTrackReferenceForFrame, KeyFrame::mnTrackReferenceForFrame · mbNotErase, Map::mnInitKFid.
use super::types::*;
use std::collections::{BTreeMap, HashMap};

/// System::eSensor
pub mod sensor {
    pub const MONOCULAR: i32 = 0;
    pub const STEREO: i32 = 1;
    pub const RGBD: i32 = 2;
    pub const IMU_MONOCULAR: i32 = 3;
    pub const IMU_STEREO: i32 = 4;
    pub const IMU_RGBD: i32 = 5;
}
#[inline]
fn is_imu(s: i32) -> bool {
    s == sensor::IMU_MONOCULAR || s == sensor::IMU_STEREO || s == sensor::IMU_RGBD
}

/// Tracking::eTrackingState
pub mod state {
    pub const SYSTEM_NOT_READY: i32 = -1;
    pub const NO_IMAGES_YET: i32 = 0;
    pub const NOT_INITIALIZED: i32 = 1;
    pub const OK: i32 = 2;
    pub const RECENTLY_LOST: i32 = 3;
    pub const LOST: i32 = 4;
    pub const OK_KLT: i32 = 5;
}

/// 패치: 놓친 지 이만큼(초) 지나도 재위치 못 하면 새 지도(원본 3.0f)
pub const LOST_NEWMAP_S: f64 = 0.2;

/// 공통 타입 밖의 원본 칸(옆 표)
#[derive(Clone, Debug, Default)]
pub struct TrackExt {
    pub mp_track_ref: HashMap<MpId, u64>,
    pub kf_track_ref: HashMap<KfId, u64>,
}

impl TrackExt {
    #[inline]
    pub fn kf_ref(&self, kf: KfId) -> u64 {
        *self.kf_track_ref.get(&kf).unwrap_or(&0)
    }
    #[inline]
    pub fn mp_ref(&self, mp: MpId) -> u64 {
        *self.mp_track_ref.get(&mp).unwrap_or(&0)
    }
}

/// MapPoint::AddObservation 이 더하는 관측 무게(오른쪽 영상 없음: 깊이 있으면 2)
#[inline]
pub fn obs_weight(u_right: f32) -> i32 {
    if u_right >= 0.0 { 2 } else { 1 }
}

/// 지역 지도 스레드 상태(LocalMapping 의 질문들)
pub trait MapperStatus {
    fn is_stopped(&self) -> bool;
    fn stop_requested(&self) -> bool;
    fn accept_keyframes(&self) -> bool;
    fn is_initializing(&self) -> bool;
    fn keyframes_in_queue(&self) -> i32;
    fn interrupt_ba(&self);
}

/// Tracking 의 판단에 쓰는 상태 칸
#[derive(Clone, Debug)]
pub struct Tracker {
    pub sensor: i32,
    pub state: i32,
    pub only_tracking: bool,
    pub velocity: bool,
    pub last_reloc_frame_id: u64,
    pub max_frames: i32,
    pub min_frames: i32,
    pub last_kf_frame_id: u64,
    pub reference_kf: Option<KfId>,
    pub last_kf: Option<KfId>,
    pub th_depth: f32,
    pub matches_inliers: i32,
    pub has_camera2: bool,
    pub time_stamp_lost: f64,
    pub insert_kfs_lost: bool,
    pub frames_to_reset_imu: u64,
    pub local_kfs: Vec<KfId>,
    pub local_mps: Vec<MpId>,
}

impl Default for Tracker {
    fn default() -> Self {
        Tracker {
            sensor: sensor::RGBD,
            state: state::NO_IMAGES_YET,
            only_tracking: false,
            velocity: false,
            last_reloc_frame_id: 0,
            max_frames: 30,
            min_frames: 0,
            last_kf_frame_id: 0,
            reference_kf: None,
            last_kf: None,
            th_depth: 0.0,
            matches_inliers: 0,
            has_camera2: false,
            time_stamp_lost: 0.0,
            insert_kfs_lost: true,
            frames_to_reset_imu: 0,
            local_kfs: Vec::new(),
            local_mps: Vec::new(),
        }
    }
}

/// KeyFrame::TrackedMapPoints
pub fn tracked_map_points(atlas: &Atlas, ext: &TrackExt, kf: KfId, min_obs: i32) -> i32 {
    let k = &atlas.kfs[&kf];
    let check = min_obs > 0;
    let mut n = 0;
    for mp in k.map_points.iter().flatten() {
        let m = &atlas.mps[mp];
        if !m.bad {
            if check {
                if m.n_obs >= min_obs {
                    n += 1;
                }
            } else {
                n += 1;
            }
        }
    }
    n
}

/// NeedNewKeyFrame 이 읽는 장면 밖 값
pub struct NeedKfCtx {
    /// mpAtlas->KeyFramesInMap()(지금 지도)
    pub n_kfs: i32,
    /// mpReferenceKF->TrackedMapPoints(nMinObs) — nMinObs 를 받아 셈
    pub ref_tracked: Box<dyn Fn(i32) -> i32>,
    pub imu_initialized: bool,
    pub last_kf_ts: Option<f64>,
}

/// Tracking::NeedNewKeyFrame
pub fn need_new_keyframe(t: &Tracker, f: &Frame, lm: &dyn MapperStatus, c: &NeedKfCtx) -> bool {
    let s = t.sensor;
    if is_imu(s) && !c.imu_initialized {
        let lt = c.last_kf_ts.unwrap_or(f64::NAN);
        return (f.timestamp - lt) >= 0.25;
    }
    if t.only_tracking {
        return false;
    }
    if lm.is_stopped() || lm.stop_requested() {
        return false;
    }
    let n_kfs = c.n_kfs;
    if f.id < t.last_reloc_frame_id.wrapping_add(t.max_frames as i64 as u64) && n_kfs > t.max_frames {
        return false;
    }
    let min_obs = if n_kfs <= 2 { 2 } else { 3 };
    let n_ref = (c.ref_tracked)(min_obs);
    let idle = lm.accept_keyframes();
    let (mut ntc, mut nntc) = (0i32, 0i32);
    if s != sensor::MONOCULAR && s != sensor::IMU_MONOCULAR {
        for i in 0..f.keys.len() {
            let d = f.depth[i];
            if d > 0.0 && d < t.th_depth {
                if f.map_points[i].is_some() && !f.outlier[i] {
                    ntc += 1;
                } else {
                    nntc += 1;
                }
            }
        }
    }
    let close = ntc < 100 && nntc > 70;
    let mut th_ref: f32 = 0.75;
    if n_kfs < 2 {
        th_ref = 0.4;
    }
    if s == sensor::MONOCULAR {
        th_ref = 0.9;
    }
    if t.has_camera2 {
        th_ref = 0.75;
    }
    if s == sensor::IMU_MONOCULAR {
        th_ref = if t.matches_inliers > 350 { 0.75 } else { 0.90 };
    }
    let inl = t.matches_inliers;
    let c1a = f.id >= t.last_kf_frame_id.wrapping_add(t.max_frames as i64 as u64);
    let c1b = f.id >= t.last_kf_frame_id.wrapping_add(t.min_frames as i64 as u64) && idle;
    let c1c = s != sensor::MONOCULAR
        && s != sensor::IMU_MONOCULAR
        && s != sensor::IMU_STEREO
        && s != sensor::IMU_RGBD
        && ((inl as f64) < (n_ref as f64) * 0.25 || close);
    let c2 = ((inl as f32) < (n_ref as f32) * th_ref || close) && inl > 15;
    let mut c3 = false;
    if let Some(lt) = c.last_kf_ts {
        if is_imu(s) && (f.timestamp - lt) >= 0.5 {
            c3 = true;
        }
    }
    let c4 = (((inl < 75) && (inl > 15)) || t.state == state::RECENTLY_LOST) && s == sensor::IMU_MONOCULAR;
    if ((c1a || c1b || c1c) && c2) || c3 || c4 {
        if idle || lm.is_initializing() {
            true
        } else {
            lm.interrupt_ba();
            if s != sensor::MONOCULAR && s != sensor::IMU_MONOCULAR { lm.keyframes_in_queue() < 3 } else { false }
        }
    } else {
        false
    }
}

/// Tracking::UpdateLocalKeyFrames(RGB-D: IMU 시간 이웃 단계 없음).
/// `use_cur` = !IMU 초기화 || 재위치 직후(원본 조건) — 참이면 지금 장면, 거짓이면 지난 장면의 점이 투표.
/// `order(kf)` = 원본 std::map/set<KeyFrame*> 순서 열쇠.
pub fn update_local_keyframes(
    t: &mut Tracker,
    voter: &mut Frame,
    cur_id: u64,
    cur_ref_kf: &mut Option<KfId>,
    atlas: &Atlas,
    ext: &mut TrackExt,
    order: &dyn Fn(KfId) -> u64,
) {
    assert!(!is_imu(t.sensor), "IMU 시간 이웃 단계는 옮기지 않음(RGB-D 경로)");
    let mut counter: BTreeMap<(u64, KfId), i32> = BTreeMap::new();
    for slot in voter.map_points.iter_mut() {
        if let Some(mp) = *slot {
            let m = &atlas.mps[&mp];
            if !m.bad {
                for kf in m.obs.keys() {
                    *counter.entry((order(*kf), *kf)).or_insert(0) += 1;
                }
            } else {
                *slot = None;
            }
        }
    }
    let mut max = 0;
    let mut kf_max: Option<KfId> = None;
    t.local_kfs.clear();
    for (&(_, kf), &n) in counter.iter() {
        if atlas.kfs[&kf].bad {
            continue;
        }
        if n > max {
            max = n;
            kf_max = Some(kf);
        }
        t.local_kfs.push(kf);
        ext.kf_track_ref.insert(kf, cur_id);
    }
    let n0 = t.local_kfs.len();
    'outer: for it in 0..n0 {
        if t.local_kfs.len() > 80 {
            break;
        }
        let kf = t.local_kfs[it];
        let k = &atlas.kfs[&kf];
        for &(nb, _) in k.ordered_conn.iter().take(10) {
            if !atlas.kfs[&nb].bad && ext.kf_ref(nb) != cur_id {
                t.local_kfs.push(nb);
                ext.kf_track_ref.insert(nb, cur_id);
                break;
            }
        }
        let mut ch: Vec<KfId> = k.children.clone();
        ch.sort_by_key(|c| (order(*c), *c));
        for c in ch {
            if !atlas.kfs[&c].bad && ext.kf_ref(c) != cur_id {
                t.local_kfs.push(c);
                ext.kf_track_ref.insert(c, cur_id);
                break;
            }
        }
        if let Some(p) = k.parent {
            if ext.kf_ref(p) != cur_id {
                t.local_kfs.push(p);
                ext.kf_track_ref.insert(p, cur_id);
                // 원본: 여기 break 는 바깥 반복을 끝냄
                break 'outer;
            }
        }
    }
    if let Some(m) = kf_max {
        t.reference_kf = Some(m);
        *cur_ref_kf = Some(m);
    }
}

/// Tracking::UpdateLocalPoints
pub fn update_local_points(t: &mut Tracker, cur_id: u64, atlas: &Atlas, ext: &mut TrackExt) {
    t.local_mps.clear();
    for &kf in t.local_kfs.iter().rev() {
        for mp in atlas.kfs[&kf].map_points.iter().flatten() {
            if ext.mp_ref(*mp) == cur_id {
                continue;
            }
            if !atlas.mps[mp].bad {
                t.local_mps.push(*mp);
                ext.mp_track_ref.insert(*mp, cur_id);
            }
        }
    }
}

// ---------------- Sophus / Eigen 산술(원본 연산 순서, 기준 빌드 -ffp-contract=off: 곱합 합치기 없음) ----------------

/// Eigen QuaternionBase::toRotationMatrix (행 우선 r[i][j])
pub fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    Se3 { q, t: [0.0; 3] }.rot()
}

/// Sophus SO3 생성자의 normalize(): coeffs /= sqrt(squaredNorm), 합은 Eigen Packet4f 줄이기 순서 (x+z)+(y+w)
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let s = [q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3]];
    let n2 = (s[0] + s[2]) + (s[1] + s[3]);
    let l = n2.sqrt();
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}

/// Sophus SO3::operator*(point): uv = v×p; uv += uv; p + w·uv + v×uv
pub fn quat_rotate(q: [f32; 4], p: [f32; 3]) -> [f32; 3] {
    Se3 { q, t: [0.0; 3] }.rotate(&p)
}

/// Frame::UpdatePoseMatrices 의 (mRwc, mOw): Twc = Tcw.inverse()
pub fn pose_inverse_matrices(tcw: &Se3) -> ([[f32; 3]; 3], [f32; 3]) {
    let qi = quat_normalize([-tcw.q[0], -tcw.q[1], -tcw.q[2], tcw.q[3]]);
    let mt = [tcw.t[0] * -1.0, tcw.t[1] * -1.0, tcw.t[2] * -1.0];
    let ow = quat_rotate(qi, mt);
    (quat_to_rot(qi), ow)
}

/// Frame::UnprojectStereo: x3D = Rwc·(x,y,z) + Ow. 행 내적은 Eigen 이 a0b0 + (a1b1 + a2b2) 로 펼침
pub fn unproject(rwc: &[[f32; 3]; 3], ow: &[f32; 3], cam: &Camera, u: f32, v: f32, z: f32) -> [f32; 3] {
    let x = (u - cam.cx) * z * cam.invfx;
    let y = (v - cam.cy) * z * cam.invfy;
    let p = [x, y, z];
    let mut r = [0f32; 3];
    for i in 0..3 {
        let s = rwc[i][0] * p[0] + (rwc[i][1] * p[1] + rwc[i][2] * p[2]);
        r[i] = s + ow[i];
    }
    r
}

/// CreateNewKeyFrame 의 RGB-D 새 지도 점 고르기 · 위치. 돌려줌: (점 번호, 세계 위치) 만들 것들, 비울 칸(관측 0 짝).
pub fn new_keyframe_points(
    f: &Frame,
    rwc: &[[f32; 3]; 3],
    ow: &[f32; 3],
    cam: &Camera,
    th_depth: f32,
    max_point: i32,
    nobs: &dyn Fn(MpId) -> i32,
) -> (Vec<(usize, [f32; 3])>, Vec<usize>) {
    let n = f.keys.len();
    let mut v: Vec<(f32, i32)> = (0..n).filter(|&i| f.depth[i] > 0.0).map(|i| (f.depth[i], i as i32)).collect();
    let mut out = Vec::new();
    let mut cleared = Vec::new();
    if v.is_empty() {
        return (out, cleared);
    }
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut np = 0;
    for &(z, i) in &v {
        let i = i as usize;
        let create = match f.map_points[i] {
            None => true,
            Some(mp) => {
                if nobs(mp) < 1 {
                    cleared.push(i);
                    true
                } else {
                    false
                }
            }
        };
        if create {
            let k = &f.keys[i];
            out.push((i, unproject(rwc, ow, cam, k.x, k.y, f.depth[i])));
        }
        np += 1;
        if z > th_depth && np > max_point {
            break;
        }
    }
    (out, cleared)
}

// ---------------- Track() 흐름 ----------------

/// Track() 의 하위 단계. 지금은 원본(FFI)이 구현, 옮겨지면 Rust.
pub trait TrackOps {
    fn stereo_initialization(&mut self) -> i32;
    fn check_replaced_in_last_frame(&mut self);
    fn track_reference_keyframe(&mut self) -> bool;
    fn track_with_motion_model(&mut self) -> bool;
    /// Relocalization(pMap): None = 지금 지도
    fn relocalization(&mut self, map: Option<MapId>) -> bool;
    fn track_local_map(&mut self) -> bool;
    fn pose_finite(&mut self) -> bool;
    fn create_map_in_atlas(&mut self);
    fn reset_active_map(&mut self);
    fn change_map(&mut self, map: MapId);
    fn need_new_keyframe(&mut self, t: &Tracker) -> bool;
    /// 새 키프레임 id 를 돌려줌. None = 지도 스레드가 멈춰 있어 SetNotStop 거절(원본은 만들지 않고 돌아감, 추적 칸도 그대로)
    fn create_new_keyframe(&mut self) -> Option<KfId>;
    fn try_graft(&mut self);
    /// 지도 표: (id, 키프레임 수) — 원본 GetAllMaps 순서
    fn all_maps(&mut self) -> Vec<(MapId, i32)>;
    fn current_map(&mut self) -> MapId;
    fn kfs_in_current_map(&mut self) -> i32;
    fn imu_initialized(&mut self) -> bool;
    /// 장면 자세가 정해졌나(mLastFrame.isSet · mCurrentFrame.isSet)
    fn last_frame_set(&mut self) -> bool;
    fn cur_frame_set(&mut self) -> bool;
    /// 속도 · 짝 정리 · 바깥점 지우기 등 자료 단계(흐름 판단 없음)
    fn post_track_cleanup(&mut self, velocity_from_poses: bool);
}

/// Tracking::CreateMapInAtlas 의 추적 상태 부분(지도 만들기 · 장면 비우기는 ops)
pub fn create_map_in_atlas(t: &mut Tracker, ops: &mut dyn TrackOps) {
    ops.create_map_in_atlas();
    t.state = state::NO_IMAGES_YET;
    t.velocity = false;
    t.last_kf = None;
    t.reference_kf = None;
}

/// Tracking::Track() 의 상태 흐름(RGB-D, 지도 만들기 켬, 패치 포함). 시간 늦은 장면 · 단계별 실행 · IMU 상태 예측은 뺌.
pub fn track_flow(t: &mut Tracker, frame_id: u64, ts: f64, ops: &mut dyn TrackOps) {
    if t.state == state::NOT_INITIALIZED || t.state == state::NO_IMAGES_YET {
        if t.state == state::NO_IMAGES_YET {
            t.state = state::NOT_INITIALIZED;
        }
        t.state = ops.stereo_initialization();
        if t.state == state::OK {
            // StereoInitialization: 첫 키프레임이 마지막 · 기준 키프레임
            t.last_kf = Some(KfId::MAX);
        }
        return;
    }
    let mut ok;
    if !t.only_tracking {
        if t.state == state::OK {
            ops.check_replaced_in_last_frame();
            if (!t.velocity && !ops.imu_initialized()) || frame_id < t.last_reloc_frame_id.wrapping_add(2) {
                ok = ops.track_reference_keyframe();
            } else {
                ok = ops.track_with_motion_model();
                if !ok {
                    ok = ops.track_reference_keyframe();
                }
            }
            if !ok {
                if frame_id <= t.last_reloc_frame_id.wrapping_add(t.frames_to_reset_imu) && is_imu(t.sensor) {
                    t.state = state::LOST;
                } else {
                    t.state = state::RECENTLY_LOST;
                    t.time_stamp_lost = ts;
                }
            }
        } else if t.state == state::RECENTLY_LOST {
            assert!(!is_imu(t.sensor));
            ok = ops.relocalization(None);
            if !ok {
                let cur = ops.current_map();
                for (m, n) in ops.all_maps() {
                    if m == cur || n == 0 {
                        continue;
                    }
                    if ops.relocalization(Some(m)) {
                        ops.change_map(m);
                        t.last_kf = None;
                        ok = true;
                        break;
                    }
                }
            }
            if ts - t.time_stamp_lost > LOST_NEWMAP_S && !ok {
                t.state = state::LOST;
                ok = false;
            }
        } else {
            // LOST: 새 지도
            if ops.kfs_in_current_map() == 0 {
                ops.reset_active_map();
            } else {
                create_map_in_atlas(t, ops);
            }
            t.last_kf = None;
            return;
        }
    } else {
        unimplemented!("위치만 추적 모드는 쓰지 않음");
    }
    if ok {
        ok = ops.track_local_map();
    }
    if ok && !ops.pose_finite() {
        ok = false;
    }
    if ok {
        t.state = state::OK;
    } else if t.state == state::OK {
        if is_imu(t.sensor) && !ops.imu_initialized() {
            ops.reset_active_map();
        }
        t.state = state::RECENTLY_LOST;
        t.time_stamp_lost = ts;
    }
    if ok || t.state == state::RECENTLY_LOST {
        let both = ops.last_frame_set() && ops.cur_frame_set();
        t.velocity = both;
        ops.post_track_cleanup(both);
        let need = ops.need_new_keyframe(t);
        if need && (ok || (t.insert_kfs_lost && t.state == state::RECENTLY_LOST && is_imu(t.sensor))) {
            // CreateNewKeyFrame 끝: mpReferenceKF · mnLastKeyFrameId · mpLastKeyFrame
            if let Some(kf) = ops.create_new_keyframe() {
                t.reference_kf = Some(kf);
                t.last_kf_frame_id = frame_id;
                t.last_kf = Some(kf);
            }
        }
    }
    if t.state == state::LOST {
        if ops.kfs_in_current_map() == 0 {
            ops.reset_active_map();
            return;
        }
        create_map_in_atlas(t, ops);
        return;
    }
    if t.state == state::OK && !t.only_tracking {
        ops.try_graft();
    }
}

// ---------------- Tracking 단계(장면 · 지도 점 자료 위) ----------------

use super::matcher::Matcher;
use crate::orb3loop::reloc::pose_optimization_frame;

/// Tracking::CheckReplacedInLastFrame: 앞 장면 점 중 대체된 것(GetReplaced)을 새 점으로
pub fn check_replaced_in_last_frame(last: &mut Frame, mps: &BTreeMap<MpId, MapPoint>) {
    for s in last.map_points.iter_mut() {
        if let Some(m) = *s {
            if let Some(r) = mps.get(&m).and_then(|p| p.replaced) {
                *s = Some(r);
            }
        }
    }
}

/// 자세 최적화 뒤 이상치 버리기(TrackReferenceKeyFrame · TrackWithMotionModel 공통, 왼쪽 영상만).
/// `reset_in_view` = mbTrackInView 도 끔(TrackReferenceKeyFrame 만). 돌려줌: 관측 있는 남은 짝 수
fn discard_outliers(cur: &mut Frame, mps: &mut BTreeMap<MpId, MapPoint>, reset_in_view: bool) -> i32 {
    let mut map = 0;
    for i in 0..cur.keys.len() {
        let Some(m) = cur.map_points[i] else { continue };
        if cur.outlier[i] {
            cur.map_points[i] = None;
            cur.outlier[i] = false;
            let p = mps.get_mut(&m).unwrap();
            if reset_in_view {
                p.track_in_view = false;
            }
            p.last_frame_seen = cur.id;
        } else if mps[&m].n_obs > 0 {
            map += 1;
        }
    }
    map
}

/// Tracking::TrackReferenceKeyFrame(관성 아님). BoW(ComputeBoW)는 부르는 쪽이 채움. `last_pose` = mLastFrame.GetPose()
pub fn track_reference_keyframe(cur: &mut Frame, last_pose: Se3, rkf: &KeyFrame, mps: &mut BTreeMap<MpId, MapPoint>) -> bool {
    let m = Matcher::new(0.7, true);
    let (n, v) = m.search_by_bow_kf_frame(rkf, cur, mps);
    if n < 15 {
        return false;
    }
    cur.map_points = v;
    cur.set_pose(last_pose);
    pose_optimization_frame(cur, mps);
    discard_outliers(cur, mps, true) >= 10
}

/// Tracking::UpdateLastFrame(지도 만들기 모드: 위치만) + TrackWithMotionModel(관성 아님).
/// `ref_pose` = mLastFrame.mpReferenceKF->GetPose(), `tlr` = mlRelativeFramePoses.back(), `velocity` = mVelocity
#[allow(clippy::too_many_arguments)]
pub fn track_with_motion_model(cur: &mut Frame, last: &mut Frame, ref_pose: Se3, tlr: Se3, velocity: Se3, sensor: i32, mps: &mut BTreeMap<MpId, MapPoint>) -> bool {
    let m = Matcher::new(0.9, true);
    last.set_pose(tlr.mul(&ref_pose));
    cur.set_pose(velocity.mul(&last.tcw.unwrap()));
    cur.map_points.iter_mut().for_each(|s| *s = None);
    let th = if sensor == sensor::STEREO { 7.0 } else { 15.0 };
    let mono = sensor == sensor::MONOCULAR || sensor == sensor::IMU_MONOCULAR;
    let mut n = m.search_by_projection_last(cur, last, mps, th, mono);
    if n < 20 {
        cur.map_points.iter_mut().for_each(|s| *s = None);
        n = m.search_by_projection_last(cur, last, mps, 2.0 * th, mono);
    }
    if n < 20 {
        return false;
    }
    pose_optimization_frame(cur, mps);
    discard_outliers(cur, mps, false) >= 10
}

/// Tracking::SearchLocalPoints(관성 아님). `far` = (mbFarPoints, mThFarPoints)
pub fn search_local_points(t: &Tracker, cur: &mut Frame, mps: &mut BTreeMap<MpId, MapPoint>, far: (bool, f32)) -> i32 {
    for s in cur.map_points.iter_mut() {
        if let Some(m) = *s {
            let p = mps.get_mut(&m).unwrap();
            if p.bad {
                *s = None;
            } else {
                p.visible += 1;
                p.last_frame_seen = cur.id;
                p.track_in_view = false;
            }
        }
    }
    let mut n_to_match = 0;
    for &m in &t.local_mps {
        let p = mps.get_mut(&m).unwrap();
        if p.last_frame_seen == cur.id || p.bad {
            continue;
        }
        if cur.is_in_frustum(p, 0.5) {
            p.visible += 1;
            n_to_match += 1;
        }
    }
    if n_to_match == 0 {
        return 0;
    }
    let mut th = if t.sensor == sensor::RGBD || t.sensor == sensor::IMU_RGBD { 3.0 } else { 1.0 };
    if is_imu(t.sensor) {
        th = 10.0;
    }
    if cur.id < t.last_reloc_frame_id.wrapping_add(2) {
        th = 5.0;
    }
    if t.state == state::LOST || t.state == state::RECENTLY_LOST {
        th = 15.0;
    }
    Matcher::new(0.8, true).search_by_projection_local(cur, &t.local_mps, mps, th, far.0, far.1)
}

/// TrackLocalMap 의 자세 최적화 뒤 부분: 점 찾음 수 올림 · 맞은 수(mnMatchesInliers) · 성공 판단
pub fn track_local_map_finish(t: &mut Tracker, cur: &mut Frame, mps: &mut BTreeMap<MpId, MapPoint>) -> bool {
    let id = cur.id;
    let mut inl = 0;
    for i in 0..cur.keys.len() {
        let Some(m) = cur.map_points[i] else { continue };
        if !cur.outlier[i] {
            let p = mps.get_mut(&m).unwrap();
            p.found += 1;
            if t.only_tracking || p.n_obs > 0 {
                inl += 1;
            }
        } else if t.sensor == sensor::STEREO {
            cur.map_points[i] = None;
        }
    }
    t.matches_inliers = inl;
    if id < t.last_reloc_frame_id.wrapping_add(t.max_frames as i64 as u64) && inl < 50 {
        return false;
    }
    if inl > 10 && t.state == state::RECENTLY_LOST {
        return true;
    }
    if t.sensor == sensor::IMU_MONOCULAR {
        return inl >= 50;
    }
    if t.sensor == sensor::IMU_STEREO || t.sensor == sensor::IMU_RGBD {
        return inl >= 15;
    }
    inl >= 30
}

/// Tracking::TrackLocalMap(관성 아님): UpdateLocalMap → SearchLocalPoints → PoseOptimization → 점 통계 · 판단.
/// `order` = UpdateLocalKeyFrames 의 키프레임 순서 열쇠
pub fn track_local_map(t: &mut Tracker, cur: &mut Frame, atlas: &mut Atlas, ext: &mut TrackExt, order: &dyn Fn(KfId) -> u64, far: (bool, f32)) -> bool {
    let id = cur.id;
    let mut rk = cur.ref_kf;
    update_local_keyframes(t, cur, id, &mut rk, atlas, ext, order);
    cur.ref_kf = rk;
    update_local_points(t, id, atlas, ext);
    search_local_points(t, cur, &mut atlas.mps, far);
    pose_optimization_frame(cur, &atlas.mps);
    track_local_map_finish(t, cur, &mut atlas.mps)
}

/// MapPoint(Pos, pRefKF, pMap) + AddObservation(kf, i) + KeyFrame::AddMapPoint + ComputeDistinctiveDescriptors
/// + UpdateNormalAndDepth + Atlas::AddMapPoint. 돌려줌: 새 점 번호
pub fn new_kf_map_point(atlas: &mut Atlas, kf: KfId, i: usize, pos: [f32; 3]) -> MpId {
    let id = atlas.next_mp;
    atlas.next_mp += 1;
    let k = &atlas.kfs[&kf];
    let map = k.map;
    let mut mp = MapPoint { id, map, pos, first_frame: k.frame_id, first_kf: kf, ref_kf: kf, visible: 1, found: 1, ..Default::default() };
    mp.add_observation(k, i);
    atlas.kfs.get_mut(&kf).unwrap().map_points[i] = Some(id);
    mp.compute_distinctive_descriptors(&atlas.kfs);
    atlas.mps.insert(id, mp);
    super::mapdata::mp_update_normal_and_depth(atlas, id);
    if let Some(mi) = atlas.maps.iter_mut().find(|x| x.id == map) {
        mi.add_map_point(id);
    }
    id
}

/// Tracking::StereoInitialization(RGB-D, 관성 · 위쪽 벡터 없음): 점 500 개 넘으면 원점 자세로 첫 키프레임과 깊이 점들을 만듦.
/// 돌려줌: 첫 키프레임(지도 스레드에 넣는 것은 부르는 쪽). 지역 키프레임 · 점, 기준 · 마지막 키프레임도 채움
pub fn stereo_initialization(t: &mut Tracker, cur: &mut Frame, atlas: &mut Atlas) -> Option<KfId> {
    if cur.keys.len() <= 500 {
        return None;
    }
    cur.set_pose(Se3::new([0.0, 0.0, 0.0, 1.0], [0.0; 3]));
    let map = atlas.maps[atlas.current].id;
    let kf = atlas.next_kf;
    atlas.next_kf += 1;
    atlas.kfs.insert(kf, KeyFrame::from_frame(kf, map, cur));
    // Atlas::AddKeyFrame → Map::AddKeyFrame(첫 키프레임이면 mnInitKFid)
    let cur_map = atlas.current;
    atlas.maps[cur_map].add_keyframe(kf);
    let (rwc, ow) = pose_inverse_matrices(cur.tcw.as_ref().unwrap());
    let cam = cur.ctx.cam;
    for i in 0..cur.keys.len() {
        let z = cur.depth[i];
        if z > 0.0 {
            let k = &cur.keys[i];
            let x = unproject(&rwc, &ow, &cam, k.x, k.y, z);
            let id = new_kf_map_point(atlas, kf, i, x);
            cur.map_points[i] = Some(id);
        }
    }
    t.last_kf_frame_id = cur.id;
    t.last_kf = Some(kf);
    t.local_kfs.push(kf);
    // mpAtlas->GetAllMapPoints(): 지금 지도 점(번호 순 집합)
    let mut all: Vec<MpId> = atlas.maps[atlas.current].mps.clone();
    all.sort_unstable();
    t.local_mps = all;
    t.reference_kf = Some(kf);
    cur.ref_kf = Some(kf);
    t.state = state::OK;
    Some(kf)
}

/// Tracking::CreateNewKeyFrame(RGB-D, 관성 아님) 중 키프레임 · 새 점 만들기(지도 스레드 멈춤 확인 · 넣기는 부르는 쪽)
pub fn create_new_keyframe(cur: &mut Frame, th_depth: f32, atlas: &mut Atlas) -> KfId {
    let map = atlas.maps[atlas.current].id;
    let kf = atlas.next_kf;
    atlas.next_kf += 1;
    atlas.kfs.insert(kf, KeyFrame::from_frame(kf, map, cur));
    cur.ref_kf = Some(kf);
    let (rwc, ow) = pose_inverse_matrices(cur.tcw.as_ref().unwrap());
    cur.rwc = rwc;
    cur.ow = ow;
    let cam = cur.ctx.cam;
    let (pts, cleared) = {
        let mps = &atlas.mps;
        new_keyframe_points(cur, &rwc, &ow, &cam, th_depth, 100, &|m| mps.get(&m).map(|p| p.n_obs).unwrap_or(0))
    };
    for c in cleared {
        cur.map_points[c] = None;
    }
    for (i, x) in pts {
        let id = new_kf_map_point(atlas, kf, i, x);
        cur.map_points[i] = Some(id);
    }
    kf
}

/// Track() 끝 단계 중 속도와 VO 짝 정리: 두 장면 자세가 있으면 mVelocity = Tcw · Twl, 관측 없는 점 칸 비움
pub fn update_motion_and_clean(cur: &mut Frame, last: Option<&Frame>, mps: &BTreeMap<MpId, MapPoint>) -> Option<Se3> {
    let v = match (cur.tcw, last.and_then(|l| l.tcw)) {
        (Some(c), Some(l)) => Some(c.mul(&l.inverse())),
        _ => None,
    };
    for i in 0..cur.keys.len() {
        if let Some(m) = cur.map_points[i] {
            if mps.get(&m).map(|p| p.n_obs).unwrap_or(0) < 1 {
                cur.outlier[i] = false;
                cur.map_points[i] = None;
            }
        }
    }
    v
}
