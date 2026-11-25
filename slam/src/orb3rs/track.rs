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
use std::collections::{BTreeMap, HashMap, HashSet};

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
    /// MapPoint::nObs(관측 수: 깊이 있는 관측 2, 없으면 1)
    pub mp_nobs: HashMap<MpId, i32>,
    pub mp_track_ref: HashMap<MpId, u64>,
    pub kf_track_ref: HashMap<KfId, u64>,
    pub kf_not_erase: HashSet<KfId>,
    pub map_init_kf: HashMap<MapId, KfId>,
}

impl TrackExt {
    #[inline]
    pub fn nobs(&self, mp: MpId) -> i32 {
        *self.mp_nobs.get(&mp).unwrap_or(&0)
    }
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
                if ext.nobs(*mp) >= min_obs {
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

// ---------------- Sophus / Eigen 산술(원본 연산 순서, -march=native 의 곱합 합치기 포함) ----------------

/// Eigen QuaternionBase::toRotationMatrix (행 우선 r[i][j]). 원본 빌드(-O3 -march=native, GCC 곱합 합치기)와
/// 비트가 같도록 합치는 자리를 맞춤(덤프로 칸마다 확인): 비대각 a·b ± c·d → fma(±c, d, a·b),
/// 대각 (1,1) · (2,2) 는 fma(tx, x, ·), (0,0) 은 합치지 않음.
pub fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * x;
    let ty = 2.0 * y;
    let tz = 2.0 * z;
    let tyy = ty * y;
    let tzz = tz * z;
    [
        [1.0 - (tyy + tzz), (-tz).mul_add(w, ty * x), ty.mul_add(w, tz * x)],
        [tz.mul_add(w, ty * x), 1.0 - tx.mul_add(x, tzz), (-tx).mul_add(w, tz * y)],
        [(-ty).mul_add(w, tz * x), tx.mul_add(w, tz * y), 1.0 - tx.mul_add(x, tyy)],
    ]
}

/// Sophus SO3 생성자의 normalize(): coeffs /= sqrt(squaredNorm), 합은 Eigen Packet4f 줄이기 순서 (x+z)+(y+w)
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let s = [q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3]];
    let n2 = (s[0] + s[2]) + (s[1] + s[3]);
    let l = n2.sqrt();
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    // GCC 곱합 합치기: a*b - c*d → fma(a, b, -(c*d))
    [
        a[1].mul_add(b[2], -(a[2] * b[1])),
        a[2].mul_add(b[0], -(a[0] * b[2])),
        a[0].mul_add(b[1], -(a[1] * b[0])),
    ]
}

/// Sophus SO3::operator*(point): uv = v×p; uv += uv; p + w·uv + v×uv
pub fn quat_rotate(q: [f32; 4], p: [f32; 3]) -> [f32; 3] {
    let v = [q[0], q[1], q[2]];
    let mut uv = cross(v, p);
    for k in 0..3 {
        uv[k] += uv[k];
    }
    let c = cross(v, uv);
    [q[3].mul_add(uv[0], p[0]) + c[0], q[3].mul_add(uv[1], p[1]) + c[1], q[3].mul_add(uv[2], p[2]) + c[2]]
}

/// Frame::UpdatePoseMatrices 의 (mRwc, mOw): Twc = Tcw.inverse()
pub fn pose_inverse_matrices(tcw: &Se3) -> ([[f32; 3]; 3], [f32; 3]) {
    let qi = quat_normalize([-tcw.q[0], -tcw.q[1], -tcw.q[2], tcw.q[3]]);
    let mt = [tcw.t[0] * -1.0, tcw.t[1] * -1.0, tcw.t[2] * -1.0];
    let ow = quat_rotate(qi, mt);
    (quat_to_rot(qi), ow)
}

/// Frame::UnprojectStereo: x3D = Rwc·(x,y,z) + Ow. 행 내적은 Eigen 이 a0b0 + (a1b1 + a2b2) 로 펼치고 GCC 가
/// 합쳐 fma(r0, x, fma(r2, z, r1·y)) 가 됨(덤프 39,427 점 모두 이 꼴로 확인).
pub fn unproject(rwc: &[[f32; 3]; 3], ow: &[f32; 3], cam: &Camera, u: f32, v: f32, z: f32) -> [f32; 3] {
    let x = (u - cam.cx) * z * cam.invfx;
    let y = (v - cam.cy) * z * cam.invfy;
    let p = [x, y, z];
    let mut r = [0f32; 3];
    for i in 0..3 {
        let s = rwc[i][0].mul_add(p[0], rwc[i][2].mul_add(p[2], rwc[i][1] * p[1]));
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
    /// 새 키프레임 id 를 돌려줌
    fn create_new_keyframe(&mut self) -> KfId;
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
            let kf = ops.create_new_keyframe();
            t.reference_kf = Some(kf);
            t.last_kf_frame_id = frame_id;
            t.last_kf = Some(kf);
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
