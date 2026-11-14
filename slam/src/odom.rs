//! RTAB-Map 0.24.2 OdometryF2F(장면-키프레임 위치 추적) 를 옮긴 것.
//! 원본: corelib/src/odometry/OdometryF2F.cpp(computeTransform), corelib/src/Odometry.cpp(process: 속도 추측, 놓침, 리셋).
//! 우리 설정: Odom/Strategy=1, Odom/VisKeyFrameThr=60, Odom/KeyFrameThr=0.3, Odom/GuessMotion=true,
//! Odom/GuessSmoothingDelay=0, Odom/ResetCountdown=0(bridge 가 직접 다시 잡음), 필터 없음, IMU 없음.
//! 위치 계산은 모두 RTAB-Map Transform 과 같은 float 연산(register::rtf) 으로 한다.
use crate::params::Params;
use crate::register::{RegInfo, RegistrationVis, Tf};
use crate::{Features, Frame, Pose};

/// 한 장면의 위치 추적 결과
#[derive(Clone)]
pub struct OdomResult {
    /// 누적 위치(지도 몸체 축). 놓쳤으면 None
    pub pose: Option<Pose>,
    /// 같은 값의 float 3x4(비트 비교용)
    pub tf: Tf,
    pub lost: bool,
    /// 맞추기 공분산. 처음 장면(키프레임 없음)·리셋 직후는 대각 9999 = "새로 시작"
    pub covariance: [[f64; 6]; 6],
    pub inliers: usize,
    pub matches: usize,
    pub keyframe_updated: bool,
    /// 키프레임 기준 이번 장면 움직임(직전 위치 기준, RTAB-Map OdometryInfo::transform)
    pub transform: Tf,
    /// 이 장면에 남은 특징(RTAB-Map 이 data.setFeatures 로 돌려주는 것). 지도 노드에 쓰면 됨
    pub features: Features,
    pub reg: Option<RegInfo>,
}

pub struct OdometryF2F {
    reg: RegistrationVis,
    key_frame_thr: f32,
    vis_key_frame_thr: i32,
    guess_motion: bool,
    reset_countdown: i32,
    // Odometry
    pose: Tf,
    velocity: Tf,
    previous_stamp: f64,
    frames_processed: i32,
    reset_current_count: i32,
    // OdometryF2F
    ref_frame: Option<Features>,
    last_key_frame_pose: Tf,
}

fn eye(v: f64) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        c[i][i] = v;
    }
    c
}

fn subset(f: &Features, idx: &[usize]) -> Features {
    Features { kps: idx.iter().map(|&i| f.kps[i]).collect(), pts3d: idx.iter().map(|&i| f.pts3d[i]).collect(), desc: idx.iter().map(|&i| f.desc[i]).collect() }
}

impl OdometryF2F {
    pub fn new(p: &Params) -> OdometryF2F {
        OdometryF2F {
            reg: RegistrationVis::new(p),
            key_frame_thr: p.get("Odom/KeyFrameThr", 0.3f32),
            vis_key_frame_thr: p.get("Odom/VisKeyFrameThr", 150i32),
            guess_motion: p.get_bool("Odom/GuessMotion", true),
            reset_countdown: p.get("Odom/ResetCountdown", 0i32),
            pose: Tf::IDENTITY,
            velocity: Tf::NULL,
            previous_stamp: 0.0,
            frames_processed: 0,
            reset_current_count: 0,
            ref_frame: None,
            last_key_frame_pose: Tf::NULL,
        }
    }

    /// 다른 모듈(루프 클로저 등)과 같은 난수 상태를 씀(원본 한 프로세스의 rand()·random_device 순서)
    pub fn set_rng(&mut self, rng: crate::register::SharedRng) {
        self.reg.set_rng(rng);
    }

    pub fn pose(&self) -> Pose {
        self.pose.to_pose()
    }
    pub fn pose_tf(&self) -> Tf {
        self.pose
    }

    /// Odometry::reset + OdometryF2F::reset
    pub fn reset(&mut self, pose: Pose) {
        self.reset_tf(Tf::from_pose(&pose));
    }
    pub fn reset_tf(&mut self, pose: Tf) {
        self.velocity = Tf::NULL;
        self.reset_current_count = 0;
        self.previous_stamp = 0.0;
        self.frames_processed = 0;
        self.pose = pose;
        self.ref_frame = None;
        self.last_key_frame_pose = Tf::NULL;
    }

    /// Odometry::process
    pub fn process(&mut self, frame: &Frame, feats: Features) -> OdomResult {
        let stamp = frame.t;
        let mut dt = if self.previous_stamp > 0.0 || (self.previous_stamp == 0.0 && self.frames_processed == 1) { stamp - self.previous_stamp } else { 0.0 };
        let mut guess = if dt > 0.0 && self.guess_motion && !self.velocity.is_null() { Tf::IDENTITY } else { Tf::NULL };
        if !(dt > 0.0 || (dt == 0.0 && self.velocity.is_null())) {
            dt = 0.0;
            self.velocity = Tf::NULL;
        }
        if !self.velocity.is_null() && self.guess_motion {
            let v = self.velocity.xyz_rpy();
            let m = |x: f32| (x as f64 * dt) as f32;
            guess = Tf::from_xyz_rpy(m(v[0]), m(v[1]), m(v[2]), m(v[3]), m(v[4]), m(v[5]));
        }
        let mut res = self.compute_transform(frame, feats, guess);
        let t = res.transform;
        if !t.is_null() {
            self.reset_current_count = self.reset_countdown;
            let mut v = t.xyz_rpy();
            if dt != 0.0 {
                for x in v.iter_mut() {
                    *x = (*x as f64 / dt) as f32;
                }
            }
            self.previous_stamp = stamp;
            if dt != 0.0 {
                // GuessSmoothingDelay=0: 마지막 속도를 그대로
                self.velocity = Tf::from_xyz_rpy(v[0], v[1], v[2], v[3], v[4], v[5]);
            } else {
                self.velocity = Tf::NULL;
            }
            self.frames_processed += 1;
            self.pose = self.pose.mul(&t);
            res.tf = self.pose;
            res.pose = Some(self.pose.to_pose());
            res.lost = false;
            return res;
        } else if self.reset_current_count > 0 {
            self.reset_current_count -= 1;
            if self.reset_current_count == 0 {
                // 원본: 마지막 위치에서 다시 시작(새 지도), 이번 장면은 첫 장면처럼 처리
                let p = self.pose;
                self.reset_tf(p);
                self.reset_current_count = self.reset_countdown;
                let r = self.compute_transform(frame, res.features.clone(), Tf::NULL);
                let mut out = r;
                out.tf = self.pose;
                out.pose = Some(self.pose.to_pose());
                out.lost = false;
                return out;
            }
            // 원본 Odometry.cpp: 속도·이전 시각 지우기는 이 else-if 블록 안에만 있다(ResetCountdown=0 이면 놓쳐도 그대로 둠)
            self.velocity = Tf::NULL;
            self.previous_stamp = 0.0;
        }
        res.pose = None;
        res.tf = Tf::NULL;
        res.lost = true;
        res
    }

    /// OdometryF2F::computeTransform
    fn compute_transform(&mut self, frame: &Frame, feats: Features, guess: Tf) -> OdomResult {
        let cam = &frame.cam;
        if self.last_key_frame_pose.is_null() {
            self.last_key_frame_pose = self.pose;
        }
        let motion_since_kf = self.last_key_frame_pose.inverse().mul(&self.pose);
        let mut covariance;
        let mut output;
        let mut reg_info: Option<RegInfo> = None;
        let mut new_feats = feats;
        let (mut inliers, mut matches, mut to_words) = (0usize, 0usize, 0usize);
        if let Some(rf) = &self.ref_frame {
            let g = if !guess.is_null() { motion_since_kf.mul(&guess) } else { Tf::NULL };
            let mut info = self.reg.compute_tf(rf, &new_feats, cam, g);
            if info.transform.is_none() && !guess.is_null() {
                // 추측으로 실패: 추측 없이 한 번 더(특징은 처음 것 그대로)
                info = self.reg.compute_tf(rf, &new_feats, cam, Tf::NULL);
            }
            output = info.tf;
            covariance = info.covariance;
            inliers = info.inliers;
            matches = info.matches;
            to_words = info.to_words;
            if let Some(k) = &info.to_kept {
                new_feats = subset(&new_feats, k);
            }
            reg_info = Some(info);
        } else {
            output = Tf::IDENTITY;
            covariance = eye(9999.0);
        }
        let mut keyframe = false;
        if !output.is_null() {
            output = motion_since_kf.inverse().mul(&output);
            let ref_kps = self.ref_frame.as_ref().map(|f| f.kps.len()).unwrap_or(0);
            if self.key_frame_thr == 0.0 || self.vis_key_frame_thr == 0 || (inliers as f32) <= self.key_frame_thr * (ref_kps as f32) || (inliers as i64) <= self.vis_key_frame_thr as i64 {
                let mut features = to_words;
                if self.ref_frame.is_none() || features == 0 {
                    // 처음 장면: 특징을 새로 뽑음(이미 뽑아 둔 것 전체)
                    features = new_feats.kps.len();
                }
                if features >= self.reg.min_inliers() {
                    self.ref_frame = Some(new_feats.clone());
                    self.last_key_frame_pose = Tf::NULL;
                    keyframe = true;
                } else if self.ref_frame.is_none() {
                    output = Tf::NULL;
                }
            }
        }
        if output.is_null() && self.ref_frame.is_none() {
            covariance = eye(9999.0);
        }
        OdomResult {
            pose: None,
            tf: Tf::NULL,
            lost: output.is_null(),
            covariance,
            inliers,
            matches,
            keyframe_updated: keyframe,
            transform: output,
            features: new_feats,
            reg: reg_info,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::register::tests::{d435, synth_scene};
    use crate::Camera;
    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    fn frame(t: f64, cam: &Camera) -> Frame {
        Frame { id: 0, t, gray: vec![], color: vec![], color_is_rgb: false, depth_mm: Default::default(), depth_raw: Default::default(), depth_per_m: 1000.0, cam: cam.clone() }
    }

    #[test]
    fn trajectory_tracks() {
        let cam = d435();
        let mut p = Params::default();
        p.set("Odom/VisKeyFrameThr", "60");
        let mut odom = OdometryF2F::new(&p);
        let mut world = Vec::new();
        let mut prev = Pose::identity();
        let mut feats_prev = None;
        let mut max_err: f64 = 0.0;
        for i in 0..40 {
            let s = i as f64 * 0.02;
            let gt = Pose::from_parts(Translation3::new(s, 0.3 * s, 0.05 * s), UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.1 * s, 0.3 * s)));
            // 장면마다 같은 세계 점을 그 위치에서 본다(from = 처음, to = 지금)
            let (_, f) = synth_scene(350, &cam, &Pose::identity(), &gt, 0.0, 0.0, 7 + i, &mut world);
            let r = odom.process(&frame(i as f64 / 30.0, &cam), f.clone());
            let est = r.pose.expect("놓침");
            let d = est.inverse() * gt;
            max_err = max_err.max(d.translation.vector.norm());
            if i == 0 {
                assert!(r.covariance[0][0] >= 9999.0);
                assert!(r.keyframe_updated);
            }
            prev = gt;
            feats_prev = Some(f);
        }
        let _ = (prev, feats_prev);
        assert!(max_err < 2e-3, "max_err={max_err}");
    }
}
