//! ORB-SLAM3 스테레오(· 스테레오-관성): 설정 파일 만들기와, `orb3` 기능이면 C 함수(cpp/orb3.cc)를 부르는 얇은 층.
//! 위치는 3×4 행 우선 [f64; 12](카메라 → 세계, 적외선 왼쪽 카메라 좌표).
use crate::StereoCalib;
#[cfg(feature = "orb3")]
pub use ffi::*;

#[cfg(feature = "orb3")]
mod ffi {
use std::ffi::{c_char, c_int, c_void, CString};

unsafe extern "C" {
    fn orb3_new(voc: *const c_char, settings: *const c_char, inertial: c_int) -> *mut c_void;
    fn orb3_track(h: *mut c_void, l: *const u8, r: *const u8, w: c_int, hh: c_int, t: f64, imu: *const f64, n: c_int, twc: *mut f64, map_id: *mut u64, ref_kf: *mut i64, imu_init: *mut c_int) -> c_int;
    fn orb3_map_changed(h: *mut c_void) -> c_int;
    fn orb3_keyframes(h: *mut c_void, ids: *mut u64, poses: *mut f64, cap: c_int) -> c_int;
    fn orb3_free(h: *mut c_void);
}

/// 추적 상태(Tracking::eTrackingState)
pub const OK: i32 = 2;
pub const RECENTLY_LOST: i32 = 3;

pub struct Track {
    pub state: i32,
    pub twc: [f64; 12],
    pub map_id: u64,
    pub ref_kf: Option<u64>,
    /// 지도가 중력에 맞춰짐(스테레오-관성: IMU 초기화 + 첫 관성 정밀화. 스테레오는 늘 거짓)
    pub gravity: bool,
}

pub struct System(*mut c_void);

// SAFETY: 한 스레드(처리 루프)만 부른다. ORB-SLAM3 안의 지도 · 루프 스레드는 원본이 잠금으로 지킨다
unsafe impl Send for System {}

impl System {
    /// 단어장(ORBvoc.txt) · 설정 파일로 만든다(단어장 읽기에 수십 초). inertial = 스테레오-관성(IMU 표본을 넣음), 아니면 스테레오
    pub fn new(voc: &str, settings: &str, inertial: bool) -> Result<System, String> {
        let (v, s) = (CString::new(voc).map_err(|e| e.to_string())?, CString::new(settings).map_err(|e| e.to_string())?);
        // SAFETY: 두 C 문자열은 이 호출 동안 살아 있음
        let h = unsafe { orb3_new(v.as_ptr(), s.as_ptr(), inertial as c_int) };
        if h.is_null() {
            return Err(format!("ORB-SLAM3 를 만들지 못함(단어장 {voc}, 설정 {settings})"));
        }
        Ok(System(h))
    }

    /// 적외선 좌우 한 쌍 + 앞 장면 뒤 IMU 표본(t · 가속도 xyz · 자이로 xyz, 카메라 시계)
    pub fn track(&mut self, l: &[u8], r: &[u8], w: u32, h: u32, t: f64, imu: &[[f64; 7]]) -> Track {
        assert!(l.len() >= (w * h) as usize && r.len() >= (w * h) as usize);
        let mut o = Track { state: 0, twc: [0.0; 12], map_id: 0, ref_kf: None, gravity: false };
        let (mut kf, mut init) = (-1i64, 0 as c_int);
        // SAFETY: 영상 버퍼는 w×h 이상(위에서 확인), IMU 는 n×7 칸, 출력 칸은 이 함수의 지역 값
        o.state = unsafe { orb3_track(self.0, l.as_ptr(), r.as_ptr(), w as c_int, h as c_int, t, imu.as_ptr() as *const f64, imu.len() as c_int, o.twc.as_mut_ptr(), &mut o.map_id, &mut kf, &mut init) };
        o.ref_kf = (kf >= 0).then_some(kf as u64);
        o.gravity = init != 0;
        o
    }

    /// 앞에 물은 뒤로 큰 지도 바뀜(루프 클로저 · 지도 합치기 · 전체 최적화 · IMU 초기화 단계)이 있었나
    pub fn map_changed(&mut self) -> bool {
        // SAFETY: 살아 있는 핸들
        unsafe { orb3_map_changed(self.0) != 0 }
    }

    /// 현재 지도의 키프레임(id, 키프레임 카메라 → 세계)
    pub fn keyframes(&mut self) -> Vec<(u64, [f64; 12])> {
        let mut cap = 256usize;
        loop {
            let (mut ids, mut poses) = (vec![0u64; cap], vec![0f64; cap * 12]);
            // SAFETY: ids · poses 는 cap 칸
            let n = unsafe { orb3_keyframes(self.0, ids.as_mut_ptr(), poses.as_mut_ptr(), cap as c_int) } as usize;
            if n <= cap {
                return (0..n).map(|i| (ids[i], poses[i * 12..i * 12 + 12].try_into().unwrap())).collect();
            }
            cap = n + 64; // 그사이 늘었을 수 있어 넉넉히
        }
    }
}

impl Drop for System {
    fn drop(&mut self) {
        // SAFETY: 한 번만 놓음
        unsafe { orb3_free(self.0) }
    }
}
}

/// IMU 잡음(ICM-42688-P): ORB-SLAM3 RealSense_D435i.yaml 과 같은 값(연속 시간, rad/s/√Hz · m/s²/√Hz · 걸음)
const NOISE_GYRO: f64 = 1e-3;
const NOISE_ACC: f64 = 1e-2;
const GYRO_WALK: f64 = 1e-6;
const ACC_WALK: f64 = 1e-4;
const IMU_HZ: f64 = 200.0;

/// ORB-SLAM3 설정(File.version 1.0, 정렬된 스테레오). t_b_c1 = 적외선 왼쪽 카메라 → IMU(3×4 행 우선)
pub fn settings_yaml(s: &StereoCalib, t_b_c1: &[f64; 12], fps: f64) -> String {
    let m = |i: usize| t_b_c1[i];
    format!(
        "%YAML:1.0\n\
File.version: \"1.0\"\n\
Camera.type: \"Rectified\"\n\
Camera1.fx: {fx}\nCamera1.fy: {fy}\nCamera1.cx: {cx}\nCamera1.cy: {cy}\n\
Stereo.b: {b}\n\
Camera.width: {w}\nCamera.height: {h}\nCamera.fps: {fps}\nCamera.RGB: 1\n\
Stereo.ThDepth: 40.0\n\
IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 0.0, 0.0, 0.0, 1.0]\n\
IMU.InsertKFsWhenLost: 0\n\
IMU.NoiseGyro: {NOISE_GYRO:e}\nIMU.NoiseAcc: {NOISE_ACC:e}\nIMU.GyroWalk: {GYRO_WALK:e}\nIMU.AccWalk: {ACC_WALK:e}\nIMU.Frequency: {IMU_HZ:.1}\n\
ORBextractor.nFeatures: 1250\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n\
Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\nViewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\n\
Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -3.5\nViewer.ViewpointF: 500.0\n",
        m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8), m(9), m(10), m(11),
        fx = s.k[0], fy = s.k[1], cx = s.k[2], cy = s.k[3], b = s.baseline, w = s.w, h = s.h, fps = fps,
    )
}
