//! ORB-SLAM3 RGB-D: 설정 파일 만들기와 C 함수(cpp/orb3.cc)를 부르는 얇은 층.
//! 위치는 3×4 행 우선 [f64; 12](카메라 → 세계, 컬러 카메라 광학 좌표, 세계 = ORB-SLAM3 지도 좌표).
use std::ffi::{c_char, c_int, c_void, CString};

unsafe extern "C" {
    fn orb3_new(voc: *const c_char, settings: *const c_char) -> *mut c_void;
    fn orb3_track(h: *mut c_void, rgb: *const u8, depth: *const u16, w: c_int, hh: c_int, t: f64, up: *const f64, twc: *mut f64, map_id: *mut u64, ref_kf: *mut i64) -> c_int;
    fn orb3_map_changed(h: *mut c_void) -> c_int;
    fn orb3_keyframes(h: *mut c_void, all: c_int, ids: *mut u64, maps: *mut u64, poses: *mut f64, cap: c_int) -> c_int;
    fn orb3_graft(h: *mut c_void, from: u64, into: u64, s: *const f64) -> c_int;
    fn orb3_wait_idle(h: *mut c_void);
    fn orb3_kf_now(h: *mut c_void, id: u64, twc: *mut f64, map: *mut u64) -> c_int;
    fn orb3_free(h: *mut c_void);
}

/// 추적 상태(Tracking::eTrackingState)
pub const OK: i32 = 2;

pub struct Track {
    pub state: i32,
    pub twc: [f64; 12],
    pub map_id: u64,
    pub ref_kf: Option<u64>,
}

pub struct System(*mut c_void);

// SAFETY: 한 스레드(처리 루프)만 부른다. ORB-SLAM3 안의 지도 · 루프 스레드는 원본이 잠금으로 지킨다
unsafe impl Send for System {}

impl System {
    /// 단어장(ORBvoc.txt) · 설정 파일로 만든다(단어장 읽기에 수 초)
    pub fn new(voc: &str, settings: &str) -> Result<System, String> {
        let (v, s) = (CString::new(voc).map_err(|e| e.to_string())?, CString::new(settings).map_err(|e| e.to_string())?);
        // SAFETY: 두 C 문자열은 이 호출 동안 살아 있음
        let h = unsafe { orb3_new(v.as_ptr(), s.as_ptr()) };
        if h.is_null() {
            return Err(format!("ORB-SLAM3 를 만들지 못함(단어장 {voc}, 설정 {settings})"));
        }
        Ok(System(h))
    }

    /// 컬러(3 채널, 설정의 Camera.RGB 순서) + 같은 격자 깊이(설정의 DepthMapFactor 단위) + IMU 위쪽(카메라 축 단위 벡터)
    pub fn track(&mut self, rgb: &[u8], depth: &[u16], w: u32, h: u32, t: f64, up: Option<[f64; 3]>) -> Track {
        let n = (w * h) as usize;
        assert!(rgb.len() >= n * 3 && depth.len() >= n);
        let mut o = Track { state: 0, twc: [0.0; 12], map_id: 0, ref_kf: None };
        let mut kf = -1i64;
        // SAFETY: 버퍼 크기는 위에서 확인, 출력 칸은 이 함수의 지역 값
        o.state = unsafe { orb3_track(self.0, rgb.as_ptr(), depth.as_ptr(), w as c_int, h as c_int, t, up.as_ref().map_or(std::ptr::null(), |u| u.as_ptr()), o.twc.as_mut_ptr(), &mut o.map_id, &mut kf) };
        o.ref_kf = (kf >= 0).then_some(kf as u64);
        o
    }

    /// 앞에 물은 뒤로 큰 지도 바뀜(루프 클로저 · 지도 합치기 · 옮겨 붙이기 · 전체 최적화)이 있었나
    pub fn map_changed(&mut self) -> bool {
        // SAFETY: 살아 있는 핸들
        unsafe { orb3_map_changed(self.0) != 0 }
    }

    /// 키프레임(id, 지도 id, 키프레임 카메라 → 세계). all = 모든 지도, 아니면 현재 지도만
    pub fn keyframes(&mut self, all: bool) -> Vec<(u64, u64, [f64; 12])> {
        let mut cap = 512usize;
        loop {
            let (mut ids, mut maps, mut poses) = (vec![0u64; cap], vec![0u64; cap], vec![0f64; cap * 12]);
            // SAFETY: ids · maps · poses 는 cap 칸
            let n = unsafe { orb3_keyframes(self.0, all as c_int, ids.as_mut_ptr(), maps.as_mut_ptr(), poses.as_mut_ptr(), cap as c_int) } as usize;
            if n <= cap {
                return (0..n).map(|i| (ids[i], maps[i], poses[i * 12..i * 12 + 12].try_into().unwrap())).collect();
            }
            cap = n + 64;
        }
    }
}

impl System {
    /// 지도 from 을 지도 into 좌표로 옮겨 붙이기 부탁(다음 장면 시작에 함). s = into 좌표 ← from 좌표
    pub fn graft(&mut self, from: u64, into: u64, s: &[f64; 12]) -> bool {
        // SAFETY: s 는 12 칸
        unsafe { orb3_graft(self.0, from, into, s.as_ptr()) != 0 }
    }
}

impl System {
    /// 지도 스레드가 쌓인 키프레임을 다 처리할 때까지 기다림(최대 속도 재생: 결과가 PC 속도에 덜 매이게)
    pub fn kf_now(&mut self, id: u64) -> Option<(u64, [f64; 12])> {
        let (mut t, mut m) = ([0f64; 12], 0u64);
        // SAFETY: t 는 12 칸
        (unsafe { orb3_kf_now(self.0, id, t.as_mut_ptr(), &mut m) } != 0).then_some((m, t))
    }

    /// 지도 스레드가 쌓인 키프레임을 다 처리할 때까지 기다림
    pub fn wait_idle(&mut self) {
        // SAFETY: 살아 있는 핸들
        unsafe { orb3_wait_idle(self.0) }
    }
}

impl Drop for System {
    fn drop(&mut self) {
        // SAFETY: 한 번만 놓음
        unsafe { orb3_free(self.0) }
    }
}

/// ORB-SLAM3 설정(File.version 1.0, 핀홀 RGB-D). depth_per_m = 깊이 값 / 미터, rgb = 컬러가 RGB 순서
pub fn settings_yaml(fx: f64, fy: f64, cx: f64, cy: f64, w: u32, h: u32, fps: f64, depth_per_m: f32, rgb: bool) -> String {
    format!(
        "%YAML:1.0\n\
File.version: \"1.0\"\n\
Camera.type: \"PinHole\"\n\
Camera1.fx: {fx:.6}\nCamera1.fy: {fy:.6}\nCamera1.cx: {cx:.6}\nCamera1.cy: {cy:.6}\n\
Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\nCamera1.p2: 0.0\n\
Camera.width: {w}\nCamera.height: {h}\nCamera.fps: {}\nCamera.RGB: {}\n\
Stereo.ThDepth: 40.0\nStereo.b: 0.05\n\
RGBD.DepthMapFactor: {depth_per_m:.1}\n\
ORBextractor.nFeatures: 1250\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n\
Viewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1.0\nViewer.GraphLineWidth: 0.9\nViewer.PointSize: 2.0\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3.0\n\
Viewer.ViewpointX: 0.0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -3.5\nViewer.ViewpointF: 500.0\n",
        fps.round() as i32,
        rgb as i32
    )
}
