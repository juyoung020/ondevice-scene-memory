//! RTAB-Map 0.24.2 에서 우리 설정(docs/rtabmap_pi.ini)이 지나가는 경로를 Rust 로 옮긴 것.
//! 원본: introlab/rtabmap 0.24.2 (BSD-3-Clause), OpenCV 4.10 (Apache-2.0) 의 해당 함수. 모듈마다 원본 파일·함수를 적는다.
//!
//! 흐름(장면마다): 입력(io) → 특징점(features) → 앞 장면과 짝짓기·PnP(register) → 위치 추적(odom)
//!                 1 초에 1 번: 지도 노드 추가 · 루프 클로저(loopclose) · 그래프 최적화(graph) → 스트림(stream)
pub mod features;
pub mod graph;
pub mod io;
pub mod loopclose;
pub mod odom;
pub mod params;
pub mod register;
pub mod stream;
pub mod app;

use nalgebra::{Isometry3, Point3};

/// 강체 변환(회전 + 이동, 미터). RTAB-Map Transform 과 같은 뜻: 몸체 축(x 앞, y 왼쪽, z 위)
pub type Pose = Isometry3<f64>;

/// 핀홀 카메라(왜곡 없음). local = 몸체 → 광학 축(RTAB-Map CameraModel::localTransform, 보통 opticalRotation)
#[derive(Clone, Debug)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
    pub local: Pose,
}

/// 한 장면: 회색조(특징점용), 컬러(스트림용, BGR 또는 RGB), 깊이(mm, 컬러 격자에 맞춤)
#[derive(Clone)]
pub struct Frame {
    pub id: i64,
    pub t: f64,
    pub gray: Vec<u8>,
    pub color: Vec<u8>,
    pub color_is_rgb: bool,
    pub depth_mm: Vec<u16>,
    /// 원본 깊이 값 / 미터(스트림에 그대로 씀: RealSense 1000, TUM 5000)
    pub depth_raw: Vec<u16>,
    pub depth_per_m: f32,
    pub cam: Camera,
}

/// 특징점(OpenCV KeyPoint 와 같은 칸)
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

/// 한 장면의 특징: 특징점, 그 3D 점(몸체 축, 깊이 없으면 NaN), BRIEF 32 바이트 설명자
#[derive(Clone, Default)]
pub struct Features {
    pub kps: Vec<KeyPoint>,
    pub pts3d: Vec<Point3<f32>>,
    pub desc: Vec<[u8; 32]>,
}
