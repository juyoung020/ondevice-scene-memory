//! 위치 추적 부품. 물체 기억 · 점 지도 · 뷰어는 이 부품이 내는 장면 스트림(위치 · 루프 보정으로 움직인 노드)만 받는다.
//!
//! 흐름(장면마다): 입력(io: D435 · 녹화 재생, 적외선 좌우 · 컬러 · 정렬한 깊이) + IMU(imu)
//!                 → 위치 백엔드(backend: ORB-SLAM3 스테레오-관성, orb3) → 스트림(stream)
pub mod app;
pub mod backend;
pub mod bufpool;
pub mod imu;
pub mod io;
pub mod orb3;
pub mod simd;
pub mod stream;

/// 핀홀 카메라(왜곡 없음): 컬러(물체 기억 · 점 지도가 쓰는 격자)
#[derive(Clone, Debug)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

/// 깊이 버퍼(장면 · 스트림이 복사 없이 나눠 듦). 마지막으로 놓는 쪽이 카메라 버퍼로 돌려줌
pub type DepthBuf = std::sync::Arc<Vec<u16>>;

/// 적외선 좌우(D435 Y8I: 깊이 크기, 정렬된 스테레오, 회색 8 비트)
#[derive(Clone)]
pub struct Ir {
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub w: u32,
    pub h: u32,
    /// 적외선 시각(초, 장면 시각과 같은 시계)
    pub t: f64,
}

/// 위치 백엔드가 적외선 좌우를 쓰는 데 필요한 보정값(입력이 알려 줌)
#[derive(Clone, Debug)]
pub struct StereoCalib {
    /// 적외선 왼쪽 내부값 fx fy cx cy(깊이 내부값과 같음), 크기
    pub k: [f64; 4],
    pub w: u32,
    pub h: u32,
    /// 좌우 기준선(m)
    pub baseline: f64,
    /// 적외선 왼쪽(= 깊이) → 컬러: p_color = R p_ir + t, 3×4 행 우선
    pub color_from_ir: [f64; 12],
}

/// 한 장면: 컬러(스트림용, BGR 또는 RGB), 깊이(mm, 컬러 격자에 맞춤), 적외선 좌우(있으면)
#[derive(Clone)]
pub struct Frame {
    pub id: i64,
    pub t: f64,
    pub color: Vec<u8>,
    pub color_is_rgb: bool,
    pub depth_mm: DepthBuf,
    /// 원본 깊이 값 / 미터(스트림에 그대로 씀: RealSense 1000, TUM 5000). 비었고 1000/m 이면 depth_mm 이 곧 원본(`io::split_depth`)
    pub depth_raw: DepthBuf,
    pub depth_per_m: f32,
    pub cam: Camera,
    pub ir: Option<Ir>,
}

/// 카메라 입력이면 버려지는 장면의 버퍼를 드라이버에 돌려줌
impl Drop for Frame {
    fn drop(&mut self) {
        if bufpool::enabled() {
            bufpool::give_rgb(std::mem::take(&mut self.color));
            bufpool::give_depth_buf(std::mem::take(&mut self.depth_mm));
            bufpool::give_depth_buf(std::mem::take(&mut self.depth_raw));
        }
    }
}

impl Frame {
    /// 스트림으로 보낼 원본 깊이를 꺼냄(1000/m 이라 원본 칸을 비워 둔 장면은 mm 깊이가 곧 원본)
    pub fn take_stream_depth(&mut self) -> DepthBuf {
        if self.depth_raw.is_empty() && io::is_mm(self.depth_per_m) {
            std::mem::take(&mut self.depth_mm)
        } else {
            std::mem::take(&mut self.depth_raw)
        }
    }
}
