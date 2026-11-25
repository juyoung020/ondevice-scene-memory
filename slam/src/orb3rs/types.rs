//! 공통 자료 구조. 원본 이름을 그대로 쓴다(mvKeys → keys 처럼 접두사만 뺌). 필드 의미 · 단위는 원본 헤더 주석과 같다.
//! 바꿀 때는 docs/orb3_port.md "공통 인터페이스" 도 같이 고친다(다른 포터가 맞춤).
use std::collections::BTreeMap;

pub type KfId = u64;
pub type MpId = u64;
pub type MapId = u64;

/// cv::KeyPoint 와 같은 칸 · 자료형(pt · size · angle · response = f32, octave · class_id = i32)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
}

/// ORB 기술자 256 비트
pub type Desc = [u8; 32];

/// 기술자 해밍 거리(ORBmatcher::DescriptorDistance 와 같은 값)
#[inline]
pub fn desc_dist(a: &Desc, b: &Desc) -> i32 {
    let mut d = 0u32;
    for i in 0..4 {
        let x = u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()) ^ u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        d += x.count_ones();
    }
    d as i32
}

/// Sophus::SE3f 를 행렬 꼴로(r 행 우선 3×3, t). 곱 · 역의 연산 순서는 Sophus/Eigen 과 맞춰야 비트가 같다:
/// 원본 Sophus 는 회전을 단위 사원수로 들고 있으므로, 사원수를 거치는 연산을 옮길 때는 q 를 같이 든다.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3 {
    /// 단위 사원수 (x, y, z, w) — Eigen::Quaternionf 저장 순서
    pub q: [f32; 4],
    pub t: [f32; 3],
}

impl Default for Se3 {
    fn default() -> Self {
        Se3 { q: [0.0, 0.0, 0.0, 1.0], t: [0.0; 3] }
    }
}

/// 핀홀 카메라(ORB_SLAM3::Pinhole, 왜곡 0 — 우리 설정은 k1..p2 = 0)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub invfx: f32,
    pub invfy: f32,
    /// mbf = 기준선 × fx (RGB-D 는 설정의 Stereo.b × fx)
    pub bf: f32,
    pub b: f32,
    /// mThDepth = bf × ThDepth / fx
    pub th_depth: f32,
    pub width: i32,
    pub height: i32,
}

/// ORBextractor 설정과 층 표(원본 생성자가 만드는 값)
#[derive(Clone, Debug, PartialEq)]
pub struct Pyramid {
    pub nfeatures: i32,
    pub scale_factor: f32,
    pub nlevels: i32,
    pub ini_th_fast: i32,
    pub min_th_fast: i32,
    pub scale: Vec<f32>,
    pub inv_scale: Vec<f32>,
    pub sigma2: Vec<f32>,
    pub inv_sigma2: Vec<f32>,
    pub log_scale_factor: f32,
}

/// 격자(FRAME_GRID_ROWS 48 × COLS 64)
pub const GRID_ROWS: usize = 48;
pub const GRID_COLS: usize = 64;

/// Frame(장면). RGB-D 단안 쪽만(오른쪽 영상 · 어안 · IMU 사전적분 칸은 쓰지 않음)
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub id: u64,
    pub timestamp: f64,
    /// mvKeys(왜곡 없음이라 mvKeysUn 과 같음)
    pub keys: Vec<KeyPoint>,
    pub desc: Vec<Desc>,
    /// mvuRight(-1 = 없음) · mvDepth(-1 = 없음)
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,
    /// mvpMapPoints(None = 짝 없음) · mvbOutlier
    pub map_points: Vec<Option<MpId>>,
    pub outlier: Vec<bool>,
    /// mGrid[col][row] = 점 번호(원본 std::vector<size_t>, 넣는 순서 유지)
    pub grid: Vec<Vec<u32>>,
    pub grid_inv_w: f32,
    pub grid_inv_h: f32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    /// mTcw(있을 때)
    pub tcw: Option<Se3>,
    pub ref_kf: Option<KfId>,
    /// BoW(DBoW2 BowVector · FeatureVector): 단어 id → 무게, 노드 id → 점 번호들. 순서는 std::map 과 같게 BTreeMap
    pub bow: BTreeMap<u32, f64>,
    pub feat: BTreeMap<u32, Vec<u32>>,
}

impl Frame {
    /// 격자 칸 번호(col, row) 목록 접근: grid[col * GRID_ROWS + row]
    #[inline]
    pub fn cell(&self, col: usize, row: usize) -> &Vec<u32> {
        &self.grid[col * GRID_ROWS + row]
    }
}

/// KeyFrame. 원본 KeyFrame 의 그래프 칸(공시성 · 신장 트리 · 루프 가장자리) 포함
#[derive(Clone, Debug, Default)]
pub struct KeyFrame {
    pub id: KfId,
    pub frame_id: u64,
    pub map: MapId,
    pub timestamp: f64,
    pub keys: Vec<KeyPoint>,
    /// mvKeys(왜곡 보정 전 x, y). UnprojectStereo 가 씀
    pub keys_raw: Vec<[f32; 2]>,
    pub desc: Vec<Desc>,
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,
    pub map_points: Vec<Option<MpId>>,
    pub grid: Vec<Vec<u32>>,
    pub tcw: Se3,
    pub bad: bool,
    pub bow: BTreeMap<u32, f64>,
    pub feat: BTreeMap<u32, Vec<u32>>,
    /// mConnectedKeyFrameWeights · mvpOrderedConnectedKeyFrames/mvOrderedWeights(무게 내림차순)
    pub conn_weights: BTreeMap<KfId, i32>,
    pub ordered_conn: Vec<(KfId, i32)>,
    pub parent: Option<KfId>,
    pub children: Vec<KfId>,
    pub loop_edges: Vec<KfId>,
    pub merge_edges: Vec<KfId>,
    /// mTcp(나쁨 표시 때 부모 기준 위치)
    pub tcp: Se3,
}

/// MapPoint
#[derive(Clone, Debug, Default)]
pub struct MapPoint {
    pub id: MpId,
    pub map: MapId,
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub desc: Desc,
    /// mObservations: 키프레임 → (왼쪽 번호, 오른쪽 번호 -1)
    pub obs: BTreeMap<KfId, (i32, i32)>,
    pub ref_kf: KfId,
    pub first_kf: KfId,
    pub visible: i32,
    pub found: i32,
    pub min_dist: f32,
    pub max_dist: f32,
    pub bad: bool,
    pub replaced: Option<MpId>,
    /// Tracking 이 쓰는 칸(mbTrackInView, mTrackProjX/Y, mnTrackScaleLevel, mTrackViewCos, mnLastFrameSeen …)
    pub track_in_view: bool,
    pub track_proj_x: f32,
    pub track_proj_y: f32,
    pub track_scale_level: i32,
    pub track_view_cos: f32,
    pub last_frame_seen: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Map {
    pub id: MapId,
    pub kfs: Vec<KfId>,
    pub mps: Vec<MpId>,
    pub origin_kf: Option<KfId>,
    pub bad: bool,
    pub change_id: i32,
}

/// Atlas: 지도들 + 키프레임 · 지도 점 저장소. 원본의 std::set<KeyFrame*>(포인터 주소 순)은 id 순으로 바꾼다
/// (주소 순은 실행마다 다를 수 있어 원본 자체가 비결정; 비트 비교 하네스에서는 id 순이 되게 원본 쪽 순회를 맞춘다).
#[derive(Clone, Debug, Default)]
pub struct Atlas {
    pub maps: Vec<Map>,
    pub current: usize,
    pub kfs: BTreeMap<KfId, KeyFrame>,
    pub mps: BTreeMap<MpId, MapPoint>,
    pub next_kf: KfId,
    pub next_mp: MpId,
}
