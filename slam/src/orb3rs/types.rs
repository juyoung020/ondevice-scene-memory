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
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    /// mThDepth = 설정 Stereo.b × ThDepth (Tracking::newParameterLoader)
    pub th_depth: f32,
    pub width: i32,
    pub height: i32,
}

/// ORBextractor 설정과 층 표(원본 생성자가 만드는 값)
#[derive(Clone, Debug, Default, PartialEq)]
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

/// 장면 공통 값(원본 Frame 의 static: 카메라 · 층 표 · 영상 경계 · 격자 칸 크기의 역수)
#[derive(Clone, Debug, Default)]
pub struct FrameCtx {
    pub cam: Camera,
    pub pyr: Pyramid,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub grid_inv_w: f32,
    pub grid_inv_h: f32,
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
    /// mTcw(있을 때)와 거기서 나온 값(UpdatePoseMatrices: mRcw · mtcw · mRwc · mOw)
    pub tcw: Option<Se3>,
    pub rcw: M3,
    pub tcw_t: V3,
    pub rwc: M3,
    pub ow: V3,
    /// 카메라 · 층 표 · 영상 경계(모든 장면이 같이 씀)
    pub ctx: std::sync::Arc<FrameCtx>,
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
    pub desc: Vec<Desc>,
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,
    pub map_points: Vec<Option<MpId>>,
    pub grid: Vec<Vec<u32>>,
    /// mTcw · mTwc(SetPose 가 둘 다 둠)
    pub tcw: Se3,
    pub twc: Se3,
    pub ctx: std::sync::Arc<FrameCtx>,
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
    /// !mbFirstConnection(처음 연결로 부모를 정했음) · mbNotErase · mbToBeErased
    pub first_conn_done: bool,
    pub not_erase: bool,
    pub to_be_erased: bool,
    /// mTcwBefMerge · mTwcBefMerge(지도 합치기 전 위치, 본질 그래프 · 점 보정이 씀)
    pub tcw_bef_merge: Se3,
    pub twc_bef_merge: Se3,
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
    /// nObs(입체 관측은 2, 단안은 1 씩)
    pub n_obs: i32,
    pub first_frame: u64,
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
    pub track_proj_xr: f32,
    pub track_depth: f32,
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
    /// mnInitKFid · mnMaxKFid · mpKFlowerID · mvpReferenceMapPoints · mnBigChangeIdx · mnMapChangeNotified
    /// (kfs · mps 는 원본 std::set 처럼 id 순 정렬을 유지한다)
    pub init_kf_id: KfId,
    pub max_kf_id: KfId,
    pub lower_kf: Option<KfId>,
    pub ref_mps: Vec<MpId>,
    pub big_change_idx: i32,
    pub change_notified: i32,
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

// ---------- Sophus SE3f · Eigen float 연산(원본 빌드와 비트 동일, ~/opt/orb3_geom 으로 확인) ----------
// 기준 원본은 -ffp-contract=off 로 빌드한 ORB-SLAM3(docs/orb3_port.md "기준 빌드": GCC 기본값은 곱 · 덧셈을 FMA 로 합치는데
// 어느 곱을 합칠지가 인라인된 자리마다 달라 같은 식도 호출 자리마다 결과가 다름 → 소스 순서 그대로 계산하는 빌드를 기준으로 삼음).
// 크기 3 벡터 합은 Eigen 반 나눔 순서 a0 + (a1 + a2).
// 사원수 4 칸은 SSE 묶음이라 제곱합이 (x²+z²)+(y²+w²)(묶음 곱 · 가로 합, FMA 없음).
pub type V3 = [f32; 3];
pub type M3 = [[f32; 3]; 3];

#[inline]
pub fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
/// Eigen Vector3f::dot: a0b0 + (a1b1 + a2b2)
#[inline]
pub fn dot(a: &V3, b: &V3) -> f32 {
    a[0] * b[0] + (a[1] * b[1] + a[2] * b[2])
}
/// Eigen Vector3f::norm
#[inline]
pub fn norm(a: &V3) -> f32 {
    dot(a, a).sqrt()
}
/// Eigen Matrix3f * Vector3f(행마다 dot 과 같은 순서)
#[inline]
pub fn mat_vec(m: &M3, p: &V3) -> V3 {
    [dot(&m[0], p), dot(&m[1], p), dot(&m[2], p)]
}
#[inline]
pub fn add(a: &V3, b: &V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Eigen Quaternionf::normalize(SSE 묶음): 제곱합 (x²+z²)+(y²+w²), 칸마다 나눔
#[inline]
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let s = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    let n = s.sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

impl Se3 {
    /// Sophus::SE3f(Quaternionf q, t): 사원수를 정규화해서 들고 있음
    pub fn new(q: [f32; 4], t: V3) -> Se3 {
        Se3 { q: quat_normalize(q), t }
    }
    /// SO3 * 점: uv = v×p, uv += uv, (p + w·uv) + v×uv
    #[inline]
    pub fn rotate(&self, p: &V3) -> V3 {
        let v = [self.q[0], self.q[1], self.q[2]];
        let w = self.q[3];
        let mut uv = cross(&v, p);
        uv = [uv[0] + uv[0], uv[1] + uv[1], uv[2] + uv[2]];
        let c = cross(&v, &uv);
        [p[0] + w * uv[0] + c[0], p[1] + w * uv[1] + c[1], p[2] + w * uv[2] + c[2]]
    }
    /// SE3 * 점 = R p + t
    #[inline]
    pub fn mul_point(&self, p: &V3) -> V3 {
        add(&self.rotate(p), &self.t)
    }
    /// SE3::inverse: R⁻¹ = 켤레(정규화), t' = R⁻¹ (−t)
    pub fn inverse(&self) -> Se3 {
        let inv = Se3 { q: quat_normalize([-self.q[0], -self.q[1], -self.q[2], self.q[3]]), t: [0.0; 3] };
        let t = inv.rotate(&[-self.t[0], -self.t[1], -self.t[2]]);
        Se3 { q: inv.q, t }
    }
    /// SE3 * SE3: 사원수 곱(Sophus 식) 뒤 정규화, t = t_a + R_a t_b
    pub fn mul(&self, o: &Se3) -> Se3 {
        let (a, b) = (&self.q, &o.q);
        let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
        let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
        let q = [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ];
        Se3 { q: quat_normalize(q), t: add(&self.t, &self.rotate(&o.t)) }
    }
    /// Quaternionf::toRotationMatrix(행 우선)
    pub fn rot(&self) -> M3 {
        let (x, y, z, w) = (self.q[0], self.q[1], self.q[2], self.q[3]);
        let (tx, ty, tz) = (2.0 * x, 2.0 * y, 2.0 * z);
        let (twx, twy, twz) = (tx * w, ty * w, tz * w);
        let (txx, txy, txz) = (tx * x, ty * x, tz * x);
        let (tyy, tyz, tzz) = (ty * y, tz * y, tz * z);
        [[1.0 - (tyy + tzz), txy - twz, txz + twy], [txy + twz, 1.0 - (txx + tzz), tyz - twx], [txz - twy, tyz + twx, 1.0 - (txx + tyy)]]
    }
}

/// Eigen Matrix3f * Matrix3f(칸마다 행 · 열 dot: a0 + (a1 + a2))
pub fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + (a[i][1] * b[1][j] + a[i][2] * b[2][j]);
        }
    }
    r
}
pub fn transpose(a: &M3) -> M3 {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}
/// Sophus::SO3f::hat
pub fn hat(t: &V3) -> M3 {
    [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]
}
/// Eigen Matrix3f::inverse(InverseImpl.h compute_inverse 크기 3: 여인수 · 행렬식 1/det 곱)
pub fn mat_inv(m: &M3) -> M3 {
    let cof = |i: usize, j: usize| -> f32 {
        let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
        m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
    };
    let c0 = [cof(0, 0), cof(1, 0), cof(2, 0)];
    let det = c0[0] * m[0][0] + (c0[1] * m[1][0] + c0[2] * m[2][0]);
    let invdet = 1.0f32 / det;
    let mut r = [[0.0f32; 3]; 3];
    let c01 = cof(0, 1) * invdet;
    let c11 = cof(1, 1) * invdet;
    let c02 = cof(0, 2) * invdet;
    r[1][2] = cof(2, 1) * invdet;
    r[2][1] = cof(1, 2) * invdet;
    r[2][2] = cof(2, 2) * invdet;
    r[1][0] = c01;
    r[1][1] = c11;
    r[2][0] = c02;
    r[0] = [c0[0] * invdet, c0[1] * invdet, c0[2] * invdet];
    r
}

/// Eigen Quaternion(Matrix3) (Shoemake): 대각 합 (m00 + m11) + m22
pub fn quat_from_mat(m: &M3) -> [f32; 4] {
    let mut q = [0.0f32; 4];
    let t = (m[0][0] + m[1][1]) + m[2][2];
    if t > 0.0 {
        let t = (t + 1.0).sqrt();
        q[3] = 0.5 * t;
        let t = 0.5 / t;
        q[0] = (m[2][1] - m[1][2]) * t;
        q[1] = (m[0][2] - m[2][0]) * t;
        q[2] = (m[1][0] - m[0][1]) * t;
    } else {
        let mut i = 0;
        if m[1][1] > m[0][0] {
            i = 1;
        }
        if m[2][2] > m[i][i] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let t = ((m[i][i] - m[j][j]) - m[k][k] + 1.0).sqrt();
        q[i] = 0.5 * t;
        let t = 0.5 / t;
        q[3] = (m[k][j] - m[j][k]) * t;
        q[j] = (m[j][i] + m[i][j]) * t;
        q[k] = (m[k][i] + m[i][k]) * t;
    }
    q
}

/// Sophus::Sim3f: 정규화 안 한 사원수(제곱 노름 = 배율, x y z w) + t
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim3 {
    pub q: [f32; 4],
    pub t: V3,
}

/// Quaternion::squaredNorm(SSE 묶음): (x²+z²)+(y²+w²)
#[inline]
pub fn quat_sqnorm(q: &[f32; 4]) -> f32 {
    (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])
}

impl Sim3 {
    /// RxSO3::scale = 사원수 제곱 노름
    pub fn scale(&self) -> f32 {
        quat_sqnorm(&self.q)
    }
    /// RxSO3::rotationMatrix: 정규화한 사원수의 회전 행렬
    pub fn rotation_matrix(&self) -> M3 {
        Se3 { q: quat_normalize(self.q), t: [0.0; 3] }.rot()
    }
    /// RxSO3 * 점: s·p + (w·(2 v×p) + v×(2 v×p))
    #[inline]
    pub fn rxso3_mul(&self, p: &V3) -> V3 {
        let s = self.scale();
        let v = [self.q[0], self.q[1], self.q[2]];
        let w = self.q[3];
        let mut c = cross(&v, p);
        c = [c[0] + c[0], c[1] + c[1], c[2] + c[2]];
        let cc = cross(&v, &c);
        [s * p[0] + (w * c[0] + cc[0]), s * p[1] + (w * c[1] + cc[1]), s * p[2] + (w * c[2] + cc[2])]
    }
    /// Sim3 * 점
    #[inline]
    pub fn mul_point(&self, p: &V3) -> V3 {
        add(&self.rxso3_mul(p), &self.t)
    }
    /// Sim3::inverse: 사원수 역(켤레 / 제곱 노름), t' = R⁻¹(−t)
    pub fn inverse(&self) -> Sim3 {
        let n2 = quat_sqnorm(&self.q);
        let q = [-self.q[0] / n2, -self.q[1] / n2, -self.q[2] / n2, self.q[3] / n2];
        let r = Sim3 { q, t: [0.0; 3] };
        let t = r.rxso3_mul(&[-self.t[0], -self.t[1], -self.t[2]]);
        Sim3 { q, t }
    }
    /// SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale())
    pub fn to_se3(&self) -> Se3 {
        let r = self.rotation_matrix();
        let s = self.scale();
        Se3 { q: quat_from_mat(&r), t: [self.t[0] / s, self.t[1] / s, self.t[2] / s] }
    }
}
