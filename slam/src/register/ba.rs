//! RegistrationVis 의 두 장면 묶음 조정(Vis/BundleAdjustment=1 → OptimizerG2O::optimizeBA) 을 옮긴 것.
//! 원본: rtabmap corelib/src/optimizer/OptimizerG2O.cpp(optimizeBA), g2o 20230806
//! (OptimizationAlgorithmLevenberg, BlockSolver<6,3> 의 Schur 소거, 6x6 풀이는 독자 구현 up-looking Cholesky(chol6_solve),
//!  types/sba 의 SBACam·EdgeProjectP2MC·EdgeProjectP2SC·EdgeSBACam, RobustKernelHuber).
//! 고정 크기 Eigen 곱의 더하기 순서는 시험(저장소 밖 probe)으로 확인한 대로 따른다:
//!  - 열 우선 결과가 3 행이면 0·1 행은 앞에서부터, 2 행은 반씩 나눠(나무) 더함. 6 행·2 행은 앞에서부터
//!  - 왼쪽이 전치(Aᵀ·B)면 두 칸 묶음(패킷)으로 더한 뒤 묶음끼리 나무, 마지막에 두 칸 합
//!
//! 그래프: 카메라 1(고정, 기준 장면) · 카메라 2(풀림, 현재 장면) · 안쪽 점마다 3D 점(소거).
//! 변: 두 카메라 사이 상대 위치(PnP 공분산의 역), 점마다 두 카메라의 재투영(깊이 있으면 스테레오 흉내, Huber 8).
use super::rtf::Tf;

// ---- 더하기 순서 ----
#[inline]
fn d_seq(a: &[f64], b: &[f64]) -> f64 {
    let mut s = a[0] * b[0];
    for k in 1..a.len() {
        s = s + a[k] * b[k];
    }
    s
}
#[inline]
fn d_tree(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 1 {
        return a[0] * b[0];
    }
    let h = n / 2;
    d_tree(&a[..h], &b[..h]) + d_tree(&a[h..], &b[h..])
}
fn tree_sum(v: &[f64]) -> f64 {
    if v.len() == 1 {
        return v[0];
    }
    let h = v.len() / 2;
    tree_sum(&v[..h]) + tree_sum(&v[h..])
}
/// 패킷(2 칸) 나무 축약
#[inline]
fn d_ptree(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return a[0] * b[0];
    }
    let np = n / 2;
    let mut l0 = [0f64; 8];
    let mut l1 = [0f64; 8];
    for p in 0..np {
        l0[p] = a[2 * p] * b[2 * p];
        l1[p] = a[2 * p + 1] * b[2 * p + 1];
    }
    let mut r = tree_sum(&l0[..np]) + tree_sum(&l1[..np]);
    if n % 2 == 1 {
        r = r + a[n - 1] * b[n - 1];
    }
    r
}

/// 열 우선 A(R×K)·B(K×C) 한 칸: 3 행이면 2 행만 나무
#[inline]
fn mm_elem(rows: usize, i: usize, a: &[f64], b: &[f64]) -> f64 {
    if rows == 3 && i == 2 { d_tree(a, b) } else { d_seq(a, b) }
}

type M3 = [[f64; 3]; 3];
type M34 = [[f64; 4]; 3];

/// A(3×3)·B(3×4)
fn m3_m34(a: &M3, b: &M34) -> M34 {
    let mut o = [[0f64; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            let bc = [b[0][j], b[1][j], b[2][j]];
            o[i][j] = mm_elem(3, i, &a[i], &bc);
        }
    }
    o
}
fn m3_m3(a: &M3, b: &M3) -> M3 {
    let mut o = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bc = [b[0][j], b[1][j], b[2][j]];
            o[i][j] = mm_elem(3, i, &a[i], &bc);
        }
    }
    o
}
fn m3_v3(a: &M3, v: &[f64; 3]) -> [f64; 3] {
    [mm_elem(3, 0, &a[0], v), mm_elem(3, 1, &a[1], v), mm_elem(3, 2, &a[2], v)]
}
fn m34_v4(a: &M34, v: &[f64; 4]) -> [f64; 3] {
    [mm_elem(3, 0, &a[0], v), mm_elem(3, 1, &a[1], v), mm_elem(3, 2, &a[2], v)]
}

// ---- 쿼터니언(double, Eigen SSE 판) [x,y,z,w] ----
type Q = [f64; 4];
fn q_mul(a: &Q, b: &Q) -> Q {
    let (x, y, z, w) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [(w * bx + y * bz) - (z * by - x * bw), (w * by + y * bw) + (z * bx - x * bz), (w * bz - y * bx) + (z * bw + x * by), (w * bw - y * by) - (z * bz + x * bx)]
}
fn q_normalize(q: &mut Q) {
    let n = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    if n > 0.0 {
        let s = n.sqrt();
        for v in q.iter_mut() {
            *v /= s;
        }
    }
}
fn q_to_mat(q: &Q) -> M3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * x;
    let ty = 2.0 * y;
    let tz = 2.0 * z;
    let twx = tx * w;
    let twy = ty * w;
    let twz = tz * w;
    let txx = tx * x;
    let txy = ty * x;
    let txz = tz * x;
    let tyy = ty * y;
    let tyz = tz * y;
    let tzz = tz * z;
    [[1.0 - (tyy + tzz), txy - twz, txz + twy], [txy + twz, 1.0 - (txx + tzz), tyz - twx], [txz - twy, tyz + twx, 1.0 - (txx + tyy)]]
}
fn q_from_mat(m: &M3) -> Q {
    let t = m[0][0] + (m[1][1] + m[2][2]);
    let mut q = [0f64; 4];
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
        let t = (m[i][i] - m[j][j] - m[k][k] + 1.0).sqrt();
        q[i] = 0.5 * t;
        let t = 0.5 / t;
        q[3] = (m[k][j] - m[j][k]) * t;
        q[j] = (m[j][i] + m[i][j]) * t;
        q[k] = (m[k][i] + m[i][k]) * t;
    }
    q
}
#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
/// Quaternion * Vector3 (_transformVector)
fn q_rot(q: &Q, v: &[f64; 3]) -> [f64; 3] {
    let qv = [q[0], q[1], q[2]];
    let mut uv = cross(&qv, v);
    uv = [uv[0] + uv[0], uv[1] + uv[1], uv[2] + uv[2]];
    let c = cross(&qv, &uv);
    [(v[0] + q[3] * uv[0]) + c[0], (v[1] + q[3] * uv[1]) + c[1], (v[2] + q[3] * uv[2]) + c[2]]
}

/// g2o::SE3Quat
#[derive(Clone, Copy)]
struct Se3 {
    r: Q,
    t: [f64; 3],
}
impl Se3 {
    fn normalize_rotation(&mut self) {
        if self.r[3] < 0.0 {
            for v in self.r.iter_mut() {
                *v *= -1.0;
            }
        }
        q_normalize(&mut self.r);
    }
    fn from_mat(r: &M3, t: [f64; 3]) -> Se3 {
        let mut s = Se3 { r: q_from_mat(r), t };
        s.normalize_rotation();
        s
    }
    fn inverse(&self) -> Se3 {
        let r = [-self.r[0], -self.r[1], -self.r[2], self.r[3]];
        let mt = [self.t[0] * -1.0, self.t[1] * -1.0, self.t[2] * -1.0];
        Se3 { r, t: q_rot(&r, &mt) }
    }
    fn mul(&self, o: &Se3) -> Se3 {
        let rt = q_rot(&self.r, &o.t);
        let mut s = Se3 { r: q_mul(&self.r, &o.r), t: [self.t[0] + rt[0], self.t[1] + rt[1], self.t[2] + rt[2]] };
        s.normalize_rotation();
        s
    }
}

/// g2o::SBACam (+ VertexCam 의 캐시)
#[derive(Clone, Copy)]
struct Cam {
    se3: Se3,
    kcam: M3,
    baseline: f64,
    w2n: M34,
    w2i: M34,
    drdx: M3,
    drdy: M3,
    drdz: M3,
}
const DRIDX: M3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, -2.0, 0.0]];
const DRIDY: M3 = [[0.0, 0.0, -2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
const DRIDZ: M3 = [[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
impl Cam {
    fn new(r: Q, t: [f64; 3], fx: f64, fy: f64, cx: f64, cy: f64, baseline: f64) -> Cam {
        let mut se3 = Se3 { r, t };
        se3.normalize_rotation();
        let kcam = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]];
        let mut c = Cam { se3, kcam, baseline, w2n: [[0.0; 4]; 3], w2i: [[0.0; 4]; 3], drdx: [[0.0; 3]; 3], drdy: [[0.0; 3]; 3], drdz: [[0.0; 3]; 3] };
        c.refresh();
        c
    }
    fn refresh(&mut self) {
        // setTransform: w2n = [Rᵀ | -Rᵀ t]
        let r = q_to_mat(&self.se3.r);
        let mut m = [[0f64; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[j][i];
            }
        }
        let tt = [self.se3.t[0], self.se3.t[1], self.se3.t[2], 1.0];
        let neg = [[-m[0][0], -m[0][1], -m[0][2], -m[0][3]], [-m[1][0], -m[1][1], -m[1][2], -m[1][3]], [-m[2][0], -m[2][1], -m[2][2], -m[2][3]]];
        let c3 = m34_v4(&neg, &tt);
        for i in 0..3 {
            m[i][3] = c3[i];
        }
        self.w2n = m;
        self.w2i = m3_m34(&self.kcam, &self.w2n);
        let r3 = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
        self.drdx = m3_m3(&DRIDX, &r3);
        self.drdy = m3_m3(&DRIDY, &r3);
        self.drdz = m3_m3(&DRIDZ, &r3);
    }
    /// SBACam::update
    fn oplus(&mut self, u: &[f64]) {
        self.se3.t = [self.se3.t[0] + u[0], self.se3.t[1] + u[1], self.se3.t[2] + u[2]];
        let sq = (u[3] * u[3] + u[4] * u[4]) + u[5] * u[5];
        let qr = [u[3], u[4], u[5], (1.0 - sq).sqrt()];
        self.se3.r = q_mul(&self.se3.r, &qr);
        q_normalize(&mut self.se3.r);
        self.refresh();
    }
}

/// 한 관측(점 → 카메라). dim 2(단안) 또는 3(스테레오 흉내)
#[derive(Clone, Copy)]
struct Obs {
    point: usize,
    cam: usize, // 0 = 카메라 1(고정), 1 = 카메라 2
    stereo: bool,
    meas: [f64; 3],
    err: [f64; 3],
    level: u8,
    word: usize,
}

const HUBER: f64 = 8.0;

/// 한 번의 BA 입력
pub struct BaInput<'a> {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub local: Tf,
    /// PnP 결과(기준 → 현재 장면, 몸체 축)
    pub transform: Tf,
    /// PnP 공분산(대각, 엡실론 바닥 적용 뒤)
    pub cov_diag: [f64; 6],
    /// 안쪽 점: (기준 장면 3D 점, 기준 화소, 기준 깊이, 현재 화소, 현재 깊이)
    pub points: &'a [BaPoint],
    pub iterations: i32,
    pub baseline: f64,
    /// 정보 행렬은 Optimizer/PixelVariance = DisparityVariance = 1(기본) 의 단위 행렬로 둔다
    pub robust_delta: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BaPoint {
    pub p3: [f32; 3],
    pub kp_from: [f32; 2],
    pub depth_from: f32,
    pub kp_to: [f32; 2],
    pub depth_to: f32,
}

pub struct BaOutput {
    /// 최적화된 현재 장면 위치(None 이면 실패)
    pub transform: Option<Tf>,
    /// 카메라 2 관측이 바깥값이 된 점 번호(입력 순서)
    pub outliers_to: Vec<bool>,
}

struct Graph {
    cams: [Cam; 2],
    pts: Vec<[f64; 3]>,
    obs: Vec<Obs>,
    // 두 카메라 사이 변
    meas_inv: Se3,
    info6: [[f64; 6]; 6],
    link_err: [f64; 6],
}

impl Graph {
    fn obs_error(&self, o: &Obs) -> [f64; 3] {
        let cam = &self.cams[o.cam];
        let p = self.pts[o.point];
        let pt = [p[0], p[1], p[2], 1.0];
        if !o.stereo {
            let pp = m34_v4(&cam.w2i, &pt);
            let perr = [pp[0] / pp[2], pp[1] / pp[2]];
            [perr[0] - o.meas[0], perr[1] - o.meas[1], 0.0]
        } else {
            let p1 = m34_v4(&cam.w2i, &pt);
            let p2 = m34_v4(&cam.w2n, &pt);
            let invp1 = 1.0 / p1[2];
            let kp0 = p1[0] * invp1;
            let kp1 = p1[1] * invp1;
            let d = [p2[0] - cam.baseline, p2[1] - 0.0, p2[2] - 0.0];
            let p2 = m3_v3(&cam.kcam, &d);
            let kp2 = p2[0] / p2[2];
            [kp0 - o.meas[0], kp1 - o.meas[1], kp2 - o.meas[2]]
        }
    }

    fn link_error(&self) -> [f64; 6] {
        let d = self.meas_inv.mul(&self.cams[0].se3.inverse().mul(&self.cams[1].se3));
        [d.t[0], d.t[1], d.t[2], d.r[0], d.r[1], d.r[2]]
    }

    fn compute_errors(&mut self) {
        self.link_err = self.link_error();
        for k in 0..self.obs.len() {
            if self.obs[k].level == 0 {
                let e = self.obs_error(&self.obs[k]);
                self.obs[k].err = e;
            }
        }
    }

    /// 단위 정보 행렬 변의 chi2 = eᵀ(I e)
    fn obs_chi2(o: &Obs) -> f64 {
        // information = 단위(pixelVariance = disparityVariance = 1) 일 때 I·e 는 e 그대로
        let e = &o.err;
        if o.stereo {
            let ie = obs_info_mul(e, 3);
            (e[0] * ie[0] + e[1] * ie[1]) + e[2] * ie[2]
        } else {
            let ie = obs_info_mul(e, 2);
            e[0] * ie[0] + e[1] * ie[1]
        }
    }

    fn link_chi2(&self) -> f64 {
        let e = &self.link_err;
        let mut ie = [0f64; 6];
        for i in 0..6 {
            ie[i] = d_seq(&self.info6[i], e);
        }
        d_ptree(e, &ie)
    }

    fn active_robust_chi2(&self) -> f64 {
        let mut chi = 0.0;
        chi += self.link_chi2();
        for o in &self.obs {
            if o.level != 0 {
                continue;
            }
            let c = Graph::obs_chi2(o);
            chi += huber(c, HUBER)[0];
        }
        chi
    }
}

/// 단위 정보 행렬과 오차의 곱(I·e). 3 행 규칙(2 행은 나무)을 그대로 따름
#[inline]
fn obs_info_mul(e: &[f64; 3], d: usize) -> [f64; 3] {
    if d == 2 {
        [1.0 * e[0] + 0.0 * e[1], 0.0 * e[0] + 1.0 * e[1], 0.0]
    } else {
        [(1.0 * e[0] + 0.0 * e[1]) + 0.0 * e[2], (0.0 * e[0] + 1.0 * e[1]) + 0.0 * e[2], 0.0 * e[0] + (0.0 * e[1] + 1.0 * e[2])]
    }
}

/// RobustKernelHuber::robustify
#[inline]
fn huber(e: f64, delta: f64) -> [f64; 3] {
    let dsqr = delta * delta;
    if e <= dsqr {
        [e, 1.0, 0.0]
    } else {
        let sqrte = e.sqrt();
        let r1 = delta / sqrte;
        [2.0 * sqrte * delta - dsqr, r1, -0.5 * r1 / e]
    }
}

/// 점 관측의 야코비안(EdgeProjectP2MC/P2SC::linearizeOplus). 반환: (점 D×3, 카메라 D×6)
fn obs_jacobian(cam: &Cam, p: &[f64; 3], stereo: bool) -> ([[f64; 3]; 3], [[f64; 6]; 3]) {
    let pt = [p[0], p[1], p[2], 1.0];
    let trans = [cam.se3.t[0], cam.se3.t[1], cam.se3.t[2], 1.0];
    let pc = m34_v4(&cam.w2n, &pt);
    let (px, py, pz) = (pc[0], pc[1], pc[2]);
    let ipz2 = 1.0 / (pz * pz);
    let ipz2fx = ipz2 * cam.kcam[0][0];
    let ipz2fy = ipz2 * cam.kcam[1][1];
    let b = cam.baseline;
    let pwt = [pt[0] - trans[0], pt[1] - trans[1], pt[2] - trans[2]];
    let mut jc = [[0f64; 6]; 3];
    let mut jp = [[0f64; 3]; 3];
    let set = |j: &mut [[f64; 6]; 3], col: usize, dp: [f64; 3]| {
        j[0][col] = (pz * dp[0] - px * dp[2]) * ipz2fx;
        j[1][col] = (pz * dp[1] - py * dp[2]) * ipz2fy;
        if stereo {
            j[2][col] = (pz * dp[0] - (px - b) * dp[2]) * ipz2fx;
        }
    };
    set(&mut jc, 3, m3_v3(&cam.drdx, &pwt));
    set(&mut jc, 4, m3_v3(&cam.drdy, &pwt));
    set(&mut jc, 5, m3_v3(&cam.drdz, &pwt));
    for c in 0..3 {
        set(&mut jc, c, [-cam.w2n[0][c], -cam.w2n[1][c], -cam.w2n[2][c]]);
    }
    for c in 0..3 {
        let dp = [cam.w2n[0][c], cam.w2n[1][c], cam.w2n[2][c]];
        jp[0][c] = (pz * dp[0] - px * dp[2]) * ipz2fx;
        jp[1][c] = (pz * dp[1] - py * dp[2]) * ipz2fy;
        if stereo {
            jp[2][c] = (pz * dp[0] - (px - b) * dp[2]) * ipz2fx;
        }
    }
    (jp, jc)
}

/// 6x6 대칭 양정치 풀이(H·x = b). 실패(양정치 아님)면 false.
/// 알고리즘 설명(위로 보는 up-looking Cholesky: 행 k 마다 L[0..k,0..k]·l = H[0..k,k] 를 열 순서 전진 대입,
/// 대각 = sqrt(H_kk - Σ l²) → 전진 대입 L·y = b(열 순서) → 후진 대입 Lᵀ·x = y(행 순서)) 에서 직접 쓴 독자 구현.
/// 고정 크기·스택 배열·할당 없음. 원본 g2o 풀이와 비트 단위로 같은 결과(저장소 밖 무작위 2백만 건·전 시퀀스로 확인).
fn chol6_solve(h: &[[f64; 6]; 6], b: &[f64; 6], x: &mut [f64; 6]) -> bool {
    let mut l = [[0f64; 6]; 6];
    for k in 0..6 {
        let mut y = [0f64; 6];
        for j in 0..k {
            y[j] = h[j][k];
        }
        let mut d = h[k][k];
        for j in 0..k {
            let lkj = y[j] / l[j][j];
            for i in j + 1..k {
                y[i] -= l[i][j] * lkj;
            }
            d -= lkj * lkj;
            l[k][j] = lkj;
        }
        if d <= 0.0 {
            return false;
        }
        l[k][k] = d.sqrt();
    }
    let mut y = *b;
    for j in 0..6 {
        y[j] /= l[j][j];
        for i in j + 1..6 {
            y[i] -= l[i][j] * y[j];
        }
    }
    for j in (0..6).rev() {
        for i in j + 1..6 {
            y[j] -= l[i][j] * y[i];
        }
        y[j] /= l[j][j];
    }
    *x = y;
    true
}

/// Eigen 3x3 역행렬(여인수)
fn inv3(m: &M3) -> M3 {
    let cof = |i: usize, j: usize| -> f64 {
        let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
        m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
    };
    let c0 = [cof(0, 0), cof(1, 0), cof(2, 0)];
    let det = (c0[0] * m[0][0] + c0[1] * m[1][0]) + c0[2] * m[2][0];
    let invdet = 1.0 / det;
    let mut r = [[0f64; 3]; 3];
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

/// 정규 방정식(카메라 2 + 점들)
struct System {
    hpp: [[f64; 6]; 6],
    bp: [f64; 6],
    hll: Vec<M3>,
    bl: Vec<[f64; 3]>,
    /// 점마다 Hpl(6×3), 카메라 2 관측이 활성일 때만
    hpl: Vec<Option<[[f64; 3]; 6]>>,
}

fn build_system(g: &Graph, active_pts: &[bool], sys: &mut System) {
    sys.hpp = [[0.0; 6]; 6];
    sys.bp = [0.0; 6];
    for i in 0..g.pts.len() {
        sys.hll[i] = [[0.0; 3]; 3];
        sys.bl[i] = [0.0; 3];
        sys.hpl[i] = None;
    }
    // 1) 두 카메라 사이 변(수치 야코비안, 카메라 2 만)
    {
        let delta = 1e-9f64;
        let scalar = 1.0 / (2.0 * delta);
        let mut jac = [[0f64; 6]; 6]; // jac[row][col]
        let mut g2 = Graph { cams: g.cams, pts: Vec::new(), obs: Vec::new(), meas_inv: g.meas_inv, info6: g.info6, link_err: [0.0; 6] };
        for d in 0..6 {
            let saved = g2.cams[1];
            let mut add = [0f64; 6];
            add[d] = delta;
            g2.cams[1].oplus(&add);
            let e1 = g2.link_error();
            g2.cams[1] = saved;
            add[d] = -delta;
            g2.cams[1].oplus(&add);
            let e2 = g2.link_error();
            g2.cams[1] = saved;
            for r in 0..6 {
                jac[r][d] = scalar * (e1[r] - e2[r]);
            }
        }
        let e = &g.link_err;
        // omega = Ω, 가중 오차 = -(Ω e)
        let mut we = [0f64; 6];
        for i in 0..6 {
            we[i] = -d_seq(&g.info6[i], e);
        }
        // AtO = Jᵀ Ω (전치 왼쪽: 패킷 나무)
        let mut ato = [[0f64; 6]; 6];
        for i in 0..6 {
            let a: [f64; 6] = [jac[0][i], jac[1][i], jac[2][i], jac[3][i], jac[4][i], jac[5][i]];
            for j in 0..6 {
                let o: [f64; 6] = [g.info6[0][j], g.info6[1][j], g.info6[2][j], g.info6[3][j], g.info6[4][j], g.info6[5][j]];
                ato[i][j] = d_ptree(&a, &o);
            }
        }
        for i in 0..6 {
            let a: [f64; 6] = [jac[0][i], jac[1][i], jac[2][i], jac[3][i], jac[4][i], jac[5][i]];
            sys.bp[i] = sys.bp[i] + d_ptree(&a, &we);
        }
        for i in 0..6 {
            for j in 0..6 {
                let bcol: [f64; 6] = [jac[0][j], jac[1][j], jac[2][j], jac[3][j], jac[4][j], jac[5][j]];
                sys.hpp[i][j] = sys.hpp[i][j] + d_seq(&ato[i], &bcol);
            }
        }
    }
    // 2) 재투영 변(입력 순서: 점마다 카메라 1, 카메라 2)
    for o in &g.obs {
        if o.level != 0 || !active_pts[o.point] {
            continue;
        }
        let dd = if o.stereo { 3 } else { 2 };
        let cam = &g.cams[o.cam];
        let (jp, jc) = obs_jacobian(cam, &g.pts[o.point], o.stereo);
        let chi = Graph::obs_chi2(o);
        let rho = huber(chi, HUBER);
        let ie = obs_info_mul(&o.err, dd);
        // omega_r = (-(Ω e)) * rho1, 강건 정보 = rho1 * Ω
        let mut we = [0f64; 3];
        for i in 0..dd {
            we[i] = -ie[i] * rho[1];
        }
        let mut om = [[0f64; 3]; 3];
        for i in 0..dd {
            for j in 0..dd {
                om[i][j] = rho[1] * if i == j { 1.0 } else { 0.0 };
            }
        }
        // 점(꼭짓점 0)
        let mut ato_p = [[0f64; 3]; 3]; // 3×D
        for i in 0..3 {
            let a = [jp[0][i], jp[1][i], jp[2][i]];
            for j in 0..dd {
                let oc = [om[0][j], om[1][j], om[2][j]];
                ato_p[i][j] = d_ptree(&a[..dd], &oc[..dd]);
            }
        }
        let p = o.point;
        for i in 0..3 {
            let a = [jp[0][i], jp[1][i], jp[2][i]];
            sys.bl[p][i] = sys.bl[p][i] + d_ptree(&a[..dd], &we[..dd]);
        }
        for i in 0..3 {
            for j in 0..3 {
                let bc = [jp[0][j], jp[1][j], jp[2][j]];
                sys.hll[p][i][j] = sys.hll[p][i][j] + mm_elem(3, i, &ato_p[i][..dd], &bc[..dd]);
            }
        }
        if o.cam == 1 {
            // 비대각(점-카메라 2): Hpl(6×3) += Jcᵀ · AtO_pᵀ
            let mut blk = sys.hpl[p].unwrap_or([[0.0; 3]; 6]);
            for r in 0..6 {
                let bcol = [jc[0][r], jc[1][r], jc[2][r]];
                for c in 0..3 {
                    let arow = [ato_p[c][0], ato_p[c][1], ato_p[c][2]];
                    blk[r][c] = blk[r][c] + d_tree(&bcol[..dd], &arow[..dd]); // 시험: 이 식은 나무 순서
                }
            }
            sys.hpl[p] = Some(blk);
            // 카메라 2(꼭짓점 1)
            let mut ato_c = [[0f64; 3]; 6];
            for i in 0..6 {
                let a = [jc[0][i], jc[1][i], jc[2][i]];
                for j in 0..dd {
                    let oc = [om[0][j], om[1][j], om[2][j]];
                    ato_c[i][j] = d_ptree(&a[..dd], &oc[..dd]);
                }
            }
            for i in 0..6 {
                let a = [jc[0][i], jc[1][i], jc[2][i]];
                sys.bp[i] = sys.bp[i] + d_ptree(&a[..dd], &we[..dd]);
            }
            for i in 0..6 {
                for j in 0..6 {
                    let bc = [jc[0][j], jc[1][j], jc[2][j]];
                    sys.hpp[i][j] = sys.hpp[i][j] + d_seq(&ato_c[i][..dd], &bc[..dd]);
                }
            }
        }
    }
}

/// BlockSolver::solve(Schur) + 점 되풀이. x = [카메라 2 (6), 점마다 3]
/// lambda: setLambda 와 같은 대각 더하기(원본은 제자리에 더했다가 되돌림, 여기선 복사본에 더함)
fn solve_schur(sys: &System, active: &[usize], lambda: f64, dinv: &mut Vec<M3>, x: &mut [f64]) -> bool {
    let mut hs = sys.hpp;
    for j in 0..6 {
        hs[j][j] += lambda;
    }
    let mut coef = [0f64; 6];
    dinv.clear();
    for &p in active {
        let mut d = sys.hll[p];
        for j in 0..3 {
            d[j][j] += lambda;
        }
        let di = inv3(&d);
        let db0 = sys.bl[p];
        let db = m3_v3(&di, &db0);
        if let Some(bi) = &sys.hpl[p] {
            // BDinv = Bi·Dinv(6 행: 앞에서부터)
            let mut bd = [[0f64; 3]; 6];
            for r in 0..6 {
                for c in 0..3 {
                    let dc = [di[0][c], di[1][c], di[2][c]];
                    bd[r][c] = d_seq(&bi[r], &dc);
                }
            }
            for r in 0..6 {
                coef[r] = coef[r] + d_seq(&bi[r], &db);
            }
            for r in 0..6 {
                for c in 0..6 {
                    hs[r][c] = hs[r][c] - d_seq(&bd[r], &bi[c]);
                }
            }
        }
        dinv.push(di);
    }
    let mut bs = [0f64; 6];
    for i in 0..6 {
        bs[i] = sys.bp[i] - coef[i];
    }
    let mut xp = [0f64; 6];
    if !chol6_solve(&hs, &bs, &mut xp) {
        return false;
    }
    x[..6].copy_from_slice(&xp);
    let cp = [-xp[0], -xp[1], -xp[2], -xp[3], -xp[4], -xp[5]];
    for (k, &p) in active.iter().enumerate() {
        let mut cl = sys.bl[p];
        if let Some(bi) = &sys.hpl[p] {
            for c in 0..3 {
                let col = [bi[0][c], bi[1][c], bi[2][c], bi[3][c], bi[4][c], bi[5][c]];
                cl[c] = cl[c] + d_ptree(&col, &cp);
            }
        }
        let v = m3_v3(&dinv[k], &cl);
        for c in 0..3 {
            x[6 + 3 * k + c] = 0.0 + v[c];
        }
    }
    true
}

/// optimize(iterations): LM. 반환: 끝까지 성공했는지(실패 = Fail)
fn levenberg(g: &mut Graph, iterations: i32) {
    // 활성 점: 레벨 0 관측이 하나라도 있는 점(번호 순)
    let npts = g.pts.len();
    let mut active_pts = vec![false; npts];
    for o in &g.obs {
        if o.level == 0 {
            active_pts[o.point] = true;
        }
    }
    let active: Vec<usize> = (0..npts).filter(|&p| active_pts[p]).collect();
    let dim = 6 + 3 * active.len();
    let mut sys = System { hpp: [[0.0; 6]; 6], bp: [0.0; 6], hll: vec![[[0.0; 3]; 3]; npts], bl: vec![[0.0; 3]; npts], hpl: vec![None; npts] };
    let mut x = vec![0f64; dim];
    let mut b = vec![0f64; dim];
    let mut backup_pts: Vec<[f64; 3]> = Vec::with_capacity(active.len());
    let mut dinv_buf: Vec<M3> = Vec::with_capacity(active.len());
    let mut lambda = -1.0f64;
    let mut ni = 2.0f64;
    let tau = 1e-5f64;
    for it in 0..iterations {
        g.compute_errors();
        let mut current_chi = g.active_robust_chi2();
        build_system(g, &active_pts, &mut sys);
        b[..6].copy_from_slice(&sys.bp);
        for (k, &p) in active.iter().enumerate() {
            b[6 + 3 * k..6 + 3 * k + 3].copy_from_slice(&sys.bl[p]);
        }
        if it == 0 {
            let mut maxd = 0f64;
            for j in 0..6 {
                maxd = max_std(sys.hpp[j][j].abs(), maxd);
            }
            for &p in &active {
                for j in 0..3 {
                    maxd = max_std(sys.hll[p][j][j].abs(), maxd);
                }
            }
            lambda = tau * maxd;
            ni = 2.0;
        }
        let mut rho;
        let mut qmax = 0;
        loop {
            let backup_cams = g.cams;
            backup_pts.clear();
            backup_pts.extend(active.iter().map(|&p| g.pts[p]));
            let ok2 = solve_schur(&sys, &active, lambda, &mut dinv_buf, &mut x);
            // update
            let xc: [f64; 6] = [x[0], x[1], x[2], x[3], x[4], x[5]];
            g.cams[1].oplus(&xc);
            for (k, &p) in active.iter().enumerate() {
                let v = &x[6 + 3 * k..6 + 3 * k + 3];
                g.pts[p] = [g.pts[p][0] + v[0], g.pts[p][1] + v[1], g.pts[p][2] + v[2]];
            }
            g.compute_errors();
            let mut temp_chi = g.active_robust_chi2();
            if !ok2 {
                temp_chi = f64::MAX;
            }
            rho = current_chi - temp_chi;
            let scale = if ok2 {
                let mut s = 0f64;
                for j in 0..dim {
                    s += x[j] * (lambda * x[j] + b[j]);
                }
                s + 1e-3
            } else {
                1.0
            };
            rho /= scale;
            if rho > 0.0 && temp_chi.is_finite() && ok2 {
                let mut alpha = 1.0 - (2.0 * rho - 1.0).powf(3.0);
                alpha = alpha.min(2.0 / 3.0);
                let sf = (1.0f64 / 3.0).max(alpha);
                lambda *= sf;
                ni = 2.0;
                current_chi = temp_chi;
            } else {
                lambda *= ni;
                ni *= 2.0;
                g.cams = backup_cams;
                for (k, &p) in active.iter().enumerate() {
                    g.pts[p] = backup_pts[k];
                }
                if !lambda.is_finite() {
                    break;
                }
            }
            qmax += 1;
            if !(rho < 0.0 && qmax < 10) {
                break;
            }
        }
        TRACE.with(|t| {
            if let Some(v) = t.borrow_mut().as_mut() {
                let c = &g.cams[1].se3;
                v.push(format!("it {it} lambda {:x} chi {:x} q {:x} {:x} {:x} {:x} t {:x} {:x} {:x}", lambda.to_bits(), current_chi.to_bits(), c.r[0].to_bits(), c.r[1].to_bits(), c.r[2].to_bits(), c.r[3].to_bits(), c.t[0].to_bits(), c.t[1].to_bits(), c.t[2].to_bits()));
                v.push(format!("  b {:x} {:x} {:x} {:x} {:x} {:x} x {:x} {:x} {:x} {:x} {:x} {:x} xl0 {:x}", b[0].to_bits(), b[1].to_bits(), b[2].to_bits(), b[3].to_bits(), b[4].to_bits(), b[5].to_bits(), x[0].to_bits(), x[1].to_bits(), x[2].to_bits(), x[3].to_bits(), x[4].to_bits(), x[5].to_bits(), x[6].to_bits()));
            }
        });
        if qmax == 10 || rho == 0.0 || !lambda.is_finite() {
            break;
        }
    }
}

thread_local! {
    /// 비교 하네스용 반복 기록(켜면 반복마다 한 줄)
    pub static TRACE: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

/// std::max(a, b) = a < b ? b : a
#[inline]
fn max_std(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// OptimizerG2O::optimizeBA(rootId=1, 두 장면)
pub fn optimize_ba(inp: &BaInput) -> BaOutput {
    let n = inp.points.len();
    let mut out = BaOutput { transform: None, outliers_to: vec![false; n] };
    // 카메라 위치 = 장면 위치 * local (float), 그 뒤 double 로
    let cam_of = |pose: &Tf| -> Cam {
        let cp = pose.mul(&inp.local).to_f64();
        let r = [[cp[0], cp[1], cp[2]], [cp[4], cp[5], cp[6]], [cp[8], cp[9], cp[10]]];
        Cam::new(q_from_mat(&r), [cp[3], cp[7], cp[11]], inp.fx, inp.fy, inp.cx, inp.cy, inp.baseline)
    };
    let c1 = cam_of(&Tf::IDENTITY);
    let c2 = cam_of(&inp.transform);
    // 측정: local⁻¹ * transform * local
    let cl = inp.local.inverse().mul(&inp.transform).mul(&inp.local).to_f64();
    let meas = Se3::from_mat(&[[cl[0], cl[1], cl[2]], [cl[4], cl[5], cl[6]], [cl[8], cl[9], cl[10]]], [cl[3], cl[7], cl[11]]);
    let mut info6 = [[0f64; 6]; 6];
    for i in 0..6 {
        info6[i][i] = 1.0 * (1.0 / inp.cov_diag[i]);
    }
    let mut g = Graph { cams: [c1, c2], pts: Vec::with_capacity(n), obs: Vec::with_capacity(2 * n), meas_inv: meas.inverse(), info6, link_err: [0.0; 6] };
    for (k, p) in inp.points.iter().enumerate() {
        g.pts.push([p.p3[0] as f64, p.p3[1] as f64, p.p3[2] as f64]);
        for cam in 0..2 {
            let (kp, depth) = if cam == 0 { (p.kp_from, p.depth_from) } else { (p.kp_to, p.depth_to) };
            let depth = depth as f64;
            let baseline = g.cams[cam].baseline;
            if depth.is_finite() && depth > 0.0 && baseline > 0.0 {
                let disparity = (baseline * g.cams[cam].kcam[0][0] / depth) as f32;
                g.obs.push(Obs { point: k, cam, stereo: true, meas: [kp[0] as f64, kp[1] as f64, (kp[0] - disparity) as f64], err: [0.0; 3], level: 0, word: k });
            } else {
                g.obs.push(Obs { point: k, cam, stereo: false, meas: [kp[0] as f64, kp[1] as f64, 0.0], err: [0.0; 3], level: 0, word: k });
            }
        }
    }
    let passes = if inp.robust_delta > 0.0 { 2 } else { 1 };
    for i in 0..passes {
        let iters = if i == 0 && inp.robust_delta > 0.0 { 5.min(inp.iterations) } else { inp.iterations };
        levenberg(&mut g, iters);
        g.compute_errors();
        let chi2 = g.active_robust_chi2();
        if chi2.is_nan() {
            return out;
        }
        if i > 0 && (chi2 > 1e12 || !chi2.is_finite()) {
            return out;
        }
        if inp.robust_delta > 0.0 {
            for o in g.obs.iter_mut() {
                if o.level == 0 && Graph::obs_chi2(o) > inp.robust_delta {
                    o.level = 1;
                }
            }
        }
    }
    for o in &g.obs {
        if o.level != 0 && o.cam == 1 {
            out.outliers_to[o.word] = true;
        }
    }
    // 끝의 큰 오차 검사(활성 변 목록 기준: 2 차 뒤에 바깥값이 된 변도 들어감)
    {
        let mut chi = g.link_chi2();
        let active_after_pass1: Vec<bool> = g.obs.iter().map(|o| o.level == 0).collect();
        let _ = active_after_pass1;
        for o in &g.obs {
            if o.level == 0 {
                chi += huber(Graph::obs_chi2(o), HUBER)[0];
            }
        }
        if chi > 1e12 {
            return out;
        }
    }
    // 카메라 2 → 장면 위치: Isometry(q.toRotationMatrix, t) 를 float 로, 그 뒤 * local⁻¹
    let c = &g.cams[1].se3;
    let r = q_to_mat(&c.r);
    let t = Tf([
        r[0][0] as f32,
        r[0][1] as f32,
        r[0][2] as f32,
        c.t[0] as f32,
        r[1][0] as f32,
        r[1][1] as f32,
        r[1][2] as f32,
        c.t[1] as f32,
        r[2][0] as f32,
        r[2][1] as f32,
        r[2][2] as f32,
        c.t[2] as f32,
    ]);
    let t = t.mul(&inp.local.inverse());
    if t.is_null() {
        return out;
    }
    out.transform = Some(t);
    out
}
