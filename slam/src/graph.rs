//! 그래프 최적화: RTAB-Map 0.24.2 OptimizerGTSAM::optimize (corelib/src/optimizer/OptimizerGTSAM.cpp) 의 우리 설정 경로.
//! 설정: Optimizer/Strategy=2(GTSAM), GTSAM/Optimizer=1(가우스-뉴턴), GTSAM/Incremental=false(iSAM2 아님),
//!       Optimizer/Iterations=20, Optimizer/Epsilon=1e-5, Optimizer/Robust=false, Optimizer/PriorsIgnored=true, Reg/Force3DoF=false.
//! 경로: 뿌리 노드 사전(prior) 인자 → 링크마다 BetweenFactor<Pose3>(정보 행렬 → 잡음 모델) → 가우스-뉴턴을 RTAB 의 반복 고리로
//!       (반복마다 오차 감소 < Epsilon 이면 멈춤, 오차 > 1e12 이면 실패) → 결과 Pose3 를 float Transform 으로.
//! GTSAM 4.2.0(빌드 플래그 POSE3_EXPMAP·ROT3_EXPMAP, 사원수·TBB 없음, Eigen 3.4.0 SSE2) 의 다음을 그대로 옮김:
//!   Pose3 between·Logmap·Expmap·AdjointMap(geom.rs), noiseModel::Gaussian::Information(대각 판정 → Diagonal/Isotropic/Unit, 아니면 Eigen LLT),
//!   NoiseModelFactor::linearize(화이트닝)·error(0.5·마할라노비스²), NonlinearFactorGraph::error(인자 순서대로 합), Values::retract.
//! 선형 풀이(solve.rs)만 원본(COLAMD + 다전선 숄레스키)과 덧셈 순서가 다르다.
//! 검증: tests::ref_dumps 가 실제 실행(rtab_bridge)에서 남긴 최적화 호출 덤프를 모두 다시 돌려 비트 비교한다.
mod ccolamd;
mod exact;
mod geom;
mod solve;

pub use geom::Pose3;
use geom::M6;
use std::collections::BTreeMap;

/// RTAB-Map Transform 의 값(3×4 행 우선 float: r11 r12 r13 x / r21 .. y / r31 .. z)
pub type Tf = [f32; 12];

/// rtabmap::Link::Type
pub mod link_type {
    pub const NEIGHBOR: i32 = 0;
    pub const GLOBAL_CLOSURE: i32 = 1;
    pub const LOCAL_SPACE_CLOSURE: i32 = 2;
    pub const LOCAL_TIME_CLOSURE: i32 = 3;
    pub const USER_CLOSURE: i32 = 4;
    pub const VIRTUAL_CLOSURE: i32 = 5;
    pub const NEIGHBOR_MERGED: i32 = 6;
    pub const POSE_PRIOR: i32 = 7;
    pub const LANDMARK: i32 = 8;
    pub const GRAVITY: i32 = 9;
}

/// rtabmap::Link 중 최적화가 읽는 칸. inf = 정보 행렬 6×6(cv::Mat 행 우선, x y z roll pitch yaw 순)
#[derive(Clone, Debug)]
pub struct Link {
    pub from: i32,
    pub to: i32,
    pub kind: i32,
    pub transform: Tf,
    pub inf: [f64; 36],
}

/// 최적화 파라미터(RTAB-Map Parameters.h 의 GTSAM 빌드 기본값)
#[derive(Clone, Debug)]
pub struct OptimizerParams {
    pub iterations: i32,
    pub epsilon: f64,
    pub variance_ignored: bool,
    pub priors_ignored: bool,
    pub gravity_sigma: f32,
    pub force_3dof: bool,
    pub gtsam_optimizer: i32,
    pub robust: bool,
    pub incremental: bool,
    /// 빠른 선형 풀이(최소 차수 + 블록 숄레스키, 원본과 마지막 몇 ulp 다를 수 있음). 기본은 원본과 비트까지 같은 풀이
    pub fast_solver: bool,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        OptimizerParams {
            iterations: 20,
            epsilon: 1e-5,
            variance_ignored: false,
            priors_ignored: true,
            gravity_sigma: 0.3,
            force_3dof: false,
            gtsam_optimizer: 1,
            robust: false,
            incremental: false,
            fast_solver: false,
        }
    }
}

impl OptimizerParams {
    pub fn from_params(p: &crate::params::Params) -> OptimizerParams {
        let d = OptimizerParams::default();
        OptimizerParams {
            iterations: p.get("Optimizer/Iterations", d.iterations),
            epsilon: p.get("Optimizer/Epsilon", d.epsilon),
            variance_ignored: p.get_bool("Optimizer/VarianceIgnored", d.variance_ignored),
            priors_ignored: p.get_bool("Optimizer/PriorsIgnored", d.priors_ignored),
            gravity_sigma: p.get("Optimizer/GravitySigma", d.gravity_sigma),
            force_3dof: p.get_bool("Reg/Force3DoF", d.force_3dof),
            gtsam_optimizer: p.get("GTSAM/Optimizer", d.gtsam_optimizer),
            robust: p.get_bool("Optimizer/Robust", d.robust),
            incremental: p.get_bool("GTSAM/Incremental", d.incremental),
            fast_solver: d.fast_solver,
        }
    }
}

/// 최적화 결과. RTAB 이 빈 결과를 내는 경우(링크 없음, 발산, 풀이 실패)는 poses 가 비어 있다
#[derive(Clone, Debug, Default)]
pub struct Optimized {
    pub poses: BTreeMap<i32, Tf>,
    /// RTAB finalError(멈춘 반복은 빼고 마지막으로 받아들인 오차)
    pub final_error: f64,
    /// RTAB iterationsDone
    pub iterations: i32,
}

/// Gaussian::Information(smart=true) 이 고르는 잡음 모델
#[derive(Clone, Debug)]
enum Noise {
    Unit,
    /// Isotropic: 1/σ
    Iso(f64),
    /// Diagonal: 1/σᵢ
    Diag([f64; 6]),
    /// Gaussian: 위 삼각 R (RᵀR = 정보 행렬)
    Full(M6),
}

impl Noise {
    /// noiseModel::Gaussian::Information(information, smart=true)
    fn information(m: &M6) -> Noise {
        // checkIfDiagonal: 대각 위쪽에 |값| > 1e-9 가 있으면 꽉 찬 행렬
        let full = (0..6).any(|i| (i + 1..6).any(|j| m[i][j].abs() > 1e-9));
        if !full {
            // Diagonal::Precisions → Variances(1/p, smart)
            let var: [f64; 6] = std::array::from_fn(|i| 1.0 / m[i][i]);
            if var.iter().all(|&v| v == var[0]) {
                // Isotropic::Variance(6, v, smart)
                if (var[0] - 1.0).abs() < 1e-9 {
                    return Noise::Unit;
                }
                return Noise::Iso(1.0 / var[0].sqrt());
            }
            return Noise::Diag(std::array::from_fn(|i| 1.0 / var[i].sqrt()));
        }
        // Eigen::LLT<MatrixXd>(아래 삼각, 크기 32 미만이라 블록 없이) → R = Lᵀ
        let mut a = *m;
        for k in 0..6 {
            let mut x = a[k][k];
            if k > 0 {
                // A10.squaredNorm(): 앞에서부터 차례로
                let mut s = a[k][0] * a[k][0];
                for j in 1..k {
                    s += a[k][j] * a[k][j];
                }
                x -= s;
            }
            if !(x > 0.0) {
                // LLT 실패: Eigen 은 여기까지 계산한 행렬을 돌려준다
                break;
            }
            let x = x.sqrt();
            a[k][k] = x;
            if k > 0 && k < 5 {
                // A21 -= A20 · A10ᵀ (GEMV: 열 순서대로 더한 뒤 한 번에 뺌)
                for i in k + 1..6 {
                    let mut s = a[i][0] * a[k][0];
                    for j in 1..k {
                        s += a[i][j] * a[k][j];
                    }
                    a[i][k] -= s;
                }
            }
            for i in k + 1..6 {
                a[i][k] /= x;
            }
        }
        let mut r = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in i..6 {
                r[i][j] = a[j][i];
            }
        }
        Noise::Full(r)
    }

    /// whiten(v)
    fn whiten(&self, v: &[f64; 6]) -> [f64; 6] {
        match self {
            Noise::Unit => *v,
            Noise::Iso(s) => v.map(|x| x * s),
            Noise::Diag(s) => std::array::from_fn(|i| v[i] * s[i]),
            Noise::Full(r) => std::array::from_fn(|i| {
                let mut s = r[i][0] * v[0];
                for k in 1..6 {
                    s += r[i][k] * v[k];
                }
                s
            }),
        }
    }

    /// WhitenInPlace(A), A 는 6×6
    fn whiten_mat(&self, a: &M6) -> M6 {
        match self {
            Noise::Unit => *a,
            Noise::Iso(s) => a.map(|row| row.map(|x| x * s)),
            Noise::Diag(s) => std::array::from_fn(|i| a[i].map(|x| x * s[i])),
            Noise::Full(r) => std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let mut s = r[i][0] * a[0][j];
                    for k in 1..6 {
                        s += r[i][k] * a[k][j];
                    }
                    s
                })
            }),
        }
    }

    /// squaredMahalanobisDistance
    fn sq_dist(&self, v: &[f64; 6]) -> f64 {
        match self {
            Noise::Unit => dot6(v, v),
            Noise::Iso(s) => dot6(v, v) * s * s,
            _ => {
                let w = self.whiten(v);
                dot6(&w, &w)
            }
        }
    }
}

/// 동적 길이 6 벡터 내적(Eigen 3.4 SSE2: 짝수 칸·홀수 칸을 따로 더한 뒤 합침)
#[inline]
fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let p: [f64; 6] = std::array::from_fn(|i| a[i] * b[i]);
    ((p[0] + p[2]) + p[4]) + ((p[1] + p[3]) + p[5])
}

const I6: M6 = {
    let mut m = [[0.0; 6]; 6];
    let mut i = 0;
    while i < 6 {
        m[i][i] = 1.0;
        i += 1;
    }
    m
};

/// 인자: 사전(뿌리, n = 1) 또는 Between(n = 2)
struct Factor {
    keys: [usize; 2],
    n: usize,
    measured: Pose3,
    noise: Noise,
}

impl Factor {
    /// 화이트닝 전 오차와 (원하면) Jacobian
    fn error_vec(&self, x: &[Pose3], jac: Option<&mut [M6; 2]>) -> [f64; 6] {
        if self.n == 1 {
            // PriorFactor::evaluateError = -Local(x, prior), H = I
            let e = x[self.keys[0]].local(&self.measured);
            if let Some(j) = jac {
                j[0] = I6;
            }
            e.map(|v| -v)
        } else {
            // BetweenFactor::evaluateError = Local(measured, p1.between(p2)), H1 = -Ad(hx⁻¹), H2 = I
            let hx = x[self.keys[0]].between(&x[self.keys[1]]);
            if let Some(j) = jac {
                j[0] = hx.inverse().adjoint().map(|row| row.map(|v| -v));
                j[1] = I6;
            }
            self.measured.local(&hx)
        }
    }

    /// NoiseModelFactor::error = 0.5 · 마할라노비스²
    fn error(&self, x: &[Pose3]) -> f64 {
        0.5 * self.noise.sq_dist(&self.error_vec(x, None))
    }
}

/// Ordering::Create(COLAMD, graph): VariableIndex(변수 = 키 순서, 변수마다 인자 번호 오름차순) → ccolamd.
/// 돌려주는 것: 소거 순서(변수 번호)
fn colamd_order(fs: &[Factor], nvars: usize) -> Vec<usize> {
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); nvars];
    for (fi, f) in fs.iter().enumerate() {
        for &k in &f.keys[..f.n] {
            cols[k].push(fi as i32);
        }
    }
    ccolamd::colamd(fs.len(), &cols)
}

/// NonlinearFactorGraph::error: 인자 순서대로 합
fn graph_error(fs: &[Factor], x: &[Pose3]) -> f64 {
    let mut s = 0.0;
    for f in fs {
        s += f.error(x);
    }
    s
}

/// GaussNewtonOptimizer::iterate: 선형화(b = -화이트닝 오차, A = 화이트닝 Jacobian) → 풀이 → retract
fn gn_step(fs: &[Factor], x: &[Pose3], ord: Option<&[usize]>) -> Option<Vec<Pose3>> {
    let delta = match ord {
        Some(ord) => {
            let lin: Vec<exact::LinFactor> = fs.iter().map(|f| linearize(f, x)).collect();
            exact::solve(x.len(), &lin, ord)?
        }
        None => {
            let mut sys = solve::Normal::new(x.len());
            for f in fs {
                let l = linearize(f, x);
                sys.add(&l.keys, &l.a, &l.b);
            }
            solve::solve(sys)?
        }
    };
    Some(x.iter().zip(&delta).map(|(p, d)| p.retract(d)).collect())
}

/// NoiseModelFactor::linearize: A = 화이트닝 Jacobian, b = -화이트닝 오차
fn linearize(f: &Factor, x: &[Pose3]) -> exact::LinFactor {
    let mut j = [[[0.0; 6]; 6]; 2];
    let e = f.error_vec(x, Some(&mut j));
    let b = f.noise.whiten(&e.map(|v| -v));
    let a: Vec<M6> = (0..f.n).map(|k| f.noise.whiten_mat(&j[k])).collect();
    exact::LinFactor { keys: f.keys[..f.n].to_vec(), a, b }
}

/// RTAB-Map OptimizerGTSAM::optimize(rootId, poses, edgeConstraints) 의 3D · 배치 · 가우스-뉴턴 경로.
/// links 는 std::multimap<int, Link> 의 순회 순서(키 = from, 같은 키는 넣은 순서)로 준다.
/// 우리 설정 밖(2D, iSAM2, Robust, 사전 사용, 중력 링크, 랜드마크, 다른 GTSAM 최적화기)이면 Err.
pub fn optimize(root: i32, poses: &BTreeMap<i32, Tf>, links: &[Link], p: &OptimizerParams) -> Result<Optimized, String> {
    if p.force_3dof || p.incremental || p.robust || !p.priors_ignored || p.gtsam_optimizer != 1 {
        return Err("우리 설정(3D·배치·가우스-뉴턴·PriorsIgnored) 밖의 최적화 설정".into());
    }
    if links.is_empty() || poses.len() < 2 || p.iterations <= 0 {
        // 원본 OptimizerGTSAM.cpp 1140 행: 위치가 하나뿐이거나 반복 0 이면 받은 위치를 그대로, 그 밖(링크 없음)은 빈 결과
        if poses.len() == 1 || p.iterations <= 0 {
            return Ok(Optimized { poses: poses.clone(), ..Optimized::default() });
        }
        return Ok(Optimized::default());
    }
    if poses.keys().any(|&id| id <= 0) {
        return Err("랜드마크(음수 id)는 옮기지 않음".into());
    }
    if p.gravity_sigma > 0.0 && links.iter().any(|l| l.from == l.to && l.kind == link_type::GRAVITY) {
        return Err("중력 링크(AttitudeFactor)는 옮기지 않음".into());
    }
    // Values: id 순서
    let ids: Vec<i32> = poses.keys().copied().collect();
    let index: BTreeMap<i32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let x0: Vec<Pose3> = poses.values().map(Pose3::from_tf).collect();
    let fs = build_factors(root, &index, &x0, links, p)?;
    // GaussNewtonOptimizer 가 만들 때 한 번 정하는 소거 순서(Ordering::Create(COLAMD, graph))
    let ord = if p.fast_solver { None } else { Some(colamd_order(&fs, x0.len())) };
    // RTAB 반복 고리
    let mut x = x0;
    let mut last_error = graph_error(&fs, &x);
    let mut it = 0;
    for i in 0..p.iterations {
        let Some(nx) = gn_step(&fs, &x, ord.as_deref()) else {
            return Ok(Optimized::default());
        };
        x = nx;
        let error = graph_error(&fs, &x);
        it += 1;
        let delta = last_error - error;
        if p.epsilon > 0.0 && error.abs() > 1e12 {
            return Ok(Optimized::default());
        }
        if i > 0 && delta < p.epsilon {
            // 음수(오차가 늘어남)면 계속
            if delta >= 0.0 {
                break;
            }
        } else if i == 0 && error >= 0.0 && error < p.epsilon {
            break;
        }
        last_error = error;
    }
    Ok(Optimized { poses: ids.iter().zip(&x).map(|(&id, q)| (id, q.to_tf())).collect(), final_error: last_error, iterations: it })
}

/// 인자 그래프: 뿌리 사전 먼저, 그다음 링크 순서대로(RTAB 이 graph.add 하는 순서)
fn build_factors(root: i32, index: &BTreeMap<i32, usize>, x0: &[Pose3], links: &[Link], p: &OptimizerParams) -> Result<Vec<Factor>, String> {
    let mut fs: Vec<Factor> = Vec::with_capacity(links.len() + 1);
    // 뿌리 사전: 위 축(up = Rᵀz) 둘레 회전은 분산 1e-9, 나머지 회전·위치는 1e-2
    if root != 0 {
        let &ri = index.get(&root).ok_or("뿌리 노드가 poses 에 없음")?;
        let rp = x0[ri];
        let up = [rp.r[2][0], rp.r[2][1], rp.r[2][2]];
        let mut inf = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                let u = up[i] * up[j];
                let id = if i == j { 1.0 } else { 0.0 };
                inf[i][j] = u / 1e-9 + (id - u) / 1e-2;
            }
            inf[i + 3][i + 3] = 1.0 / 1e-2;
        }
        fs.push(Factor { keys: [ri, ri], n: 1, measured: rp, noise: Noise::information(&inf) });
    }
    for l in links {
        if l.from == l.to {
            // PriorsIgnored: 사전 링크는 건너뜀
            continue;
        }
        let a = *index.get(&l.from).ok_or("링크의 from 이 poses 에 없음")?;
        let b = *index.get(&l.to).ok_or("링크의 to 가 poses 에 없음")?;
        // memcpy(cv 행 우선 → Eigen 열 우선) = 전치. 그다음 위치·회전 블록 자리를 바꿈(gtsam 은 회전 먼저)
        let info: M6 = if p.variance_ignored { I6 } else { std::array::from_fn(|i| std::array::from_fn(|j| l.inf[j * 6 + i])) };
        let sw = |i: usize| (i + 3) % 6;
        let mg: M6 = std::array::from_fn(|i| std::array::from_fn(|j| info[sw(i)][sw(j)]));
        fs.push(Factor { keys: [a, b], n: 2, measured: Pose3::from_tf(&l.transform), noise: Noise::information(&mg) });
    }
    Ok(fs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        let xi = [0.1, -0.2, 0.3, 1.0, 2.0, -0.5];
        let p = Pose3::expmap(&xi);
        let l = Pose3::logmap(&p);
        for i in 0..6 {
            assert!((l[i] - xi[i]).abs() < 1e-12);
        }
        let q = p.compose(&p.inverse());
        assert!((q.r[0][0] - 1.0).abs() < 1e-12 && q.t[2].abs() < 1e-12);
    }

    #[test]
    fn ring_with_loop() {
        // 8 노드 고리 + 루프 하나: 수렴하고 뿌리는 그대로
        let mut poses = BTreeMap::new();
        let mut links = Vec::new();
        let step = Pose3::expmap(&[0.0, 0.0, std::f64::consts::PI / 4.0, 1.0, 0.0, 0.0]);
        let mut cur = Pose3::from_tf(&[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]);
        let mut inf = [0.0; 36];
        for i in 0..6 {
            inf[i * 6 + i] = if i < 3 { 1000.0 } else { 100.0 };
        }
        for id in 1..=8 {
            let noisy = cur.retract(&[0.01, 0.0, 0.02 * id as f64, 0.05, 0.0, 0.0]);
            poses.insert(id, if id == 1 { cur.to_tf() } else { noisy.to_tf() });
            if id < 8 {
                links.push(Link { from: id, to: id + 1, kind: link_type::NEIGHBOR, transform: step.to_tf(), inf });
            }
            cur = cur.compose(&step);
        }
        links.push(Link { from: 8, to: 1, kind: link_type::GLOBAL_CLOSURE, transform: step.to_tf(), inf });
        let r = optimize(1, &poses, &links, &OptimizerParams::default()).unwrap();
        assert_eq!(r.poses.len(), 8);
        // 뿌리 사전은 무한히 단단하지 않아 아주 조금(1e-15 수준) 움직일 수 있음
        assert!(r.poses[&1].iter().zip(&poses[&1]).all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(r.final_error < 1e-3, "{}", r.final_error);
    }

    // ---- 원본과 비트 비교 ----
    // 덤프 opt_*.bin: LD_PRELOAD 로 rtab_bridge 안의 OptimizerGTSAM::optimize 입력·출력을 그대로 남긴 것(저장소 밖).
    struct Rd<'a>(&'a [u8], usize);
    impl Rd<'_> {
        fn take<const N: usize>(&mut self) -> [u8; N] {
            let v = self.0[self.1..self.1 + N].try_into().unwrap();
            self.1 += N;
            v
        }
        fn i32(&mut self) -> i32 {
            i32::from_le_bytes(self.take())
        }
        fn f32(&mut self) -> f32 {
            f32::from_le_bytes(self.take())
        }
        fn f64(&mut self) -> f64 {
            f64::from_le_bytes(self.take())
        }
        fn poses(&mut self) -> BTreeMap<i32, Tf> {
            let n = self.i32();
            (0..n).map(|_| (self.i32(), std::array::from_fn(|_| self.f32()))).collect()
        }
    }

    pub(super) struct Dump {
        pub root: i32,
        pub iters: i32,
        pub eps: f64,
        pub poses: BTreeMap<i32, Tf>,
        pub links: Vec<Link>,
        pub out: BTreeMap<i32, Tf>,
        pub fe: f64,
        pub it: i32,
        pub us: f64,
    }

    pub(super) fn read_dump(path: &std::path::Path) -> Dump {
        let b = std::fs::read(path).unwrap();
        let mut r = Rd(&b, 0);
        let _magic = r.i32();
        let root = r.i32();
        let iters = r.i32();
        let _ = r.i32();
        let eps = r.f64();
        let poses = r.poses();
        let nl = r.i32();
        let links = (0..nl)
            .map(|_| {
                let (_key, from, to, kind) = (r.i32(), r.i32(), r.i32(), r.i32());
                let transform = std::array::from_fn(|_| r.f32());
                let inf = std::array::from_fn(|_| r.f64());
                Link { from, to, kind, transform, inf }
            })
            .collect();
        let out = r.poses();
        let fe = r.f64();
        let it = r.i32();
        let us = r.f64();
        Dump { root, iters, eps, poses, links, out, fe, it, us }
    }

    /// float 두 값의 ulp 거리
    fn ulp32(a: f32, b: f32) -> u64 {
        let k = |x: f32| {
            let i = x.to_bits() as i32;
            if i < 0 { i32::MIN as i64 - i as i64 } else { i as i64 }
        };
        (k(a) - k(b)).unsigned_abs()
    }

    pub(super) fn dump_files() -> Vec<std::path::PathBuf> {
        let root = std::env::var("SLAM_GTSAM_REF").unwrap_or_else(|_| format!("{}/slamref_gtsam", std::env::var("HOME").unwrap_or_default()));
        let mut files = Vec::new();
        for d in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            for f in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
                let p = f.path();
                if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("opt_")) && p.extension().is_some_and(|e| e == "bin") {
                    files.push(p);
                }
            }
        }
        files.sort();
        files
    }

    /// f64 두 값의 ulp 거리
    fn ulp64(a: f64, b: f64) -> u64 {
        let k = |x: f64| {
            let i = x.to_bits() as i64;
            if i < 0 { i64::MIN as i128 - i as i128 } else { i as i128 }
        };
        (k(a) - k(b)).unsigned_abs() as u64
    }

    /// 단계별 비교: replay(저장소 밖 C++ 하네스)가 GTSAM 으로 같은 그래프를 돌리며 남긴 trace(opt_*.trace) 와
    /// 반복마다 (1) 원본 값에서의 오차 (2) 원본 값에서의 선형화 A·b (3) 원본 δ 로 retract (4) 원본 선형계를 우리 풀이로 → δ ulp
    #[test]
    fn ref_traces() {
        let files = dump_files();
        let (mut n_err, mut s_err, mut n_ab, mut s_ab, mut n_ret, mut s_ret, mut n_d, mut max_d, mut s_d) = (0, 0, 0usize, 0usize, 0usize, 0usize, 0usize, 0f64, 0usize);
        let mut max_ab = 0u64;
        let (mut n_ord, mut s_ord) = (0, 0);
        for f in &files {
            let tp = f.with_extension("trace");
            let Ok(tb) = std::fs::read(&tp) else { continue };
            let d = read_dump(f);
            // optimize() 와 같은 인자 목록
            let ids: Vec<i32> = d.poses.keys().copied().collect();
            let index: BTreeMap<i32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let x0: Vec<Pose3> = d.poses.values().map(Pose3::from_tf).collect();
            let fs = build_factors(d.root, &index, &x0, &d.links, &OptimizerParams::default()).unwrap();
            let mut r = Rd(&tb, 0);
            let i64r = |r: &mut Rd| i64::from_le_bytes(r.take());
            let nord = i64r(&mut r);
            let gord: Vec<usize> = (0..nord).map(|_| index[&(i64r(&mut r) as i32)]).collect();
            n_ord += 1;
            if colamd_order(&fs, x0.len()) == gord {
                s_ord += 1;
            } else if s_ord + 3 > n_ord {
                common::info!("  순서 다름: {}", tp.display());
            }
            let vals = |r: &mut Rd| -> Vec<Pose3> {
                let n = i64::from_le_bytes(r.take());
                (0..n)
                    .map(|_| {
                        let _k = i64::from_le_bytes(r.take());
                        let c: [f64; 9] = std::array::from_fn(|_| r.f64());
                        let t: [f64; 3] = std::array::from_fn(|_| r.f64());
                        Pose3 { r: std::array::from_fn(|i| std::array::from_fn(|j| c[j * 3 + i])), t }
                    })
                    .collect()
            };
            let mut x = vals(&mut r);
            let e0 = r.f64();
            assert!(x.iter().zip(&x0).all(|(a, b)| a == b), "초기 값이 다름");
            n_err += 1;
            s_err += (graph_error(&fs, &x).to_bits() == e0.to_bits()) as usize;
            loop {
                let nf = i64r(&mut r);
                if nf < 0 {
                    break;
                }
                // 선형 인자
                let mut lins: Vec<exact::LinFactor> = Vec::with_capacity(nf as usize);
                for (fi, fac) in fs.iter().enumerate().take(nf as usize) {
                    let nk = i64r(&mut r) as usize;
                    let keys: Vec<usize> = (0..nk).map(|_| index[&(i64r(&mut r) as i32)]).collect();
                    let rows = i64r(&mut r) as usize;
                    assert_eq!(rows, 6);
                    let a: Vec<M6> = (0..nk)
                        .map(|_| {
                            let c: [f64; 36] = std::array::from_fn(|_| r.f64());
                            std::array::from_fn(|i| std::array::from_fn(|j| c[j * 6 + i]))
                        })
                        .collect();
                    let b: [f64; 6] = std::array::from_fn(|_| r.f64());
                    // 우리 선형화
                    let mut j = [[[0.0; 6]; 6]; 2];
                    let e = fac.error_vec(&x, Some(&mut j));
                    let ob = fac.noise.whiten(&e.map(|v| -v));
                    assert_eq!(keys[..], fac.keys[..fac.n], "인자 {fi} 키");
                    for k in 0..nk {
                        let oa = fac.noise.whiten_mat(&j[k]);
                        for p in 0..6 {
                            for q in 0..6 {
                                n_ab += 1;
                                if oa[p][q].to_bits() == a[k][p][q].to_bits() || (oa[p][q] == 0.0 && a[k][p][q] == 0.0) {
                                    s_ab += 1;
                                } else {
                                    max_ab = max_ab.max(ulp64(oa[p][q], a[k][p][q]));
                                }
                            }
                        }
                    }
                    for p in 0..6 {
                        n_ab += 1;
                        if ob[p].to_bits() == b[p].to_bits() {
                            s_ab += 1;
                        } else {
                            max_ab = max_ab.max(ulp64(ob[p], b[p]));
                        }
                    }
                    lins.push(exact::LinFactor { keys, a, b });
                }
                let nd = i64r(&mut r);
                let mut delta = vec![[0.0; 6]; x.len()];
                for _ in 0..nd {
                    let k = index[&(i64r(&mut r) as i32)];
                    delta[k] = std::array::from_fn(|_| r.f64());
                }
                // 원본 선형계를 정확 이식 풀이로(같은 순서 gord)
                let ours = exact::solve(x.len(), &lins, &gord).expect("정확 풀이 실패");
                if n_d == s_d && ours.iter().flatten().zip(delta.iter().flatten()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    common::info!("  첫 δ 불일치: {}", tp.display());
                }
                for (a, b) in ours.iter().flatten().zip(delta.iter().flatten()) {
                    n_d += 1;
                    if a.to_bits() == b.to_bits() {
                        s_d += 1;
                    }
                    max_d = max_d.max((a - b).abs());
                }
                let nx = vals(&mut r);
                let e = r.f64();
                for (k, p) in x.iter().enumerate() {
                    n_ret += 1;
                    s_ret += (p.retract(&delta[k]) == nx[k]) as usize;
                }
                x = nx;
                n_err += 1;
                s_err += (graph_error(&fs, &x).to_bits() == e.to_bits()) as usize;
            }
        }
        common::info!(
            "단계별: 순서 같음 {s_ord}/{n_ord}, 오차 같음 {s_err}/{n_err}, 선형화 A·b 같음 {s_ab}/{n_ab}(최대 {max_ab} ulp), retract 같음 {s_ret}/{n_ret}, δ 같음 {s_d}/{n_d}(최대 차이 {max_d:e})"
        );
    }

    /// SLAM_GTSAM_REF(기본 ~/slamref_gtsam) 아래 폴더들의 opt_*.bin 을 모두 다시 돌려 비교. 덤프가 없으면 건너뜀
    #[test]
    fn ref_dumps() {
        let files = dump_files();
        if files.is_empty() {
            common::info!("덤프 없음, 건너뜀");
            return;
        }
        let (mut calls, mut same_calls, mut vals, mut same_vals, mut max_ulp, mut same_it, mut same_fe) = (0, 0, 0usize, 0usize, 0u64, 0, 0);
        for f in &files {
            let d = read_dump(f);
            let p = OptimizerParams { iterations: d.iters, epsilon: d.eps, ..Default::default() };
            let reps = 5;
            let t0 = std::time::Instant::now();
            let mut r = Optimized::default();
            for _ in 0..reps {
                r = optimize(d.root, &d.poses, &d.links, &p).unwrap();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let pf = OptimizerParams { fast_solver: true, ..p.clone() };
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = optimize(d.root, &d.poses, &d.links, &pf).unwrap();
            }
            let us_fast = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
            calls += 1;
            let mut call_same = r.poses.len() == d.out.len();
            let mut call_max = 0;
            for (id, t) in &d.out {
                let Some(q) = r.poses.get(id) else {
                    call_same = false;
                    continue;
                };
                for k in 0..12 {
                    vals += 1;
                    if q[k].to_bits() == t[k].to_bits() {
                        same_vals += 1;
                    } else {
                        call_same = false;
                        call_max = call_max.max(ulp32(q[k], t[k]));
                    }
                }
            }
            max_ulp = max_ulp.max(call_max);
            same_calls += call_same as i32;
            same_it += (r.iterations == d.it) as i32;
            same_fe += (r.final_error.to_bits() == d.fe.to_bits()) as i32;
            common::info!(
                "{}: 노드 {} 링크 {} | 출력 {} | 반복 {}/{} | 최종 오차 {:e}/{:e} | 최대 {} ulp | 원본 {:.0} us, 정확 Rust {:.0} us, 빠른 Rust {:.0} us",
                f.display(),
                d.poses.len(),
                d.links.len(),
                if call_same { "같음" } else { "다름" },
                r.iterations,
                d.it,
                r.final_error,
                d.fe,
                call_max,
                d.us,
                us,
                us_fast
            );
        }
        common::info!("합계: 호출 {calls}, 출력 전부 같음 {same_calls}, float 값 같음 {same_vals}/{vals}, 최대 {max_ulp} ulp, 반복 수 같음 {same_it}, 최종 오차 비트 같음 {same_fe}");
    }
}
