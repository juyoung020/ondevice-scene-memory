//! Optimizer::OptimizeSim3 · OptimizeEssentialGraph(루프 닫기 · 지도 합치기)가 부르는 g2o 경로를 기준 빌드와 비트 같게.
//! 원본: UZ-SLAMLab/ORB_SLAM3(GPL-3.0) src/Optimizer.cc · include/OptimizableTypes.h + Thirdparty/g2o(BSD)
//! types_seven_dof_expmap · SparseOptimizer · OptimizationAlgorithmLevenberg · BlockSolverX/BlockSolver_7_3 ·
//! LinearSolverDense(Eigen LDLT<MatrixXd>) · LinearSolverEigen(SimplicialLDLT — `orb3ba::ldlt` 를 그대로 씀).
//!
//! 기준 빌드: -O3 -march=native(AVX-512) -ffp-contract=off. Eigen 이 묶음(pmadd)으로 계산하는 자리만 FMA:
//! - 고정 7×7 결과에 작은 곱을 더하는 대입(Packet2d 조각 벡터화): 모든 열에서 0..6 행이 묶음 · 6 행 낱개.
//!   묶음 행 = k 순서 FMA 사슬, 낱개 행 = 곱을 반씩 나눠 더함(`tree`). 왼쪽이 행 우선(전치)이면 전부 낱개.
//! - 연속 메모리 두 벡터의 고정 7 길이 내적(Packet2d 펼침): ((m0+(m2+m4)) + (m1+(m3+m5))) + m6 (`vred7`).
//! - Eigen GEMV(LDLT 의 A21 −= A20·temp): 앞 4 행 · 다음 2 행은 묶음(FMA 사슬), 나머지 낱개.
//! 하네스: 저장소 밖 WSL `~/opt/orb3_cv/s3opt`(원본 덤프) · `~/opt/orb3_cv/rs --bin s3opt`(비교).
use super::g2osim3::{G2oSim3, V3d};
use crate::orb3ba::ldlt::Ldlt;
use std::collections::BTreeMap;

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// 고정 7×7 대입(Packet2d 조각 벡터화, 열마다 정렬 시작 0)에서 i 행이 묶음으로 계산되는지: 0..6 행 묶음 · 6 행 낱개
#[inline(always)]
fn packet_row(i: usize) -> bool {
    i < 6
}

/// 낱개 곱 합(Eigen redux_novec_unroller: 반으로 나눠 왼쪽 + 오른쪽)
fn tree(m: &[f64]) -> f64 {
    if m.len() == 1 {
        m[0]
    } else {
        let h = m.len() / 2;
        tree(&m[..h]) + tree(&m[h..])
    }
}

/// 묶음 곱: c = l0·r0, c = fma(lk, rk, c)
#[inline]
fn chain(l: &[f64], r: &[f64]) -> f64 {
    let mut c = l[0] * r[0];
    for k in 1..l.len() {
        c = fma(l[k], r[k], c);
    }
    c
}

/// 연속 7 길이 내적(Packet2d 3 개 펼침 + 남은 1 개)
#[inline]
fn vred7(a: &[f64], b: &[f64]) -> f64 {
    let m = |i: usize| a[i] * b[i];
    ((m(0) + (m(2) + m(4))) + (m(1) + (m(3) + m(5)))) + m(6)
}

/// h(7×7 열 우선) += L(7×d) · R(d×7). l(i) = L 의 i 행, r(j) = R 의 j 열
fn add_prod7(h: &mut [f64; 49], l: &[[f64; 7]; 7], r: &[[f64; 7]; 7], d: usize) {
    for j in 0..7 {
        for i in 0..7 {
            let v = if packet_row(i) { chain(&l[i][..d], &r[j][..d]) } else { tree(&prods(&l[i][..d], &r[j][..d])) };
            h[j * 7 + i] += v;
        }
    }
}

fn prods(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// g2o RobustKernelHuber::robustify → (ρ0, ρ1)
#[inline]
pub fn huber(e: f64, delta: f64) -> (f64, f64) {
    let dsqr = delta * delta;
    if e <= dsqr {
        (e, 1.0)
    } else {
        let sqrte = e.sqrt();
        (2.0 * sqrte * delta - dsqr, delta / sqrte)
    }
}

/// VertexSim3Expmap::oplusImpl: (고정 축척이면 갱신 6 = 0, 입력도 바뀜) exp(δ)·S
#[inline]
fn oplus(s: &G2oSim3, upd: &mut [f64], fix_scale: bool) -> G2oSim3 {
    if fix_scale {
        upd[6] = 0.0;
    }
    let u: [f64; 7] = upd[..7].try_into().unwrap();
    G2oSim3::exp(&u).mul(s)
}

// ---------------------------------------------------------------------------------------------
// Eigen 3.4 LDLT<MatrixXd>(아래 삼각, 대각 피벗) — g2o LinearSolverDense

/// LinearSolverDense::solve: compute(H) → isPositive 면 solve(b)
pub fn dense_ldlt_solve(h: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut m = h.to_vec();
    let at = |r: usize, c: usize| c * n + r;
    let mut tr = vec![0usize; n];
    let mut temp = vec![0.0f64; n];
    // 0 ZeroSign, 1 Pos, 2 Neg, 3 Indef
    let mut sign = 0u8;
    if n <= 1 {
        tr.iter_mut().enumerate().for_each(|(i, t)| *t = i);
        sign = if n == 0 { 0 } else if m[0] > 0.0 { 1 } else if m[0] < 0.0 { 2 } else { 0 };
    } else {
        for k in 0..n {
            // 남은 대각 중 절댓값 최대(첫 것)
            let mut big = k;
            let mut bv = m[at(k, k)].abs();
            for i in k + 1..n {
                let v = m[at(i, i)].abs();
                if v > bv {
                    bv = v;
                    big = i;
                }
            }
            tr[k] = big;
            if k != big {
                for c in 0..k {
                    m.swap(at(k, c), at(big, c));
                }
                for r in big + 1..n {
                    m.swap(at(r, k), at(r, big));
                }
                m.swap(at(k, k), at(big, big));
                for i in k + 1..big {
                    let tmp = m[at(i, k)];
                    m[at(i, k)] = m[at(big, i)];
                    m[at(big, i)] = tmp;
                }
            }
            let rs = n - k - 1;
            if k > 0 {
                for j in 0..k {
                    temp[j] = m[at(j, j)] * m[at(k, j)];
                }
                let mut s = m[at(k, 0)] * temp[0];
                for j in 1..k {
                    s += m[at(k, j)] * temp[j];
                }
                m[at(k, k)] -= s;
                if rs > 0 {
                    // GEMV: 앞 4 행 · 다음 2 행 묶음, 나머지 낱개 (AVX-512 의 반 · 4 분의 1 묶음)
                    let p4 = if rs >= 4 { 4 } else { 0 };
                    let p2 = if rs - p4 >= 2 { p4 + 2 } else { p4 };
                    for r in 0..rs {
                        let row = k + 1 + r;
                        let c = if r < p2 {
                            let mut c = m[at(row, 0)] * temp[0];
                            for j in 1..k {
                                c = fma(m[at(row, j)], temp[j], c);
                            }
                            c
                        } else {
                            let mut c = 0.0 + m[at(row, 0)] * temp[0];
                            for j in 1..k {
                                c += m[at(row, j)] * temp[j];
                            }
                            c
                        };
                        m[at(row, k)] -= c;
                    }
                }
            }
            let akk = m[at(k, k)];
            let valid = akk.abs() > 0.0;
            if k == 0 && !valid {
                sign = 0;
                for j in 0..n {
                    tr[j] = j;
                }
                break;
            }
            if rs > 0 && valid {
                for r in k + 1..n {
                    m[at(r, k)] /= akk;
                }
            }
            match sign {
                1 => {
                    if akk < 0.0 {
                        sign = 3
                    }
                }
                2 => {
                    if akk > 0.0 {
                        sign = 3
                    }
                }
                0 => {
                    if akk > 0.0 {
                        sign = 1
                    } else if akk < 0.0 {
                        sign = 2
                    }
                }
                _ => {}
            }
        }
    }
    if !(sign == 1 || sign == 0) {
        return None;
    }
    let mut x = b.to_vec();
    for k in 0..n {
        if tr[k] != k {
            x.swap(k, tr[k]);
        }
    }
    // L 전진(열 우선 단위 아래 삼각): x[s..] −= x[i]·L[s.., i]
    for i in 0..n {
        if x[i] != 0.0 {
            let xi = x[i];
            for r in i + 1..n {
                x[r] -= xi * m[at(r, i)];
            }
        }
    }
    for i in 0..n {
        let d = m[at(i, i)];
        if d.abs() > f64::MIN_POSITIVE {
            x[i] /= d;
        } else {
            x[i] = 0.0;
        }
    }
    // Lᵀ 후진(행 우선 단위 위 삼각): x[i] −= Σ L[j,i]·x[j] (j > i, 낱개 차례 합)
    for i in (0..n).rev() {
        let k = n - 1 - i;
        if k > 0 {
            let mut s = m[at(i + 1, i)] * x[i + 1];
            for j in i + 2..n {
                s += m[at(j, i)] * x[j];
            }
            x[i] -= s;
        }
    }
    for k in (0..n).rev() {
        if tr[k] != k {
            x.swap(k, tr[k]);
        }
    }
    Some(x)
}

// ---------------------------------------------------------------------------------------------
// OptimizeSim3

/// EdgeSim3ProjectXYZ(inverse = false: x1 = π1(S12·X2)) · EdgeInverseSim3ProjectXYZ(inverse = true: x2 = π2(S12⁻¹·X1))
#[derive(Clone, Debug)]
pub struct ProjEdge {
    pub inverse: bool,
    /// 고정 점(카메라 좌표, double)
    pub pt: V3d,
    pub obs: [f64; 2],
    /// 2×2 열 우선
    pub info: [f64; 4],
    /// Huber δ(없으면 None)
    pub delta: Option<f64>,
    pub cam: [f32; 4],
}

/// Pinhole::project(Vector3d): fx·x/z + cx
#[inline]
fn project(cam: &[f32; 4], v: &V3d) -> [f64; 2] {
    [cam[0] as f64 * v[0] / v[2] + cam[2] as f64, cam[1] as f64 * v[1] / v[2] + cam[3] as f64]
}

impl ProjEdge {
    fn error(&self, s: &G2oSim3) -> [f64; 2] {
        let p = if self.inverse { s.inverse().map(&self.pt) } else { s.map(&self.pt) };
        let q = project(&self.cam, &p);
        [self.obs[0] - q[0], self.obs[1] - q[1]]
    }
}

/// χ² = e·(Ω·e)
#[inline]
fn chi2_2(e: &[f64; 2], o: &[f64; 4]) -> f64 {
    let w0 = o[0] * e[0] + o[2] * e[1];
    let w1 = o[1] * e[0] + o[3] * e[1];
    e[0] * w0 + e[1] * w1
}

/// 정점 하나(Sim3)짜리 g2o Levenberg. 돌려줌: 반복 수, 간선마다 마지막으로 계산한 오차
struct OneVertex<'a> {
    edges: &'a [ProjEdge],
    active: &'a [bool],
    fix_scale: bool,
    est: G2oSim3,
    err: Vec<[f64; 2]>,
}

impl<'a> OneVertex<'a> {
    fn compute_errors(&mut self) {
        for (k, e) in self.edges.iter().enumerate() {
            if self.active[k] {
                self.err[k] = e.error(&self.est);
            }
        }
    }
    fn robust_chi2(&self) -> f64 {
        let mut s = 0.0;
        for (k, e) in self.edges.iter().enumerate() {
            if self.active[k] {
                let c = chi2_2(&self.err[k], &e.info);
                s += match e.delta {
                    Some(d) => huber(c, d).0,
                    None => c,
                };
            }
        }
        s
    }
    /// BlockSolverX::buildSystem → (H 7×7 열 우선, b)
    fn build(&self) -> ([f64; 49], [f64; 7]) {
        let mut h = [0.0; 49];
        let mut b = [0.0; 7];
        let delta = 1e-9;
        let scalar = 1.0 / (2.0 * delta);
        for (k, e) in self.edges.iter().enumerate() {
            if !self.active[k] {
                continue;
            }
            // 수치 야코비안 2×7(열 d)
            let mut jac = [[0.0f64; 2]; 7];
            let mut add = [0.0f64; 7];
            for d in 0..7 {
                add[d] = delta;
                let ep = e.error(&oplus(&self.est, &mut add, self.fix_scale));
                add[d] = -delta;
                let em = e.error(&oplus(&self.est, &mut add, self.fix_scale));
                add[d] = 0.0;
                jac[d] = [scalar * (ep[0] - em[0]), scalar * (ep[1] - em[1])];
            }
            let er = &self.err[k];
            let o = &e.info;
            // omega_r = (−Ω)·e (2 행 묶음)
            let mut om_r = [fma(-o[2], er[1], -o[0] * er[0]), fma(-o[3], er[1], -o[1] * er[0])];
            let w: [f64; 4] = match e.delta {
                Some(dl) => {
                    let (_, r1) = huber(chi2_2(er, o), dl);
                    om_r = [om_r[0] * r1, om_r[1] * r1];
                    [r1 * o[0], r1 * o[1], r1 * o[2], r1 * o[3]]
                }
                None => *o,
            };
            // b += Bᵀ ω (낱개)
            for i in 0..7 {
                b[i] += jac[i][0] * om_r[0] + jac[i][1] * om_r[1];
            }
            // T = Bᵀ W (7×2, 낱개), H += T · B
            let mut l = [[0.0; 7]; 7];
            let mut r = [[0.0; 7]; 7];
            for i in 0..7 {
                l[i][0] = jac[i][0] * w[0] + jac[i][1] * w[1];
                l[i][1] = jac[i][0] * w[2] + jac[i][1] * w[3];
                r[i][0] = jac[i][0];
                r[i][1] = jac[i][1];
            }
            add_prod7(&mut h, &l, &r, 2);
        }
        (h, b)
    }

    fn optimize(&mut self, iters: usize) -> usize {
        let tau = 1e-5;
        let (mut lambda, mut ni, mut nbad) = (0.0f64, 2.0f64, 0);
        let mut it = 0;
        let mut xprev = [0.0f64; 7];
        for i in 0..iters {
            self.compute_errors();
            let mut cur = self.robust_chi2();
            let ini = cur;
            let (h, b) = self.build();
            if i == 0 {
                let mut md = 0.0f64;
                for j in 0..7 {
                    md = h[j * 8].abs().max(md);
                }
                lambda = tau * md;
                ni = 2.0;
                nbad = 0;
            }
            let mut rho;
            let mut q = 0;
            loop {
                let saved = self.est;
                let mut hl = h;
                for j in 0..7 {
                    hl[j * 8] += lambda;
                }
                let x = dense_ldlt_solve(&hl, 7, &b);
                let ok = x.is_some();
                let mut xv: [f64; 7] = match &x {
                    Some(v) => v[..7].try_into().unwrap(),
                    None => xprev,
                };
                self.est = oplus(&self.est, &mut xv, self.fix_scale);
                xprev = xv;
                self.compute_errors();
                let mut temp = self.robust_chi2();
                if !ok {
                    temp = f64::MAX;
                }
                rho = cur - temp;
                let mut scale = 0.0;
                for j in 0..7 {
                    scale += xv[j] * (lambda * xv[j] + b[j]);
                }
                scale += 1e-3;
                rho /= scale;
                if rho > 0.0 && temp.is_finite() {
                    let alpha = 1.0 - (2.0 * rho - 1.0).powf(3.0);
                    let alpha = alpha.min(2.0 / 3.0);
                    lambda *= (1.0f64 / 3.0).max(alpha);
                    ni = 2.0;
                    cur = temp;
                } else {
                    lambda *= ni;
                    ni *= 2.0;
                    self.est = saved;
                }
                q += 1;
                if !(rho < 0.0 && q < 10) {
                    break;
                }
            }
            it += 1;
            if q == 10 || rho == 0.0 {
                break;
            }
            if (ini - cur) * 1e3 < ini {
                nbad += 1;
            } else {
                nbad = 0;
            }
            if nbad >= 3 {
                break;
            }
        }
        it
    }
}

/// OptimizeSim3 의 짝 하나(원본이 간선 두 개를 넣은 짝만): 원본 vpMatches1 번호 · KF1/KF2 카메라 좌표 점 · 관측
#[derive(Clone, Debug)]
pub struct Sim3Pair {
    pub idx: i32,
    /// KF1 카메라 좌표의 점1(e21 이 씀) · KF2 카메라 좌표의 점2(e12 가 씀)
    pub p1c: V3d,
    pub p2c: V3d,
    pub obs1: [f64; 2],
    pub info1: [f64; 4],
    pub obs2: [f64; 2],
    pub info2: [f64; 4],
    pub in_kf2: bool,
}

#[derive(Clone, Debug)]
pub struct Sim3Problem {
    pub s12: G2oSim3,
    pub fix_scale: bool,
    pub th2: f32,
    pub cam1: [f32; 4],
    pub cam2: [f32; 4],
    pub pairs: Vec<Sim3Pair>,
}

#[derive(Clone, Debug)]
pub struct Sim3Result {
    /// 원본 반환값(0 = 이상치 정리 뒤 10 쌍 미만)
    pub n_in: i32,
    /// 결과 S12(0 이면 입력 그대로)
    pub s12: G2oSim3,
    /// 짝마다 vpMatches1 을 NULL 로 지웠는지
    pub rejected: Vec<bool>,
    /// 첫 최적화 반복 수 · 그 뒤 S12 · 간선 χ²(e12, e21) — 시험용
    pub it1: usize,
    pub mid_s12: G2oSim3,
    pub mid_chi2: Vec<[f64; 2]>,
    pub it2: i32,
}

/// Optimizer::OptimizeSim3 의 최적화 · 이상치 정리 부분(그래프 만들기 뒤부터 끝까지)
pub fn optimize_sim3(p: &Sim3Problem) -> Sim3Result {
    let delta = p.th2.sqrt() as f64;
    let mut edges = Vec::with_capacity(p.pairs.len() * 2);
    for q in &p.pairs {
        edges.push(ProjEdge { inverse: false, pt: q.p2c, obs: q.obs1, info: q.info1, delta: Some(delta), cam: p.cam1 });
        edges.push(ProjEdge { inverse: true, pt: q.p1c, obs: q.obs2, info: q.info2, delta: Some(delta), cam: p.cam2 });
    }
    let n = p.pairs.len();
    let mut active = vec![true; 2 * n];
    let mut ov = OneVertex { edges: &edges, active: &active, fix_scale: p.fix_scale, est: p.s12, err: vec![[0.0; 2]; 2 * n] };
    let it1 = ov.optimize(5);
    let mid_s12 = ov.est;
    let th2 = p.th2 as f64;
    let mid_chi2: Vec<[f64; 2]> =
        (0..n).map(|k| [chi2_2(&ov.err[2 * k], &edges[2 * k].info), chi2_2(&ov.err[2 * k + 1], &edges[2 * k + 1].info)]).collect();
    let mut est = ov.est;
    let mut err = ov.err;
    let mut rejected = vec![false; n];
    let mut nbad = 0;
    for k in 0..n {
        if mid_chi2[k][0] > th2 || mid_chi2[k][1] > th2 {
            rejected[k] = true;
            active[2 * k] = false;
            active[2 * k + 1] = false;
            nbad += 1;
            continue;
        }
        edges[2 * k].delta = None;
        edges[2 * k + 1].delta = None;
    }
    let more = if nbad > 0 { 10 } else { 5 };
    if (n as i32) - nbad < 10 {
        return Sim3Result { n_in: 0, s12: p.s12, rejected, it1, mid_s12, mid_chi2, it2: -1 };
    }
    let mut ov2 = OneVertex { edges: &edges, active: &active, fix_scale: p.fix_scale, est, err };
    let it2 = ov2.optimize(more);
    est = ov2.est;
    err = ov2.err;
    let _ = err;
    let mut n_in = 0;
    for k in 0..n {
        if !active[2 * k] {
            continue;
        }
        let c12 = chi2_2(&edges[2 * k].error(&est), &edges[2 * k].info);
        let c21 = chi2_2(&edges[2 * k + 1].error(&est), &edges[2 * k + 1].info);
        if c12 > th2 || c21 > th2 {
            rejected[k] = true;
        } else {
            n_in += 1;
        }
    }
    Sim3Result { n_in, s12: est, rejected, it1, mid_s12, mid_chi2, it2: it2 as i32 }
}

// ---------------------------------------------------------------------------------------------
// OptimizeEssentialGraph(두 판 공통 최적화): g2o::VertexSim3Expmap 정점 · EdgeSim3(정보 = 단위) · λ 초기 1e-16 · 20 번

#[derive(Clone, Debug)]
pub struct EgVertex {
    pub id: i32,
    pub fixed: bool,
    pub fix_scale: bool,
    pub est: G2oSim3,
}

/// EdgeSim3: 오차 = log(측정 · S0 · S1⁻¹). v0 · v1 = 정점 번호(입력 배열 자리)
#[derive(Clone, Debug)]
pub struct EgEdge {
    pub v0: usize,
    pub v1: usize,
    pub meas: G2oSim3,
}

#[inline]
pub fn eg_error(m: &G2oSim3, a: &G2oSim3, b: &G2oSim3) -> [f64; 7] {
    m.mul(a).mul(&b.inverse()).log()
}

/// χ² = e·(I·e): 7 길이 내적
#[inline]
pub fn chi2_7(e: &[f64; 7]) -> f64 {
    vred7(e, e)
}

pub struct EgOutcome {
    pub iterations: usize,
    /// 간선마다 마지막으로 계산한 χ²
    pub chi2: Vec<f64>,
}

/// 본질 그래프 풀이 상태(BlockSolver_7_3 의 Hpp 블록 · b)
pub struct EgSys<'a> {
    pub es: &'a [EgEdge],
    pub active: Vec<bool>,
    /// 정점마다 헤세 번호(고정이면 usize::MAX)
    pub hidx: Vec<usize>,
    pub nv: usize,
    /// (블록 열, 블록 행) 행 ≤ 열 → 7×7 열 우선
    pub blocks: BTreeMap<(usize, usize), [f64; 49]>,
    pub keys: Vec<(usize, usize)>,
    pub b: Vec<f64>,
    pub err: Vec<[f64; 7]>,
    pub ldlt: Option<Ldlt>,
}

impl<'a> EgSys<'a> {
    /// initializeOptimization + buildStructure
    pub fn new(vs: &[EgVertex], es: &'a [EgEdge]) -> EgSys<'a> {
        debug_assert!(vs.windows(2).all(|w| w[0].id < w[1].id));
        let mut hidx = vec![usize::MAX; vs.len()];
        let mut nv = 0;
        for (i, v) in vs.iter().enumerate() {
            if !v.fixed {
                hidx[i] = nv;
                nv += 1;
            }
        }
        let active: Vec<bool> = es.iter().map(|e| !(vs[e.v0].fixed && vs[e.v1].fixed)).collect();
        let mut blocks: BTreeMap<(usize, usize), [f64; 49]> = BTreeMap::new();
        for i in 0..nv {
            blocks.insert((i, i), [0.0; 49]);
        }
        for (k, e) in es.iter().enumerate() {
            if !active[k] {
                continue;
            }
            let (a, b) = (hidx[e.v0], hidx[e.v1]);
            if a != usize::MAX && b != usize::MAX {
                blocks.insert((a.max(b), a.min(b)), [0.0; 49]);
            }
        }
        let keys = blocks.keys().copied().collect();
        EgSys { es, active, hidx, nv, blocks, keys, b: vec![0.0; nv * 7], err: vec![[0.0; 7]; es.len()], ldlt: None }
    }

    pub fn compute_errors(&mut self, vs: &[EgVertex]) {
        for (k, e) in self.es.iter().enumerate() {
            if self.active[k] {
                self.err[k] = eg_error(&e.meas, &vs[e.v0].est, &vs[e.v1].est);
            }
        }
    }

    pub fn chi2_sum(&self) -> f64 {
        let mut s = 0.0;
        for k in 0..self.es.len() {
            if self.active[k] {
                s += chi2_7(&self.err[k]);
            }
        }
        s
    }

    /// 간선 하나의 수치 야코비안(열 d = 오차 7 길이): (정점 0, 정점 1)
    pub fn jacobians(&self, vs: &[EgVertex], k: usize) -> ([[f64; 7]; 7], [[f64; 7]; 7]) {
        let e = &self.es[k];
        let delta = 1e-9;
        let scalar = 1.0 / (2.0 * delta);
        let mut ja = [[0.0f64; 7]; 7];
        let mut jb = [[0.0f64; 7]; 7];
        let mut add = [0.0f64; 7];
        if !vs[e.v0].fixed {
            let v = &vs[e.v0];
            for d in 0..7 {
                add[d] = delta;
                let ep = eg_error(&e.meas, &oplus(&v.est, &mut add, v.fix_scale), &vs[e.v1].est);
                add[d] = -delta;
                let em = eg_error(&e.meas, &oplus(&v.est, &mut add, v.fix_scale), &vs[e.v1].est);
                add[d] = 0.0;
                for r in 0..7 {
                    ja[d][r] = scalar * (ep[r] - em[r]);
                }
            }
        }
        if !vs[e.v1].fixed {
            let v = &vs[e.v1];
            for d in 0..7 {
                add[d] = delta;
                let ep = eg_error(&e.meas, &vs[e.v0].est, &oplus(&v.est, &mut add, v.fix_scale));
                add[d] = -delta;
                let em = eg_error(&e.meas, &vs[e.v0].est, &oplus(&v.est, &mut add, v.fix_scale));
                add[d] = 0.0;
                for r in 0..7 {
                    jb[d][r] = scalar * (ep[r] - em[r]);
                }
            }
        }
        (ja, jb)
    }

    /// BlockSolver::buildSystem
    pub fn build(&mut self, vs: &[EgVertex]) {
        for v in self.blocks.values_mut() {
            *v = [0.0; 49];
        }
        self.b.iter_mut().for_each(|x| *x = 0.0);
        for k in 0..self.es.len() {
            if !self.active[k] {
                continue;
            }
            let e = &self.es[k];
            let (f0, f1) = (vs[e.v0].fixed, vs[e.v1].fixed);
            let (ja, jb) = self.jacobians(vs, k);
            let er = self.err[k];
            // omega_r = (−I)·e
            let om_r: [f64; 7] = std::array::from_fn(|r| {
                let mut c = (if r == 0 { -1.0 } else { -0.0 }) * er[0];
                for q in 1..7 {
                    c = fma(if q == r { -1.0 } else { -0.0 }, er[q], c);
                }
                c
            });
            let ha = self.hidx[e.v0];
            let hb = self.hidx[e.v1];
            if !f0 {
                let ato = xt_omega(&ja);
                for i in 0..7 {
                    self.b[ha * 7 + i] += vred7(&ja[i], &om_r);
                }
                add_prod7(self.blocks.get_mut(&(ha, ha)).unwrap(), &ato, &ja, 7);
                if !f1 {
                    if ha > hb {
                        // _hessianTransposed += Bᵀ · AtOᵀ (낱개): (i, j) = Σ B(q,i)·AtO(j,q)
                        let blk = self.blocks.get_mut(&(ha, hb)).unwrap();
                        for j in 0..7 {
                            for i in 0..7 {
                                let m: [f64; 7] = std::array::from_fn(|q| jb[i][q] * ato[j][q]);
                                blk[j * 7 + i] += tree(&m);
                            }
                        }
                    } else {
                        add_prod7(self.blocks.get_mut(&(hb, ha)).unwrap(), &ato, &jb, 7);
                    }
                }
            }
            if !f1 {
                for i in 0..7 {
                    self.b[hb * 7 + i] += vred7(&jb[i], &om_r);
                }
                let bto = xt_omega(&jb);
                add_prod7(self.blocks.get_mut(&(hb, hb)).unwrap(), &bto, &jb, 7);
            }
        }
    }

    pub fn max_diag(&self) -> f64 {
        let mut m = 0.0f64;
        for i in 0..self.nv {
            let h = &self.blocks[&(i, i)];
            for j in 0..7 {
                m = h[j * 8].abs().max(m);
            }
        }
        m
    }

    /// setLambda · LinearSolverEigen::solve(fillCCS 위 삼각) · restoreDiagonal. 실패면 None
    pub fn solve(&mut self, lambda: f64) -> Option<Vec<f64>> {
        let first = self.ldlt.is_none();
        let mut vals = Vec::new();
        let (mut ap, mut ai) = (vec![0usize], Vec::new());
        let keys = &self.keys;
        let mut k0 = 0;
        for c in 0..self.nv {
            let mut k1 = k0;
            while k1 < keys.len() && keys[k1].0 == c {
                k1 += 1;
            }
            for cc in 0..7 {
                for &(_, r) in &keys[k0..k1] {
                    let blk = &self.blocks[&(c, r)];
                    let n = if r == c { cc + 1 } else { 7 };
                    for rr in 0..n {
                        let mut v = blk[cc * 7 + rr];
                        if r == c && rr == cc {
                            v += lambda;
                        }
                        vals.push(v);
                        if first {
                            ai.push(r * 7 + rr);
                        }
                    }
                }
                if first {
                    ap.push(ai.len());
                }
            }
            k0 = k1;
        }
        if first {
            self.ldlt = Some(Ldlt::analyze(self.nv * 7, ap, ai));
        }
        let ld = self.ldlt.as_mut().unwrap();
        if !ld.factorize(&vals) {
            return None;
        }
        Some(ld.solve(&self.b))
    }

    /// SparseOptimizer::update(x): 정점마다 exp(δ)·S (고정 축척이면 x 의 6 번도 0 으로)
    pub fn update(&self, vs: &mut [EgVertex], x: &mut [f64]) {
        for (i, v) in vs.iter_mut().enumerate() {
            if self.hidx[i] != usize::MAX {
                let h = self.hidx[i];
                v.est = oplus(&v.est, &mut x[h * 7..h * 7 + 7], v.fix_scale);
            }
        }
    }
}

/// XᵀΩ(Ω = 단위, 7×7): (i, j) = vred7(X 열 i, Ω 열 j)
fn xt_omega(x: &[[f64; 7]; 7]) -> [[f64; 7]; 7] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let w: [f64; 7] = std::array::from_fn(|q| if q == j { 1.0 } else { 0.0 });
            vred7(&x[i], &w)
        })
    })
}

/// SparseOptimizer.initializeOptimization(); optimize(iters) — 정점 값은 제자리에서 바뀐다.
/// 정점 배열은 id 순이어야 한다(원본 _activeVertices 순서). λ 초기값 1e-16(원본 setUserLambdaInit).
pub fn optimize_essential_graph(vs: &mut [EgVertex], es: &[EgEdge], iters: usize) -> EgOutcome {
    let mut sy = EgSys::new(vs, es);
    let nv = sy.nv;
    let (mut lambda, mut ni, mut nbad) = (0.0f64, 2.0f64, 0);
    let mut it = 0;
    let mut xprev = vec![0.0f64; nv * 7];
    for iter in 0..iters {
        sy.compute_errors(vs);
        let mut cur = sy.chi2_sum();
        let ini = cur;
        sy.build(vs);
        if iter == 0 {
            lambda = 1e-16;
            ni = 2.0;
            nbad = 0;
        }
        let mut rho;
        let mut q = 0;
        loop {
            let saved: Vec<G2oSim3> = vs.iter().map(|v| v.est).collect();
            let xs = sy.solve(lambda);
            let ok = xs.is_some();
            let mut x = xs.unwrap_or_else(|| xprev.clone());
            sy.update(vs, &mut x);
            sy.compute_errors(vs);
            let mut temp = sy.chi2_sum();
            if !ok {
                temp = f64::MAX;
            }
            rho = cur - temp;
            let mut scale = 0.0;
            for j in 0..nv * 7 {
                scale += x[j] * (lambda * x[j] + sy.b[j]);
            }
            scale += 1e-3;
            rho /= scale;
            xprev = x;
            if rho > 0.0 && temp.is_finite() {
                let alpha = 1.0 - (2.0 * rho - 1.0).powf(3.0);
                let alpha = alpha.min(2.0 / 3.0);
                lambda *= (1.0f64 / 3.0).max(alpha);
                ni = 2.0;
                cur = temp;
            } else {
                lambda *= ni;
                ni *= 2.0;
                for (v, s) in vs.iter_mut().zip(&saved) {
                    v.est = *s;
                }
            }
            q += 1;
            if !(rho < 0.0 && q < 10) {
                break;
            }
        }
        it += 1;
        if q == 10 || rho == 0.0 {
            break;
        }
        if (ini - cur) * 1e3 < ini {
            nbad += 1;
        } else {
            nbad = 0;
        }
        if nbad >= 3 {
            break;
        }
    }
    let chi2 = (0..es.len()).map(|k| chi2_7(&sy.err[k])).collect();
    EgOutcome { iterations: it, chi2 }
}

/// 루프 판 마무리: 키프레임 자세 Tiw(Sophus SE3f: 사원수 [x,y,z,w] · 이동) = (q(f32) 정규화, t(f32)/s(f32))
pub fn sim3_to_se3f(s: &G2oSim3) -> ([f32; 4], [f32; 3]) {
    let q = [s.r[0] as f32, s.r[1] as f32, s.r[2] as f32, s.r[3] as f32];
    // Sophus SO3 정규화: 길이 = sqrt((x²+z²)+(y²+w²)), 성분 / 길이
    let sq = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    let len = sq.sqrt();
    let qn = [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
    let sf = s.s as f32;
    let t = [s.t[0] as f32 / sf, s.t[1] as f32 / sf, s.t[2] as f32 / sf];
    (qn, t)
}

/// 루프 판 점 고치기: P' = (S_corr⁻¹)·(S_ref·P) (double 로, 결과 f32)
pub fn correct_point(srw: &G2oSim3, corrected_swr: &G2oSim3, p: &[f32; 3]) -> [f32; 3] {
    let pd = [p[0] as f64, p[1] as f64, p[2] as f64];
    let o = corrected_swr.map(&srw.map(&pd));
    [o[0] as f32, o[1] as f32, o[2] as f32]
}

/// 시험용: OptimizeSim3 첫 최적화의 첫 반복 단계 (H, b, λ 넣어 푼 x)
pub fn sim3_first_stage(p: &Sim3Problem) -> ([f64; 49], [f64; 7], Option<Vec<f64>>) {
    let delta = p.th2.sqrt() as f64;
    let mut edges = Vec::with_capacity(p.pairs.len() * 2);
    for q in &p.pairs {
        edges.push(ProjEdge { inverse: false, pt: q.p2c, obs: q.obs1, info: q.info1, delta: Some(delta), cam: p.cam1 });
        edges.push(ProjEdge { inverse: true, pt: q.p1c, obs: q.obs2, info: q.info2, delta: Some(delta), cam: p.cam2 });
    }
    let active = vec![true; edges.len()];
    let mut ov = OneVertex { edges: &edges, active: &active, fix_scale: p.fix_scale, est: p.s12, err: vec![[0.0; 2]; edges.len()] };
    ov.compute_errors();
    let (h, b) = ov.build();
    let mut md = 0.0f64;
    for j in 0..7 {
        md = h[j * 8].abs().max(md);
    }
    let mut hl = h;
    for j in 0..7 {
        hl[j * 8] += 1e-5 * md;
    }
    (h, b, dense_ldlt_solve(&hl, 7, &b))
}

// ---------------------------------------------------------------------------------------------
// 그래프 만들기(원본 함수 앞부분). 입력은 원본이 KeyFrame · MapPoint 에서 읽는 값만 — Atlas 쪽 호출자가 채운다.

/// Eigen Matrix3f · Vector3f + Vector3f(행마다 a0b0 + (a1b1 + a2b2), `orb3rs::types::mat_vec` 와 같은 순서)
#[inline]
fn rp_t(r: &[[f32; 3]; 3], p: &[f32; 3], t: &[f32; 3]) -> [f32; 3] {
    let d = |m: &[f32; 3]| m[0] * p[0] + (m[1] * p[1] + m[2] * p[2]);
    [d(&r[0]) + t[0], d(&r[1]) + t[1], d(&r[2]) + t[2]]
}

/// OptimizeSim3 의 키프레임 쪽 입력: Tcw 회전(행 우선) · 이동, mvKeysUn(x, y, octave), mvInvLevelSigma2, 카메라 [fx, fy, cx, cy]
pub struct Sim3Kf<'a> {
    pub r: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub keys_un: &'a [(f32, f32, i32)],
    pub inv_level_sigma2: &'a [f32],
    pub cam: [f32; 4],
}

/// vpMatches1[i] 가 있는 자리 하나: KF1 의 i 번 지도 점(없으면 None) · 짝 지도 점 · 그 점의 KF2 번호(없으면 −1) · mnTrackScaleLevel
#[derive(Clone, Debug)]
pub struct Sim3Match {
    pub i: usize,
    /// (세계 위치, 나쁨)
    pub mp1: Option<([f32; 3], bool)>,
    pub mp2: ([f32; 3], bool),
    pub i2: i32,
    pub track_scale_level2: i32,
}

/// Optimizer::OptimizeSim3 의 그래프 만들기: 간선 두 개를 넣는 짝만 남긴 문제
pub fn build_sim3_problem(kf1: &Sim3Kf, kf2: &Sim3Kf, matches: &[Sim3Match], s12: G2oSim3, th2: f32, fix_scale: bool, all_points: bool) -> Sim3Problem {
    let mut pairs = Vec::new();
    let dv = |p: &[f32; 3]| [p[0] as f64, p[1] as f64, p[2] as f64];
    let info = |s: f32| [s as f64, 0.0, 0.0, s as f64];
    for m in matches {
        let Some((w1, bad1)) = m.mp1 else { continue };
        let (w2, bad2) = m.mp2;
        if bad1 || bad2 {
            continue;
        }
        let p1c = rp_t(&kf1.r, &w1, &kf1.t);
        let p2c = rp_t(&kf2.r, &w2, &kf2.t);
        if m.i2 < 0 && !all_points {
            continue;
        }
        if p2c[2] < 0.0 {
            continue;
        }
        let k1 = kf1.keys_un[m.i];
        let (obs2, oct2, in_kf2) = if m.i2 >= 0 {
            let k2 = kf2.keys_un[m.i2 as usize];
            ([k2.0 as f64, k2.1 as f64], k2.2, true)
        } else {
            // 원본 cv::KeyPoint(Point2f, mnTrackScaleLevel): 둘째 인자는 size 라 octave = 0
            let invz = 1.0f32 / p2c[2];
            ([(p2c[0] * invz) as f64, (p2c[1] * invz) as f64], 0, false)
        };
        pairs.push(Sim3Pair {
            idx: m.i as i32,
            p1c: dv(&p1c),
            p2c: dv(&p2c),
            obs1: [k1.0 as f64, k1.1 as f64],
            info1: info(kf1.inv_level_sigma2[k1.2 as usize]),
            obs2,
            info2: info(kf2.inv_level_sigma2[oct2 as usize]),
            in_kf2,
        });
    }
    Sim3Problem { s12, fix_scale, th2, cam1: kf1.cam, cam2: kf2.cam, pairs }
}

/// g2o::Sim3(Tcw.cast<double>().unit_quaternion(), t, 1): 사원수를 double 로 바꿔 다시 정규화(Sophus SO3 생성자)
pub fn se3f_to_sim3(q: &[f32; 4], t: &[f32; 3]) -> G2oSim3 {
    let d = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
    let n = ((d[0] * d[0] + d[2] * d[2]) + (d[1] * d[1] + d[3] * d[3])).sqrt();
    G2oSim3 { r: [d[0] / n, d[1] / n, d[2] / n, d[3] / n], t: [t[0] as f64, t[1] as f64, t[2] as f64], s: 1.0 }
}

/// 루프 판 OptimizeEssentialGraph 의 키프레임 입력(원본 vpKFs 순서로 준다)
#[derive(Clone, Debug)]
pub struct EgKf {
    pub id: u32,
    pub bad: bool,
    /// GetPose(): 사원수 [x, y, z, w] · 이동
    pub q: [f32; 4],
    pub t: [f32; 3],
    pub corrected: Option<G2oSim3>,
    pub non_corrected: Option<G2oSim3>,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
    /// GetLoopEdges() 순서
    pub loop_edges: Vec<u32>,
    /// GetCovisiblesByWeight(100): (id, 나쁨)
    pub covis100: Vec<(u32, bool)>,
}

/// 루프 판 입력. loop_conn = LoopConnections(순서대로): (키프레임, [(이웃, GetWeight)])
pub struct EgLoopInput {
    pub cur_kf: u32,
    pub loop_kf: u32,
    pub init_kf: u32,
    pub fix_scale: bool,
    pub kfs: Vec<EgKf>,
    pub loop_conn: Vec<(u32, Vec<(u32, i32)>)>,
}

/// 루프 판 그래프. 정점은 id 순(최적화 입력), scw = vScw(정점 첫 값, 없는 id 는 기본 Sim3)
pub struct EgGraph {
    pub vertices: Vec<EgVertex>,
    pub edges: Vec<EgEdge>,
    pub scw: BTreeMap<u32, G2oSim3>,
}

/// Optimizer::OptimizeEssentialGraph(Map*, …) 의 그래프 만들기(관성 간선 없음 — RGB-D)
pub fn build_eg_loop(inp: &EgLoopInput) -> EgGraph {
    let min_feat = 100;
    let mut scw: BTreeMap<u32, G2oSim3> = BTreeMap::new();
    let mut vlist: Vec<EgVertex> = Vec::new();
    let by_id: BTreeMap<u32, &EgKf> = inp.kfs.iter().map(|k| (k.id, k)).collect();
    for k in &inp.kfs {
        if k.bad {
            continue;
        }
        let s = match k.corrected {
            Some(c) => c,
            None => se3f_to_sim3(&k.q, &k.t),
        };
        scw.insert(k.id, s);
        vlist.push(EgVertex { id: k.id as i32, fixed: k.id == inp.init_kf, fix_scale: inp.fix_scale, est: s });
    }
    vlist.sort_by_key(|v| v.id);
    let pos: BTreeMap<u32, usize> = vlist.iter().enumerate().map(|(i, v)| (v.id as u32, i)).collect();
    let vscw = |id: u32| scw.get(&id).copied().unwrap_or_default();
    let nonc = |id: u32| by_id.get(&id).and_then(|k| k.non_corrected);
    let mut edges = Vec::new();
    let mut push = |a: u32, b: u32, m: G2oSim3, edges: &mut Vec<EgEdge>| {
        // 원본 addEdge 는 없는 정점(null)이 있으면 넣지 않음
        if let (Some(&i), Some(&j)) = (pos.get(&a), pos.get(&b)) {
            edges.push(EgEdge { v0: i, v1: j, meas: m });
        }
    };
    let mut inserted = std::collections::BTreeSet::new();
    for (i, conns) in &inp.loop_conn {
        let swi = vscw(*i).inverse();
        for &(j, w) in conns {
            if (*i != inp.cur_kf || j != inp.loop_kf) && w < min_feat {
                continue;
            }
            let sji = vscw(j).mul(&swi);
            push(*i, j, sji, &mut edges);
            inserted.insert((*i.min(&j), *i.max(&j)));
        }
    }
    for k in &inp.kfs {
        let swi = match k.non_corrected {
            Some(s) => s.inverse(),
            None => vscw(k.id).inverse(),
        };
        if let Some(p) = k.parent {
            let sjw = nonc(p).unwrap_or_else(|| vscw(p));
            push(k.id, p, sjw.mul(&swi), &mut edges);
        }
        for &l in &k.loop_edges {
            if l < k.id {
                let slw = nonc(l).unwrap_or_else(|| vscw(l));
                push(k.id, l, slw.mul(&swi), &mut edges);
            }
        }
        for &(n, nbad) in &k.covis100 {
            if Some(n) != k.parent && !k.children.contains(&n) && !nbad && n < k.id {
                if inserted.contains(&(n.min(k.id), n.max(k.id))) {
                    continue;
                }
                let snw = nonc(n).unwrap_or_else(|| vscw(n));
                push(k.id, n, snw.mul(&swi), &mut edges);
            }
        }
    }
    EgGraph { vertices: vlist, edges, scw }
}

/// 루프 판 지도 점 입력(vpMPs 순서)
#[derive(Clone, Debug)]
pub struct EgMp {
    pub bad: bool,
    pub corrected_by_kf: u64,
    pub corrected_reference: u64,
    pub ref_kf: u32,
    pub pos: [f32; 3],
}

/// 루프 판 마무리 점 고치기: 나쁘지 않은 점마다 (기준 키프레임 id, 새 위치)
pub fn eg_loop_points(g: &EgGraph, cur_kf: u32, mps: &[EgMp]) -> Vec<(u32, [f32; 3])> {
    let mut out = Vec::new();
    for m in mps {
        if m.bad {
            continue;
        }
        let r = if m.corrected_by_kf == cur_kf as u64 { m.corrected_reference as u32 } else { m.ref_kf };
        let srw = g.scw.get(&r).copied().unwrap_or_default();
        let cswr = match g.vertices.binary_search_by_key(&(r as i32), |v| v.id) {
            Ok(i) => g.vertices[i].est.inverse(),
            Err(_) => G2oSim3::default(),
        };
        out.push((r, correct_point(&srw, &cswr, &m.pos)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_ldlt_solves() {
        // 대칭 양의 정부호 7×7: H = Bᵀ B + I
        let mut h = [0.0f64; 49];
        for i in 0..7 {
            for j in 0..7 {
                let mut s = 0.0;
                for k in 0..9 {
                    let b = |r: usize, c: usize| ((r * 7 + c) as f64 * 0.37).sin();
                    s += b(k, i) * b(k, j);
                }
                h[j * 7 + i] = s + if i == j { 1.0 } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..7).map(|i| i as f64 - 3.0).collect();
        let x = dense_ldlt_solve(&h, 7, &b).unwrap();
        for i in 0..7 {
            let r: f64 = (0..7).map(|j| h[j * 7 + i] * x[j]).sum();
            assert!((r - b[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn essential_graph_pulls_to_loop() {
        // 0(고정) — 1 — 2 사슬 + 2→0 루프 간선: 측정과 맞는 자리로 모인다
        let step = G2oSim3::exp(&[0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let mut vs: Vec<EgVertex> = (0..3)
            .map(|i| EgVertex { id: i, fixed: i == 0, fix_scale: true, est: G2oSim3::default() })
            .collect();
        vs[1].est = step;
        vs[2].est = step.mul(&step).mul(&G2oSim3::exp(&[0.0, 0.0, 0.05, 0.1, 0.0, 0.0, 0.0]));
        // 측정 = S_j · S_i⁻¹ (원본 Sji = Sjw · Swi), 참값 S1 = step, S2 = step²
        let inv = step.inverse();
        let es = vec![
            EgEdge { v0: 1, v1: 0, meas: inv },
            EgEdge { v0: 2, v1: 1, meas: inv },
            EgEdge { v0: 2, v1: 0, meas: inv.mul(&inv) },
        ];
        let before: f64 = es.iter().map(|e| chi2_7(&eg_error(&e.meas, &vs[e.v0].est, &vs[e.v1].est))).sum();
        let o = optimize_essential_graph(&mut vs, &es, 20);
        let after: f64 = o.chi2.iter().sum();
        assert!(after < before * 1e-6, "{before} → {after}");
    }
}
