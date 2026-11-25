//! g2o Sim3 자세 그래프(VertexSim3Expmap · EdgeSim3, BlockSolver_7_3 + LinearSolverEigen + Levenberg)를 기준 빌드 원본과
//! 비트 같게. ORB-SLAM3 의 본질 그래프 최적화(합치기 판 `OptimizeEssentialGraph(pCurKF, …)`)가 쓰는 g2o 경로.
//! g2o::Sim3 연산 · 희소 LDLT(AMD) 는 BA 포터의 `orb3ba::{sim3, ldlt}` 를 그대로 쓴다.
//!
//! 원본 순서(libg2o · 기준 빌드 역어셈블 확인):
//! - 오차 e = log(C · v0 · v1⁻¹), 정보 행렬 Ω = I(이 쓰임에서는 항상) 이라 Ω·e = e, AᵀΩ = Aᵀ 는 정확히 같음.
//! - χ² = eᵀ(Ωe): 7 길이 내적 = p6 + (((p4+p2)+p0) + ((p5+p3)+p1)) (SSE 2 묶음 세 개 + 낱개).
//! - 야코비안: 수치(±1e-9, 차이 × 1/(2·1e-9)), 정점 oplus = exp(δ)·T, 척도 고정 정점은 δ₆ = 0.
//! - 이차 형식: b += Aᵀ(−e)(7 길이 내적), H += (Aᵀ)·A 는 Eigen 지연 곱 — 0..5 행은 k=0 곱에서 시작해 k=1..6 FMA 사슬,
//!   6 행은 ((m6+m5)+(m4+m3)) + ((m2+m1)+m0) 를 더함.
//! - BlockSolver 는 점이 없어도 슈어 경로(Hschur = 0 + Hpp, b 그대로) → 위 삼각 CCS → SimplicialLDLT.
use crate::orb3ba::ldlt::Ldlt;
use crate::orb3ba::sim3::Sim3;

#[derive(Clone, Debug)]
pub struct PgVertex {
    pub id: i32,
    pub fixed: bool,
    pub fix_scale: bool,
    pub est: Sim3,
}
#[derive(Clone, Debug)]
pub struct PgEdge {
    /// setVertex(0, ·) · setVertex(1, ·) 의 정점 번호(입력 vertices 안)
    pub v0: usize,
    pub v1: usize,
    pub meas: Sim3,
}
#[derive(Clone, Debug, Default)]
pub struct PoseGraph {
    pub vertices: Vec<PgVertex>,
    /// 원본 addEdge 순서
    pub edges: Vec<PgEdge>,
    /// setUserLambdaInit(0 이면 τ·최대 대각)
    pub lambda_init: f64,
}

type M7 = [f64; 49]; // 열 우선
const DELTA: f64 = 1e-9;

#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
/// 7 길이 내적(원본 묶음 순서)
#[inline]
pub fn dot7(a: &[f64], b: &[f64]) -> f64 {
    let p = |i: usize| a[i] * b[i];
    let l0 = (p(4) + p(2)) + p(0);
    let l1 = (p(5) + p(3)) + p(1);
    p(6) + (l0 + l1)
}
/// 7 항 낱개 곱 나무 합 ((p3+p4)+(p5+p6)) + ((p1+p2)+p0)
#[inline]
fn tree7(a: &[f64], b: &[f64]) -> f64 {
    let p = |i: usize| a[i] * b[i];
    ((p(3) + p(4)) + (p(5) + p(6))) + ((p(1) + p(2)) + p(0))
}
/// dst += L·R (7×7, 열 우선) — Eigen 지연 곱 더하기
fn lazy_add(dst: &mut M7, l: &M7, r: &M7) {
    for j in 0..7 {
        let rc = &r[j * 7..j * 7 + 7];
        for row in 0..6 {
            let mut acc = l[row] * rc[0];
            for k in 1..7 {
                acc = fma(l[k * 7 + row], rc[k], acc);
            }
            dst[j * 7 + row] = acc + dst[j * 7 + row];
        }
        let m = |k: usize| l[k * 7 + 6] * rc[k];
        let s = ((m(6) + m(5)) + (m(4) + m(3))) + ((m(2) + m(1)) + m(0));
        dst[j * 7 + 6] = s + dst[j * 7 + 6];
    }
}
fn transpose7(a: &M7) -> M7 {
    std::array::from_fn(|k| a[(k % 7) * 7 + k / 7])
}

/// VertexSim3Expmap::oplusImpl: (척도 고정이면 δ₆ = 0) exp(δ)·T
fn oplus(est: &Sim3, d: &[f64; 7], fix_scale: bool) -> Sim3 {
    let mut u = *d;
    if fix_scale {
        u[6] = 0.0;
    }
    Sim3::exp(&u).mul(est)
}
fn edge_error(meas: &Sim3, a: &Sim3, b: &Sim3) -> [f64; 7] {
    meas.mul(a).mul(&b.inverse()).log()
}

pub struct Outcome {
    pub est: Vec<Sim3>,
    pub iterations: usize,
    /// 반복마다(postIteration) 정점 값
    pub trace: Vec<Vec<Sim3>>,
}

struct Sys<'a> {
    g: &'a PoseGraph,
    est: Vec<Sim3>,
    active: Vec<bool>,
    err: Vec<[f64; 7]>,
    /// 정점 → 헤세 번호(고정이면 NONE)
    hidx: Vec<usize>,
    order: Vec<usize>,
    /// 블록 (행, 열) 위 삼각 → 값
    h: std::collections::BTreeMap<(usize, usize), M7>,
    b: Vec<f64>,
}
const NONE: usize = usize::MAX;

impl<'a> Sys<'a> {
    fn new(g: &'a PoseGraph) -> Sys<'a> {
        let n = g.vertices.len();
        let active: Vec<bool> = g.edges.iter().map(|e| !(g.vertices[e.v0].fixed && g.vertices[e.v1].fixed)).collect();
        let mut has = vec![false; n];
        for (e, &a) in g.edges.iter().zip(&active) {
            if a {
                has[e.v0] = true;
                has[e.v1] = true;
            }
        }
        let mut av: Vec<usize> = (0..n).filter(|&i| has[i]).collect();
        av.sort_by_key(|&i| g.vertices[i].id);
        let mut hidx = vec![NONE; n];
        let mut order = Vec::new();
        for &i in &av {
            if !g.vertices[i].fixed {
                hidx[i] = order.len();
                order.push(i);
            }
        }
        let mut h = std::collections::BTreeMap::new();
        for &i in &order {
            h.insert((hidx[i], hidx[i]), [0.0; 49]);
        }
        for (e, &a) in g.edges.iter().zip(&active) {
            if !a {
                continue;
            }
            let (x, y) = (hidx[e.v0], hidx[e.v1]);
            if x != NONE && y != NONE {
                h.insert((x.min(y), x.max(y)), [0.0; 49]);
            }
        }
        let np = order.len();
        Sys { g, est: g.vertices.iter().map(|v| v.est).collect(), err: vec![[0.0; 7]; g.edges.len()], active, hidx, order, h, b: vec![0.0; np * 7] }
    }
    fn compute_errors(&mut self) {
        for (k, e) in self.g.edges.iter().enumerate() {
            if self.active[k] {
                self.err[k] = edge_error(&e.meas, &self.est[e.v0], &self.est[e.v1]);
            }
        }
    }
    fn chi2_of(e: &[f64; 7]) -> f64 {
        dot7(e, e)
    }
    fn total_chi2(&self) -> f64 {
        let mut s = 0.0;
        for k in 0..self.g.edges.len() {
            if self.active[k] {
                s += Self::chi2_of(&self.err[k]);
            }
        }
        s
    }
    /// BaseBinaryEdge::linearizeOplus(수치)
    fn jacobian(&self, k: usize, which: usize) -> M7 {
        let e = &self.g.edges[k];
        let v = if which == 0 { e.v0 } else { e.v1 };
        let fs = self.g.vertices[v].fix_scale;
        let scalar = 1.0 / (2.0 * DELTA);
        let mut j = [0.0; 49];
        for d in 0..7 {
            let mut add = [0.0; 7];
            add[d] = DELTA;
            let p = oplus(&self.est[v], &add, fs);
            let e1 = if which == 0 { edge_error(&e.meas, &p, &self.est[e.v1]) } else { edge_error(&e.meas, &self.est[e.v0], &p) };
            add[d] = -DELTA;
            let p = oplus(&self.est[v], &add, fs);
            let e2 = if which == 0 { edge_error(&e.meas, &p, &self.est[e.v1]) } else { edge_error(&e.meas, &self.est[e.v0], &p) };
            for r in 0..7 {
                j[d * 7 + r] = scalar * (e1[r] - e2[r]);
            }
        }
        j
    }
    fn build_system(&mut self) -> Vec<(M7, M7)> {
        for v in self.h.values_mut() {
            *v = [0.0; 49];
        }
        self.b.iter_mut().for_each(|x| *x = 0.0);
        let mut jac = Vec::new();
        for k in 0..self.g.edges.len() {
            if !self.active[k] {
                jac.push(([0.0; 49], [0.0; 49]));
                continue;
            }
            let e = &self.g.edges[k];
            let (fi, fj) = (self.g.vertices[e.v0].fixed, self.g.vertices[e.v1].fixed);
            let a = if fi { [0.0; 49] } else { self.jacobian(k, 0) };
            let bm = if fj { [0.0; 49] } else { self.jacobian(k, 1) };
            let w: [f64; 7] = std::array::from_fn(|i| -self.err[k][i]);
            let (hi, hj) = (self.hidx[e.v0], self.hidx[e.v1]);
            let at = transpose7(&a);
            if !fi {
                for r in 0..7 {
                    self.b[hi * 7 + r] = dot7(&a[r * 7..r * 7 + 7], &w) + self.b[hi * 7 + r];
                }
                lazy_add(self.h.get_mut(&(hi, hi)).unwrap(), &at, &a);
                if !fj {
                    if hi < hj {
                        lazy_add(self.h.get_mut(&(hi, hj)).unwrap(), &at, &bm);
                    } else {
                        // _hessianTransposed += Bᵀ·AtOᵀ: 칸마다 낱개 곱 나무 합 ((p3+p4)+(p5+p6)) + ((p1+p2)+p0)
                        let d = self.h.get_mut(&(hj, hi)).unwrap();
                        for c in 0..7 {
                            for r in 0..7 {
                                d[c * 7 + r] = tree7(&bm[r * 7..r * 7 + 7], &a[c * 7..c * 7 + 7]) + d[c * 7 + r];
                            }
                        }
                    }
                }
            }
            if !fj {
                for r in 0..7 {
                    self.b[hj * 7 + r] = dot7(&bm[r * 7..r * 7 + 7], &w) + self.b[hj * 7 + r];
                }
                let bt = transpose7(&bm);
                lazy_add(self.h.get_mut(&(hj, hj)).unwrap(), &bt, &bm);
            }
            jac.push((a, bm));
        }
        jac
    }
    fn max_diag(&self) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..self.order.len() {
            let d = &self.h[&(i, i)];
            for k in 0..7 {
                m = m.max(d[k * 8].abs());
            }
        }
        m
    }
    /// setLambda · 슈어 경로 · LDLT · restoreDiagonal
    fn solve(&mut self, ld: &mut Option<Ldlt>, lambda: f64) -> Option<Vec<f64>> {
        let np = self.order.len();
        let mut sch = self.h.clone();
        for v in sch.values_mut() {
            *v = [0.0; 49];
        }
        for (key, blk) in &self.h {
            let d = sch.get_mut(key).unwrap();
            for k in 0..49 {
                let add = if key.0 == key.1 && k % 8 == 0 { blk[k] + lambda } else { blk[k] };
                d[k] += add;
            }
        }
        let first = ld.is_none();
        let (mut ap, mut ai, mut vals) = (vec![0usize], Vec::new(), Vec::new());
        let keys: Vec<(usize, usize)> = sch.keys().copied().collect();
        // 위 삼각 CCS: 블록 열 c · 블록 안 열 cc · 블록 행(오름차순) · 블록 안 행
        for c in 0..np {
            let rows: Vec<usize> = keys.iter().filter(|k| k.1 == c).map(|k| k.0).collect();
            for cc in 0..7 {
                for &r in &rows {
                    let blk = &sch[&(r, c)];
                    let n = if r == c { cc + 1 } else { 7 };
                    for rr in 0..n {
                        vals.push(blk[cc * 7 + rr]);
                        if first {
                            ai.push(r * 7 + rr);
                        }
                    }
                }
                if first {
                    ap.push(ai.len());
                }
            }
        }
        if first {
            *ld = Some(Ldlt::analyze(np * 7, ap, ai));
        }
        let l = ld.as_mut().unwrap();
        if !l.factorize(&vals) {
            return None;
        }
        let bs: Vec<f64> = self.b.iter().map(|&x| x - 0.0).collect();
        Some(l.solve(&bs))
    }
    /// SparseOptimizer::update: 헤세 순서로 oplus. 척도 고정 정점은 x 의 그 칸도 0 이 됨(원본은 갱신 배열을 고쳐 씀)
    fn update(&mut self, x: &mut [f64]) {
        for (i, &v) in self.order.iter().enumerate() {
            let fs = self.g.vertices[v].fix_scale;
            if fs {
                x[i * 7 + 6] = 0.0;
            }
            let d: [f64; 7] = std::array::from_fn(|j| x[i * 7 + j]);
            self.est[v] = oplus(&self.est[v], &d, false);
        }
    }
}

/// 첫 선형화(시험용): 오차 · χ² 합 · 야코비안 · H 블록 · b · 첫 λ · x
pub struct FirstLin {
    pub chi2: f64,
    pub err: Vec<[f64; 7]>,
    pub jac: Vec<([f64; 49], [f64; 49])>,
    pub h: std::collections::BTreeMap<(usize, usize), [f64; 49]>,
    pub b: Vec<f64>,
    pub lambda: f64,
    pub x: Option<Vec<f64>>,
}
pub fn first_linearization(g: &PoseGraph) -> FirstLin {
    let mut s = Sys::new(g);
    s.compute_errors();
    let chi2 = s.total_chi2();
    let jac = s.build_system();
    let lambda = if g.lambda_init > 0.0 { g.lambda_init } else { 1e-5 * s.max_diag() };
    let mut ld = None;
    let x = s.solve(&mut ld, lambda);
    FirstLin { chi2, err: s.err.clone(), jac, h: s.h.clone(), b: s.b.clone(), lambda, x }
}

/// optimizer.initializeOptimization(); optimizer.optimize(iters) (Levenberg, Raul 멈춤 조건, 시도 10 번)
pub fn optimize(g: &PoseGraph, iters: usize) -> Outcome {
    let mut s = Sys::new(g);
    let mut ld: Option<Ldlt> = None;
    let mut lambda = 0.0;
    let mut ni = 2.0;
    let mut nbad = 0;
    let mut it = 0;
    let mut trace = Vec::new();
    for i in 0..iters {
        s.compute_errors();
        let mut cur = s.total_chi2();
        let ini = cur;
        s.build_system();
        if i == 0 {
            lambda = if g.lambda_init > 0.0 { g.lambda_init } else { 1e-5 * s.max_diag() };
            ni = 2.0;
            nbad = 0;
        }
        let mut rho;
        let mut q = 0;
        loop {
            let saved = s.est.clone();
            let x = s.solve(&mut ld, lambda);
            let ok = x.is_some();
            let mut xv = x.unwrap_or_else(|| vec![0.0; s.order.len() * 7]);
            s.update(&mut xv);
            s.compute_errors();
            let mut temp = s.total_chi2();
            if !ok {
                temp = f64::MAX;
            }
            rho = cur - temp;
            let mut scale = 0.0;
            for j in 0..xv.len() {
                scale += xv[j] * (lambda * xv[j] + s.b[j]);
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
                s.est = saved;
            }
            q += 1;
            if !(rho < 0.0 && q < 10) {
                break;
            }
        }
        // 원본 optimize 는 Terminate 여도 반복 수를 세고 postIteration 을 부름
        it += 1;
        trace.push(s.est.clone());
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
    Outcome { est: s.est, iterations: it, trace }
}

#[cfg(test)]
#[path = "sim3pg_bit.rs"]
mod bit;
