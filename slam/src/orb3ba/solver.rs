//! g2o SparseOptimizer + OptimizationAlgorithmLevenberg + BlockSolver<6,3> 경로(ORB-SLAM3 판 g2o).
//! 순서: 정점 = id 순(_activeVertices 정렬), 헤세 번호 = 고정 아닌 자세 먼저 · 다음 점(buildIndexMapping),
//! 간선 = 추가 순서(EdgeIDCompare), 모든 정점이 고정인 간선은 빠짐.
use super::edges::{self, Kind};
use super::ldlt::Ldlt;
use super::se3::{self, fused};
use super::{Est, Problem};
use std::collections::BTreeMap;

/// 결과: 정점 값(입력 순서), 간선마다 마지막 χ²(원본 e->chi2(): 마지막 computeActiveErrors 의 오차), 반복 수
#[derive(Clone, Debug)]
pub struct Outcome {
    pub est: Vec<Est>,
    pub chi2: Vec<f64>,
    pub iterations: usize,
}

/// 풀이 상태(원본의 정점 · 간선 · 블록 행렬)
pub struct Sys<'a> {
    pub p: &'a Problem,
    pub est: Vec<Est>,
    /// 간선마다 활성 여부 · 오차
    pub active: Vec<bool>,
    pub err: Vec<[f64; 3]>,
    /// 정점마다 헤세 번호(자세: 0.., 점: 0..; 없으면 usize::MAX)
    pub hidx: Vec<usize>,
    /// 헤세 번호 → 정점 (자세, 점)
    pub poses: Vec<usize>,
    pub points: Vec<usize>,
    pub hpp: Vec<[f64; 36]>,
    pub bp: Vec<[f64; 6]>,
    pub hll: Vec<[f64; 9]>,
    pub bl: Vec<[f64; 3]>,
    /// (자세, 점) → Hpl 블록(6×3 열 우선)
    pub hpl: BTreeMap<(usize, usize), [f64; 18]>,
    /// 점마다 (자세 번호 오름차순, Hpl 키)
    pub pl_cols: Vec<Vec<usize>>,
    /// 활성 정점(_activeVertices: id 순)
    pub act_v: Vec<usize>,
}

pub const NONE: usize = usize::MAX;

fn pose_of(e: &Est) -> (&[f64; 4], &[f64; 3]) {
    match e {
        Est::Pose { q, t } => (q, t),
        _ => panic!("자세 정점이 아님"),
    }
}
fn point_of(e: &Est) -> &[f64; 3] {
    match e {
        Est::Point(p) => p,
        _ => panic!("점 정점이 아님"),
    }
}

impl<'a> Sys<'a> {
    pub fn new(p: &'a Problem) -> Sys<'a> {
        let n = p.vertices.len();
        let fixed = |i: usize| p.vertices[i].fixed;
        let active: Vec<bool> = p
            .edges
            .iter()
            .map(|e| match e.kind {
                Kind::Gravity => !fixed(e.v0),
                _ => !(fixed(e.v0) && fixed(e.v1)),
            })
            .collect();
        let mut has = vec![false; n];
        for (e, &a) in p.edges.iter().zip(&active) {
            if a {
                has[e.v0] = true;
                if !matches!(e.kind, Kind::Gravity) {
                    has[e.v1] = true;
                }
            }
        }
        let mut order: Vec<usize> = (0..n).filter(|&i| has[i]).collect();
        order.sort_by_key(|&i| p.vertices[i].id);
        let mut hidx = vec![NONE; n];
        let mut poses = Vec::new();
        let mut points = Vec::new();
        for &i in &order {
            if fixed(i) {
                continue;
            }
            match p.vertices[i].est {
                Est::Pose { .. } => {
                    hidx[i] = poses.len();
                    poses.push(i);
                }
                Est::Point(_) => {}
            }
        }
        for &i in &order {
            if fixed(i) {
                continue;
            }
            if let Est::Point(_) = p.vertices[i].est {
                hidx[i] = points.len();
                points.push(i);
            }
        }
        let mut pl_cols = vec![Vec::new(); points.len()];
        let mut hpl = BTreeMap::new();
        for (e, &a) in p.edges.iter().zip(&active) {
            if !a || matches!(e.kind, Kind::Gravity) {
                continue;
            }
            let (pi, li) = (hidx[e.v1], hidx[e.v0]);
            if pi != NONE && li != NONE && !hpl.contains_key(&(pi, li)) {
                hpl.insert((pi, li), [0.0; 18]);
                pl_cols[li].push(pi);
            }
        }
        for c in pl_cols.iter_mut() {
            c.sort_unstable();
        }
        Sys {
            p,
            est: p.vertices.iter().map(|v| v.est.clone()).collect(),
            err: vec![[0.0; 3]; p.edges.len()],
            active,
            hpp: vec![[0.0; 36]; poses.len()],
            bp: vec![[0.0; 6]; poses.len()],
            hll: vec![[0.0; 9]; points.len()],
            bl: vec![[0.0; 3]; points.len()],
            hidx,
            poses,
            points,
            hpl,
            pl_cols,
            act_v: order,
        }
    }

    fn edge_error(&self, k: usize) -> [f64; 3] {
        let e = &self.p.edges[k];
        match e.kind {
            Kind::Gravity => {
                let (q, t) = pose_of(&self.est[e.v0]);
                edges::error(e, q, t, &[0.0; 3])
            }
            _ => {
                let (q, t) = pose_of(&self.est[e.v1]);
                edges::error(e, q, t, point_of(&self.est[e.v0]))
            }
        }
    }

    /// SparseOptimizer::computeActiveErrors
    pub fn compute_errors(&mut self) {
        for k in 0..self.p.edges.len() {
            if self.active[k] {
                self.err[k] = self.edge_error(k);
            }
        }
    }

    pub fn chi2(&self, k: usize) -> f64 {
        edges::chi2(self.p.edges[k].dim(), &self.err[k], &self.p.edges[k].info)
    }

    /// activeRobustChi2: 간선 순서대로 ρ0(또는 χ²) 합
    pub fn robust_chi2(&self) -> f64 {
        let mut s = 0.0;
        for k in 0..self.p.edges.len() {
            if !self.active[k] {
                continue;
            }
            let c = self.chi2(k);
            s += match self.p.edges[k].delta {
                Some(d) => edges::huber(c, d).0,
                None => c,
            };
        }
        s
    }

    /// 중력 간선 수치 야코비안(BaseUnaryEdge::linearizeOplus: ±1e-9, oplus · computeError · pop)
    fn gravity_jacobian(&self, k: usize) -> [f64; 18] {
        let e = &self.p.edges[k];
        let (q0, t0) = pose_of(&self.est[e.v0]);
        let delta = 1e-9;
        let scalar = 1.0 / (2.0 * delta);
        let mut j = [0.0; 18];
        for d in 0..6 {
            let mut add = [0.0; 6];
            add[d] = delta;
            let (mut q, mut t) = (*q0, *t0);
            se3::oplus_se3(&mut q, &mut t, &add);
            let e1 = edges::error(e, &q, &t, &[0.0; 3]);
            add[d] = -delta;
            let (mut q, mut t) = (*q0, *t0);
            se3::oplus_se3(&mut q, &mut t, &add);
            let e2 = edges::error(e, &q, &t, &[0.0; 3]);
            for r in 0..3 {
                j[d * 3 + r] = scalar * (e1[r] - e2[r]);
            }
        }
        j
    }

    /// 간선 하나를 선형화해 블록에 더함. 돌려줌: 야코비안(시험용)
    pub fn add_edge(&mut self, k: usize) -> ([f64; 9], [f64; 18]) {
        let e = &self.p.edges[k];
        let err = self.err[k];
        match e.kind {
            Kind::Gravity => {
                let j = self.gravity_jacobian(k);
                let pi = self.hidx[e.v0];
                edges::quad_gravity(e, &err, &j, &mut self.hpp[pi], &mut self.bp[pi]);
                ([0.0; 9], j)
            }
            _ => {
                let (q, t) = pose_of(&self.est[e.v1]);
                let pt = point_of(&self.est[e.v0]);
                let mut xi = [0.0; 9];
                let mut xj = [0.0; 18];
                edges::jacobian(e, q, t, pt, &mut xi, &mut xj);
                let li = self.hidx[e.v0];
                let pi = self.hidx[e.v1];
                let pose = if pi != NONE {
                    let h = self.hpl.get_mut(&(pi, li)).unwrap();
                    Some((&mut self.hpp[pi], &mut self.bp[pi], h))
                } else {
                    None
                };
                edges::quad_binary(e, &err, &xi, &xj, &mut self.hll[li], &mut self.bl[li], pose);
                (xi, xj)
            }
        }
    }

    /// BlockSolver::buildSystem
    pub fn build_system(&mut self) {
        for h in self.hpp.iter_mut() {
            *h = [0.0; 36];
        }
        for b in self.bp.iter_mut() {
            *b = [0.0; 6];
        }
        for h in self.hll.iter_mut() {
            *h = [0.0; 9];
        }
        for b in self.bl.iter_mut() {
            *b = [0.0; 3];
        }
        for h in self.hpl.values_mut() {
            *h = [0.0; 18];
        }
        for k in 0..self.p.edges.len() {
            if self.active[k] {
                self.add_edge(k);
            }
        }
    }
}

/// Eigen compute_inverse<3×3>: 0 열 여인수 → det = (c0·m00 + c1·m10) + c2·m20 → 1/det 곱
pub fn inv3(m: &[f64; 9]) -> [f64; 9] {
    let a = |i: usize, j: usize| m[j * 3 + i];
    let cof = |i: usize, j: usize| {
        let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
        a(i1, j1) * a(i2, j2) - a(i1, j2) * a(i2, j1)
    };
    let c0 = [cof(0, 0), cof(1, 0), cof(2, 0)];
    let det = (c0[0] * a(0, 0) + c0[1] * a(1, 0)) + c0[2] * a(2, 0);
    let id = 1.0 / det;
    let mut r = [0.0; 9];
    r[2 * 3 + 1] = cof(2, 1) * id;
    r[3 + 2] = cof(1, 2) * id;
    r[2 * 3 + 2] = cof(2, 2) * id;
    r[1] = cof(0, 1) * id;
    r[3 + 1] = cof(1, 1) * id;
    r[2] = cof(0, 2) * id;
    r[0] = c0[0] * id;
    r[3] = c0[1] * id;
    r[6] = c0[2] * id;
    r
}

/// 슈어 보수 · 선형 풀이 상태(BlockSolver 의 _Hschur · _DInvSchur · 선형 풀이기)
#[derive(Default)]
pub struct Schur {
    /// (블록 열, 블록 행) → 6×6 열 우선, 행 ≤ 열 (fillCCS 순서 = 키 순서)
    pub h: BTreeMap<(usize, usize), [f64; 36]>,
    pub dinv: Vec<[f64; 9]>,
    pub ldlt: Option<Ldlt>,
    pub bschur: Vec<f64>,
}

impl<'a> Sys<'a> {
    /// buildStructure 의 슈어 패턴: Hpp 대각 + 같은 점을 보는 자세 쌍
    pub fn schur_pattern(&self) -> Schur {
        let mut h = BTreeMap::new();
        for i in 0..self.poses.len() {
            h.insert((i, i), [0.0; 36]);
        }
        for c in &self.pl_cols {
            for (a, &i1) in c.iter().enumerate() {
                for &i2 in &c[a..] {
                    h.insert((i2, i1), [0.0; 36]);
                }
            }
        }
        Schur { h, dinv: vec![[0.0; 9]; self.points.len()], ldlt: None, bschur: vec![] }
    }

    /// 최대 대각 절댓값(computeLambdaInit, indexMapping 순서: 자세 다음 점)
    pub fn max_diag(&self) -> f64 {
        let mut m = 0.0f64;
        for h in &self.hpp {
            for j in 0..6 {
                m = h[j * 7].abs().max(m);
            }
        }
        for h in &self.hll {
            for j in 0..3 {
                m = h[j * 4].abs().max(m);
            }
        }
        m
    }

    pub fn b_vec(&self) -> Vec<f64> {
        let mut b = Vec::with_capacity(self.poses.len() * 6 + self.points.len() * 3);
        for x in &self.bp {
            b.extend_from_slice(x);
        }
        for x in &self.bl {
            b.extend_from_slice(x);
        }
        b
    }

    /// setLambda · BlockSolver::solve · restoreDiagonal. 돌려줌 x(자세 6·np 다음 점 3·nl), 실패면 None
    pub fn solve(&mut self, sc: &mut Schur, lambda: f64) -> Option<Vec<f64>> {
        let np = self.poses.len();
        let nl = self.points.len();
        let bk_p: Vec<[f64; 6]> = self.hpp.iter().map(|h| std::array::from_fn(|j| h[j * 7])).collect();
        let bk_l: Vec<[f64; 3]> = self.hll.iter().map(|h| std::array::from_fn(|j| h[j * 4])).collect();
        for h in self.hpp.iter_mut() {
            for j in 0..6 {
                h[j * 7] += lambda;
            }
        }
        for h in self.hll.iter_mut() {
            for j in 0..3 {
                h[j * 4] += lambda;
            }
        }
        for v in sc.h.values_mut() {
            *v = [0.0; 36];
        }
        for i in 0..np {
            let d = sc.h.get_mut(&(i, i)).unwrap();
            for k in 0..36 {
                d[k] += self.hpp[i][k];
            }
        }
        let mut coef = vec![0.0; np * 6];
        for li in 0..nl {
            let dinv = inv3(&self.hll[li]);
            sc.dinv[li] = dinv;
            let b = self.bl[li];
            let db = mv3f(&dinv, &b);
            let col = &self.pl_cols[li];
            for (a, &i1) in col.iter().enumerate() {
                let bi = &self.hpl[&(i1, li)];
                // BDinv = Bi · Dinv (6×3)
                let mut bd = [0.0; 18];
                for j in 0..3 {
                    for r in 0..6 {
                        bd[j * 6 + r] = fused(bi[12 + r], dinv[j * 3 + 2], fused(bi[6 + r], dinv[j * 3 + 1], bi[r] * dinv[j * 3]));
                    }
                }
                // Bb += Bi · db
                for r in 0..6 {
                    coef[i1 * 6 + r] += fused(bi[12 + r], db[2], fused(bi[6 + r], db[1], bi[r] * db[0]));
                }
                for &i2 in &col[a..] {
                    let bj = &self.hpl[&(i2, li)];
                    let t = sc.h.get_mut(&(i2, i1)).unwrap();
                    for j in 0..6 {
                        for r in 0..6 {
                            let v = fused(bd[12 + r], bj[12 + j], fused(bd[6 + r], bj[6 + j], bd[r] * bj[j]));
                            t[j * 6 + r] -= v;
                        }
                    }
                }
            }
        }
        let mut bs = vec![0.0; np * 6];
        for i in 0..np {
            for r in 0..6 {
                bs[i * 6 + r] = self.bp[i][r];
            }
        }
        for i in 0..np * 6 {
            bs[i] -= coef[i];
        }
        // 위 삼각 CCS(블록 열 · 블록 안 열 · 블록 행 · 블록 안 행 순)
        let first = sc.ldlt.is_none();
        let mut vals = Vec::new();
        let (mut ap, mut ai) = (vec![0usize], Vec::new());
        let keys: Vec<(usize, usize)> = sc.h.keys().copied().collect();
        let mut k0 = 0;
        for c in 0..np {
            let mut k1 = k0;
            while k1 < keys.len() && keys[k1].0 == c {
                k1 += 1;
            }
            for cc in 0..6 {
                for &(_, r) in &keys[k0..k1] {
                    let blk = &sc.h[&(c, r)];
                    let n = if r == c { cc + 1 } else { 6 };
                    for rr in 0..n {
                        vals.push(blk[cc * 6 + rr]);
                        if first {
                            ai.push(r * 6 + rr);
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
            sc.ldlt = Some(Ldlt::analyze(np * 6, ap, ai));
        }
        let ld = sc.ldlt.as_mut().unwrap();
        let ok = ld.factorize(&vals);
        sc.bschur = bs.clone();
        for (h, b) in self.hpp.iter_mut().zip(&bk_p) {
            for j in 0..6 {
                h[j * 7] = b[j];
            }
        }
        for (h, b) in self.hll.iter_mut().zip(&bk_l) {
            for j in 0..3 {
                h[j * 4] = b[j];
            }
        }
        if !ok {
            return None;
        }
        let xp = ld.solve(&bs);
        let mut x = xp.clone();
        x.resize(np * 6 + nl * 3, 0.0);
        let cp: Vec<f64> = xp.iter().map(|v| -v).collect();
        for li in 0..nl {
            let mut cl = self.bl[li];
            for &r in &self.pl_cols[li] {
                let b = &self.hpl[&(r, li)];
                for j in 0..3 {
                    cl[j] += dot6(&b[j * 6..j * 6 + 6], &cp[r * 6..r * 6 + 6]);
                }
            }
            let xl = mv3f(&sc.dinv[li], &cl);
            for j in 0..3 {
                x[np * 6 + li * 3 + j] = 0.0 + xl[j];
            }
        }
        Some(x)
    }

    /// SparseOptimizer::update(x): 자세는 exp(δ)·T, 점은 += δ
    pub fn update(&mut self, x: &[f64]) {
        let np = self.poses.len();
        for (i, &v) in self.poses.iter().enumerate() {
            if let Est::Pose { q, t } = &mut self.est[v] {
                let d: [f64; 6] = std::array::from_fn(|j| x[i * 6 + j]);
                se3::oplus_se3(q, t, &d);
            }
        }
        for (i, &v) in self.points.iter().enumerate() {
            if let Est::Point(p) = &mut self.est[v] {
                for j in 0..3 {
                    p[j] += x[np * 6 + i * 3 + j];
                }
            }
        }
    }
}

/// 3×3 · 3 (Eigen 고정 크기 곱: 0·1 행 = 묶음 vfmadd 차례, 2 행 = m0 + (m1 + m2))
#[inline]
pub fn mv3f(a: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        fused(a[6], v[2], fused(a[3], v[1], a[0] * v[0])),
        fused(a[7], v[2], fused(a[4], v[1], a[1] * v[0])),
        a[2] * v[0] + (a[5] * v[1] + a[8] * v[2]),
    ]
}
/// atxpy 의 6 길이 내적(SSE 2 묶음 세 개): ((m2+m4)+m0) + ((m3+m5)+m1)
#[inline]
fn dot6(a: &[f64], b: &[f64]) -> f64 {
    let m = |i: usize| a[i] * b[i];
    ((m(2) + m(4)) + m(0)) + ((m(3) + m(5)) + m(1))
}

/// 원본 LBA 의 optimizer.initializeOptimization(); optimizer.optimize(iters)
/// (OptimizationAlgorithmLevenberg::solve, Raul 의 멈춤 조건 · 시도 10 번 포함)
pub fn optimize(p: &Problem, iters: usize) -> Outcome {
    let mut s = Sys::new(p);
    let mut sc = s.schur_pattern();
    let tau = 1e-5;
    let mut lambda = 0.0;
    let mut ni = 2.0;
    let mut nbad = 0;
    let mut it = 0;
    for i in 0..iters {
        s.compute_errors();
        let mut cur = s.robust_chi2();
        let ini = cur;
        s.build_system();
        if i == 0 {
            lambda = if p.lambda_init > 0.0 { p.lambda_init } else { tau * s.max_diag() };
            ni = 2.0;
            nbad = 0;
        }
        let mut rho;
        let mut q = 0;
        loop {
            let saved = s.est.clone();
            let x = s.solve(&mut sc, lambda);
            let xv = x.clone().unwrap_or_else(|| vec![0.0; s.poses.len() * 6 + s.points.len() * 3]);
            s.update(&xv);
            s.compute_errors();
            let mut temp = s.robust_chi2();
            if x.is_none() {
                temp = f64::MAX;
            }
            rho = cur - temp;
            let b = s.b_vec();
            let mut scale = 0.0;
            for j in 0..xv.len() {
                scale += xv[j] * (lambda * xv[j] + b[j]);
            }
            scale += 1e-3;
            rho /= scale;
            if rho > 0.0 && temp.is_finite() {
                let alpha = 1.0 - (2.0 * rho - 1.0).powf(3.0);
                let alpha = alpha.min(2.0 / 3.0);
                let f = (1.0f64 / 3.0).max(alpha);
                lambda *= f;
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
    let chi2 = (0..p.edges.len()).map(|k| if s.active[k] { s.chi2(k) } else { 0.0 }).collect();
    Outcome { est: s.est, chi2, iterations: it }
}
