//! GTSAM 4.2 GaussianFactorGraph::optimize(ordering, EliminatePreferCholesky) 를 덧셈 순서까지 그대로 옮긴 선형 풀이.
//! 원본 경로: EliminationTree(EliminationTree-inst.h) → JunctionTree(JunctionTree-inst.h, mergeChildren) →
//!   클리크마다 EliminateCholesky(Scatter → HessianFactor 합치기 → choleskyPartial → split) → GaussianBayesTree::optimize(OptimizeClique).
//! 밀집 연산은 Eigen 3.4.0(gtsam_bg 빌드: x86-64 SSE2, FMA 없음, -O3) 이 실제로 도는 커널의 합 순서를 따른다:
//!   - JacobianFactor::updateHessian: 비대각 블록 = 지연 계수 곱(redux, 두 갈래 합), 대각 블록 = rankUpdate(GEBP, 차례 합)
//!   - LLT<Upper>: llt_inplace<Lower>::unblocked / blocked(크기 32 이상, 블록 8) — squaredNorm(redux), 행 우선 GEMV, 행 하나면 dot
//!   - TRSM(triangular_solve_matrix OnTheLeft·Lower·RowMajor 삼각): 패널 4(mr = nr = 4), 패널 밖은 GEBP(alpha = -1)
//!   - SYRK(general_matrix_matrix_triangular_product Upper): tribb_kernel(블록 4) + GEBP, mc = ⌊mc/4⌋·4
//!   - GEBP 누적: 4 행 묶음 = 차례 합, 끝 2 행 묶음 × 4 열 패널 = k 를 8 단위로 짝·홀 따로(C·D) 더해 합친 뒤 나머지 차례,
//!     나머지 1 행 = 차례 합. 결과 = res + alpha·acc.
//!   - 뒤로 대입: d − S·xS(열 우선 GEMV, 열 128 이상이면 16 열씩), 위 삼각 풀이(패널 8, 아래에서 위로)
//! 블록 크기(kc, mc)는 Eigen 의 computeProductBlockingSizes 를 그대로 옮기고, 캐시 크기는 원본이 도는 PC(Ryzen 9 9950X,
//! Eigen manage_caching_sizes: L1 49152, L2 1048576, L3 67108864)의 값을 쓴다. 이 크기들에서는 우리 그래프의 kc 가 깊이 전체라
//! 쪼개지지 않는다(그래도 쪼개지는 경우까지 원본 고리를 그대로 둠).
//! 메모리 정렬: 원본에서 정렬에 따라 갈리는 곳은 redux(squaredNorm·dot) 뿐인데, 그 식이 직접 접근 없는 식(cwiseAbs2·곱)이라
//! first_default_aligned = 0 으로 늘 같다 → 정렬과 무관하게 결정적. 그 밖의 커널은 비정렬 적재라 값에 영향 없음.
use super::geom::M6;

const L1: i64 = 49152;
const L2: i64 = 1048576;
const L3: i64 = 67108864;

/// 선형 인자(JacobianFactor, 잡음 모델 없음 = 이미 화이트닝): 키(변수 번호, 인자 안 순서), 키마다 6×6 A, b
pub struct LinFactor {
    pub keys: Vec<usize>,
    pub a: Vec<M6>,
    pub b: [f64; 6],
}

// ---------------- Eigen 커널 ----------------

/// computeProductBlockingSizes<double,double,KcFactor>(k, m, n, 1) (evaluateProductBlockingSizesHeuristic)
fn blocking(kcf: i64, mut k: i64, mut m: i64, mut n: i64) -> (i64, i64, i64) {
    if k.max(m.max(n)) < 48 {
        return (k, m, n);
    }
    let k_sub = 4 * 4 * 8;
    let k_div = kcf * (4 * 8 + 4 * 8);
    let max_kc = (((L1 - k_sub) / k_div) & !7).max(1);
    let old_k = k;
    if k > max_kc {
        k = if k % max_kc == 0 { max_kc } else { max_kc - 8 * ((max_kc - 1 - (k % max_kc)) / (8 * (k / max_kc + 1))) };
    }
    let actual_l2 = 1572864;
    let lhs_bytes = m * k * 8;
    let remaining_l1 = L1 - k_sub - lhs_bytes;
    let max_nc = if remaining_l1 >= 4 * 8 * k { remaining_l1 / (k * 8) } else { (3 * actual_l2) / (2 * 2 * max_kc * 8) };
    let nc = (actual_l2 / (2 * k * 8)).min(max_nc) & !3;
    if n > nc {
        n = if n % nc == 0 { nc } else { nc - 4 * ((nc - (n % nc)) / (4 * (n / nc + 1))) };
    } else if old_k == k {
        let problem_size = k * n * 8;
        let mut actual_lm = actual_l2;
        let mut max_mc = m;
        if problem_size <= 1024 {
            actual_lm = L1;
        } else if L3 != 0 && problem_size <= 32768 {
            actual_lm = L2;
            max_mc = max_mc.min(576);
        }
        let mut mc = (actual_lm / (3 * k * 8)).min(max_mc);
        if mc > 4 {
            mc -= mc % 4;
        } else if mc == 0 {
            return (k, m, n);
        }
        m = if m % mc == 0 { mc } else { mc - 4 * ((mc - (m % mc)) / (4 * (m / mc + 1))) };
    }
    (k, m, n)
}

/// GEBP 누적(acc) 한 칸: 이 gebp 호출의 행 수 m·열 수 n 안에서 (행 r, 열 c), 깊이 depth, p(k) = 곱
#[inline]
fn gebp_acc(r: usize, m: usize, c: usize, n: usize, depth: usize, p: impl Fn(usize) -> f64) -> f64 {
    let pm2 = (m / 4) * 4;
    let pm1 = pm2 + ((m - pm2) / 2) * 2;
    let pc4 = (n / 4) * 4;
    if r >= pm2 && r < pm1 && c < pc4 {
        let pk = (depth / 8) * 8;
        let (mut cc, mut dd) = (0.0f64, 0.0f64);
        let mut k = 0;
        while k < pk {
            cc = p(k) + cc;
            dd = p(k + 1) + dd;
            k += 2;
        }
        let mut acc = cc + dd;
        for k in pk..depth {
            acc = p(k) + acc;
        }
        acc
    } else {
        let mut acc = 0.0f64;
        for k in 0..depth {
            acc = p(k) + acc;
        }
        acc
    }
}

/// redux 합(LinearVectorizedTraversal, 꾸러미 2, 정렬 시작 0): 두 갈래
#[inline]
fn redux2(n: usize, p: impl Fn(usize) -> f64) -> f64 {
    if n < 2 {
        let mut res = p(0);
        for i in 1..n {
            res = res + p(i);
        }
        return res;
    }
    let aligned = (n / 2) * 2;
    let aligned2 = (n / 4) * 4;
    let (mut r0a, mut r0b) = (p(0), p(1));
    if aligned > 2 {
        let (mut r1a, mut r1b) = (p(2), p(3));
        let mut idx = 4;
        while idx < aligned2 {
            r0a = r0a + p(idx);
            r0b = r0b + p(idx + 1);
            r1a = r1a + p(idx + 2);
            r1b = r1b + p(idx + 3);
            idx += 4;
        }
        r0a = r0a + r1a;
        r0b = r0b + r1b;
        if aligned > aligned2 {
            r0a = r0a + p(aligned2);
            r0b = r0b + p(aligned2 + 1);
        }
    }
    let mut res = r0a + r0b;
    for i in aligned..n {
        res = res + p(i);
    }
    res
}

/// 열 우선 행렬 보기: (r, c) = d[off + c·ld + r]
#[derive(Clone, Copy)]
struct V {
    off: usize,
    ld: usize,
}
impl V {
    #[inline]
    fn at(&self, r: usize, c: usize) -> usize {
        self.off + c * self.ld + r
    }
    fn sub(&self, r: usize, c: usize) -> V {
        V { off: self.at(r, c), ld: self.ld }
    }
}

/// triangular_solve_matrix<OnTheLeft, Lower, TriStorage = RowMajor, Other = ColMajor>:
/// L·X = B, L(i,j) = t[tri.at(j, i)] (= 위 삼각 R 의 전치), B 는 size × cols (o 안에서 제자리)
fn trsm(t: &[f64], tri: V, o: &mut [f64], oth: V, size: usize, cols: usize) {
    if size == 0 || cols == 0 {
        return;
    }
    let l = |i: usize, j: usize, t: &[f64]| t[tri.at(j, i)];
    let (kc, mc, _) = blocking(4, size as i64, size as i64, cols as i64);
    let (kc, mc) = (kc as usize, (mc as usize).min(size));
    let mut k2 = 0;
    while k2 < size {
        let actual_kc = (size - k2).min(kc);
        let mut k1 = 0;
        while k1 < actual_kc {
            let pw = (actual_kc - k1).min(4);
            for k in 0..pw {
                let i = k2 + k1 + k;
                let s = k2 + k1;
                let a = 1.0 / l(i, i, t);
                for j in 0..cols {
                    let mut b = 0.0f64;
                    for i3 in 0..k {
                        b = b + l(i, s + i3, t) * o[oth.at(s + i3, j)];
                    }
                    let x = &mut o[oth.at(i, j)];
                    *x = (*x - b) * a;
                }
            }
            let length_target = actual_kc - k1 - pw;
            let start_block = k2 + k1;
            if length_target > 0 {
                let start_target = k2 + k1 + pw;
                for j in 0..cols {
                    for r in 0..length_target {
                        let acc = gebp_acc(r, length_target, j, cols, pw, |kk| l(start_target + r, start_block + kk, t) * o[oth.at(start_block + kk, j)]);
                        let x = &mut o[oth.at(start_target + r, j)];
                        *x = *x + (-1.0) * acc;
                    }
                }
            }
            k1 += 4;
        }
        // 이 kc 블록 아래 행들
        let start = k2 + kc;
        let mut i2 = start;
        while i2 < size {
            let amc = mc.min(size - i2);
            for j in 0..cols {
                for r in 0..amc {
                    let acc = gebp_acc(r, amc, j, cols, actual_kc, |kk| l(i2 + r, k2 + kk, t) * o[oth.at(k2 + kk, j)]);
                    let x = &mut o[oth.at(i2 + r, j)];
                    *x = *x + (-1.0) * acc;
                }
            }
            i2 += mc;
        }
        k2 += kc;
    }
}

/// C.selfadjointView<Upper>().rankUpdate(Bᵀ, alpha): C(r,c) (r ≤ c) += alpha · Σ_k B(k,r)·B(k,c).
/// d 하나에 C·B 가 같이 있음(cv: C 보기, bv: B 보기, B 는 depth × size)
fn syrk_upper(d: &mut [f64], cv: V, bv: V, size: usize, depth: usize, alpha: f64) {
    if size == 0 || depth == 0 {
        return;
    }
    let (kc, mc0, _) = blocking(1, depth as i64, size as i64, size as i64);
    let kc = kc as usize;
    let mut mc = (size as i64).min(mc0) as usize;
    if mc > 4 {
        mc = (mc / 4) * 4;
    }
    let mut k2 = 0;
    while k2 < depth {
        let akc = (k2 + kc).min(depth) - k2;
        let mut i2 = 0;
        while i2 < size {
            let amc = (i2 + mc).min(size) - i2;
            // tribb_kernel(블록 4) 와 오른쪽 gebp
            for r in i2..i2 + amc {
                for c in r..size {
                    let (rr, m, cc, n);
                    if c >= i2 + amc {
                        rr = r - i2;
                        m = amc;
                        cc = c - (i2 + amc);
                        n = size - (i2 + amc);
                    } else {
                        let jb = ((c - i2) / 4) * 4;
                        let abs = 4.min(amc - jb);
                        if r - i2 < jb {
                            rr = r - i2;
                            m = jb;
                        } else {
                            rr = r - i2 - jb;
                            m = abs;
                        }
                        cc = c - i2 - jb;
                        n = abs;
                    }
                    let acc = gebp_acc(rr, m, cc, n, akc, |k| d[bv.at(k2 + k, r)] * d[bv.at(k2 + k, c)]);
                    let x = &mut d[cv.at(r, c)];
                    *x = *x + alpha * acc;
                }
            }
            i2 += mc;
        }
        k2 += kc;
    }
}

/// llt_inplace<Lower>::unblocked 를 전치 보기로: 저장(열 우선) 위 삼각 S 를 R(SᵀS... 가 아니라 RᵀR = A)로 제자리 분해.
/// 실패하면 실패한 열 번호
fn llt_unblocked(d: &mut [f64], s: V, size: usize) -> Result<(), usize> {
    for k in 0..size {
        let rs = size - k - 1;
        let mut x = d[s.at(k, k)];
        if k > 0 {
            // A10.squaredNorm() = S(0..k, k)
            let nrm = redux2(k, |i| {
                let v = d[s.at(i, k)];
                v * v
            });
            x = x - nrm;
        }
        if x <= 0.0 {
            return Err(k);
        }
        let x = x.sqrt();
        d[s.at(k, k)] = x;
        if k > 0 && rs > 0 {
            // A21 -= A20 · A10ᵀ: A21(r) = S(k, k+1+r), A20(r, j) = S(j, k+1+r), A10(j) = S(j, k)
            if rs == 1 {
                let dot = redux2(k, |j| d[s.at(j, k + 1)] * d[s.at(j, k)]);
                let y = &mut d[s.at(k, k + 1)];
                *y = *y + (-1.0) * dot;
            } else {
                for r in 0..rs {
                    let col = k + 1 + r;
                    let (mut c0, mut c1) = (0.0f64, 0.0f64);
                    let mut j = 0;
                    while j + 2 <= k {
                        c0 = d[s.at(j, col)] * d[s.at(j, k)] + c0;
                        c1 = d[s.at(j + 1, col)] * d[s.at(j + 1, k)] + c1;
                        j += 2;
                    }
                    let mut cc = c0 + c1;
                    while j < k {
                        cc = cc + d[s.at(j, col)] * d[s.at(j, k)];
                        j += 1;
                    }
                    let y = &mut d[s.at(k, col)];
                    *y = *y + (-1.0) * cc;
                }
            }
        }
        if rs > 0 {
            for r in 0..rs {
                let y = &mut d[s.at(k, k + 1 + r)];
                *y = *y / x;
            }
        }
    }
    Ok(())
}

/// LLT<MatrixXd, Upper> 의 분해(Transpose 보기에서 llt_inplace<Lower>::blocked)
fn llt(d: &mut [f64], s: V, size: usize) -> Result<(), usize> {
    if size < 32 {
        return llt_unblocked(d, s, size);
    }
    let mut bsz = size / 8;
    bsz = (bsz / 16) * 16;
    bsz = bsz.max(8).min(128);
    let mut k = 0;
    while k < size {
        let bs = bsz.min(size - k);
        let rs = size - k - bs;
        llt_unblocked(d, s.sub(k, k), bs).map_err(|e| k + e)?;
        if rs > 0 {
            // A21 ← A21 · A11⁻ᴴ  (저장에서는 R11ᵀ · X = S(k.., k+bs..))
            let (t, o) = (s.sub(k, k), s.sub(k, k + bs));
            let copy = d.to_vec();
            trsm(&copy, t, d, o, bs, rs);
            // 위 trsm 은 삼각 부분을 읽기만 하므로 사본으로 읽어도 같은 값(삼각과 대상이 겹치지 않음)
            syrk_upper(d, s.sub(k + bs, k + bs), s.sub(k, k + bs), rs, bs, -1.0);
        }
        k += bs;
    }
    Ok(())
}

// ---------------- 소거 나무 · 접합 나무 ----------------

struct JNode {
    frontal: Vec<usize>,
    factors: Vec<usize>,
    children: Vec<usize>,
}

/// 헤시안 인자(choleskyPartial 뒤 남은 부분): 키, 행렬(열 우선 n×n), 남은 블록 시작
struct Hess {
    keys: Vec<usize>,
    m: Vec<f64>,
    n: usize,
    start: usize,
}

struct Cond {
    frontal: Vec<usize>,
    parents: Vec<usize>,
    /// [R S d] (열 우선, 행 = 6·프런탈)
    rsd: Vec<f64>,
    rows: usize,
    cols: usize,
}

/// GaussianFactorGraph::optimize(ordering, EliminatePreferCholesky). 실패(IndeterminantLinearSystem)면 None
pub fn solve(nvars: usize, fs: &[LinFactor], ord: &[usize]) -> Option<Vec<[f64; 6]>> {
    let n = ord.len();
    let m = fs.len();
    // VariableIndex
    let mut structure: Vec<Vec<usize>> = vec![Vec::new(); nvars];
    for (i, f) in fs.iter().enumerate() {
        for &k in &f.keys {
            structure[k].push(i);
        }
    }
    // EliminationTree
    const NONE: usize = usize::MAX;
    let mut parents = vec![NONE; n];
    let mut prev_col = vec![NONE; m];
    let mut e_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut e_factors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        for &i in &structure[ord[j]] {
            if prev_col[i] != NONE {
                let mut r = prev_col[i];
                while parents[r] != NONE {
                    r = parents[r];
                }
                if r != j {
                    parents[r] = j;
                    e_children[j].push(r);
                }
            } else {
                e_factors[j].push(i);
            }
            prev_col[i] = j;
        }
    }
    let e_roots: Vec<usize> = (0..n).filter(|&j| parents[j] == NONE).collect();

    // JunctionTree: 깊이 우선(자식 순서대로), 끝날 때 기호 소거 → 자식 합치기
    let mut jt: Vec<JNode> = Vec::with_capacity(n + 1);
    jt.push(JNode { frontal: vec![], factors: vec![], children: vec![] }); // 0 = 가짜 뿌리
    // (기호 분리 키 집합, 조건부 부모 수) 를 돌려줌
    fn build(e: usize, parent: usize, ord: &[usize], e_children: &[Vec<usize>], e_factors: &[Vec<usize>], fs: &[LinFactor], jt: &mut Vec<JNode>) -> Vec<usize> {
        let me = jt.len();
        jt.push(JNode { frontal: vec![ord[e]], factors: e_factors[e].clone(), children: vec![] });
        jt[parent].children.push(me);
        let mut child_seps: Vec<Vec<usize>> = Vec::with_capacity(e_children[e].len());
        for &c in &e_children[e] {
            child_seps.push(build(c, me, ord, e_children, e_factors, fs, jt));
        }
        // EliminateSymbolic(인자들 + 자식 분리 인자들, [key])
        let key = ord[e];
        let mut all: Vec<usize> = Vec::new();
        for &f in &e_factors[e] {
            all.extend_from_slice(&fs[f].keys);
        }
        for s in &child_seps {
            all.extend_from_slice(s);
        }
        all.sort_unstable();
        all.dedup();
        let sep: Vec<usize> = all.into_iter().filter(|&k| k != key).collect();
        let my_nr_parents = sep.len();
        // mergeChildren
        let kids = jt[me].children.clone();
        let mut merge = vec![false; kids.len()];
        let mut my_nr_frontals = 1;
        for (i, &c) in kids.iter().enumerate() {
            if my_nr_parents + my_nr_frontals == child_seps[i].len() {
                my_nr_frontals += jt[c].frontal.len();
                merge[i] = true;
            }
        }
        let mut new_children = Vec::new();
        for (i, &c) in kids.iter().enumerate() {
            if merge[i] {
                let cf: Vec<usize> = jt[c].frontal.iter().rev().copied().collect();
                jt[me].frontal.extend(cf);
                let cfac = jt[c].factors.clone();
                jt[me].factors.extend(cfac);
                let cch = jt[c].children.clone();
                new_children.extend(cch);
            } else {
                new_children.push(c);
            }
        }
        jt[me].children = new_children;
        jt[me].frontal.reverse();
        sep
    }
    for &r in &e_roots {
        build(r, 0, ord, &e_children, &e_factors, fs, &mut jt);
    }

    // 소거(깊이 우선, 자식 결과를 자식 순서대로 모음)
    let mut conds: Vec<Option<Cond>> = (0..jt.len()).map(|_| None).collect();
    let mut cparent = vec![0usize; jt.len()];
    fn elim(j: usize, jt: &[JNode], fs: &[LinFactor], conds: &mut Vec<Option<Cond>>, cparent: &mut Vec<usize>) -> Result<Option<Hess>, ()> {
        let mut child_factors: Vec<Option<Hess>> = Vec::with_capacity(jt[j].children.len());
        for &c in &jt[j].children {
            cparent[c] = j;
            child_factors.push(elim(c, jt, fs, conds, cparent)?);
        }
        let node = &jt[j];
        // Scatter: 프런탈 순서대로, 나머지 키는 정렬
        let mut keys: Vec<usize> = node.frontal.clone();
        let mut rest: Vec<usize> = Vec::new();
        for &f in &node.factors {
            for &k in &fs[f].keys {
                if !keys.contains(&k) && !rest.contains(&k) {
                    rest.push(k);
                }
            }
        }
        for h in child_factors.iter().flatten() {
            for &k in &h.keys {
                if !keys.contains(&k) && !rest.contains(&k) {
                    rest.push(k);
                }
            }
        }
        rest.sort_unstable();
        keys.extend(rest);
        let nk = keys.len();
        let nn = 6 * nk + 1;
        let slot = |k: usize| keys.iter().position(|&x| x == k).unwrap();
        let mut mm = vec![0.0f64; nn * nn];
        let at = |r: usize, c: usize| c * nn + r;
        // 원래 야코비안 인자들
        for &f in &node.factors {
            let lf = &fs[f];
            let nb = lf.keys.len();
            let col = |blk: usize, k: usize, c: usize| if blk < nb { lf.a[blk][k][c] } else { lf.b[k] };
            let dim = |blk: usize| if blk < nb { 6 } else { 1 };
            let off = |blk: usize| if blk < nb { 6 * slot(lf.keys[blk]) } else { 6 * nk };
            for jb in 0..=nb {
                for ib in 0..jb {
                    let (oi, oj) = (off(ib), off(jb));
                    for c in 0..dim(jb) {
                        for r in 0..dim(ib) {
                            let v = redux2(6, |k| col(ib, k, r) * col(jb, k, c));
                            if oi < oj {
                                mm[at(oi + r, oj + c)] += v;
                            } else {
                                mm[at(oj + c, oi + r)] += v;
                            }
                        }
                    }
                }
                let oj = off(jb);
                for c in 0..dim(jb) {
                    for r in 0..=c {
                        let mut acc = 0.0f64;
                        for k in 0..6 {
                            acc = col(jb, k, r) * col(jb, k, c) + acc;
                        }
                        mm[at(oj + r, oj + c)] = mm[at(oj + r, oj + c)] + acc;
                    }
                }
            }
        }
        // 자식 헤시안
        for h in child_factors.iter().flatten() {
            let hn = h.keys.len();
            let hat = |r: usize, c: usize| h.m[c * h.n + h.start * (h.n + 1) + r];
            let hoff = |b: usize| 6 * b;
            let hdim = |b: usize| if b < hn { 6 } else { 1 };
            let poff = |b: usize| if b < hn { 6 * slot(h.keys[b]) } else { 6 * nk };
            for jb in 0..=hn {
                for ib in 0..=jb {
                    let (pi, pj) = (poff(ib), poff(jb));
                    if ib == jb {
                        for c in 0..hdim(jb) {
                            for r in 0..=c {
                                mm[at(pj + r, pj + c)] += hat(hoff(jb) + r, hoff(jb) + c);
                            }
                        }
                    } else {
                        for c in 0..hdim(jb) {
                            for r in 0..hdim(ib) {
                                let v = hat(hoff(ib) + r, hoff(jb) + c);
                                if pi < pj {
                                    mm[at(pi + r, pj + c)] += v;
                                } else {
                                    mm[at(pj + c, pi + r)] += v;
                                }
                            }
                        }
                    }
                }
            }
        }
        // choleskyPartial
        let nf = node.frontal.len();
        let nfd = 6 * nf;
        let mut lc = vec![0.0f64; nfd * nfd];
        for c in 0..nfd {
            for r in 0..=c {
                lc[c * nfd + r] = mm[at(r, c)];
            }
        }
        llt(&mut lc, V { off: 0, ld: nfd }, nfd).map_err(|_| ())?;
        for c in 0..nfd {
            for r in 0..=c {
                mm[at(r, c)] = lc[c * nfd + r];
            }
        }
        let big = V { off: 0, ld: nn };
        if nfd < nn {
            let copy = mm.clone();
            trsm(&copy, big, &mut mm, big.sub(0, nfd), nfd, nn - nfd);
            syrk_upper(&mut mm, big.sub(nfd, nfd), big.sub(0, nfd), nn - nfd, nfd, -1.0);
        }
        let (e2, e1) = (frexp_exp(mm[at(nfd - 2, nfd - 2)]), frexp_exp(mm[at(nfd - 1, nfd - 1)]));
        if !(e2 - e1 < 12) {
            return Err(());
        }
        // split: [R S d] = 위 nfd 행, 아래 삼각 0
        let mut rsd = vec![0.0f64; nfd * nn];
        for c in 0..nn {
            for r in 0..nfd.min(c + 1) {
                rsd[c * nfd + r] = mm[at(r, c)];
            }
        }
        conds[j] = Some(Cond { frontal: node.frontal.clone(), parents: keys[nf..].to_vec(), rsd, rows: nfd, cols: nn });
        if nk > nf {
            Ok(Some(Hess { keys: keys[nf..].to_vec(), m: mm, n: nn, start: nfd }))
        } else {
            Ok(None)
        }
    }
    let roots = jt[0].children.clone();
    for &r in &roots {
        if elim(r, &jt, fs, &mut conds, &mut cparent).is_err() {
            return None;
        }
    }

    // 뒤로 대입(위에서 아래로)
    let mut x: Vec<[f64; 6]> = vec![[0.0; 6]; nvars];
    fn backsub(j: usize, jt: &[JNode], conds: &[Option<Cond>], x: &mut Vec<[f64; 6]>) -> bool {
        let c = conds[j].as_ref().unwrap();
        let (rows, cols) = (c.rows, c.cols);
        let at = |r: usize, cc: usize| cc * rows + r;
        let ns = cols - 1 - rows;
        let xs: Vec<f64> = c.parents.iter().flat_map(|&p| x[p]).collect();
        // rhs = d - S·xS (열 우선 GEMV, alpha = -1)
        let mut rhs: Vec<f64> = (0..rows).map(|r| c.rsd[at(r, cols - 1)]).collect();
        let block_cols = if ns < 128 { ns.max(1) } else if rows * 8 < 32000 { 16 } else { 4 };
        let mut j2 = 0;
        while j2 < ns {
            let jend = (j2 + block_cols).min(ns);
            for r in 0..rows {
                let mut acc = 0.0f64;
                for jj in j2..jend {
                    acc = c.rsd[at(r, rows + jj)] * xs[jj] + acc;
                }
                rhs[r] = acc * (-1.0) + rhs[r];
            }
            j2 += block_cols;
        }
        // R x = rhs (위 삼각, 패널 8, 아래에서 위로)
        let size = rows;
        let mut pi = size;
        while pi > 0 {
            let apw = pi.min(8);
            let start_block = pi - apw;
            for k in 0..apw {
                let i = pi - k - 1;
                if rhs[i] != 0.0 {
                    rhs[i] /= c.rsd[at(i, i)];
                    let r = apw - k - 1;
                    let s = i - r;
                    for t in s..s + r {
                        rhs[t] -= rhs[i] * c.rsd[at(t, i)];
                    }
                }
            }
            let r = start_block;
            if r > 0 {
                for row in 0..r {
                    let mut acc = 0.0f64;
                    for jj in start_block..start_block + apw {
                        acc = c.rsd[at(row, jj)] * rhs[jj] + acc;
                    }
                    rhs[row] = acc * (-1.0) + rhs[row];
                }
            }
            pi -= apw;
        }
        if rhs.iter().any(|v| v.is_nan()) {
            return false;
        }
        for (q, &k) in c.frontal.iter().enumerate() {
            x[k] = std::array::from_fn(|i| rhs[6 * q + i]);
        }
        for &ch in &jt[j].children {
            if !backsub(ch, jt, conds, x) {
                return false;
            }
        }
        true
    }
    for &r in &roots {
        if !backsub(r, &jt, &conds, &mut x) {
            return None;
        }
    }
    let _ = cparent;
    Some(x)
}

/// frexp 의 지수
fn frexp_exp(v: f64) -> i32 {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        // 비정규수
        let m = bits & ((1u64 << 52) - 1);
        return -1022 - (m.leading_zeros() as i32 - 12) ;
    }
    e - 1022
}
