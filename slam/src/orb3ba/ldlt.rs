//! g2o LinearSolverEigen = Eigen 3.4 SimplicialLDLT<SparseMatrix<double>, Upper, AMDOrdering<int>> 를 그대로.
//! - 입력: 슈어 행렬의 위 삼각(블록 열 순서 · 블록 안 열 · 행 순서 = SparseBlockMatrix::fillCCS(upper))
//! - analyzePattern(첫 풀이 한 번): C = 대칭 전체 패턴(대각 포함, 열 안 행 오름차순) → minimum_degree_ordering(Amd.h, 정수)
//!   → P = Pinv⁻¹, ap = P·A·Pᵀ 의 위 삼각(permute_symm_to_symm 의 열 안 순서 그대로) → 소거 나무 · 열 개수
//! - factorize(매번): 위로 보는 LDLT. y[Li] −= Lx·yi, d −= l·yi (소스 순서, 축약 없음)
//! - solve: x = P·b → L 전진(tmp ≠ 0 일 때 x −= L·tmp) → x = (1/d)·x → Lᵀ 후진(tmp −= L·x) → Pinv
#[derive(Default, Clone)]
pub struct Ldlt {
    n: usize,
    /// 입력 행렬 패턴(위 삼각, 열 우선)
    ap_src: (Vec<usize>, Vec<usize>),
    /// P · Pinv (Eigen indices)
    p: Vec<usize>,
    pinv: Vec<usize>,
    /// ap 패턴 · 원본 값 자리(입력 nz 번호)
    ap_p: Vec<usize>,
    ap_i: Vec<usize>,
    ap_src_k: Vec<usize>,
    parent: Vec<isize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<f64>,
    d: Vec<f64>,
}

fn amd_flip(i: i64) -> i64 {
    -i - 2
}

fn cs_wclear(mut mark: i64, lemax: i64, w: &mut [i64], n: usize) -> i64 {
    if mark < 2 || (mark + lemax < 0) {
        for k in 0..n {
            if w[k] != 0 {
                w[k] = 1;
            }
        }
        mark = 2;
    }
    mark
}

fn cs_tdfs(j: i64, mut k: i64, head: &mut [i64], next: &[i64], post: &mut [i64], stack: &mut [i64]) -> i64 {
    let mut top: i64 = 0;
    stack[0] = j;
    while top >= 0 {
        let p = stack[top as usize];
        let i = head[p as usize];
        if i == -1 {
            top -= 1;
            post[k as usize] = p;
            k += 1;
        } else {
            head[p as usize] = next[i as usize];
            top += 1;
            stack[top as usize] = i;
        }
    }
    k
}

/// Eigen internal::minimum_degree_ordering (CSparse cs_amd). cp/ci = 대칭 전체 패턴(대각 포함). 돌려줌 perm(= Pinv)
pub fn amd(n: usize, cp_in: &[usize], ci_in: &[usize]) -> Vec<usize> {
    let ni = n as i64;
    let mut dense = std::cmp::max(16i64, (10.0 * (n as f64).sqrt()) as i64);
    dense = std::cmp::min(ni - 2, dense);
    let mut cnz = ci_in.len() as i64;
    let t = cnz + cnz / 5 + 2 * ni;
    let mut cp: Vec<i64> = cp_in.iter().map(|&x| x as i64).collect();
    let mut ci: Vec<i64> = vec![0; t as usize];
    for (k, &x) in ci_in.iter().enumerate() {
        ci[k] = x as i64;
    }
    let m = n + 1;
    let mut len = vec![0i64; m];
    let mut nv = vec![0i64; m];
    let mut next = vec![0i64; m];
    let mut head = vec![0i64; m];
    let mut elen = vec![0i64; m];
    let mut degree = vec![0i64; m];
    let mut w = vec![0i64; m];
    let mut hhead = vec![0i64; m];
    let mut last = vec![0i64; m];
    for k in 0..n {
        len[k] = cp[k + 1] - cp[k];
    }
    len[n] = 0;
    let nzmax = t;
    for i in 0..=n {
        head[i] = -1;
        last[i] = -1;
        next[i] = -1;
        hhead[i] = -1;
        nv[i] = 1;
        w[i] = 1;
        elen[i] = 0;
        degree[i] = len[i];
    }
    let mut mark = cs_wclear(0, 0, &mut w, n);
    let mut nel: i64 = 0;
    for i in 0..n {
        let mut has_diag = false;
        let mut p = cp[i];
        while p < cp[i + 1] {
            if ci[p as usize] == i as i64 {
                has_diag = true;
                break;
            }
            p += 1;
        }
        let d = degree[i];
        if d == 1 && has_diag {
            elen[i] = -2;
            nel += 1;
            cp[i] = -1;
            w[i] = 0;
        } else if d > dense || !has_diag {
            nv[i] = 0;
            elen[i] = -1;
            nel += 1;
            cp[i] = amd_flip(ni);
            nv[n] += 1;
        } else {
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = i as i64;
            }
            next[i] = head[d as usize];
            head[d as usize] = i as i64;
        }
    }
    elen[n] = -2;
    cp[n] = -1;
    w[n] = 0;
    let mut mindeg: i64 = 0;
    let mut lemax: i64 = 0;
    macro_rules! u {
        ($e:expr) => {
            ($e) as usize
        };
    }
    while nel < ni {
        let mut k: i64 = -1;
        while mindeg < ni {
            k = head[u!(mindeg)];
            if k != -1 {
                break;
            }
            mindeg += 1;
        }
        if next[u!(k)] != -1 {
            last[u!(next[u!(k)])] = -1;
        }
        head[u!(mindeg)] = next[u!(k)];
        let elenk = elen[u!(k)];
        let mut nvk = nv[u!(k)];
        nel += nvk;
        if elenk > 0 && cnz + mindeg >= nzmax {
            for j in 0..n {
                let p = cp[j];
                if p >= 0 {
                    cp[j] = ci[u!(p)];
                    ci[u!(p)] = amd_flip(j as i64);
                }
            }
            let mut q: i64 = 0;
            let mut p: i64 = 0;
            while p < cnz {
                let j = amd_flip(ci[u!(p)]);
                p += 1;
                if j >= 0 {
                    ci[u!(q)] = cp[u!(j)];
                    cp[u!(j)] = q;
                    q += 1;
                    for _ in 0..(len[u!(j)] - 1).max(0) {
                        ci[u!(q)] = ci[u!(p)];
                        q += 1;
                        p += 1;
                    }
                }
            }
            cnz = q;
        }
        let mut dk: i64 = 0;
        nv[u!(k)] = -nvk;
        let mut p = cp[u!(k)];
        let pk1 = if elenk == 0 { p } else { cnz };
        let mut pk2 = pk1;
        for k1 in 1..=elenk + 1 {
            let (e, mut pj, ln);
            if k1 > elenk {
                e = k;
                pj = p;
                ln = len[u!(k)] - elenk;
            } else {
                e = ci[u!(p)];
                p += 1;
                pj = cp[u!(e)];
                ln = len[u!(e)];
            }
            for _ in 1..=ln {
                let i = ci[u!(pj)];
                pj += 1;
                let nvi = nv[u!(i)];
                if nvi <= 0 {
                    continue;
                }
                dk += nvi;
                nv[u!(i)] = -nvi;
                ci[u!(pk2)] = i;
                pk2 += 1;
                if next[u!(i)] != -1 {
                    last[u!(next[u!(i)])] = last[u!(i)];
                }
                if last[u!(i)] != -1 {
                    next[u!(last[u!(i)])] = next[u!(i)];
                } else {
                    head[u!(degree[u!(i)])] = next[u!(i)];
                }
            }
            if e != k {
                cp[u!(e)] = amd_flip(k);
                w[u!(e)] = 0;
            }
        }
        if elenk != 0 {
            cnz = pk2;
        }
        degree[u!(k)] = dk;
        cp[u!(k)] = pk1;
        len[u!(k)] = pk2 - pk1;
        elen[u!(k)] = -2;
        mark = cs_wclear(mark, lemax, &mut w, n);
        for pk in pk1..pk2 {
            let i = ci[u!(pk)];
            let eln = elen[u!(i)];
            if eln <= 0 {
                continue;
            }
            let nvi = -nv[u!(i)];
            let wnvi = mark - nvi;
            let mut p = cp[u!(i)];
            while p <= cp[u!(i)] + eln - 1 {
                let e = ci[u!(p)];
                if w[u!(e)] >= mark {
                    w[u!(e)] -= nvi;
                } else if w[u!(e)] != 0 {
                    w[u!(e)] = degree[u!(e)] + wnvi;
                }
                p += 1;
            }
        }
        for pk in pk1..pk2 {
            let i = ci[u!(pk)];
            let p1 = cp[u!(i)];
            let p2 = p1 + elen[u!(i)] - 1;
            let mut pn = p1;
            let mut h: i64 = 0;
            let mut d: i64 = 0;
            let mut p = p1;
            while p <= p2 {
                let e = ci[u!(p)];
                if w[u!(e)] != 0 {
                    let dext = w[u!(e)] - mark;
                    if dext > 0 {
                        d += dext;
                        ci[u!(pn)] = e;
                        pn += 1;
                        h += e;
                    } else {
                        cp[u!(e)] = amd_flip(k);
                        w[u!(e)] = 0;
                    }
                }
                p += 1;
            }
            elen[u!(i)] = pn - p1 + 1;
            let p3 = pn;
            let p4 = p1 + len[u!(i)];
            let mut p = p2 + 1;
            while p < p4 {
                let j = ci[u!(p)];
                p += 1;
                let nvj = nv[u!(j)];
                if nvj <= 0 {
                    continue;
                }
                d += nvj;
                ci[u!(pn)] = j;
                pn += 1;
                h += j;
            }
            if d == 0 {
                cp[u!(i)] = amd_flip(k);
                let nvi = -nv[u!(i)];
                dk -= nvi;
                nvk += nvi;
                nel += nvi;
                nv[u!(i)] = 0;
                elen[u!(i)] = -1;
            } else {
                degree[u!(i)] = degree[u!(i)].min(d);
                ci[u!(pn)] = ci[u!(p3)];
                ci[u!(p3)] = ci[u!(p1)];
                ci[u!(p1)] = k;
                len[u!(i)] = pn - p1 + 1;
                h %= ni;
                next[u!(i)] = hhead[u!(h)];
                hhead[u!(h)] = i;
                last[u!(i)] = h;
            }
        }
        degree[u!(k)] = dk;
        lemax = lemax.max(dk);
        mark = cs_wclear(mark + lemax, lemax, &mut w, n);
        for pk in pk1..pk2 {
            let mut i = ci[u!(pk)];
            if nv[u!(i)] >= 0 {
                continue;
            }
            let h = last[u!(i)];
            i = hhead[u!(h)];
            hhead[u!(h)] = -1;
            while i != -1 && next[u!(i)] != -1 {
                let ln = len[u!(i)];
                let eln = elen[u!(i)];
                let mut p = cp[u!(i)] + 1;
                while p <= cp[u!(i)] + ln - 1 {
                    w[u!(ci[u!(p)])] = mark;
                    p += 1;
                }
                let mut jlast = i;
                let mut j = next[u!(i)];
                while j != -1 {
                    let mut ok = len[u!(j)] == ln && elen[u!(j)] == eln;
                    let mut p = cp[u!(j)] + 1;
                    while ok && p <= cp[u!(j)] + ln - 1 {
                        if w[u!(ci[u!(p)])] != mark {
                            ok = false;
                        }
                        p += 1;
                    }
                    if ok {
                        cp[u!(j)] = amd_flip(i);
                        nv[u!(i)] += nv[u!(j)];
                        nv[u!(j)] = 0;
                        elen[u!(j)] = -1;
                        j = next[u!(j)];
                        next[u!(jlast)] = j;
                    } else {
                        jlast = j;
                        j = next[u!(j)];
                    }
                }
                i = next[u!(i)];
                mark += 1;
            }
        }
        let mut p = pk1;
        for pk in pk1..pk2 {
            let i = ci[u!(pk)];
            let nvi = -nv[u!(i)];
            if nvi <= 0 {
                continue;
            }
            nv[u!(i)] = nvi;
            let mut d = degree[u!(i)] + dk - nvi;
            d = d.min(ni - nel - nvi);
            if head[u!(d)] != -1 {
                last[u!(head[u!(d)])] = i;
            }
            next[u!(i)] = head[u!(d)];
            last[u!(i)] = -1;
            head[u!(d)] = i;
            mindeg = mindeg.min(d);
            degree[u!(i)] = d;
            ci[u!(p)] = i;
            p += 1;
        }
        nv[u!(k)] = nvk;
        len[u!(k)] = p - pk1;
        if len[u!(k)] == 0 {
            cp[u!(k)] = -1;
            w[u!(k)] = 0;
        }
        if elenk != 0 {
            cnz = p;
        }
    }
    for i in 0..n {
        cp[i] = amd_flip(cp[i]);
    }
    for j in 0..=n {
        head[j] = -1;
    }
    for j in (0..=n).rev() {
        if nv[j] > 0 {
            continue;
        }
        next[j] = head[u!(cp[j])];
        head[u!(cp[j])] = j as i64;
    }
    for e in (0..=n).rev() {
        if nv[e] <= 0 {
            continue;
        }
        if cp[e] != -1 {
            next[e] = head[u!(cp[e])];
            head[u!(cp[e])] = e as i64;
        }
    }
    let mut perm = vec![0i64; m];
    let mut k: i64 = 0;
    for i in 0..=n {
        if cp[i] == -1 {
            k = cs_tdfs(i as i64, k, &mut head, &next, &mut perm, &mut w);
        }
    }
    perm.truncate(n);
    perm.into_iter().map(|x| x as usize).collect()
}

impl Ldlt {
    /// 위 삼각 패턴(열 시작 ap, 행 ai)으로 순서 · 기호 분해
    pub fn analyze(n: usize, ap: Vec<usize>, ai: Vec<usize>) -> Ldlt {
        // C = selfadjointView<Upper>: 열 c = 자기 열의 행(≤c, 오름차순) + 뒤 열 j>c 의 (c,j) 거울(j 오름차순) → 정렬됨
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            for k in ap[j]..ap[j + 1] {
                let i = ai[k];
                if i <= j {
                    cols[j].push(i);
                }
            }
        }
        let mut full: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            for &i in &cols[j] {
                if i == j {
                    full[i].push(i);
                } else {
                    full[j].push(i);
                    full[i].push(j);
                }
            }
        }
        // permute_symm_to_fullsymm 의 채우기 순서: j 오름차순으로 열 j 의 행, (i<j 면 열 i 에 j) → 열마다 정렬과 같음
        for c in full.iter_mut() {
            c.sort_unstable();
        }
        let mut cp = vec![0usize; n + 1];
        let mut ci = Vec::new();
        for j in 0..n {
            ci.extend_from_slice(&full[j]);
            cp[j + 1] = ci.len();
        }
        let pinv = amd(n, &cp, &ci);
        let mut p = vec![0usize; n];
        for (i, &x) in pinv.iter().enumerate() {
            p[x] = i;
        }
        // ap = P A Pᵀ 위 삼각: 열 max(ip,jp) 에 행 min(ip,jp), 원본 열 j · 행 순으로 채움
        let mut count = vec![0usize; n];
        for j in 0..n {
            for k in ap[j]..ap[j + 1] {
                let i = ai[k];
                if i > j {
                    continue;
                }
                count[p[i].max(p[j])] += 1;
            }
        }
        let mut dp = vec![0usize; n + 1];
        for j in 0..n {
            dp[j + 1] = dp[j] + count[j];
        }
        let mut pos: Vec<usize> = dp[..n].to_vec();
        let mut di = vec![0usize; dp[n]];
        let mut dk = vec![0usize; dp[n]];
        for j in 0..n {
            for k in ap[j]..ap[j + 1] {
                let i = ai[k];
                if i > j {
                    continue;
                }
                let c = p[i].max(p[j]);
                let q = pos[c];
                pos[c] += 1;
                di[q] = p[i].min(p[j]);
                dk[q] = k;
            }
        }
        // analyzePattern_preordered: 소거 나무 · 열 개수
        let mut parent = vec![-1isize; n];
        let mut nz = vec![0usize; n];
        let mut tags = vec![0usize; n];
        for k in 0..n {
            parent[k] = -1;
            tags[k] = k;
            nz[k] = 0;
            for q in dp[k]..dp[k + 1] {
                let mut i = di[q];
                if i < k {
                    while tags[i] != k {
                        if parent[i] == -1 {
                            parent[i] = k as isize;
                        }
                        nz[i] += 1;
                        tags[i] = k;
                        i = parent[i] as usize;
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + nz[k];
        }
        Ldlt {
            n,
            ap_src: (ap, ai),
            p,
            pinv,
            ap_p: dp,
            ap_i: di,
            ap_src_k: dk,
            parent,
            li: vec![0; lp[n]],
            lx: vec![0.0; lp[n]],
            lp,
            d: vec![0.0; n],
        }
    }

    /// 수치 분해(입력 값 = 위 삼각 패턴 순서). 실패(대각 0)면 false
    pub fn factorize(&mut self, ax: &[f64]) -> bool {
        let n = self.n;
        let mut y = vec![0.0f64; n];
        let mut pattern = vec![0usize; n];
        let mut tags = vec![0usize; n];
        let mut nz = vec![0usize; n];
        for k in 0..n {
            y[k] = 0.0;
            let mut top = n;
            tags[k] = k;
            nz[k] = 0;
            for q in self.ap_p[k]..self.ap_p[k + 1] {
                let mut i = self.ap_i[q];
                if i <= k {
                    y[i] += ax[self.ap_src_k[q]];
                    let mut len = 0;
                    while tags[i] != k {
                        pattern[len] = i;
                        len += 1;
                        tags[i] = k;
                        i = self.parent[i] as usize;
                    }
                    while len > 0 {
                        top -= 1;
                        len -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }
            let mut d = y[k];
            y[k] = 0.0;
            while top < n {
                let i = pattern[top];
                let yi = y[i];
                y[i] = 0.0;
                let l_ki = yi / self.d[i];
                let p2 = self.lp[i] + nz[i];
                let mut p = self.lp[i];
                while p < p2 {
                    let r = self.li[p];
                    y[r] = -self.lx[p] * yi + y[r];
                    p += 1;
                }
                d -= l_ki * yi;
                self.li[p] = k;
                self.lx[p] = l_ki;
                nz[i] += 1;
                top += 1;
            }
            self.d[k] = d;
            if d == 0.0 {
                return false;
            }
        }
        true
    }

    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[self.p[i]] = b[i];
        }
        for i in 0..n {
            let t = x[i];
            if t != 0.0 {
                for p in self.lp[i]..self.lp[i + 1] {
                    let r = self.li[p];
                    x[r] = -self.lx[p] * t + x[r];
                }
            }
        }
        for i in 0..n {
            x[i] *= 1.0 / self.d[i];
        }
        for i in (0..n).rev() {
            let mut t = x[i];
            for p in self.lp[i]..self.lp[i + 1] {
                t -= self.lx[p] * x[self.li[p]];
            }
            x[i] = t;
        }
        let mut o = vec![0.0; n];
        for i in 0..n {
            o[self.pinv[i]] = x[i];
        }
        let _ = &self.ap_src;
        o
    }
}
