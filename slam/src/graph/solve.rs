//! 가우스-뉴턴 한 걸음의 선형 풀이: 6×6 블록 희소 정규방정식 (Σ AᵀA) δ = Σ Aᵀb 를 블록 숄레스키로.
//! 원본(GTSAM 4.2 GaussianFactorGraph::optimize: COLAMD 순서 · 다전선(multifrontal) 숄레스키 · Eigen 블록 커널)과
//! 수학은 같고 덧셈 순서만 다르다 → δ 가 원본과 마지막 몇 ulp 다를 수 있음(graph.rs 머리말의 검증 결과 참고).
//! 순서: 최소 차수(동률이면 작은 번호), 블록 숄레스키는 오른쪽으로 갱신(right-looking).
use super::geom::M6;
use std::collections::{BTreeSet, HashMap};

/// 정규방정식: 대각 블록, 아래 삼각 블록 (i > j), 오른쪽 변
pub struct Normal {
    pub diag: Vec<M6>,
    pub off: HashMap<(usize, usize), M6>,
    pub g: Vec<[f64; 6]>,
}

impl Normal {
    pub fn new(n: usize) -> Normal {
        Normal { diag: vec![[[0.0; 6]; 6]; n], off: HashMap::new(), g: vec![[0.0; 6]; n] }
    }

    /// 인자 하나(화이트닝된 A 블록들, b)를 더함
    pub fn add(&mut self, keys: &[usize], a: &[M6], b: &[f64; 6]) {
        for (p, &ki) in keys.iter().enumerate() {
            for r in 0..6 {
                let mut s = 0.0;
                for k in 0..6 {
                    s += a[p][k][r] * b[k];
                }
                self.g[ki][r] += s;
            }
            for (q, &kj) in keys.iter().enumerate() {
                if kj > ki || (kj == ki && q != p) {
                    continue;
                }
                let blk = if ki == kj { &mut self.diag[ki] } else { self.off.entry((ki, kj)).or_insert([[0.0; 6]; 6]) };
                for r in 0..6 {
                    for c in 0..6 {
                        let mut s = 0.0;
                        for k in 0..6 {
                            s += a[p][k][r] * a[q][k][c];
                        }
                        blk[r][c] += s;
                    }
                }
            }
        }
    }
}

/// 6×6 대칭 양의 정부호 → 아래 삼각 L (L Lᵀ = m). 양이 아니면 None(GTSAM IndeterminantLinearSystem 에 해당)
fn chol6(m: &M6) -> Option<M6> {
    let mut l = [[0.0; 6]; 6];
    for j in 0..6 {
        let mut d = m[j][j];
        for k in 0..j {
            d -= l[j][k] * l[j][k];
        }
        if !(d > 0.0) {
            return None;
        }
        let d = d.sqrt();
        l[j][j] = d;
        for i in j + 1..6 {
            let mut s = m[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = s / d;
        }
    }
    Some(l)
}

/// X Lᵀ = B 를 X 에 대해 (L 아래 삼각)
fn solve_xlt(b: &M6, l: &M6) -> M6 {
    let mut x = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            let mut s = b[r][c];
            for k in 0..c {
                s -= x[r][k] * l[c][k];
            }
            x[r][c] = s / l[c][c];
        }
    }
    x
}

/// a -= x yᵀ
fn sub_xyt(a: &mut M6, x: &M6, y: &M6) {
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += x[r][k] * y[c][k];
            }
            a[r][c] -= s;
        }
    }
}

/// 최소 차수 순서(변수 그래프에서 이웃 수가 가장 적은 것부터, 지운 변수의 이웃끼리는 이어 붙임)
fn min_degree(n: usize, off: &HashMap<(usize, usize), M6>) -> Vec<usize> {
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for &(i, j) in off.keys() {
        adj[i].insert(j);
        adj[j].insert(i);
    }
    let mut alive = vec![true; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let v = (0..n).filter(|&v| alive[v]).min_by_key(|&v| (adj[v].len(), v)).unwrap();
        alive[v] = false;
        let nb: Vec<usize> = adj[v].iter().copied().collect();
        for &a in &nb {
            adj[a].remove(&v);
            for &b in &nb {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
        adj[v].clear();
        order.push(v);
    }
    order
}

/// (Σ AᵀA) δ = Σ Aᵀb 의 δ (변수 번호 순)
pub fn solve(sys: Normal) -> Option<Vec<[f64; 6]>> {
    let n = sys.diag.len();
    let order = min_degree(n, &sys.off);
    let mut pos = vec![0usize; n];
    for (p, &v) in order.iter().enumerate() {
        pos[v] = p;
    }
    // 순서대로 다시 번호 매김: 아래 삼각(행 > 열)
    let mut diag: Vec<M6> = vec![[[0.0; 6]; 6]; n];
    let mut rhs = vec![[0.0; 6]; n];
    for v in 0..n {
        diag[pos[v]] = sys.diag[v];
        rhs[pos[v]] = sys.g[v];
    }
    let mut low: HashMap<(usize, usize), M6> = HashMap::with_capacity(sys.off.len() * 2);
    let mut cols: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (&(i, j), b) in &sys.off {
        let (pi, pj) = (pos[i], pos[j]);
        if pi > pj {
            low.insert((pi, pj), *b);
            cols[pj].insert(pi);
        } else {
            let mut t = [[0.0; 6]; 6];
            for r in 0..6 {
                for c in 0..6 {
                    t[r][c] = b[c][r];
                }
            }
            low.insert((pj, pi), t);
            cols[pi].insert(pj);
        }
    }
    // 분해
    let mut lcols: Vec<Vec<(usize, M6)>> = Vec::with_capacity(n);
    let mut ldiag: Vec<M6> = Vec::with_capacity(n);
    for k in 0..n {
        let lkk = chol6(&diag[k])?;
        let rows: Vec<usize> = cols[k].iter().copied().collect();
        let mut col: Vec<(usize, M6)> = Vec::with_capacity(rows.len());
        for &i in &rows {
            let h = low.remove(&(i, k)).unwrap();
            col.push((i, solve_xlt(&h, &lkk)));
        }
        for a in 0..col.len() {
            let (ia, ref la) = col[a];
            sub_xyt(&mut diag[ia], la, la);
            for b in 0..a {
                let (ib, ref lb) = col[b];
                // ia > ib (rows 가 오름차순)
                let e = low.entry((ia, ib)).or_insert_with(|| {
                    cols[ib].insert(ia);
                    [[0.0; 6]; 6]
                });
                sub_xyt(e, la, lb);
            }
        }
        ldiag.push(lkk);
        lcols.push(col);
    }
    // L y = g
    let mut y = rhs;
    for k in 0..n {
        let l = &ldiag[k];
        let mut v = y[k];
        for r in 0..6 {
            let mut s = v[r];
            for c in 0..r {
                s -= l[r][c] * v[c];
            }
            v[r] = s / l[r][r];
        }
        y[k] = v;
        for (i, b) in &lcols[k] {
            for r in 0..6 {
                let mut s = 0.0;
                for c in 0..6 {
                    s += b[r][c] * v[c];
                }
                y[*i][r] -= s;
            }
        }
    }
    // Lᵀ x = y
    for k in (0..n).rev() {
        let mut v = y[k];
        for (i, b) in &lcols[k] {
            let xi = y[*i];
            for r in 0..6 {
                let mut s = 0.0;
                for c in 0..6 {
                    s += b[c][r] * xi[c];
                }
                v[r] -= s;
            }
        }
        let l = &ldiag[k];
        for r in (0..6).rev() {
            let mut s = v[r];
            for c in r + 1..6 {
                s -= l[c][r] * v[c];
            }
            v[r] = s / l[r][r];
        }
        y[k] = v;
    }
    let mut out = vec![[0.0; 6]; n];
    for v in 0..n {
        out[v] = y[pos[v]];
    }
    Some(out)
}
