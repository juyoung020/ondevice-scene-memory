//! Eigen 3.4.0 EigenSolver<Matrix4f>(RealSchur · HessenbergDecomposition · Householder · JacobiRotation)를
//! 기준 빌드(-ffp-contract=off, docs/orb3_port.md 공통 규칙)가 계산하는 순서 그대로 옮긴 것: 곱한 뒤 더함(FMA 없음).
//! 덧셈 묶음 순서는 GIMPLE(-fdump-tree-optimized)에서 읽은 Eigen 전개 순서. 행렬은 열 우선 [f32; 16].

#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}
/// c - a*b (한 번 반올림, GIMPLE .FNMA)
#[inline(always)]
fn msub(a: f32, b: f32, c: f32) -> f32 {
    c - a * b
}
#[inline(always)]
fn ix(r: usize, c: usize) -> usize {
    r + 4 * c
}

const EPS: f32 = f32::EPSILON;
const MINP: f32 = f32::MIN_POSITIVE;

/// 동적 길이 내적(Redux, 스칼라 경로): 첫 곱 + 이어지는 곱 더함
#[inline]
fn dot_dyn(n: usize, a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32) -> f32 {
    dot_dyn_t(false, n, a, b)
}
/// f = true: Eigen gemv 커널(pmadd = _mm*_fmadd_ps 내장 함수라 -ffp-contract=off 에서도 FMA)
fn dot_dyn_t(f: bool, n: usize, a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32) -> f32 {
    let mut r = a(0) * b(0);
    for k in 1..n {
        r = if f { a(k).mul_add(b(k), r) } else { madd(a(k), b(k), r) };
    }
    r
}

/// makeHouseholder(동적 길이 꼬리): (essential, tau, beta)
fn make_householder_dyn(v: &[f32]) -> ([f32; 3], f32, f32) {
    let n = v.len();
    let tail = &v[1..];
    let tsq = if n == 1 { 0.0 } else { dot_dyn(n - 1, |k| tail[k], |k| tail[k]) };
    let c0 = v[0];
    let mut ess = [0f32; 3];
    if tsq <= MINP {
        (ess, 0.0, c0)
    } else {
        let mut beta = madd(c0, c0, tsq).sqrt();
        if c0 >= 0.0 {
            beta = -beta;
        }
        let d = c0 - beta;
        for k in 0..n - 1 {
            ess[k] = tail[k] / d;
        }
        (ess, (beta - c0) / beta, beta)
    }
}

/// 고정 길이 3/2 makeHouseholder (performFrancisQRStep 안)
fn make_householder_fix(v: &[f32]) -> ([f32; 2], f32, f32) {
    let c0 = v[0];
    let tsq = if v.len() == 3 { madd(v[2], v[2], v[1] * v[1]) } else { v[1] * v[1] };
    let mut ess = [0f32; 2];
    if tsq <= MINP {
        (ess, 0.0, c0)
    } else {
        let mut beta = madd(c0, c0, tsq).sqrt();
        if c0 >= 0.0 {
            beta = -beta;
        }
        let d = c0 - beta;
        for k in 0..v.len() - 1 {
            ess[k] = v[k + 1] / d;
        }
        (ess, (beta - c0) / beta, beta)
    }
}

/// block(r0,c0,rows,cols).applyHouseholderOnTheLeft(ess, tau). fixed=true 면 고정 길이 ess(2·1) 판
fn hh_left(m: &mut [f32; 16], r0: usize, c0: usize, rows: usize, cols: usize, ess: &[f32], tau: f32, fixed: bool) {
    if rows == 1 {
        let f = 1.0 - tau;
        for j in 0..cols {
            m[ix(r0, c0 + j)] *= f;
        }
        return;
    }
    if tau == 0.0 {
        return;
    }
    let mut tmp = [0f32; 4];
    for j in 0..cols {
        let c = c0 + j;
        let t = if fixed && rows == 3 {
            madd(ess[1], m[ix(r0 + 2, c)], ess[0] * m[ix(r0 + 1, c)])
        } else {
            dot_dyn(rows - 1, |k| ess[k], |k| m[ix(r0 + 1 + k, c)])
        };
        tmp[j] = t + m[ix(r0, c)];
    }
    for j in 0..cols {
        m[ix(r0, c0 + j)] = msub(tau, tmp[j], m[ix(r0, c0 + j)]);
    }
    let mut te = [0f32; 3];
    for k in 0..rows - 1 {
        te[k] = tau * ess[k];
    }
    for j in 0..cols {
        for k in 0..rows - 1 {
            let p = ix(r0 + 1 + k, c0 + j);
            m[p] = msub(te[k], tmp[j], m[p]);
        }
    }
}

/// block(r0,c0,rows,cols).applyHouseholderOnTheRight(ess, tau)
fn hh_right(m: &mut [f32; 16], r0: usize, c0: usize, rows: usize, cols: usize, ess: &[f32], tau: f32, fixed: bool) {
    if cols == 1 {
        let f = 1.0 - tau;
        for i in 0..rows {
            m[ix(r0 + i, c0)] *= f;
        }
        return;
    }
    if tau == 0.0 {
        return;
    }
    let mut tmp = [0f32; 4];
    for i in 0..rows {
        let r = r0 + i;
        let t = if fixed && cols == 3 {
            madd(m[ix(r, c0 + 2)], ess[1], m[ix(r, c0 + 1)] * ess[0])
        } else {
            dot_dyn_t(true, cols - 1, |k| m[ix(r, c0 + 1 + k)], |k| ess[k])
        };
        tmp[i] = t + m[ix(r, c0)];
    }
    for i in 0..rows {
        m[ix(r0 + i, c0)] = msub(tau, tmp[i], m[ix(r0 + i, c0)]);
    }
    for i in 0..rows {
        let tt = tau * tmp[i];
        for k in 0..cols - 1 {
            let p = ix(r0 + i, c0 + 1 + k);
            m[p] = msub(tt, ess[k], m[p]);
        }
    }
}

/// apply_rotation_in_the_plane(x, y, (c, s)) : x' = c*x + s*y, y' = -s*x + c*y (곱 더함 순서)
fn rot_plane(m: &mut [f32; 16], xs: &[usize], ys: &[usize], c: f32, s: f32) {
    if c == 1.0 && s == 0.0 {
        return;
    }
    for (&px, &py) in xs.iter().zip(ys) {
        let (xi, yi) = (m[px], m[py]);
        m[px] = madd(c, xi, s * yi);
        m[py] = madd(-s, xi, c * yi);
    }
}

/// JacobiRotation::makeGivens(p, q) → (c, s)
fn make_givens(p: f32, q: f32) -> (f32, f32) {
    if q == 0.0 {
        (if p < 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else if p == 0.0 {
        (0.0, if q < 0.0 { 1.0 } else { -1.0 })
    } else if p.abs() > q.abs() {
        let t = q / p;
        let mut u = madd(t, t, 1.0).sqrt();
        if p < 0.0 {
            u = -u;
        }
        let c = 1.0 / u;
        (c, -t * c)
    } else {
        let t = p / q;
        let mut u = madd(t, t, 1.0).sqrt();
        if q < 0.0 {
            u = -u;
        }
        let s = -1.0 / u;
        (-t * s, s)
    }
}

pub struct Eig4 {
    /// 정규화 전 고유벡터(doComputeEigenvectors 결과)
    pub evec_raw: [f32; 16],
    pub t: [f32; 16],
    pub u: [f32; 16],
    pub eval_re: [f32; 4],
    pub eval_im: [f32; 4],
    pub evec: [f32; 16],
    pub ok: bool,
}

/// HessenbergDecomposition::_compute → (행렬(아래에 반사 벡터), hCoeffs)
pub fn hessenberg(a: &mut [f32; 16]) -> [f32; 3] {
    let mut h = [0f32; 3];
    for i in 0..3 {
        let rem = 3 - i;
        let v: Vec<f32> = (0..rem).map(|k| a[ix(i + 1 + k, i)]).collect();
        let (ess, tau, beta) = make_householder_dyn(&v);
        for k in 0..rem - 1 {
            a[ix(i + 2 + k, i)] = ess[k];
        }
        a[ix(i + 1, i)] = beta;
        h[i] = tau;
        let e: Vec<f32> = (0..rem - 1).map(|k| a[ix(i + 2 + k, i)]).collect();
        hh_left(a, i + 1, i + 1, rem, rem, &e, tau, false);
        hh_right(a, 0, i + 1, 4, rem, &e, tau, false);
    }
    h
}

pub struct Schur {
    pub t: [f32; 16],
    pub u: [f32; 16],
    pub exshift: f32,
}

impl Schur {
    pub fn split_off_two_rows(&mut self, iu: usize) {
        let t = &mut self.t;
        let p = (t[ix(iu - 1, iu - 1)] - t[ix(iu, iu)]) * 0.5;
        let q = madd(p, p, t[ix(iu, iu - 1)] * t[ix(iu - 1, iu)]);
        t[ix(iu, iu)] += self.exshift;
        t[ix(iu - 1, iu - 1)] += self.exshift;
        if q >= 0.0 {
            let z = q.abs().sqrt();
            let (c, s) = if p >= 0.0 { make_givens(p + z, t[ix(iu, iu - 1)]) } else { make_givens(p - z, t[ix(iu, iu - 1)]) };
            // rightCols(size-iu+1).applyOnTheLeft(iu-1, iu, rot.adjoint()) : (c, -s)
            let xs: Vec<usize> = (iu - 1..4).map(|c| ix(iu - 1, c)).collect();
            let ys: Vec<usize> = (iu - 1..4).map(|c| ix(iu, c)).collect();
            rot_plane(t, &xs, &ys, c, -s);
            // topRows(iu+1).applyOnTheRight(iu-1, iu, rot) : transpose → (c, -s)
            let xs: Vec<usize> = (0..=iu).map(|r| ix(r, iu - 1)).collect();
            let ys: Vec<usize> = (0..=iu).map(|r| ix(r, iu)).collect();
            rot_plane(t, &xs, &ys, c, -s);
            t[ix(iu, iu - 1)] = 0.0;
            let xs: Vec<usize> = (0..4).map(|r| ix(r, iu - 1)).collect();
            let ys: Vec<usize> = (0..4).map(|r| ix(r, iu)).collect();
            rot_plane(&mut self.u, &xs, &ys, c, -s);
        }
        if iu > 1 {
            self.t[ix(iu - 1, iu - 2)] = 0.0;
        }
    }

    fn compute_shift(&mut self, iu: usize, iter: usize) -> [f32; 3] {
        let t = &mut self.t;
        let mut sh = [t[ix(iu, iu)], t[ix(iu - 1, iu - 1)], t[ix(iu, iu - 1)] * t[ix(iu - 1, iu)]];
        if iter == 10 {
            self.exshift += sh[0];
            for i in 0..=iu {
                t[ix(i, i)] -= sh[0];
            }
            let s = t[ix(iu, iu - 1)].abs() + t[ix(iu - 1, iu - 2)].abs();
            sh[0] = 0.75 * s;
            sh[1] = 0.75 * s;
            sh[2] = s * (-0.4375 * s);
        }
        if iter == 30 {
            let h = (sh[1] - sh[0]) * 0.5;
            let mut s = madd(h, h, sh[2]);
            if s > 0.0 {
                s = s.sqrt();
                if sh[1] < sh[0] {
                    s = -s;
                }
                s += h;
                s = sh[0] - sh[2] / s;
                self.exshift += s;
                for i in 0..=iu {
                    t[ix(i, i)] -= s;
                }
                sh = [0.964; 3];
            }
        }
        sh
    }

    fn init_francis(&self, il: usize, iu: usize, sh: &[f32; 3]) -> (usize, [f32; 3]) {
        let t = &self.t;
        let mut im = iu - 2;
        loop {
            let tmm = t[ix(im, im)];
            let r = sh[0] - tmm;
            let s = sh[1] - tmm;
            let v0 = (r * s - sh[2]) / t[ix(im + 1, im)] + t[ix(im, im + 1)];
            let v1 = ((t[ix(im + 1, im + 1)] - tmm) - r) - s;
            let v2 = t[ix(im + 2, im + 1)];
            let v = [v0, v1, v2];
            if im == il {
                return (im, v);
            }
            let lhs = t[ix(im, im - 1)] * (v1.abs() + v2.abs());
            let rhs = v0 * ((t[ix(im - 1, im - 1)].abs() + tmm.abs()) + t[ix(im + 1, im + 1)].abs());
            if lhs.abs() < rhs * EPS {
                return (im, v);
            }
            im -= 1;
        }
    }

    pub fn francis_step(&mut self, il: usize, im: usize, iu: usize, first_v: [f32; 3]) {
        for k in im..=iu - 2 {
            let first = k == im;
            let v = if first { first_v } else { [self.t[ix(k, k - 1)], self.t[ix(k + 1, k - 1)], self.t[ix(k + 2, k - 1)]] };
            let (ess, tau, beta) = make_householder_fix(&v);
            if beta != 0.0 {
                if first && k > il {
                    self.t[ix(k, k - 1)] = -self.t[ix(k, k - 1)];
                } else if !first {
                    self.t[ix(k, k - 1)] = beta;
                }
                hh_left(&mut self.t, k, k, 3, 4 - k, &ess, tau, true);
                hh_right(&mut self.t, 0, k, iu.min(k + 3) + 1, 3, &ess, tau, true);
                hh_right(&mut self.u, 0, k, 4, 3, &ess, tau, true);
            }
        }
        let v = [self.t[ix(iu - 1, iu - 2)], self.t[ix(iu, iu - 2)]];
        let (ess, tau, beta) = make_householder_fix(&v);
        if beta != 0.0 {
            self.t[ix(iu - 1, iu - 2)] = beta;
            hh_left(&mut self.t, iu - 1, iu - 1, 2, 4 - iu + 1, &ess[..1], tau, true);
            hh_right(&mut self.t, 0, iu - 1, iu + 1, 2, &ess[..1], tau, true);
            hh_right(&mut self.u, 0, iu - 1, 4, 2, &ess[..1], tau, true);
        }
        for i in im + 2..=iu {
            self.t[ix(i, i - 2)] = 0.0;
            if i > im + 2 {
                self.t[ix(i, i - 3)] = 0.0;
            }
        }
    }
}

/// EigenSolver<Matrix4f>::compute(a, true) + eigenvectors()(열마다 정규화, 실수부)
pub fn eigen_solver4(a: &[f32; 16]) -> Eig4 {
    // RealSchur::compute
    let mut scale = 0f32;
    for &x in a {
        let y = x.abs();
        if y > scale {
            scale = y;
        }
    }
    let mut out = Eig4 { t: [0.0; 16], u: [0.0; 16], eval_re: [0.0; 4], eval_im: [0.0; 4], evec: [0.0; 16], evec_raw: [0.0; 16], ok: true };
    if scale < MINP {
        for i in 0..4 {
            out.u[ix(i, i)] = 1.0;
        }
        // 고유값 0, 고유벡터 단위
        out.evec = out.u;
        return out;
    }
    let mut h = [0f32; 16];
    for i in 0..16 {
        h[i] = a[i] / scale;
    }
    let hc = hessenberg(&mut h);
    // matrixQ().evalTo(U)
    let mut u = [0f32; 16];
    for i in 0..4 {
        u[ix(i, i)] = 1.0;
    }
    for k in (0..3).rev() {
        let cs = 3 - k;
        let e: Vec<f32> = (0..2 - k).map(|j| h[ix(k + 2 + j, k)]).collect();
        hh_left(&mut u, 4 - cs, 4 - cs, cs, cs, &e, hc[k], false);
    }
    // matrixH
    let mut t = h;
    for c in 0..4 {
        for r in c + 2..4 {
            t[ix(r, c)] = 0.0;
        }
    }
    let mut sc = Schur { t, u, exshift: 0.0 };
    // computeFromHessenberg
    let mut norm = 0f32;
    for j in 0..4 {
        let n = 4.min(j + 2);
        let mut s = sc.t[ix(0, j)].abs();
        for r in 1..n {
            s += sc.t[ix(r, j)].abs();
        }
        norm += s;
    }
    let caz = (norm * (EPS * EPS)).max(MINP);
    let caz = if norm * (EPS * EPS) < MINP { MINP } else { caz };
    let max_iters = 160;
    let (mut iter, mut total) = (0usize, 0usize);
    if norm != 0.0 {
        let mut iu: isize = 3;
        while iu >= 0 {
            let iuu = iu as usize;
            // findSmallSubdiagEntry
            let mut il = iuu;
            while il > 0 {
                let s = sc.t[ix(il - 1, il - 1)].abs() + sc.t[ix(il, il)].abs();
                let s2 = s * EPS;
                let s = if s2 < caz { caz } else { s2 };
                if sc.t[ix(il, il - 1)].abs() <= s {
                    break;
                }
                il -= 1;
            }
            if il == iuu {
                sc.t[ix(iuu, iuu)] += sc.exshift;
                if iuu > 0 {
                    sc.t[ix(iuu, iuu - 1)] = 0.0;
                }
                iu -= 1;
                iter = 0;
            } else if il + 1 == iuu {
                sc.split_off_two_rows(iuu);
                iu -= 2;
                iter = 0;
            } else {
                let sh = sc.compute_shift(iuu, iter);
                iter += 1;
                total += 1;
                if total > max_iters {
                    break;
                }
                let (im, v) = sc.init_francis(il, iuu, &sh);
                sc.francis_step(il, im, iuu, v);
            }
        }
    }
    if total > max_iters {
        out.ok = false;
        return out;
    }
    for x in sc.t.iter_mut() {
        *x *= scale;
    }
    out.t = sc.t;
    out.u = sc.u;
    // EigenSolver::compute: 고유값
    let tt = &out.t;
    let mut i = 0;
    while i < 4 {
        if i == 3 || tt[ix(i + 1, i)] == 0.0 {
            out.eval_re[i] = tt[ix(i, i)];
            i += 1;
        } else {
            let p = 0.5 * (tt[ix(i, i)] - tt[ix(i + 1, i + 1)]);
            let (mut t0, mut t1) = (tt[ix(i + 1, i)], tt[ix(i, i + 1)]);
            let mx = p.abs().max(t0.abs().max(t1.abs()));
            t0 /= mx;
            t1 /= mx;
            let p0 = p / mx;
            let z = mx * madd(p0, p0, t0 * t1).abs().sqrt();
            out.eval_re[i] = tt[ix(i + 1, i + 1)] + p;
            out.eval_im[i] = z;
            out.eval_re[i + 1] = p + tt[ix(i + 1, i + 1)];
            out.eval_im[i + 1] = -z;
            i += 2;
        }
    }
    eigenvectors_real(&mut out);
    out
}

/// doComputeEigenvectors (모든 고유값이 실수인 경로) + eigenvectors() 정규화
fn eigenvectors_real(o: &mut Eig4) {
    if o.eval_im.iter().any(|&x| x != 0.0) {
        // 대칭 N 에선 생기지 않는 복소 경로: 원본과 같은지 확인하지 않음
        o.ok = false;
    }
    let mut t = o.t;
    let norm = eig_norm(&t);
    let mut ev = o.u;
    do_compute_eigenvectors(&mut ev, &mut t, &o.eval_re, norm);
    o.evec_raw = ev;
    normalize_cols(&mut ev);
    o.evec = ev;
}

/// doComputeEigenvectors 머리의 norm
pub fn eig_norm(t: &[f32; 16]) -> f32 {
    let mut norm = 0f32;
    for j in 0..4 {
        let s0 = (j as usize).saturating_sub(1);
        let mut s = t[ix(j, s0)].abs();
        for c in s0 + 1..4 {
            s += t[ix(j, c)].abs();
        }
        norm += s;
    }
    norm
}

/// doComputeEigenvectors 의 실수 고유값 경로(ev = U 를 고유벡터로 덮어씀)
pub fn do_compute_eigenvectors(ev: &mut [f32; 16], t: &mut [f32; 16], eval_re: &[f32; 4], norm: f32) {
    let o_eval_re = eval_re;
    if norm != 0.0 {
        for n in (0..4).rev() {
            let p = o_eval_re[n];
            let mut l = n;
            t[ix(n, n)] = 1.0;
            for i in (0..n).rev() {
                let w = t[ix(i, i)] - p;
                let r = dot_dyn(n - l + 1, |k| t[ix(i, l + k)], |k| t[ix(l + k, n)]);
                l = i;
                if w != 0.0 {
                    t[ix(i, n)] = -r / w;
                } else {
                    t[ix(i, n)] = -r / (EPS * norm);
                }
                let tv = t[ix(i, n)].abs();
                if (EPS * tv) * tv > 1.0 {
                    for k in i..4 {
                        t[ix(k, n)] /= tv;
                    }
                }
            }
        }
        // 되돌리기: m_tmp = eivec.leftCols(j+1) * T.col(j).head(j+1)
        for j in (0..4).rev() {
            let mut tmp = [0f32; 4];
            for r in 0..4 {
                let mut acc = ev[ix(r, 0)] * t[ix(0, j)];
                for k in 1..=j {
                    acc = ev[ix(r, k)].mul_add(t[ix(k, j)], acc); // 행렬 곱(gemv): Eigen pmadd 는 기준 빌드에서도 FMA
                }
                tmp[r] = acc;
            }
            for r in 0..4 {
                ev[ix(r, j)] = tmp[r];
            }
        }
    }
}

/// eigenvectors(): 열을 복소수로 바꿔 normalize. 합 = (a3+a2)+(a1+a0), 나눗셈은 복소 나눗셈 그대로
/// (x·conj(s))/(s·s) → 실수부 = (x*s)/(s*s)
pub fn normalize_cols(ev: &mut [f32; 16]) {
    for j in 0..4 {
        let a: [f32; 4] = std::array::from_fn(|r| ev[ix(r, j)] * ev[ix(r, j)]);
        let z = (a[3] + a[2]) + (a[1] + a[0]);
        if z > 0.0 {
            let s = z.sqrt();
            let ss = s * s;
            for r in 0..4 {
                ev[ix(r, j)] = (ev[ix(r, j)] * s) / ss;
            }
        }
    }
}
