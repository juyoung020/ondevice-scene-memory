//! OpenCV 4.10 calib3d/src/epnp.cpp 를 같은 연산 순서로 옮긴 것.
//! RANSAC 안에서만 쓰므로 점은 늘 6 개(PnPRansac 의 model_points)이다. 배열은 고정 크기로 둔다.
use super::cvmath::{invert3_svd, jacobi_svd, mul_transposed_ata, solve_svd, svd_square};

pub const NP: usize = 6;

pub struct Epnp {
    uc: f64,
    vc: f64,
    fu: f64,
    fv: f64,
    pws: [f64; 3 * NP],
    us: [f64; 2 * NP],
    alphas: [f64; 4 * NP],
    pcs: [f64; 3 * NP],
    cws: [[f64; 3]; 4],
    ccs: [[f64; 3]; 4],
}

impl Epnp {
    /// opoints: 물체 점(float), upts: undistortPoints 결과(정규 좌표, float)
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, opoints: &[[f32; 3]; NP], upts: &[[f32; 2]; NP]) -> Epnp {
        let mut e = Epnp {
            uc: cx,
            vc: cy,
            fu: fx,
            fv: fy,
            pws: [0.0; 3 * NP],
            us: [0.0; 2 * NP],
            alphas: [0.0; 4 * NP],
            pcs: [0.0; 3 * NP],
            cws: [[0.0; 3]; 4],
            ccs: [[0.0; 3]; 4],
        };
        for i in 0..NP {
            e.pws[3 * i] = opoints[i][0] as f64;
            e.pws[3 * i + 1] = opoints[i][1] as f64;
            e.pws[3 * i + 2] = opoints[i][2] as f64;
            e.us[2 * i] = upts[i][0] as f64 * e.fu + e.uc;
            e.us[2 * i + 1] = upts[i][1] as f64 * e.fv + e.vc;
        }
        e
    }

    fn choose_control_points(&mut self) {
        let n = NP as f64;
        self.cws[0] = [0.0; 3];
        for i in 0..NP {
            for j in 0..3 {
                self.cws[0][j] += self.pws[3 * i + j];
            }
        }
        for j in 0..3 {
            self.cws[0][j] /= n;
        }
        let mut pw0 = [0f64; 3 * NP];
        for i in 0..NP {
            for j in 0..3 {
                pw0[3 * i + j] = self.pws[3 * i + j] - self.cws[0][j];
            }
        }
        let mut ptp = [0f64; 9];
        mul_transposed_ata(&pw0, NP, 3, &mut ptp);
        let a = [[ptp[0], ptp[1], ptp[2]], [ptp[3], ptp[4], ptp[5]], [ptp[6], ptp[7], ptp[8]]];
        // cvSVD(.., U_T): uct 의 행 = 왼쪽 특이벡터
        let (dc, uct, _) = svd_square::<3>(&a);
        for i in 1..4 {
            let k = (dc[i - 1] / n).sqrt();
            for j in 0..3 {
                self.cws[i][j] = self.cws[0][j] + k * uct[i - 1][j];
            }
        }
    }

    fn compute_barycentric_coordinates(&mut self) {
        let mut cc = [[0f64; 3]; 3];
        for i in 0..3 {
            for j in 1..4 {
                cc[i][j - 1] = self.cws[j][i] - self.cws[0][i];
            }
        }
        let ci = invert3_svd(&cc);
        for i in 0..NP {
            let pi = [self.pws[3 * i], self.pws[3 * i + 1], self.pws[3 * i + 2]];
            for j in 0..3 {
                self.alphas[4 * i + 1 + j] = ci[j][0] * (pi[0] - self.cws[0][0]) + ci[j][1] * (pi[1] - self.cws[0][1]) + ci[j][2] * (pi[2] - self.cws[0][2]);
            }
            let a = &mut self.alphas[4 * i..4 * i + 4];
            a[0] = 1.0 - a[1] - a[2] - a[3];
        }
    }

    /// 반환: (R 행 우선, t)
    pub fn compute_pose(&mut self) -> ([f64; 9], [f64; 3]) {
        self.choose_control_points();
        self.compute_barycentric_coordinates();
        let mut m = [0f64; 2 * NP * 12];
        for i in 0..NP {
            let row = 2 * i;
            let as_ = [self.alphas[4 * i], self.alphas[4 * i + 1], self.alphas[4 * i + 2], self.alphas[4 * i + 3]];
            let (u, v) = (self.us[2 * i], self.us[2 * i + 1]);
            for k in 0..4 {
                m[row * 12 + 3 * k] = as_[k] * self.fu;
                m[row * 12 + 3 * k + 1] = 0.0;
                m[row * 12 + 3 * k + 2] = as_[k] * (self.uc - u);
                m[(row + 1) * 12 + 3 * k] = 0.0;
                m[(row + 1) * 12 + 3 * k + 1] = as_[k] * self.fv;
                m[(row + 1) * 12 + 3 * k + 2] = as_[k] * (self.vc - v);
            }
        }
        let mut mtm = [0f64; 144];
        mul_transposed_ata(&m, 2 * NP, 12, &mut mtm);
        // cvSVD(MtM, D, Ut, 0, MODIFY_A | U_T): ut 의 행 = temp_a(전치된 MtM) 를 회전한 결과
        let mut at = [0f64; 144];
        for i in 0..12 {
            for k in 0..12 {
                at[i * 12 + k] = mtm[k * 12 + i];
            }
        }
        let mut d = [0f64; 12];
        let mut vt = [0f64; 144];
        jacobi_svd(&mut at, 12, &mut d, Some((&mut vt, 12)), 12, 12, 12);
        let ut = at;

        let mut l = [0f64; 60];
        compute_l_6x10(&ut, &mut l);
        let rho = self.compute_rho();

        let mut betas = [[0f64; 4]; 4];
        let mut rep = [0f64; 4];
        let mut rs = [[0f64; 9]; 4];
        let mut ts = [[0f64; 3]; 4];

        find_betas_approx_1(&l, &rho, &mut betas[1]);
        gauss_newton(&l, &rho, &mut betas[1]);
        rep[1] = self.compute_r_and_t(&ut, &betas[1], &mut rs[1], &mut ts[1]);

        find_betas_approx_2(&l, &rho, &mut betas[2]);
        gauss_newton(&l, &rho, &mut betas[2]);
        rep[2] = self.compute_r_and_t(&ut, &betas[2], &mut rs[2], &mut ts[2]);

        find_betas_approx_3(&l, &rho, &mut betas[3]);
        gauss_newton(&l, &rho, &mut betas[3]);
        rep[3] = self.compute_r_and_t(&ut, &betas[3], &mut rs[3], &mut ts[3]);

        let mut n = 1;
        if rep[2] < rep[1] {
            n = 2;
        }
        if rep[3] < rep[n] {
            n = 3;
        }
        (rs[n], ts[n])
    }

    fn compute_rho(&self) -> [f64; 6] {
        let c = &self.cws;
        [dist2(&c[0], &c[1]), dist2(&c[0], &c[2]), dist2(&c[0], &c[3]), dist2(&c[1], &c[2]), dist2(&c[1], &c[3]), dist2(&c[2], &c[3])]
    }

    fn compute_ccs(&mut self, betas: &[f64; 4], ut: &[f64; 144]) {
        for i in 0..4 {
            self.ccs[i] = [0.0; 3];
        }
        for i in 0..4 {
            let v = &ut[12 * (11 - i)..12 * (12 - i)];
            for j in 0..4 {
                for k in 0..3 {
                    self.ccs[j][k] += betas[i] * v[3 * j + k];
                }
            }
        }
    }

    fn compute_pcs(&mut self) {
        for i in 0..NP {
            let a = [self.alphas[4 * i], self.alphas[4 * i + 1], self.alphas[4 * i + 2], self.alphas[4 * i + 3]];
            for j in 0..3 {
                self.pcs[3 * i + j] = a[0] * self.ccs[0][j] + a[1] * self.ccs[1][j] + a[2] * self.ccs[2][j] + a[3] * self.ccs[3][j];
            }
        }
    }

    fn solve_for_sign(&mut self) {
        if self.pcs[2] < 0.0 {
            for i in 0..4 {
                for j in 0..3 {
                    self.ccs[i][j] = -self.ccs[i][j];
                }
            }
            for i in 0..NP {
                self.pcs[3 * i] = -self.pcs[3 * i];
                self.pcs[3 * i + 1] = -self.pcs[3 * i + 1];
                self.pcs[3 * i + 2] = -self.pcs[3 * i + 2];
            }
        }
    }

    fn compute_r_and_t(&mut self, ut: &[f64; 144], betas: &[f64; 4], r: &mut [f64; 9], t: &mut [f64; 3]) -> f64 {
        self.compute_ccs(betas, ut);
        self.compute_pcs();
        self.solve_for_sign();
        self.estimate_r_and_t(r, t);
        self.reprojection_error(r, t)
    }

    fn estimate_r_and_t(&self, r: &mut [f64; 9], t: &mut [f64; 3]) {
        let n = NP as f64;
        let mut pc0 = [0f64; 3];
        let mut pw0 = [0f64; 3];
        for i in 0..NP {
            for j in 0..3 {
                pc0[j] += self.pcs[3 * i + j];
                pw0[j] += self.pws[3 * i + j];
            }
        }
        for j in 0..3 {
            pc0[j] /= n;
            pw0[j] /= n;
        }
        let mut abt = [[0f64; 3]; 3];
        for i in 0..NP {
            let pc = &self.pcs[3 * i..3 * i + 3];
            let pw = &self.pws[3 * i..3 * i + 3];
            for j in 0..3 {
                abt[j][0] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
                abt[j][1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
                abt[j][2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
            }
        }
        // cvSVD(ABt, D, U, V, MODIFY_A): abt_u = U(열=왼쪽 벡터), abt_v = V(열=오른쪽 벡터)
        let (_d, ut, vt) = svd_square::<3>(&abt);
        // R[i][j] = dot(abt_u 행 i, abt_v 행 j) = Σ U[i][k] V[j][k] = Σ ut[k][i] vt[k][j]
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = ut[0][i] * vt[0][j] + ut[1][i] * vt[1][j] + ut[2][i] * vt[2][j];
            }
        }
        let det = r[0] * r[4] * r[8] + r[1] * r[5] * r[6] + r[2] * r[3] * r[7] - r[2] * r[4] * r[6] - r[1] * r[3] * r[8] - r[0] * r[5] * r[7];
        if det < 0.0 {
            r[6] = -r[6];
            r[7] = -r[7];
            r[8] = -r[8];
        }
        t[0] = pc0[0] - dot(&r[0..3], &pw0);
        t[1] = pc0[1] - dot(&r[3..6], &pw0);
        t[2] = pc0[2] - dot(&r[6..9], &pw0);
    }

    fn reprojection_error(&self, r: &[f64; 9], t: &[f64; 3]) -> f64 {
        let mut sum2 = 0f64;
        for i in 0..NP {
            let pw = &self.pws[3 * i..3 * i + 3];
            let xc = dot(&r[0..3], pw) + t[0];
            let yc = dot(&r[3..6], pw) + t[1];
            let inv_zc = 1.0 / (dot(&r[6..9], pw) + t[2]);
            let ue = self.uc + self.fu * xc * inv_zc;
            let ve = self.vc + self.fv * yc * inv_zc;
            let (u, v) = (self.us[2 * i], self.us[2 * i + 1]);
            sum2 += ((u - ue) * (u - ue) + (v - ve) * (v - ve)).sqrt();
        }
        sum2 / NP as f64
    }
}

#[inline]
fn dist2(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2])
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn compute_l_6x10(ut: &[f64; 144], l: &mut [f64; 60]) {
    let v = [&ut[12 * 11..12 * 12], &ut[12 * 10..12 * 11], &ut[12 * 9..12 * 10], &ut[12 * 8..12 * 9]];
    let mut dv = [[[0f64; 3]; 6]; 4];
    for i in 0..4 {
        let (mut a, mut b) = (0usize, 1usize);
        for j in 0..6 {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];
            b += 1;
            if b > 3 {
                a += 1;
                b = a + 1;
            }
        }
    }
    for i in 0..6 {
        let row = &mut l[10 * i..10 * i + 10];
        row[0] = dot(&dv[0][i], &dv[0][i]);
        row[1] = 2.0 * dot(&dv[0][i], &dv[1][i]);
        row[2] = dot(&dv[1][i], &dv[1][i]);
        row[3] = 2.0 * dot(&dv[0][i], &dv[2][i]);
        row[4] = 2.0 * dot(&dv[1][i], &dv[2][i]);
        row[5] = dot(&dv[2][i], &dv[2][i]);
        row[6] = 2.0 * dot(&dv[0][i], &dv[3][i]);
        row[7] = 2.0 * dot(&dv[1][i], &dv[3][i]);
        row[8] = 2.0 * dot(&dv[2][i], &dv[3][i]);
        row[9] = dot(&dv[3][i], &dv[3][i]);
    }
}

fn find_betas_approx_1(l: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let mut a = [[0f64; 4]; 6];
    for i in 0..6 {
        a[i] = [l[10 * i], l[10 * i + 1], l[10 * i + 3], l[10 * i + 6]];
    }
    let b4 = solve_svd::<6, 4>(&a, rho);
    if b4[0] < 0.0 {
        betas[0] = (-b4[0]).sqrt();
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = b4[0].sqrt();
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

fn find_betas_approx_2(l: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let mut a = [[0f64; 3]; 6];
    for i in 0..6 {
        a[i] = [l[10 * i], l[10 * i + 1], l[10 * i + 2]];
    }
    let b3 = solve_svd::<6, 3>(&a, rho);
    if b3[0] < 0.0 {
        betas[0] = (-b3[0]).sqrt();
        betas[1] = if b3[2] < 0.0 { (-b3[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b3[0].sqrt();
        betas[1] = if b3[2] > 0.0 { b3[2].sqrt() } else { 0.0 };
    }
    if b3[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = 0.0;
    betas[3] = 0.0;
}

fn find_betas_approx_3(l: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let mut a = [[0f64; 5]; 6];
    for i in 0..6 {
        a[i] = [l[10 * i], l[10 * i + 1], l[10 * i + 2], l[10 * i + 3], l[10 * i + 4]];
    }
    let b5 = solve_svd::<6, 5>(&a, rho);
    if b5[0] < 0.0 {
        betas[0] = (-b5[0]).sqrt();
        betas[1] = if b5[2] < 0.0 { (-b5[2]).sqrt() } else { 0.0 };
    } else {
        betas[0] = b5[0].sqrt();
        betas[1] = if b5[2] > 0.0 { b5[2].sqrt() } else { 0.0 };
    }
    if b5[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

fn gauss_newton(l: &[f64; 60], rho: &[f64; 6], betas: &mut [f64; 4]) {
    // x 는 반복 사이에 남는다(qr_solve 가 특이 행렬에서 일찍 끝나면 이전 값이 다시 더해짐)
    let mut x = [0f64; 4];
    for _ in 0..5 {
        let mut a = [0f64; 24];
        let mut b = [0f64; 6];
        for i in 0..6 {
            let rl = &l[i * 10..i * 10 + 10];
            let ra = &mut a[i * 4..i * 4 + 4];
            ra[0] = 2.0 * rl[0] * betas[0] + rl[1] * betas[1] + rl[3] * betas[2] + rl[6] * betas[3];
            ra[1] = rl[1] * betas[0] + 2.0 * rl[2] * betas[1] + rl[4] * betas[2] + rl[7] * betas[3];
            ra[2] = rl[3] * betas[0] + rl[4] * betas[1] + 2.0 * rl[5] * betas[2] + rl[8] * betas[3];
            ra[3] = rl[6] * betas[0] + rl[7] * betas[1] + rl[8] * betas[2] + 2.0 * rl[9] * betas[3];
            b[i] = rho[i]
                - (rl[0] * betas[0] * betas[0]
                    + rl[1] * betas[0] * betas[1]
                    + rl[2] * betas[1] * betas[1]
                    + rl[3] * betas[0] * betas[2]
                    + rl[4] * betas[1] * betas[2]
                    + rl[5] * betas[2] * betas[2]
                    + rl[6] * betas[0] * betas[3]
                    + rl[7] * betas[1] * betas[3]
                    + rl[8] * betas[2] * betas[3]
                    + rl[9] * betas[3] * betas[3]);
        }
        qr_solve(&mut a, &mut b, &mut x);
        for i in 0..4 {
            betas[i] += x[i];
        }
    }
}

/// epnp::qr_solve(6x4)
fn qr_solve(a: &mut [f64; 24], b: &mut [f64; 6], x: &mut [f64; 4]) {
    const NR: usize = 6;
    const NC: usize = 4;
    let mut a1 = [0f64; NR];
    let mut a2 = [0f64; NR];
    let mut kk = 0usize; // ppAkk 위치
    for k in 0..NC {
        let mut ik1 = kk;
        let mut eta = a[ik1].abs();
        for _ in k + 1..NR {
            let elt = a[ik1].abs();
            if eta < elt {
                eta = elt;
            }
            ik1 += NC;
        }
        if eta == 0.0 {
            // A1[k] = A2[k] = 0 후 반환(x 는 그대로)
            return;
        }
        let mut ik2 = kk;
        let mut sum2 = 0f64;
        let inv_eta = 1.0 / eta;
        for _ in k..NR {
            a[ik2] *= inv_eta;
            sum2 += a[ik2] * a[ik2];
            ik2 += NC;
        }
        let mut sigma = sum2.sqrt();
        if a[kk] < 0.0 {
            sigma = -sigma;
        }
        a[kk] += sigma;
        a1[k] = sigma * a[kk];
        a2[k] = -eta * sigma;
        for j in k + 1..NC {
            let mut ik = kk;
            let mut sum = 0f64;
            for _ in k..NR {
                sum += a[ik] * a[ik + j - k];
                ik += NC;
            }
            let tau = sum / a1[k];
            ik = kk;
            for _ in k..NR {
                a[ik + j - k] -= tau * a[ik];
                ik += NC;
            }
        }
        kk += NC + 1;
    }
    // b <- Qt b
    let mut jj = 0usize;
    for j in 0..NC {
        let mut ij = jj;
        let mut tau = 0f64;
        for i in j..NR {
            tau += a[ij] * b[i];
            ij += NC;
        }
        tau /= a1[j];
        ij = jj;
        for i in j..NR {
            b[i] -= tau * a[ij];
            ij += NC;
        }
        jj += NC + 1;
    }
    // X = R⁻¹ b
    x[NC - 1] = b[NC - 1] / a2[NC - 1];
    for i in (0..NC - 1).rev() {
        let mut ij = i * NC + (i + 1);
        let mut sum = 0f64;
        for j in i + 1..NC {
            sum += a[ij] * x[j];
            ij += 1;
        }
        x[i] = (b[i] - sum) / a2[i];
    }
}
