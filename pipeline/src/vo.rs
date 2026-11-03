//! 카메라만으로 포즈(시각 오도메트리): SuperPoint 특징점(NPU) → 이전 키프레임과 기술자 상호 최근접 짝 →
//! 이전 장면 깊이로 3D(세계) → 현재 장면 깊이로 3D(카메라) 3점 RANSAC(Horn 쿼터니언 정렬)로 초기값 →
//! 맞는 짝들의 재투영 오차를 가우스-뉴턴(후버 가중)으로 줄여 현재 카메라 포즈.
use crate::geom::*;

pub struct Feat {
    pub kp: Vec<(f32, f32)>, // 화소 (u, v)
    pub desc: Vec<f32>,      // kp 마다 256, L2 정규화
}

/// SuperPoint 출력(semi [65,Hc,Wc], desc [256,Hc,Wc]) → 칸마다 가장 센 점 하나, 점수 상위 n 개
pub fn superpoint_decode(semi: &[f32], desc: &[f32], hc: usize, wc: usize, n: usize, thr: f32) -> Feat {
    let cells = hc * wc;
    let mut cand: Vec<(f32, f32, f32)> = vec![];
    for cy in 0..hc {
        for cx in 0..wc {
            let i = cy * wc + cx;
            let mut mx = f32::MIN;
            for c in 0..65 {
                mx = mx.max(semi[c * cells + i]);
            }
            let mut sum = 0.0;
            for c in 0..65 {
                sum += (semi[c * cells + i] - mx).exp();
            }
            let (mut best, mut bc) = (0.0f32, 0usize);
            for c in 0..64 {
                let p = (semi[c * cells + i] - mx).exp() / sum;
                if p > best {
                    best = p;
                    bc = c;
                }
            }
            if best > thr {
                let u = (cx * 8 + bc % 8) as f32;
                let v = (cy * 8 + bc / 8) as f32;
                cand.push((best, u, v));
            }
        }
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    cand.truncate(n);
    let mut kp = vec![];
    let mut d = vec![];
    for (_, u, v) in cand {
        // 기술자: 칸 좌표로 쌍선형 표본
        let fx = ((u + 0.5) / 8.0 - 0.5).clamp(0.0, (wc - 1) as f32);
        let fy = ((v + 0.5) / 8.0 - 0.5).clamp(0.0, (hc - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(wc - 1), (y0 + 1).min(hc - 1));
        let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
        let mut v256 = [0f32; 256];
        let mut nn = 0.0;
        for (c, slot) in v256.iter_mut().enumerate() {
            let b = c * cells;
            let top = desc[b + y0 * wc + x0] * (1.0 - ax) + desc[b + y0 * wc + x1] * ax;
            let bot = desc[b + y1 * wc + x0] * (1.0 - ax) + desc[b + y1 * wc + x1] * ax;
            *slot = top * (1.0 - ay) + bot * ay;
            nn += *slot * *slot;
        }
        let nn = nn.sqrt().max(1e-6);
        d.extend(v256.iter().map(|x| x / nn));
        kp.push((u, v));
    }
    Feat { kp, desc: d }
}

/// 상호 최근접 + 유사도 문턱: (a 번호, b 번호)
pub fn match_mutual(a: &Feat, b: &Feat, min_sim: f32) -> Vec<(usize, usize)> {
    let (na, nb) = (a.kp.len(), b.kp.len());
    let mut best_a = vec![(usize::MAX, -1.0f32); na];
    let mut best_b = vec![(usize::MAX, -1.0f32); nb];
    for i in 0..na {
        let da = &a.desc[i * 256..i * 256 + 256];
        for j in 0..nb {
            let db = &b.desc[j * 256..j * 256 + 256];
            let s: f32 = da.iter().zip(db).map(|(x, y)| x * y).sum();
            if s > best_a[i].1 {
                best_a[i] = (j, s);
            }
            if s > best_b[j].1 {
                best_b[j] = (i, s);
            }
        }
    }
    (0..na).filter(|&i| best_a[i].0 != usize::MAX && best_b[best_a[i].0].0 == i && best_a[i].1 > min_sim).map(|i| (i, best_a[i].0)).collect()
}

/// Horn: 카메라 점 q 를 세계 점 p 로 보내는 (R, t) — p ≈ R q + t
fn horn(p: &[V3], q: &[V3]) -> (M3, V3) {
    let n = p.len() as f64;
    let mp = [0, 1, 2].map(|k| p.iter().map(|x| x[k]).sum::<f64>() / n);
    let mq = [0, 1, 2].map(|k| q.iter().map(|x| x[k]).sum::<f64>() / n);
    let mut s = [[0.0; 3]; 3];
    for (a, b) in p.iter().zip(q) {
        let (a, b) = (sub(a, &mp), sub(b, &mq));
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += b[i] * a[j];
            }
        }
    }
    let (sxx, sxy, sxz, syx, syy, syz, szx, szy, szz) = (s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], s[2][0], s[2][1], s[2][2]);
    let nm = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    // 가장 큰 고유벡터: 이동 거듭제곱
    let shift = nm.iter().map(|r| r.iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max);
    let mut v = [1.0, 0.0, 0.0, 0.0];
    for _ in 0..60 {
        let mut w = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                w[i] += (nm[i][j] + if i == j { shift } else { 0.0 }) * v[j];
            }
        }
        let l = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v = w.map(|x| x / l);
    }
    let r = quat_to_m([v[1], v[2], v[3], v[0]]);
    let t = sub(&mp, &mv(&r, &mq));
    (r, t)
}

fn so3_exp(w: V3) -> M3 {
    let th = norm(&w);
    if th < 1e-12 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = w.map(|x| x / th);
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let kk = mm(&kx, &kx);
    let (s, c) = (th.sin(), th.cos());
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = if i == j { 1.0 } else { 0.0 } + s * kx[i][j] + (1.0 - c) * kk[i][j];
        }
    }
    r
}

fn reproj(r: &M3, t: &V3, x: &V3, k: [f64; 4]) -> Option<(f64, f64)> {
    let d = sub(x, t);
    let pc = [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2]);
    (pc[2] > 0.05).then(|| (k[0] * pc[0] / pc[2] + k[2], k[1] * pc[1] / pc[2] + k[3]))
}

/// 6×6 선형계 풀기(부분 피벗 가우스)
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let p = (c..6).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..6 {
            let f = a[r][c] / a[c][c];
            for k in c..6 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 6];
    for r in (0..6).rev() {
        let mut s = b[r];
        for k in r + 1..6 {
            s -= a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// 두 장면 기하(깊이 안 씀)로 상대 회전·이동 방향: 정규화 좌표 8점 RANSAC → 에센셜 행렬 → 네 해 중 앞쪽에 점이 많은 것.
/// 반환: (R, t̂, 정상 짝 번호들, 삼각측량 깊이(이전 장면, t̂ 단위)) — 이전 카메라 좌표 X0 를 현재로: X1 = R·X0 + t̂·s
pub fn relative_pose(x0: &[(f64, f64)], x1: &[(f64, f64)]) -> Option<(M3, V3, Vec<usize>, Vec<f64>)> {
    use nalgebra::{Matrix3, SMatrix, Vector3};
    let n = x0.len();
    if n < 20 {
        return None;
    }
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    let fit = |idx: &[usize]| -> Option<Matrix3<f64>> {
        let mut a = nalgebra::DMatrix::<f64>::zeros(idx.len().max(9), 9);
        for (r, &i) in idx.iter().enumerate() {
            let ((u0, v0), (u1, v1)) = (x0[i], x1[i]);
            let row = [u1 * u0, u1 * v0, u1, v1 * u0, v1 * v0, v1, u0, v0, 1.0];
            for c in 0..9 {
                a[(r, c)] = row[c];
            }
        }
        let svd = a.svd(false, true);
        let vt = svd.v_t?;
        let e = vt.row(vt.nrows() - 1);
        let em = Matrix3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
        let s = em.svd(true, true);
        let (u, vt) = (s.u?, s.v_t?);
        Some(u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * vt)
    };
    let sampson = |e: &Matrix3<f64>, i: usize| -> f64 {
        let (p0, p1) = (Vector3::new(x0[i].0, x0[i].1, 1.0), Vector3::new(x1[i].0, x1[i].1, 1.0));
        let (ep0, etp1) = (e * p0, e.transpose() * p1);
        let num = p1.dot(&ep0);
        num * num / (ep0[0] * ep0[0] + ep0[1] * ep0[1] + etp1[0] * etp1[0] + etp1[1] * etp1[1]).max(1e-18)
    };
    let thr = (1.5f64 / 530.0).powi(2); // 약 1.5 화소
    let (mut best, mut bin) = (None, vec![]);
    for _ in 0..300 {
        let mut s = [0usize; 8];
        for k in 0..8 {
            s[k] = rnd(n);
        }
        let Some(e) = fit(&s) else { continue };
        let inl: Vec<usize> = (0..n).filter(|&i| sampson(&e, i) < thr).collect();
        if inl.len() > bin.len() {
            bin = inl;
            best = Some(e);
        }
    }
    if bin.len() < 20 {
        return None;
    }
    let e = fit(&bin).or(best)?;
    let inl: Vec<usize> = (0..n).filter(|&i| sampson(&e, i) < thr).collect();
    let s = e.svd(true, true);
    let (mut u, mut vt) = (s.u?, s.v_t?);
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let cands = [(u * w * vt, u.column(2).into_owned()), (u * w * vt, -u.column(2).into_owned()),
                 (u * w.transpose() * vt, u.column(2).into_owned()), (u * w.transpose() * vt, -u.column(2).into_owned())];
    // 삼각측량(중점법)으로 앞쪽 점 수가 가장 많은 해
    let tri = |r: &Matrix3<f64>, t: &Vector3<f64>, i: usize| -> Option<f64> {
        let d0 = Vector3::new(x0[i].0, x0[i].1, 1.0);
        let d1 = Vector3::new(x1[i].0, x1[i].1, 1.0);
        // X1 = R X0 + t, X0 = z0 d0, X1 = z1 d1 → z0 R d0 - z1 d1 = -t
        let a = SMatrix::<f64, 3, 2>::from_columns(&[r * d0, -d1]);
        let ata = a.transpose() * a;
        let z = ata.try_inverse()? * (a.transpose() * (-t));
        (z[0] > 0.0 && z[1] > 0.0).then_some(z[0])
    };
    let mut bestc = None;
    let mut bestn = 0;
    for (r, t) in &cands {
        let c = inl.iter().filter(|&&i| tri(r, t, i).is_some()).count();
        if c > bestn {
            bestn = c;
            bestc = Some((*r, *t));
        }
    }
    let (r, t) = bestc?;
    let mut ids = vec![];
    let mut zs = vec![];
    for &i in &inl {
        if let Some(z) = tri(&r, &t, i) {
            ids.push(i);
            zs.push(z);
        }
    }
    let rm = [[r[(0, 0)], r[(0, 1)], r[(0, 2)]], [r[(1, 0)], r[(1, 1)], r[(1, 2)]], [r[(2, 0)], r[(2, 1)], r[(2, 2)]]];
    Some((rm, [t[0], t[1], t[2]], ids, zs))
}

/// world: 이전 장면 점(세계), cam: 같은 짝의 현재 장면 점(카메라, 깊이 있으면), uv: 현재 화소.
/// 반환: (R, t) 카메라→세계, 맞는 짝 수
pub fn estimate(world: &[V3], cam: &[Option<V3>], uv: &[(f64, f64)], k: [f64; 4], prior: (M3, V3)) -> Option<(M3, V3, usize)> {
    let n = world.len();
    if n < 12 {
        return None;
    }
    let idx3: Vec<usize> = (0..n).filter(|&i| cam[i].is_some()).collect();
    // RANSAC(3점 Horn) — 결정적 난수
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = |m: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as usize
    };
    let count = |r: &M3, t: &V3| -> usize { (0..n).filter(|&i| reproj(r, t, &world[i], k).is_some_and(|(u, v)| (u - uv[i].0).hypot(v - uv[i].1) < 4.0)).count() };
    let (mut br, mut bt) = prior;
    let mut bn = count(&br, &bt);
    if idx3.len() >= 3 {
        for _ in 0..200 {
            let s = [idx3[rnd(idx3.len())], idx3[rnd(idx3.len())], idx3[rnd(idx3.len())]];
            if s[0] == s[1] || s[1] == s[2] || s[0] == s[2] {
                continue;
            }
            let p: Vec<V3> = s.iter().map(|&i| world[i]).collect();
            let q: Vec<V3> = s.iter().map(|&i| cam[i].unwrap()).collect();
            let (r, t) = horn(&p, &q);
            let c = count(&r, &t);
            if c > bn {
                bn = c;
                br = r;
                bt = t;
            }
        }
    }
    // 가우스-뉴턴: 재투영 오차, 후버 가중
    let (mut r, mut t) = (br, bt);
    for _ in 0..12 {
        let mut h = [[0.0; 6]; 6];
        let mut g = [0.0; 6];
        let mut used = 0;
        for i in 0..n {
            let Some((u0, v0)) = reproj(&r, &t, &world[i], k) else { continue };
            let e = [u0 - uv[i].0, v0 - uv[i].1];
            let en = (e[0] * e[0] + e[1] * e[1]).sqrt();
            if en > 12.0 {
                continue;
            }
            let wgt = if en < 2.0 { 1.0 } else { 2.0 / en };
            let mut jac = [[0.0; 6]; 2];
            for p in 0..6 {
                let mut d = [0.0; 6];
                d[p] = 1e-6;
                let r2 = mm(&so3_exp([d[0], d[1], d[2]]), &r);
                let t2 = add(&t, &[d[3], d[4], d[5]]);
                if let Some((u1, v1)) = reproj(&r2, &t2, &world[i], k) {
                    jac[0][p] = (u1 - u0) / 1e-6;
                    jac[1][p] = (v1 - v0) / 1e-6;
                }
            }
            for a in 0..6 {
                for b in 0..6 {
                    h[a][b] += wgt * (jac[0][a] * jac[0][b] + jac[1][a] * jac[1][b]);
                }
                g[a] -= wgt * (jac[0][a] * e[0] + jac[1][a] * e[1]);
            }
            used += 1;
        }
        if used < 8 {
            return None;
        }
        for (a, row) in h.iter_mut().enumerate() {
            row[a] += 1e-6;
        }
        let Some(dx) = solve6(h, g) else { break };
        r = mm(&so3_exp([dx[0], dx[1], dx[2]]), &r);
        t = add(&t, &[dx[3], dx[4], dx[5]]);
        if dx.iter().map(|x| x * x).sum::<f64>() < 1e-12 {
            break;
        }
    }
    let inl = count(&r, &t);
    (inl >= 12).then_some((r, t, inl))
}
