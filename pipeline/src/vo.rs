//! 카메라만으로 포즈(시각 오도메트리): SuperPoint 특징점(NPU) → 이전 키프레임과 기술자 상호 최근접 짝 →
//! 기준 장면 랜드마크(여러 장면 관측 평균)로 3D(세계) → 현재 장면 깊이로 3D(카메라) 3점 RANSAC(Horn 쿼터니언 정렬)로 초기값 →
//! 맞는 짝들의 재투영 오차를 가우스-뉴턴(후버 가중)으로 줄여 현재 카메라 포즈.
use crate::geom::*;

#[derive(Clone)]
pub struct Feat {
    pub kp: Vec<(f32, f32)>, // 화소 (u, v)
    pub desc: Vec<f32>,      // kp 마다 256, L2 정규화
}

/// SuperPoint 출력(semi [65,Hc,Wc], desc [256,Hc,Wc]) → 칸마다 가장 센 점 하나, 점수 상위 n 개
#[cfg_attr(not(feature = "dxrt"), allow(dead_code))]
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

/// 256 차원 내적(8 갈래 누산 → 컴파일러가 SIMD 로)
#[inline]
fn dot256(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    acc.iter().sum()
}

/// 상호 최근접 + 유사도 문턱: (a 번호, b 번호). a 의 행을 코어마다 나눠 계산하고 b 쪽 최댓값은 합친다.
pub fn match_mutual(a: &Feat, b: &Feat, min_sim: f32) -> Vec<(usize, usize)> {
    let (na, nb) = (a.kp.len(), b.kp.len());
    if na == 0 || nb == 0 {
        return vec![];
    }
    let nth = std::thread::available_parallelism().map_or(4, |x| x.get()).min(8);
    let per = na.div_ceil(nth).max(1);
    let parts: Vec<(Vec<(usize, f32)>, Vec<(usize, f32)>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..na)
            .step_by(per)
            .map(|s0| {
                sc.spawn(move || {
                    let e0 = (s0 + per).min(na);
                    let mut ba = vec![(usize::MAX, -1.0f32); e0 - s0];
                    let mut bb = vec![(usize::MAX, -1.0f32); nb];
                    for i in s0..e0 {
                        let da = &a.desc[i * 256..i * 256 + 256];
                        for j in 0..nb {
                            let s = dot256(da, &b.desc[j * 256..j * 256 + 256]);
                            if s > ba[i - s0].1 {
                                ba[i - s0] = (j, s);
                            }
                            if s > bb[j].1 {
                                bb[j] = (i, s);
                            }
                        }
                    }
                    (ba, bb)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut best_a = Vec::with_capacity(na);
    let mut best_b = vec![(usize::MAX, -1.0f32); nb];
    for (ba, bb) in parts {
        best_a.extend(ba);
        for (j, x) in bb.into_iter().enumerate() {
            if x.1 > best_b[j].1 {
                best_b[j] = x;
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

pub(crate) fn so3_exp(w: V3) -> M3 {
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
pub(crate) fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
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
