//! 광류 위치 추적의 국소 묶음 조정(키프레임 창). RTAB-Map OdometryF2M 의 국소 지도 묶음 조정(OdomF2M/BundleAdjustment)과
//! 같은 생각: 최근 키프레임 몇 장의 위치와 그 장들이 본 점을 함께 다듬는다. RGB-D 깊이는 RTAB-Map g2o 처럼 가짜 기선(g2o/Baseline 0.075 m)의
//! 오른쪽 눈 화소(u − bf/z)로 넣어 크기를 묶는다. 가장 오래된 키프레임은 고정.
use nalgebra::{DMatrix, DVector, Matrix3, Matrix6x3, Rotation3, Vector3, Vector6};
use std::collections::HashMap;

/// 키프레임 하나: 세계 → 광학 축(tcw), 관측 (점 번호, u, v, 깊이 m 또는 NaN)
pub struct Kf {
    pub stamp: f64,
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
    pub obs: Vec<(u64, [f32; 2], f32)>,
}

const BASELINE: f64 = 0.075;
const CHI2_3: f64 = 7.815;
const CHI2_2: f64 = 5.991;

/// 관측 한 개의 잔차와 야코비안(포즈 6: [ω, v] 왼쪽 곱 · 점 3)
fn eval(r: &Rotation3<f64>, t: &Vector3<f64>, p: &Vector3<f64>, k: &[f64; 4], uv: [f32; 2], d: f32) -> Option<(usize, [f64; 3], [Vector6<f64>; 3], [Vector3<f64>; 3])> {
    let c = r * p + t;
    if c[2] <= 0.05 {
        return None;
    }
    let iz = 1.0 / c[2];
    let bf = k[0] * BASELINE;
    let u = k[0] * c[0] * iz + k[2];
    let v = k[1] * c[1] * iz + k[3];
    let mut e = [u - uv[0] as f64, v - uv[1] as f64, 0.0];
    let du = Vector3::new(k[0] * iz, 0.0, -k[0] * c[0] * iz * iz);
    let dv = Vector3::new(0.0, k[1] * iz, -k[1] * c[1] * iz * iz);
    let dr = Vector3::new(k[0] * iz, 0.0, -k[0] * c[0] * iz * iz + bf * iz * iz);
    let jc = |d: &Vector3<f64>| -> Vector6<f64> {
        let a = c.cross(d);
        Vector6::new(a[0], a[1], a[2], d[0], d[1], d[2])
    };
    let rm = r.matrix();
    let jp = |d: &Vector3<f64>| -> Vector3<f64> { rm.transpose() * d };
    let n = if d.is_finite() && d > 0.0 {
        e[2] = (u - bf * iz) - (uv[0] as f64 - bf / d as f64);
        3
    } else {
        2
    };
    Some((n, e, [jc(&du), jc(&dv), jc(&dr)], [jp(&du), jp(&dv), jp(&dr)]))
}

/// 창 전체를 다듬음. kfs[0] 고정. pts: 점 번호 → 세계 위치(바뀜). 돌려줌: 바깥값(카이제곱 넘는) 관측을 뺀 수
pub fn solve(kfs: &mut [Kf], pts: &mut HashMap<u64, Vector3<f64>>, k: &[f64; 4], iters: usize) -> usize {
    let nc = kfs.len();
    if nc < 2 {
        return 0;
    }
    let nf = nc - 1;
    // 점별 관측 목록
    let mut pidx: HashMap<u64, usize> = HashMap::new();
    let mut plist: Vec<u64> = Vec::new();
    let mut pobs: Vec<Vec<(usize, usize)>> = Vec::new();
    for (ci, kf) in kfs.iter().enumerate() {
        for (oi, o) in kf.obs.iter().enumerate() {
            if !pts.contains_key(&o.0) {
                continue;
            }
            let j = *pidx.entry(o.0).or_insert_with(|| {
                plist.push(o.0);
                pobs.push(Vec::new());
                plist.len() - 1
            });
            pobs[j].push((ci, oi));
        }
    }
    let huber = |n: usize, s2: f64| -> f64 {
        let th = if n == 3 { CHI2_3 } else { CHI2_2 };
        if s2 <= th { 1.0 } else { th.sqrt() / s2.sqrt() }
    };
    let cost = |kfs: &[Kf], pts: &HashMap<u64, Vector3<f64>>| -> f64 {
        let mut s = 0.0;
        for kf in kfs {
            for o in &kf.obs {
                let Some(p) = pts.get(&o.0) else { continue };
                if let Some((n, e, _, _)) = eval(&kf.r, &kf.t, p, k, o.1, o.2) {
                    let s2: f64 = e[..n].iter().map(|x| x * x).sum();
                    let th = if n == 3 { CHI2_3 } else { CHI2_2 };
                    s += if s2 <= th { s2 } else { 2.0 * th.sqrt() * s2.sqrt() - th };
                }
            }
        }
        s
    };
    let mut lambda = 1e-3;
    let mut cur = cost(kfs, pts);
    for _ in 0..iters {
        let mut hcc = DMatrix::<f64>::zeros(6 * nf, 6 * nf);
        let mut bc = DVector::<f64>::zeros(6 * nf);
        let mut hpp_inv: Vec<Matrix3<f64>> = Vec::with_capacity(plist.len());
        let mut bp: Vec<Vector3<f64>> = Vec::with_capacity(plist.len());
        let mut hcp: Vec<Vec<(usize, Matrix6x3<f64>)>> = Vec::with_capacity(plist.len());
        for (j, id) in plist.iter().enumerate() {
            let p = pts[id];
            let mut h = Matrix3::<f64>::zeros();
            let mut b = Vector3::<f64>::zeros();
            let mut blocks: HashMap<usize, Matrix6x3<f64>> = HashMap::new();
            for &(ci, oi) in &pobs[j] {
                let kf = &kfs[ci];
                let o = &kf.obs[oi];
                let Some((n, e, jcs, jps)) = eval(&kf.r, &kf.t, &p, k, o.1, o.2) else { continue };
                let s2: f64 = e[..n].iter().map(|x| x * x).sum();
                let w = huber(n, s2);
                for r in 0..n {
                    h += w * jps[r] * jps[r].transpose();
                    b -= w * jps[r] * e[r];
                    if ci > 0 {
                        let c = ci - 1;
                        let mut blk = hcc.fixed_view_mut::<6, 6>(6 * c, 6 * c);
                        blk += w * jcs[r] * jcs[r].transpose();
                        let mut bb = bc.fixed_rows_mut::<6>(6 * c);
                        bb -= w * jcs[r] * e[r];
                        *blocks.entry(c).or_insert_with(Matrix6x3::zeros) += w * jcs[r] * jps[r].transpose();
                    }
                }
            }
            for i in 0..3 {
                h[(i, i)] *= 1.0 + lambda;
                h[(i, i)] += 1e-9;
            }
            let hi = h.try_inverse().unwrap_or_else(Matrix3::zeros);
            hpp_inv.push(hi);
            bp.push(b);
            hcp.push(blocks.into_iter().collect());
        }
        // Schur
        let mut s = hcc.clone();
        for i in 0..6 * nf {
            s[(i, i)] *= 1.0 + lambda;
            s[(i, i)] += 1e-9;
        }
        let mut rhs = bc.clone();
        for j in 0..plist.len() {
            let hi = &hpp_inv[j];
            for (a, wa) in &hcp[j] {
                let t = wa * hi;
                let mut rr = rhs.fixed_rows_mut::<6>(6 * a);
                rr -= t * bp[j];
                for (c, wc) in &hcp[j] {
                    let mut blk = s.fixed_view_mut::<6, 6>(6 * a, 6 * c);
                    blk -= t * wc.transpose();
                }
            }
        }
        let Some(dc) = s.cholesky().map(|ch| ch.solve(&rhs)) else {
            lambda *= 10.0;
            continue;
        };
        // 시도
        let saved: Vec<(Rotation3<f64>, Vector3<f64>)> = kfs.iter().map(|f| (f.r, f.t)).collect();
        let saved_p: Vec<Vector3<f64>> = plist.iter().map(|id| pts[id]).collect();
        for c in 0..nf {
            let d = dc.fixed_rows::<6>(6 * c);
            let dr = Rotation3::new(Vector3::new(d[0], d[1], d[2]));
            let kf = &mut kfs[c + 1];
            kf.r = dr * kf.r;
            kf.t = dr * kf.t + Vector3::new(d[3], d[4], d[5]);
        }
        for (j, id) in plist.iter().enumerate() {
            let mut b = bp[j];
            for (c, wc) in &hcp[j] {
                b -= wc.transpose() * dc.fixed_rows::<6>(6 * c);
            }
            *pts.get_mut(id).unwrap() += hpp_inv[j] * b;
        }
        let nw = cost(kfs, pts);
        if nw < cur {
            let done = (cur - nw) < 1e-6 * cur;
            cur = nw;
            lambda = (lambda * 0.3).max(1e-7);
            if done {
                break;
            }
        } else {
            for (f, s) in kfs.iter_mut().zip(saved) {
                f.r = s.0;
                f.t = s.1;
            }
            for (id, p) in plist.iter().zip(saved_p) {
                pts.insert(*id, p);
            }
            lambda *= 10.0;
        }
    }
    // 바깥값 관측 뺌
    let mut removed = 0;
    for kf in kfs.iter_mut() {
        let (r, t) = (kf.r, kf.t);
        kf.obs.retain(|o| {
            let Some(p) = pts.get(&o.0) else { return false };
            match eval(&r, &t, p, k, o.1, o.2) {
                Some((n, e, _, _)) => {
                    let s2: f64 = e[..n].iter().map(|x| x * x).sum();
                    let ok = s2 <= if n == 3 { CHI2_3 } else { CHI2_2 } * 4.0;
                    if !ok {
                        removed += 1;
                    }
                    ok
                }
                None => {
                    removed += 1;
                    false
                }
            }
        });
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ba_recovers_pose_noise() {
        let k = [525.0, 525.0, 320.0, 240.0];
        let mut pts_true = HashMap::new();
        let mut s = 1u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for i in 0..200u64 {
            pts_true.insert(i, Vector3::new(rnd() * 3.0, rnd() * 2.0, 2.0 + rnd() * 2.0));
        }
        let poses: Vec<(Rotation3<f64>, Vector3<f64>)> = (0..4).map(|i| (Rotation3::new(Vector3::new(0.0, 0.02 * i as f64, 0.0)), Vector3::new(-0.05 * i as f64, 0.0, 0.0))).collect();
        let mut kfs: Vec<Kf> = poses
            .iter()
            .map(|(r, t)| {
                let obs = pts_true
                    .iter()
                    .filter_map(|(id, p)| {
                        let c = r * p + t;
                        Some((*id, [(k[0] * c[0] / c[2] + k[2]) as f32, (k[1] * c[1] / c[2] + k[3]) as f32], c[2] as f32))
                    })
                    .collect();
                Kf { stamp: 0.0, r: *r, t: *t, obs }
            })
            .collect();
        // 흔들기
        kfs[3].t += Vector3::new(0.03, -0.02, 0.01);
        kfs[2].r = Rotation3::new(Vector3::new(0.01, 0.0, 0.0)) * kfs[2].r;
        let mut pts: HashMap<u64, Vector3<f64>> = pts_true.iter().map(|(i, p)| (*i, p + Vector3::new(rnd() * 0.02, rnd() * 0.02, rnd() * 0.02))).collect();
        solve(&mut kfs, &mut pts, &k, 20);
        for (kf, (_, t)) in kfs.iter().zip(&poses) {
            assert!((kf.t - t).norm() < 1e-3, "{:?} vs {:?}", kf.t, t);
        }
    }
}
