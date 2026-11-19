//! RTAB-Map 의 cv3::solvePnPRansac(corelib/src/opencv/solvepnp.cpp, OpenCV 3 판을 복사한 것) 을 옮긴 것.
//! util3d::solvePnPRansac 가 이것을 부르고, Vis/PnPRefineIterations=0(g2o 빌드 기본값) 이라 그 뒤 다듬기는 없다.
//!
//! 원본에서 그대로 따른 특이점:
//! - 안쪽 점 판정은 재투영 거리(제곱 아님) <= 문턱값² (2 화소 → 4 화소)
//! - 난수는 cv::RNG((uint64)-1), 매번 같은 순서
//! - RANSAC 뒤 안쪽 점으로 다시 푸는 solvePnP(ITERATIVE) 의 결과는 쓰이지 않는다(가장 좋았던 EPnP 모델을 그대로 내보냄).
//!   그 호출은 늘 성공하므로 결과에 영향이 없어 여기서는 부르지 않는다(같은 비트, 더 빠름).
use super::cvmath::{cv_round, rodrigues_m2v, rodrigues_v2m, undistort_one, CvRng, K};
use super::epnp::{Epnp, NP};

/// RANSACUpdateNumIters
fn update_num_iters(p: f64, ep: f64, model_points: i32, max_iters: i32) -> i32 {
    let p = p.max(0.0).min(1.0);
    let ep = ep.max(0.0).min(1.0);
    let num = (1.0 - p).max(f64::MIN_POSITIVE);
    let denom = 1.0 - (1.0 - ep).powf(model_points as f64);
    if denom < f64::MIN_POSITIVE {
        return 0;
    }
    let num = num.ln();
    let denom = denom.ln();
    if denom >= 0.0 || -num >= max_iters as f64 * (-denom) {
        max_iters
    } else {
        cv_round(num / denom)
    }
}

/// 한 모델(rvec, tvec) 로 모든 점의 재투영 거리(float) → 안쪽 점 수
fn find_inliers(op: &[[f32; 3]], ip: &[[f32; 2]], rvec: &[f64; 3], tvec: &[f64; 3], k: &K, thresh: f64, mask: &mut [u8]) -> usize {
    let r = rodrigues_v2m(*rvec);
    let t = (thresh * thresh) as f32;
    let mut nz = 0;
    for i in 0..op.len() {
        // project_one 의 왜곡 0 경로를 줄인 것: 유한한 값에선 1+0·r2… = 1, x·1·1+0·a1… = x 라 결과 비트가 같고,
        // x 가 무한·NaN 이면 둘 다 안쪽 점이 아니다(err 가 inf/NaN)
        let p = op[i];
        let (xx, yy, zz) = (p[0] as f64, p[1] as f64, p[2] as f64);
        let x = r[0] * xx + r[1] * yy + r[2] * zz + tvec[0];
        let y = r[3] * xx + r[4] * yy + r[5] * zz + tvec[1];
        let z = r[6] * xx + r[7] * yy + r[8] * zz + tvec[2];
        let z = if z != 0.0 { 1.0 / z } else { 1.0 };
        let (px, py) = ((x * z) * k.fx + k.cx, (y * z) * k.fy + k.cy);
        let dx = ip[i][0] - px as f32;
        let dy = ip[i][1] - py as f32;
        let err = ((dx as f64) * (dx as f64) + (dy as f64) * (dy as f64)).sqrt() as f32;
        let f = (err <= t) as u8;
        mask[i] = f;
        nz += f as usize;
    }
    nz
}

/// 6 점 EPnP(solvePnPGeneric 의 SOLVEPNP_EPNP 경로)
fn epnp6(op: &[[f32; 3]; NP], ip: &[[f32; 2]; NP], k: &K) -> ([f64; 3], [f64; 3]) {
    let mut up = [[0f32; 2]; NP];
    for i in 0..NP {
        let (x, y) = undistort_one(ip[i][0], ip[i][1], k);
        up[i] = [x, y];
    }
    let mut e = Epnp::new(k.fx, k.fy, k.cx, k.cy, op, &up);
    let (r, t) = e.compute_pose();
    (rodrigues_m2v(&r), t)
}

/// 결과: (rvec, tvec, 안쪽 점 번호들). 실패면 None
pub struct PnpOut {
    pub rvec: [f64; 3],
    pub tvec: [f64; 3],
    pub inliers: Vec<usize>,
}

/// cv3::solvePnPRansac(useExtrinsicGuess=true, confidence 0.99, flags 는 RANSAC 안에서 EPnP)
pub fn solve_pnp_ransac(op: &[[f32; 3]], ip: &[[f32; 2]], k: &K, iterations: i32, reproj_error: f32, mask_buf: &mut Vec<u8>, best_buf: &mut Vec<u8>) -> Option<PnpOut> {
    let count = op.len();
    let model_points = NP;
    let threshold = reproj_error as f64;
    let confidence = 0.99f64;
    if count < model_points {
        return None;
    }
    mask_buf.clear();
    mask_buf.resize(count, 0);
    best_buf.clear();
    best_buf.resize(count, 0);
    let mut niters = iterations.max(1);
    let mut rng = CvRng::new(u64::MAX);
    let mut max_good = 0usize;
    let mut best: Option<([f64; 3], [f64; 3])> = None;
    if count == model_points {
        let mut o6 = [[0f32; 3]; NP];
        let mut i6 = [[0f32; 2]; NP];
        o6.copy_from_slice(op);
        i6.copy_from_slice(ip);
        let (r, t) = epnp6(&o6, &i6, k);
        return Some(PnpOut { rvec: r, tvec: t, inliers: (0..count).collect() });
    }
    // 후보(표본 6 개 → EPnP → 안쪽 점 세기)는 서로 독립이다. 표본 순서는 난수에만 달려 있어 미리 뽑을 수 있으므로
    // 묶음으로 나란히 계산하고, 결과는 원본처럼 하나씩 차례로 반영하며 그때의 niters 에서 멈춘다(같은 비트).
    // 넘쳐 계산한 후보는 버린다.
    let mut iter = 0i32;
    let batch = std::env::var("SLAM_PNP_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).max(1);
    let mut subsets: Vec<[usize; NP]> = Vec::with_capacity(batch);
    while iter < niters {
        let nb = ((niters - iter) as usize).min(batch);
        subsets.clear();
        for _ in 0..nb {
            // getSubset: 서로 다른 6 개
            let mut idx = [0usize; NP];
            let mut i = 0;
            while i < model_points {
                loop {
                    let v = rng.uniform(0, count as i32) as usize;
                    idx[i] = v;
                    if !idx[..i].contains(&v) {
                        break;
                    }
                }
                i += 1;
            }
            subsets.push(idx);
        }
        let eval = |idx: &[usize; NP]| {
            let mut o6 = [[0f32; 3]; NP];
            let mut i6 = [[0f32; 2]; NP];
            for j in 0..NP {
                o6[j] = op[idx[j]];
                i6[j] = ip[idx[j]];
            }
            let (r, t) = epnp6(&o6, &i6, k);
            let mut m = Vec::new();
            m.resize(count, 0u8);
            let good = find_inliers(op, ip, &r, &t, k, threshold, &mut m);
            (r, t, good, m)
        };
        let res: Vec<_> = if nb > 1 && count >= 64 {
            use rayon::prelude::*;
            subsets.par_iter().map(eval).collect()
        } else {
            subsets.iter().map(eval).collect()
        };
        for (r, t, good, m) in res {
            if iter >= niters {
                break;
            }
            if good > max_good.max(model_points - 1) {
                *best_buf = m;
                best = Some((r, t));
                max_good = good;
                niters = update_num_iters(confidence, (count - good) as f64 / count as f64, model_points as i32, niters);
            }
            iter += 1;
        }
    }
    let _ = &mask_buf;
    let (r, t) = best?;
    if max_good == 0 {
        return None;
    }
    let inliers = best_buf.iter().enumerate().filter(|(_, m)| **m != 0).map(|(i, _)| i).collect();
    Some(PnpOut { rvec: r, tvec: t, inliers })
}
