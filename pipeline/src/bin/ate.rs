//! 경로 정렬 오차: 추정 경로와 정답 경로(둘 다 TUM 형식 "t x y z qx qy qz qw")를 시각으로 짝지어(0.02 초 안)
//! 회전·이동으로 가장 잘 겹치게 맞춘 뒤(Horn 사원수 방법) 위치 오차 RMSE.
//! 자세도 있으면 맞춘 뒤 카메라 자세 차이(도)의 중앙값·최대도 낸다: 카메라 축 약속(OpenCV 광학 축)이 같으면 몇 도, 다르면 90° 안팎.
//!   ate <추정> <정답> [짝 시각 차(초, 기본 0.02)] [정렬한 추정 경로를 쓸 파일]
//! 네 번째 인자를 주면 추정 경로를 정답 좌표계로 옮겨(위치·자세 모두) TUM 형식으로 쓴다.
//!   ate --lc <LC_LOG 파일> <정답>
//! 신경망 루프 기록(검증해 본 후보마다 한 줄, slam/src/lcnet.rs)의 상대 자세를 정답 상대 자세와 견줌: 줄마다 특징 자세 · 깊이 ICP 자세의 오차(m, °).
#[path = "../horn.rs"]
mod horn;
use horn::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).is_some_and(|x| x == "--lc") {
        return lc(&a[2], &a[3]);
    }
    let (est, gt) = (read_tum(&a[1]), read_tum(&a[2]));
    let tol: f64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let idx = pair_by_time(&est, &gt, tol);
    let pairs: Vec<(V3, V3)> = idx.iter().map(|&(i, j)| (est[i].1, gt[j].1)).collect();
    let n = pairs.len() as f64;
    if pairs.len() < 3 {
        println!("{{\"pairs\":{}}}", pairs.len());
        return;
    }
    let (r, ce, cg) = horn(&pairs);
    let mut se = 0.0;
    let mut mx: f64 = 0.0;
    for (e, g) in &pairs {
        let d = apply(&r, &ce, &cg, e);
        let e2 = (0..3).map(|k| (d[k] - g[k]).powi(2)).sum::<f64>();
        se += e2;
        mx = mx.max(e2.sqrt());
    }
    // 자세 차이: 맞춘 회전 R · 추정 자세 와 정답 자세 사이 각
    let mut rot: Vec<f64> = idx.iter().filter_map(|&(i, j)| Some(angle_deg(&mm(&r, &quat_m(est[i].2?)), &quat_m(gt[j].2?)))).collect();
    rot.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let rot_s = if rot.is_empty() { String::new() } else { format!(",\"rot_med_deg\":{:.2},\"rot_max_deg\":{:.2}", rot[rot.len() / 2], rot[rot.len() - 1]) };
    // 1 초 창 = 장면 사이 흔들림, 10 초 창 = 밀림(루프 보정이 든 창은 소수라 중앙값에는 위치 추적의 밀림만 남음)
    let rpe_s = rpe(&est, &gt, &idx, 1.0, "rpe") + &rpe(&est, &gt, &idx, 10.0, "rpe10");
    println!("{{\"pairs\":{},\"ate_rmse_m\":{:.4},\"max_m\":{:.4}{rot_s}{rpe_s}}}", pairs.len(), (se / n).sqrt(), mx);
    if let Some(out) = a.get(4) {
        // 정렬: x' = R (x - ce) + cg, 자세 R_cam' = R · R_cam
        let mut s = String::from("# aligned to ground truth\n");
        for (t, p, q) in &est {
            let Some(q) = q else { continue };
            let p = apply(&r, &ce, &cg, p);
            let m = mm(&r, &quat_m(*q));
            let tr = m[0][0] + m[1][1] + m[2][2];
            let qw2 = (1.0 + tr).max(1e-12).sqrt() / 2.0;
            let q = [(m[2][1] - m[1][2]) / (4.0 * qw2), (m[0][2] - m[2][0]) / (4.0 * qw2), (m[1][0] - m[0][1]) / (4.0 * qw2), qw2];
            s += &format!("{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}\n", t, p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
        }
        std::fs::write(out, s).unwrap();
    }
}

/// 상대 위치 오차(RPE): dt 초 떨어진 두 장면의 상대 움직임(추정 · 정답)을 견줌. 전체 맞춤 · 루프 보정과 상관없이 위치 추적이 밀리는 정도를 본다.
/// 돌려줌: JSON 조각(이동 RMSE cm · 회전 RMSE °, 이동 오차 합 / 정답 이동 합 %)
fn rpe(est: &[(f64, V3, Option<[f64; 4]>)], gt: &[(f64, V3, Option<[f64; 4]>)], idx: &[(usize, usize)], dt: f64, name: &str) -> String {
    let tr = |m: &M3| -> M3 { [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i])) };
    let mv = |m: &M3, v: &V3| -> V3 { [0, 1, 2].map(|i| (0..3).map(|k| m[i][k] * v[k]).sum()) };
    // 상대 움직임 a⁻¹ b = (Raᵀ Rb, Raᵀ (pb − pa))
    let rel = |a: &(M3, V3), b: &(M3, V3)| -> (M3, V3) {
        let rt = tr(&a.0);
        (mm(&rt, &b.0), mv(&rt, &[b.1[0] - a.1[0], b.1[1] - a.1[1], b.1[2] - a.1[2]]))
    };
    let pose = |v: &[(f64, V3, Option<[f64; 4]>)], i: usize| v[i].2.map(|q| (quat_m(q), v[i].1));
    let (mut st, mut sr, mut n, mut sum_e, mut sum_g) = (0.0, 0.0, 0usize, 0.0, 0.0);
    let (mut tes, mut angs): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    let mut k = 0usize;
    for a in 0..idx.len() {
        let ta = gt[idx[a].1].0;
        while k < idx.len() && gt[idx[k].1].0 < ta + dt {
            k += 1;
        }
        if k >= idx.len() {
            break;
        }
        if gt[idx[k].1].0 - (ta + dt) > 0.05 {
            continue;
        }
        let (Some(e0), Some(e1), Some(g0), Some(g1)) = (pose(est, idx[a].0), pose(est, idx[k].0), pose(gt, idx[a].1), pose(gt, idx[k].1)) else { continue };
        let (re, rg) = (rel(&e0, &e1), rel(&g0, &g1));
        let d = rel(&rg, &re);
        let te = (d.1[0] * d.1[0] + d.1[1] * d.1[1] + d.1[2] * d.1[2]).sqrt();
        let ang = angle_deg(&rg.0, &re.0);
        tes.push(te);
        angs.push(ang);
        st += te * te;
        sr += ang * ang;
        sum_e += te;
        sum_g += (rg.1[0] * rg.1[0] + rg.1[1] * rg.1[1] + rg.1[2] * rg.1[2]).sqrt();
        n += 1;
    }
    if n == 0 {
        return String::new();
    }
    tes.sort_by(|a, b| a.total_cmp(b));
    angs.sort_by(|a, b| a.total_cmp(b));
    format!(
        ",\"{name}_pairs\":{n},\"{name}_trans_cm\":{:.2},\"{name}_trans_med_cm\":{:.2},\"{name}_rot_deg\":{:.2},\"{name}_rot_med_deg\":{:.2},\"{name}_drift_pct\":{:.1}",
        (st / n as f64).sqrt() * 100.0,
        tes[n / 2] * 100.0,
        (sr / n as f64).sqrt(),
        angs[n / 2],
        100.0 * sum_e / sum_g.max(1e-9)
    )
}

/// 신경망 루프 기록 ↔ 정답: 질의 · 후보 장면의 정답 자세로 상대 자세(X_q = T X_d)를 만들어 기록의 두 자세와 견줌
fn lc(log: &str, gt: &str) {
    let gt = read_tum(gt);
    let tr = |m: &M3| -> M3 { [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i])) };
    let mv = |m: &M3, v: &V3| -> V3 { [0, 1, 2].map(|i| (0..3).map(|k| m[i][k] * v[k]).sum()) };
    let at = |t: f64| -> Option<(M3, V3)> {
        let i = gt.partition_point(|g| g.0 < t);
        let j = [i.saturating_sub(1), i.min(gt.len() - 1)].into_iter().min_by(|a, b| (gt[*a].0 - t).abs().total_cmp(&(gt[*b].0 - t).abs()))?;
        ((gt[j].0 - t).abs() < 0.03).then(|| (quat_m(gt[j].2.unwrap()), gt[j].1))
    };
    println!("# tq td idq idd score inl hyp feat_ok icp_ok ratio rms | 특징 오차 m ° | ICP 오차 m ° | 정답 이동 m 회전 °");
    for l in std::fs::read_to_string(log).unwrap().lines() {
        let c: String = l.chars().filter(|ch| !"[],".contains(*ch)).collect();
        let v: Vec<f64> = c.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if v.len() < 37 {
            continue;
        }
        let (Some(q), Some(d)) = (at(v[0]), at(v[1])) else { continue };
        // X_q = Rqᵀ (Rd X_d + pd − pq)
        let rt = tr(&q.0);
        let (gr, gt_t) = (mm(&rt, &d.0), mv(&rt, &[d.1[0] - q.1[0], d.1[1] - q.1[1], d.1[2] - q.1[2]]));
        let err = |o: usize| -> (f64, f64) {
            if v[o..o + 12].iter().any(|x| !x.is_finite()) {
                return (f64::NAN, f64::NAN);
            }
            let r: M3 = [0, 1, 2].map(|i| [0, 1, 2].map(|j| v[o + i * 4 + j]));
            let t = [v[o + 3], v[o + 7], v[o + 11]];
            let dt = [t[0] - gt_t[0], t[1] - gt_t[1], t[2] - gt_t[2]];
            ((dt[0] * dt[0] + dt[1] * dt[1] + dt[2] * dt[2]).sqrt(), angle_deg(&r, &gr))
        };
        let (ef, ei) = (err(13), err(25));
        let id: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        println!(
            "{:.3} {:.3} {} {} {:.3} {} {} {} {} {:.2} {:.4} | {:.3} {:.2} | {:.3} {:.2} | {:.2} {:.1}",
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], ef.0, ef.1, ei.0, ei.1,
            (gt_t[0] * gt_t[0] + gt_t[1] * gt_t[1] + gt_t[2] * gt_t[2]).sqrt(), angle_deg(&gr, &id)
        );
    }
}
