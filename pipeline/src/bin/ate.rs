//! 경로 정렬 오차: 추정 경로와 정답 경로(둘 다 TUM 형식 "t x y z qx qy qz qw")를 시각으로 짝지어(0.02 초 안)
//! 회전·이동으로 가장 잘 겹치게 맞춘 뒤(Horn 사원수 방법) 위치 오차 RMSE.
//! 자세도 있으면 맞춘 뒤 카메라 자세 차이(도)의 중앙값·최대도 낸다: 카메라 축 약속(OpenCV 광학 축)이 같으면 몇 도, 다르면 90° 안팎.
//!   ate <추정> <정답> [짝 시각 차(초, 기본 0.02)] [정렬한 추정 경로를 쓸 파일]
//! 네 번째 인자를 주면 추정 경로를 정답 좌표계로 옮겨(위치·자세 모두) TUM 형식으로 쓴다.
#[path = "../horn.rs"]
mod horn;
use horn::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
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
    println!("{{\"pairs\":{},\"ate_rmse_m\":{:.4},\"max_m\":{:.4}{rot_s}}}", pairs.len(), (se / n).sqrt(), mx);
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
