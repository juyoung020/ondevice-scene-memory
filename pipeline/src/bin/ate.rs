//! 경로 정렬 오차: 추정 경로와 정답 경로(둘 다 TUM 형식 "t x y z qx qy qz qw")를 시각으로 짝지어(0.02 초 안)
//! 회전·이동으로 가장 잘 겹치게 맞춘 뒤(Horn 사원수 방법) 위치 오차 RMSE. 자세 축 약속이 서로 달라도 위치만 본다.
//!   ate <추정> <정답> [짝 시각 차(초, 기본 0.02)]
fn read(p: &str) -> Vec<(f64, [f64; 3])> {
    std::fs::read_to_string(p)
        .expect("read")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let v: Vec<f64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            (v.len() >= 4).then(|| (v[0], [v[1], v[2], v[3]]))
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (est, gt) = (read(&a[1]), read(&a[2]));
    let tol: f64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    // 짝: 정답 시각마다 가장 가까운 추정
    let mut pairs = vec![];
    for (t, g) in &gt {
        let i = est.partition_point(|e| e.0 < *t);
        let best = [i.wrapping_sub(1), i].into_iter().filter(|&j| j < est.len()).min_by(|&x, &y| (est[x].0 - t).abs().partial_cmp(&(est[y].0 - t).abs()).unwrap());
        if let Some(j) = best {
            if (est[j].0 - t).abs() <= tol {
                pairs.push((est[j].1, *g));
            }
        }
    }
    let n = pairs.len() as f64;
    if pairs.len() < 3 {
        println!("{{\"pairs\":{}}}", pairs.len());
        return;
    }
    let mean = |f: &dyn Fn(&([f64; 3], [f64; 3])) -> [f64; 3]| {
        let mut m = [0.0; 3];
        for p in &pairs {
            let v = f(p);
            for k in 0..3 {
                m[k] += v[k] / n;
            }
        }
        m
    };
    let (ce, cg) = (mean(&|p| p.0), mean(&|p| p.1));
    // 교차 공분산 S = Σ (e - ce)(g - cg)ᵀ, Horn 의 4×4 N 의 가장 큰 고유벡터 = 회전(e → g)
    let mut s = [[0.0; 3]; 3];
    for (e, g) in &pairs {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += (e[i] - ce[i]) * (g[j] - cg[j]);
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
    let shift: f64 = nm.iter().flatten().map(|x| x.abs()).sum();
    let mut q = [1.0, 0.0, 0.0, 0.0];
    for _ in 0..500 {
        let mut nq = [0.0; 4];
        for i in 0..4 {
            nq[i] = (0..4).map(|j| nm[i][j] * q[j]).sum::<f64>() + shift * q[i];
        }
        let l = nq.iter().map(|x| x * x).sum::<f64>().sqrt();
        q = nq.map(|x| x / l);
    }
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    let mut se = 0.0;
    let mut mx: f64 = 0.0;
    for (e, g) in &pairs {
        let d = [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * (e[j] - ce[j])).sum::<f64>() + cg[i] - g[i]);
        let e2 = d.iter().map(|v| v * v).sum::<f64>();
        se += e2;
        mx = mx.max(e2.sqrt());
    }
    println!("{{\"pairs\":{},\"ate_rmse_m\":{:.4},\"max_m\":{:.4}}}", pairs.len(), (se / n).sqrt(), mx);
}
