//! 경로 맞추기(ate): TUM 위치 파일 읽기, 시각 짝, Horn 사원수 방법 회전·이동.
#![allow(dead_code)]

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

/// TUM 형식 "t x y z qx qy qz qw" → (시각, 위치, 사원수(x, y, z, w))
pub fn read_tum(p: &str) -> Vec<(f64, V3, Option<[f64; 4]>)> {
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    common::tum::rows(&text).filter(|v| v.len() >= 4).map(|v| (v[0], [v[1], v[2], v[3]], (v.len() >= 8).then(|| [v[4], v[5], v[6], v[7]]))).collect()
}

/// 정답 시각마다 tol 초 안에서 가장 가까운 추정: (추정 번호, 정답 번호)
pub fn pair_by_time(est: &[(f64, V3, Option<[f64; 4]>)], gt: &[(f64, V3, Option<[f64; 4]>)], tol: f64) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for (gi, (t, _, _)) in gt.iter().enumerate() {
        let i = est.partition_point(|e| e.0 < *t);
        let best = [i.wrapping_sub(1), i].into_iter().filter(|&j| j < est.len()).min_by(|&x, &y| (est[x].0 - t).abs().partial_cmp(&(est[y].0 - t).abs()).unwrap());
        if let Some(j) = best {
            if (est[j].0 - t).abs() <= tol {
                out.push((j, gi));
            }
        }
    }
    out
}

/// e → g 를 가장 잘 겹치는 회전 R 와 두 무게중심: g ≈ R (e - ce) + cg
pub fn horn(pairs: &[(V3, V3)]) -> (M3, V3, V3) {
    let n = pairs.len() as f64;
    let mut ce = [0.0; 3];
    let mut cg = [0.0; 3];
    for (e, g) in pairs {
        for k in 0..3 {
            ce[k] += e[k] / n;
            cg[k] += g[k] / n;
        }
    }
    // 교차 공분산 S = Σ (e - ce)(g - cg)ᵀ, Horn 의 4×4 N 의 가장 큰 고유벡터 = 회전(e → g)
    let mut s = [[0.0; 3]; 3];
    for (e, g) in pairs {
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
    (r, ce, cg)
}

/// x' = R (x - ce) + cg
pub fn apply(r: &M3, ce: &V3, cg: &V3, x: &V3) -> V3 {
    [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * (x[j] - ce[j])).sum::<f64>() + cg[i])
}

/// 사원수(x, y, z, w) → 회전 행렬
pub use common::quat::quat_m;

pub fn mm(a: &M3, b: &M3) -> M3 {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum::<f64>()))
}

/// 두 회전 사이 각(도)
pub fn angle_deg(a: &M3, b: &M3) -> f64 {
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a[k][i] * b[k][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}
