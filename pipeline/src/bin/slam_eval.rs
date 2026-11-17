//! 위치 · 지도 품질 지표(재생 벤치): 같은 녹화를 카메라 전용 · IMU 모드로 돌린 결과를 같은 잣대로 잰다.
//!   slam_eval --gt <정답 TUM> --poses <보낸 위치 TUM> [--final <최적화된 노드 TUM>] [--stats <통계 CSV>]
//!             [--imu <imu0.csv> [--calib <yaml>]] [--stream <장면 스트림>] [--every 2] [--tag 이름]
//! 지표
//!  - ATE: 보낸 위치(믿는 위치만) · 최종 노드 · 화면에 보낸 위치(스트림, 임시 · 맡김 포함)를 정답에 SE3(· Sim3)으로 맞춘 위치 오차 RMSE(cm), 위치를 낸 장면 비율
//!  - 놓침: 놓친 장면 수 · 놓친 횟수(놓침이 시작된 수), 새 지도 수(통계 CSV), 다시 붙기(새 지도를 연 뒤 믿는 위치가 다시 나오기까지 초, 끝까지 못 붙은 수)
//!  - 튐: 화면에 보낸 위치(스트림, 임시 · 맡김 포함)의 장면 사이 움직임이 IMU 회전(없으면 정답)과 3° + 회전의 20 % 넘게,
//!        또는 정답 이동과 3 cm + 이동의 20 % 넘게 다른 장면 수
//!  - 지도 두께: 스트림을 물체 기억과 같은 규칙으로 쌓은 최종 점 지도(놓침 · 위치만 장면 빼고, 노드 기준 상대 위치 × 최종 노드)를
//!        정답 지도에서 찾은 큰 평면 3 개(책상 · 바닥 · 벽) 위 10 cm 칸마다 칸 중앙값 ±5 cm 안 점의 평면 거리 p90 − p10, 그 중앙값(cm). 두 겹이면 커진다.
//!  - 겹 어긋남: 5 cm 칸마다 2 초 묶음별 평균 위치의 법선 방향 차(다시 본 면이 처음 면과 떨어진 정도). 2 cm 넘는 칸 비율 · 평균(cm), 정답 위치 지도 값과 비교
//!        같은 장면을 정답 위치로 쌓은 지도의 두께 · 두 겹 비율(깊이 잡음 · 물건 몫)을 함께 낸다
//!  - 지도 오차: 쌓인 점에서 같은 장면을 정답 위치로 쌓은 지도까지 거리 p50 · p90(cm), 5 cm 넘게 떨어진 점 비율(%, 두 겹 · 어긋난 면)
//!  - 처리 시간: 통계 CSV 의 위치 추적 · 지도 갱신 ms
#[path = "../horn.rs"]
mod horn;
use horn::*;
use std::collections::BTreeMap;

type T34 = [f64; 12];

fn arg<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}

fn t_r(t: &T34) -> M3 {
    [[t[0], t[1], t[2]], [t[4], t[5], t[6]], [t[8], t[9], t[10]]]
}
fn t_p(t: &T34) -> V3 {
    [t[3], t[7], t[11]]
}
fn t_of(r: &M3, p: &V3) -> T34 {
    [r[0][0], r[0][1], r[0][2], p[0], r[1][0], r[1][1], r[1][2], p[1], r[2][0], r[2][1], r[2][2], p[2]]
}
fn mt(a: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn mv(a: &M3, v: &V3) -> V3 {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}
fn tmul(a: &T34, b: &T34) -> T34 {
    let (ra, rb) = (t_r(a), t_r(b));
    let r = mm(&ra, &rb);
    let p = mv(&ra, &t_p(b));
    let pa = t_p(a);
    t_of(&r, &[p[0] + pa[0], p[1] + pa[1], p[2] + pa[2]])
}
fn tinv(a: &T34) -> T34 {
    let rt = mt(&t_r(a));
    let p = mv(&rt, &t_p(a));
    t_of(&rt, &[-p[0], -p[1], -p[2]])
}
fn exp3(w: V3) -> M3 {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let (a, b) = if th < 1e-9 { (1.0, 0.5) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    let k2 = mm(&k, &k);
    std::array::from_fn(|i| std::array::from_fn(|j| (i == j) as i32 as f64 + a * k[i][j] + b * k2[i][j]))
}
fn log3(m: &M3) -> V3 {
    let c = ((m[0][0] + m[1][1] + m[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let th = c.acos();
    let k = if th < 1e-7 { 0.5 } else { 0.5 * th / th.sin() };
    [k * (m[2][1] - m[1][2]), k * (m[0][2] - m[2][0]), k * (m[1][0] - m[0][1])]
}
fn norm(v: &V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// 정답 위치 보간(위치 선형, 회전 slerp). 앞뒤 정답이 0.05 초 넘게 떨어졌으면 None
struct Gt(Vec<(f64, V3, M3)>);
impl Gt {
    fn at(&self, t: f64) -> Option<T34> {
        let g = &self.0;
        let j = g.partition_point(|x| x.0 <= t);
        if j == 0 || j >= g.len() {
            return None;
        }
        let (a, b) = (&g[j - 1], &g[j]);
        if b.0 - a.0 > 0.05 {
            return None;
        }
        let u = (t - a.0) / (b.0 - a.0);
        let r = mm(&a.2, &exp3(log3(&mm(&mt(&a.2), &b.2)).map(|x| x * u)));
        Some(t_of(&r, &std::array::from_fn(|i| a.1[i] + u * (b.1[i] - a.1[i]))))
    }
}

/// SE3(scale = false) · Sim3 맞춤 후 RMSE(m): e → g
fn ate(pairs: &[(V3, V3)], sim: bool) -> (f64, M3, V3, V3, f64) {
    let (r, ce, cg) = horn(pairs);
    let s = if sim {
        let (mut num, mut den) = (0.0, 0.0);
        for (e, g) in pairs {
            let d = mv(&r, &[e[0] - ce[0], e[1] - ce[1], e[2] - ce[2]]);
            num += (0..3).map(|k| d[k] * (g[k] - cg[k])).sum::<f64>();
            den += (0..3).map(|k| (e[k] - ce[k]).powi(2)).sum::<f64>();
        }
        num / den
    } else {
        1.0
    };
    let mut se = 0.0;
    for (e, g) in pairs {
        let d = mv(&r, &[e[0] - ce[0], e[1] - ce[1], e[2] - ce[2]]);
        se += (0..3).map(|k| (s * d[k] + cg[k] - g[k]).powi(2)).sum::<f64>();
    }
    ((se / pairs.len() as f64).sqrt(), r, ce, cg, s)
}

/// 추정 위치 파일(TUM) → 정답 짝 → (SE3 cm, Sim3 cm, 척도, 짝 수)
fn ate_file(path: &str, gt: &Gt) -> Option<(f64, f64, f64, usize)> {
    let est = read_tum(path);
    let pairs: Vec<(V3, V3)> = est.iter().filter_map(|(t, p, _)| Some((*p, t_p(&gt.at(*t)?)))).collect();
    if pairs.len() < 10 {
        return None;
    }
    Some((ate(&pairs, false).0 * 100.0, ate(&pairs, true).0 * 100.0, ate(&pairs, true).4, pairs.len()))
}

/// IMU(Kalibr csv) 자이로 → 카메라 광학 축 회전(t0 → t1, 카메라 시각)
struct Imu {
    s: Vec<(f64, V3)>,
    r_ci: M3,
    shift: f64,
}
impl Imu {
    fn load(path: &str, calib: &str) -> Imu {
        let text = std::fs::read_to_string(calib).unwrap_or_else(|e| panic!("{calib}: {e}"));
        let mut rows: Vec<Vec<f64>> = vec![];
        let (mut in_t, mut shift) = (false, 0.0);
        for l in text.lines() {
            let s = l.trim();
            if s.starts_with("T_cam_imu:") {
                in_t = true;
                continue;
            }
            if in_t && s.starts_with('-') && rows.len() < 3 {
                rows.push(s.trim_start_matches('-').trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|x| x.trim().parse().unwrap()).collect());
                continue;
            }
            in_t = false;
            if let Some(v) = s.strip_prefix("timeshift_cam_imu:") {
                shift = v.trim().parse().unwrap();
            }
        }
        let r_ci = std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j]));
        let t = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut s: Vec<(f64, V3)> = vec![];
        for l in t.lines() {
            let v: Vec<f64> = l.split(',').map_while(|x| x.trim().parse().ok()).collect();
            if v.len() >= 7 && s.last().is_none_or(|p| v[0] * 1e-9 > p.0) {
                s.push((v[0] * 1e-9, [v[1], v[2], v[3]]));
            }
        }
        Imu { s, r_ci, shift }
    }
    fn rot(&self, t0: f64, t1: f64) -> Option<M3> {
        let (a, b) = (t0 + self.shift, t1 + self.shift);
        let i = self.s.partition_point(|x| x.0 <= a);
        if i == 0 || i >= self.s.len() {
            return None;
        }
        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut tp = a;
        for x in &self.s[i..] {
            let end = x.0.min(b);
            if end - tp > 0.02 {
                return None;
            }
            r = mm(&r, &exp3(x.1.map(|w| w * (end - tp))));
            tp = end;
            if x.0 >= b {
                break;
            }
        }
        if b - tp > 0.01 {
            return None;
        }
        Some(mm(&mm(&self.r_ci, &r), &mt(&self.r_ci)))
    }
}

/// 3×3 대칭 행렬의 가장 작은 고유벡터(야코비)
fn smallest_eig(c: [[f64; 3]; 3]) -> V3 {
    let mut a = c;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut mx) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > mx {
                    (p, q, mx) = (i, j, a[i][j].abs());
                }
            }
        }
        if mx < 1e-15 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c_, s_) = (th.cos(), th.sin());
        let mut j: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        j[p][p] = c_;
        j[q][q] = c_;
        j[p][q] = s_;
        j[q][p] = -s_;
        a = mm(&mm(&mt(&j), &a), &j);
        v = mm(&v, &j);
    }
    let k = (0..3).min_by(|&x, &y| a[x][x].partial_cmp(&a[y][y]).unwrap()).unwrap();
    [v[0][k], v[1][k], v[2][k]]
}

/// 평면(n·x = d)
#[derive(Clone, Copy)]
struct Plane {
    n: V3,
    d: f64,
}

/// 점 무리에서 큰 평면 k 개(RANSAC 2 cm → 안쪽 점으로 다시 맞춤). 찾은 평면의 안쪽 점(3 cm)은 다음 평면 찾기에서 뺌
fn find_planes(pts: &[[f32; 3]], k: usize) -> Vec<(Plane, usize)> {
    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut rnd = |m: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % m as u64) as usize
    };
    let step = (pts.len() / 60_000).max(1);
    let mut sub: Vec<V3> = pts.iter().step_by(step).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let mut out = vec![];
    for _ in 0..k {
        if sub.len() < 1000 {
            break;
        }
        let mut best = (0usize, Plane { n: [0.0, 0.0, 1.0], d: 0.0 });
        for _ in 0..400 {
            let (a, b, c) = (sub[rnd(sub.len())], sub[rnd(sub.len())], sub[rnd(sub.len())]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let l = norm(&n);
            if l < 1e-6 {
                continue;
            }
            let n = n.map(|x| x / l);
            let d = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
            let cnt = sub.iter().filter(|p| (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - d).abs() < 0.02).count();
            if cnt > best.0 {
                best = (cnt, Plane { n, d });
            }
        }
        // 안쪽 점으로 다시 맞춤(주성분)
        let mut pl = best.1;
        for _ in 0..2 {
            let ins: Vec<&V3> = sub.iter().filter(|p| (pl.n[0] * p[0] + pl.n[1] * p[1] + pl.n[2] * p[2] - pl.d).abs() < 0.02).collect();
            let m = ins.len() as f64;
            let c0: V3 = std::array::from_fn(|i| ins.iter().map(|p| p[i]).sum::<f64>() / m);
            let mut cv = [[0.0; 3]; 3];
            for p in &ins {
                for i in 0..3 {
                    for j in 0..3 {
                        cv[i][j] += (p[i] - c0[i]) * (p[j] - c0[j]);
                    }
                }
            }
            let n = smallest_eig(cv);
            pl = Plane { n, d: n[0] * c0[0] + n[1] * c0[1] + n[2] * c0[2] };
        }
        let cnt = sub.iter().filter(|p| (pl.n[0] * p[0] + pl.n[1] * p[1] + pl.n[2] * p[2] - pl.d).abs() < 0.02).count();
        out.push((pl, cnt * step));
        sub.retain(|p| (pl.n[0] * p[0] + pl.n[1] * p[1] + pl.n[2] * p[2] - pl.d).abs() >= 0.03);
    }
    out
}

/// 평면 위 10 cm 칸마다(평면 ±band 점 300 개 이상: 추정 지도는 전체 맞춤 뒤 남은 어긋남을 넘게 25 cm, 그 절반 이상이 칸 중앙값 ±2 cm) 칸 중앙값 ±5 cm 안 점의 평면 거리 p90 − p10.
/// (칸 두께 중앙값 cm, 칸 수, 두 겹 칸 비율 %)
fn thickness(pts: &[[f32; 3]], pl: &Plane, band: f64) -> (f64, usize, f64) {
    let n = pl.n;
    let u0 = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = {
        let c = [n[1] * u0[2] - n[2] * u0[1], n[2] * u0[0] - n[0] * u0[2], n[0] * u0[1] - n[1] * u0[0]];
        let l = norm(&c);
        c.map(|x| x / l)
    };
    let v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    let mut cells: BTreeMap<(i32, i32), Vec<f32>> = BTreeMap::new();
    for p in pts {
        let q = [p[0] as f64, p[1] as f64, p[2] as f64];
        let d = n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - pl.d;
        if d.abs() < band {
            let (a, b) = (u[0] * q[0] + u[1] * q[1] + u[2] * q[2], v[0] * q[0] + v[1] * q[1] + v[2] * q[2]);
            cells.entry(((a / 0.1).floor() as i32, (b / 0.1).floor() as i32)).or_default().push(d as f32);
        }
    }
    let (mut sp, mut far, mut tot) = (vec![], 0usize, 0usize);
    let (mut n_dbl, mut n_plane) = (0usize, 0usize);
    for (_, mut ds) in cells {
        if ds.len() < 300 {
            continue;
        }
        // 두 겹 칸: 1 cm 칸 히스토그램에서 주 봉우리(3 칸 합, 점의 40 % 이상)와 3 cm 넘게 떨어진 둘째 봉우리가 주 봉우리의 25 % 이상
        let nb = (2.0 * band / 0.01).ceil() as usize + 1;
        let mut hist = vec![0usize; nb];
        for d in &ds {
            hist[(((*d as f64) + band) / 0.01).floor().clamp(0.0, (nb - 1) as f64) as usize] += 1;
        }
        let tri = |i: usize| hist[i.saturating_sub(1)] + hist[i] + if i + 1 < nb { hist[i + 1] } else { 0 };
        let main = (0..nb).max_by_key(|&i| (tri(i), usize::MAX - i)).unwrap();
        if tri(main) * 10 >= ds.len() * 4 {
            n_plane += 1;
            let second = (0..nb).filter(|&i| i.abs_diff(main) >= 3).map(tri).max().unwrap_or(0);
            n_dbl += (second * 4 >= tri(main)) as usize;
        }
        ds.sort_by(|a, b| a.total_cmp(b));
        let med = ds[ds.len() / 2];
        let near = ds.iter().filter(|d| (*d - med).abs() < 0.02).count();
        if near * 2 < ds.len() {
            continue;
        }
        // 칸 중앙값 ±5 cm 안(그 밖은 책상 위 물건 등 다른 면)
        let band: Vec<f32> = ds.iter().copied().filter(|d| (*d - med).abs() < 0.05).collect();
        let q = |f: f64| band[((band.len() - 1) as f64 * f) as usize];
        sp.push((q(0.9) - q(0.1)) as f64 * 100.0);
        far += band.iter().filter(|d| (*d - med).abs() > 0.015).count();
        tot += band.len();
    }
    let _ = (far, tot);
    if sp.is_empty() {
        return (f64::NAN, 0, f64::NAN);
    }
    sp.sort_by(|a, b| a.total_cmp(b));
    (sp[sp.len() / 2], sp.len(), 100.0 * n_dbl as f64 / n_plane.max(1) as f64)
}

/// 겹 어긋남: 5 cm 칸(점 40 개 이상, 평평함)마다 2 초 묶음(점 8 개 이상)별 평균을 칸 법선에 투영해 최대 − 최소.
/// (2 cm 넘는 칸 비율 %, 평균 cm, 잰 칸 수)
fn layer_gap(pts: &[[f32; 3]], seg: &[u32]) -> (f64, f64, usize) {
    let mut vox: std::collections::HashMap<(i32, i32, i32), Vec<u32>> = std::collections::HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        vox.entry(((p[0] / 0.05).floor() as i32, (p[1] / 0.05).floor() as i32, (p[2] / 0.05).floor() as i32)).or_default().push(i as u32);
    }
    let (mut n, mut n2, mut sum) = (0usize, 0usize, 0f64);
    for ids in vox.values() {
        if ids.len() < 40 {
            continue;
        }
        let m = ids.len() as f64;
        let c: V3 = std::array::from_fn(|k| ids.iter().map(|&i| pts[i as usize][k] as f64).sum::<f64>() / m);
        let mut cv = [[0.0; 3]; 3];
        for &i in ids {
            let p = pts[i as usize];
            for a in 0..3 {
                for b in 0..3 {
                    cv[a][b] += (p[a] as f64 - c[a]) * (p[b] as f64 - c[b]) / m;
                }
            }
        }
        let nv = smallest_eig(cv);
        let tr = cv[0][0] + cv[1][1] + cv[2][2];
        let lam = (0..3).map(|a| (0..3).map(|b| nv[a] * cv[a][b] * nv[b]).sum::<f64>()).sum::<f64>();
        // 평평한 칸만(두 겹 2 cm 는 법선 방향 분산 1 cm² 라 이 문턱 안)
        if lam > 0.25 * tr {
            continue;
        }
        let mut by: BTreeMap<u32, (f64, usize)> = BTreeMap::new();
        for &i in ids {
            let p = pts[i as usize];
            let d = (0..3).map(|a| nv[a] * (p[a] as f64 - c[a])).sum::<f64>();
            let e = by.entry(seg[i as usize]).or_insert((0.0, 0));
            e.0 += d;
            e.1 += 1;
        }
        let means: Vec<f64> = by.values().filter(|v| v.1 >= 8).map(|v| v.0 / v.1 as f64).collect();
        if means.len() < 2 {
            continue;
        }
        let gap = means.iter().cloned().fold(f64::MIN, f64::max) - means.iter().cloned().fold(f64::MAX, f64::min);
        n += 1;
        n2 += (gap > 0.02) as usize;
        sum += gap;
    }
    (100.0 * n2 as f64 / n.max(1) as f64, 100.0 * sum / n.max(1) as f64, n)
}

/// 장면 깊이 → 점(카메라 광학 축), 4 화소 간격, 0.4 – 3.5 m
fn backproject(f: &slam::stream::OutFrame, pose: &T34, out: &mut Vec<[f32; 3]>) {
    let (w, h) = (f.width as usize, f.height as usize);
    let (r, p) = (t_r(pose), t_p(pose));
    for y in (0..h).step_by(4) {
        for x in (0..w).step_by(4) {
            let z = f.depth[y * w + x] as f64 / f.depth_per_m as f64;
            if !(0.4..3.5).contains(&z) {
                continue;
            }
            let c = [(x as f64 - f.cx) / f.fx * z, (y as f64 - f.cy) / f.fy * z, z];
            let q = mv(&r, &c);
            out.push([(q[0] + p[0]) as f32, (q[1] + p[1]) as f32, (q[2] + p[2]) as f32]);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(gt_path) = arg(&a, "--gt") else {
        eprintln!("slam_eval --gt <정답> --poses <보낸 위치> [--final <노드>] [--stats <csv>] [--imu <imu0.csv> [--calib yaml]] [--stream <스트림>] [--every 2] [--tag 이름]");
        std::process::exit(2);
    };
    let tag = arg(&a, "--tag").unwrap_or("-");
    let every: usize = arg(&a, "--every").map(|s| s.parse().unwrap()).unwrap_or(2);
    let gt = Gt(read_tum(gt_path).into_iter().filter_map(|(t, p, q)| Some((t, p, quat_m(q?)))).collect());
    let mut js: Vec<(String, String)> = vec![("tag".into(), format!("\"{tag}\""))];
    macro_rules! put {
        ($k:expr, $v:expr) => {
            js.push(($k.to_string(), $v))
        };
    }

    // ATE
    if let Some(p) = arg(&a, "--poses") {
        if let Some((se3, sim3, s, n)) = ate_file(p, &gt) {
            put!("ate_cm", format!("{se3:.2}"));
            put!("ate_sim3_cm", format!("{sim3:.2}"));
            put!("scale", format!("{s:.3}"));
            put!("posed", n.to_string());
        }
    }
    if let Some(p) = arg(&a, "--final") {
        if let Some((se3, sim3, _, n)) = ate_file(p, &gt) {
            put!("ate_final_cm", format!("{se3:.2}"));
            put!("ate_final_sim3_cm", format!("{sim3:.2}"));
            put!("nodes", n.to_string());
        }
    }
    // 통계 CSV: 놓침 · 새 지도 · 처리 시간
    let mut frames_total = 0usize;
    if let Some(p) = arg(&a, "--stats") {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        let mut lines = text.lines();
        let head: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
        let col = |k: &str| head.iter().position(|h| *h == k);
        let rows: Vec<Vec<f64>> = lines.map(|l| l.split(',').map(|x| x.parse().unwrap_or(0.0)).collect()).collect();
        frames_total = rows.len();
        let get = |r: &Vec<f64>, k: &str| col(k).and_then(|i| r.get(i).copied()).unwrap_or(0.0);
        let lost: Vec<bool> = rows.iter().map(|r| get(r, "lost") > 0.5).collect();
        let episodes = (0..lost.len()).filter(|&i| lost[i] && (i == 0 || !lost[i - 1])).count();
        put!("frames", rows.len().to_string());
        put!("lost_frames", lost.iter().filter(|x| **x).count().to_string());
        put!("lost_events", episodes.to_string());
        put!("new_maps", rows.iter().filter(|r| get(r, "new_map") > 0.5).count().to_string());
        put!("no_map", rows.iter().filter(|r| get(r, "no_map") > 0.5).count().to_string());
        put!("held", rows.iter().filter(|r| get(r, "held") > 0.5).count().to_string());
        let mean = |k: &str| {
            let v: Vec<f64> = rows.iter().skip(1).map(|r| get(r, k)).filter(|x| *x >= 0.0).collect();
            v.iter().sum::<f64>() / v.len().max(1) as f64
        };
        let mut od: Vec<f64> = rows.iter().skip(1).map(|r| get(r, "odom_ms")).collect();
        od.sort_by(|a, b| a.total_cmp(b));
        put!("odom_ms", format!("{:.2}", mean("odom_ms")));
        put!("odom_p95_ms", format!("{:.2}", od.get(((od.len().max(1) - 1) as f64 * 0.95) as usize).copied().unwrap_or(0.0)));
        put!("slam_ms", format!("{:.2}", mean("slam_ms")));
    }
    if frames_total > 0 {
        if let Some(n) = js.iter().find(|(k, _)| k == "posed").map(|(_, v)| v.parse::<f64>().unwrap()) {
            js.push(("posed_pct".into(), format!("{:.1}", 100.0 * n / frames_total as f64)));
        }
    }
    let imu = arg(&a, "--imu").map(|p| Imu::load(p, arg(&a, "--calib").unwrap_or("docs/imu_cam_calib.yaml")));

    // 스트림: 튐 · 지도 두께
    if let Some(sp) = arg(&a, "--stream") {
        let file: Box<dyn std::io::Read> = if sp == "-" { Box::new(std::io::stdin().lock()) } else { Box::new(std::fs::File::open(sp).unwrap_or_else(|e| panic!("{sp}: {e}"))) };
        let mut rd = slam::stream::Reader::new(std::io::BufReader::with_capacity(1 << 22, file));
        // 노드 위치 · 맡김(물체 기억 kfstore 와 같음): 장면 = 노드 × 상대 위치
        let mut nodes: BTreeMap<u32, (T34, bool)> = BTreeMap::new();
        struct Kf {
            anchor: u32,
            rel: T34,
            t: f64,
            pts: Vec<[f32; 3]>,
            gt_pts: Vec<[f32; 3]>,
        }
        let mut kfs: Vec<Kf> = vec![];
        let (mut prev, mut n_jump, mut n_jump_ev, mut in_jump, mut n_pairs) = (None::<(f64, T34)>, 0usize, 0usize, false, 0usize);
        let (mut rot_errs, mut tr_errs) = (vec![], vec![]);
        let mut seq = 0usize;
        let mut shown_pairs: Vec<(V3, V3)> = vec![];
        let (mut rejoin, mut rejoin_t0): (Vec<f64>, Option<f64>) = (vec![], None);
        while let Some(f) = rd.recv() {
            if f.reset {
                nodes.clear();
                kfs.clear();
                prev = None;
                continue;
            }
            if f.is_graph() {
                for g in &f.graph {
                    nodes.insert(g.id, (g.t, g.held != 0));
                }
                continue;
            }
            let lost = f.flags & slam::stream::LOST != 0;
            let held = f.flags & slam::stream::HELD != 0;
            let shown = !lost || held;
            // 다시 붙기: 새 지도를 연 장면부터 믿는 위치(임시 · 맡김 · 놓침 아님)가 다시 나온 장면까지
            if f.flags & slam::stream::NEW_MAP != 0 && rejoin_t0.is_none() {
                rejoin_t0 = Some(f.t);
            } else if let Some(t0) = rejoin_t0.filter(|_| !lost && !held && f.flags & slam::stream::PROVISIONAL == 0) {
                rejoin.push(f.t - t0);
                rejoin_t0 = None;
            }
            // 튐: 화면에 보낸 위치의 장면 사이 움직임
            if shown {
                let pose = f.pose;
                if let Some((t0, p0)) = prev.filter(|(t0, _)| f.t - t0 < 0.1) {
                    let rel = tmul(&tinv(&p0), &pose);
                    let refr = imu.as_ref().and_then(|u| u.rot(t0, f.t)).or_else(|| Some(t_r(&tmul(&tinv(&gt.at(t0)?), &gt.at(f.t)?))));
                    let gtrel = gt.at(t0).zip(gt.at(f.t)).map(|(a, b)| tmul(&tinv(&a), &b));
                    let mut bad = false;
                    if let Some(r) = refr {
                        let e = norm(&log3(&mm(&mt(&r), &t_r(&rel)))).to_degrees();
                        let th = norm(&log3(&r)).to_degrees();
                        rot_errs.push(e);
                        bad |= e > 3.0 + 0.2 * th;
                    }
                    if let Some(g) = gtrel {
                        let (pe, pg) = (t_p(&rel), t_p(&g));
                        let e = norm(&[pe[0] - pg[0], pe[1] - pg[1], pe[2] - pg[2]]);
                        tr_errs.push(e * 100.0);
                        bad |= e > 0.03 + 0.2 * norm(&pg);
                    }
                    n_pairs += 1;
                    n_jump += bad as usize;
                    n_jump_ev += (bad && !in_jump) as usize;
                    in_jump = bad;
                }
                prev = Some((f.t, pose));
                if let Some(g) = gt.at(f.t) {
                    shown_pairs.push((t_p(&pose), t_p(&g)));
                }
            } else {
                prev = None;
            }
            // 지도: 놓침(맡김 아님) · 위치만 장면 빼고, every 장마다
            let in_map = shown && f.flags & slam::stream::NO_MAP == 0;
            seq += 1;
            if in_map && seq % every == 0 && f.width > 0 {
                let (anchor, rel) = match nodes.get(&f.anchor) {
                    Some((n, _)) if f.anchor != 0 => (f.anchor, tmul(&tinv(n), &f.pose)),
                    _ => (0, f.pose),
                };
                let mut pts = vec![];
                backproject(&f, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], &mut pts);
                let mut gt_pts = vec![];
                if let Some(g) = gt.at(f.t) {
                    backproject(&f, &g, &mut gt_pts);
                }
                kfs.push(Kf { anchor, rel, t: f.t, pts, gt_pts });
            }
        }
        if shown_pairs.len() >= 10 {
            put!("ate_shown_cm", format!("{:.2}", ate(&shown_pairs, false).0 * 100.0));
            put!("shown", shown_pairs.len().to_string());
        }
        put!("rejoined", rejoin.len().to_string());
        put!("unjoined", (rejoin_t0.is_some() as usize).to_string());
        if !rejoin.is_empty() {
            put!("rejoin_mean_s", format!("{:.2}", rejoin.iter().sum::<f64>() / rejoin.len() as f64));
            put!("rejoin_max_s", format!("{:.2}", rejoin.iter().cloned().fold(0.0, f64::max)));
        }
        put!("jump_frames", n_jump.to_string());
        put!("jump_events", n_jump_ev.to_string());
        put!("jump_pairs", n_pairs.to_string());
        let pct = |v: &mut Vec<f64>, q: f64| {
            v.sort_by(|a, b| a.total_cmp(b));
            v.get(((v.len().max(1) - 1) as f64 * q) as usize).copied().unwrap_or(f64::NAN)
        };
        put!("step_rot_p99_deg", format!("{:.2}", pct(&mut rot_errs, 0.99)));
        put!("step_tr_p99_cm", format!("{:.2}", pct(&mut tr_errs, 0.99)));
        // 최종 지도: 맡긴 노드가 끝까지 맡김이면 뺌, 노드 기준이면 최종 노드 × 상대 위치
        let mut est_pts: Vec<[f32; 3]> = vec![];
        let mut gt_pts: Vec<[f32; 3]> = vec![];
        let (mut est_seg, mut gt_seg): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        let mut centers: Vec<(V3, V3)> = vec![];
        let mut n_kf = 0usize;
        let t_first = kfs.first().map_or(0.0, |k| k.t);
        for k in &kfs {
            let pose = match nodes.get(&k.anchor) {
                Some((_, true)) => continue,
                Some((n, false)) if k.anchor != 0 => tmul(n, &k.rel),
                _ => k.rel,
            };
            n_kf += 1;
            let (r, p) = (t_r(&pose), t_p(&pose));
            for q in &k.pts {
                let w = mv(&r, &[q[0] as f64, q[1] as f64, q[2] as f64]);
                est_pts.push([(w[0] + p[0]) as f32, (w[1] + p[1]) as f32, (w[2] + p[2]) as f32]);
                est_seg.push(((k.t - t_first) / 2.0) as u32);
            }
            gt_pts.extend_from_slice(&k.gt_pts);
            gt_seg.extend(std::iter::repeat_n(((k.t - t_first) / 2.0) as u32, k.gt_pts.len()));
            if let Some(g) = gt.at(k.t) {
                centers.push((p, t_p(&g)));
            }
        }
        put!("map_frames", n_kf.to_string());
        if centers.len() >= 10 {
            // 지도를 정답 좌표로 옮김(지도에 쌓인 장면 위치 ↔ 정답, SE3)
            let (_, r, ce, cg, _) = ate(&centers, false);
            for q in est_pts.iter_mut() {
                let w = apply(&r, &ce, &cg, &[q[0] as f64, q[1] as f64, q[2] as f64]);
                *q = [w[0] as f32, w[1] as f32, w[2] as f32];
            }
            // 지도 오차: 쌓인 점에서 같은 장면을 정답 위치로 쌓은 지도의 가장 가까운 점까지 거리(정답 점은 1 cm 칸마다 하나, 3 cm 칸 해시로 6 cm 까지 찾음).
            // 5 cm 넘게 떨어진 점 = 정답 지도에 없는 면(두 겹 · 어긋나 쌓인 면)
            let cell = 0.03f32;
            let key = |p: &[f32; 3]| ((p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32);
            let mut hash: std::collections::HashMap<(i32, i32, i32), Vec<[f32; 3]>> = std::collections::HashMap::new();
            let mut seen1 = std::collections::HashSet::new();
            for p in &gt_pts {
                if seen1.insert(((p[0] * 100.0).floor() as i32, (p[1] * 100.0).floor() as i32, (p[2] * 100.0).floor() as i32)) {
                    hash.entry(key(p)).or_default().push(*p);
                }
            }
            let mut errs: Vec<f32> = est_pts
                .iter()
                .step_by(8)
                .map(|p| {
                    let (x, y, z) = key(p);
                    let mut best = 0.06f32 * 0.06;
                    for dx in -2..=2 {
                        for dy in -2..=2 {
                            for dz in -2..=2 {
                                if let Some(v) = hash.get(&(x + dx, y + dy, z + dz)) {
                                    for q in v {
                                        best = best.min((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2));
                                    }
                                }
                            }
                        }
                    }
                    best.sqrt()
                })
                .collect();
            errs.sort_by(|a, b| a.total_cmp(b));
            if !errs.is_empty() {
                let q = |f: f64| errs[((errs.len() - 1) as f64 * f) as usize] as f64 * 100.0;
                put!("map_err_p50_cm", format!("{:.2}", q(0.5)));
                put!("map_err_p90_cm", format!("{:.2}", q(0.9)));
                put!("ghost_pct", format!("{:.2}", 100.0 * errs.iter().filter(|e| **e > 0.05).count() as f64 / errs.len() as f64));
            }
            // 겹 어긋남: 5 cm 칸마다 2 초 묶음별 평균 위치를 칸 법선 방향으로 비교(다시 본 면이 처음 쌓인 면과 떨어져 있으면 두 겹). 전체 맞춤과 상관없음
            let (g2, gm, gv) = layer_gap(&est_pts, &est_seg);
            let (h2, hm, _) = layer_gap(&gt_pts, &gt_seg);
            put!("layer2_pct", format!("{g2:.1}"));
            put!("layer_gap_cm", format!("{gm:.2}"));
            put!("gt_layer2_pct", format!("{h2:.1}"));
            put!("gt_layer_gap_cm", format!("{hm:.2}"));
            put!("layer_vox", gv.to_string());
            let planes = find_planes(&gt_pts, 3);
            let (mut se, mut sg, mut sd, mut sdg, mut wsum) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let mut desc = vec![];
            for (i, (pl, cnt)) in planes.iter().enumerate() {
                let (te, ce_, de) = thickness(&est_pts, pl, 0.25);
                let (tg, _, dg) = thickness(&gt_pts, pl, 0.25);
                if te.is_finite() && tg.is_finite() {
                    let w = *cnt as f64;
                    se += w * te;
                    sg += w * tg;
                    sd += w * de;
                    sdg += w * dg;
                    wsum += w;
                }
                desc.push(format!("{{\"n\":[{:.2},{:.2},{:.2}],\"pts\":{cnt},\"cells\":{ce_},\"thick_cm\":{te:.2},\"gt_thick_cm\":{tg:.2},\"double_pct\":{de:.1},\"gt_double_pct\":{dg:.1}}}", pl.n[0], pl.n[1], pl.n[2]));
                let _ = i;
            }
            if wsum > 0.0 {
                put!("thick_cm", format!("{:.2}", se / wsum));
                put!("gt_thick_cm", format!("{:.2}", sg / wsum));
                put!("double_pct", format!("{:.1}", sd / wsum));
                put!("gt_double_pct", format!("{:.1}", sdg / wsum));
            }
            put!("planes", format!("[{}]", desc.join(",")));
        }
    }
    let body: Vec<String> = js.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    println!("{{{}}}", body.join(","));
}
