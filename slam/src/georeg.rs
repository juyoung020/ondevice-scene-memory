//! 깊이 점 지도로 새 지도를 기존 지도에 다시 맞추기(IMU 모드): 놓쳐 새 지도가 열리면 새 지도 1 초 남짓의 장면으로 작은 점 지도를 만들고,
//! 놓치기 직전 몇 초의 기존 지도 점과 기하로 정합한다. 시각 특징이 안 맞는 흔들린 장면에서도 면 모양으로 붙이는 것이 목표.
//!  ① 대략: 초기값 = 새 지도 첫 장면을 마지막으로 보낸 위치에 맞춘 표시 변환(놓친 사이 자이로 회전이 들어 있음). 지도는 IMU 중력으로
//!     z 가 위라 기울기(roll · pitch)는 고정, yaw ±8° · 이동 ±0.3 m 격자(4° · 10 cm)에서 겹치는 점 수로 고름
//!  ② 정밀: 4 자유도(yaw + 이동) 점-평면 ICP, 짝 = 이웃 10 → 5 cm 안 · 점-평면 6 → 1.5 cm
//!  ③ 받아들이기: 겹침(10 cm 안 점 비율) · 맞음(5 cm 안 · 점-평면 1.5 cm 안 비율) · 잔차 RMSE · 미끄러짐(평면 하나뿐인 장면) · 다른 시작점이 다른 답으로 가는지(모호함)
//! 셈은 f32 · 결정적(해시 대신 정렬한 칸 목록), 다른 스레드에서 돌려도 결과가 같다.
use crate::register::Tf;
use std::collections::HashMap;

pub type P3 = [f32; 3];

/// 장면 깊이(mm) → 세계 점: twc = 카메라 광학 → 세계, step 화소 간격, 0.3 – 3.0 m
pub fn backproject(depth_mm: &[u16], w: usize, h: usize, k: [f64; 4], twc: &Tf, step: usize, out: &mut Vec<P3>) {
    if depth_mm.len() < w * h {
        return;
    }
    let (fx, fy, cx, cy) = (k[0] as f32, k[1] as f32, k[2] as f32, k[3] as f32);
    for y in (step / 2..h).step_by(step) {
        for x in (step / 2..w).step_by(step) {
            let z = depth_mm[y * w + x] as f32 * 1e-3;
            if !(0.3..3.0).contains(&z) {
                continue;
            }
            out.push(twc.transform_point([(x as f32 - cx) / fx * z, (y as f32 - cy) / fy * z, z]));
        }
    }
}

fn vkey(p: &P3, c: f32) -> (i32, i32, i32) {
    ((p[0] / c).floor() as i32, (p[1] / c).floor() as i32, (p[2] / c).floor() as i32)
}

/// 복셀 무게중심으로 줄임(칸 순서대로 정렬해 결정적)
pub fn voxel(pts: &[P3], c: f32) -> Vec<P3> {
    let mut m: HashMap<(i32, i32, i32), ([f64; 3], u32)> = HashMap::new();
    for p in pts {
        let e = m.entry(vkey(p, c)).or_insert(([0.0; 3], 0));
        for i in 0..3 {
            e.0[i] += p[i] as f64;
        }
        e.1 += 1;
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|(_, (s, n))| [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32]).collect()
}

/// 정합 대상(기존 지도 점): 점 · 법선 · 이웃 찾기 칸
pub struct Target {
    pts: Vec<P3>,
    nrm: Vec<P3>,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    cell: f32,
    /// 거친 탐색용: 점이 있는 4 cm 칸과 그 이웃 칸(약 4 – 8 cm 안이면 맞음으로 셈)
    occ: std::collections::HashSet<(i32, i32, i32)>,
}

/// 3×3 대칭 행렬의 가장 작은 고윳값 · 고유벡터(야코비)
fn eig3_min(c: [[f32; 3]; 3]) -> (f32, P3) {
    let mut a = c.map(|r| r.map(|x| x as f64));
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        let (mut p, mut q, mut mx) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > mx {
                    (p, q, mx) = (i, j, a[i][j].abs());
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (cs, sn) = (th.cos(), th.sin());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = cs * akp - sn * akq;
            a[k][q] = sn * akp + cs * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = cs * apk - sn * aqk;
            a[q][k] = sn * apk + cs * aqk;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = cs * vkp - sn * vkq;
            v[k][q] = sn * vkp + cs * vkq;
        }
    }
    let k = (0..3).min_by(|&x, &y| a[x][x].total_cmp(&a[y][y])).unwrap();
    (a[k][k] as f32, [v[0][k] as f32, v[1][k] as f32, v[2][k] as f32])
}

impl Target {
    /// raw 를 voxel 칸으로 줄이고, 둘레(반지름 2.5 칸) 점의 주성분으로 법선을 구함. 법선을 못 구한(이웃 5 개 미만 · 평평하지 않은) 점은 뺌
    pub fn build(raw: &[P3], vox: f32) -> Target {
        let v = voxel(raw, vox);
        let cell = vox * 2.5;
        let mut g: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in v.iter().enumerate() {
            g.entry(vkey(p, cell)).or_default().push(i as u32);
        }
        let (mut pts, mut nrm) = (Vec::new(), Vec::new());
        for p in &v {
            let (x, y, z) = vkey(p, cell);
            let mut nb: Vec<P3> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(ids) = g.get(&(x + dx, y + dy, z + dz)) {
                            for &j in ids {
                                let q = v[j as usize];
                                if (0..3).map(|i| (q[i] - p[i]).powi(2)).sum::<f32>() < cell * cell {
                                    nb.push(q);
                                }
                            }
                        }
                    }
                }
            }
            if nb.len() < 5 {
                pts.push(*p);
                nrm.push([0.0; 3]); // 법선 모름: 겹침 · 맞음(점-점)에만 씀
                continue;
            }
            let n = nb.len() as f32;
            let m: P3 = std::array::from_fn(|i| nb.iter().map(|q| q[i]).sum::<f32>() / n);
            let mut c = [[0f32; 3]; 3];
            for q in &nb {
                for i in 0..3 {
                    for j in 0..3 {
                        c[i][j] += (q[i] - m[i]) * (q[j] - m[j]) / n;
                    }
                }
            }
            let tr = c[0][0] + c[1][1] + c[2][2];
            let (l, nv) = eig3_min(c);
            pts.push(*p);
            // 평평하지 않으면(모서리 · 물건 · 잡음) 법선 없음: 점-평면 ICP 짝에서 뺌
            nrm.push(if tr <= 0.0 || l / tr > 0.08 { [0.0; 3] } else { nv });
        }
        let cell = 0.10;
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            grid.entry(vkey(p, cell)).or_default().push(i as u32);
        }
        let mut occ = std::collections::HashSet::new();
        for p in &pts {
            let (x, y, z) = vkey(p, 0.04);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        occ.insert((x + dx, y + dy, z + dz));
                    }
                }
            }
        }
        Target { pts, nrm, grid, cell, occ }
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    /// 반지름 r(≤ 10 cm) 안 가장 가까운 점(번호, 거리²)
    fn nearest(&self, p: &P3, r: f32) -> Option<(u32, f32)> {
        let (x, y, z) = vkey(p, self.cell);
        let mut best: Option<(u32, f32)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ids) = self.grid.get(&(x + dx, y + dy, z + dz)) {
                        for &j in ids {
                            let q = &self.pts[j as usize];
                            let d = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2);
                            if best.is_none_or(|b| d < b.1) {
                                best = Some((j, d));
                            }
                        }
                    }
                }
            }
        }
        best.filter(|b| b.1 < r * r)
    }
}

/// 4 자유도 변환: 세계 z 축 회전(yaw, 중심 c 둘레) + 이동. p' = Rz(yaw)(p − c) + c + t
#[derive(Clone, Copy, Debug)]
struct D4 {
    yaw: f32,
    t: P3,
}

impl D4 {
    fn apply(&self, c: &P3, p: &P3) -> P3 {
        let (s, co) = self.yaw.sin_cos();
        let (x, y) = (p[0] - c[0], p[1] - c[1]);
        [co * x - s * y + c[0] + self.t[0], s * x + co * y + c[1] + self.t[1], p[2] + self.t[2]]
    }
    /// 세계 변환(Tf)으로: 원래 변환 init 뒤에 이 4 자유도를 곱함
    fn to_tf(&self, c: &P3) -> Tf {
        let (s, co) = self.yaw.sin_cos();
        // p' = R p + (c + t − R c)
        let tx = c[0] + self.t[0] - (co * c[0] - s * c[1]);
        let ty = c[1] + self.t[1] - (s * c[0] + co * c[1]);
        Tf([co, -s, 0.0, tx, s, co, 0.0, ty, 0.0, 0.0, 1.0, self.t[2]])
    }
}

/// 4×4 연립(가우스 소거, 부분 피벗)
fn solve4(a: [[f64; 4]; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let mut m = [[0f64; 5]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&a[i]);
        m[i][4] = b[i];
    }
    for c in 0..4 {
        let p = (c..4).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        for r in 0..4 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..5 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| m[i][4] / m[i][i]))
}

/// 정합 결과
#[derive(Clone, Copy, Debug)]
pub struct GeoResult {
    /// 새 지도 세계 → 기존 지도 세계
    pub t: Tf,
    /// 10 cm 안에 기존 점이 있는 새 점 비율(겹침)
    pub overlap: f32,
    /// 맞음: 5 cm 안 이웃이 있고 점-평면 거리 2 cm 안(평평하지 않은 곳은 점-점 3 cm 안)인 비율
    pub fitness: f32,
    /// 맞은 점의 점-평면 RMSE(m)
    pub rmse: f32,
    /// 짝 법선의 퍼짐(가장 작은 고윳값 / 점 수, 평면 하나면 0, 세 면이 고르면 약 0.33)
    pub spread: f32,
    /// 초기값에서 옮긴 크기(m, °)
    pub moved: (f32, f32),
}

/// 받아들이는 문턱
pub struct Accept {
    pub overlap: f32,
    pub fitness: f32,
    pub rmse: f32,
    pub spread: f32,
}

impl Default for Accept {
    fn default() -> Accept {
        Accept { overlap: 0.5, fitness: 0.5, rmse: 0.015, spread: 0.03 }
    }
}

/// src(새 지도 세계 점, 이미 줄인 것)를 init(새 → 기존 초기 변환)에서 target 에 맞춤. 받아들일 수 없으면 Err(이유)
pub fn register(target: &Target, src: &[P3], init: &Tf, acc: &Accept) -> Result<GeoResult, String> {
    if target.len() < 200 || src.len() < 100 {
        return Err(format!("점 모자람(기존 {} · 새 {})", target.len(), src.len()));
    }
    let s0: Vec<P3> = src.iter().map(|p| init.transform_point(*p)).collect();
    let n = s0.len() as f32;
    let c: P3 = std::array::from_fn(|i| s0.iter().map(|p| p[i]).sum::<f32>() / n);
    // ① 거친 탐색: 400 점이 기존 점 칸(4 cm, 이웃 포함)에 드는 수
    let step = (s0.len() / 400).max(1);
    let sub: Vec<P3> = s0.iter().step_by(step).copied().collect();
    let score = |d: &D4| -> f32 {
        sub.iter().filter(|p| target.occ.contains(&vkey(&d.apply(&c, p), 0.04))).count() as f32
    };
    let mut hyp: Vec<(f32, D4)> = Vec::new();
    for yi in -2..=2 {
        for xi in -3..=3 {
            for zi in -3..=3 {
                for wi in -3..=3 {
                    let d = D4 { yaw: (yi as f32 * 4.0).to_radians(), t: [xi as f32 * 0.1, zi as f32 * 0.1, wi as f32 * 0.1] };
                    hyp.push((score(&d), d));
                }
            }
        }
    }
    hyp.sort_by(|a, b| b.0.total_cmp(&a.0));
    if std::env::var_os("GEO_DBG").is_some() {
        let z = score(&D4 { yaw: 0.0, t: [0.0; 3] });
        eprintln!("georeg 대상 {} 점 · 새 {} 점, 초기값 겹침 {:.2}, 거친 탐색 최고 {:.2} {:?}", target.len(), s0.len(), z / sub.len() as f32, hyp[0].0 / sub.len() as f32, hyp[0].1);
    }
    // ② 위 세 시작점(서로 15 cm · 6° 넘게 떨어진 것)에서 ICP
    let mut starts: Vec<D4> = Vec::new();
    for (_, d) in &hyp {
        if starts.iter().all(|s| (s.t[0] - d.t[0]).hypot(s.t[1] - d.t[1]).hypot(s.t[2] - d.t[2]) > 0.15 || (s.yaw - d.yaw).abs() > 6f32.to_radians()) {
            starts.push(*d);
        }
        if starts.len() == 3 {
            break;
        }
    }
    let mut outs: Vec<(D4, f32, f32, f32, f32)> = Vec::new();
    for st in starts {
        let mut d = st;
        for it in 0..20 {
            // 짝: 이웃 반지름 안 가장 가까운 점, 점-평면 거리 문턱(복셀 4 cm 라 점-점 거리는 맞아도 2 – 3 cm 남음)
            let (rad, thr) = if it < 6 { (0.10f32, 0.06f32) } else if it < 12 { (0.06, 0.03) } else { (0.05, 0.015) };
            let (mut a, mut b) = ([[0f64; 4]; 4], [0f64; 4]);
            let mut m = 0usize;
            for p in &s0 {
                let q = d.apply(&c, p);
                let Some((j, _)) = target.nearest(&q, rad) else { continue };
                let (tq, nv) = (&target.pts[j as usize], &target.nrm[j as usize]);
                if nv == &[0.0; 3] {
                    continue;
                }
                let r = (0..3).map(|i| nv[i] * (q[i] - tq[i])).sum::<f32>() as f64;
                if r.abs() > thr as f64 {
                    continue;
                }
                // d q / d yaw = ẑ × (q − c)
                let jy = (nv[0] * -(q[1] - c[1]) + nv[1] * (q[0] - c[0])) as f64;
                let jr = [jy, nv[0] as f64, nv[1] as f64, nv[2] as f64];
                let w = if r.abs() < 0.01 { 1.0 } else { 0.01 / r.abs() };
                for i in 0..4 {
                    for k in 0..4 {
                        a[i][k] += w * jr[i] * jr[k];
                    }
                    b[i] -= w * jr[i] * r;
                }
                m += 1;
            }
            if m < 50 {
                break;
            }
            let Some(x) = solve4(a, b) else { break };
            d.yaw += x[0] as f32;
            for i in 0..3 {
                d.t[i] += x[1 + i] as f32;
            }
            if x.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-5 {
                break;
            }
        }
        // 평가
        let (mut near10, mut fit, mut se) = (0usize, 0usize, 0f64);
        let mut nn = [[0f32; 3]; 3];
        for p in &s0 {
            let q = d.apply(&c, p);
            let Some((j, d2)) = target.nearest(&q, 0.10) else { continue };
            near10 += 1;
            let nv = &target.nrm[j as usize];
            let tq = &target.pts[j as usize];
            let r = (0..3).map(|i| nv[i] * (q[i] - tq[i])).sum::<f32>();
            let planar = nv != &[0.0; 3];
            if d2 < 0.05 * 0.05 && if planar { r.abs() < 0.02 } else { d2 < 0.03 * 0.03 } {
                fit += 1;
                se += (r * r) as f64;
                for i in 0..3 {
                    for k in 0..3 {
                        nn[i][k] += nv[i] * nv[k];
                    }
                }
            }
        }
        let spread = if fit > 0 { eig3_min(nn.map(|r| r.map(|x| x / fit as f32))).0 } else { 0.0 };
        outs.push((d, near10 as f32 / n, fit as f32 / n, if fit > 0 { (se / fit as f64).sqrt() as f32 } else { f32::MAX }, spread));
    }
    outs.sort_by(|a, b| b.2.total_cmp(&a.2));
    if std::env::var_os("GEO_DBG").is_some() {
        for o in &outs {
            eprintln!("georeg ICP 끝 yaw {:.1}° 이동 {:?} 겹침 {:.2} 맞음 {:.2} RMSE {:.1} mm 면 퍼짐 {:.3}", o.0.yaw.to_degrees(), o.0.t, o.1, o.2, o.3 * 1e3, o.4);
        }
    }
    let (d, overlap, fitness, rmse, spread) = outs[0];
    let moved = ((d.t[0] * d.t[0] + d.t[1] * d.t[1] + d.t[2] * d.t[2]).sqrt(), d.yaw.to_degrees().abs());
    let res = GeoResult { t: d.to_tf(&c).mul(init), overlap, fitness, rmse, spread, moved };
    // 모호함: 다른 시작점이 10 cm · 5° 넘게 다른 답으로 가서 맞음이 거의 같으면 버림
    if let Some(o) = outs.iter().skip(1).find(|o| (o.0.t[0] - d.t[0]).hypot(o.0.t[1] - d.t[1]).hypot(o.0.t[2] - d.t[2]) > 0.10 || (o.0.yaw - d.yaw).abs() > 5f32.to_radians()) {
        if o.2 > 0.9 * fitness {
            return Err(format!("모호함(맞음 {:.2} · 다른 답 {:.2})", fitness, o.2));
        }
    }
    if overlap < acc.overlap {
        return Err(format!("겹침 모자람 {overlap:.2}"));
    }
    if fitness < acc.fitness || rmse > acc.rmse {
        return Err(format!("안 맞음(맞음 {fitness:.2} · RMSE {:.1} mm)", rmse * 1e3));
    }
    if spread < acc.spread {
        return Err(format!("미끄러짐(면 퍼짐 {spread:.3})"));
    }
    if moved.0 > 0.5 || moved.1 > 12.0 {
        return Err(format!("너무 많이 옮김({:.2} m · {:.1}°)", moved.0, moved.1));
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 책상 모서리 같은 세 면 + 상자: 새 지도를 yaw 5° · (0.12, −0.08, 0.03) m 어긋나게 둔 것을 되찾고, 평면 하나뿐이면 미끄러짐으로 버림
    #[test]
    fn recovers_offset_and_rejects_single_plane() {
        let mut world: Vec<P3> = Vec::new();
        for i in 0..120 {
            for j in 0..120 {
                let (u, v) = (i as f32 * 0.01, j as f32 * 0.01);
                world.push([u, v, 0.0]); // 바닥
                world.push([u, 0.0, v * 0.8]); // 벽 1
                world.push([0.0, u, v * 0.8]); // 벽 2
            }
        }
        for i in 0..30 {
            for j in 0..30 {
                let (u, v) = (0.4 + i as f32 * 0.01, 0.5 + j as f32 * 0.01);
                world.push([u, v, 0.3]); // 상자 윗면
            }
        }
        let target = Target::build(&world, 0.02);
        let truth = D4 { yaw: 5f32.to_radians(), t: [0.12, -0.08, 0.03] }.to_tf(&[0.0; 3]);
        // 새 지도 점 = 진짜 변환의 역으로 옮긴 같은 장면(일부)
        let inv = truth.inverse();
        let src: Vec<P3> = voxel(&world.iter().filter(|p| p[0] < 0.9 && p[1] < 0.9).map(|p| inv.transform_point(*p)).collect::<Vec<_>>(), 0.03);
        let r = register(&target, &src, &Tf::IDENTITY, &Accept::default()).expect("맞춤");
        let e = r.t.mul(&inv);
        assert!(e.0[3].abs() < 0.01 && e.0[7].abs() < 0.01 && e.0[11].abs() < 0.01, "{:?} {r:?}", e.0);
        assert!(e.0[1].abs() < 0.005, "{:?}", e.0);
        // 바닥 하나뿐: 미끄러짐
        let floor: Vec<P3> = world.iter().filter(|p| p[2] == 0.0).copied().collect();
        let t2 = Target::build(&floor, 0.02);
        let s2 = voxel(&floor.iter().filter(|p| p[0] < 0.6).map(|p| [p[0] + 0.05, p[1], p[2]]).collect::<Vec<_>>(), 0.03);
        assert!(register(&t2, &s2, &Tf::IDENTITY, &Accept::default()).is_err());
    }
}
