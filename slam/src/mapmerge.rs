//! 끝난 뒤 지도 통째 맞춤: 실행 중 못 이은 지도(빈틈 뒤 새 지도)를 가장 큰 지도에 깊이 자유 공간 검사로 옮겨 붙인다.
//!
//! 가까이서 본 장면과 멀리서 본 같은 곳은 특징 짝이 거의 안 맞아(강의실: 처음 책상 · 창가 조각) 장면 한 장끼리는 못 가른다.
//! 대신 지도 하나의 깊이 점 전부를 큰 지도 장면들의 깊이 영상에 투영해, 깊이가 맞는 점과 "큰 지도가 그 뒤를 봤는데 비어 있던" 점(자유 공간 침범)을
//! 센다. 반복되는 책상 · 모니터 줄에서 한 줄 어긋난 변환은 다른 줄 · 벽 · 창 쪽에서 자유 공간을 크게 침범해 갈린다.
//! 후보 변환 = 약한 잇기 짝(XFeat 검증만 통과)이 낸 변환. 지도는 중력 맞춤(z 위)이라 방위 + 이동 4 자유도, 높이는 두 지도 바닥 높이 차에서 시작.
use std::collections::{HashMap, HashSet};

/// 3×4 행 우선 강체 변환
pub type T34 = [f64; 12];

/// 저장 깊이 격자(원본 640×480 을 4 화소마다)
pub const SW: usize = 160;
pub const SH: usize = 120;
pub const SCALE: f64 = 4.0;
/// 쓰는 깊이 범위(m): 먼 깊이는 잡음이 커 자유 공간 판정이 흔들림
const ZMIN: f64 = 0.3;
const ZMAX: f64 = 3.5;
/// 점 구름 칸(m), 맞출 지도 점 수 최대, 큰 지도 쪽 장면 수 최대
const VOX: f64 = 0.05;
const MAX_PTS: usize = 5000;
const MAX_REF: usize = 200;
/// 받는 기준: 맞음 / (맞음 + 침범) ≥ ACCEPT, 맞은 투영 수 ≥ MIN_AGREE, 다른 후보 최고와 차 ≥ MARGIN
/// (강의실: 맞는 변환 0.84, 한 줄 어긋난 변환 다듬은 뒤 0.42 · 0.64)
const ACCEPT: f64 = 0.75;
const MIN_AGREE: usize = 1500;
const MARGIN: f64 = 0.1;
/// 맞출 지도의 최소 장면 수(저장 깊이 장 수)
const MIN_FRAMES: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct Intr {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// 깊이 장면 하나: 지도 좌표 카메라 위치(광학 → 지도)와 작은 깊이 격자(mm)
pub struct Frame<'a> {
    pub pose: T34,
    pub d: &'a [u16],
}

pub fn mul(a: &T34, b: &T34) -> T34 {
    let mut o = [0.0; 12];
    for i in 0..3 {
        for j in 0..4 {
            o[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + if j == 3 { a[i * 4 + 3] } else { 0.0 };
        }
    }
    o
}

pub fn inv(a: &T34) -> T34 {
    let mut o = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 4 + j] = a[j * 4 + i];
        }
        o[i * 4 + 3] = -(o[i * 4] * a[3] + o[i * 4 + 1] * a[7] + o[i * 4 + 2] * a[11]);
    }
    o
}

fn apply(t: &T34, p: &[f64; 3]) -> [f64; 3] {
    [
        t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[3],
        t[4] * p[0] + t[5] * p[1] + t[6] * p[2] + t[7],
        t[8] * p[0] + t[9] * p[1] + t[10] * p[2] + t[11],
    ]
}

/// z 축 방위 + 이동
pub fn yaw_t(yaw: f64, x: f64, y: f64, z: f64) -> T34 {
    let (s, c) = yaw.sin_cos();
    [c, -s, 0.0, x, s, c, 0.0, y, 0.0, 0.0, 1.0, z]
}

/// 장면들의 깊이 점(지도 좌표, 5 cm 칸마다 하나)
fn cloud(fr: &[Frame], k: &Intr) -> Vec<[f64; 3]> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for f in fr {
        for v in (0..SH).step_by(2) {
            for u in (0..SW).step_by(2) {
                let z = f.d[v * SW + u] as f64 * 1e-3;
                if !(ZMIN..ZMAX).contains(&z) {
                    continue;
                }
                let pc = [(u as f64 * SCALE - k.cx) / k.fx * z, (v as f64 * SCALE - k.cy) / k.fy * z, z];
                let p = apply(&f.pose, &pc);
                let key = ((p[0] / VOX).floor() as i32, (p[1] / VOX).floor() as i32, (p[2] / VOX).floor() as i32);
                if seen.insert(key) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// 바닥 높이: 점 높이 5 cm 칸 가운데 가장 많은 칸의 1/4 이상인 가장 낮은 칸
fn floor(p: &[[f64; 3]]) -> Option<f64> {
    let lo = p.iter().map(|x| x[2]).fold(f64::MAX, f64::min);
    if !lo.is_finite() || p.len() < 300 {
        return None;
    }
    let mut h: HashMap<i64, usize> = HashMap::new();
    for x in p {
        *h.entry(((x[2] - lo) / VOX) as i64).or_default() += 1;
    }
    let mx = *h.values().max()?;
    let b = h.iter().filter(|x| *x.1 * 4 >= mx).map(|x| *x.0).min()?;
    Some(lo + (b as f64 + 0.5) * VOX)
}

/// 큰 지도 쪽 장면: 지도 → 카메라 변환과 깊이
struct Ref<'a> {
    w2c: T34,
    d: &'a [u16],
}

/// 점(맞출 지도 좌표)을 h 로 옮겨 큰 지도 장면들에 투영: (맞음, 침범)
fn score(pts: &[[f64; 3]], refs: &[Ref], k: &Intr, h: &T34) -> (usize, usize) {
    use rayon::prelude::*;
    let moved: Vec<[f64; 3]> = pts.iter().map(|p| apply(h, p)).collect();
    refs.par_iter()
        .map(|r| {
            let (mut ag, mut vi) = (0usize, 0usize);
            for p in &moved {
                let c = apply(&r.w2c, p);
                let z = c[2];
                if !(ZMIN..ZMAX).contains(&z) {
                    continue;
                }
                let u = ((c[0] / z * k.fx + k.cx) / SCALE).round();
                let v = ((c[1] / z * k.fy + k.cy) / SCALE).round();
                if u < 0.0 || v < 0.0 || u >= SW as f64 || v >= SH as f64 {
                    continue;
                }
                let dm = r.d[v as usize * SW + u as usize];
                if dm == 0 {
                    continue;
                }
                let dm = dm as f64 * 1e-3;
                let tol = 0.05 + 0.02 * z * z;
                if (dm - z).abs() < tol {
                    ag += 1;
                } else if dm > z + 2.0 * tol {
                    vi += 1;
                }
            }
            (ag, vi)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn ratio(s: (usize, usize)) -> f64 {
    s.0 as f64 / (s.0 + s.1).max(1) as f64
}

/// 다듬을 때 최대로 할 값: 맞음 × 맞음 비율(비율만 보면 점이 겹치지 않고 가려지는 쪽으로 빠져도 1 이 됨)
fn objective(s: (usize, usize)) -> f64 {
    s.0 as f64 * ratio(s)
}

/// 점-점 ICP(방위 + 이동 4 자유도)로 마지막 다듬기: 자유 공간 점수는 깊이 허용폭(10 cm 넘음) 안에서 평평해 몇 cm 남음
fn polish(pts: &[[f64; 3]], dst: &[[f64; 3]], x0: [f64; 4]) -> [f64; 4] {
    const C: f64 = 0.1;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, q) in dst.iter().enumerate() {
        grid.entry(((q[0] / C).floor() as i32, (q[1] / C).floor() as i32, (q[2] / C).floor() as i32)).or_default().push(i);
    }
    let mut x = x0;
    for it in 0..20 {
        let dmax = (0.15 - it as f64 * 0.01).max(0.05);
        let t = yaw_t(x[0], x[1], x[2], x[3]);
        let mut pairs = Vec::new();
        for p in pts {
            let a = apply(&t, p);
            let c = ((a[0] / C).floor() as i32, (a[1] / C).floor() as i32, (a[2] / C).floor() as i32);
            let mut best = (dmax * dmax, usize::MAX);
            for di in -1..=1 {
                for dj in -1..=1 {
                    for dk in -1..=1 {
                        if let Some(v) = grid.get(&(c.0 + di, c.1 + dj, c.2 + dk)) {
                            for &j in v {
                                let q = dst[j];
                                let d2 = (q[0] - a[0]).powi(2) + (q[1] - a[1]).powi(2) + (q[2] - a[2]).powi(2);
                                if d2 < best.0 {
                                    best = (d2, j);
                                }
                            }
                        }
                    }
                }
            }
            if best.1 != usize::MAX {
                pairs.push((a, dst[best.1]));
            }
        }
        if pairs.len() < 50 {
            return x;
        }
        let n = pairs.len() as f64;
        let mp = pairs.iter().fold([0.0; 3], |s, (a, _)| [s[0] + a[0] / n, s[1] + a[1] / n, s[2] + a[2] / n]);
        let mq = pairs.iter().fold([0.0; 3], |s, (_, q)| [s[0] + q[0] / n, s[1] + q[1] / n, s[2] + q[2] / n]);
        let (mut sn, mut cs) = (0.0, 0.0);
        for (a, q) in &pairs {
            let (px, py, qx, qy) = (a[0] - mp[0], a[1] - mp[1], q[0] - mq[0], q[1] - mq[1]);
            sn += px * qy - py * qx;
            cs += px * qx + py * qy;
        }
        let d = sn.atan2(cs);
        // 새 변환 = 회전 d(점 평균 둘레) + 이동, 지금 변환 뒤에 곱함
        let (s_, c_) = d.sin_cos();
        let tx = mq[0] - (c_ * mp[0] - s_ * mp[1]);
        let ty = mq[1] - (s_ * mp[0] + c_ * mp[1]);
        let nt = mul(&yaw_t(d, tx, ty, mq[2] - mp[2]), &t);
        x = [nt[4].atan2(nt[0]), nt[3], nt[7], nt[11]];
    }
    x
}

/// 방위 · x · y · z 를 거칠게 → 곱게 넓혀 자유 공간 점수 최대로
fn refine(pts: &[[f64; 3]], refs: &[Ref], k: &Intr, x0: [f64; 4]) -> ([f64; 4], (usize, usize)) {
    let mut best = (x0, score(pts, refs, k, &yaw_t(x0[0], x0[1], x0[2], x0[3])));
    for &(da, dt, dz) in &[(4f64.to_radians(), 0.3, 0.1), (2f64.to_radians(), 0.15, 0.05), (1f64.to_radians(), 0.07, 0.03)] {
        for _ in 0..4 {
            let c = best.0;
            let mut improved = false;
            for i in -1..=1 {
                for j in -1..=1 {
                    for l in -1..=1 {
                        for m in -1..=1 {
                            if i == 0 && j == 0 && l == 0 && m == 0 {
                                continue;
                            }
                            let x = [c[0] + i as f64 * da, c[1] + j as f64 * dt, c[2] + l as f64 * dt, c[3] + m as f64 * dz];
                            let s = score(pts, refs, k, &yaw_t(x[0], x[1], x[2], x[3]));
                            if objective(s) > objective(best.1) {
                                best = (x, s);
                                improved = true;
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    best
}

/// 지도별 장면(지도 좌표)과 지도 사이 후보 변환(b ← a)으로, 가장 큰 지도 main 에 옮겨 붙일 수 있는 지도마다 main ← 그 지도 변환.
/// log: 시도 · 결과 줄
pub fn merge(maps: &HashMap<u64, Vec<Frame>>, hyps: &[(u64, u64, T34)], k: &Intr, main: u64, log: &mut Vec<String>) -> HashMap<u64, T34> {
    let mut xf: HashMap<u64, T34> = HashMap::new();
    xf.insert(main, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let mut order: Vec<u64> = maps.iter().filter(|x| *x.0 != main && x.1.len() >= MIN_FRAMES).map(|x| *x.0).collect();
    order.sort_by_key(|m| std::cmp::Reverse(maps[m].len()));
    loop {
        let mut changed = false;
        for &m in &order {
            if xf.contains_key(&m) {
                continue;
            }
            // 후보: m 과 이미 붙은 지도 사이 변환을 main ← m 으로
            let mut cands: Vec<T34> = Vec::new();
            for &(a, b, t) in hyps {
                let h = if a == m && xf.contains_key(&b) {
                    mul(&xf[&b], &t)
                } else if b == m && xf.contains_key(&a) {
                    mul(&xf[&a], &inv(&t))
                } else {
                    continue;
                };
                cands.push(h);
            }
            if cands.is_empty() {
                continue;
            }
            let fr = &maps[&m];
            let mut pts = cloud(fr, k);
            let Some(fl) = floor(&pts) else { continue };
            if pts.len() > MAX_PTS {
                let st = pts.len() as f64 / MAX_PTS as f64;
                pts = (0..MAX_PTS).map(|i| pts[(i as f64 * st) as usize]).collect();
            }
            // 큰 쪽 장면: 붙은 지도 모두(main 좌표)
            let mut all: Vec<(T34, &[u16])> = Vec::new();
            for (mm, t) in &xf {
                for f in &maps[mm] {
                    all.push((mul(t, &f.pose), f.d));
                }
            }
            let st = (all.len() as f64 / MAX_REF as f64).max(1.0);
            let refs: Vec<Ref> = (0..all.len().min(MAX_REF)).map(|i| all[(i as f64 * st) as usize]).map(|(p, d)| Ref { w2c: inv(&p), d }).collect();
            let dst_fr: Vec<Frame> = all.iter().map(|(p, d)| Frame { pose: *p, d }).collect();
            let dst = cloud(&dst_fr, k);
            // 시작점: 후보 방위 · 이동, 높이는 바닥 높이 차(큰 쪽 바닥은 옮긴 점 둘레 4 m 안에서만: 큰 지도가 곳에 따라 기울어 있음).
            // 가까운 시작점(0.3 m · 5°)은 하나만
            let cen = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]).map(|v| v / pts.len().max(1) as f64);
            let mut starts: Vec<[f64; 4]> = Vec::new();
            for h in &cands {
                let c = apply(h, &cen);
                let near: Vec<[f64; 3]> = dst.iter().filter(|q| (q[0] - c[0]).hypot(q[1] - c[1]) < 4.0).copied().collect();
                let Some(mf) = floor(&near) else { continue };
                let x = [h[4].atan2(h[0]), h[3], h[7], mf - fl];
                let dup = starts.iter().any(|s| {
                    let da = (s[0] - x[0] + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                    da.abs() < 5f64.to_radians() && (s[1] - x[1]).hypot(s[2] - x[2]) < 0.3
                });
                if !dup {
                    starts.push(x);
                }
            }
            let mut res: Vec<([f64; 4], (usize, usize))> = starts
                .iter()
                .map(|x| {
                    let r = refine(&pts, &refs, k, *x);
                    let px = polish(&pts, &dst, r.0);
                    let ps = score(&pts, &refs, k, &yaw_t(px[0], px[1], px[2], px[3]));
                    // ICP 가 자유 공간 비율을 거의 안 깎으면 ICP 값(맞음 수는 허용폭 안에서 조금 어긋날 때 오히려 큼)
                    if ratio(ps) >= ratio(r.1) - 0.03 { (px, ps) } else { r }
                })
                .collect();
            res.sort_by(|a, b| ratio(b.1).total_cmp(&ratio(a.1)));
            for r in &res {
                log.push(format!(
                    "지도 {m}(장면 {}, 점 {}): 방위 {:.1}° 이동 {:.2} {:.2} {:.2} → 맞음 {} 침범 {} ({:.3})",
                    fr.len(),
                    pts.len(),
                    r.0[0].to_degrees(),
                    r.0[1],
                    r.0[2],
                    r.0[3],
                    r.1 .0,
                    r.1 .1,
                    ratio(r.1)
                ));
            }
            let Some(best) = res.first() else { continue };
            // 다른 곳(0.5 m 또는 10° 넘게 떨어진) 후보 최고와의 차
            let second = res
                .iter()
                .skip(1)
                .filter(|r| {
                    let da = (r.0[0] - best.0[0] + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                    da.abs() > 10f64.to_radians() || (r.0[1] - best.0[1]).hypot(r.0[2] - best.0[2]) > 0.5
                })
                .map(|r| ratio(r.1))
                .fold(0.0, f64::max);
            let ok = ratio(best.1) >= ACCEPT && best.1 .0 >= MIN_AGREE && ratio(best.1) - second >= MARGIN;
            log.push(format!("지도 {m}: 가장 좋은 {:.3} · 다른 곳 최고 {second:.3} → {}", ratio(best.1), if ok { "옮겨 붙임" } else { "안 붙임" }));
            if ok {
                xf.insert(m, yaw_t(best.0[0], best.0[1], best.0[2], best.0[3]));
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    xf
}

/// 조각 다시 놓기: 끝 지도를 빈틈(0.4 초 넘게 장면 없음)마다 조각으로 나눠, 가장 큰 조각부터 나머지 조각을 지금 자리에서 시작해
/// 자유 공간 점수로 다시 맞춤(실행 중 잘못 옮겨 붙인 조각 · 표류한 조각). 받은 조각만 옮기고 나머지는 그대로.
/// frames: (시각, 위치, _) 끝 지도 장면, depth: (시각, 작은 깊이). 돌려줌: 옮긴 조각 수
pub fn realign_pieces(frames: &mut [(f64, T34, u64)], depth: &[(f64, &[u16])], k: &Intr, log: &mut Vec<String>) -> usize {
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|a, b| frames[*a].0.total_cmp(&frames[*b].0));
    let mut piece = vec![0u64; frames.len()];
    let mut cur = 0u64;
    for w in 0..order.len() {
        if w > 0 && frames[order[w]].0 - frames[order[w - 1]].0 > 0.4 {
            cur += 1;
        }
        piece[order[w]] = cur;
    }
    let at: HashMap<u64, usize> = frames.iter().enumerate().map(|(i, f)| (f.0.to_bits(), i)).collect();
    let mut maps: HashMap<u64, Vec<Frame>> = HashMap::new();
    for (t, d) in depth {
        if let Some(&i) = at.get(&t.to_bits()) {
            maps.entry(piece[i]).or_default().push(Frame { pose: frames[i].1, d });
        }
    }
    let mut cnt: HashMap<u64, usize> = HashMap::new();
    for p in &piece {
        *cnt.entry(*p).or_default() += 1;
    }
    // 조각마다 나머지 모두에 대한 지금 자리 점수가 낮은 것부터, 지금 자리에서 다듬어 크게 나아지면 옮김(두 바퀴)
    let mut moved = 0;
    let mut keys: Vec<u64> = maps.keys().copied().collect();
    keys.sort_unstable();
    let ident = [0.0f64; 4];
    for _round in 0..2 {
        let mut todo: Vec<(f64, u64, Vec<[f64; 3]>, (usize, usize))> = Vec::new();
        for &p in &keys {
            if maps[&p].len() < MIN_FRAMES {
                continue;
            }
            let mut pts = cloud(&maps[&p], k);
            if pts.len() > MAX_PTS {
                let st = pts.len() as f64 / MAX_PTS as f64;
                pts = (0..MAX_PTS).map(|i| pts[(i as f64 * st) as usize]).collect();
            }
            let others: Vec<&Frame> = keys.iter().filter(|q| **q != p).flat_map(|q| maps[q].iter()).collect();
            let st = (others.len() as f64 / MAX_REF as f64).max(1.0);
            let refs: Vec<Ref> = (0..others.len().min(MAX_REF)).map(|i| others[(i as f64 * st) as usize]).map(|f| Ref { w2c: inv(&f.pose), d: f.d }).collect();
            let s0 = score(&pts, &refs, k, &yaw_t(0.0, 0.0, 0.0, 0.0));
            todo.push((ratio(s0), p, pts, s0));
        }
        todo.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (r0, p, pts, s0) in todo {
            if r0 >= ACCEPT || s0.0 + s0.1 < MIN_AGREE {
                continue;
            }
            let others: Vec<&Frame> = keys.iter().filter(|q| **q != p).flat_map(|q| maps[q].iter()).collect();
            let st = (others.len() as f64 / MAX_REF as f64).max(1.0);
            let refs: Vec<Ref> = (0..others.len().min(MAX_REF)).map(|i| others[(i as f64 * st) as usize]).map(|f| Ref { w2c: inv(&f.pose), d: f.d }).collect();
            let (x, s1) = refine(&pts, &refs, k, ident);
            let ok = ratio(s1) >= ratio(s0) + 0.15 && ratio(s1) >= 0.7 && s1.0 >= MIN_AGREE;
            log.push(format!("조각 {p}(장면 {}): 지금 {:.3} → 다듬어 {:.3} (방위 {:.1}° 이동 {:.2} {:.2} {:.2}) {}", cnt[&p], r0, ratio(s1), x[0].to_degrees(), x[1], x[2], x[3], if ok { "옮김" } else { "그대로" }));
            if ok {
                let t = yaw_t(x[0], x[1], x[2], x[3]);
                for (i, f) in frames.iter_mut().enumerate() {
                    if piece[i] == p {
                        f.1 = mul(&t, &f.1);
                    }
                }
                for f in maps.get_mut(&p).unwrap().iter_mut() {
                    f.pose = mul(&t, &f.pose);
                }
                moved += 1;
            }
        }
    }
    log.push(format!("조각 {} 개, 옮긴 조각 {moved}", cnt.len()));
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 상자 방(벽 4 면 + 바닥)을 두 지도에서 보고, 한 지도를 알려진 변환으로 옮겨 둔 뒤 되찾는지
    #[test]
    fn recovers_known_offset() {
        let k = Intr { fx: 614.0, fy: 614.0, cx: 320.0, cy: 240.0 };
        // 방: x ∈ [-3, 3], y ∈ [-2, 4], 바닥 z = -1.0, 천장 없음. 광선 추적으로 깊이 만들기
        let render = |pose: &T34| -> Vec<u16> {
            let mut d = vec![0u16; SW * SH];
            for v in 0..SH {
                for u in 0..SW {
                    let dir_c = [(u as f64 * SCALE - k.cx) / k.fx, (v as f64 * SCALE - k.cy) / k.fy, 1.0];
                    let o = [pose[3], pose[7], pose[11]];
                    let dw = [
                        pose[0] * dir_c[0] + pose[1] * dir_c[1] + pose[2] * dir_c[2],
                        pose[4] * dir_c[0] + pose[5] * dir_c[1] + pose[6] * dir_c[2],
                        pose[8] * dir_c[0] + pose[9] * dir_c[1] + pose[10] * dir_c[2],
                    ];
                    let mut best = f64::MAX;
                    for (ax, val) in [(0usize, -3.0), (0, 3.0), (1, -2.0), (1, 4.0), (2, -1.0)] {
                        if dw[ax].abs() > 1e-9 {
                            let s = (val - o[ax]) / dw[ax];
                            if s > 0.0 && s < best {
                                best = s;
                            }
                        }
                    }
                    // 기둥(장애물) 하나: x ∈ [0.5, 0.8], y ∈ [1, 1.3]
                    for (ax, val) in [(0usize, 0.5), (0, 0.8), (1, 1.0), (1, 1.3)] {
                        if dw[ax].abs() > 1e-9 {
                            let s = (val - o[ax]) / dw[ax];
                            let p = [o[0] + s * dw[0], o[1] + s * dw[1]];
                            if s > 0.0 && s < best && (0.5..=0.8).contains(&p[0]) && (1.0..=1.3).contains(&p[1]) {
                                best = s;
                            }
                        }
                    }
                    let z = best; // dir_c z = 1 → s 가 곧 광학 z
                    if z < 3.4 {
                        d[v * SW + u] = (z * 1000.0) as u16;
                    }
                }
            }
            d
        };
        // 카메라: 수평(광학 z → 세계 방향), 광학 y = 아래(세계 -z)
        let cam = |x: f64, y: f64, yaw: f64| -> T34 {
            let (s, c) = yaw.sin_cos();
            // 광학 x = 오른쪽, y = 아래, z = 앞
            let fwd = [c, s, 0.0];
            let right = [s, -c, 0.0];
            [right[0], 0.0, fwd[0], x, right[1], 0.0, fwd[1], y, right[2], -1.0, fwd[2], 0.0]
        };
        let main_poses: Vec<T34> = (0..24).map(|i| cam(0.0, 1.0, i as f64 * 15f64.to_radians())).collect();
        let sub_world: Vec<T34> = (0..8).map(|i| cam(-1.5, 0.0, 2.0 + i as f64 * 0.3)).collect();
        // 맞출 지도 좌표 = 세계를 truth⁻¹ 로 옮긴 것
        let truth = yaw_t(0.7, 1.2, -0.4, 0.2);
        let ti = inv(&truth);
        let main_d: Vec<Vec<u16>> = main_poses.iter().map(render).collect();
        let sub_d: Vec<Vec<u16>> = sub_world.iter().map(render).collect();
        let mut maps: HashMap<u64, Vec<Frame>> = HashMap::new();
        maps.insert(1, main_poses.iter().zip(&main_d).map(|(p, d)| Frame { pose: *p, d }).collect());
        maps.insert(2, sub_world.iter().zip(&sub_d).map(|(p, d)| Frame { pose: mul(&ti, p), d }).collect());
        // 후보: 참값에서 조금 어긋난 것(2 ← 1 이 아니라 1 ← 2 로 주는 hyps: (a=2, b=1, 1 ← 2))
        let noisy = mul(&yaw_t(0.05, 0.15, -0.1, 0.0), &truth);
        let mut log = Vec::new();
        let xf = merge(&maps, &[(2, 1, noisy)], &k, 1, &mut log);
        let got = xf.get(&2).expect("붙여야 함");
        let d = ((got[3] - truth[3]).powi(2) + (got[7] - truth[7]).powi(2) + (got[11] - truth[11]).powi(2)).sqrt();
        assert!(d < 0.1, "{log:?} {got:?}");
    }
}
