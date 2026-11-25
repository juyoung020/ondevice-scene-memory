//! 끝난 지도 다듬기: 다른 때 같은 곳을 본 장면끼리 깊이 ICP 로 상대 위치를 재고, 방위 + 이동(4 자유도) 위치 그래프를 풀어
//! 지도 조각 사이 몇십 cm 어긋남(벽 · 책상 두 겹)을 없앤다. 기울기는 앞에서 중력으로 맞춰 두었으므로 방위만 푼다.
//!
//! 마디 = 작은 깊이(160×120)가 있는 장면. 간선: ① 같은 ORB 지도에서 시간상 이웃한 마디(지금 상대 위치를 그대로 측정으로)
//! ② 다른 때(3 초 넘게 떨어진) 1.5 m · 35° 안에서 본 마디 짝을 깊이 점-면 ICP 로 다듬은 상대 위치(겹친 점 · 잔차로 거름, 코시 가중)
//! ③ 마디 하나짜리 바닥 높이: 그 장면 깊이로 잰 카메라 높이로 z − 높이 = 바닥(한 층 방은 바닥이 평평), 코시 가중.
pub type T34 = [f64; 12];
use crate::mapmerge::{inv, mul, Intr, SCALE, SH, SW};

const ZMIN: f64 = 0.3;
/// 지도 · 위치 추적에 쓰는 가장 먼 깊이(m, 병렬 A4)
pub const NEAR_M: f64 = 2.5;
const ZMAX: f64 = NEAR_M;
/// 고리 후보: 시간 차(초) 이상, 카메라 거리(m) · 보는 방향 각 이하, 마디마다 최대 수
const LOOP_DT: f64 = 3.0;
const LOOP_DIST: f64 = 1.5;
const LOOP_ANG: f64 = 35.0;
const LOOP_PER_NODE: usize = 4;
/// ICP 받는 기준: 맞은 점(5 cm 안) 비율 · 수, 고친 크기 한계
const ICP_INLIER_FRAC: f64 = 0.7;
const ICP_MIN_INLIERS: usize = 300;
const ICP_MAX_T: f64 = 0.2;
const ICP_MAX_YAW: f64 = 4.0;
/// 간선 표준편차: 이웃(m, °), 고리(m, °)
const SEQ_SIG: (f64, f64) = (0.03, 0.7);
/// 이웃 간선: 몇 칸 건너 이웃까지 깊이 ICP 로 잴지 · ICP 가 ORB 위치를 고칠 수 있는 최대 이동(m, 튐 바로잡기)
const SEQ_SKIP: usize = 2;
const SEQ_ICP_MAX_T: f64 = 0.4;
const LOOP_SIG: (f64, f64) = (0.03, 0.5);

pub struct Node<'a> {
    pub t: f64,
    pub pose: T34,
    /// 원래 ORB 지도(같은 지도 안 이웃만 이웃 간선)
    pub seg: u64,
    pub d: &'a [u16],
}

fn apply(t: &T34, p: &[f64; 3]) -> [f64; 3] {
    [
        t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[3],
        t[4] * p[0] + t[5] * p[1] + t[6] * p[2] + t[7],
        t[8] * p[0] + t[9] * p[1] + t[10] * p[2] + t[11],
    ]
}

fn rot(t: &T34, p: &[f64; 3]) -> [f64; 3] {
    [t[0] * p[0] + t[1] * p[1] + t[2] * p[2], t[4] * p[0] + t[5] * p[1] + t[6] * p[2], t[8] * p[0] + t[9] * p[1] + t[10] * p[2]]
}

fn yaw_of(t: &T34) -> f64 {
    // 광학 z(앞)의 수평 방향이 아니라 회전 전체의 z 축 둘레 각: R = Rz(yaw) · R_rp 로 보고 x 축 수평 성분 대신
    // 앞 방향(광학 z)과 오른쪽(광학 x) 의 수평 성분 합으로 안정하게
    let f = [t[2], t[6]];
    f[1].atan2(f[0])
}

fn rz(a: f64) -> T34 {
    let (s, c) = a.sin_cos();
    [c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// 깊이 격자 → 카메라 점 · 법선(격자 이웃 외적), 없으면 None
fn cam_pts(d: &[u16], k: &Intr) -> Vec<Option<([f64; 3], [f64; 3])>> {
    let bp = |u: usize, v: usize| -> Option<[f64; 3]> {
        let z = d[v * SW + u] as f64 * 1e-3;
        (ZMIN..ZMAX).contains(&z).then(|| [(u as f64 * SCALE - k.cx) / k.fx * z, (v as f64 * SCALE - k.cy) / k.fy * z, z])
    };
    let mut out = vec![None; SW * SH];
    for v in 1..SH - 1 {
        for u in 1..SW - 1 {
            let (Some(p), Some(a), Some(b), Some(c), Some(e)) = (bp(u, v), bp(u + 1, v), bp(u - 1, v), bp(u, v + 1), bp(u, v - 1)) else { continue };
            let dx = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let dy = [c[0] - e[0], c[1] - e[1], c[2] - e[2]];
            // 이웃 깊이 차가 크면(모서리) 버림
            if (a[2] - b[2]).abs() > 0.05 * p[2] + 0.02 || (c[2] - e[2]).abs() > 0.05 * p[2] + 0.02 {
                continue;
            }
            let mut n = [dx[1] * dy[2] - dx[2] * dy[1], dx[2] * dy[0] - dx[0] * dy[2], dx[0] * dy[1] - dx[1] * dy[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l < 1e-12 {
                continue;
            }
            n = n.map(|x| x / l);
            out[v * SW + u] = Some((p, n));
        }
    }
    out
}

/// 맨해튼 방위 바로잡기의 가우스 폭(초)과 수평 법선 판정(|n.z| 이하)
const MAN_SIG: f64 = 2.0;
const MAN_HZ: f64 = 0.15;

/// 끝난 지도 방위 바로잡기(병렬 A2): 강의실은 벽 · 책상이 서로 직각이라 장면 깊이의 수평 법선 방위(4 배 각 원형 평균)가 모든 장면에서
/// 같아야 함. ORB 지도 안 방위 표류(같은 녹화에서 10–35° 로 흔들려 책상 줄이 V 자)를 장면마다 측정의 가우스(MAN_SIG) 가중 평균으로 재
/// 전체 기준과의 차만큼 z 축으로 돌리고, 위치는 시간 순 이웃 장면 이동을 같은 회전으로 돌려 다시 이어 붙임.
/// frames: 시간 순 (시각, 위치, 지도), deps: (시각, 160×120 깊이). 돌려줌: (기준 방위 °, 가장 큰 바로잡음 °, 잰 장면 수)
pub fn manhattan(frames: &mut [(f64, T34, u64)], deps: &[(f64, &[u16])], k: &Intr) -> (f64, f64, usize) {
    let at: std::collections::HashMap<u64, usize> = frames.iter().enumerate().map(|(i, f)| (f.0.to_bits(), i)).collect();
    // (시각, cos4a, sin4a, 강도)
    let mut ms: Vec<(f64, f64, f64, f64)> = Vec::new();
    for (t, d) in deps {
        let Some(&i) = at.get(&t.to_bits()) else { continue };
        let p = &frames[i].1;
        let (mut c, mut s, mut n) = (0.0, 0.0, 0usize);
        for (_, nc) in cam_pts(d, k).into_iter().flatten() {
            let w = rot(p, &nc);
            if w[2].abs() >= MAN_HZ {
                continue;
            }
            let a = 4.0 * w[1].atan2(w[0]);
            c += a.cos();
            s += a.sin();
            n += 1;
        }
        if n >= 200 {
            let (c, s) = (c / n as f64, s / n as f64);
            ms.push((*t, c, s, c * c + s * s));
        }
    }
    if ms.is_empty() {
        return (0.0, 0.0, 0);
    }
    let rc = ms.iter().fold((0.0, 0.0), |a, m| (a.0 + m.1 * m.3, a.1 + m.2 * m.3));
    let refa = rc.1.atan2(rc.0) / 4.0;
    let mut e: Vec<Option<f64>> = frames
        .iter()
        .map(|f| {
            let (mut c, mut s, mut w) = (0.0, 0.0, 0.0);
            let lo = ms.partition_point(|m| m.0 < f.0 - 3.0 * MAN_SIG);
            for m in ms[lo..].iter().take_while(|m| m.0 <= f.0 + 3.0 * MAN_SIG) {
                let g = m.3 * (-0.5 * ((m.0 - f.0) / MAN_SIG).powi(2)).exp();
                c += g * m.1;
                s += g * m.2;
                w += g;
            }
            (w > 1e-3).then(|| wrap(s.atan2(c) - 4.0 * refa) / 4.0)
        })
        .collect();
    // 빈 곳은 가장 가까운 값
    let known: Vec<usize> = (0..e.len()).filter(|&i| e[i].is_some()).collect();
    for i in 0..e.len() {
        if e[i].is_none() {
            let j = known.partition_point(|&j| j < i);
            let c = [j.checked_sub(1).map(|x| known[x]), known.get(j).copied()].into_iter().flatten().min_by_key(|&x| x.abs_diff(i));
            e[i] = c.and_then(|c| e[c]);
        }
    }
    let mut maxa: f64 = 0.0;
    let (mut pp, mut pn) = ([0.0; 3], [0.0; 3]);
    for (i, f) in frames.iter_mut().enumerate() {
        let a = e[i].unwrap_or(0.0);
        maxa = maxa.max(a.abs());
        let z = rz(-a);
        let p = [f.1[3], f.1[7], f.1[11]];
        let n = if i == 0 { p } else { let d = rot(&z, &[p[0] - pp[0], p[1] - pp[1], p[2] - pp[2]]); [pn[0] + d[0], pn[1] + d[1], pn[2] + d[2]] };
        pp = p;
        pn = n;
        let mut q = mul(&z, &f.1);
        q[3] = n[0];
        q[7] = n[1];
        q[11] = n[2];
        f.1 = q;
    }
    (refa.to_degrees(), maxa.to_degrees(), ms.len())
}

/// 마디 j 를 마디 i 에 맞추는 4 자유도 ICP(세계 좌표에서 j 에 곱할 변환 Δ). 돌려줌: (Δ, 맞은 점 수, 맞은 비율)
fn icp(pi: &T34, ti: &[Option<([f64; 3], [f64; 3])>], pj: &T34, sj: &[[f64; 3]], k: &Intr) -> Option<(T34, usize, f64)> {
    let wi = inv(pi);
    let src: Vec<[f64; 3]> = sj.iter().map(|p| apply(pj, p)).collect();
    let c0 = src.iter().fold([0.0; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]).map(|v| v / src.len().max(1) as f64);
    let mut delta: T34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let assoc = |delta: &T34, dmax: f64| -> (Vec<([f64; 3], [f64; 3], [f64; 3])>, usize) {
        let mut pairs = Vec::new();
        let mut seen = 0;
        for p in &src {
            let a = apply(delta, p);
            let c = apply(&wi, &a);
            if c[2] < ZMIN || c[2] > ZMAX {
                continue;
            }
            let u = ((c[0] / c[2] * k.fx + k.cx) / SCALE).round();
            let v = ((c[1] / c[2] * k.fy + k.cy) / SCALE).round();
            if u < 1.0 || v < 1.0 || u >= (SW - 1) as f64 || v >= (SH - 1) as f64 {
                continue;
            }
            let Some((q, n)) = ti[v as usize * SW + u as usize] else { continue };
            seen += 1;
            let qw = apply(pi, &q);
            let nw = rot(pi, &n);
            let d = ((a[0] - qw[0]).powi(2) + (a[1] - qw[1]).powi(2) + (a[2] - qw[2]).powi(2)).sqrt();
            if d < dmax {
                pairs.push((a, qw, nw));
            }
        }
        (pairs, seen)
    };
    for it in 0..15 {
        let dmax = (0.25 * 0.8f64.powi(it)).max(0.05);
        let (pairs, _) = assoc(&delta, dmax);
        if pairs.len() < 50 {
            return None;
        }
        let mut h = nalgebra::Matrix4::<f64>::zeros();
        let mut g = nalgebra::Vector4::<f64>::zeros();
        for (a, q, n) in &pairs {
            let r = n[0] * (a[0] - q[0]) + n[1] * (a[1] - q[1]) + n[2] * (a[2] - q[2]);
            let j = nalgebra::Vector4::new(n[0] * -(a[1] - c0[1]) + n[1] * (a[0] - c0[0]), n[0], n[1], n[2]);
            h += j * j.transpose();
            g += j * r;
        }
        h += nalgebra::Matrix4::identity() * 1e-6;
        let x = h.lu().solve(&(-g))?;
        // Δ ← T(c0) Rz(x0) T(-c0) + (x1, x2, x3) 를 앞에 곱함
        let r = rz(x[0]);
        let rc = rot(&r, &c0);
        let step: T34 = [r[0], r[1], 0.0, c0[0] - rc[0] + x[1], r[4], r[5], 0.0, c0[1] - rc[1] + x[2], 0.0, 0.0, 1.0, x[3]];
        delta = mul(&step, &delta);
        if x[0].abs() < 1e-5 && (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt() < 1e-4 && it > 5 {
            break;
        }
    }
    let (pairs, seen) = assoc(&delta, 0.05);
    let frac = pairs.len() as f64 / seen.max(1) as f64;
    Some((delta, pairs.len(), frac))
}

/// 위치 그래프 간선: i → j 측정(방위 차, i 방위 축으로 본 이동), 표준편차, 고리인지
struct Edge {
    i: usize,
    j: usize,
    dyaw: f64,
    dp: [f64; 3],
    sig: (f64, f64),
    /// 높이 표준편차 배수(이웃 간선은 크게: 기울었던 지도의 높이 차를 바닥 쪽이 이기게)
    zmul: f64,
    robust: bool,
}

fn meas(pi: &T34, pj: &T34) -> (f64, [f64; 3]) {
    let (yi, yj) = (yaw_of(pi), yaw_of(pj));
    let d = [pj[3] - pi[3], pj[7] - pi[7], pj[11] - pi[11]];
    let (s, c) = yi.sin_cos();
    (wrap(yj - yi), [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
}

/// 바닥 높이 표준편차(m)
const FLOOR_SIG: f64 = 0.03;

/// 카메라에서 바닥까지 높이: 세계 위를 향한 법선(cos > 0.95)의 점들 카메라 아래 깊이를 2 cm 칸으로 모아 1.0–2.2 m 에서 가장 낮은 큰 면(이웃 칸 합 100 점 이상)
fn floor_height(pose: &T34, t: &[Option<([f64; 3], [f64; 3])>]) -> Option<f64> {
    let mut hst = [0usize; 60];
    let mut sum = [0.0f64; 60];
    for (p, n) in t.iter().flatten() {
        let wn = rot(pose, n);
        if wn[2].abs() < 0.95 {
            continue;
        }
        let h = -rot(pose, p)[2];
        if !(1.0..2.2).contains(&h) {
            continue;
        }
        let b = ((h - 1.0) / 0.02) as usize;
        hst[b] += 1;
        sum[b] += h;
    }
    // 가장 낮은 큰 면(의자 앉는 면 · 책상은 바닥보다 위): 이웃 칸 합 100 점 이상인 가장 깊은 칸
    let win = |b: usize| (b.saturating_sub(1)..(b + 2).min(60)).map(|q| hst[q]).sum::<usize>();
    let b = (0..60).rev().find(|&b| win(b) >= 100 && hst[b] * 3 >= win(b))?;
    // 이웃 칸까지 평균
    let (mut s, mut k) = (0.0, 0usize);
    for q in b.saturating_sub(1)..(b + 2).min(60) {
        s += sum[q];
        k += hst[q];
    }
    Some(s / k as f64)
}

/// 다듬기: 마디마다 세계 쪽에 곱할 변환(새 = Δ · 옛)
pub fn refine(nodes: &[Node], k: &Intr, log: &mut Vec<String>) -> Vec<T34> {
    let n = nodes.len();
    let ident: T34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    if n < 3 {
        return vec![ident; n];
    }
    // 깊이 점(이웃 · 고리 ICP 공용)
    use rayon::prelude::*;
    let tgt: Vec<Vec<Option<([f64; 3], [f64; 3])>>> = nodes.par_iter().map(|x| cam_pts(x.d, k)).collect();
    let src: Vec<Vec<[f64; 3]>> = tgt
        .iter()
        .map(|t| (0..SH).step_by(2).flat_map(|v| (0..SW).step_by(2).map(move |u| (u, v))).filter_map(|(u, v)| t[v * SW + u].map(|x| x.0)).collect())
        .collect();
    let mut edges = Vec::new();
    // ① 이웃: 시간 순 이웃(2 초 안, 한 칸 · 두 칸 건너)을 깊이 ICP 로 다듬은 측정. ICP 가 안 맞으면 같은 ORB 지도 안 이웃만 ORB 측정 그대로
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| nodes[*a].t.total_cmp(&nodes[*b].t));
    let mut pairs: Vec<(usize, usize, bool)> = Vec::new();
    for w in 0..order.len() {
        for s_ in 1..=SEQ_SKIP {
            if w + s_ < order.len() {
                let (a, b) = (order[w], order[w + s_]);
                if nodes[b].t - nodes[a].t < 2.0 {
                    pairs.push((a, b, s_ == 1));
                }
            }
        }
    }
    let seq: Vec<Option<Edge>> = pairs
        .par_iter()
        .map(|&(a, b, first)| {
            let same = nodes[a].seg == nodes[b].seg;
            let icp_e = (src[b].len() >= 500).then(|| icp(&nodes[a].pose, &tgt[a], &nodes[b].pose, &src[b], k)).flatten().and_then(|(delta, inl, frac)| {
                let pb = mul(&delta, &nodes[b].pose);
                let mv = ((pb[3] - nodes[b].pose[3]).powi(2) + (pb[7] - nodes[b].pose[7]).powi(2) + (pb[11] - nodes[b].pose[11]).powi(2)).sqrt();
                (inl >= ICP_MIN_INLIERS && frac >= ICP_INLIER_FRAC && mv < SEQ_ICP_MAX_T).then(|| meas(&nodes[a].pose, &pb))
            });
            match icp_e {
                Some((dy, dp)) => Some(Edge { i: a, j: b, dyaw: dy, dp, sig: SEQ_SIG, zmul: 3.0, robust: true }),
                None if same && first => {
                    let (dy, dp) = meas(&nodes[a].pose, &nodes[b].pose);
                    Some(Edge { i: a, j: b, dyaw: dy, dp, sig: SEQ_SIG, zmul: 3.0, robust: false })
                }
                None => None,
            }
        })
        .collect();
    let n_seq_icp = seq.iter().flatten().filter(|e| e.robust).count();
    edges.extend(seq.into_iter().flatten());
    let n_seq = edges.len();
    let fwd = |p: &T34| [p[2], p[6], p[10]];
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        let mut near: Vec<(f64, usize)> = Vec::new();
        for j in 0..n {
            if j <= i || (nodes[j].t - nodes[i].t).abs() < LOOP_DT {
                continue;
            }
            let (pi, pj) = (&nodes[i].pose, &nodes[j].pose);
            let d = ((pi[3] - pj[3]).powi(2) + (pi[7] - pj[7]).powi(2) + (pi[11] - pj[11]).powi(2)).sqrt();
            if d > LOOP_DIST {
                continue;
            }
            let (fi, fj) = (fwd(pi), fwd(pj));
            let ca = (fi[0] * fj[0] + fi[1] * fj[1] + fi[2] * fj[2]).clamp(-1.0, 1.0);
            if ca.acos().to_degrees() > LOOP_ANG {
                continue;
            }
            near.push((d + 0.02 * ca.acos().to_degrees(), j));
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        // 같은 때 몰린 후보는 하나만(서로 2 초 넘게 떨어진 것끼리)
        let mut picked: Vec<usize> = Vec::new();
        for (_, j) in near {
            if picked.iter().all(|&p| (nodes[p].t - nodes[j].t).abs() > 2.0) {
                picked.push(j);
            }
            if picked.len() >= LOOP_PER_NODE {
                break;
            }
        }
        cands.extend(picked.into_iter().map(|j| (i, j)));
    }
    let found: Vec<Option<Edge>> = cands
        .par_iter()
        .map(|&(i, j)| {
            if src[j].len() < 500 {
                return None;
            }
            let (delta, inl, frac) = icp(&nodes[i].pose, &tgt[i], &nodes[j].pose, &src[j], k)?;
            let dt = (delta[3] * delta[3] + delta[7] * delta[7] + delta[11] * delta[11]).sqrt();
            let dyaw = delta[4].atan2(delta[0]).to_degrees().abs();
            let _ = dt;
            if inl < ICP_MIN_INLIERS || frac < ICP_INLIER_FRAC || dyaw > ICP_MAX_YAW {
                return None;
            }
            // Δ 는 세계 원점 둘레 변환이라 이동 크기는 위치에 따라 큼: j 카메라가 옮겨진 거리로 다시 잼
            let pj2 = mul(&delta, &nodes[j].pose);
            let mv = ((pj2[3] - nodes[j].pose[3]).powi(2) + (pj2[7] - nodes[j].pose[7]).powi(2) + (pj2[11] - nodes[j].pose[11]).powi(2)).sqrt();
            if mv > ICP_MAX_T {
                return None;
            }
            let (dy, dp) = meas(&nodes[i].pose, &pj2);
            Some(Edge { i, j, dyaw: dy, dp, sig: LOOP_SIG, zmul: 1.0, robust: true })
        })
        .collect();
    let n_try = cands.len();
    edges.extend(found.into_iter().flatten());
    let n_loop = edges.len() - n_seq;
    // ③ 바닥 높이: 마디마다 카메라에서 바닥까지 높이(위 향한 평면 가운데 카메라 아래 1.0–2.2 m 의 가장 큰 무리, 책상 면은 0.8 m 안팎이라 빠짐)
    let floor_h: Vec<Option<f64>> = nodes.iter().zip(&tgt).map(|(m, t)| floor_height(&m.pose, t)).collect();
    let fz: Vec<f64> = nodes.iter().zip(&floor_h).filter_map(|(m, h)| h.map(|h| m.pose[11] - h)).collect();
    let floor_ref = {
        let mut v = fz.clone();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    if let Some(f) = floor_ref {
        let mut v: Vec<f64> = fz.iter().map(|z| (z - f).abs()).collect();
        v.sort_by(f64::total_cmp);
        log.push(format!("바닥: 잰 마디 {}/{n}, 바닥 높이 {f:.3} m, 마디 바닥 어긋남 중앙 {:.3} · p90 {:.3} m", fz.len(), v[v.len() / 2], v[v.len() * 9 / 10]));
    }
    // 위치 그래프(가우스-뉴턴, 마디 0 고정): 변수 마디마다 (방위, x, y, z)
    let mut x: Vec<[f64; 4]> = nodes.iter().map(|m| [yaw_of(&m.pose), m.pose[3], m.pose[7], m.pose[11]]).collect();
    let x0 = x.clone();
    let dim = 4 * n;
    let mut last_cost = f64::MAX;
    for _ in 0..15 {
        let mut h = nalgebra::DMatrix::<f64>::zeros(dim, dim);
        let mut g = nalgebra::DVector::<f64>::zeros(dim);
        let mut cost = 0.0;
        for e in &edges {
            let (xi, xj) = (x[e.i], x[e.j]);
            let (s, c) = xi[0].sin_cos();
            let d = [xj[1] - xi[1], xj[2] - xi[2], xj[3] - xi[3]];
            let r = [
                wrap(xj[0] - xi[0] - e.dyaw) / e.sig.1.to_radians(),
                (c * d[0] + s * d[1] - e.dp[0]) / e.sig.0,
                (-s * d[0] + c * d[1] - e.dp[1]) / e.sig.0,
                (d[2] - e.dp[2]) / (e.sig.0 * e.zmul),
            ];
            let chi2: f64 = r.iter().map(|v| v * v).sum();
            // 코시 가중(척도 3 시그마)
            let w = if e.robust { 1.0 / (1.0 + chi2 / 9.0) } else { 1.0 };
            cost += if e.robust { 9.0 * (1.0 + chi2 / 9.0).ln() } else { chi2 };
            // 야코비안(4 × 8: i 4 개, j 4 개)
            let mut jm = [[0.0f64; 8]; 4];
            let iy = 1.0 / e.sig.1.to_radians();
            let it = 1.0 / e.sig.0;
            jm[0][0] = -iy;
            jm[0][4] = iy;
            // e_p = Rz(-yi) d - dp
            jm[1][0] = (-s * d[0] + c * d[1]) * it;
            jm[2][0] = (-c * d[0] - s * d[1]) * it;
            jm[1][1] = -c * it;
            jm[1][2] = -s * it;
            jm[2][1] = s * it;
            jm[2][2] = -c * it;
            jm[3][3] = -it / e.zmul;
            jm[1][5] = c * it;
            jm[1][6] = s * it;
            jm[2][5] = -s * it;
            jm[2][6] = c * it;
            jm[3][7] = it / e.zmul;
            let idx = |q: usize| if q < 4 { 4 * e.i + q } else { 4 * e.j + q - 4 };
            for a in 0..8 {
                let mut ga = 0.0;
                for row in 0..4 {
                    ga += jm[row][a] * r[row];
                }
                g[idx(a)] += w * ga;
                for b in 0..8 {
                    let mut hab = 0.0;
                    for row in 0..4 {
                        hab += jm[row][a] * jm[row][b];
                    }
                    h[(idx(a), idx(b))] += w * hab;
                }
            }
        }
        // 바닥 높이(마디 하나짜리): z − 카메라 높이 = 바닥, 코시 가중. 모든 마디에 처음 값 쪽 약한 끌림(떨어진 조각이 떠다니지 않게)
        for i in 0..n {
            if let (Some(hh), Some(f)) = (floor_h[i], floor_ref) {
                let r = (x[i][3] - hh - f) / FLOOR_SIG;
                let w = 1.0 / (1.0 + r * r / 9.0);
                cost += 9.0 * (1.0 + r * r / 9.0).ln();
                g[4 * i + 3] += w * r / FLOOR_SIG;
                h[(4 * i + 3, 4 * i + 3)] += w / (FLOOR_SIG * FLOOR_SIG);
            }
            for q in 0..4 {
                let sg = if q == 0 { 10f64.to_radians() } else { 10.0 };
                g[4 * i + q] += (x[i][q] - x0[i][q]) / (sg * sg);
                h[(4 * i + q, 4 * i + q)] += 1.0 / (sg * sg);
            }
        }
        // 마디 0 고정
        for q in 0..4 {
            h[(q, q)] += 1e9;
        }
        for q in 0..dim {
            h[(q, q)] += 1e-9;
        }
        let Some(ch) = h.cholesky() else { break };
        let dx = ch.solve(&(-g));
        for (i, xi) in x.iter_mut().enumerate() {
            for q in 0..4 {
                xi[q] += dx[4 * i + q];
            }
        }
        if (last_cost - cost).abs() < 1e-4 * cost {
            break;
        }
        last_cost = cost;
    }
    let mut big = (0.0f64, 0.0f64);
    let out: Vec<T34> = (0..n)
        .map(|i| {
            // Δ = T(p_new) Rz(yaw_new - yaw_old) T(-p_old)
            let dyaw = wrap(x[i][0] - x0[i][0]);
            let r = rz(dyaw);
            let po = [x0[i][1], x0[i][2], x0[i][3]];
            let rp = rot(&r, &po);
            big.0 = big.0.max(((x[i][1] - po[0]).powi(2) + (x[i][2] - po[1]).powi(2) + (x[i][3] - po[2]).powi(2)).sqrt());
            big.1 = big.1.max(dyaw.abs().to_degrees());
            [r[0], r[1], 0.0, x[i][1] - rp[0], r[4], r[5], 0.0, x[i][2] - rp[1], 0.0, 0.0, 1.0, x[i][3] - rp[2]]
        })
        .collect();
    log.push(format!("마디 {n}, 이웃 간선 {n_seq}(깊이 ICP {n_seq_icp}), 고리 후보 {n_try} → 받은 고리 {n_loop}, 가장 큰 고침 {:.2} m · {:.1}°", big.0, big.1));
    out
}

/// 장면마다(시각, 원래 ORB 지도) 가장 가까운 마디(같은 지도 우선)의 Δ 를 곱함
pub fn apply_deltas(frames: &mut [(f64, T34, u64)], node_t: &[(f64, u64)], deltas: &[T34]) {
    if deltas.is_empty() {
        return;
    }
    let mut by_seg: std::collections::HashMap<u64, Vec<(f64, usize)>> = std::collections::HashMap::new();
    for (i, (t, s)) in node_t.iter().enumerate() {
        by_seg.entry(*s).or_default().push((*t, i));
    }
    for v in by_seg.values_mut() {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let mut all: Vec<(f64, usize)> = node_t.iter().enumerate().map(|(i, x)| (x.0, i)).collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let nearest = |v: &[(f64, usize)], t: f64| -> usize {
        let k = v.partition_point(|x| x.0 < t);
        let a = k.saturating_sub(1);
        let b = k.min(v.len() - 1);
        if (v[a].0 - t).abs() <= (v[b].0 - t).abs() { v[a].1 } else { v[b].1 }
    };
    for f in frames.iter_mut() {
        let i = match by_seg.get(&f.2) {
            Some(v) => nearest(v, f.0),
            None => nearest(&all, f.0),
        };
        f.1 = mul(&deltas[i], &f.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 끝난 실행 결과로 다듬기 시험(손으로): REFINE_POSES=poses_final REFINE_MAPS=stitch.log.maps REFINE_OUT=출력 cargo test refine_offline -- --ignored
    #[test]
    #[ignore]
    fn refine_offline() {
        let poses = std::env::var("REFINE_POSES").unwrap();
        let maps = std::env::var("REFINE_MAPS").unwrap();
        let out = std::env::var("REFINE_OUT").unwrap();
        let ds = std::env::var("REFINE_DS").unwrap_or_else(|_| format!("{}/rec_dl/ds_1118_121652/cam", std::env::var("HOME").unwrap()));
        let key = |t: f64| (t * 1000.0).round() as i64;
        let mut seg: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();
        for l in std::fs::read_to_string(&maps).unwrap().lines() {
            let v: Vec<&str> = l.split_whitespace().collect();
            seg.insert(key(v[1].parse().unwrap()), v[0].parse().unwrap());
        }
        let mut dpath: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
        for l in std::fs::read_to_string(format!("{ds}/depth.txt")).unwrap().lines() {
            if l.starts_with('#') {
                continue;
            }
            let v: Vec<&str> = l.split_whitespace().collect();
            dpath.insert(key(v[0].parse().unwrap()), format!("{ds}/{}", v[1]));
        }
        let q2t = |v: &[f64]| -> T34 {
            let (x, y, z, w) = (v[3], v[4], v[5], v[6]);
            [
                1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), v[0],
                2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), v[1],
                2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), v[2],
            ]
        };
        let mut frames: Vec<(f64, T34, u64)> = Vec::new();
        for l in std::fs::read_to_string(&poses).unwrap().lines() {
            if l.starts_with('#') || l.trim().is_empty() {
                continue;
            }
            let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            frames.push((v[0], q2t(&v[1..]), *seg.get(&key(v[0])).unwrap_or(&0)));
        }
        let mut depths = Vec::new();
        let mut nt = Vec::new();
        let mut np = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            if i % 10 != 0 {
                continue;
            }
            let img = image::open(&dpath[&key(f.0)]).unwrap().into_luma16();
            let d: Vec<u16> = (0..SH * SW).map(|q| img.get_pixel(((q % SW) * 4) as u32, ((q / SW) * 4) as u32).0[0]).collect();
            depths.push(d);
            nt.push((f.0, f.2));
            np.push(f.1);
        }
        let nodes: Vec<Node> = (0..depths.len()).map(|i| Node { t: nt[i].0, pose: np[i], seg: nt[i].1, d: &depths[i] }).collect();
        let k = Intr { fx: 614.201904, fy: 614.689209, cx: 324.950439, cy: 241.556976 };
        let mut log = Vec::new();
        let t0 = std::time::Instant::now();
        let deps: Vec<(f64, &[u16])> = nt.iter().zip(&depths).map(|(t, d)| (t.0, d.as_slice())).collect();
        // 기울기 바로잡기 전 위치(stitch.log.maps 의 큰 지도)로 시험: LEVEL_IMU=imu0.csv 면 IMU 를 읽어 끝난 뒤와 같은 level() 을 먼저
        if let Ok(ip) = std::env::var("LEVEL_IMU") {
            let cal = crate::imu::Calib::parse(crate::imu::DEFAULT_CALIB).unwrap();
            let mut imu: Vec<(f64, [f64; 3], f64)> = Vec::new();
            for l in std::fs::read_to_string(ip).unwrap().lines() {
                if l.starts_with('#') {
                    continue;
                }
                let v: Vec<f64> = l.split(',').map(|x| x.trim().parse().unwrap()).collect();
                let r = &cal.r_ci;
                let a = [v[4], v[5], v[6]];
                let ac = [0, 1, 2].map(|i| r[i][0] as f64 * a[0] + r[i][1] as f64 * a[1] + r[i][2] as f64 * a[2]);
                imu.push((v[0] * 1e-9 - cal.shift, ac, (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt()));
            }
            let (n, m) = crate::orb3_backend::level(&mut frames, &imu);
            eprintln!("level: {n} 장, 최대 {m:.2}°");
        }
        if std::env::var("NO_MAN").is_err() {
            let (r, m, n) = manhattan(&mut frames, &deps, &k);
            eprintln!("manhattan: 기준 {r:.1}°, 최대 {m:.1}°, 잰 장면 {n}");
        }
        let mut pf = frames.clone();
        // 조각 다시 놓기(실행에는 안 씀)
        if std::env::var("REALIGN").is_ok() {
            crate::mapmerge::realign_pieces(&mut pf, &deps, &k, &mut log);
        }
        eprintln!("{log:?}");
        log.clear();
        frames = pf;
        let nodes: Vec<Node> = (0..depths.len()).map(|i| {
            let p = frames.iter().find(|f| f.0 == nt[i].0).unwrap().1;
            Node { t: nt[i].0, pose: p, seg: nt[i].1, d: &depths[i] }
        }).collect();
        if std::env::var("NO_REFINE").is_ok() {
            let mut o = String::new();
            for f in &frames {
                o += &crate::orb3_backend::tum_line(f.0, &f.1);
                o += "
";
            }
            std::fs::write(&out, o).unwrap();
            return;
        }
        let dl = refine(&nodes, &k, &mut log);
        eprintln!("{log:?} {:.1} s", t0.elapsed().as_secs_f64());
        apply_deltas(&mut frames, &nt, &dl);
        let mut o = String::new();
        for f in &frames {
            o += &crate::orb3_backend::tum_line(f.0, &f.1);
            o += "
";
        }
        std::fs::write(out, o).unwrap();
    }
}
