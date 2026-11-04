//! 특징점 없이 RGB-D 로 위치(직접 방식, DVO): 앞 장면의 밝기·깊이로 화소들을 이번 장면에 옮겨 보고,
//! 밝기 차이가 가장 작아지는 움직임(회전·이동)을 거친 해상도부터 고운 해상도로 가우스-뉴턴(후버 가중)으로 찾는다.
//! 무늬가 적거나 조금 흐린 장면에서도 밝기 기울기만 있으면 버틴다. 깊이는 앞 장면 것만 쓴다.
use crate::geom::*;

/// 흑백 피라미드: 단계마다 (가로, 세로, 밝기 0–1)
pub struct Pyr {
    pub lv: Vec<(usize, usize, Vec<f32>)>,
}

pub fn pyramid(rgb: &[u8], w: usize, h: usize, n: usize) -> Pyr {
    let g: Vec<f32> = rgb.chunks_exact(3).map(|c| (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0).collect();
    let mut lv = vec![(w, h, g)];
    for _ in 1..n {
        let (pw, _, pg) = lv.last().unwrap();
        let (pw, ph) = (*pw, lv.last().unwrap().1);
        let (nw, nh) = (pw / 2, ph / 2);
        let mut ng = vec![0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let i = 2 * y * pw + 2 * x;
                ng[y * nw + x] = 0.25 * (pg[i] + pg[i + 1] + pg[i + pw] + pg[i + pw + 1]);
            }
        }
        lv.push((nw, nh, ng));
    }
    Pyr { lv }
}

/// 쌍선형 밝기와 기울기(중앙 차분)
#[inline]
fn samp(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<(f32, f32, f32)> {
    if x < 1.0 || y < 1.0 || x >= (w - 2) as f32 || y >= (h - 2) as f32 {
        return None;
    }
    let bl = |x: f32, y: f32| {
        let (x0, y0) = (x as usize, y as usize);
        let (ax, ay) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * w + x0;
        let top = img[i] + (img[i + 1] - img[i]) * ax;
        let bot = img[i + w] + (img[i + w + 1] - img[i + w]) * ax;
        top + (bot - top) * ay
    };
    Some((bl(x, y), 0.5 * (bl(x + 1.0, y) - bl(x - 1.0, y)), 0.5 * (bl(x, y + 1.0) - bl(x, y - 1.0))))
}

/// 앞 장면(밝기 피라미드·깊이) → 이번 장면 움직임. init: (R, t) 앞 카메라 좌표 → 이번 카메라 좌표의 첫 추정.
/// 반환: (R, t, 쓰인 화소 수, 평균 밝기 오차, 깊이 배율 = 앞 깊이로 옮긴 거리 / 이번 깊이 중앙값)
/// minlv: 가장 고운 단계 번호(딥러닝 깊이는 1: 640×480 단계는 깊이가 흐려 도움이 안 되고 시간만 40 % 더 듦)
pub fn track(prev: &Pyr, prev_depth: &[f32], cur: &Pyr, cur_depth: &[f32], k: [f64; 4], init: (M3, V3), minlv: usize) -> Option<(M3, V3, usize, f64, f64)> {

    let (w0, h0) = (prev.lv[0].0, prev.lv[0].1);
    let nth = std::thread::available_parallelism().map_or(1, |n| n.get()).min(4);
    let level = |l: usize, r_io: &mut M3, t_io: &mut V3, n_io: &mut usize, e_io: &mut f64| {
        let (mut r, mut t) = (*r_io, *t_io);
        let (mut last_n, mut last_err) = (*n_io, *e_io);
        let s = (1usize << l) as f64;
        let (fx, fy) = (k[0] / s, k[1] / s);
        let (cx, cy) = ((k[2] + 0.5) / s - 0.5, (k[3] + 0.5) / s - 0.5);
        let (w, h, i0) = (prev.lv[l].0, prev.lv[l].1, &prev.lv[l].2);
        let i1 = &cur.lv[l].2;
        // 앞 장면 점: 밝기 기울기가 있고 깊이가 있는 화소(맨 고운 단계는 2 화소 간격)
        let step = if l == 0 { 2 } else { 1 };
        let mut pts: Vec<(V3, f32)> = vec![];
        for y in (2..h - 2).step_by(step) {
            for x in (2..w - 2).step_by(step) {
                let i = y * w + x;
                let gx = i0[i + 1] - i0[i - 1];
                let gy = i0[i + w] - i0[i - w];
                if gx * gx + gy * gy < 0.0004 {
                    continue;
                }
                let z = prev_depth[(y << l).min(h0 - 1) * w0 + (x << l).min(w0 - 1)] as f64;
                if !(0.05..6.0).contains(&z) {
                    continue;
                }
                pts.push(([(x as f64 - cx) / fx * z, (y as f64 - cy) / fy * z, z], i0[i]));
            }
        }
        if pts.len() < 100 {
            return;
        }
        let mut bias = 0f32; // 노출 변화(밝기 전체 차이)
        let iters = [6, 8, 10, 12][l.min(3)];
        for _ in 0..iters {
            // 점마다 (H, g, 개수, Σw·r, Σw, Σ|r|) 를 쌓는다. 점이 많으면 코어마다 나눠 쌓고 더한다(보드 4 코어)
            let acc = |chunk: &[(V3, f32)]| {
                let mut hm = [[0.0f64; 6]; 6];
                let mut g = [0.0f64; 6];
                let (mut n, mut sr, mut sw, mut sabs) = (0usize, 0f64, 0f64, 0f64);
                for (p0, v0) in chunk {
                    let p1 = add(&mv(&r, p0), &t);
                    if p1[2] < 0.03 {
                        continue;
                    }
                    let (u, v) = (fx * p1[0] / p1[2] + cx, fy * p1[1] / p1[2] + cy);
                    let Some((v1, gx, gy)) = samp(i1, w, h, u as f32, v as f32) else { continue };
                    let res = (v1 - v0 - bias) as f64;
                    let wgt = if res.abs() < 0.04 { 1.0 } else { 0.04 / res.abs() };
                    let iz = 1.0 / p1[2];
                    // 화면 기울기 × 투영 미분 → 점 위치에 대한 미분, 이동 = 그대로, 회전 = p1 × ju
                    let ju = [gx as f64 * fx * iz, gy as f64 * fy * iz, -(gx as f64 * fx * p1[0] + gy as f64 * fy * p1[1]) * iz * iz];
                    let jr = cross(&p1, &ju);
                    let jac = [ju[0], ju[1], ju[2], jr[0], jr[1], jr[2]];
                    for a in 0..6 {
                        for b in a..6 {
                            hm[a][b] += wgt * jac[a] * jac[b];
                        }
                        g[a] -= wgt * jac[a] * res;
                    }
                    n += 1;
                    sr += wgt * res;
                    sw += wgt;
                    sabs += res.abs();
                }
                (hm, g, n, sr, sw, sabs)
            };
            let parts: Vec<_> = if pts.len() >= 4000 && nth > 1 {
                std::thread::scope(|sc| {
                    let hs: Vec<_> = pts.chunks(pts.len().div_ceil(nth)).map(|c| sc.spawn(move || acc(c))).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                })
            } else {
                vec![acc(&pts)]
            };
            let mut hm = [[0.0; 6]; 6];
            let mut g = [0.0; 6];
            let (mut n, mut sr, mut sw, mut sabs) = (0usize, 0f64, 0f64, 0f64);
            for (h2, g2, n2, sr2, sw2, sa2) in parts {
                for a in 0..6 {
                    for b in a..6 {
                        hm[a][b] += h2[a][b];
                    }
                    g[a] += g2[a];
                }
                n += n2;
                sr += sr2;
                sw += sw2;
                sabs += sa2;
            }
            for a in 0..6 {
                for b in 0..a {
                    hm[a][b] = hm[b][a];
                }
            }            if n < 50 {
                break;
            }
            bias += (sr / sw) as f32;
            last_n = n;
            last_err = sabs / n as f64;
            for (a, row) in hm.iter_mut().enumerate() {
                row[a] += 1e-6;
            }
            let Some(dx) = crate::vo::solve6(hm, g) else { break };
            let dr = crate::vo::so3_exp([dx[3], dx[4], dx[5]]);
            r = mm(&dr, &r);
            t = add(&mv(&dr, &t), &[dx[0], dx[1], dx[2]]);
            if dx.iter().map(|x| x * x).sum::<f64>() < 1e-10 {
                break;
            }
        }
        (*r_io, *t_io, *n_io, *e_io) = (r, t, last_n, last_err);
    };
    let (mut r, mut t) = init;
    let (mut last_n, mut last_err) = (0usize, 1.0f64);
    for l in (minlv..prev.lv.len()).rev() {
        level(l, &mut r, &mut t, &mut last_n, &mut last_err);
    }
    if std::env::var_os("DVO_DEBUG").is_some() {
        eprintln!("dvo n {last_n} err {last_err:.3}");
    }
    if last_n < 300 || last_err > 0.15 {
        return None;
    }
    // 깊이 배율 이어 붙이기: 앞 깊이로 옮긴 점의 거리 / 이번 장면 깊이(같은 화소)
    let (fx, fy, cx, cy) = (k[0], k[1], k[2], k[3]);
    let mut rs = vec![];
    for y in (8..h0 - 8).step_by(12) {
        for x in (8..w0 - 8).step_by(12) {
            let z = prev_depth[y * w0 + x] as f64;
            if !(0.2..6.0).contains(&z) {
                continue;
            }
            let p1 = add(&mv(&r, &[(x as f64 - cx) / fx * z, (y as f64 - cy) / fy * z, z]), &t);
            if p1[2] < 0.2 {
                continue;
            }
            let (u, v) = ((fx * p1[0] / p1[2] + cx).round(), (fy * p1[1] / p1[2] + cy).round());
            if u < 0.0 || v < 0.0 || u >= w0 as f64 || v >= h0 as f64 {
                continue;
            }
            let zc = cur_depth[v as usize * w0 + u as usize] as f64;
            if zc > 0.2 {
                rs.push(p1[2] / zc);
            }
        }
    }
    let sc = if rs.len() >= 30 {
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rs[rs.len() / 2].clamp(0.5, 2.0)
    } else {
        1.0
    };
    Some((r, t, last_n, last_err, sc))
}
