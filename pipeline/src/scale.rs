//! 딥러닝 깊이의 배율을 두 장면 삼각측량으로 맞춘다(포즈를 알 때: TUM 정답, 실기기는 VIO).
//! 현재 장면에서 무늬가 뚜렷한 점(16 화소 칸마다 기울기가 가장 큰 점)을 골라, 이전 장면으로 깊이 후보(0.3–5 m, 로그 간격)를
//! 하나씩 투영해 7×7 조각 NCC 가 가장 높은 깊이를 그 점의 실제 거리로 본다. 실제 거리 / 예측 깊이 의 중앙값 = 배율.
//! 앞 장면의 배율과 무관하게 장면마다 새로 재므로 오차가 쌓이지 않는다.
use crate::geom::*;

pub struct View {
    pub gray: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub r: M3,
    pub p: V3,
}

pub fn gray(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks(3).map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32).collect()
}

fn patch(g: &[f32], w: usize, h: usize, u: f64, v: f64, out: &mut [f32; 49]) -> bool {
    if u < 4.0 || v < 4.0 || u >= (w - 5) as f64 || v >= (h - 5) as f64 {
        return false;
    }
    let (x0, y0) = (u.floor() as usize, v.floor() as usize);
    let (ax, ay) = ((u - x0 as f64) as f32, (v - y0 as f64) as f32);
    let mut k = 0;
    for dy in 0..7 {
        for dx in 0..7 {
            let (x, y) = (x0 + dx - 3, y0 + dy - 3);
            let a = g[y * w + x] * (1.0 - ax) + g[y * w + x + 1] * ax;
            let b = g[(y + 1) * w + x] * (1.0 - ax) + g[(y + 1) * w + x + 1] * ax;
            out[k] = a * (1.0 - ay) + b * ay;
            k += 1;
        }
    }
    true
}

fn ncc(a: &[f32; 49], b: &[f32; 49]) -> f32 {
    let ma = a.iter().sum::<f32>() / 49.0;
    let mb = b.iter().sum::<f32>() / 49.0;
    let (mut s, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..49 {
        let (x, y) = (a[i] - ma, b[i] - mb);
        s += x * y;
        sa += x * x;
        sb += y * y;
    }
    if sa < 1e-3 || sb < 1e-3 { -1.0 } else { s / (sa * sb).sqrt() }
}

/// 반환: (배율, 쓴 점 수). 점이 30 개 미만이거나 두 장면 사이가 3 cm 미만이면 None.
pub fn metric_scale(cur: &View, prev: &View, k: [f64; 4], pred: &[f32]) -> Option<(f64, usize)> {
    let base = norm(&sub(&cur.p, &prev.p));
    if base < 0.03 {
        return None;
    }
    let (w, h) = (cur.w, cur.h);
    let [fx, fy, cx, cy] = k;
    let g = &cur.gray;
    let nd = 96;
    let (zmin, zmax) = (0.3f64, 5.0f64);
    let mut ratios = vec![];
    let mut pa = [0f32; 49];
    let mut pb = [0f32; 49];
    for gy in (8..h - 8).step_by(16) {
        for gx in (8..w - 8).step_by(16) {
            // 칸 안에서 기울기가 가장 큰 점
            let (mut best, mut bu, mut bv) = (0.0f32, 0usize, 0usize);
            for v in gy..(gy + 16).min(h - 8) {
                for u in gx..(gx + 16).min(w - 8) {
                    let dx = g[v * w + u + 1] - g[v * w + u - 1];
                    let dy = g[(v + 1) * w + u] - g[(v - 1) * w + u];
                    let m = dx * dx + dy * dy;
                    if m > best {
                        best = m;
                        bu = u;
                        bv = v;
                    }
                }
            }
            if best < 400.0 || !patch(g, w, h, bu as f64, bv as f64, &mut pa) {
                continue;
            }
            let pd = pred[bv * w + bu] as f64;
            if pd <= 0.1 {
                continue;
            }
            let ray = [(bu as f64 - cx) / fx, (bv as f64 - cy) / fy, 1.0];
            let (mut s1, mut s2, mut zb) = (-1.0f32, -1.0f32, 0.0);
            let mut prev_peak_i = 0usize;
            for i in 0..nd {
                let z = zmin * (zmax / zmin).powf(i as f64 / (nd - 1) as f64);
                let xw = add(&mv(&cur.r, &ray.map(|c| c * z)), &cur.p);
                let d = sub(&xw, &prev.p);
                let pc = [0, 1, 2].map(|j| prev.r[0][j] * d[0] + prev.r[1][j] * d[1] + prev.r[2][j] * d[2]);
                if pc[2] < 0.1 {
                    continue;
                }
                let (u2, v2) = (fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
                if !patch(&prev.gray, prev.w, prev.h, u2, v2, &mut pb) {
                    continue;
                }
                let c = ncc(&pa, &pb);
                if c > s1 {
                    if i > prev_peak_i + 3 {
                        s2 = s1; // 멀리 떨어진 다른 봉우리
                    }
                    s1 = c;
                    zb = z;
                    prev_peak_i = i;
                } else if c > s2 && i > prev_peak_i + 3 {
                    s2 = c;
                }
            }
            // 확실한 짝만: 높은 상관, 두 번째 봉우리와 차이
            if s1 > 0.9 && s1 - s2 > 0.1 && zb > zmin * 1.05 && zb < zmax * 0.95 {
                ratios.push(zb / pd);
            }
        }
    }
    if ratios.len() < 30 {
        return None;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((ratios[ratios.len() / 2], ratios.len()))
}
