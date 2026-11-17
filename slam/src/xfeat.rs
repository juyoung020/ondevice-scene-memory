//! XFeat 희소 특징(verlab/accelerated_features, CVPR 2024, Apache-2.0)의 신경망 앞뒤를 Rust 로 옮긴 것.
//! 신경망 본체(합성곱 백본 + 머리 셋)는 NPU(.dxnn) 또는 PC ONNX 가 돌리고, 여기는 그 앞뒤만 한다.
//!
//! 원본 경로(`XFeat.detectAndCompute` + `XFeat.match`, top_k 1000, 문턱 0.05, min_cossim −1):
//! 1. 입력: RGB 채널 평균 → InstanceNorm(eps 1e-5) = [`normalize_rgb`]. NPU 판은 [`quantize_input`] 로 u8 로 바꿔 넣는다
//!    (모델 첫머리에서 (q − 112) / 32 로 되돌림).
//! 2. 신경망 출력: feats [64,60,80] · kpts 로짓 [65,60,80] · heat(신뢰도) [1,60,80] (480×640 입력).
//! 3. [`detect`]: 로짓 softmax(65) → 쓰레기 칸 빼고 8×8 펼쳐 480×640 점 확률 → 5×5 최대 거르기(NMS, 문턱 0.05)
//!    → 점수 = 점 확률(가장 가까운 칸) × 신뢰도(쌍선형) → 상위 k → feats 를 채널 정규화 후 쌍삼차(bicubic) 보간 → 정규화.
//!    좌표 · 보간 식은 PyTorch `grid_sample`(align_corners=False, 0 채움)과 같은 순서의 f32 계산.
//! 4. [`match_mnn`]: 코사인 행렬의 상호 최근접(행 · 열 argmax 가 서로를 가리키는 짝).
//!
//! 원본은 하위 화소 보정이 없는 희소 경로다(점 좌표는 화소 격자). XFeat* (조밀 + fine_matcher) 는 옮기지 않았다.

pub const H: usize = 480;
pub const W: usize = 640;
pub const HC: usize = H / 8;
pub const WC: usize = W / 8;
pub const DIM: usize = 64;
/// NPU 판 입력 u8 변환: q = clamp(round(x · QA + QZ), 0, 255)
pub const QA: f32 = 32.0;
pub const QZ: f32 = 112.0;

/// 신경망 출력 셋(모두 NCHW f32, 배치 1)
pub struct Maps<'a> {
    pub feats: &'a [f32], // [64, HC, WC]
    pub kpts: &'a [f32],  // [65, HC, WC]
    pub heat: &'a [f32],  // [1, HC, WC]
}

/// 한 장의 XFeat 특징: 점(x, y 화소), 점수, 기술자(점마다 64 개, 단위 길이)
#[derive(Clone, Default, Debug)]
pub struct XFeatures {
    pub kps: Vec<[f32; 2]>,
    pub scores: Vec<f32>,
    pub desc: Vec<f32>,
}
impl XFeatures {
    pub fn len(&self) -> usize {
        self.kps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kps.is_empty()
    }
    pub fn d(&self, i: usize) -> &[f32] {
        &self.desc[i * DIM..(i + 1) * DIM]
    }
}

/// RGB(u8, 줄 순서) → 회색 = 채널 평균 → InstanceNorm. 평균 · 분산은 f64 로 모은다(PyTorch 와 1e-7 급 차이)
pub fn normalize_rgb(rgb: &[u8], out: &mut Vec<f32>) {
    let n = rgb.len() / 3;
    out.resize(n, 0.0);
    let mut s = 0f64;
    for (o, p) in out.iter_mut().zip(rgb.chunks_exact(3)) {
        *o = (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0;
        s += *o as f64;
    }
    let mean = s / n as f64;
    let mut v = 0f64;
    for &g in out.iter() {
        let d = g as f64 - mean;
        v += d * d;
    }
    let var = v / n as f64;
    let inv = (1.0 / (var + 1e-5).sqrt()) as f32;
    let m = mean as f32;
    for o in out.iter_mut() {
        *o = (*o - m) * inv;
    }
}

/// NPU 판 입력(u8, 480×640×1)
pub fn quantize_input(x: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.extend(x.iter().map(|&v| (v * QA + QZ).round_ties_even().clamp(0.0, 255.0) as u8));
}

/// detect 의 작업 버퍼(장면마다 다시 씀)
#[derive(Default)]
pub struct Scratch {
    heat_full: Vec<f32>,
    rowmax: Vec<f32>,
    fnorm: Vec<f32>,
    cand: Vec<(u32, u32)>,
    score: Vec<f32>,
    order: Vec<u32>,
}

/// PyTorch grid_sample(align_corners=False) 의 좌표: 정규화 [-1,1] → 칸 좌표.
/// normgrid: 2·(x/(W−1)) − 1, unnormalize: (g + 1)·(size/2) − 0.5 (모두 f32)
#[inline]
fn grid_coord(x: f32, w_in: usize, size: usize) -> f32 {
    let g = 2.0f32 * (x / (w_in - 1) as f32) - 1.0;
    (g + 1.0) * (size as f32 / 2.0) - 0.5
}

#[inline]
fn cubic1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}
#[inline]
fn cubic2(x: f32, a: f32) -> f32 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}
#[inline]
fn cubic_coeffs(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let u = 1.0 - t;
    [cubic2(t + 1.0, A), cubic1(t, A), cubic1(u, A), cubic2(u + 1.0, A)]
}

/// 신경망 출력 → 점 · 점수 · 기술자(원본 detectAndCompute 와 같은 계산 순서)
pub fn detect(m: &Maps, top_k: usize, threshold: f32, sc: &mut Scratch, out: &mut XFeatures) {
    let plane = HC * WC;
    assert_eq!(m.kpts.len(), 65 * plane);
    assert_eq!(m.feats.len(), DIM * plane);
    assert_eq!(m.heat.len(), plane);
    // 1) 칸마다 softmax(65) → 64 칸을 8×8 화소로
    sc.heat_full.resize(H * W, 0.0);
    let mut e = [0f32; 65];
    for cy in 0..HC {
        for cx in 0..WC {
            let o = cy * WC + cx;
            let mut mx = f32::NEG_INFINITY;
            for c in 0..65 {
                mx = mx.max(m.kpts[c * plane + o]);
            }
            let mut s = 0f32;
            for c in 0..65 {
                e[c] = (m.kpts[c * plane + o] - mx).exp();
                s += e[c];
            }
            for c in 0..64 {
                sc.heat_full[(cy * 8 + c / 8) * W + cx * 8 + c % 8] = e[c] / s;
            }
        }
    }
    // 2) 5×5 최대 거르기(가로 최대 → 세로 최대, 밖은 −∞) 와 문턱
    let hf = &sc.heat_full;
    sc.rowmax.resize(H * W, 0.0);
    for y in 0..H {
        let r = &hf[y * W..(y + 1) * W];
        let o = &mut sc.rowmax[y * W..(y + 1) * W];
        for x in 0..W {
            let (a, b) = (x.saturating_sub(2), (x + 3).min(W));
            o[x] = r[a..b].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
    }
    sc.cand.clear();
    for y in 0..H {
        let (a, b) = (y.saturating_sub(2), (y + 3).min(H));
        for x in 0..W {
            let v = hf[y * W + x];
            if v <= threshold {
                continue;
            }
            let mut mx = f32::NEG_INFINITY;
            for yy in a..b {
                mx = mx.max(sc.rowmax[yy * W + x]);
            }
            if v == mx {
                sc.cand.push((x as u32, y as u32));
            }
        }
    }
    // 3) 점수 = 점 확률(가장 가까운 화소) × 신뢰도(쌍선형, 60×80)
    sc.score.clear();
    for &(x, y) in &sc.cand {
        let (xf, yf) = (x as f32, y as f32);
        let near = {
            let ix = grid_coord(xf, W, W).round_ties_even();
            let iy = grid_coord(yf, H, H).round_ties_even();
            if ix >= 0.0 && iy >= 0.0 && (ix as usize) < W && (iy as usize) < H { hf[iy as usize * W + ix as usize] } else { 0.0 }
        };
        let bil = bilinear(m.heat, WC, HC, grid_coord(xf, W, WC), grid_coord(yf, H, HC));
        let mut s = near * bil;
        if x == 0 && y == 0 {
            s = -1.0; // 원본: 좌표 (0,0) 은 채움 자리로 보고 −1
        }
        sc.score.push(s);
    }
    // 4) 점수 내림차순(같으면 앞 순서) 상위 k, 점수 > 0 만
    sc.order.clear();
    sc.order.extend(0..sc.cand.len() as u32);
    let scs = &sc.score;
    sc.order.sort_by(|&a, &b| scs[b as usize].partial_cmp(&scs[a as usize]).unwrap_or(std::cmp::Ordering::Equal));
    sc.order.truncate(top_k);
    // 5) feats 채널 정규화(칸마다) → 쌍삼차 보간 → 정규화
    sc.fnorm.resize(DIM * plane, 0.0);
    for o in 0..plane {
        let mut s = 0f32;
        for c in 0..DIM {
            let v = m.feats[c * plane + o];
            s += v * v;
        }
        let d = s.sqrt().max(1e-12);
        for c in 0..DIM {
            sc.fnorm[c * plane + o] = m.feats[c * plane + o] / d;
        }
    }
    out.kps.clear();
    out.scores.clear();
    out.desc.clear();
    for &i in &sc.order {
        let s = sc.score[i as usize];
        if s <= 0.0 {
            continue;
        }
        let (x, y) = sc.cand[i as usize];
        let mut d = [0f32; DIM];
        bicubic(&sc.fnorm, WC, HC, grid_coord(x as f32, W, WC), grid_coord(y as f32, H, HC), &mut d);
        let mut n = 0f32;
        for v in &d {
            n += v * v;
        }
        let n = n.sqrt().max(1e-12);
        out.kps.push([x as f32, y as f32]);
        out.scores.push(s);
        out.desc.extend(d.iter().map(|v| v / n));
    }
}

/// 한 채널 지도(w×h)의 쌍선형 값, 밖은 0
fn bilinear(map: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (xw, yn) = (x.floor(), y.floor());
    let (tw, tn) = (x - xw, y - yn);
    let (e, s) = (1.0 - tw, 1.0 - tn);
    let (nw, ne, sw, se) = (s * e, s * tw, tn * e, tn * tw);
    let at = |xx: f32, yy: f32| -> f32 {
        if xx >= 0.0 && yy >= 0.0 && (xx as usize) < w && (yy as usize) < h { map[yy as usize * w + xx as usize] } else { 0.0 }
    };
    at(xw, yn) * nw + at(xw + 1.0, yn) * ne + at(xw, yn + 1.0) * sw + at(xw + 1.0, yn + 1.0) * se
}

/// 64 채널 지도([64,h,w])의 쌍삼차 값(A = −0.75, 밖은 0)
fn bicubic(map: &[f32], w: usize, h: usize, x: f32, y: f32, out: &mut [f32; DIM]) {
    let (ix, iy) = (x.floor(), y.floor());
    let (cx, cy) = (cubic_coeffs(x - ix), cubic_coeffs(y - iy));
    let plane = w * h;
    let (ix, iy) = (ix as i64, iy as i64);
    let idx = |xx: i64, yy: i64| -> Option<usize> {
        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h { Some(yy as usize * w + xx as usize) } else { None }
    };
    let mut taps = [[None; 4]; 4];
    for (r, row) in taps.iter_mut().enumerate() {
        for (c, t) in row.iter_mut().enumerate() {
            *t = idx(ix - 1 + c as i64, iy - 1 + r as i64);
        }
    }
    for (ch, o) in out.iter_mut().enumerate() {
        let m = &map[ch * plane..(ch + 1) * plane];
        let v = |t: Option<usize>| t.map_or(0.0, |i| m[i]);
        let mut rows = [0f32; 4];
        for r in 0..4 {
            let t = &taps[r];
            rows[r] = cx[0] * v(t[0]) + cx[1] * v(t[1]) + cx[2] * v(t[2]) + cx[3] * v(t[3]);
        }
        *o = cy[0] * rows[0] + cy[1] * rows[1] + cy[2] * rows[2] + cy[3] * rows[3];
    }
}

/// 64 차원 내적(aarch64 는 NEON 4 칸 × 4 누적, 다른 기계는 같은 묶음 순서의 스칼라)
#[inline]
pub fn dot64(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= DIM && b.len() >= DIM);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 두 조각 모두 64 개 이상
    unsafe {
        use core::arch::aarch64::*;
        let (p, q) = (a.as_ptr(), b.as_ptr());
        let mut s = [vdupq_n_f32(0.0); 4];
        for k in 0..4 {
            for j in 0..4 {
                let o = k * 16 + j * 4;
                s[j] = vfmaq_f32(s[j], vld1q_f32(p.add(o)), vld1q_f32(q.add(o)));
            }
        }
        return vaddvq_f32(vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3])));
    }
    #[allow(unreachable_code)]
    {
        let mut s = [0f32; 16];
        for k in 0..4 {
            for j in 0..16 {
                s[j] += a[k * 16 + j] * b[k * 16 + j];
            }
        }
        s.iter().sum()
    }
}

/// 상호 최근접 짝(원본 `XFeat.match`, min_cossim ≤ 0): (질의 번호, 대상 번호). sim 은 작업 버퍼(n1×n2)
pub fn match_mnn(a: &XFeatures, b: &XFeatures, min_cossim: f32, sim: &mut Vec<f32>, out: &mut Vec<(u32, u32)>) {
    out.clear();
    let (n1, n2) = (a.len(), b.len());
    if n1 == 0 || n2 == 0 {
        return;
    }
    sim.resize(n1 * n2, 0.0);
    for i in 0..n1 {
        let di = a.d(i);
        let row = &mut sim[i * n2..(i + 1) * n2];
        for (j, r) in row.iter_mut().enumerate() {
            *r = dot64(di, b.d(j));
        }
    }
    // 열마다 가장 큰 행(같으면 앞 번호)
    let mut colbest = vec![(f32::NEG_INFINITY, 0u32); n2];
    for i in 0..n1 {
        let row = &sim[i * n2..(i + 1) * n2];
        for (j, &v) in row.iter().enumerate() {
            if v > colbest[j].0 {
                colbest[j] = (v, i as u32);
            }
        }
    }
    for i in 0..n1 {
        let row = &sim[i * n2..(i + 1) * n2];
        let (mut bj, mut bv) = (0usize, f32::NEG_INFINITY);
        for (j, &v) in row.iter().enumerate() {
            if v > bv {
                bv = v;
                bj = j;
            }
        }
        if colbest[bj].1 as usize == i && (min_cossim <= 0.0 || bv > min_cossim) {
            out.push((i as u32, bj as u32));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_coord_matches_torch_identity() {
        // 같은 크기 지도: x → x·W/(W−1) − 0.5 근처, 끝 화소는 반올림이 한 칸 밖으로(원본도 그 점은 점수 0)
        assert_eq!(grid_coord(0.0, W, W).round_ties_even(), 0.0);
        assert_eq!(grid_coord(100.0, W, W).round_ties_even(), 100.0);
        assert_eq!(grid_coord(639.0, W, W).round_ties_even(), 640.0);
    }

    #[test]
    fn cubic_coeffs_sum_to_one() {
        for t in [0.0f32, 0.25, 0.5, 0.9] {
            let c = cubic_coeffs(t);
            assert!((c.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert_eq!(cubic_coeffs(0.0), [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn mnn_finds_identity() {
        let mut f = XFeatures::default();
        for i in 0..10 {
            let mut d = [0f32; DIM];
            d[i] = 1.0;
            f.kps.push([i as f32, 0.0]);
            f.scores.push(1.0);
            f.desc.extend(d);
        }
        let (mut sim, mut out) = (Vec::new(), Vec::new());
        match_mnn(&f, &f, -1.0, &mut sim, &mut out);
        assert_eq!(out, (0..10).map(|i| (i, i)).collect::<Vec<_>>());
    }

    #[test]
    fn detect_single_peak() {
        let plane = HC * WC;
        let feats = vec![1.0f32; DIM * plane];
        let mut kp = vec![0f32; 65 * plane];
        // 칸 (10, 20) 의 채널 27 = 화소 (20*8+3, 10*8+3) 에 큰 로짓
        kp[27 * plane + 10 * WC + 20] = 12.0;
        let heat = vec![0.5f32; plane];
        let (mut sc, mut out) = (Scratch::default(), XFeatures::default());
        detect(&Maps { feats: &feats, kpts: &kp, heat: &heat }, 1000, 0.05, &mut sc, &mut out);
        assert_eq!(out.kps, vec![[163.0, 83.0]]);
        let n: f32 = out.d(0).iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-5);
    }
}
