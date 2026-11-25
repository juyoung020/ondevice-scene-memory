//! ORBextractor(ORB-SLAM3 src/ORBextractor.cc) 옮김: 피라미드 → 칸별 FAST → 사분 트리 분배 → 방향(IC_Angle) → 흐림 → rBRIEF.
//! 원본과 비트 동일(점 순서 · 좌표 · 크기 · 각도 · 응답 · 기술자). 원본이 부르는 OpenCV 함수도 같은 정수 · 실수 계산으로 옮겼다:
//! - cv::resize INTER_LINEAR 8U(resize.cpp: 계수 11 비트 고정소수, 세로는 SIMD 식 `((S>>4)*b)>>16` 합 + 2 >> 2 — 4.6 은 줄 끝까지 이 식)
//! - cv::GaussianBlur 7×7 σ2 8U(smooth: 비트 정확 고정소수 커널 [18,34,48,56,48,34,18]/256, 반사 101)
//! - cv::FAST 9/16 + 비최대 억제(fast.cpp: 점수 = 연속 9 칸 차이의 최소 · 최대 중 큰 것 − 1)
//! - cv::fastAtan2(mathfuncs_core: 7 차 다항식), cvRound(짝수 반올림)
//! - std::sort(사분 트리 노드 정렬은 불안정 정렬이라 libstdc++ 알고리즘 그대로: stdsort)
use super::stdsort;
use super::types::{Desc, KeyPoint, Pyramid};

const PATCH_SIZE: i32 = 31;
const HALF_PATCH_SIZE: i32 = 15;
const EDGE_THRESHOLD: i32 = 19;

/// cvRound(float): x86 은 OpenCV 와 같은 cvtss2si(짝수 반올림), 그 밖은 round_ties_even(aarch64 fcvtns 와 같음)
#[inline]
pub fn cv_round(v: f32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE 는 x86_64 기본
    unsafe {
        use std::arch::x86_64::*;
        _mm_cvtss_si32(_mm_set_ss(v))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v.round_ties_even() as i32
    }
}
#[inline]
fn cv_round_d(v: f64) -> i32 {
    v.round_ties_even() as i32
}

impl Pyramid {
    /// ORBextractor 생성자가 만드는 층 표
    pub fn new(nfeatures: i32, scale_factor: f32, nlevels: i32, ini_th_fast: i32, min_th_fast: i32) -> Pyramid {
        let n = nlevels as usize;
        let mut scale = vec![1.0f32; n];
        let mut sigma2 = vec![1.0f32; n];
        for i in 1..n {
            scale[i] = scale[i - 1] * scale_factor;
            sigma2[i] = scale[i] * scale[i];
        }
        let inv_scale = scale.iter().map(|s| 1.0f32 / s).collect();
        let inv_sigma2 = sigma2.iter().map(|s| 1.0f32 / s).collect();
        Pyramid { nfeatures, scale_factor, nlevels, ini_th_fast, min_th_fast, scale, inv_scale, sigma2, inv_sigma2, log_scale_factor: scale_factor.ln() }
    }
}

pub struct Extractor {
    pub p: Pyramid,
    feats_per_level: Vec<i32>,
    umax: [i32; (HALF_PATCH_SIZE + 1) as usize],
    /// 층 영상(테두리 없이: 원본의 테두리 칸은 어느 계산도 읽지 않음)
    levels: Vec<(Vec<u8>, usize, usize)>,
}

impl Extractor {
    pub fn new(p: Pyramid) -> Extractor {
        let n = p.nlevels as usize;
        let factor = 1.0f32 / p.scale_factor;
        let mut per = (p.nfeatures as f32) * (1.0 - factor) / (1.0 - (factor as f64).powf(p.nlevels as f64) as f32);
        let mut feats = vec![0i32; n];
        let mut sum = 0;
        for l in 0..n - 1 {
            feats[l] = cv_round(per);
            sum += feats[l];
            per *= factor;
        }
        feats[n - 1] = (p.nfeatures - sum).max(0);
        let mut umax = [0i32; 16];
        let vmax = (HALF_PATCH_SIZE as f32 * 2f32.sqrt() / 2.0 + 1.0).floor() as i32;
        let vmin = (HALF_PATCH_SIZE as f32 * 2f32.sqrt() / 2.0).ceil() as i32;
        let hp2 = (HALF_PATCH_SIZE * HALF_PATCH_SIZE) as f64;
        for v in 0..=vmax {
            umax[v as usize] = cv_round_d((hp2 - (v * v) as f64).sqrt());
        }
        let mut v0 = 0usize;
        let mut v = HALF_PATCH_SIZE;
        while v >= vmin {
            while umax[v0] == umax[v0 + 1] {
                v0 += 1;
            }
            umax[v as usize] = v0 as i32;
            v0 += 1;
            v -= 1;
        }
        Extractor { p, feats_per_level: feats, umax, levels: Vec::new() }
    }

    /// 층 영상(디버그 · 비교용)
    pub fn level(&self, l: usize) -> (&[u8], usize, usize) {
        let (ref v, w, h) = self.levels[l];
        (v, w, h)
    }

    /// 원본 operator()(image, mask 없음, vLappingArea = [0, 0]): 특징점(0 층 좌표) · 기술자
    pub fn extract(&mut self, gray: &[u8], w: usize, h: usize) -> (Vec<KeyPoint>, Vec<Desc>) {
        use rayon::prelude::*;
        self.compute_pyramid(gray, w, h);
        // 층마다 따로(원본은 한 스레드로 층을 차례로 돎; 층끼리 주고받는 값이 없어 결과 같음)
        let per: Vec<(Vec<KeyPoint>, Vec<Desc>)> = (0..self.p.nlevels as usize)
            .into_par_iter()
            .map(|level| {
                let mut keys = self.level_keys(level);
                if keys.is_empty() {
                    return (keys, Vec::new());
                }
                let (ref img, lw, lh) = self.levels[level];
                let mut blur = Vec::new();
                gaussian7(img, lw, lh, &mut blur);
                let scale = self.p.scale[level];
                let descs = keys
                    .iter_mut()
                    .map(|k| {
                        let d = orb_descriptor(k, &blur, lw);
                        if level != 0 {
                            k.x *= scale;
                            k.y *= scale;
                        }
                        d
                    })
                    .collect();
                (keys, descs)
            })
            .collect();
        let total: usize = per.iter().map(|k| k.0.len()).sum();
        let mut kps = vec![KeyPoint::default(); total];
        let mut descs = vec![[0u8; 32]; total];
        let (mut mono, mut stereo) = (0usize, total as isize - 1);
        for (keys, ds) in per {
            for (k, d) in keys.into_iter().zip(ds) {
                // vLappingArea = [0, 0](RGB-D): 원본 순서 규칙 그대로
                if k.x >= 0.0 && k.x <= 0.0 {
                    kps[stereo as usize] = k;
                    descs[stereo as usize] = d;
                    stereo -= 1;
                } else {
                    kps[mono] = k;
                    descs[mono] = d;
                    mono += 1;
                }
            }
        }
        (kps, descs)
    }

    fn compute_pyramid(&mut self, gray: &[u8], w: usize, h: usize) {
        let n = self.p.nlevels as usize;
        self.levels.resize_with(n, || (Vec::new(), 0, 0));
        for l in 0..n {
            let s = self.p.inv_scale[l];
            let (lw, lh) = (cv_round(w as f32 * s) as usize, cv_round(h as f32 * s) as usize);
            let mut dst = std::mem::take(&mut self.levels[l].0);
            dst.resize(lw * lh, 0);
            if l == 0 {
                dst.copy_from_slice(&gray[..w * h]);
            } else {
                let (ref src, sw, sh) = self.levels[l - 1];
                resize_linear(src, sw, sh, &mut dst, lw, lh);
            }
            self.levels[l] = (dst, lw, lh);
        }
    }

    /// 한 층의 특징점(ComputeKeyPointsOctTree 의 층 하나 + computeOrientation), 층 좌표
    fn level_keys(&self, level: usize) -> Vec<KeyPoint> {
        const W: f32 = 35.0;
        {
            let (ref img, iw, ih) = self.levels[level];
            let min_bx = EDGE_THRESHOLD - 3;
            let min_by = min_bx;
            let max_bx = iw as i32 - EDGE_THRESHOLD + 3;
            let max_by = ih as i32 - EDGE_THRESHOLD + 3;
            let mut to_dist: Vec<KeyPoint> = Vec::with_capacity(self.p.nfeatures as usize * 10);
            let width = (max_bx - min_bx) as f32;
            let height = (max_by - min_by) as f32;
            let ncols = (width / W) as i32;
            let nrows = (height / W) as i32;
            let wcell = (width / ncols as f32).ceil() as i32;
            let hcell = (height / nrows as f32).ceil() as i32;
            let mut cell = Vec::new();
            for i in 0..nrows {
                let ini_y = (min_by + i * hcell) as f32;
                let mut max_y = ini_y + hcell as f32 + 6.0;
                if ini_y >= (max_by - 3) as f32 {
                    continue;
                }
                if max_y > max_by as f32 {
                    max_y = max_by as f32;
                }
                for j in 0..ncols {
                    let ini_x = (min_bx + j * wcell) as f32;
                    let mut max_x = ini_x + wcell as f32 + 6.0;
                    if ini_x >= (max_bx - 6) as f32 {
                        continue;
                    }
                    if max_x > max_bx as f32 {
                        max_x = max_bx as f32;
                    }
                    // Mat::rowRange · colRange(float → int 잘라냄)
                    let (y0, y1, x0, x1) = (ini_y as usize, max_y as usize, ini_x as usize, max_x as usize);
                    cell.clear();
                    fast(img, iw, x0, y0, x1 - x0, y1 - y0, self.p.ini_th_fast, &mut cell);
                    if cell.is_empty() {
                        fast(img, iw, x0, y0, x1 - x0, y1 - y0, self.p.min_th_fast, &mut cell);
                    }
                    for k in cell.iter() {
                        let mut k = *k;
                        k.x += (j * wcell) as f32;
                        k.y += (i * hcell) as f32;
                        to_dist.push(k);
                    }
                }
            }
            let mut keys = distribute_octtree(&to_dist, min_bx, max_bx, min_by, max_by, self.feats_per_level[level], self.p.nfeatures);
            let scaled_patch = (PATCH_SIZE as f32 * self.p.scale[level]) as i32;
            for k in keys.iter_mut() {
                k.x += min_bx as f32;
                k.y += min_by as f32;
                k.octave = level as i32;
                k.size = scaled_patch as f32;
                k.angle = ic_angle(img, iw, k.x, k.y, &self.umax);
            }
            keys
        }
    }
}

/// cv::resize(INTER_LINEAR) 8U 1 채널, 크기를 정해 줌(inv_scale = 목표 / 원본)
pub fn resize_linear(src: &[u8], sw: usize, sh: usize, dst: &mut [u8], dw: usize, dh: usize) {
    let inv_x = dw as f64 / sw as f64;
    let inv_y = dh as f64 / sh as f64;
    let (scale_x, scale_y) = (1.0 / inv_x, 1.0 / inv_y);
    let mut xofs = vec![0usize; dw];
    let mut alpha = vec![[0i16; 2]; dw];
    let mut xmax = dw;
    for dx in 0..dw {
        let mut fx = ((dx as f64 + 0.5) * scale_x - 0.5) as f32;
        let mut sx = fx.floor() as i32;
        fx -= sx as f32;
        if sx < 0 {
            fx = 0.0;
            sx = 0;
        }
        if sx + 1 >= sw as i32 {
            xmax = xmax.min(dx);
            if sx >= sw as i32 - 1 {
                fx = 0.0;
                sx = sw as i32 - 1;
            }
        }
        xofs[dx] = sx as usize;
        alpha[dx] = [sat_i16(cv_round((1.0 - fx) * 2048.0)), sat_i16(cv_round(fx * 2048.0))];
    }
    let mut yofs = vec![0i32; dh];
    let mut beta = vec![[0i16; 2]; dh];
    for dy in 0..dh {
        let mut fy = ((dy as f64 + 0.5) * scale_y - 0.5) as f32;
        let sy = fy.floor() as i32;
        fy -= sy as f32;
        yofs[dy] = sy;
        beta[dy] = [sat_i16(cv_round((1.0 - fy) * 2048.0)), sat_i16(cv_round(fy * 2048.0))];
    }
    let hrow = |sy: usize, out: &mut [i32]| {
        let s = &src[sy * sw..sy * sw + sw];
        for dx in 0..xmax {
            let sx = xofs[dx];
            out[dx] = s[sx] as i32 * alpha[dx][0] as i32 + s[sx + 1] as i32 * alpha[dx][1] as i32;
        }
        for dx in xmax..dw {
            out[dx] = s[xofs[dx]] as i32 * 2048;
        }
    };
    let (mut r0, mut r1) = (vec![0i32; dw], vec![0i32; dw]);
    let (mut c0, mut c1) = (usize::MAX, usize::MAX);
    for dy in 0..dh {
        let clip = |y: i32| -> usize { if y < 0 { 0 } else if y >= sh as i32 { sh - 1 } else { y as usize } };
        let (y0, y1) = (clip(yofs[dy]), clip(yofs[dy] + 1));
        if y0 != c0 {
            if y0 == c1 {
                std::mem::swap(&mut r0, &mut r1);
                c0 = c1;
                c1 = usize::MAX;
            } else {
                hrow(y0, &mut r0);
                c0 = y0;
            }
        }
        if y1 != c1 {
            if y1 == c0 {
                r1.copy_from_slice(&r0);
            } else {
                hrow(y1, &mut r1);
            }
            c1 = y1;
        }
        let (b0, b1) = (beta[dy][0] as i32, beta[dy][1] as i32);
        let d = &mut dst[dy * dw..dy * dw + dw];
        // 시스템 OpenCV 4.6 은 줄 전체를 SIMD 식으로 계산(끝 칸도): ((S >> 4) · b) >> 16 두 개 합 → (+2) >> 2
        for x in 0..dw {
            let a = ((r0[x] >> 4) * b0) >> 16;
            let b = ((r1[x] >> 4) * b1) >> 16;
            d[x] = ((a + b + 2) >> 2).clamp(0, 255) as u8;
        }
    }
}

#[inline]
fn sat_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// cv::GaussianBlur(7×7, σ = 2, BORDER_REFLECT_101) 8U: 비트 정확 고정소수 경로
pub fn gaussian7(src: &[u8], w: usize, h: usize, dst: &mut Vec<u8>) {
    // 정수 합이라 계산 순서와 무관하게 같은 값: 가로 Σk·p(≤ 65280, u16) → 세로 Σk·가로 + 2^15 >> 16
    fn refl(p: isize, n: usize) -> usize {
        let n = n as isize;
        if n == 1 {
            return 0;
        }
        let mut p = p;
        loop {
            if p < 0 {
                p = -p;
            } else if p >= n {
                p = 2 * n - 2 - p;
            } else {
                return p as usize;
            }
        }
    }
    let mut hor = vec![0u16; w * h];
    let mut pad = vec![0u16; w + 6];
    for y in 0..h {
        let s = &src[y * w..y * w + w];
        for (i, v) in pad.iter_mut().enumerate() {
            *v = s[refl(i as isize - 3, w)] as u16;
        }
        let o = &mut hor[y * w..y * w + w];
        let (p0, p1, p2, p3, p4, p5, p6) = (&pad[0..w], &pad[1..w + 1], &pad[2..w + 2], &pad[3..w + 3], &pad[4..w + 4], &pad[5..w + 5], &pad[6..w + 6]);
        for x in 0..w {
            o[x] = 18 * (p0[x] + p6[x]) + 34 * (p1[x] + p5[x]) + 48 * (p2[x] + p4[x]) + 56 * p3[x];
        }
    }
    dst.resize(w * h, 0);
    for y in 0..h {
        let r: [&[u16]; 7] = std::array::from_fn(|k| {
            let ry = refl(y as isize + k as isize - 3, h);
            &hor[ry * w..ry * w + w]
        });
        let o = &mut dst[y * w..y * w + w];
        for x in 0..w {
            let acc = 18 * (r[0][x] as u32 + r[6][x] as u32) + 34 * (r[1][x] as u32 + r[5][x] as u32) + 48 * (r[2][x] as u32 + r[4][x] as u32) + 56 * r[3][x] as u32;
            o[x] = ((acc + (1 << 15)) >> 16).min(255) as u8;
        }
    }
}

/// cv::FAST(TYPE_9_16, 비최대 억제) — 영상 img(한 줄 stride 칸)의 (x0, y0) 부터 cw × ch 구역. 좌표는 구역 기준
pub fn fast(img: &[u8], stride: usize, x0: usize, y0: usize, cw: usize, ch: usize, threshold: i32, out: &mut Vec<KeyPoint>) {
    const OFF: [(i32, i32); 16] = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)];
    let st = stride as isize;
    let mut pixel = [0isize; 25];
    for k in 0..16 {
        pixel[k] = OFF[k].0 as isize + OFF[k].1 as isize * st;
    }
    for k in 16..25 {
        pixel[k] = pixel[k - 16];
    }
    let th = threshold.clamp(0, 255);
    let mut tab = [0u8; 512];
    for i in -255i32..=255 {
        tab[(i + 255) as usize] = if i < -th { 1 } else if i > th { 2 } else { 0 };
    }
    let mut buf = [vec![0u8; cw], vec![0u8; cw], vec![0u8; cw]];
    let mut cpos: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    if ch < 5 || cw < 7 {
        return;
    }
    assert!((y0 + ch - 1) * stride + x0 + cw <= img.len() && x0 + cw <= stride);
    for i in 3..ch - 2 {
        let ci = (i - 3) % 3;
        buf[ci].iter_mut().for_each(|v| *v = 0);
        cpos[ci].clear();
        if i < ch - 3 {
            let row = (y0 + i) as isize * st + x0 as isize;
            for j in 3..cw.saturating_sub(3) {
                let c = (row + j as isize) as usize;
                let v = img[c] as i32;
                // SAFETY: 구역 안 (3 ≤ j < cw-3, 3 ≤ i < ch-3) 이라 반지름 3 원은 구역 안(위 assert)
                let px = |k: usize| unsafe { *img.get_unchecked((c as isize + pixel[k]) as usize) } as i32;
                let t = |x: i32| tab[(x - v + 255) as usize];
                let mut d = t(px(0)) | t(px(8));
                if d == 0 {
                    continue;
                }
                d &= t(px(2)) | t(px(10));
                d &= t(px(4)) | t(px(12));
                d &= t(px(6)) | t(px(14));
                if d == 0 {
                    continue;
                }
                d &= t(px(1)) | t(px(9));
                d &= t(px(3)) | t(px(11));
                d &= t(px(5)) | t(px(13));
                d &= t(px(7)) | t(px(15));
                let mut corner = false;
                if d & 1 != 0 {
                    let vt = v - th;
                    let mut count = 0;
                    for k in 0..25 {
                        if px(k) < vt {
                            count += 1;
                            if count > 8 {
                                corner = true;
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }
                if !corner && d & 2 != 0 {
                    let vt = v + th;
                    let mut count = 0;
                    for k in 0..25 {
                        if px(k) > vt {
                            count += 1;
                            if count > 8 {
                                corner = true;
                                break;
                            }
                        } else {
                            count = 0;
                        }
                    }
                }
                if corner {
                    cpos[ci].push(j);
                    let mut dd = [0i32; 25];
                    for k in 0..25 {
                        dd[k] = v - px(k);
                    }
                    let mut best = i32::MIN;
                    for k in 0..16 {
                        let (mut a, mut b) = (dd[k], dd[k]);
                        for m in 1..9 {
                            a = a.min(dd[k + m]);
                            b = b.max(dd[k + m]);
                        }
                        best = best.max(a).max(-b);
                    }
                    buf[ci][j] = (best - 1) as u8;
                }
            }
        }
        if i == 3 {
            continue;
        }
        let (pi, ppi) = ((i - 4 + 3) % 3, (i - 5 + 3) % 3);
        let (prev, pprev, curr) = (&buf[pi], &buf[ppi], &buf[ci]);
        for &j in cpos[pi].iter() {
            let s = prev[j];
            if s > prev[j + 1] && s > prev[j - 1] && s > pprev[j - 1] && s > pprev[j] && s > pprev[j + 1] && s > curr[j - 1] && s > curr[j] && s > curr[j + 1] {
                out.push(KeyPoint { x: j as f32, y: (i - 1) as f32, size: 7.0, angle: -1.0, response: s as f32, octave: 0, class_id: -1 });
            }
        }
    }
}

struct Node {
    ul: (i32, i32),
    ur: (i32, i32),
    bl: (i32, i32),
    br: (i32, i32),
    keys: Vec<KeyPoint>,
    no_more: bool,
    prev: usize,
    next: usize,
}

const NIL: usize = usize::MAX;

/// std::list<ExtractorNode> 흉내: 앞에 넣기 · 지우기 · 앞에서부터 돌기
struct List {
    nodes: Vec<Node>,
    head: usize,
    len: usize,
}

impl List {
    fn push_front(&mut self, mut n: Node) -> usize {
        n.prev = NIL;
        n.next = self.head;
        let id = self.nodes.len();
        if self.head != NIL {
            self.nodes[self.head].prev = id;
        }
        self.nodes.push(n);
        self.head = id;
        self.len += 1;
        id
    }
    fn push_back_tail(&mut self, tail: &mut usize, mut n: Node) -> usize {
        n.prev = *tail;
        n.next = NIL;
        let id = self.nodes.len();
        if *tail != NIL {
            self.nodes[*tail].next = id;
        } else {
            self.head = id;
        }
        self.nodes.push(n);
        *tail = id;
        self.len += 1;
        id
    }
    /// 지우고 다음 노드
    fn erase(&mut self, id: usize) -> usize {
        let (p, n) = (self.nodes[id].prev, self.nodes[id].next);
        if p != NIL {
            self.nodes[p].next = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.nodes[n].prev = p;
        }
        self.len -= 1;
        self.nodes[id].keys = Vec::new();
        n
    }
}

fn divide(n: &Node) -> [Node; 4] {
    let half_x = ((n.ur.0 - n.ul.0) as f32 / 2.0).ceil() as i32;
    let half_y = ((n.br.1 - n.ul.1) as f32 / 2.0).ceil() as i32;
    let mk = |ul, ur, bl, br| Node { ul, ur, bl, br, keys: Vec::with_capacity(n.keys.len()), no_more: false, prev: NIL, next: NIL };
    let n1ur = (n.ul.0 + half_x, n.ul.1);
    let n1bl = (n.ul.0, n.ul.1 + half_y);
    let n1br = (n.ul.0 + half_x, n.ul.1 + half_y);
    let mut c = [
        mk(n.ul, n1ur, n1bl, n1br),
        mk(n1ur, n.ur, n1br, (n.ur.0, n.ul.1 + half_y)),
        mk(n1bl, n1br, n.bl, (n1br.0, n.bl.1)),
        mk(n1br, (n.ur.0, n.ul.1 + half_y), (n1br.0, n.bl.1), n.br),
    ];
    for k in n.keys.iter() {
        let i = if k.x < n1ur.0 as f32 {
            if k.y < n1br.1 as f32 { 0 } else { 2 }
        } else if k.y < n1br.1 as f32 {
            1
        } else {
            3
        };
        c[i].keys.push(*k);
    }
    for ch in c.iter_mut() {
        if ch.keys.len() == 1 {
            ch.no_more = true;
        }
    }
    c
}

fn distribute_octtree(keys: &[KeyPoint], min_x: i32, max_x: i32, min_y: i32, max_y: i32, n: i32, nfeatures: i32) -> Vec<KeyPoint> {
    let n_ini = ((max_x - min_x) as f32 / (max_y - min_y) as f32).round() as i32;
    let hx = (max_x - min_x) as f32 / n_ini as f32;
    let mut l = List { nodes: Vec::with_capacity(keys.len() / 2 + 16), head: NIL, len: 0 };
    let mut tail = NIL;
    let mut ini = Vec::with_capacity(n_ini as usize);
    for i in 0..n_ini {
        let ulx = (hx * i as f32) as i32;
        let urx = (hx * (i + 1) as f32) as i32;
        let node = Node { ul: (ulx, 0), ur: (urx, 0), bl: (ulx, max_y - min_y), br: (urx, max_y - min_y), keys: Vec::with_capacity(keys.len()), no_more: false, prev: NIL, next: NIL };
        ini.push(l.push_back_tail(&mut tail, node));
    }
    for k in keys {
        let idx = (k.x / hx) as usize;
        l.nodes[ini[idx]].keys.push(*k);
    }
    let mut it = l.head;
    while it != NIL {
        if l.nodes[it].keys.len() == 1 {
            l.nodes[it].no_more = true;
            it = l.nodes[it].next;
        } else if l.nodes[it].keys.is_empty() {
            it = l.erase(it);
        } else {
            it = l.nodes[it].next;
        }
    }
    let mut finish = false;
    let mut size_ptr: Vec<(i32, usize)> = Vec::with_capacity(l.len * 4);
    let mut less = |a: &(i32, usize), b: &(i32, usize), nodes: &Vec<Node>| -> bool {
        if a.0 < b.0 {
            true
        } else if a.0 > b.0 {
            false
        } else {
            nodes[a.1].ul.0 < nodes[b.1].ul.0
        }
    };
    while !finish {
        let mut prev_size = l.len;
        let mut it = l.head;
        let mut n_expand = 0;
        size_ptr.clear();
        while it != NIL {
            if l.nodes[it].no_more {
                it = l.nodes[it].next;
                continue;
            }
            let ch = divide(&l.nodes[it]);
            for c in ch {
                if !c.keys.is_empty() {
                    let s = c.keys.len();
                    let id = l.push_front(c);
                    if s > 1 {
                        n_expand += 1;
                        size_ptr.push((s as i32, id));
                    }
                }
            }
            it = l.erase(it);
        }
        if l.len as i32 >= n || l.len == prev_size {
            finish = true;
        } else if (l.len as i32 + n_expand * 3) > n {
            while !finish {
                prev_size = l.len;
                let mut prev_sp = size_ptr.clone();
                size_ptr.clear();
                {
                    let nodes = &l.nodes;
                    stdsort::sort_by(&mut prev_sp, &mut |a, b| less(a, b, nodes));
                }
                for j in (0..prev_sp.len()).rev() {
                    let ch = divide(&l.nodes[prev_sp[j].1]);
                    for c in ch {
                        if !c.keys.is_empty() {
                            let s = c.keys.len();
                            let id = l.push_front(c);
                            if s > 1 {
                                size_ptr.push((s as i32, id));
                            }
                        }
                    }
                    l.erase(prev_sp[j].1);
                    if l.len as i32 >= n {
                        break;
                    }
                }
                if l.len as i32 >= n || l.len == prev_size {
                    finish = true;
                }
            }
        }
    }
    let mut res = Vec::with_capacity(nfeatures as usize);
    let mut it = l.head;
    while it != NIL {
        let ks = &l.nodes[it].keys;
        let mut best = 0;
        let mut maxr = ks[0].response;
        for k in 1..ks.len() {
            if ks[k].response > maxr {
                best = k;
                maxr = ks[k].response;
            }
        }
        res.push(ks[best]);
        it = l.nodes[it].next;
    }
    res
}

/// cv::fastAtan2(도)
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    let r = (180.0 / std::f64::consts::PI) as f32;
    let (p1, p3, p5, p7) = (0.9997878412794807f32 * r, -0.3258083974640975f32 * r, 0.1555786518463281f32 * r, -0.04432655554792128f32 * r);
    let (ax, ay) = (x.abs(), y.abs());
    let mut a;
    if ax >= ay {
        let c = ay / (ax + f64::EPSILON as f32);
        let c2 = c * c;
        a = (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    } else {
        let c = ax / (ay + f64::EPSILON as f32);
        let c2 = c * c;
        a = 90.0 - (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    }
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

fn ic_angle(img: &[u8], stride: usize, x: f32, y: f32, umax: &[i32; 16]) -> f32 {
    let c = cv_round(y) as isize * stride as isize + cv_round(x) as isize;
    let st = stride as isize;
    assert!(c - 15 * st - 15 >= 0 && ((c + 15 * st + 15) as usize) < img.len());
    // SAFETY: 원 반지름 15 안(위 assert)
    let px = |o: isize| unsafe { *img.get_unchecked((c + o) as usize) } as i32;
    let (mut m01, mut m10) = (0i32, 0i32);
    for u in -HALF_PATCH_SIZE..=HALF_PATCH_SIZE {
        m10 += u * px(u as isize);
    }
    for v in 1..=HALF_PATCH_SIZE {
        let mut vsum = 0;
        let d = umax[v as usize];
        for u in -d..=d {
            let (p, m) = (px(u as isize + v as isize * st), px(u as isize - v as isize * st));
            vsum += p - m;
            m10 += u * (p + m);
        }
        m01 += v * vsum;
    }
    fast_atan2(m01 as f32, m10 as f32)
}

fn orb_descriptor(k: &KeyPoint, img: &[u8], stride: usize) -> Desc {
    let factor_pi = (std::f64::consts::PI / 180.0f32 as f64) as f32;
    let angle = k.angle * factor_pi;
    let (a, b) = (angle.cos(), angle.sin());
    let c = cv_round(k.y) as isize * stride as isize + cv_round(k.x) as isize;
    let st = stride as isize;
    // 무늬 좌표 |x|,|y| ≤ 13 → 돌려도 반지름 ≤ 19
    assert!(c - 19 * st - 19 >= 0 && ((c + 19 * st + 19) as usize) < img.len());
    let get = |idx: usize| -> u8 {
        let (px, py) = (BIT_PATTERN_31[idx * 2] as f32, BIT_PATTERN_31[idx * 2 + 1] as f32);
        // SAFETY: 위 assert
        unsafe { *img.get_unchecked((c + cv_round(px * b + py * a) as isize * st + cv_round(px * a - py * b) as isize) as usize) }
    };
    let mut d = [0u8; 32];
    for i in 0..32 {
        let base = i * 16;
        let mut val = 0u8;
        for bit in 0..8 {
            let (t0, t1) = (get(base + bit * 2), get(base + bit * 2 + 1));
            val |= ((t0 < t1) as u8) << bit;
        }
        d[i] = val;
    }
    d
}

include!("bit_pattern_31.in");
