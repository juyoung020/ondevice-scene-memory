//! ORB 추출의 FAST 를 NPU 로 돕기(docs/orb3_port.md "포터 NPU").
//!
//! NPU 모델 `orb_maps`(tools/orbnpu/orb_maps.py)는 피라미드 8 층을 한 장(모자이크)으로 받아 화소마다 FAST 점수 S
//! (16 이웃 중 이어진 9 칸 차이의 min · −max 중 큰 것, 127 에서 자름)를 낸다. 8 비트 NPU 의 S 는 정확한 값과 최대
//! `margin` 만큼 다를 수 있으므로, S 는 **후보 거르개**로만 쓴다: 문턱 th 에서 S_npu ≥ th + 1 − margin 인 화소만 CPU 가
//! 정확한 점수를 계산하고, 나머지는 구석이 아니다(정확한 점수 ≤ th). 그래서 결과(점 · 응답 · 순서)는 원본
//! `cv::FAST` · `orb3rs::extract::fast` 와 **비트 같다**(오차 한계가 맞는 한 — 보드 확인 필요).
//!
//! 맞물릴 자리(opt/orb3-rust `orb3rs::extract`):
//! 1. `compute_pyramid` 뒤 [`Mosaic::pack`] 으로 층들을 한 장에 → `nnrun::Net::submit`(보드 DXRT 비동기).
//! 2. `wait` 뒤 [`Mosaic::level_scores`] 로 층마다 S(u8, 층 크기)를 꺼냄.
//! 3. `level_keys` 의 `fast(img, iw, x0, y0, cw, ch, th, &mut cell)` 를
//!    `fast_npu(img, iw, x0, y0, cw, ch, th, &s, MARGIN, &mut cell)` 로(같은 출력). 나머지 단계는 그대로.
//! 블러 · 기술자 · 방향은 CPU 그대로(블러를 NPU 로 하면 기술자 해밍 평균 2/256 이 바뀌어 쓰지 않음).

/// 8 비트 NPU S 오차 한계(PC 8 비트 흉내에서 최대 1 · 늘 아래로) + 여유 1
pub const MARGIN: i32 = 2;
/// 원본 ORBextractor 의 층 테두리(모자이크 칸 배치가 이 테두리를 포함한 크기 — 모델 컴파일 크기와 같게)
const EDGE: usize = 19;
pub const MOS_W: usize = 1280;

/// 모자이크 배치: 층마다 (x, y, w, h) — 테두리 포함 사각형. orbnpu_bench `layout` 과 같은 규칙
pub struct Mosaic {
    pub slots: Vec<(usize, usize, usize, usize)>,
    pub h: usize,
    pub buf: Vec<u8>,
}

impl Mosaic {
    /// 층 크기(테두리 없이)로 배치를 정함
    pub fn new(level_sizes: &[(usize, usize)]) -> Mosaic {
        let (mut x, mut y, mut rowh) = (0usize, 0usize, 0usize);
        let mut slots = Vec::new();
        for &(lw, lh) in level_sizes {
            let (w, h) = (lw + 2 * EDGE, lh + 2 * EDGE);
            if x + w > MOS_W {
                x = 0;
                y += rowh;
                rowh = 0;
            }
            slots.push((x, y, w, h));
            x += w;
            rowh = rowh.max(h);
        }
        let h = (y + rowh).div_ceil(32) * 32;
        Mosaic { slots, h, buf: vec![0; h * MOS_W] }
    }

    /// 층 영상을 모자이크에 복사(테두리 칸은 0 그대로: S 를 읽는 곳(층 가장자리 13 칸 안쪽)은 테두리에 닿지 않음)
    pub fn pack(&mut self, levels: &[(&[u8], usize, usize)]) -> &[u8] {
        for (l, &(img, lw, lh)) in levels.iter().enumerate() {
            let (x, y, ..) = self.slots[l];
            for r in 0..lh {
                let o = (y + EDGE + r) * MOS_W + x + EDGE;
                self.buf[o..o + lw].copy_from_slice(&img[r * lw..(r + 1) * lw]);
            }
        }
        &self.buf
    }

    /// 모델 출력 S(모자이크 f32) → 층 l 의 S(u8, lw × lh, 반올림 · 0–255)
    pub fn level_scores(&self, s: &[f32], l: usize, lw: usize, lh: usize, out: &mut Vec<u8>) {
        let (x, y, ..) = self.slots[l];
        out.clear();
        out.reserve(lw * lh);
        for r in 0..lh {
            let o = (y + EDGE + r) * MOS_W + x + EDGE;
            out.extend(s[o..o + lw].iter().map(|&v| v.round().clamp(0.0, 255.0) as u8));
        }
    }
}

/// 화소 하나의 정확한 FAST 점수(구석 ⇔ 점수 > 문턱, cv::FAST 응답 = 점수 − 1)
#[inline]
fn exact_score(img: &[u8], c: usize, pixel: &[isize; 25]) -> i32 {
    let v = img[c] as i32;
    let mut d = [0i32; 25];
    for k in 0..25 {
        // SAFETY: 부르는 쪽이 반지름 3 원이 영상 안인 화소만 넘김(fast_npu 의 assert)
        d[k] = v - unsafe { *img.get_unchecked((c as isize + pixel[k]) as usize) } as i32;
    }
    let mut best = i32::MIN;
    for k in 0..16 {
        let (mut a, mut b) = (d[k], d[k]);
        for m in 1..9 {
            a = a.min(d[k + m]);
            b = b.max(d[k + m]);
        }
        best = best.max(a).max(-b);
    }
    best
}

/// `orb3rs::extract::fast` 와 같은 출력(cv::FAST 9/16 + 비최대 억제, 좌표는 구역 기준). `s` 는 같은 층의 NPU S
/// (stride 같음), `margin` 은 S 오차 한계
#[allow(clippy::too_many_arguments)]
pub fn fast_npu(img: &[u8], stride: usize, x0: usize, y0: usize, cw: usize, ch: usize, threshold: i32, s: &[u8], margin: i32, out: &mut Vec<KeyPointOut>) {
    const OFF: [(i32, i32); 16] = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3), (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)];
    if ch < 5 || cw < 7 {
        return;
    }
    assert!((y0 + ch - 1) * stride + x0 + cw <= img.len() && x0 + cw <= stride && s.len() == img.len());
    let st = stride as isize;
    let mut pixel = [0isize; 25];
    for k in 0..16 {
        pixel[k] = OFF[k].0 as isize + OFF[k].1 as isize * st;
    }
    for k in 16..25 {
        pixel[k] = pixel[k - 16];
    }
    let th = threshold.clamp(0, 255);
    let lo = (th + 1 - margin).clamp(0, 256);
    let mut buf = [vec![0u8; cw], vec![0u8; cw], vec![0u8; cw]];
    let mut cpos: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 3..ch - 2 {
        let ci = (i - 3) % 3;
        buf[ci].iter_mut().for_each(|v| *v = 0);
        cpos[ci].clear();
        if i < ch - 3 {
            let row = (y0 + i) * stride + x0;
            let srow = &s[row + 3..row + cw.saturating_sub(3).max(3)];
            // 후보(S_npu ≥ lo)만: 대부분 줄은 비교 한 번에 지나감
            for (jj, &sv) in srow.iter().enumerate() {
                if (sv as i32) < lo {
                    continue;
                }
                let j = jj + 3;
                let best = exact_score(img, row + j, &pixel);
                if best > th {
                    cpos[ci].push(j);
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
            let v = prev[j];
            if v > prev[j + 1] && v > prev[j - 1] && v > pprev[j - 1] && v > pprev[j] && v > pprev[j + 1] && v > curr[j - 1] && v > curr[j] && v > curr[j + 1] {
                out.push(KeyPointOut { x: j as f32, y: (i - 1) as f32, response: v as f32 });
            }
        }
    }
}

/// fast_npu 출력(orb3rs `KeyPoint { x, y, size: 7, angle: -1, response, octave: 0, class_id: -1 }` 로 그대로 옮김)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyPointOut {
    pub x: f32,
    pub y: f32,
    pub response: f32,
}
