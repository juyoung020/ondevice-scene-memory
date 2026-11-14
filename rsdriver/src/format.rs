//! 화소 형식 바꾸기. YUYV(YUY2) → RGB8 은 librealsense v2.59.1 `unpack_yuy2<RGB8>` 과 같은 정수 계산.
//! 이 라이브러리는 빌드마다 식이 다르다(`src/proc/color-formats-converter.cpp`, `src/proc/neon/image-neon.cpp`):
//! - [`Yuy2Rule::Sse`] (x86 SSSE3 빌드): R = (298c>>8) + (409e>>8), 항마다 버림 후 합, 0..255 로 자름
//! - [`Yuy2Rule::Neon`] (aarch64 BUILD_WITH_NEON): 같은 식을 i16 곱(넘치면 감김)으로. U 가 끝값(|U−128| ≥ 127)이면 B 가 감김
//! - [`Yuy2Rule::Generic`] (그 밖): (298c + 409e + 128) >> 8
//! [`Yuy2Rule::NATIVE`] = 이 빌드 대상에서 librealsense 가 쓰는 규칙.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yuy2Rule {
    Sse,
    Neon,
    Generic,
}

impl Yuy2Rule {
    #[cfg(target_arch = "aarch64")]
    pub const NATIVE: Yuy2Rule = Yuy2Rule::Neon;
    #[cfg(not(target_arch = "aarch64"))]
    pub const NATIVE: Yuy2Rule = Yuy2Rule::Sse;
}

#[inline(always)]
fn clamp(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline(always)]
fn px(y: u8, u: u8, v: u8, rule: Yuy2Rule) -> [u8; 3] {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    match rule {
        Yuy2Rule::Sse => {
            // _mm_mulhi_epi16(x<<4, k<<4) = (x*k) >> 8 (산술), 합은 i16 범위 안
            let m = |x: i32, k: i32| (x * k) >> 8;
            [clamp(m(c, 298) + m(e, 409)), clamp(m(c, 298) - m(d, 100) - m(e, 208)), clamp(m(c, 298) + m(d, 516))]
        }
        Yuy2Rule::Neon => {
            // vmulq_s16 는 하위 16 비트(감김), vshrq_n_s16 8 은 산술, 더하기·빼기도 i16 감김, 끝에 max(0)·포화 u8
            let m = |x: i32, k: i32| (((x as i16).wrapping_mul(k as i16)) >> 8) as i16;
            let c16 = c as i16;
            let t0 = m(c, 298 - 256).wrapping_add(c16);
            let t1 = d as i16;
            let t2 = e as i16;
            let r = t0.wrapping_add(m(e, 409 - 256).wrapping_add(t2));
            let g = t0.wrapping_sub(m(d, 100)).wrapping_sub(m(e, 208));
            let b = t0.wrapping_add(m(d, 516 - 256).wrapping_add(t1));
            let sat = |x: i16| (x.max(0) as u16).min(255) as u8;
            [sat(r), sat(g), sat(b)]
        }
        Yuy2Rule::Generic => {
            [clamp((298 * c + 409 * e + 128) >> 8), clamp((298 * c - 100 * d - 208 * e + 128) >> 8), clamp((298 * c + 516 * d + 128) >> 8)]
        }
    }
}

/// YUYV(가로 w, 줄 간격 stride 바이트) → RGB8(빽빽하게 w*3). 줄 수 = out.len() / (w*3)
pub fn yuyv_to_rgb8(src: &[u8], stride: usize, w: usize, out: &mut [u8], rule: Yuy2Rule) {
    let h = out.len() / (w * 3);
    for r in 0..h {
        let s = &src[r * stride..r * stride + w * 2];
        let o = &mut out[r * w * 3..(r + 1) * w * 3];
        for (q, os) in s.chunks_exact(4).zip(o.chunks_exact_mut(6)) {
            let a = px(q[0], q[1], q[3], rule);
            let b = px(q[2], q[1], q[3], rule);
            os[..3].copy_from_slice(&a);
            os[3..].copy_from_slice(&b);
        }
    }
}

/// RGB 화소 하나의 회색조: OpenCV cvtColor(BGR2GRAY) 15 비트 계수(slam `io::to_gray` 의 RGB 차례와 같은 식)
#[inline(always)]
pub fn gray_of(p: &[u8; 3]) -> u8 {
    ((p[2] as u32 * 3735 + p[1] as u32 * 19235 + p[0] as u32 * 9798 + (1 << 14)) >> 15) as u8
}

/// `yuyv_to_rgb8` + 회색조를 한 번 훑기로(값은 RGB 를 만든 뒤 `gray_of` 를 따로 돌린 것과 같음). gray 길이 = out.len() / 3
pub fn yuyv_to_rgb8_gray(src: &[u8], stride: usize, w: usize, out: &mut [u8], gray: &mut [u8], rule: Yuy2Rule) {
    let h = out.len() / (w * 3);
    for r in 0..h {
        let s = &src[r * stride..r * stride + w * 2];
        let o = &mut out[r * w * 3..(r + 1) * w * 3];
        let g = &mut gray[r * w..(r + 1) * w];
        for ((q, os), gs) in s.chunks_exact(4).zip(o.chunks_exact_mut(6)).zip(g.chunks_exact_mut(2)) {
            let a = px(q[0], q[1], q[3], rule);
            let b = px(q[2], q[1], q[3], rule);
            os[..3].copy_from_slice(&a);
            os[3..].copy_from_slice(&b);
            gs[0] = gray_of(&a);
            gs[1] = gray_of(&b);
        }
    }
}

/// Z16(줄 간격 stride 바이트) → u16 빽빽하게
pub fn z16_copy(src: &[u8], stride: usize, w: usize, out: &mut [u16]) {
    let h = out.len() / w;
    for r in 0..h {
        let s = &src[r * stride..r * stride + w * 2];
        for (o, b) in out[r * w..(r + 1) * w].iter_mut().zip(s.chunks_exact(2)) {
            *o = u16::from_le_bytes([b[0], b[1]]);
        }
    }
}

#[cfg(test)]
mod fused_tests {
    use super::*;

    #[test]
    fn rgb_gray_fused_same() {
        let (w, h, stride) = (64usize, 9usize, 64 * 2 + 6);
        let mut s = 7u64;
        let src: Vec<u8> = (0..stride * h).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 56) as u8 }).collect();
        for rule in [Yuy2Rule::Sse, Yuy2Rule::Neon, Yuy2Rule::Generic] {
            let mut a = vec![0u8; w * h * 3];
            yuyv_to_rgb8(&src, stride, w, &mut a, rule);
            let ga: Vec<u8> = a.chunks_exact(3).map(|p| gray_of(&[p[0], p[1], p[2]])).collect();
            let (mut b, mut gb) = (vec![1u8; w * h * 3], vec![1u8; w * h]);
            yuyv_to_rgb8_gray(&src, stride, w, &mut b, &mut gb, rule);
            assert_eq!(a, b);
            assert_eq!(ga, gb);
        }
        // 모든 Y·U·V 조합의 화소
        for y in 0..=255u8 { for u in (0..=255u8).step_by(3) { for v in (0..=255u8).step_by(5) {
            let src = [y, u, y, v];
            let (mut o, mut g) = ([0u8; 6], [0u8; 2]);
            yuyv_to_rgb8_gray(&src, 4, 2, &mut o, &mut g, Yuy2Rule::NATIVE);
            assert_eq!(g[0], gray_of(&[o[0], o[1], o[2]]));
        }}}
    }
}
