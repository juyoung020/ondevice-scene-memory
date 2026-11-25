//! 원소별 함수를 슬라이스째: 반복을 AVX-512 · AVX2 기능을 켠 함수 안에 두어 그 폭으로 벡터화(기본 x86 빌드는 SSE2 4 칸).
//! 값은 기능과 상관없이 같다(곱셈 · 덧셈 차례 그대로, FMA 로 합치지 않음).
#![allow(unsafe_op_in_unsafe_fn)]
use super::{erf, exp, sigmoid, tanh};

#[cfg(target_arch = "x86_64")]
fn level() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static LV: AtomicU8 = AtomicU8::new(0);
    let mut lv = LV.load(Relaxed);
    if lv == 0 {
        lv = if std::is_x86_feature_detected!("avx512f") {
            3
        } else if std::is_x86_feature_detected!("avx2") {
            2
        } else {
            1
        };
        LV.store(lv, Relaxed);
    }
    lv
}

macro_rules! slice_fn {
    ($(#[$m:meta])* $name:ident, |$x:ident| $body:expr) => {
        $(#[$m])*
        pub fn $name(v: &mut [f32]) {
            #[inline(always)]
            fn body(v: &mut [f32]) {
                for $x in v.iter_mut() {
                    *$x = $body;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx512f,avx512vl,avx512dq,avx512bw,avx2")]
                unsafe fn w512(v: &mut [f32]) {
                    body(v)
                }
                #[target_feature(enable = "avx2")]
                unsafe fn w2(v: &mut [f32]) {
                    body(v)
                }
                match level() {
                    // SAFETY: 기능 확인함
                    3 => return unsafe { w512(v) },
                    2 => return unsafe { w2(v) },
                    _ => {}
                }
            }
            body(v)
        }
    };
}

slice_fn!(relu, |x| x.max(0.0));
slice_fn!(
    /// x · sigmoid(x)
    silu, |x| *x * sigmoid(*x));
slice_fn!(sigmoid_s, |x| sigmoid(*x));
slice_fn!(tanh_s, |x| tanh(*x));
slice_fn!(erf_s, |x| erf(*x));
slice_fn!(exp_s, |x| exp(*x));
slice_fn!(sqrt_s, |x| x.sqrt());
slice_fn!(neg_s, |x| -*x);

/// 줄 하나 softmax(제자리)
/// 줄(길이 d)마다 softmax(제자리), 이 CPU 의 넓은 벡터로
pub fn softmax_rows(v: &mut [f32], d: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx512f,avx512vl,avx512dq,avx512bw,avx2")]
        unsafe fn w512(v: &mut [f32], d: usize) {
            v.chunks_mut(d).for_each(softmax_row)
        }
        #[target_feature(enable = "avx2")]
        unsafe fn w2(v: &mut [f32], d: usize) {
            v.chunks_mut(d).for_each(softmax_row)
        }
        match level() {
            // SAFETY: 기능 확인함
            3 => return unsafe { w512(v, d) },
            2 => return unsafe { w2(v, d) },
            _ => {}
        }
    }
    v.chunks_mut(d).for_each(softmax_row)
}

/// 줄 하나 softmax(제자리). 최대 · 합은 16 칸 갈래로 나눠 더함(벡터로 돌게)
#[inline(always)]
fn softmax_row(row: &mut [f32]) {
    let mut ml = [f32::NEG_INFINITY; 16];
    let mut ch = row.chunks_exact(16);
    for c in &mut ch {
        for (a, &b) in ml.iter_mut().zip(c) {
            *a = a.max(b);
        }
    }
    let m = ch.remainder().iter().fold(ml.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)), |a, &b| a.max(b));
    row.iter_mut().for_each(|x| *x -= m);
    exp_s(row);
    let mut sl = [0f32; 16];
    let mut ch = row.chunks_exact(16);
    for c in &mut ch {
        for (a, &b) in sl.iter_mut().zip(c) {
            *a += b;
        }
    }
    let s: f32 = sl.iter().sum::<f32>() + ch.remainder().iter().sum::<f32>();
    let k = 1.0 / s;
    row.iter_mut().for_each(|x| *x *= k);
}
