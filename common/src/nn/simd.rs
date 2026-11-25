//! 원소별 반복을 이 CPU 의 넓은 벡터로: f 를 AVX-512 · AVX2 기능을 켠 함수 안에서 불러, 안에 펼쳐진 반복이 그 명령으로 벡터화되게 한다.
//! 저장소는 x86 에 대상 CPU 를 따로 주지 않아(기본 SSE2) 원소별 반복이 4 칸씩만 돌았다. 곱셈 · 덧셈 차례는 그대로라 값은 같다.
#![allow(unsafe_op_in_unsafe_fn)]

#[inline(always)]
pub fn run<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
        static LV: AtomicU8 = AtomicU8::new(0);
        let mut lv = LV.load(Relaxed);
        if lv == 0 {
            lv = if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512dq") {
                3
            } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                2
            } else {
                1
            };
            LV.store(lv, Relaxed);
        }
        match lv {
            // SAFETY: 기능을 확인함
            3 => return unsafe { w512(f) },
            2 => return unsafe { w2(f) },
            _ => {}
        }
    }
    f()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,avx2,fma,bmi1,bmi2")]
unsafe fn w512<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2")]
unsafe fn w2<R>(f: impl FnOnce() -> R) -> R {
    f()
}
