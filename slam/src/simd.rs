//! 손으로 짠 aarch64 NEON 판(보드 Cortex-A76)과 그 스칼라 판. 두 판이 비트까지 같다.
//! 다른 기계(x86 PC)는 스칼라 판.

/// ICM-42688 레지스터 12 바이트(가속도 x y z, 자이로 x y z, 큰 끝 i16; 버퍼는 16 바이트) → (가속도 m/s², 자이로 rad/s)
#[inline(always)]
pub fn imu_raw(b: &[u8; 16], acc_s: f32, gyr_s: f32) -> ([f32; 4], [f32; 4]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 16 바이트 읽기·쓰기
    unsafe {
        use core::arch::aarch64::*;
        let s = vreinterpretq_s16_u8(vrev16q_u8(vld1q_u8(b.as_ptr()))); // 바이트 바꿈 → i16 8 개
        let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))); // ax ay az gx
        let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))); // gy gz _ _
        let sl: [f32; 4] = [acc_s, acc_s, acc_s, gyr_s];
        let (mut l, mut h) = ([0f32; 4], [0f32; 4]);
        vst1q_f32(l.as_mut_ptr(), vmulq_f32(lo, vld1q_f32(sl.as_ptr())));
        vst1q_f32(h.as_mut_ptr(), vmulq_n_f32(hi, gyr_s));
        return ([l[0], l[1], l[2], 0.0], [l[3], h[0], h[1], 0.0]);
    }
    #[allow(unreachable_code)]
    imu_raw_scalar(b, acc_s, gyr_s)
}

#[inline(always)]
pub fn imu_raw_scalar(b: &[u8; 16], acc_s: f32, gyr_s: f32) -> ([f32; 4], [f32; 4]) {
    let v = |i: usize| i16::from_be_bytes([b[2 * i], b[2 * i + 1]]) as f32;
    ([v(0) * acc_s, v(1) * acc_s, v(2) * acc_s, 0.0], [v(3) * gyr_s, v(4) * gyr_s, v(5) * gyr_s, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 레지스터 변환: NEON 과 스칼라가 비트까지 같다
    #[test]
    fn imu_raw_same_as_scalar() {
        let mut s = 3u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for _ in 0..10_000 {
            let mut raw = [0u8; 16];
            for x in raw.iter_mut().take(12) {
                *x = r();
            }
            let (x, y) = (imu_raw(&raw, 0.0047884, 0.0010653), imu_raw_scalar(&raw, 0.0047884, 0.0010653));
            assert_eq!((x.0.map(f32::to_bits), x.1.map(f32::to_bits)), (y.0.map(f32::to_bits), y.1.map(f32::to_bits)));
        }
        let mut raw = [0u8; 16];
        raw[4] = 0x08; // az = 2048 = 1 g(±16 g)
        raw[7] = 8; // gx = 8
        let (acc, gyr) = imu_raw(&raw, 16.0 * 9.80665 / 32768.0, 1.0);
        assert!((acc[2] - 9.80665).abs() < 1e-4 && gyr[0] == 8.0);
    }
}
