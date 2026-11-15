//! 손으로 짠 aarch64 NEON 판(보드 Cortex-A76)과 그 스칼라 판. 정수 결과라 두 판이 비트까지 같다(시험은 qemu-aarch64 로 돌림).
//! 다른 기계(x86 PC)는 스칼라 판(컴파일러가 popcnt 등으로 바꿈).

/// 256 비트(32 바이트 BRIEF/ORB 설명자) 두 개의 서로 다른 비트 수
#[inline(always)]
pub fn hamming256(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 는 NEON 이 늘 있음, 32 바이트 읽기
    unsafe {
        use core::arch::aarch64::*;
        let p = a.as_ptr() as *const u8;
        let q = b.as_ptr() as *const u8;
        let x0 = vcntq_u8(veorq_u8(vld1q_u8(p), vld1q_u8(q)));
        let x1 = vcntq_u8(veorq_u8(vld1q_u8(p.add(16)), vld1q_u8(q.add(16))));
        return vaddlvq_u8(vaddq_u8(x0, x1)) as u32; // 바이트당 ≤ 16, 합은 u16 으로
    }
    #[allow(unreachable_code)]
    hamming256_scalar(a, b)
}

#[inline(always)]
pub fn hamming256_scalar(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones() + (a[2] ^ b[2]).count_ones() + (a[3] ^ b[3]).count_ones()
}


/// 가로 9 칸 합: o[k] = Σ col[k..k+9] (col.len() = o.len() + 8). i32 라 NEON 판과 값이 같다
#[inline]
pub fn hsum9(col: &[i32], o: &mut [i32]) {
    let n = o.len();
    assert!(col.len() >= n + 8);
    #[allow(unused_mut)]
    let mut k = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: k + 4 ≤ n 이면 col[k .. k+12] 와 o[k .. k+4] 는 범위 안
    unsafe {
        use core::arch::aarch64::*;
        let (c, op) = (col.as_ptr(), o.as_mut_ptr());
        while k + 4 <= n {
            let p = c.add(k);
            let a = vaddq_s32(vaddq_s32(vld1q_s32(p), vld1q_s32(p.add(1))), vaddq_s32(vld1q_s32(p.add(2)), vld1q_s32(p.add(3))));
            let b = vaddq_s32(vaddq_s32(vld1q_s32(p.add(4)), vld1q_s32(p.add(5))), vaddq_s32(vld1q_s32(p.add(6)), vld1q_s32(p.add(7))));
            vst1q_s32(op.add(k), vaddq_s32(vaddq_s32(a, b), vld1q_s32(p.add(8))));
            k += 4;
        }
    }
    hsum9_scalar(&col[k..], &mut o[k..]);
}

#[inline]
pub fn hsum9_scalar(col: &[i32], o: &mut [i32]) {
    let n = o.len();
    if n == 0 {
        return;
    }
    let mut s: i32 = col[..9].iter().sum();
    o[0] = s;
    for k in 1..n {
        s += col[k + 8] - col[k - 1];
        o[k] = s;
    }
}

// ---- IMU(imu.rs): 자이로 사전적분 · 회전 보정 · 좌표 변환 · 레지스터 변환 ----
// 3×3 은 행마다 4 칸(넷째 칸 0)으로 둔다(NEON 한 레지스터 = 한 행). 곱·덧셈 순서를 스칼라 판과 똑같이 둬(FMA 없음) 두 판이 비트까지 같다.

/// 3×3 행렬, 행 우선, 행마다 넷째 칸은 0
pub type M34 = [[f32; 4]; 3];
pub const M34_I: M34 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];

/// a·b
#[inline(always)]
pub fn m34_mul(a: &M34, b: &M34) -> M34 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 행마다 16 바이트 읽기·쓰기
    unsafe {
        use core::arch::aarch64::*;
        let (b0, b1, b2) = (vld1q_f32(b[0].as_ptr()), vld1q_f32(b[1].as_ptr()), vld1q_f32(b[2].as_ptr()));
        let mut o = [[0f32; 4]; 3];
        for i in 0..3 {
            let r = vaddq_f32(vaddq_f32(vmulq_n_f32(b0, a[i][0]), vmulq_n_f32(b1, a[i][1])), vmulq_n_f32(b2, a[i][2]));
            vst1q_f32(o[i].as_mut_ptr(), r);
        }
        return o;
    }
    #[allow(unreachable_code)]
    m34_mul_scalar(a, b)
}

#[inline(always)]
pub fn m34_mul_scalar(a: &M34, b: &M34) -> M34 {
    let mut o = [[0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            o[i][j] = (b[0][j] * a[i][0] + b[1][j] * a[i][1]) + b[2][j] * a[i][2];
        }
    }
    o
}

/// 전치
#[inline(always)]
pub fn m34_t(a: &M34) -> M34 {
    [[a[0][0], a[1][0], a[2][0], 0.0], [a[0][1], a[1][1], a[2][1], 0.0], [a[0][2], a[1][2], a[2][2], 0.0]]
}

/// m·v(좌표 변환, v 넷째 칸은 0 으로 봄)
#[inline(always)]
pub fn m34_vec(m: &M34, v: &[f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 16 바이트 읽기
    unsafe {
        use core::arch::aarch64::*;
        let x = vsetq_lane_f32::<3>(0.0, vld1q_f32(v.as_ptr()));
        // 가로 합(vaddvq, 짝끼리 먼저) = (a0 + a1) + (a2 + a3): 스칼라 판과 같은 순서
        return [vaddvq_f32(vmulq_f32(vld1q_f32(m[0].as_ptr()), x)), vaddvq_f32(vmulq_f32(vld1q_f32(m[1].as_ptr()), x)), vaddvq_f32(vmulq_f32(vld1q_f32(m[2].as_ptr()), x)), 0.0];
    }
    #[allow(unreachable_code)]
    m34_vec_scalar(m, v)
}

#[inline(always)]
pub fn m34_vec_scalar(m: &M34, v: &[f32; 4]) -> [f32; 4] {
    let d = |r: &[f32; 4]| (r[0] * v[0] + r[1] * v[1]) + (r[2] * v[2] + r[3] * 0.0);
    [d(&m[0]), d(&m[1]), d(&m[2]), 0.0]
}

/// 회전 벡터의 회전 행렬(Rodrigues). 두 판이 같이 쓰는 스칼라 식
#[inline(always)]
pub fn so3_exp(v: [f32; 4]) -> M34 {
    let th2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    // 작은 각은 급수(sin θ/θ, (1−cos θ)/θ²)
    let (a, b) = if th2 < 1e-8 {
        (1.0 - th2 / 6.0, 0.5 - th2 / 24.0)
    } else {
        let th = th2.sqrt();
        (th.sin() / th, (1.0 - th.cos()) / th2)
    };
    let (x, y, z) = (v[0], v[1], v[2]);
    [
        [1.0 - b * (y * y + z * z), b * x * y - a * z, b * x * z + a * y, 0.0],
        [b * x * y + a * z, 1.0 - b * (x * x + z * z), b * y * z - a * x, 0.0],
        [b * x * z - a * y, b * y * z + a * x, 1.0 - b * (x * x + y * y), 0.0],
    ]
}

/// 자이로 한 걸음: r · Exp((w − bias)·dt)
#[inline(always)]
pub fn gyro_step(r: &M34, w: &[f32; 4], bias: &[f32; 4], dt: f32) -> M34 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 16 바이트 읽기·쓰기
    unsafe {
        use core::arch::aarch64::*;
        let mut v = [0f32; 4];
        vst1q_f32(v.as_mut_ptr(), vmulq_n_f32(vsubq_f32(vld1q_f32(w.as_ptr()), vld1q_f32(bias.as_ptr())), dt));
        return m34_mul(r, &so3_exp(v));
    }
    #[allow(unreachable_code)]
    gyro_step_scalar(r, w, bias, dt)
}

#[inline(always)]
pub fn gyro_step_scalar(r: &M34, w: &[f32; 4], bias: &[f32; 4], dt: f32) -> M34 {
    let v = [(w[0] - bias[0]) * dt, (w[1] - bias[1]) * dt, (w[2] - bias[2]) * dt, (w[3] - bias[3]) * dt];
    m34_mul_scalar(r, &so3_exp(v))
}

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
    #[test]
    fn hamming_same_as_scalar() {
        let mut s = 1u64;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); s };
        for _ in 0..200_000 {
            let a = [r(), r(), r(), r()];
            let b = [r(), r(), r(), r()];
            assert_eq!(hamming256(&a, &b), hamming256_scalar(&a, &b));
        }
        assert_eq!(hamming256(&[0; 4], &[u64::MAX; 4]), 256);
    }
    #[test]
    fn hsum9_same_as_scalar() {
        let mut s = 7u64;
        for n in 0..70 {
            let col: Vec<i32> = (0..n + 8).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 40) as i32 % 2296 }).collect();
            let (mut a, mut b) = (vec![0; n], vec![0; n]);
            hsum9(&col, &mut a);
            hsum9_scalar(&col, &mut b);
            assert_eq!(a, b);
        }
    }

    fn bits(m: &M34) -> Vec<u32> {
        m.iter().flatten().map(|x| x.to_bits()).collect()
    }

    /// IMU 판: NEON 과 스칼라가 비트까지 같다(자이로 400 걸음 이어 곱하기, 좌표 변환, 레지스터 변환)
    #[test]
    fn imu_same_as_scalar() {
        let mut s = 3u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let bias = [0.01, -0.02, 0.005, 0.0];
        let (mut a, mut b) = (M34_I, M34_I);
        for _ in 0..400 {
            let w = [r() * 6.0, r() * 6.0, r() * 6.0, 0.0];
            let dt = 0.005 + r() * 0.001;
            a = gyro_step(&a, &w, &bias, dt);
            b = gyro_step_scalar(&b, &w, &bias, dt);
            assert_eq!(bits(&a), bits(&b));
            let v = [r(), r(), r(), 0.0];
            assert_eq!(m34_vec(&a, &v).map(f32::to_bits), m34_vec_scalar(&b, &v).map(f32::to_bits));
            let c = m34_t(&a);
            assert_eq!(bits(&m34_mul(&c, &a)), bits(&m34_mul_scalar(&c, &b)));
        }
        // 회전으로 남아 있음(직교)
        let p = m34_mul(&m34_t(&a), &a);
        for i in 0..3 {
            for j in 0..3 {
                assert!((p[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-4);
            }
        }
        for _ in 0..10_000 {
            let mut raw = [0u8; 16];
            for x in raw.iter_mut().take(12) {
                *x = (r() * 127.0 + 128.0) as u8;
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
