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
}
