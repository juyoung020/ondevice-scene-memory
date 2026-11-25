//! f32 행렬곱 핵심: C[m, n] = A[m, k] · B[k, n].
//! A 는 MR 줄씩 묶어 [k][MR] 로(줄 묶음), B 는 NB 열씩 [k][NB] 판(panel)으로 펴 둔다. 스레드는 판을 나눠 맡고,
//! 판 하나를 캐시에 둔 채 모든 줄 묶음을 지나가게 한다. 합은 0 에서 시작해 k = 0, 1, … 차례로 FMA(커널마다 같은 비트).
#![allow(unsafe_op_in_unsafe_fn)]
use rayon::prelude::*;

pub const NB: usize = 32;

#[inline(always)]
unsafe fn tile_scalar<const M: usize>(a: *const f32, k: usize, mr: usize, panel: *const f32, out: &mut [[f32; NB]], add: bool) {
    let mut acc = [[0f32; NB]; M];
    if add {
        acc.copy_from_slice(&out[..M]);
    }
    for j in 0..k {
        let b = std::slice::from_raw_parts(panel.add(j * NB), NB);
        for m in 0..M {
            let x = *a.add(j * mr + m);
            for c in 0..NB {
                acc[m][c] = x.mul_add(b[c], acc[m][c]);
            }
        }
    }
    out[..M].copy_from_slice(&acc);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tile_avx512<const M: usize>(a: *const f32, k: usize, mr: usize, panel: *const f32, out: &mut [[f32; NB]], add: bool) {
    use std::arch::x86_64::*;
    let mut a0 = [_mm512_setzero_ps(); M];
    let mut a1 = [_mm512_setzero_ps(); M];
    if add {
        for m in 0..M {
            a0[m] = _mm512_loadu_ps(out[m].as_ptr());
            a1[m] = _mm512_loadu_ps(out[m].as_ptr().add(16));
        }
    }
    for j in 0..k {
        let w0 = _mm512_loadu_ps(panel.add(j * NB));
        let w1 = _mm512_loadu_ps(panel.add(j * NB + 16));
        for m in 0..M {
            let x = _mm512_set1_ps(*a.add(j * mr + m));
            a0[m] = _mm512_fmadd_ps(x, w0, a0[m]);
            a1[m] = _mm512_fmadd_ps(x, w1, a1[m]);
        }
    }
    for m in 0..M {
        _mm512_storeu_ps(out[m].as_mut_ptr(), a0[m]);
        _mm512_storeu_ps(out[m].as_mut_ptr().add(16), a1[m]);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tile_avx2<const M: usize>(a: *const f32, k: usize, mr: usize, panel: *const f32, out: &mut [[f32; NB]], add: bool) {
    use std::arch::x86_64::*;
    let mut acc = [[_mm256_setzero_ps(); 4]; M];
    if add {
        for m in 0..M {
            for q in 0..4 {
                acc[m][q] = _mm256_loadu_ps(out[m].as_ptr().add(q * 8));
            }
        }
    }
    for j in 0..k {
        let w: [__m256; 4] = std::array::from_fn(|q| _mm256_loadu_ps(panel.add(j * NB + q * 8)));
        for m in 0..M {
            let x = _mm256_broadcast_ss(&*a.add(j * mr + m));
            for q in 0..4 {
                acc[m][q] = _mm256_fmadd_ps(x, w[q], acc[m][q]);
            }
        }
    }
    for m in 0..M {
        for q in 0..4 {
            _mm256_storeu_ps(out[m].as_mut_ptr().add(q * 8), acc[m][q]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn tile_neon<const M: usize>(a: *const f32, k: usize, mr: usize, panel: *const f32, out: &mut [[f32; NB]], add: bool) {
    use std::arch::aarch64::*;
    let mut acc = [[vdupq_n_f32(0.0); 8]; M];
    if add {
        for m in 0..M {
            for q in 0..8 {
                acc[m][q] = vld1q_f32(out[m].as_ptr().add(q * 4));
            }
        }
    }
    for j in 0..k {
        let x: [f32; M] = std::array::from_fn(|m| *a.add(j * mr + m));
        for h in 0..2 {
            let w: [float32x4_t; 4] = std::array::from_fn(|q| vld1q_f32(panel.add(j * NB + h * 16 + q * 4)));
            for m in 0..M {
                for q in 0..4 {
                    acc[m][h * 4 + q] = vfmaq_n_f32(acc[m][h * 4 + q], w[q], x[m]);
                }
            }
        }
    }
    for m in 0..M {
        for q in 0..8 {
            vst1q_f32(out[m].as_mut_ptr().add(q * 4), acc[m][q]);
        }
    }
}

/// 줄 묶음 여럿(rows 줄)을 판 하나에 통과. `a` 는 [`pack_a`] 모양(첫 줄부터)
macro_rules! rows_through {
    ($name:ident, $tile:ident, $mr:literal, $($attr:tt)*) => {
        $($attr)*
        unsafe fn $name(a: &[f32], k: usize, rows: usize, panel: &[f32], out: &mut [[f32; NB]]) {
            let (a, p) = (a.as_ptr(), panel.as_ptr());
            // k 를 KC 씩: 판 조각(KC · NB)이 L1 에 머문 채 모든 줄 묶음이 지나감
            let mut k0 = 0;
            while k0 < k {
                let kl = KC.min(k - k0);
                let add = k0 > 0;
                let pk = p.add(k0 * NB);
                let mut r = 0;
                while r + $mr <= rows {
                    $tile::<$mr>(a.add(r * k + k0 * $mr), kl, $mr, pk, &mut out[r..], add);
                    r += $mr;
                }
                let m = rows - r;
                let (x, o) = (a.add(r * k + k0 * m), &mut out[r..]);
                match m {
                    0 => {}
                    1 => $tile::<1>(x, kl, m, pk, o, add),
                    2 => $tile::<2>(x, kl, m, pk, o, add),
                    3 => $tile::<3>(x, kl, m, pk, o, add),
                    4 => $tile::<4>(x, kl, m, pk, o, add),
                    5 => $tile::<5>(x, kl, m, pk, o, add),
                    6 => $tile::<6>(x, kl, m, pk, o, add),
                    7 => $tile::<7>(x, kl, m, pk, o, add),
                    8 => $tile::<8>(x, kl, m, pk, o, add),
                    9 => $tile::<9>(x, kl, m, pk, o, add),
                    10 => $tile::<10>(x, kl, m, pk, o, add),
                    _ => $tile::<11>(x, kl, m, pk, o, add),
                }
                k0 += kl;
            }
        }
    };
}

const KC: usize = 256;

rows_through!(rows_scalar, tile_scalar, 3, #[inline(never)]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx512, tile_avx512, 12, #[target_feature(enable = "avx512f")]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx2, tile_avx2, 3, #[target_feature(enable = "avx2,fma")]);
#[cfg(target_arch = "aarch64")]
rows_through!(rows_neon, tile_neon, 3, #[target_feature(enable = "neon")]);

type RowsFn = (unsafe fn(&[f32], usize, usize, &[f32], &mut [[f32; NB]]), usize);

fn kernel() -> RowsFn {
    static K: std::sync::OnceLock<RowsFn> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return (rows_avx512, 12);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return (rows_avx2, 3);
            }
        }
        #[cfg(target_arch = "aarch64")]
        return (rows_neon, 3);
        #[allow(unreachable_code)]
        (rows_scalar, 3)
    })
}

/// 줄 묶음 크기(이 CPU 커널)
pub fn mr() -> usize {
    kernel().1
}

/// A[m, k](줄 사이 lda) → MR 줄 묶음마다 [k][묶음 줄 수]
pub fn pack_a(a: &[f32], m: usize, k: usize, lda: usize) -> Vec<f32> {
    let mr = mr();
    let mut p = super::buf::zeros::<f32>(m * k);
    let mut r0 = 0;
    while r0 < m {
        let rr = mr.min(m - r0);
        let t = &mut p[r0 * k..(r0 + rr) * k];
        for i in 0..rr {
            let row = &a[(r0 + i) * lda..(r0 + i) * lda + k];
            for (j, &v) in row.iter().enumerate() {
                t[j * rr + i] = v;
            }
        }
        r0 += rr;
    }
    p
}

/// B[k, n](줄 사이 ldb) 의 판 p(열 p·NB ..) → [k][NB], 넘치는 열은 0
pub fn pack_b_panel(b: &[f32], k: usize, n: usize, ldb: usize, p: usize, out: &mut [f32]) {
    let c0 = p * NB;
    let w = NB.min(n - c0);
    for j in 0..k {
        let o = &mut out[j * NB..(j + 1) * NB];
        o[..w].copy_from_slice(&b[j * ldb + c0..j * ldb + c0 + w]);
        o[w..].fill(0.0);
    }
}

/// B 전체를 판으로
pub fn pack_b(b: &[f32], k: usize, n: usize, ldb: usize) -> Vec<f32> {
    let np = n.div_ceil(NB);
    let mut p = vec![0f32; np * k * NB];
    p.par_chunks_mut(k * NB).enumerate().for_each(|(i, o)| pack_b_panel(b, k, n, ldb, i, o));
    p
}

/// B[k, n] 를 transB(B 가 [n, k] 로 저장) 에서 판으로
pub fn pack_bt(bt: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut b = vec![0f32; k * n];
    for i in 0..n {
        for j in 0..k {
            b[j * n + i] = bt[i * k + j];
        }
    }
    pack_b(&b, k, n, n)
}

/// 판을 어디서 얻나
pub enum Panels<'a> {
    /// 미리 편 판들([`pack_b`])
    Packed(&'a [f32]),
    /// 판 번호 → 버퍼([k][NB])에 채우기
    Make(&'a (dyn Fn(usize, &mut [f32]) + Sync)),
}

/// 판 결과 한 덩어리(줄 r0.., 열 c0..c0+w) 를 받아 쓰는 곳
pub trait Sink: Sync {
    /// rows: 줄마다 NB 값, w 개만 유효
    fn put(&self, r0: usize, c0: usize, w: usize, rows: &mut [[f32; NB]]);
}

/// C = A · B, A 는 [`pack_a`], 결과는 sink 로. 일은 (판, 줄 덩어리) 로 나눔
pub fn gemm(ap: &[f32], m: usize, k: usize, n: usize, b: Panels, sink: &dyn Sink) {
    let (kern, mr) = kernel();
    let np = n.div_ceil(NB);
    let th = super::pool::threads();
    // 판이 모자라면 줄도 나눔(MR 배수)
    let splits = if np >= 2 * th { 1 } else { (2 * th).div_ceil(np).min(m.div_ceil(mr)).max(1) };
    let chunk = m.div_ceil(splits).div_ceil(mr) * mr;
    let nr = m.div_ceil(chunk);
    // 판 하나 일이 작으면(작은 m · k) 판 여럿을 한 일로: 일 수 ≈ 스레드 × 4
    let per = if nr == 1 && np > 4 * th && m * k < 64 * 1024 { np / (4 * th) } else { 1 }.max(1);
    let ng = np.div_ceil(per);
    super::pool::for_each(ng * nr, &|t| {
        let (pg, ri) = (t / nr, t % nr);
        let r0 = ri * chunk;
        let rows = chunk.min(m - r0);
        thread_local! {
            static BUF: std::cell::RefCell<(Vec<f32>, Vec<[f32; NB]>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        BUF.with(|bf| {
            let mut bf = bf.borrow_mut();
            let (pb, ob) = &mut *bf;
            for p in pg * per..((pg + 1) * per).min(np) {
                let panel: &[f32] = match &b {
                    Panels::Packed(all) => &all[p * k * NB..(p + 1) * k * NB],
                    Panels::Make(f) => {
                        pb.resize(k * NB, 0.0);
                        f(p, pb);
                        pb
                    }
                };
                ob.resize(rows, [0.0; NB]);
                unsafe { kern(&ap[r0 * k..], k, rows, panel, ob) };
                sink.put(r0, p * NB, NB.min(n - p * NB), ob);
            }
        });
    });
}

/// 줄마다 쓰는 출력(줄 사이 ldc) + 열 편향 · 줄 편향 · 활성
pub struct RowMajor<'a> {
    pub out: *mut f32,
    pub ldc: usize,
    pub col_bias: Option<&'a [f32]>,
    pub row_bias: Option<&'a [f32]>,
    pub act: super::Act,
}
unsafe impl Sync for RowMajor<'_> {}

impl Sink for RowMajor<'_> {
    fn put(&self, r0: usize, c0: usize, w: usize, rows: &mut [[f32; NB]]) {
        for (i, r) in rows.iter_mut().enumerate() {
            let row = r0 + i;
            let v = &mut r[..w];
            if let Some(cb) = self.col_bias {
                for (x, b) in v.iter_mut().zip(&cb[c0..c0 + w]) {
                    *x += b;
                }
            }
            if let Some(rb) = self.row_bias {
                let b = rb[row];
                v.iter_mut().for_each(|x| *x += b);
            }
            self.act.apply(v);
            // SAFETY: 일마다 (줄, 열) 칸이 겹치지 않음
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), self.out.add(row * self.ldc + c0), w) };
        }
    }
}
