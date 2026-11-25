//! f32 행렬곱 핵심: C[m, n] = A[m, k] · B[k, n].
//! A 는 MR 줄씩 묶어 [k][MR] 로(줄 묶음). B 는 NB 열씩 [k][NB] 판(panel)으로 펴 두거나(펴기),
//! 바로 읽기(B 의 j 째 줄 = base + offs[j] 에서 NB 개 연속 — 보폭 1 합성곱은 덧댄 입력을 그대로 읽음).
//! 스레드는 판을 나눠 맡고, 판(k 는 KC 씩)을 캐시에 둔 채 모든 줄 묶음을 지나가게 한다.
//! 합은 0 에서 시작해 k = 0, 1, … 차례로 FMA(커널마다 같은 비트).
#![allow(unsafe_op_in_unsafe_fn)]
use rayon::prelude::*;

pub const NB: usize = 32;

/// B 의 j 째 줄 시작: 펴 둔 판이면 bp + j·NB, 바로 읽기면 bp + offs[j]
#[inline(always)]
unsafe fn brow<const D: bool>(bp: *const f32, offs: *const usize, j: usize) -> *const f32 {
    if D { bp.add(*offs.add(j)) } else { bp.add(j * NB) }
}

#[inline(always)]
unsafe fn tile_scalar<const M: usize, const D: bool>(a: *const f32, k: usize, mr: usize, bp: *const f32, offs: *const usize, out: &mut [[f32; NB]], add: bool) {
    let mut acc = [[0f32; NB]; M];
    if add {
        acc.copy_from_slice(&out[..M]);
    }
    for j in 0..k {
        let b = std::slice::from_raw_parts(brow::<D>(bp, offs, j), NB);
        for m in 0..M {
            let x = *a.add(j * mr + m);
            for c in 0..NB {
                acc[m][c] += x * b[c];
            }
        }
    }
    out[..M].copy_from_slice(&acc);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tile_avx512<const M: usize, const D: bool>(a: *const f32, k: usize, mr: usize, bp: *const f32, offs: *const usize, out: &mut [[f32; NB]], add: bool) {
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
        let b = brow::<D>(bp, offs, j);
        if D {
            _mm_prefetch::<_MM_HINT_T0>(b.add(NB) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(b.add(NB + 16) as *const i8);
        }
        let w0 = _mm512_loadu_ps(b);
        let w1 = _mm512_loadu_ps(b.add(16));
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
unsafe fn tile_avx2<const M: usize, const D: bool>(a: *const f32, k: usize, mr: usize, bp: *const f32, offs: *const usize, out: &mut [[f32; NB]], add: bool) {
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
        let b = brow::<D>(bp, offs, j);
        let w: [__m256; 4] = std::array::from_fn(|q| _mm256_loadu_ps(b.add(q * 8)));
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
unsafe fn tile_neon<const M: usize, const D: bool>(a: *const f32, k: usize, mr: usize, bp: *const f32, offs: *const usize, out: &mut [[f32; NB]], add: bool) {
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
        let b = brow::<D>(bp, offs, j);
        for h in 0..2 {
            let w: [float32x4_t; 4] = std::array::from_fn(|q| vld1q_f32(b.add(h * 16 + q * 4)));
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

/// 줄 묶음 여럿(rows 줄)을 판 하나에 통과. `a` 는 [`pack_a`] 모양(첫 줄부터, 묶음 mr 줄)
macro_rules! rows_through {
    ($name:ident, $tile:ident, $($attr:tt)*) => {
        $($attr)*
        unsafe fn $name<const D: bool>(a: &[f32], k: usize, rows: usize, mr: usize, bp: *const f32, offs: *const usize, out: &mut [[f32; NB]]) {
            let a = a.as_ptr();
            // k 를 KC 씩: 판 조각(KC · NB)이 L1 에 머문 채 모든 줄 묶음이 지나감
            let mut k0 = 0;
            while k0 < k {
                let kl = KC.min(k - k0);
                let add = k0 > 0;
                let (pk, ok) = if D { (bp, offs.add(k0)) } else { (bp.add(k0 * NB), offs) };
                let mut r = 0;
                while r < rows {
                    let m = mr.min(rows - r);
                    let (x, o) = (a.add(r * k + k0 * m), &mut out[r..]);
                    match m {
                        1 => $tile::<1, D>(x, kl, m, pk, ok, o, add),
                        2 => $tile::<2, D>(x, kl, m, pk, ok, o, add),
                        3 => $tile::<3, D>(x, kl, m, pk, ok, o, add),
                        4 => $tile::<4, D>(x, kl, m, pk, ok, o, add),
                        5 => $tile::<5, D>(x, kl, m, pk, ok, o, add),
                        6 => $tile::<6, D>(x, kl, m, pk, ok, o, add),
                        7 => $tile::<7, D>(x, kl, m, pk, ok, o, add),
                        8 => $tile::<8, D>(x, kl, m, pk, ok, o, add),
                        9 => $tile::<9, D>(x, kl, m, pk, ok, o, add),
                        10 => $tile::<10, D>(x, kl, m, pk, ok, o, add),
                        11 => $tile::<11, D>(x, kl, m, pk, ok, o, add),
                        _ => $tile::<12, D>(x, kl, m, pk, ok, o, add),
                    }
                    r += m;
                }
                k0 += kl;
            }
        }
    };
}

const KC: usize = 256;

rows_through!(rows_scalar, tile_scalar, #[inline(never)]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx512, tile_avx512, #[target_feature(enable = "avx512f")]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx2, tile_avx2, #[target_feature(enable = "avx2,fma")]);
#[cfg(target_arch = "aarch64")]
rows_through!(rows_neon, tile_neon, #[target_feature(enable = "neon")]);

type Rows = unsafe fn(&[f32], usize, usize, usize, *const f32, *const usize, &mut [[f32; NB]]);
/// (펴 둔 판, 바로 읽기, MR)
type Kern = (Rows, Rows, usize);

fn kernel() -> Kern {
    static K: std::sync::OnceLock<Kern> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return (rows_avx512::<false>, rows_avx512::<true>, 12);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return (rows_avx2::<false>, rows_avx2::<true>, 3);
            }
        }
        #[cfg(target_arch = "aarch64")]
        return (rows_neon::<false>, rows_neon::<true>, 3);
        #[allow(unreachable_code)]
        (rows_scalar::<false>, rows_scalar::<true>, 3)
    })
}

/// m 줄에 쓸 줄 묶음 크기: 커널 최대(AVX-512 12, 그 밖 3)의 절반 이상에서 버리는 줄(마지막 묶음의 빈 줄)이 가장 적은 것
/// (16 · 32 · 64 채널 → 8, 193 토큰 → 11). 큰 묶음과 작은 묶음의 FMA 비율은 거의 같고, 남는 줄 묶음이 느리다
pub fn mr_for(m: usize) -> usize {
    let mx = kernel().2;
    let mut best = mx;
    for t in (mx.div_ceil(2)..=mx).rev() {
        if m.div_ceil(t) * t < m.div_ceil(best) * best {
            best = t;
        }
    }
    best
}

/// A[m, k](줄 사이 lda) → MR 줄 묶음마다 [k][묶음 줄 수]
pub fn pack_a(a: &[f32], m: usize, k: usize, lda: usize) -> Vec<f32> {
    let mr = mr_for(m);
    let mut p = super::buf::take::<f32>(m * k);
    // 묶음 차례로(끝 묶음만 줄 수가 적음), 크면 묶음들을 나눠 맡김
    let tiles = m.div_ceil(mr);
    let body = |t: usize, dst: &mut [f32]| {
        let r0 = t * mr;
        let rr = mr.min(m - r0);
        for i in 0..rr {
            let row = &a[(r0 + i) * lda..(r0 + i) * lda + k];
            for (j, &v) in row.iter().enumerate() {
                dst[j * rr + i] = v;
            }
        }
    };
    if m * k > 1 << 16 {
        super::pool::chunks_mut(&mut p, mr * k, body);
    } else {
        for (t, d) in p.chunks_mut(mr * k).enumerate() {
            body(t, d);
        }
    }
    let _ = tiles;
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
    /// 바로 읽기: B[j][c] = base[offs[j] + c] (base 는 마지막 판이 NB 개 읽어도 되게 꼬리를 둠)
    Direct { base: &'a [f32], offs: &'a [usize] },
}

/// 판 결과 한 덩어리(줄 r0.., 열 c0..c0+w) 를 받아 쓰는 곳
pub trait Sink: Sync {
    /// rows: 줄마다 NB 값, w 개만 유효
    fn put(&self, r0: usize, c0: usize, w: usize, rows: &mut [[f32; NB]]);
}

/// C = A · B, A 는 [`pack_a`], 결과는 sink 로. 일은 (판, 줄 덩어리) 로 나눔
pub fn gemm(ap: &[f32], m: usize, k: usize, n: usize, b: Panels, sink: &dyn Sink) {
    let (kern, kdir, _) = kernel();
    let mr = mr_for(m);
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
            let (p0, p1) = (pg * per, ((pg + 1) * per).min(np));
            // 펴 둔 판 · 바로 읽기에서 판이 여럿이면 줄 띠(2 묶음)마다 판들을 지나감: 출력 줄 흐름이 동시에 띠 줄 수만큼만
            // (64 줄을 한꺼번에 쓰면 출력이 캐시를 넘을 때 줄마다 떨어진 쓰기 흐름 64 개로 느려짐)
            let band = if p1 - p0 > 1 && !matches!(b, Panels::Make(_)) { 2 * mr } else { rows };
            let mut rb = 0;
            while rb < rows {
                let rr = band.min(rows - rb);
                let a0 = &ap[(r0 + rb) * k..];
                for p in p0..p1 {
                    ob.resize(rr, [0.0; NB]);
                    match &b {
                        Panels::Packed(all) => unsafe { kern(a0, k, rr, mr, all[p * k * NB..(p + 1) * k * NB].as_ptr(), std::ptr::null(), ob) },
                        Panels::Make(f) => {
                            pb.resize(k * NB, 0.0);
                            f(p, pb);
                            unsafe { kern(a0, k, rr, mr, pb.as_ptr(), std::ptr::null(), ob) }
                        }
                        Panels::Direct { base, offs } => unsafe { kdir(a0, k, rr, mr, base.as_ptr().add(p * NB), offs.as_ptr(), ob) },
                    }
                    sink.put(r0 + rb, p * NB, NB.min(n - p * NB), ob);
                }
                rb += rr;
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
        super::simd::run(|| self.put_in(r0, c0, w, rows))
    }
}

impl RowMajor<'_> {
    #[inline(always)]
    fn put_in(&self, r0: usize, c0: usize, w: usize, rows: &mut [[f32; NB]]) {
        // 편향 → 활성(덩어리 통째, 남는 열도 계산만 함) → 줄마다 복사
        for (i, r) in rows.iter_mut().enumerate() {
            if let Some(cb) = self.col_bias {
                for (x, b) in r[..w].iter_mut().zip(&cb[c0..c0 + w]) {
                    *x += b;
                }
            }
            if let Some(rb) = self.row_bias {
                let b = rb[r0 + i];
                r.iter_mut().for_each(|x| *x += b);
            }
        }
        self.act.apply_in(rows.as_flattened_mut());
        for (i, r) in rows.iter().enumerate() {
            // SAFETY: 일마다 (줄, 열) 칸이 겹치지 않음
            unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), self.out.add((r0 + i) * self.ldc + c0), w) };
        }
    }
}

/// 판 하나를 바로 읽기로: B[j][c] = bp[offs[j] + c](c < NB), A 는 [`pack_a`] 의 m 줄 전부 → out[m]
/// (부르는 쪽이 NB 개 읽어도 되는 꼬리를 보장)
pub unsafe fn direct_panel(ap: &[f32], m: usize, k: usize, bp: *const f32, offs: &[usize], out: &mut [[f32; NB]]) {
    let (_, kdir, _) = kernel();
    kdir(ap, k, m, mr_for(m), bp, offs.as_ptr(), out)
}
