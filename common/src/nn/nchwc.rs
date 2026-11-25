//! NCHWc 배치: 채널을 B 개씩 묶어 [N][C/B][H][W][B] 로 둔다(AVX-512 B = 16, 그 밖 B = 8). 한 화소의 B 채널이 연속이라
//! 합성곱이 입력 값 하나를 퍼뜨려 출력 채널 B 개(묶음 여럿)에 곱해 더한다(MLAS ConvNchwcFloatKernel 방식) — 채널 평면을 따로
//! 읽고 쓰는 흐름이 없다. 합성곱 · 채널별 합성곱 · ConvTranspose · 원소별 · Concat · Split · Resize · 풀링이 이 배치를 그대로
//! 읽고 쓰며, 이 배치를 모르는 노드 앞에서만 NCHW 로 바꾼다([`accepts`]). 첫 층은 NCHW 입력(묶음 1)을 그대로 읽는다.
//! 곱셈-덧셈 차례: 입력 묶음 → (ky, kx) → 묶음 안 채널, 합은 0 에서 시작해 FMA, 끝에 편향 · 활성(MLAS 와 같은 차례).
#![allow(unsafe_op_in_unsafe_fn)]
use super::ops::{ConvGeo, geo, need};
use super::{Act, Data, Node, Pre, Tensor, buf, pool, simd};
use std::sync::OnceLock;

type R = Result<Vec<Tensor>, String>;

/// 채널 묶음 크기 B
pub fn bk() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") {
            return 16;
        }
        8
    })
}

/// (묶음 수 최대, 묶음 수별 화소 수 최대)
fn caps() -> (usize, [usize; 5]) {
    if bk() == 16 { (4, [0, 12, 8, 8, 6]) } else { (2, [0, 12, 6, 0, 0]) }
}

struct P<T>(*mut T);
unsafe impl<T> Sync for P<T> {}
unsafe impl<T> Send for P<T> {}
impl<T> P<T> {
    #[inline(always)]
    fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

fn dims4(t: &Tensor) -> Result<(usize, usize, usize, usize), String> {
    if t.shape.len() != 4 {
        return Err(format!("NCHWc 는 4 차원만 {:?}", t.shape));
    }
    Ok((t.shape[0], t.shape[1], t.shape[2], t.shape[3]))
}

/// NCHW → NCHWc(남는 채널 칸은 0)
pub fn to_blocked(t: &Tensor) -> Result<Tensor, String> {
    let b = bk();
    let (nb, c, h, w) = dims4(t)?;
    let (hw, ncb) = (h * w, c.div_ceil(b));
    let v = t.fv()?;
    let mut o = buf::take::<f32>(nb * ncb * hw * b);
    const PC: usize = 1024;
    let npc = hw.div_ceil(PC);
    let op = P(o.as_mut_ptr());
    pool::for_each(nb * ncb * npc, &|i| {
        simd::run(|| {
            let (pl, pc) = (i / npc, i % npc);
            let (bb, cb) = (pl / ncb, pl % ncb);
            let (p0, p1) = (pc * PC, ((pc + 1) * PC).min(hw));
            // SAFETY: 일마다 (평면, 화소 범위) 가 겹치지 않음
            let d = unsafe { std::slice::from_raw_parts_mut(op.at((pl * hw + p0) * b), (p1 - p0) * b) };
            let lanes = b.min(c - cb * b);
            for l in 0..b {
                if l >= lanes {
                    for p in 0..p1 - p0 {
                        d[p * b + l] = 0.0;
                    }
                } else {
                    let s = &v[(bb * c + cb * b + l) * hw + p0..(bb * c + cb * b + l) * hw + p1];
                    for (p, &x) in s.iter().enumerate() {
                        d[p * b + l] = x;
                    }
                }
            }
        })
    });
    Ok(Tensor { shape: t.shape.clone(), data: Data::F(o), blk: b })
}

/// NCHWc → NCHW
pub fn to_plain(t: &Tensor) -> Result<Tensor, String> {
    let b = t.blk;
    let (nb, c, h, w) = dims4(t)?;
    let (hw, ncb) = (h * w, c.div_ceil(b));
    let v = t.fv()?;
    let mut o = buf::take::<f32>(nb * c * hw);
    const PC: usize = 1024;
    let npc = hw.div_ceil(PC);
    let op = P(o.as_mut_ptr());
    pool::for_each(nb * ncb * npc, &|i| {
        simd::run(|| {
            let (pl, pc) = (i / npc, i % npc);
            let (bb, cb) = (pl / ncb, pl % ncb);
            let (p0, p1) = (pc * PC, ((pc + 1) * PC).min(hw));
            let s = &v[(pl * hw + p0) * b..(pl * hw + p1) * b];
            for l in 0..b.min(c - cb * b) {
                // SAFETY: 일마다 (채널, 화소 범위) 가 겹치지 않음
                let d = unsafe { std::slice::from_raw_parts_mut(op.at((bb * c + cb * b + l) * hw + p0), p1 - p0) };
                for (p, x) in d.iter_mut().enumerate() {
                    *x = s[p * b + l];
                }
            }
        })
    });
    Ok(Tensor::f(t.shape.clone(), o))
}

/// 이 노드가 입력을 NCHWc 그대로 읽을 수 있나(아니면 실행 전에 NCHW 로 바꿈)
pub(crate) fn accepts(n: &Node, ins: &[Option<&Tensor>]) -> bool {
    let b = bk();
    let blk = |t: &&Tensor| t.blk > 0;
    match n.op.as_str() {
        "Conv" | "ConvTranspose" | "Identity" | "Relu" | "Sigmoid" | "Tanh" | "Sqrt" | "Erf" | "Neg" | "Exp" | "MaxPool" | "AveragePool" | "Resize" => true,
        "Add" | "Sub" | "Mul" | "Div" => {
            let (Some(x), Some(y)) = (ins.first().copied().flatten(), ins.get(1).copied().flatten()) else {
                return false;
            };
            let f = matches!(x.data, Data::F(_)) && matches!(y.data, Data::F(_));
            f && ((x.blk > 0 && y.blk > 0 && x.shape == y.shape) || (x.blk > 0 && y.len() == 1) || (y.blk > 0 && x.len() == 1))
        }
        "Concat" => {
            let ts: Vec<&Tensor> = ins.iter().flatten().copied().collect();
            n.ai("axis", 0) == 1 && ts.iter().all(blk) && ts.iter().all(|t| t.shape.len() == 4) && ts[..ts.len() - 1].iter().all(|t| t.shape[1] % b == 0)
        }
        "Split" => {
            let Some(x) = ins.first().copied().flatten() else { return false };
            if n.ai("axis", 0) != 1 || x.shape.len() != 4 {
                return false;
            }
            match ins.get(1).copied().flatten().map(|s| s.ints()).or(n.ais("split")) {
                Some(s) => s[..s.len().saturating_sub(1)].iter().all(|&v| v as usize % b == 0),
                None => false,
            }
        }
        _ => false,
    }
}

// ---------------------------------------------------------------- 무게

/// 상수 합성곱 무게: 원래 [Co][C][kh][kw](ConvTranspose 는 가상 [kh·kw·Co'][C][1][1]), 입력 묶음 크기별로 처음 쓸 때 편다
pub(crate) struct ConvPre {
    pub w: Vec<f32>,
    pub co: usize,
    pub c: usize,
    pub kh: usize,
    pub kw: usize,
    pub dw: bool,
    packs: [OnceLock<Vec<f32>>; 2],
}

impl ConvPre {
    pub fn new(w: Vec<f32>, co: usize, c: usize, kh: usize, kw: usize, dw: bool) -> ConvPre {
        ConvPre { w, co, c, kh, kw, dw, packs: [OnceLock::new(), OnceLock::new()] }
    }
    fn pack(&self, bi: usize) -> &[f32] {
        self.packs[(bi != 1) as usize].get_or_init(|| if self.dw { pack_dw(&self.w, self.co, self.kh, self.kw) } else { pack_w(&self.w, self.co, self.c, self.kh, self.kw, bi) })
    }
}

/// [Co][C][kh][kw] → 출력 묶음 무리(커널이 한 번에 맡는 fb 개)마다 [C/bi][kh][kw][bi][fb][B](넘치는 칸 0)
fn pack_w(w: &[f32], co: usize, c: usize, kh: usize, kw: usize, bi: usize) -> Vec<f32> {
    let b = bk();
    let fbm = caps().0;
    let (ncb, ngi, taps) = (co.div_ceil(b), c.div_ceil(bi), kh * kw);
    let mut p = vec![0f32; ncb * ngi * taps * bi * b];
    for o in 0..co {
        let (cb, l) = (o / b, o % b);
        let (v, f) = (cb / fbm * fbm, cb % fbm);
        let fb = fbm.min(ncb - v);
        for ci in 0..c {
            for t in 0..taps {
                p[v * ngi * taps * bi * b + (((ci / bi) * taps + t) * bi + ci % bi) * fb * b + f * b + l] = w[(o * c + ci) * taps + t];
            }
        }
    }
    p
}

/// 채널별 [C][1][kh][kw] → [C/B][kh][kw][B]
fn pack_dw(w: &[f32], c: usize, kh: usize, kw: usize) -> Vec<f32> {
    let b = bk();
    let taps = kh * kw;
    let mut p = vec![0f32; c.div_ceil(b) * taps * b];
    for ci in 0..c {
        for t in 0..taps {
            p[((ci / b) * taps + t) * b + ci % b] = w[ci * taps + t];
        }
    }
    p
}

/// ConvTranspose 무게 [C][Co][kh][kw] → 가상 1×1 합성곱 [(dy·kw + dx)·Co' + co][C](Co' = Co 를 B 배수로)
pub(crate) fn conv_t_virtual(w: &[f32], ci: usize, co: usize, kh: usize, kw: usize) -> (Vec<f32>, usize) {
    let cop = co.div_ceil(bk()) * bk();
    let taps = kh * kw;
    let mut v = vec![0f32; taps * cop * ci];
    for c in 0..ci {
        for o in 0..co {
            for t in 0..taps {
                v[(t * cop + o) * ci + c] = w[(c * co + o) * taps + t];
            }
        }
    }
    (v, taps * cop)
}

// ---------------------------------------------------------------- 커널

/// 타일 계산 인자. 입력 (묶음 gi, 화소) 시작 = g[gi] + off0, 탭 (ky, kx) 는 (ky − ky0)·kys + (kx − kx0)·kxs, 화소 q 는 q·ps,
/// 묶음 안 채널 ii(< bi) 는 + ii. 무게(출력 묶음 fb 개 무리) = w + ((gi·taps + ky·kw + kx)·bi + ii)·fb·B + f·B
#[derive(Clone, Copy)]
struct Kp {
    g: *const usize,
    off0: usize,
    gi0: usize,
    gi1: usize,
    bi: usize,
    /// 0 = 묶음마다 채널 bi 개 모두, 아니면 앞 채널 nl 개만(남는 칸이 있는 마지막 묶음)
    nl: usize,
    kys: usize,
    kxs: usize,
    ps: usize,
    w: *const f32,
    taps: usize,
    kw: usize,
    /// 끝 처리: 0 = 날값, 1 = 편향만, 2 = 편향 · Relu, 3 = 편향 · SiLU. 묶음 f 화소 q 자리 = out + f·fs + q·os, 편향 = eb + f·B
    /// (eb 가 null 이면 편향 없음)
    ea: u8,
    eb: *const f32,
    fs: usize,
    os: usize,
    /// 1×1: 입력을 이만큼(칸) 앞서 미리 읽기(0 = 안 함, x86 만)
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pf: usize,
    /// NCHW 출력(채널이 B/4 이하): 채널 l 의 화소 q 자리 = out + l·ss + q, 쓰는 채널 = sm 비트(0 = NCHWc)
    sm: u32,
    ss: usize,
    /// 합 시작값(null = 0): ip + f·ifs + q·ios
    ip: *const f32,
    ifs: usize,
    ios: usize,
}

/// x · sigmoid(x) 를 [`super::exp`] · [`super::sigmoid`] 와 같은 차례(곱셈 · 덧셈 따로)로
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn silu512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let s = _mm512_set1_ps;
    let z = _mm512_min_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_setzero_ps(), x), s(-87.0)), s(88.0));
    let t = _mm512_add_ps(_mm512_mul_ps(z, s(std::f32::consts::LOG2_E)), s(12_582_912.0));
    let n = _mm512_sub_ps(t, s(12_582_912.0));
    let r = _mm512_add_ps(_mm512_sub_ps(z, _mm512_mul_ps(n, s(0.693_359_4))), _mm512_mul_ps(n, s(2.121_944_4e-4)));
    let mut p = s(1.987_569_1e-4);
    for c in [1.398_199_9e-3, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 5.000_000_1e-1] {
        p = _mm512_add_ps(_mm512_mul_ps(p, r), s(c));
    }
    let y = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(p, r), r), r), s(1.0));
    let e = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_sub_epi32(_mm512_castps_si512(t), _mm512_set1_epi32(0x4B40_0000)), _mm512_set1_epi32(127))));
    let ex = _mm512_mul_ps(y, e);
    _mm512_mul_ps(x, _mm512_div_ps(s(1.0), _mm512_add_ps(s(1.0), ex)))
}

/// 커널 하나하나를 따로 둔 함수로(큰 분기 함수에 모두 펼치면 누적이 레지스터에 머물지 못하는 판이 생김)
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn t512<const FB: usize, const PX: usize, const BI: usize, const NI: usize>(p: &Kp, ky0: usize, ky1: usize, kx0: usize, kx1: usize, out: *mut f32) {
    use std::arch::x86_64::*;
    let mut acc = [[_mm512_setzero_ps(); PX]; FB];
    if !p.ip.is_null() {
        for f in 0..FB {
            for q in 0..PX {
                acc[f][q] = _mm512_loadu_ps(p.ip.add(f * p.ifs + q * p.ios));
            }
        }
    }
    let wk = BI * FB * 16;
    for gi in p.gi0..p.gi1 {
        let gb = (*p.g.add(gi) as *const f32).add(p.off0);
        if p.pf > 0 {
            for q in 0..PX {
                _mm_prefetch::<_MM_HINT_T0>(gb.add(p.pf + q * p.ps) as *const i8);
            }
        }
        let wgp = p.w.add(gi * p.taps * wk);
        for ky in ky0..ky1 {
            let xr = gb.add((ky - ky0) * p.kys);
            let wr = wgp.add(ky * p.kw * wk);
            for kx in kx0..kx1 {
                let xp = xr.add((kx - kx0) * p.kxs);
                let wt = wr.add(kx * wk);
                for ii in 0..NI {
                    let wv: [__m512; FB] = std::array::from_fn(|f| _mm512_loadu_ps(wt.add((ii * FB + f) * 16)));
                    for q in 0..PX {
                        let x = _mm512_set1_ps(*xp.add(q * p.ps + ii));
                        for f in 0..FB {
                            acc[f][q] = _mm512_fmadd_ps(x, wv[f], acc[f][q]);
                        }
                    }
                }
            }
        }
    }
    ep512::<FB, PX>(p, &acc, out)
}

/// 끝 처리(따로 둔 함수: 상수들이 계산 고리 밖으로 올라가 누적 레지스터를 밀어내지 않게)
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn ep512<const FB: usize, const PX: usize>(p: &Kp, acc: &[[std::arch::x86_64::__m512; PX]; FB], out: *mut f32) {
    use std::arch::x86_64::*;
    let plain = p.sm != 0 && p.ea != 0;
    let mut st = [[0f32; 16]; PX];
    for f in 0..FB {
        let bv = if p.ea != 0 && !p.eb.is_null() { Some(_mm512_loadu_ps(p.eb.add(f * 16))) } else { None };
        for q in 0..PX {
            let mut v = acc[f][q];
            if let Some(b) = bv {
                v = _mm512_add_ps(v, b);
            }
            match p.ea {
                2 => v = _mm512_max_ps(v, _mm512_setzero_ps()),
                3 => v = silu512(v),
                _ => {}
            }
            if plain {
                _mm512_storeu_ps(st[q].as_mut_ptr(), v);
            } else {
                _mm512_storeu_ps(out.add(f * p.fs + q * p.os), v);
            }
        }
    }
    if plain {
        // NCHW: 채널마다 화소 PX 개를 모아(16 칸 간격 읽기) 이어 씀
        let idx = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(16));
        let m = ((1u32 << PX) - 1) as __mmask16;
        for l in 0..16 {
            if p.sm >> l & 1 != 0 {
                let v = _mm512_mask_i32gather_ps::<4>(_mm512_setzero_ps(), m, idx, st.as_ptr().cast::<f32>().add(l) as _);
                _mm512_mask_storeu_ps(out.add(l * p.ss), m, v);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn tneon<const FB: usize, const PX: usize, const BI: usize, const NI: usize>(p: &Kp, ky0: usize, ky1: usize, kx0: usize, kx1: usize, out: *mut f32) {
    use std::arch::aarch64::*;
    let mut acc = [[[vdupq_n_f32(0.0); 2]; PX]; FB];
    if !p.ip.is_null() {
        for f in 0..FB {
            for q in 0..PX {
                for h in 0..2 {
                    acc[f][q][h] = vld1q_f32(p.ip.add(f * p.ifs + q * p.ios + h * 4));
                }
            }
        }
    }
    let wk = BI * FB * 8;
    for gi in p.gi0..p.gi1 {
        let gb = (*p.g.add(gi) as *const f32).add(p.off0);
        let wgp = p.w.add(gi * p.taps * wk);
        for ky in ky0..ky1 {
            let xr = gb.add((ky - ky0) * p.kys);
            let wr = wgp.add(ky * p.kw * wk);
            for kx in kx0..kx1 {
                let xp = xr.add((kx - kx0) * p.kxs);
                let wt = wr.add(kx * wk);
                for ii in 0..NI {
                    let wv: [[float32x4_t; 2]; FB] = std::array::from_fn(|f| [vld1q_f32(wt.add((ii * FB + f) * 8)), vld1q_f32(wt.add((ii * FB + f) * 8 + 4))]);
                    for q in 0..PX {
                        let x = *xp.add(q * p.ps + ii);
                        for f in 0..FB {
                            acc[f][q][0] = vfmaq_n_f32(acc[f][q][0], wv[f][0], x);
                            acc[f][q][1] = vfmaq_n_f32(acc[f][q][1], wv[f][1], x);
                        }
                    }
                }
            }
        }
    }
    for f in 0..FB {
        let bv = if p.ea != 0 && !p.eb.is_null() { Some([vld1q_f32(p.eb.add(f * 8)), vld1q_f32(p.eb.add(f * 8 + 4))]) } else { None };
        for q in 0..PX {
            for h in 0..2 {
                let mut v = acc[f][q][h];
                if let Some(b) = bv {
                    v = vaddq_f32(v, b[h]);
                }
                if p.ea == 2 {
                    v = vmaxq_f32(v, vdupq_n_f32(0.0));
                }
                if p.sm != 0 && p.ea != 0 {
                    let mut a = [0f32; 4];
                    vst1q_f32(a.as_mut_ptr(), v);
                    for (l, &x) in a.iter().enumerate() {
                        if p.sm >> (h * 4 + l) & 1 != 0 {
                            *out.add((h * 4 + l) * p.ss + q) = x;
                        }
                    }
                } else {
                    vst1q_f32(out.add(f * p.fs + q * p.os + h * 4), v);
                }
            }
        }
    }
}

/// 넓은 벡터 없는 곳(B = 8, 날값만)
#[allow(dead_code)]
#[inline(never)]
unsafe fn tgen<const FB: usize, const PX: usize, const BI: usize, const NI: usize>(p: &Kp, ky0: usize, ky1: usize, kx0: usize, kx1: usize, out: *mut f32) {
    let mut acc = [[[0f32; 8]; PX]; FB];
    if !p.ip.is_null() {
        for f in 0..FB {
            for q in 0..PX {
                std::ptr::copy_nonoverlapping(p.ip.add(f * p.ifs + q * p.ios), acc[f][q].as_mut_ptr(), 8);
            }
        }
    }
    let wk = BI * FB * 8;
    for gi in p.gi0..p.gi1 {
        let gb = (*p.g.add(gi) as *const f32).add(p.off0);
        let wgp = p.w.add(gi * p.taps * wk);
        for ky in ky0..ky1 {
            let xr = gb.add((ky - ky0) * p.kys);
            let wr = wgp.add(ky * p.kw * wk);
            for kx in kx0..kx1 {
                let xp = xr.add((kx - kx0) * p.kxs);
                let wt = wr.add(kx * wk);
                for ii in 0..NI {
                    for q in 0..PX {
                        let x = *xp.add(q * p.ps + ii);
                        for f in 0..FB {
                            let w = std::slice::from_raw_parts(wt.add((ii * FB + f) * 8), 8);
                            for l in 0..8 {
                                acc[f][q][l] += x * w[l];
                            }
                        }
                    }
                }
            }
        }
    }
    for f in 0..FB {
        for q in 0..PX {
            std::ptr::copy_nonoverlapping(acc[f][q].as_ptr(), out.add(f * p.fs + q * p.os), 8);
        }
    }
}

macro_rules! px_arms {
    ($t:ident, $f:literal, $bi:literal, $ni:literal, $px:expr, $args:tt, $($q:literal),*) => {
        match $px {
            $($q => $t::<$f, $q, $bi, $ni> $args,)*
            _ => unreachable!(),
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! fb_arms {
    ($t:ident, $bi:literal, $ni:literal, $fb:expr, $px:expr, $args:tt) => {
        match $fb {
            1 => px_arms!($t, 1, $bi, $ni, $px, $args, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            2 => px_arms!($t, 2, $bi, $ni, $px, $args, 1, 2, 3, 4, 5, 6, 7, 8),
            3 => px_arms!($t, 3, $bi, $ni, $px, $args, 1, 2, 3, 4, 5, 6, 7, 8),
            _ => px_arms!($t, 4, $bi, $ni, $px, $args, 1, 2, 3, 4, 5, 6),
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn d512(p: &Kp, fb: usize, px: usize, a: usize, b: usize, c: usize, d: usize, out: *mut f32) {
    if p.bi == 1 {
        return fb_arms!(t512, 1, 1, fb, px, (p, a, b, c, d, out));
    }
    // 묶음 gi1 − 1 이 마지막(남는 칸 있음)이면 그 묶음은 따로: 앞 묶음들의 합을 시작값으로 실제 채널 수(4 · 8 로 올림)까지
    match p.nl {
        0 => fb_arms!(t512, 16, 16, fb, px, (p, a, b, c, d, out)),
        4 => fb_arms!(t512, 16, 4, fb, px, (p, a, b, c, d, out)),
        _ => fb_arms!(t512, 16, 8, fb, px, (p, a, b, c, d, out)),
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dneon(p: &Kp, fb: usize, px: usize, a: usize, b: usize, c: usize, d: usize, out: *mut f32) {
    match (p.bi, p.nl, fb) {
        (1, _, 1) => px_arms!(tneon, 1, 1, 1, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (1, _, _) => px_arms!(tneon, 2, 1, 1, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
        (_, 0, 1) => px_arms!(tneon, 1, 8, 8, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (_, 0, _) => px_arms!(tneon, 2, 8, 8, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
        (_, _, 1) => px_arms!(tneon, 1, 8, 4, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        _ => px_arms!(tneon, 2, 8, 4, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
    }
}

#[allow(dead_code)]
#[inline(never)]
unsafe fn dgen(p: &Kp, fb: usize, px: usize, a: usize, b: usize, c: usize, d: usize, out: *mut f32) {
    match (p.bi, p.nl, fb) {
        (1, _, 1) => px_arms!(tgen, 1, 1, 1, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (1, _, _) => px_arms!(tgen, 2, 1, 1, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
        (_, 0, 1) => px_arms!(tgen, 1, 8, 8, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (_, 0, _) => px_arms!(tgen, 2, 8, 8, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
        (_, _, 1) => px_arms!(tgen, 1, 8, 4, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        _ => px_arms!(tgen, 2, 8, 4, px, (p, a, b, c, d, out), 1, 2, 3, 4, 5, 6),
    }
}

/// 출력 묶음 fb 개 × 화소 px 개(끝 처리 · 쓰는 곳은 p 에). 입력의 마지막 묶음에 남는 칸이 있으면(bl < bi) 앞 묶음들을
/// 날값으로 tb 에 둔 뒤 마지막 묶음을 그 합에 이어 실제 채널 수(올림)까지만
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn tile_all(p: &Kp, gn: usize, bl: usize, tb: *mut f32, fb: usize, px: usize, ky0: usize, ky1: usize, kx0: usize, kx1: usize, out: *mut f32) {
    let nl = if p.gi1 == gn && bl < p.bi { if bl <= 4 { 4 } else { 8 } } else { 0 };
    if nl == 0 || nl >= p.bi {
        return tile(p, fb, px, ky0, ky1, kx0, kx1, out);
    }
    let mut p = *p;
    let p = &mut p;
    let (ea, fs, os, gi1) = (p.ea, p.fs, p.os, p.gi1);
    if p.gi0 < gi1 - 1 {
        p.gi1 = gi1 - 1;
        (p.ea, p.fs, p.os) = (0, px * bk(), bk());
        tile(p, fb, px, ky0, ky1, kx0, kx1, tb);
        (p.ip, p.ifs, p.ios) = (tb, px * bk(), bk());
    }
    (p.gi0, p.gi1, p.nl, p.ea, p.fs, p.os) = (gi1 - 1, gi1, nl, ea, fs, os);
    tile(p, fb, px, ky0, ky1, kx0, kx1, out);
}

/// 출력 묶음 fb 개 × 화소 px 개(끝 처리 · 쓰는 곳은 p 에)
#[inline(always)]
unsafe fn tile(p: &Kp, fb: usize, px: usize, ky0: usize, ky1: usize, kx0: usize, kx1: usize, out: *mut f32) {
    #[cfg(target_arch = "x86_64")]
    if bk() == 16 {
        return d512(p, fb, px, ky0, ky1, kx0, kx1, out);
    }
    #[cfg(target_arch = "aarch64")]
    return dneon(p, fb, px, ky0, ky1, kx0, kx1, out);
    #[allow(unreachable_code)]
    dgen(p, fb, px, ky0, ky1, kx0, kx1, out)
}

// ---------------------------------------------------------------- 합성곱

/// 탭 범위: 출력 자리 o 에서 0 ≤ o·s + k·d − p < len 인 k ∈ [k0, k1), 그리고 k0 의 입력 자리
fn krange(o: usize, s: usize, d: usize, p: usize, len: usize, k: usize) -> (usize, usize, usize) {
    let base = (o * s) as isize - p as isize;
    let k0 = if base >= 0 { 0 } else { ((-base) as usize).div_ceil(d) };
    let last = len as isize - 1 - base;
    let k1 = if last < 0 { 0 } else { (last as usize / d + 1).min(k) };
    if k0 >= k1 { (0, 0, 0) } else { (k0, k1, (base + (k0 * d) as isize) as usize) }
}

/// 가운데(모든 kx 가 입력 안) 출력 열 범위
fn inner(g: &ConvGeo) -> (usize, usize) {
    let num = g.w as isize - 1 + g.pl as isize - ((g.kw - 1) * g.dw) as isize;
    (g.pl.div_ceil(g.sw), if num < 0 { 0 } else { (num as usize / g.sw + 1).min(g.ow) })
}

/// 합성곱 입력: 묶음 평면 주소(배치마다 ngi 개), 묶음 크기 bi(NCHWc 면 B, NCHW 면 1)
struct Src {
    nb: usize,
    c: usize,
    h: usize,
    w: usize,
    bi: usize,
    ngi: usize,
    g: Vec<usize>,
    _tmp: Vec<Tensor>,
}

/// 입력 조각들(n.cat: Concat · Split 묶은 것 포함): 모두 NCHWc 이고 채널 경계가 B 에 맞으면 묶음 평면을 그대로,
/// 아니면 NCHW 채널 평면들(NCHWc 조각은 바꿔서)
fn src(n: &Node, ins: &[Option<&Tensor>]) -> Result<Src, String> {
    let b = bk();
    let parts: Vec<(&Tensor, usize, usize)> = if n.cat.is_empty() {
        let x = need(ins, 0)?;
        vec![(x, 0, *x.shape.get(1).ok_or("합성곱 입력 모양")?)]
    } else {
        n.cat
            .iter()
            .map(|&(i, off, cnt)| {
                let t = need(ins, i)?;
                Ok((t, off, if cnt == 0 { t.shape[1] - off } else { cnt }))
            })
            .collect::<Result<_, String>>()?
    };
    let (nb, _, h, w) = dims4(parts[0].0)?;
    for p in &parts {
        let (pn, _, ph, pw) = dims4(p.0)?;
        if (pn, ph, pw) != (nb, h, w) {
            return Err(format!("이어 붙일 모양 {:?} ≠ {:?}", p.0.shape, parts[0].0.shape));
        }
    }
    let c: usize = parts.iter().map(|p| p.2).sum();
    let hw = h * w;
    let last = parts.len() - 1;
    let aligned = parts.iter().enumerate().all(|(i, p)| p.0.blk == b && p.1 % b == 0 && (i == last || p.2 % b == 0));
    let mut g = vec![];
    let mut tmp = vec![];
    if aligned {
        for bb in 0..nb {
            for &(t, off, cnt) in &parts {
                let ncb = t.shape[1].div_ceil(b);
                let base = t.fv()?.as_ptr();
                for gi in off / b..(off + cnt).div_ceil(b) {
                    g.push(unsafe { base.add((bb * ncb + gi) * hw * b) } as usize);
                }
            }
        }
        return Ok(Src { nb, c, h, w, bi: b, ngi: c.div_ceil(b), g, _tmp: tmp });
    }
    let mut which = vec![];
    for &(t, _, _) in &parts {
        if t.blk > 0 {
            tmp.push(to_plain(t)?);
            which.push(Some(tmp.len() - 1));
        } else {
            which.push(None);
        }
    }
    for bb in 0..nb {
        for (k, &(t, off, cnt)) in parts.iter().enumerate() {
            let pt = match which[k] {
                Some(i) => &tmp[i],
                None => t,
            };
            let base = pt.fv()?.as_ptr();
            let tc = t.shape[1];
            for ci in off..off + cnt {
                g.push(unsafe { base.add((bb * tc + ci) * hw) } as usize);
            }
        }
    }
    Ok(Src { nb, c, h, w, bi: 1, ngi: c, g, _tmp: tmp })
}

/// 출력 쓰기 방식
enum Mode {
    Blk,
    /// NCHW(채널 B/4 이하, 출력 묶음 하나)
    Plain,
    /// ConvTranspose(보폭 = 크기): 가상 묶음 vb = 탭 · ncb + 묶음, 입력 화소 (y, x) → 출력 (y·kh + dy, x·kw + dx)
    T {
        kh: usize,
        kw: usize,
        w: usize,
        ow: usize,
    },
}

/// 타일 결과를 받아 편향 · 활성 후 출력에 씀
struct Out {
    p: *mut f32,
    mode: Mode,
    co: usize,
    ncb: usize,
    ohw: usize,
    /// 줄 폭(화소 자리 = oy · rw + ox)
    rw: usize,
    bias: Option<Vec<f32>>,
    act: Act,
}
unsafe impl Sync for Out {}

impl Out {
    #[allow(clippy::too_many_arguments)]
    fn new(p: *mut f32, mode: Mode, co: usize, ohw: usize, rw: usize, bias: Option<&[f32]>, act: &Act) -> Out {
        let b = bk();
        let ncb = co.div_ceil(b);
        let bias = bias.map(|bs| {
            let mut v = vec![0f32; ncb * b];
            v[..co].copy_from_slice(&bs[..co]);
            v
        });
        Out { p, mode, co, ncb, ohw, rw, bias, act: act.clone() }
    }

    /// 커널이 끝 처리까지 해서 출력에 바로 쓸 수 있으면 그 번호([`Kp::ea`]), 아니면 0
    fn direct(&self) -> u8 {
        if matches!(self.mode, Mode::T { .. }) {
            return 0;
        }
        let a = match self.act {
            Act::None => 1,
            Act::Relu => 2,
            Act::Silu => 3,
            _ => return 0,
        };
        #[cfg(target_arch = "x86_64")]
        if bk() == 16 {
            return a;
        }
        #[cfg(target_arch = "aarch64")]
        return if a == 3 { 0 } else { a };
        #[allow(unreachable_code)]
        0
    }

    /// 배치 bb, (가상) 묶음 vb, 자리 (oy, ox) 부터 화소 px 개: t = [px][B]
    #[inline(always)]
    fn put(&self, bb: usize, vb: usize, oy: usize, ox: usize, px: usize, t: &mut [f32]) {
        let b = bk();
        let cb = if let Mode::T { .. } = self.mode { vb % self.ncb } else { vb };
        if let Some(bs) = &self.bias {
            let bs = &bs[cb * b..(cb + 1) * b];
            for q in t.chunks_mut(b) {
                for (x, y) in q.iter_mut().zip(bs) {
                    *x += y;
                }
            }
        }
        self.act.apply_in(t);
        let lanes = b.min(self.co - cb * b);
        if lanes < b && matches!(self.act, Act::Prog(_)) {
            // 남는 채널 칸은 0(묶은 원소별 연산이 0 이 아닌 값을 낼 수 있음)
            for q in t.chunks_mut(b) {
                q[lanes..].fill(0.0);
            }
        }
        let pix = oy * self.rw + ox;
        unsafe {
            match self.mode {
                Mode::Blk => std::ptr::copy_nonoverlapping(t.as_ptr(), self.p.add(((bb * self.ncb + cb) * self.ohw + pix) * b), px * b),
                Mode::Plain => {
                    for l in 0..lanes {
                        let d = self.p.add((bb * self.co + l) * self.ohw + pix);
                        for q in 0..px {
                            *d.add(q) = t[q * b + l];
                        }
                    }
                }
                Mode::T { kh, kw, w, ow } => {
                    let tap = vb / self.ncb;
                    let (dy, dx) = (tap / kw, tap % kw);
                    let (mut y, mut x) = (pix / w, pix % w);
                    for q in 0..px {
                        let o = (y * kh + dy) * ow + x * kw + dx;
                        std::ptr::copy_nonoverlapping(t.as_ptr().add(q * b), self.p.add(((bb * self.ncb + cb) * self.ohw + o) * b), b);
                        x += 1;
                        if x == w {
                            x = 0;
                            y += 1;
                        }
                    }
                }
            }
        }
    }
}

/// 합성곱(group 1) 본체: 일 = (출력 줄 조각 묶음, 출력 묶음 무리 범위). 줄마다 출력 묶음 fb 개씩 화소 px 개 타일로,
/// 가장자리(덧댐에 걸리는 화소)는 화소 하나씩 탭 범위를 줄여서. 1×1(펼친 줄)은 입력 채널 128 개씩 나눠 그 무게가 L1 에
/// 머문 채 줄 조각 전체를 지나가고, 조각 합은 일꾼 버퍼에 이어 더함(MLAS 와 같은 방식)
fn gconv(s: &Src, g: &ConvGeo, wp: &[f32], nvb: usize, out: &Out) {
    let b = bk();
    let (fbm, pxm) = caps();
    let (kh, kw) = (g.kh, g.kw);
    let flat = kh == 1 && kw == 1 && g.sh == 1 && g.sw == 1 && g.pt == 0 && g.pl == 0 && g.oh == g.h && g.ow == g.w;
    let (rows, rw) = if flat { (1, g.oh * g.ow) } else { (g.oh, g.ow) };
    let xch = if flat { 240 } else { rw };
    let nseg = rw.div_ceil(xch);
    let units = s.nb * rows * nseg;
    let nfg = nvb.div_ceil(fbm);
    let gpb = if flat { (128 / s.bi).max(1) } else { s.ngi };
    let nbat = s.ngi.div_ceil(gpb);
    let fmas = s.nb * g.oh * g.ow * nvb * b * s.ngi * s.bi * kh * kw;
    let th = pool::threads();
    let want = (fmas >> 21).clamp(1, th);
    let target = if want == 1 { 1 } else { 4 * want };
    let fs = if units >= target { 1 } else { target.div_ceil(units).min(nfg).max(1) };
    let ut = units.min(target.div_ceil(fs)).max(1);
    let taps = kh * kw;
    let (rs, ps, kys, kxs) = if flat { (0, s.bi, 0, 0) } else { (g.w * s.bi, g.sw * s.bi, g.dh * g.w * s.bi, g.dw * s.bi) };
    // 가운데(모든 kx 가 입력 안) 화소 범위
    let (oxl, oxh) = if flat { (0, rw) } else { inner(g) };
    let ea = out.direct();
    let bl = s.c - (s.ngi - 1) * s.bi;
    let plain = matches!(out.mode, Mode::Plain);
    // 출력 자리(묶음 v, 화소 pix)
    let dst = |bb: usize, v: usize, pix: usize| if plain { unsafe { out.p.add(bb * out.co * out.ohw + pix) } } else { unsafe { out.p.add(((bb * out.ncb + v) * out.ohw + pix) * b) } };
    let task = |t: usize| {
        simd::run(|| {
            let (ui, fi) = (t / fs, t % fs);
            let (u0, u1) = (ui * units / ut, (ui + 1) * units / ut);
            let (f0, f1) = (fi * nfg / fs, (fi + 1) * nfg / fs);
            thread_local! {
                static SC: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            SC.with(|sc| {
                let mut sc = sc.borrow_mut();
                let mut tb = [0f32; 512];
                for u in u0..u1 {
                    let bb = u / (rows * nseg);
                    let r = (u / nseg) % rows;
                    let sg = u % nseg;
                    let (xa, xb) = (sg * xch, ((sg + 1) * xch).min(rw));
                    let (ky0, ky1, iy0) = if flat { (0, 1, 0) } else { krange(r, g.sh, g.dh, g.pt, g.h, kh) };
                    for fg in f0..f1 {
                        let v = fg * fbm;
                        let fb = fbm.min(nvb - v);
                        let eb = if ea > 0 { out.bias.as_ref().map_or(std::ptr::null(), |bs| bs[v * b..].as_ptr()) } else { std::ptr::null() };
                        let mut kp = Kp {
                            g: s.g[bb * s.ngi..].as_ptr(),
                            off0: 0,
                            gi0: 0,
                            gi1: s.ngi,
                            bi: s.bi,
                            nl: 0,
                            sm: if plain { (1u32 << out.co) - 1 } else { 0 },
                            ss: out.ohw,
                            kys,
                            kxs,
                            ps,
                            w: wp[v * s.ngi * taps * s.bi * b..].as_ptr(),
                            taps,
                            kw,
                            ea,
                            eb,
                            fs: out.ohw * b,
                            os: b,
                            pf: if flat && s.bi == b { 2 * pxm[fb] * b } else { 0 },
                            ip: std::ptr::null(),
                            ifs: 0,
                            ios: 0,
                        };
                        if nbat > 1 {
                            // 입력 채널 나눠 이어 더하기: 합은 sc[f][화소][B]
                            let n = xb - xa;
                            sc.resize(fb * xch * b, 0.0);
                            for bt in 0..nbat {
                                let last = bt + 1 == nbat;
                                kp.gi0 = bt * gpb;
                                kp.gi1 = ((bt + 1) * gpb).min(s.ngi);
                                let mut ox = xa;
                                while ox < xb {
                                    let px = pxm[fb].min(xb - ox);
                                    let sp = sc[(ox - xa) * b..].as_mut_ptr();
                                    kp.off0 = ox * s.bi;
                                    kp.ip = if bt == 0 { std::ptr::null() } else { sp };
                                    (kp.ifs, kp.ios) = (xch * b, b);
                                    // SAFETY: 탭 범위는 입력 안, 쓰는 곳은 이 일의 칸
                                    if last && ea > 0 {
                                        (kp.ea, kp.fs, kp.os) = (ea, out.ohw * b, b);
                                        unsafe { tile_all(&kp, s.ngi, bl, tb.as_mut_ptr(), fb, px, 0, 1, 0, 1, dst(bb, v, ox)) };
                                    } else {
                                        (kp.ea, kp.fs, kp.os) = (0, xch * b, b);
                                        unsafe { tile_all(&kp, s.ngi, bl, tb.as_mut_ptr(), fb, px, 0, 1, 0, 1, sp) };
                                    }
                                    ox += px;
                                }
                            }
                            if ea == 0 {
                                for f in 0..fb {
                                    out.put(bb, v + f, 0, xa, n, &mut sc[f * xch * b..f * xch * b + n * b]);
                                }
                            }
                            continue;
                        }
                        let mut ox = xa;
                        while ox < xb {
                            let (px, kx0, kx1, ix0) = if ox < oxl || ox >= oxh {
                                let (a, c, i) = krange(ox, g.sw, g.dw, g.pl, g.w, kw);
                                (1, a, c, i)
                            } else {
                                (pxm[fb].min(xb - ox).min(oxh - ox), 0, kw, ox * g.sw - g.pl)
                            };
                            let empty = ky0 >= ky1 || kx0 >= kx1;
                            kp.off0 = if empty { 0 } else { iy0 * rs + ix0 * s.bi };
                            let (a, c) = if empty { (0, 0) } else { (ky0, ky1) };
                            if ea > 0 {
                                // SAFETY: 출력 묶음 v.. 의 (r, ox..ox+px) 칸, 일마다 겹치지 않음
                                unsafe { tile_all(&kp, s.ngi, bl, tb.as_mut_ptr(), fb, px, a, c, kx0, kx1, dst(bb, v, r * out.rw + ox)) };
                            } else {
                                (kp.fs, kp.os) = (px * b, b);
                                // SAFETY: tb 는 fb · px · B ≤ 512
                                unsafe { tile_all(&kp, s.ngi, bl, tb.as_mut_ptr(), fb, px, a, c, kx0, kx1, tb.as_mut_ptr()) };
                                for f in 0..fb {
                                    out.put(bb, v + f, r, ox, px, &mut tb[f * px * b..(f + 1) * px * b]);
                                }
                            }
                            ox += px;
                        }
                    }
                }
            })
        })
    };
    if want < th {
        pool::with_limit(want, || pool::for_each(ut * fs, &task));
    } else {
        pool::for_each(ut * fs, &task);
    }
}

/// 합성곱 출력(늘 NCHWc, 남는 채널 칸 포함)
/// 합성곱 출력 버퍼 · NCHWc 인가: 채널이 B/4 이하면 NCHW(덧댄 칸이 대부분인 배치는 메모리 흐름만 늘림 — 다음 합성곱은
/// NCHW 입력을 그대로 읽음), 아니면 NCHWc(남는 채널 칸 포함)
fn out_buf(nb: usize, co: usize, ohw: usize) -> (Vec<f32>, bool) {
    let b = bk();
    let blk = co * 4 > b;
    (buf::take::<f32>(if blk { nb * co.div_ceil(b) * ohw * b } else { nb * co * ohw }), blk)
}

fn mk(shape: Vec<usize>, v: Vec<f32>, blk: bool) -> Tensor {
    Tensor { shape, data: Data::F(v), blk: if blk { bk() } else { 0 } }
}

pub(crate) fn conv(n: &Node, ins: &[Option<&Tensor>]) -> R {
    let wt = need(ins, 1)?;
    let bias = ins.get(2).copied().flatten().map(|b| b.fv()).transpose()?;
    let (co, kh, kw) = (wt.shape[0], wt.shape[2], wt.shape[3]);
    let grp = n.ai("group", 1) as usize;
    if grp != 1 {
        return dwconv(n, ins, wt, bias);
    }
    let s = src(n, ins)?;
    if wt.shape[1] != s.c {
        return Err(format!("합성곱 무게 {:?} 입력 채널 {}", wt.shape, s.c));
    }
    let g = geo(n, &[s.nb, s.c, s.h, s.w], kh, kw);
    let owned;
    let wp: &[f32] = match &n.pre {
        Pre::Conv(p) => p.pack(s.bi),
        _ => {
            owned = pack_w(wt.fv()?, co, s.c, kh, kw, s.bi);
            &owned
        }
    };
    let ohw = g.oh * g.ow;
    let (mut o, blk) = out_buf(s.nb, co, ohw);
    let flat = kh == 1 && kw == 1 && g.sh == 1 && g.sw == 1 && g.pt == 0 && g.pl == 0 && g.oh == g.h && g.ow == g.w;
    let out = Out::new(o.as_mut_ptr(), if blk { Mode::Blk } else { Mode::Plain }, co, ohw, if flat { ohw } else { g.ow }, bias, &n.act);
    gconv(&s, &g, wp, co.div_ceil(bk()), &out);
    Ok(vec![mk(vec![s.nb, co, g.oh, g.ow], o, blk)])
}

/// ConvTranspose(보폭 = 크기, 덧댐 0): 입력 화소마다 1×1 합성곱으로 kh · kw · Co 값을 내어 출력 자리로
pub(crate) fn conv_t(n: &Node, ins: &[Option<&Tensor>]) -> R {
    let x = need(ins, 0)?;
    let wt = need(ins, 1)?;
    let bias = ins.get(2).copied().flatten().map(|b| b.fv()).transpose()?;
    let (ci, co, kh, kw) = (wt.shape[0], wt.shape[1], wt.shape[2], wt.shape[3]);
    let st = n.ais("strides").unwrap_or(vec![1, 1]);
    let pd = n.ais("pads").unwrap_or(vec![0, 0, 0, 0]);
    if st[0] as usize != kh || st[1] as usize != kw || pd.iter().any(|&v| v != 0) || n.ai("group", 1) != 1 {
        return Err("ConvTranspose 는 보폭 = 크기 · 덧댐 0 만".into());
    }
    let Pre::Conv(pre) = &n.pre else {
        return Err("ConvTranspose 무게가 상수가 아님".into());
    };
    let s = src(n, &[Some(x)])?;
    if s.c != ci {
        return Err("ConvTranspose 입력 채널".into());
    }
    let (h, w) = (s.h, s.w);
    let (oh, ow) = (h * kh, w * kw);
    let g = ConvGeo { c: ci, h, w, kh: 1, kw: 1, sh: 1, sw: 1, dh: 1, dw: 1, pt: 0, pl: 0, oh: h, ow: w };
    let (mut o, blk) = out_buf(s.nb, co, oh * ow);
    if !blk {
        return Err("ConvTranspose 출력 채널이 너무 적음".into());
    }
    let out = Out::new(o.as_mut_ptr(), Mode::T { kh, kw, w, ow }, co, oh * ow, h * w, bias, &n.act);
    gconv(&s, &g, pre.pack(s.bi), pre.co / bk(), &out);
    Ok(vec![mk(vec![s.nb, co, oh, ow], o, true)])
}

/// 채널별 합성곱(group = C = Co): 묶음 평면에서 화소마다 탭 차례(ky, kx)로 B 칸 곱해 더하기, 편향 · 활성은 줄 통째
fn dwconv(n: &Node, ins: &[Option<&Tensor>], wt: &Tensor, bias: Option<&[f32]>) -> R {
    let x = need(ins, 0)?;
    let (nb, c, h, w) = dims4(x)?;
    let (co, kh, kw) = (wt.shape[0], wt.shape[2], wt.shape[3]);
    if n.ai("group", 1) as usize != c || co != c || wt.shape[1] != 1 || !n.cat.is_empty() {
        return Err(format!("group {} 합성곱 없음", n.ai("group", 1)));
    }
    let b = bk();
    let tmp;
    let xb = if x.blk == b {
        x
    } else {
        tmp = to_blocked(x)?;
        &tmp
    };
    let owned;
    let wp: &[f32] = match &n.pre {
        Pre::Conv(p) => p.pack(b),
        _ => {
            owned = pack_dw(wt.fv()?, c, kh, kw);
            &owned
        }
    };
    let g = geo(n, &[nb, c, h, w], kh, kw);
    let ncb = c.div_ceil(b);
    let ohw = g.oh * g.ow;
    let (mut o, blk) = out_buf(nb, co, ohw);
    let out = Out::new(o.as_mut_ptr(), if blk { Mode::Blk } else { Mode::Plain }, co, ohw, g.ow, bias, &n.act);
    let xv = xb.fv()?;
    let th = pool::threads();
    let rch = (4 * th).div_ceil(nb * ncb).clamp(1, g.oh);
    let rpt = g.oh.div_ceil(rch);
    let g = &g;
    pool::for_each(nb * ncb * rch, &|t| {
        simd::run(|| {
            let (pl, ri) = (t / rch, t % rch);
            let (bb, cb) = (pl / ncb, pl % ncb);
            let plane = &xv[pl * h * w * b..(pl + 1) * h * w * b];
            let wk = &wp[cb * kh * kw * b..(cb + 1) * kh * kw * b];
            thread_local! {
                static ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            ROW.with(|rb| {
                let mut rb = rb.borrow_mut();
                rb.resize(g.ow * b, 0.0);
                for oy in ri * rpt..((ri + 1) * rpt).min(g.oh) {
                    if b == 16 {
                        // SAFETY: B = 16 은 AVX-512 확인 뒤에만
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            dw16(plane, g, wk, oy, &mut rb)
                        };
                    } else {
                        dw_row::<8>(plane, g, wk, oy, &mut rb);
                    }
                    out.put(bb, cb, oy, 0, g.ow, &mut rb);
                }
            })
        })
    });
    Ok(vec![mk(vec![nb, co, g.oh, g.ow], o, blk)])
}

/// 묶음 16(AVX-512) 판을 그 기능으로 컴파일(클로저 안에서 부르면 기능 없는 판으로 남을 수 있음)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,avx2,fma")]
unsafe fn dw16(plane: &[f32], g: &ConvGeo, wk: &[f32], oy: usize, o: &mut [f32]) {
    dw_row::<16>(plane, g, wk, oy, o)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq,avx2,fma")]
unsafe fn pool16(plane: &[f32], g: &ConvGeo, oy: usize, row: &mut [f32], max: bool, incl: bool) {
    pool_row::<16>(plane, g, oy, row, max, incl)
}

#[inline(always)]
fn dw_row<const BB: usize>(plane: &[f32], g: &ConvGeo, wk: &[f32], oy: usize, o: &mut [f32]) {
    let (ky0, ky1, iy0) = krange(oy, g.sh, g.dh, g.pt, g.h, g.kh);
    let (oxl, oxh) = inner(g);
    for ox in 0..g.ow {
        let (kx0, kx1, ix0) = if ox >= oxl && ox < oxh { (0, g.kw, ox * g.sw - g.pl) } else { krange(ox, g.sw, g.dw, g.pl, g.w, g.kw) };
        let mut acc = [0f32; BB];
        if kx0 < kx1 {
            for ky in ky0..ky1 {
                let iy = iy0 + (ky - ky0) * g.dh;
                for kx in kx0..kx1 {
                    let ix = ix0 + (kx - kx0) * g.dw;
                    let xv: &[f32; BB] = plane[(iy * g.w + ix) * BB..(iy * g.w + ix + 1) * BB].try_into().unwrap();
                    let wv: &[f32; BB] = wk[(ky * g.kw + kx) * BB..(ky * g.kw + kx + 1) * BB].try_into().unwrap();
                    for l in 0..BB {
                        acc[l] += wv[l] * xv[l];
                    }
                }
            }
        }
        o[ox * BB..(ox + 1) * BB].copy_from_slice(&acc);
    }
}

// ---------------------------------------------------------------- 그 밖 노드

/// 원소별 두 입력(같은 모양 NCHWc 둘, 또는 한쪽이 값 하나)
pub(crate) fn binary(op: &str, a: &Tensor, b: &Tensor) -> R {
    let (x, y) = (a.fv()?, b.fv()?);
    let blk = a.blk.max(b.blk);
    let shape = if a.blk > 0 { a.shape.clone() } else { b.shape.clone() };
    let len = x.len().max(y.len());
    let mut o = buf::take::<f32>(len);
    let f: fn(f32, f32) -> f32 = match op {
        "Add" => |p, q| p + q,
        "Sub" => |p, q| p - q,
        "Mul" => |p, q| p * q,
        _ => |p, q| p / q,
    };
    const PC: usize = 1 << 15;
    pool::chunks_mut(&mut o, PC, |c, ch| {
        let s = c * PC;
        for (i, v) in ch.iter_mut().enumerate() {
            let (p, q) = (if x.len() == 1 { x[0] } else { x[s + i] }, if y.len() == 1 { y[0] } else { y[s + i] });
            *v = f(p, q);
        }
    });
    Ok(vec![Tensor { shape, data: Data::F(o), blk }])
}

/// 채널 축 이어 붙이기(앞 조각들은 채널 수가 B 배수): 묶음 평면을 차례로 복사
pub(crate) fn concat(ts: &[&Tensor]) -> R {
    let b = bk();
    let (nb, _, h, w) = dims4(ts[0])?;
    let hw = h * w;
    let c: usize = ts.iter().map(|t| t.shape[1]).sum();
    let ncb = c.div_ceil(b);
    let mut o = buf::take::<f32>(nb * ncb * hw * b);
    let mut cp = vec![];
    let mut d = 0;
    for bb in 0..nb {
        for (k, t) in ts.iter().enumerate() {
            let tn = t.shape[1].div_ceil(b);
            cp.push((d, k, bb * tn * hw * b, tn * hw * b));
            d += tn * hw * b;
        }
    }
    let srcs: Vec<&[f32]> = ts.iter().map(|t| t.fv()).collect::<Result<_, _>>()?;
    let op = P(o.as_mut_ptr());
    pool::for_each(cp.len(), &|i| {
        let (d, k, s, l) = cp[i];
        // SAFETY: 복사끼리 출력 자리가 겹치지 않음
        unsafe { std::ptr::copy_nonoverlapping(srcs[k].as_ptr().add(s), op.at(d), l) };
    });
    Ok(vec![Tensor { shape: vec![nb, c, h, w], data: Data::F(o), blk: b }])
}

/// 채널 축 나누기(앞 크기들은 B 배수): 묶음 평면 범위 복사. keep[i] == 0 은 빈 텐서
pub(crate) fn split(t: &Tensor, sizes: &[usize], keep: Option<&[i64]>) -> R {
    let b = bk();
    let (nb, c, h, w) = dims4(t)?;
    let hw = h * w;
    let ncb = c.div_ceil(b);
    let v = t.fv()?;
    let mut outs = vec![];
    let mut start = 0;
    for (i, &sz) in sizes.iter().enumerate() {
        if keep.is_some_and(|k| k[i] == 0) {
            outs.push(Tensor::f(vec![0], vec![]));
        } else {
            let (g0, g1) = (start / b, (start + sz).div_ceil(b));
            let l = (g1 - g0) * hw * b;
            let mut o = buf::take::<f32>(nb * l);
            for bb in 0..nb {
                o[bb * l..(bb + 1) * l].copy_from_slice(&v[(bb * ncb + g0) * hw * b..(bb * ncb + g1) * hw * b]);
            }
            outs.push(Tensor { shape: vec![nb, sz, h, w], data: Data::F(o), blk: b });
        }
        start += sz;
    }
    Ok(outs)
}

/// 풀링(NCHWc): 화소마다 B 칸을 탭 차례로 최대 · 합
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_blk(x: &Tensor, g: &ConvGeo, max: bool, incl: bool) -> R {
    let b = x.blk;
    let (nb, c, _, _) = dims4(x)?;
    let ncb = c.div_ceil(b);
    let xv = x.fv()?;
    let ohw = g.oh * g.ow;
    let mut o = buf::take::<f32>(nb * ncb * ohw * b);
    let th = pool::threads();
    let rch = (4 * th).div_ceil(nb * ncb).clamp(1, g.oh);
    let rpt = g.oh.div_ceil(rch);
    let op = P(o.as_mut_ptr());
    pool::for_each(nb * ncb * rch, &|t| {
        simd::run(|| {
            let (pl, ri) = (t / rch, t % rch);
            let plane = &xv[pl * g.h * g.w * b..(pl + 1) * g.h * g.w * b];
            for oy in ri * rpt..((ri + 1) * rpt).min(g.oh) {
                // SAFETY: 줄은 일마다 겹치지 않음
                let row = unsafe { std::slice::from_raw_parts_mut(op.at((pl * ohw + oy * g.ow) * b), g.ow * b) };
                if b == 16 {
                    // SAFETY: B = 16 은 AVX-512 확인 뒤에만
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        pool16(plane, g, oy, row, max, incl)
                    };
                } else {
                    pool_row::<8>(plane, g, oy, row, max, incl);
                }
            }
        })
    });
    Ok(vec![Tensor { shape: vec![nb, c, g.oh, g.ow], data: Data::F(o), blk: b }])
}

/// 풀링 출력 한 줄(NCHWc): 화소마다 B 칸을 탭 차례로 최대 · 합
#[inline(always)]
fn pool_row<const BB: usize>(plane: &[f32], g: &ConvGeo, oy: usize, row: &mut [f32], max: bool, incl: bool) {
    let (ky0, ky1, iy0) = krange(oy, g.sh, g.dh, g.pt, g.h, g.kh);
    let (oxl, oxh) = inner(g);
    for ox in 0..g.ow {
        let (kx0, kx1, ix0) = if ox >= oxl && ox < oxh { (0, g.kw, ox * g.sw - g.pl) } else { krange(ox, g.sw, g.dw, g.pl, g.w, g.kw) };
        let mut a = [if max { f32::NEG_INFINITY } else { 0.0 }; BB];
        let mut cnt = 0usize;
        if kx0 < kx1 {
            for ky in ky0..ky1 {
                let iy = iy0 + (ky - ky0) * g.dh;
                for kx in kx0..kx1 {
                    let ix = ix0 + (kx - kx0) * g.dw;
                    let s: &[f32; BB] = plane[(iy * g.w + ix) * BB..(iy * g.w + ix + 1) * BB].try_into().unwrap();
                    for l in 0..BB {
                        a[l] = if max { a[l].max(s[l]) } else { a[l] + s[l] };
                    }
                    cnt += 1;
                }
            }
        }
        if !max {
            let d = (if incl { g.kh * g.kw } else { cnt.max(1) }) as f32;
            a.iter_mut().for_each(|p| *p /= d);
        }
        row[ox * BB..(ox + 1) * BB].copy_from_slice(&a);
    }
}

/// Resize(H · W 만, NCHWc): 가장 가까운 칸 · 선형, 칸 고르기 · 무게는 NCHW 판과 같음
pub(crate) fn resize(x: &Tensor, oh: usize, ow: usize, ys: &[(usize, usize, f32)], xs: &[(usize, usize, f32)], linear: bool) -> R {
    let b = x.blk;
    let (nb, c, h, w) = dims4(x)?;
    let ncb = c.div_ceil(b);
    let xv = x.fv()?;
    let mut o = buf::take::<f32>(nb * ncb * oh * ow * b);
    pool::chunks_mut(&mut o, ow * b, |r, op| {
        let (pl, y) = (r / oh, r % oh);
        let plane = &xv[pl * h * w * b..(pl + 1) * h * w * b];
        let (y0, y1, fy) = ys[y];
        for (xx, &(x0, x1, fx)) in xs.iter().enumerate() {
            let d = &mut op[xx * b..(xx + 1) * b];
            if !linear {
                d.copy_from_slice(&plane[(y0 * w + x0) * b..(y0 * w + x0 + 1) * b]);
                continue;
            }
            let (p00, p01) = (&plane[(y0 * w + x0) * b..], &plane[(y0 * w + x1) * b..]);
            let (p10, p11) = (&plane[(y1 * w + x0) * b..], &plane[(y1 * w + x1) * b..]);
            for l in 0..b {
                let a = p00[l] * (1.0 - fx) + p01[l] * fx;
                let bb = p10[l] * (1.0 - fx) + p11[l] * fx;
                d[l] = a * (1.0 - fy) + bb * fy;
            }
        }
    });
    Ok(vec![Tensor { shape: vec![nb, c, oh, ow], data: Data::F(o), blk: b }])
}
