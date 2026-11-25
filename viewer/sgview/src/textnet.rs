//! SigLIP2 text encoder (base patch32-256) in plain Rust: token + position embedding → 12 pre-LN transformer blocks
//! (12 heads × 64, GELU tanh, LayerNorm eps 1e-6, no attention mask) → final LayerNorm on the last token → head (Linear).
//! Weights are read from `text_model_fp16.onnx` on the first query (initializers; matrices stored [in, out]) and kept as f16;
//! the 256000-row token table stays in the file and only the rows a query uses are read.
//! Output is `pooler_output` (not normalised), matching ONNX Runtime within f32 rounding (docs/npu_models.md "글 인코더").
use rayon::prelude::*;
use std::path::Path;

const D: usize = 768;
const F: usize = 3072;
const H: usize = 12;
const HD: usize = 64;
const T: usize = 64;
const LAYERS: usize = 12;

/// weight matrix kept as f16 bits, panel-major [out / NB][in][NB] (see [`pack`]), + f32 bias
type Lin = (Vec<u16>, Vec<f32>);

struct Layer {
    ln1: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    /// q, k and v side by side (packed panels of q, then k, then v; bias q | k | v)
    qkv: Lin,
    o: Lin,
    f1: Lin,
    f2: Lin,
}

/// The encoder: weights are read on the first [`TextNet::run`] (a viewer that is never searched holds none of them)
pub struct TextNet {
    path: std::path::PathBuf,
    w: std::sync::OnceLock<Result<Weights, String>>,
}

struct Weights {
    /// token table (256000 × 768 f16) stays in the model file: rows are read when a token is used
    tok: std::sync::Mutex<std::fs::File>,
    tok_at: u64,
    pos: Vec<f32>,
    layers: Vec<Layer>,
    lnf: (Vec<f32>, Vec<f32>),
    /// head.weight as stored by Gemm(transB = 1): [out, in]
    head: (Vec<f32>, Vec<f32>),
    vocab: usize,
}

// ---------------------------------------------------------------- minimal protobuf reading (ONNX initializers)
fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
    let (mut v, mut s) = (0u64, 0);
    loop {
        let x = *b.get(*i)?;
        *i += 1;
        v |= ((x & 0x7f) as u64) << s;
        if x < 0x80 {
            return Some(v);
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
}

fn f16(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        let v = m as f32 * (1.0 / 16_777_216.0);
        return if s != 0 { -v } else { v };
    }
    f32::from_bits(if e == 0x1f { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 112) << 23) | (m << 13) })
}

/// Bytes [at, at + n) of the file
fn read_at(f: &std::fs::File, at: u64, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f;
    f.seek(SeekFrom::Start(at))?;
    let mut b = vec![0u8; n];
    let k = f.take(n as u64).read(&mut b)?;
    b.truncate(k);
    Ok(b)
}

/// Fields of the message stored in file bytes [start, end), reading only keys and lengths (bodies are skipped):
/// (field number, varint value, absolute byte range of a length-delimited body)
fn file_fields(f: &std::fs::File, start: u64, end: u64) -> Option<Vec<(u32, u64, std::ops::Range<u64>)>> {
    let mut out = vec![];
    let mut at = start;
    while at < end {
        let head = read_at(f, at, 20).ok()?;
        let mut i = 0;
        let key = varint(&head, &mut i)?;
        let (no, wt) = ((key >> 3) as u32, key & 7);
        match wt {
            0 => {
                let v = varint(&head, &mut i)?;
                out.push((no, v, 0..0));
                at += i as u64;
            }
            1 => at += i as u64 + 8,
            5 => at += i as u64 + 4,
            2 => {
                let n = varint(&head, &mut i)?;
                let b0 = at + i as u64;
                out.push((no, 0, b0..b0 + n));
                at = b0 + n;
            }
            _ => return None,
        }
    }
    (at == end).then_some(out)
}

/// fp16 initializers of the graph in file order: (name, dims, absolute byte range of the raw data). Reads only the small fields
fn initializers(f: &std::fs::File) -> Result<Vec<(String, Vec<i64>, std::ops::Range<u64>)>, String> {
    let bad = || "text model: not an ONNX file".to_string();
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let top = file_fields(f, 0, len).ok_or_else(bad)?;
    let g = top.iter().find(|x| x.0 == 7).ok_or_else(bad)?.2.clone();
    let mut out = vec![];
    for (no, _, r) in file_fields(f, g.start, g.end).ok_or_else(bad)? {
        if no != 5 {
            continue;
        }
        let (mut name, mut dims, mut raw, mut dtype) = (String::new(), vec![], None, 0);
        for (n, v, r) in file_fields(f, r.start, r.end).ok_or_else(bad)? {
            match n {
                1 if r.is_empty() => dims.push(v as i64),
                1 => {
                    let p = read_at(f, r.start, (r.end - r.start) as usize).map_err(|e| e.to_string())?;
                    let mut i = 0;
                    while i < p.len() {
                        dims.push(varint(&p, &mut i).ok_or_else(bad)? as i64);
                    }
                }
                2 => dtype = v,
                8 => name = String::from_utf8_lossy(&read_at(f, r.start, (r.end - r.start) as usize).map_err(|e| e.to_string())?).into_owned(),
                9 => raw = Some(r),
                _ => {}
            }
        }
        if dtype != 10 {
            continue; // only fp16 weights matter here
        }
        let Some(r) = raw else { continue };
        out.push((name, dims, r));
    }
    Ok(out)
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let m = x.iter().sum::<f32>() / n;
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / n;
    let r = 1.0 / (v + 1e-6).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - m) * r * w[i] + b[i];
    }
}

/// Columns per panel: a matrix is stored panel by panel ([n / NB][k][NB] f16, repacked at load), so a thread streams one panel
/// contiguously, unpacks it to f32 once (k · NB · 4 B ≤ 384 KB, stays in L2) and runs every row through it
const NB: usize = 32;

/// [k, n] row-major f16 → panel-major [n / NB][k][NB]
fn pack(w: &[u16], k: usize, n: usize) -> Vec<u16> {
    let mut p = Vec::with_capacity(k * n);
    for c0 in (0..n).step_by(NB) {
        for j in 0..k {
            p.extend_from_slice(&w[j * n + c0..j * n + c0 + NB]);
        }
    }
    p
}

/// f16 bits → f32 (exact conversion; hardware F16C / FCVTL where present, same values as [`f16`])
fn unpack(h: &[u16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
        unsafe { unpack_f16c(h, out) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let n8 = h.len() / 8 * 8;
        for (hc, oc) in h[..n8].chunks_exact(8).zip(out[..n8].chunks_exact_mut(8)) {
            unsafe {
                std::arch::asm!(
                    "ldr q0, [{s}]",
                    "fcvtl v1.4s, v0.4h",
                    "fcvtl2 v2.4s, v0.8h",
                    "stp q1, q2, [{d}]",
                    s = in(reg) hc.as_ptr(),
                    d = in(reg) oc.as_mut_ptr(),
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack),
                );
            }
        }
        for (o, &x) in out[n8..].iter_mut().zip(&h[n8..]) {
            *o = f16(x);
        }
        return;
    }
    #[allow(unreachable_code)]
    for (o, &x) in out.iter_mut().zip(h) {
        *o = f16(x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn unpack_f16c(h: &[u16], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n8 = h.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let v = _mm_loadu_si128(h.as_ptr().add(i) as *const __m128i);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_cvtph_ps(v));
        i += 8;
    }
    for j in n8..h.len() {
        out[j] = f16(h[j]);
    }
}

// Tiles: M rows × NB columns. Every kernel starts a sum at the bias and adds input j = 0, 1, … in order with fused
// multiply-add; fma is exactly rounded, so AVX-512, AVX2, NEON and the scalar fallback give the same bits.

#[inline(always)]
unsafe fn tile_scalar<const M: usize>(x: &[f32], k: usize, mr: usize, panel: &[f32], bias: &[f32], out: &mut [[f32; NB]]) {
    let mut acc = [[0f32; NB]; M];
    for a in acc.iter_mut() {
        a.copy_from_slice(bias);
    }
    for j in 0..k {
        let b = &panel[j * NB..(j + 1) * NB];
        for m in 0..M {
            let a = x[j * mr + m];
            for c in 0..NB {
                acc[m][c] = a.mul_add(b[c], acc[m][c]);
            }
        }
    }
    out[..M].copy_from_slice(&acc);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tile_avx512<const M: usize>(x: &[f32], k: usize, mr: usize, panel: &[f32], bias: &[f32], out: &mut [[f32; NB]]) {
    use std::arch::x86_64::*;
    let mut a0 = [_mm512_loadu_ps(bias.as_ptr()); M];
    let mut a1 = [_mm512_loadu_ps(bias.as_ptr().add(16)); M];
    let xp = x.as_ptr();
    let pp = panel.as_ptr();
    for j in 0..k {
        let w0 = _mm512_loadu_ps(pp.add(j * NB));
        let w1 = _mm512_loadu_ps(pp.add(j * NB + 16));
        for m in 0..M {
            let a = _mm512_set1_ps(*xp.add(j * mr + m));
            a0[m] = _mm512_fmadd_ps(a, w0, a0[m]);
            a1[m] = _mm512_fmadd_ps(a, w1, a1[m]);
        }
    }
    for m in 0..M {
        _mm512_storeu_ps(out[m].as_mut_ptr(), a0[m]);
        _mm512_storeu_ps(out[m].as_mut_ptr().add(16), a1[m]);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tile_avx2<const M: usize>(x: &[f32], k: usize, mr: usize, panel: &[f32], bias: &[f32], out: &mut [[f32; NB]]) {
    use std::arch::x86_64::*;
    let mut acc = [[_mm256_setzero_ps(); 4]; M];
    for m in 0..M {
        for q in 0..4 {
            acc[m][q] = _mm256_loadu_ps(bias.as_ptr().add(q * 8));
        }
    }
    let xp = x.as_ptr();
    let pp = panel.as_ptr();
    for j in 0..k {
        let w: [__m256; 4] = std::array::from_fn(|q| _mm256_loadu_ps(pp.add(j * NB + q * 8)));
        for m in 0..M {
            let a = _mm256_broadcast_ss(&*xp.add(j * mr + m));
            for q in 0..4 {
                acc[m][q] = _mm256_fmadd_ps(a, w[q], acc[m][q]);
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
unsafe fn tile_neon<const M: usize>(x: &[f32], k: usize, mr: usize, panel: &[f32], bias: &[f32], out: &mut [[f32; NB]]) {
    use std::arch::aarch64::*;
    let mut acc = [[vdupq_n_f32(0.0); 8]; M];
    for m in 0..M {
        for q in 0..8 {
            acc[m][q] = vld1q_f32(bias.as_ptr().add(q * 4));
        }
    }
    let xp = x.as_ptr();
    let pp = panel.as_ptr();
    for j in 0..k {
        let a: [f32; M] = std::array::from_fn(|m| *xp.add(j * mr + m));
        for h in 0..2 {
            let w: [float32x4_t; 4] = std::array::from_fn(|q| vld1q_f32(pp.add(j * NB + h * 16 + q * 4)));
            for m in 0..M {
                for q in 0..4 {
                    acc[m][h * 4 + q] = vfmaq_n_f32(acc[m][h * 4 + q], w[q], a[m]);
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

/// All rows through one panel. `x` is packed by [`pack_rows`]: tiles of MR rows, each [k][MR] (the last tile holds the rest)
macro_rules! rows_through {
    ($name:ident, $tile:ident, $mr:literal, $($attr:tt)*) => {
        $($attr)*
        unsafe fn $name(x: &[f32], k: usize, rows: usize, panel: &[f32], bias: &[f32], out: &mut [[f32; NB]]) {
            let mut r = 0;
            while r + $mr <= rows {
                $tile::<$mr>(&x[r * k..], k, $mr, panel, bias, &mut out[r..]);
                r += $mr;
            }
            let (x, o, m) = (&x[r * k..], &mut out[r..], rows - r);
            match m {
                0 => {}
                1 => $tile::<1>(x, k, m, panel, bias, o),
                2 => $tile::<2>(x, k, m, panel, bias, o),
                3 => $tile::<3>(x, k, m, panel, bias, o),
                4 => $tile::<4>(x, k, m, panel, bias, o),
                5 => $tile::<5>(x, k, m, panel, bias, o),
                6 => $tile::<6>(x, k, m, panel, bias, o),
                7 => $tile::<7>(x, k, m, panel, bias, o),
                8 => $tile::<8>(x, k, m, panel, bias, o),
                9 => $tile::<9>(x, k, m, panel, bias, o),
                10 => $tile::<10>(x, k, m, panel, bias, o),
                _ => $tile::<11>(x, k, m, panel, bias, o),
            }
        }
    };
}

/// x[rows, k] → tiles of `mr` rows stored [k][m] (m = mr, or the rest for the last tile), so a tile reads its inputs contiguously
fn pack_rows(x: &[f32], k: usize, mr: usize) -> Vec<f32> {
    let rows = x.len() / k;
    let mut p = vec![0f32; x.len()];
    let mut r0 = 0;
    while r0 < rows {
        let m = mr.min(rows - r0);
        let t = &mut p[r0 * k..(r0 + m) * k];
        for i in 0..m {
            for (j, &v) in x[(r0 + i) * k..(r0 + i + 1) * k].iter().enumerate() {
                t[j * m + i] = v;
            }
        }
        r0 += m;
    }
    p
}

rows_through!(rows_scalar, tile_scalar, 3, #[inline(never)]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx512, tile_avx512, 12, #[target_feature(enable = "avx512f")]);
#[cfg(target_arch = "x86_64")]
rows_through!(rows_avx2, tile_avx2, 3, #[target_feature(enable = "avx2,fma")]);
#[cfg(target_arch = "aarch64")]
rows_through!(rows_neon, tile_neon, 3, #[target_feature(enable = "neon")]);

type RowsFn = (unsafe fn(&[f32], usize, usize, &[f32], &[f32], &mut [[f32; NB]]), usize);

/// The widest kernel this CPU runs
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

/// out[rows, n] = x[rows, k] · w[k, n] + b. Threads split the column panels, so each weight is read from memory once per call
/// (the weights, not the 64 token rows, are what does not fit in cache)
fn linear(x: &[f32], k: usize, w: &Lin, out: &mut [f32]) {
    let n = w.1.len();
    let rows = x.len() / k;
    let (kern, mr) = kernel();
    let xp = pack_rows(x, k, mr);
    let x = &xp[..];
    let blocks: Vec<Vec<[f32; NB]>> = (0..n / NB)
        .into_par_iter()
        .map(|p| {
            thread_local! { static PANEL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) }; }
            PANEL.with(|pb| {
                let mut pb = pb.borrow_mut();
                pb.resize(k * NB, 0.0);
                unpack(&w.0[p * k * NB..(p + 1) * k * NB], &mut pb);
                let mut o = vec![[0f32; NB]; rows];
                unsafe { kern(x, k, rows, &pb, &w.1[p * NB..(p + 1) * NB], &mut o) };
                o
            })
        })
        .collect();
    for (p, o) in blocks.iter().enumerate() {
        for r in 0..rows {
            out[r * n + p * NB..r * n + (p + 1) * NB].copy_from_slice(&o[r]);
        }
    }
}

/// e^z, Cephes single-precision polynomial (relative error ~1e-7) in plain arithmetic so loops over it vectorise
#[inline(always)]
fn exp(z: f32) -> f32 {
    let z = z.clamp(-87.0, 88.0);
    let t = z * std::f32::consts::LOG2_E + 12_582_912.0;
    let n = t - 12_582_912.0;
    let r = z - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((t.to_bits() as i32 - 0x4B40_0000 + 127) as u32) << 23)
}

/// GELU, tanh form: 0.5 x (1 + tanh(y)) = x / (1 + e^(-2y))
#[inline(always)]
fn gelu(x: f32) -> f32 {
    x / (1.0 + exp(-1.595_769_2 * (x + 0.044715 * x * x * x)))
}

impl TextNet {
    /// Remembers the model file; weights are read on the first run
    pub fn load(path: &Path) -> Result<TextNet, String> {
        if !path.exists() {
            return Err(format!("{}: no such file", path.display()));
        }
        Ok(TextNet { path: path.to_path_buf(), w: std::sync::OnceLock::new() })
    }

    fn weights(&self) -> Result<&Weights, String> {
        self.w.get_or_init(|| Weights::read(&self.path)).as_ref().map_err(|e| e.clone())
    }

    /// TOKENS token ids → pooler_output (768)
    pub fn run(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.weights()?.run(ids)
    }
}

impl Weights {
    fn read(path: &Path) -> Result<Weights, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let inits = initializers(&file)?;
        let bits = |r: &std::ops::Range<u64>| -> Result<Vec<u16>, String> {
            let b = read_at(&file, r.start, (r.end - r.start) as usize).map_err(|e| e.to_string())?;
            Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        };
        let mut by: std::collections::HashMap<String, (Vec<i64>, std::ops::Range<u64>)> = Default::default();
        let mut mats = vec![]; // unnamed MatMul weights in file order: q k v out fc1 fc2 per layer
        for (n, d, v) in inits {
            if n.starts_with("onnx::MatMul") {
                mats.push((d, v));
            } else {
                by.insert(n, (d, v));
            }
        }
        if mats.len() != LAYERS * 6 {
            return Err(format!("text model: {} matrices, expected {}", mats.len(), LAYERS * 6));
        }
        let take = |n: &str| -> Result<(Vec<i64>, std::ops::Range<u64>), String> { by.get(n).cloned().ok_or(format!("text model: no {n}")) };
        let vec = |n: &str| -> Result<Vec<f32>, String> { Ok(bits(&take(n)?.1)?.iter().map(|&h| f16(h)).collect()) };
        let (tdims, tok) = take("text_model.embeddings.token_embedding.weight")?;
        let vocab = tdims[0] as usize;
        let pos = vec("text_model.embeddings.position_embedding.weight")?;
        let mut mats = mats.into_iter();
        let mut layers = vec![];
        for l in 0..LAYERS {
            let p = |s: &str| format!("text_model.encoder.layers.{l}.{s}");
            let mut m = |shape: [i64; 2], bias: &str| -> Result<Lin, String> {
                let (d, v) = mats.next().unwrap();
                if d != shape {
                    return Err(format!("text model: layer {l} matrix {d:?}, expected {shape:?}"));
                }
                Ok((pack(&bits(&v)?, shape[0] as usize, shape[1] as usize), vec(&p(bias))?))
            };
            let q = m([768, 768], "self_attn.q_proj.bias")?;
            let k = m([768, 768], "self_attn.k_proj.bias")?;
            let v = m([768, 768], "self_attn.v_proj.bias")?;
            let o = m([768, 768], "self_attn.out_proj.bias")?;
            let f1 = m([768, 3072], "mlp.fc1.bias")?;
            let f2 = m([3072, 768], "mlp.fc2.bias")?;
            layers.push(Layer {
                ln1: (vec(&p("layer_norm1.weight"))?, vec(&p("layer_norm1.bias"))?),
                ln2: (vec(&p("layer_norm2.weight"))?, vec(&p("layer_norm2.bias"))?),
                qkv: ([q.0, k.0, v.0].concat(), [q.1, k.1, v.1].concat()),
                o,
                f1,
                f2,
            });
        }
        let (lnf, head) = ((vec("text_model.final_layer_norm.weight")?, vec("text_model.final_layer_norm.bias")?), (vec("text_model.head.weight")?, vec("text_model.head.bias")?));
        Ok(Weights {
            tok_at: tok.start,
            tok: std::sync::Mutex::new(file),
            pos,
            layers,
            lnf,
            head,
            vocab,
        })
    }

    fn run(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.len() != T {
            return Err(format!("text encoder takes {T} tokens"));
        }
        let prof = std::env::var_os("TEXTNET_PROF").is_some();
        let mut tm = [0f64; 6];
        let mut t0 = std::time::Instant::now();
        let mut lap = |i: usize, t0: &mut std::time::Instant| {
            if prof {
                tm[i] += t0.elapsed().as_secs_f64();
                *t0 = std::time::Instant::now();
            }
        };
        // a token's row is read once per query (padding repeats one id)
        let mut x = vec![0f32; T * D];
        let mut seen: Vec<(u32, usize)> = vec![];
        for (t, &id) in ids.iter().enumerate() {
            if id as usize >= self.vocab {
                return Err(format!("token {id} outside the table"));
            }
            if let Some(&(_, t1)) = seen.iter().find(|s| s.0 == id) {
                x.copy_within(t1 * D..(t1 + 1) * D, t * D);
            } else {
                let row = {
                    let f = self.tok.lock().unwrap();
                    read_at(&f, self.tok_at + (id as usize * D * 2) as u64, D * 2).map_err(|e| e.to_string())?
                };
                for d in 0..D {
                    x[t * D + d] = f16(u16::from_le_bytes([row[2 * d], row[2 * d + 1]]));
                }
                seen.push((id, t));
            }
        }
        for t in 0..T {
            for d in 0..D {
                x[t * D + d] += self.pos[t * D + d];
            }
        }
        lap(0, &mut t0);
        let (mut h, mut c, mut o) = (vec![0f32; T * D], vec![0f32; T * D], vec![0f32; T * D]);
        let mut qkv = vec![0f32; T * 3 * D];
        let mut f = vec![0f32; T * F];
        // keys per head, transposed: kt[head][d][s]
        let mut kt = vec![0f32; H * HD * T];
        for (l, ly) in self.layers.iter().enumerate() {
            // the last block only feeds the last token onward
            let rows = if l + 1 == LAYERS { T - 1..T } else { 0..T };
            for t in 0..T {
                layer_norm(&x[t * D..(t + 1) * D], &ly.ln1.0, &ly.ln1.1, &mut h[t * D..(t + 1) * D]);
            }
            lap(1, &mut t0);
            linear(&h, D, &ly.qkv, &mut qkv);
            lap(2, &mut t0);
            for s in 0..T {
                for hh in 0..H {
                    for d in 0..HD {
                        kt[(hh * HD + d) * T + s] = qkv[s * 3 * D + D + hh * HD + d];
                    }
                }
            }
            let (qkv, kt) = (&qkv, &kt);
            let r = rows.start * D..rows.end * D;
            c[r.clone()].par_chunks_mut(D).enumerate().for_each(|(ti, cr)| {
                let t = rows.start + ti;
                for hh in 0..H {
                    let qv = &qkv[t * 3 * D + hh * HD..t * 3 * D + (hh + 1) * HD];
                    let mut a = [0f32; T];
                    for d in 0..HD {
                        let kr: &[f32; T] = kt[(hh * HD + d) * T..(hh * HD + d + 1) * T].try_into().unwrap();
                        let qd = qv[d];
                        for s in 0..T {
                            a[s] += qd * kr[s];
                        }
                    }
                    let mut mx = f32::NEG_INFINITY;
                    for s in a.iter_mut() {
                        *s *= 0.125;
                        mx = mx.max(*s);
                    }
                    let mut sum = 0.0;
                    for s in a.iter_mut() {
                        *s = exp(*s - mx);
                        sum += *s;
                    }
                    let out: &mut [f32; HD] = (&mut cr[hh * HD..(hh + 1) * HD]).try_into().unwrap();
                    *out = [0.0; HD];
                    for s in 0..T {
                        let w = a[s] / sum;
                        let vv: &[f32; HD] = qkv[s * 3 * D + 2 * D + hh * HD..s * 3 * D + 2 * D + (hh + 1) * HD].try_into().unwrap();
                        for d in 0..HD {
                            out[d] += w * vv[d];
                        }
                    }
                }
            });
            lap(3, &mut t0);
            linear(&c[r.clone()], D, &ly.o, &mut o[r.clone()]);
            for i in r.clone() {
                x[i] += o[i];
            }
            for t in rows.clone() {
                layer_norm(&x[t * D..(t + 1) * D], &ly.ln2.0, &ly.ln2.1, &mut h[t * D..(t + 1) * D]);
            }
            lap(1, &mut t0);
            let rf = rows.start * F..rows.end * F;
            linear(&h[r.clone()], D, &ly.f1, &mut f[rf.clone()]);
            lap(2, &mut t0);
            f[rf.clone()].par_chunks_mut(F).for_each(|y| y.iter_mut().for_each(|y| *y = gelu(*y)));
            lap(4, &mut t0);
            linear(&f[rf], F, &ly.f2, &mut o[r.clone()]);
            lap(2, &mut t0);
            for i in r {
                x[i] += o[i];
            }
        }
        let mut last = vec![0f32; D];
        layer_norm(&x[(T - 1) * D..], &self.lnf.0, &self.lnf.1, &mut last);
        // head: Gemm with weight [out, in]
        let (w, b) = &self.head;
        let out = (0..D).map(|j| b[j] + w[j * D..(j + 1) * D].iter().zip(&last).map(|(a, c)| a * c).sum::<f32>()).collect();
        lap(5, &mut t0);
        if prof {
            eprintln!("prof ms: embed {:.2} ln {:.2} linear {:.2} attn {:.2} gelu {:.2} head {:.2}", tm[0] * 1e3, tm[1] * 1e3, tm[2] * 1e3, tm[3] * 1e3, tm[4] * 1e3, tm[5] * 1e3);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod kbench {
    #[test]
    fn kernel_speed() {
        if std::env::var_os("TEXTNET_KBENCH").is_none() { return; }
        let (k, n, rows) = (3072usize, 768usize, 64usize);
        let w: Vec<u16> = (0..k * n).map(|i| 0x3000 + (i % 1000) as u16).collect();
        let lin = (super::pack(&w, k, n), vec![0.1f32; n]);
        let x: Vec<f32> = (0..rows * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let mut o = vec![0f32; rows * n];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            for _ in 0..3 { super::linear(&x, k, &lin, &mut o); }
            let t0 = std::time::Instant::now();
            for _ in 0..20 { super::linear(&x, k, &lin, &mut o); }
            let dt = t0.elapsed().as_secs_f64() / 20.0;
            println!("kbench 1 thread: {:.2} ms, {:.1} GFLOPS, kernel mr {}", dt * 1e3, 2.0 * (k * n * rows) as f64 / dt / 1e9, super::kernel().1);
        });
    }
}
