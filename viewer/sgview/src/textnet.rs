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

/// weight matrix kept as f16 bits [in, out] + f32 bias
type Lin = (Vec<u16>, Vec<f32>);

struct Layer {
    ln1: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    q: Lin,
    k: Lin,
    v: Lin,
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

/// Columns per block: a block's sums for all rows (64 × 64 f32 = 16 KB) stay in L1 while the weights stream once
const CB: usize = 64;

/// out[rows, n] = x[rows, k] · w[k, n] + b. Threads split the columns, so each weight is read from memory once per call
/// (the weights, not the 64 token rows, are what does not fit in cache)
fn linear(x: &[f32], k: usize, w: &Lin, out: &mut [f32]) {
    let n = w.1.len();
    let rows = x.len() / k;
    let blocks: Vec<Vec<f32>> = (0..n / CB)
        .into_par_iter()
        .map(|cb| {
            let c0 = cb * CB;
            let mut acc = vec![0f32; rows * CB];
            for r in 0..rows {
                acc[r * CB..(r + 1) * CB].copy_from_slice(&w.1[c0..c0 + CB]);
            }
            for j in 0..k {
                let mut wr = [0f32; CB];
                for (o, &h) in wr.iter_mut().zip(&w.0[j * n + c0..j * n + c0 + CB]) {
                    *o = f16(h);
                }
                for r in 0..rows {
                    let a = x[r * k + j];
                    let o: &mut [f32; CB] = (&mut acc[r * CB..(r + 1) * CB]).try_into().unwrap();
                    for c in 0..CB {
                        o[c] += a * wr[c];
                    }
                }
            }
            acc
        })
        .collect();
    for (cb, acc) in blocks.iter().enumerate() {
        for r in 0..rows {
            out[r * n + cb * CB..r * n + (cb + 1) * CB].copy_from_slice(&acc[r * CB..(r + 1) * CB]);
        }
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
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
                Ok((bits(&v)?, vec(&p(bias))?))
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
                q,
                k,
                v,
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
        let mut x = vec![0f32; T * D];
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.vocab {
                return Err(format!("token {id} outside the table"));
            }
            let row = {
                let f = self.tok.lock().unwrap();
                read_at(&f, self.tok_at + (id * D * 2) as u64, D * 2).map_err(|e| e.to_string())?
            };
            for d in 0..D {
                x[t * D + d] = f16(u16::from_le_bytes([row[2 * d], row[2 * d + 1]])) + self.pos[t * D + d];
            }
        }
        let (mut h, mut q, mut k, mut v, mut c, mut o) = (vec![0f32; T * D], vec![0f32; T * D], vec![0f32; T * D], vec![0f32; T * D], vec![0f32; T * D], vec![0f32; T * D]);
        let mut f = vec![0f32; T * F];
        for (l, ly) in self.layers.iter().enumerate() {
            // the last block only feeds the last token onward
            let rows = if l + 1 == LAYERS { T - 1..T } else { 0..T };
            for t in 0..T {
                layer_norm(&x[t * D..(t + 1) * D], &ly.ln1.0, &ly.ln1.1, &mut h[t * D..(t + 1) * D]);
            }
            let r = rows.start * D..rows.end * D;
            linear(&h[r.clone()], D, &ly.q, &mut q[r.clone()]);
            linear(&h, D, &ly.k, &mut k);
            linear(&h, D, &ly.v, &mut v);
            c[r.clone()].par_chunks_mut(D).enumerate().for_each(|(ti, cr)| {
                let t = rows.start + ti;
                let mut a = [0f32; T];
                for hh in 0..H {
                    let qv = &q[t * D + hh * HD..t * D + (hh + 1) * HD];
                    let mut mx = f32::NEG_INFINITY;
                    for s in 0..T {
                        let kv = &k[s * D + hh * HD..s * D + (hh + 1) * HD];
                        a[s] = qv.iter().zip(kv).map(|(x, y)| x * y).sum::<f32>() * 0.125;
                        mx = mx.max(a[s]);
                    }
                    let mut sum = 0.0;
                    for s in a.iter_mut() {
                        *s = (*s - mx).exp();
                        sum += *s;
                    }
                    let out = &mut cr[hh * HD..(hh + 1) * HD];
                    out.iter_mut().for_each(|x| *x = 0.0);
                    for s in 0..T {
                        let w = a[s] / sum;
                        let vv = &v[s * D + hh * HD..s * D + (hh + 1) * HD];
                        for (oo, &y) in out.iter_mut().zip(vv) {
                            *oo += w * y;
                        }
                    }
                }
            });
            linear(&c[r.clone()], D, &ly.o, &mut o[r.clone()]);
            for i in r.clone() {
                x[i] += o[i];
            }
            for t in rows.clone() {
                layer_norm(&x[t * D..(t + 1) * D], &ly.ln2.0, &ly.ln2.1, &mut h[t * D..(t + 1) * D]);
            }
            let rf = rows.start * F..rows.end * F;
            linear(&h[r.clone()], D, &ly.f1, &mut f[rf.clone()]);
            f[rf.clone()].par_iter_mut().for_each(|y| *y = gelu(*y));
            linear(&f[rf], F, &ly.f2, &mut o[r.clone()]);
            for i in r {
                x[i] += o[i];
            }
        }
        let mut last = vec![0f32; D];
        layer_norm(&x[(T - 1) * D..], &self.lnf.0, &self.lnf.1, &mut last);
        // head: Gemm with weight [out, in]
        let (w, b) = &self.head;
        Ok((0..D).map(|j| b[j] + w[j * D..(j + 1) * D].iter().zip(&last).map(|(a, c)| a * c).sum::<f32>()).collect())
    }
}
