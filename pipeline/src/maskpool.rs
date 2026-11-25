//! 마스크 안 토큰으로 사진 벡터 만들기(이름 붙이기용). SigLIP2 그림 인코더 끝의 주의 풀링 머리(MAP head)를 다시 계산하되,
//! 사진 64 조각(32×32 화소) 토큰마다 물체 마스크가 덮은 몫을 무게로 줘 마스크 밖 토큰(둘레 책상 · 모니터)은 보지 않는다.
//! 사진 전체 벡터(pooler_output)는 둘레가 사진 대부분이면 둘레 이름(의자 등받이 → desk)이 붙었다(docs/pose_experiments.md 병렬 C).
//!
//! 머리 계산(토큰 h_n, 탐침 q, 머리 12 개 × 64):
//!   a_hn = softmax_n((q_h · k_hn)/8 + ln w_n), k = h W_k + b_k, v = h W_v + b_v
//!   o_h = Σ a_hn v_hn = (Σ a_hn h_n) W_v,h + b_v,h        (무게 합 1 이라 b_v 는 그대로)
//!   q_h · k_hn = h_n · (W_k,h q_h) + 상수                    (상수는 softmax 에서 지워짐)
//! 그래서 토큰마다 768×1536 곱 대신 768×12 곱 · 머리별 768×64 곱만 한다(64 토큰 사진 한 장 약 6 M 곱).
//! 무게는 vision_model.onnx 의 초기값(initializer)에서 읽는다. 머리 출력이 ONNX pooler_output 과 같음(모든 무게 1)은 tools/naming/maphead.py 로 확인.

/// 토큰 수(256×256 사진, 32 화소 조각 → 8×8)
pub const TOKENS: usize = 64;
const D: usize = crate::embed::DIM;
const HEADS: usize = 12;
const HD: usize = D / HEADS;

pub struct Head {
    /// 머리별 W_k,h q_h / 8: [HEADS][D]
    qk: Vec<f32>,
    /// W_v: [D][D] (입력 행, 출력 열 = 머리 h 는 열 h·64..)
    wv: Vec<f32>,
    bv: Vec<f32>,
    /// out_proj: [출력][입력] (Gemm transB)
    wo: Vec<f32>,
    bo: Vec<f32>,
    lw: Vec<f32>,
    lb: Vec<f32>,
    /// fc1 [D][4D], fc2 [4D][D] (입력 행)
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

// ---- ONNX 초기값 읽기(필요한 필드만: ModelProto.graph(7) → GraphProto.initializer(5) → TensorProto name(8) · raw_data(9))
fn varint(b: &[u8], p: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for s in (0..64).step_by(7) {
        let c = *b.get(*p)?;
        *p += 1;
        v |= ((c & 0x7f) as u64) << s;
        if c < 0x80 {
            return Some(v);
        }
    }
    None
}
/// 메시지 b 의 필드들: (번호, 길이 있는 값이면 그 범위)
fn fields(b: &[u8]) -> Vec<(u64, Option<std::ops::Range<usize>>)> {
    let (mut p, mut o) = (0, vec![]);
    while p < b.len() {
        let Some(k) = varint(b, &mut p) else { break };
        match k & 7 {
            0 => {
                varint(b, &mut p);
                o.push((k >> 3, None));
            }
            1 => p += 8,
            5 => p += 4,
            2 => {
                let Some(n) = varint(b, &mut p) else { break };
                let r = p..(p + n as usize).min(b.len());
                p = r.end;
                o.push((k >> 3, Some(r)));
            }
            _ => break,
        }
    }
    o
}

impl Head {
    /// vision_model.onnx 에서 머리 무게를 읽는다. 머리가 없는 파일(.dxnn 등)이면 None
    pub fn load(model: &str) -> Option<Head> {
        if !model.ends_with(".onnx") {
            return None;
        }
        let b = std::fs::read(model).ok()?;
        let graph = fields(&b).into_iter().find(|f| f.0 == 7)?.1?;
        let g = &b[graph.clone()];
        let mut t: std::collections::HashMap<String, Vec<f32>> = Default::default();
        let want = |n: &str| n.contains("/head/") || n.starts_with("vision_model.head.") || n == "onnx::MatMul_2025" || n == "onnx::MatMul_2026";
        for (k, r) in fields(g) {
            let (5, Some(r)) = (k, r) else { continue };
            let tb = &g[r];
            let (mut name, mut raw) = (String::new(), None);
            for (k, r) in fields(tb) {
                match (k, r) {
                    (8, Some(r)) => name = String::from_utf8_lossy(&tb[r]).into_owned(),
                    (9, Some(r)) => raw = Some(r),
                    _ => {}
                }
            }
            if let (true, Some(r)) = (want(&name), raw) {
                t.insert(name, tb[r].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
            }
        }
        let mut get = |k: &str, n: usize| t.remove(k).filter(|v| v.len() == n);
        let probe = get("/vision_model/head/Expand_output_0", D)?;
        let (wq, bq) = (get("/vision_model/head/attention/Transpose_2_output_0", D * D)?, get("/vision_model/head/attention/Slice_3_output_0", D)?);
        let (wkv, bkv) = (get("/vision_model/head/attention/Transpose_3_output_0", D * 2 * D)?, get("/vision_model/head/attention/Slice_4_output_0", 2 * D)?);
        // q = probe W_q + b_q, qk[h][i] = Σ_d W_k[i][h·64 + d] q[h·64 + d] / 8
        let q: Vec<f32> = (0..D).map(|j| bq[j] + (0..D).map(|i| probe[i] * wq[i * D + j]).sum::<f32>()).collect();
        let mut qk = vec![0f32; HEADS * D];
        for h in 0..HEADS {
            for i in 0..D {
                qk[h * D + i] = (0..HD).map(|d| wkv[i * 2 * D + h * HD + d] * q[h * HD + d]).sum::<f32>() / (HD as f32).sqrt();
            }
        }
        let wv: Vec<f32> = (0..D).flat_map(|i| wkv[i * 2 * D + D..i * 2 * D + 2 * D].to_vec()).collect();
        Some(Head {
            qk,
            wv,
            bv: bkv[D..].to_vec(),
            wo: get("vision_model.head.attention.out_proj.weight", D * D)?,
            bo: get("vision_model.head.attention.out_proj.bias", D)?,
            lw: get("vision_model.head.layernorm.weight", D)?,
            lb: get("vision_model.head.layernorm.bias", D)?,
            w1: get("onnx::MatMul_2025", D * 4 * D)?,
            b1: get("vision_model.head.mlp.fc1.bias", 4 * D)?,
            w2: get("onnx::MatMul_2026", 4 * D * D)?,
            b2: get("vision_model.head.mlp.fc2.bias", D)?,
        })
    }

    /// 토큰 h([TOKENS][D], 인코더 last_hidden_state)를 무게 w(토큰마다 0..1, 0 = 안 봄)로 모은 사진 벡터(길이 1 아님)
    pub fn pool(&self, h: &[f32], w: &[f32; TOKENS]) -> Vec<f32> {
        // 머리별 주의 무게
        let mut a = [[0f32; TOKENS]; HEADS];
        for (hd, ah) in a.iter_mut().enumerate() {
            let qk = &self.qk[hd * D..(hd + 1) * D];
            for (n, x) in ah.iter_mut().enumerate() {
                *x = if w[n] > 0.0 { h[n * D..(n + 1) * D].iter().zip(qk).map(|(p, q)| p * q).sum::<f32>() + w[n].ln() } else { f32::NEG_INFINITY };
            }
            let m = ah.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            ah.iter_mut().for_each(|x| {
                *x = (*x - m).exp();
                s += *x;
            });
            ah.iter_mut().for_each(|x| *x /= s);
        }
        // 머리별 토큰 가중 평균 → W_v 의 그 머리 열
        let mut o = self.bv.clone();
        let mut hb = vec![0f32; D];
        for (hd, ah) in a.iter().enumerate() {
            hb.iter_mut().for_each(|x| *x = 0.0);
            for (n, &an) in ah.iter().enumerate().filter(|x| *x.1 > 0.0) {
                hb.iter_mut().zip(&h[n * D..(n + 1) * D]).for_each(|(x, y)| *x += an * y);
            }
            for (i, &x) in hb.iter().enumerate() {
                let row = &self.wv[i * D + hd * HD..i * D + (hd + 1) * HD];
                o[hd * HD..(hd + 1) * HD].iter_mut().zip(row).for_each(|(y, w)| *y += x * w);
            }
        }
        let r: Vec<f32> = (0..D).map(|j| self.bo[j] + self.wo[j * D..(j + 1) * D].iter().zip(&o).map(|(w, x)| w * x).sum::<f32>()).collect();
        let mean = r.iter().sum::<f32>() / D as f32;
        let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / D as f32;
        let x: Vec<f32> = r.iter().enumerate().map(|(j, v)| (v - mean) / (var + 1e-6).sqrt() * self.lw[j] + self.lb[j]).collect();
        let mut u = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            u.iter_mut().zip(&self.w1[i * 4 * D..(i + 1) * 4 * D]).for_each(|(y, w)| *y += xi * w);
        }
        u.iter_mut().for_each(|v| *v = 0.5 * *v * (1.0 + (0.797_884_6 * (*v + 0.044715 * *v * *v * *v)).tanh()));
        let mut out = r;
        out.iter_mut().zip(&self.b2).for_each(|(y, b)| *y += b);
        for (i, &ui) in u.iter().enumerate() {
            out.iter_mut().zip(&self.w2[i * D..(i + 1) * D]).for_each(|(y, w)| *y += ui * w);
        }
        out
    }
}

/// 사진(w×h) 마스크(0 · 255)가 256×256 으로 줄인 사진의 8×8 조각마다 덮은 몫(0..1)
pub fn weights(mask: &[u8], w: usize, h: usize) -> [f32; TOKENS] {
    let mut s = [0f32; TOKENS];
    let mut c = [0f32; TOKENS];
    for y in 0..h {
        let ty = (y * 8 / h).min(7);
        for x in 0..w {
            let k = ty * 8 + (x * 8 / w).min(7);
            c[k] += 1.0;
            if mask[y * w + x] > 127 {
                s[k] += 1.0;
            }
        }
    }
    std::array::from_fn(|k| if c[k] > 0.0 { s[k] / c[k] } else { 0.0 })
}

#[cfg(test)]
mod tests {
    #[test]
    fn weights_fraction() {
        let mut m = vec![0u8; 16 * 16];
        for y in 0..8 {
            for x in 0..2 {
                m[y * 16 + x] = 255;
            }
        }
        let w = super::weights(&m, 16, 16);
        assert_eq!(w[0], 1.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[8 * 4], 0.0);
    }
}

/// Rust 머리 = ONNX pooler_output 확인(모든 무게 1) · 마스크 무게 판을 파이썬 계산(tools/naming/maphead.py)과 견줄 값 출력.
/// 모델이 없으면 건너뜀: ONDEVICE_MODELS=<모델 폴더> cargo test --release maskpool -- --nocapture
#[cfg(all(test, feature = "onnx"))]
mod onnx_tests {
    #[test]
    fn head_matches_onnx() {
        let dir = std::env::var("ONDEVICE_MODELS").unwrap_or_default();
        let model = format!("{dir}/siglip2-base-patch32-256/vision_model.onnx");
        let Some(h) = super::Head::load(&model) else { return };
        let mut sess = ort::session::Session::builder(common::ort::env()).unwrap().commit_from_file(&model).unwrap();
        // 정해진 무늬 입력(파이썬과 같게): x[i] = sin(i · 0.001)
        let x: Vec<f32> = (0..3 * 256 * 256).map(|i| (i as f32 * 0.001).sin()).collect();
        let inp = ort::value::TensorRef::from_array_view(([1usize, 3, 256, 256], x.as_slice())).unwrap();
        let out = sess.run(ort::inputs![inp]).unwrap();
        let po = out["pooler_output"].try_extract_tensor::<f32>().unwrap().1.to_vec();
        let lh = out["last_hidden_state"].try_extract_tensor::<f32>().unwrap().1.to_vec();
        let all = h.pool(&lh, &[1.0; super::TOKENS]);
        let d = all.iter().zip(&po).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // 마스크 무게 판: 왼쪽 위 4×4 조각만, 몫 (k % 4 + 1)/4
        let w: [f32; super::TOKENS] = std::array::from_fn(|k| if k % 8 < 4 && k / 8 < 4 { (k % 4 + 1) as f32 / 4.0 } else { 0.0 });
        let m = h.pool(&lh, &w);
        println!("maskpool: 모든 무게 1 → pooler 와 최대 차 {d:e}, 마스크 판 앞 4 값 {:?} 합 {}", &m[..4], m.iter().sum::<f32>());
        assert!(d < 1e-3, "{d}");
    }
}
