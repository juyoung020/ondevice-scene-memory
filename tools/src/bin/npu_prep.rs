//! NPU(DX-M1) 컴파일 전 준비 도구. 자세한 순서는 docs/npu_models.md.
//!
//!   npu_prep siglip-patches <영상 인코더.onnx> <나올.onnx>
//!     SigLIP2 영상 인코더의 조각 나누기 합성곱(32×32, 보폭 32)을 행렬곱으로 바꾸고(입력 `patches` [1,64,3072]),
//!     끝의 모양 바꾸기·고르기(Reshape·Transpose·Gather)를 없애 출력 `pooler_output` [1,768] 까지 NPU 에 다 오르게 한다.
//!     가중치 파일(external data)은 원래 것을 그대로 가리키므로 나올 파일은 입력과 같은 폴더에 둔다.
//!   npu_prep cut <in.onnx> <out.onnx> <출력 텐서,…>
//!     주어진 텐서까지만 남긴 모델(그 뒤 노드·쓰이지 않는 가중치를 뺌). 분할 모델을 머리 출력에서 끊어
//!     모델 안 후처리(상위 300 고르기, CPU)를 없앨 때 쓴다(DX-COM 의 --compile_output_nodes 는 뒤를 CPU 작업으로 남김).
//!   npu_prep patches-calib <사진 폴더> <나올.npy> [장 수=200]
//!     보정 자료: png·jpg 를 256×256 쌍선형으로 줄여 (값/127.5 − 1) 조각 배치로 [N,64,3072] f32 .npy (DX-COM dataloader 가 한 장씩 꺼내 묶음 차원 1 을 붙임) 로 쓴다
//!     (pipeline/src/embed.rs 의 preprocess_into(patches = true) 와 같은 식).
use std::path::Path;
use tools::pb::{Msg, Val};

const DROP: [&str; 7] = [
    "/embeddings/patch_embedding/Conv",
    "/embeddings/Reshape",
    "/embeddings/Transpose",
    "/head/attention/Reshape_7",
    "/head/attention/Transpose_7",
    "Reshape_145",
    "/head/Gather",
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("siglip-patches") if a.len() == 4 => siglip_patches(&a[2], &a[3]),
        Some("cut") if a.len() == 5 => cut(&a[2], &a[3], &a[4].split(',').collect::<Vec<_>>()),
        Some("patches-calib") if a.len() >= 4 => patches_calib(&a[2], &a[3], a.get(4).map_or(200, |s| s.parse().unwrap())),
        _ => Err("쓰기: npu_prep siglip-patches <in.onnx> <out.onnx> | npu_prep cut <in.onnx> <out.onnx> <출력,…> | npu_prep patches-calib <사진 폴더> <out.npy> [N]".into()),
    };
    if let Err(e) = r {
        eprintln!("npu_prep: {e}");
        std::process::exit(1);
    }
}

/// TensorProto 의 f32 값(raw_data · float_data · external data)
fn tensor_f32(t: &Msg, dir: &Path) -> Result<Vec<f32>, String> {
    let le = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>();
    if let Some(Val::Bytes(b)) = t.0.iter().find(|(n, _)| *n == 9).map(|x| &x.1) {
        return Ok(le(b));
    }
    let ext = t.msgs(13);
    if !ext.is_empty() {
        let kv = |k: &str| ext.iter().find(|e| e.str(1).as_deref() == Some(k)).and_then(|e| e.str(2));
        let loc = kv("location").ok_or("external data 에 location 없음")?;
        let off: usize = kv("offset").map_or(Ok(0), |s| s.parse()).map_err(|e| format!("{e}"))?;
        let len: usize = kv("length").ok_or("external data 에 length 없음")?.parse().map_err(|e| format!("{e}"))?;
        let f = std::fs::read(dir.join(&loc)).map_err(|e| format!("{loc}: {e}"))?;
        return Ok(le(f.get(off..off + len).ok_or("external data 범위 밖")?));
    }
    Ok(t.0.iter().filter(|(n, _)| *n == 4).flat_map(|(_, v)| match v {
        Val::Bytes(b) => le(b),
        Val::I32(b) => vec![f32::from_le_bytes(*b)],
        _ => vec![],
    }).collect())
}

fn tensor(name: &str, dims: &[i64], v: &[f32]) -> Msg {
    let mut t = Msg::default();
    for &d in dims {
        t.push_int(1, d);
    }
    t.push_int(2, 1); // FLOAT
    t.push_str(8, name);
    t.0.push((9, Val::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect())));
    t
}

fn node(op: &str, name: &str, ins: &[&str], outs: &[&str]) -> Msg {
    let mut n = Msg::default();
    for i in ins {
        n.push_str(1, i);
    }
    for o in outs {
        n.push_str(2, o);
    }
    n.push_str(3, name);
    n.push_str(4, op);
    n
}

/// ValueInfoProto: 실수 텐서
fn value_info(name: &str, dims: &[i64]) -> Msg {
    let mut shape = Msg::default();
    for &d in dims {
        let mut dim = Msg::default();
        dim.push_int(1, d);
        shape.push_msg(1, &dim);
    }
    let mut tt = Msg::default();
    tt.push_int(1, 1);
    tt.push_msg(2, &shape);
    let mut ty = Msg::default();
    ty.push_msg(1, &tt);
    let mut vi = Msg::default();
    vi.push_str(1, name);
    vi.push_msg(2, &ty);
    vi
}

fn siglip_patches(src: &str, dst: &str) -> Result<(), String> {
    let dir = Path::new(src).parent().unwrap_or(Path::new("."));
    if Path::new(dst).parent().unwrap_or(Path::new(".")).canonicalize().ok() != dir.canonicalize().ok() {
        return Err("나올 파일은 입력과 같은 폴더에(가중치 파일을 그대로 가리킴)".into());
    }
    let mut model = Msg::parse(&std::fs::read(src).map_err(|e| format!("{src}: {e}"))?)?;
    let gi = model.0.iter().position(|(n, _)| *n == 7).ok_or("graph 없음")?;
    let Val::Bytes(gb) = &model.0[gi].1 else { return Err("graph 형식".into()) };
    let g = Msg::parse(gb)?;
    let inits = g.msgs(5);
    let find = |name: &str| inits.iter().find(|t| t.str(8).as_deref() == Some(name)).ok_or(format!("가중치 {name} 없음"));
    let w = tensor_f32(find("p.v.embeddings.patch_embedding.weight")?, dir)?;
    let b = tensor_f32(find("p.v.embeddings.patch_embedding.bias")?, dir)?;
    if w.len() != 768 * 3072 || b.len() != 768 {
        return Err(format!("조각 합성곱 크기가 다름: {} · {}", w.len(), b.len()));
    }
    // [768, 3·32·32] → [3072, 768] (행렬곱 오른쪽)
    let mut wt = vec![0f32; 3072 * 768];
    for o in 0..768 {
        for i in 0..3072 {
            wt[i * 768 + o] = w[o * 3072 + i];
        }
    }
    let rename = |s: &str| match s {
        "/head/attention/Transpose_7_output_0" => "/head/attention/Gemm_output_0".to_string(),
        "/head/mlp/fc2/Add_output_0" => "_v_2052".to_string(),
        s => s.to_string(),
    };
    let mut ng = Msg::default();
    ng.push_msg(1, &node("MatMul", "patch_mm", &["patches", "patch_w"], &["pe_mm"]));
    ng.push_msg(1, &node("Add", "patch_add", &["pe_mm", "patch_b"], &["/embeddings/Transpose_output_0"]));
    let mut dropped = 0;
    for (no, v) in &g.0 {
        match no {
            1 => {
                let Val::Bytes(nb) = v else { continue };
                let n = Msg::parse(nb)?;
                let name = n.str(3).unwrap_or_default();
                if DROP.contains(&name.as_str()) {
                    dropped += 1;
                    continue;
                }
                let mut m = Msg::default();
                for (k, x) in &n.0 {
                    match (k, x) {
                        (1, Val::Bytes(s)) => m.push_str(1, &rename(&String::from_utf8_lossy(s))),
                        (2, Val::Bytes(_)) if name == "/head/Add" => m.push_str(2, "pooler_output"),
                        _ => m.0.push((*k, x.clone())),
                    }
                }
                ng.push_msg(1, &m);
            }
            11 | 12 | 13 => {} // 입력·출력은 아래에서 새로, 중간 모양 정보는 버림
            _ => ng.0.push((*no, v.clone())),
        }
    }
    if dropped != DROP.len() {
        return Err(format!("없앨 노드 {} 개 가운데 {dropped} 개만 찾음(다른 모델?)", DROP.len()));
    }
    ng.push_msg(5, &tensor("patch_w", &[3072, 768], &wt));
    ng.push_msg(5, &tensor("patch_b", &[768], &b));
    ng.push_msg(11, &value_info("patches", &[1, 64, 3072]));
    ng.push_msg(12, &value_info("pooler_output", &[1, 768]));
    model.0[gi].1 = Val::Bytes(ng.bytes());
    std::fs::write(dst, model.bytes()).map_err(|e| format!("{dst}: {e}"))?;
    println!("{dst}: 입력 patches [1,64,3072], 출력 pooler_output [1,768], 노드 {dropped} 개 없앰");
    Ok(())
}

fn cut(src: &str, dst: &str, outs: &[&str]) -> Result<(), String> {
    let mut model = Msg::parse(&std::fs::read(src).map_err(|e| format!("{src}: {e}"))?)?;
    let gi = model.0.iter().position(|(n, _)| *n == 7).ok_or("graph 없음")?;
    let Val::Bytes(gb) = &model.0[gi].1 else { return Err("graph 형식".into()) };
    let g = Msg::parse(gb)?;
    let nodes = g.msgs(1);
    // 뒤에서부터: 필요한 텐서를 내는 노드만 남김
    let mut need: std::collections::HashSet<String> = outs.iter().map(|s| s.to_string()).collect();
    let mut keep = vec![false; nodes.len()];
    for (i, n) in nodes.iter().enumerate().rev() {
        if n.strs(2).iter().any(|o| need.contains(o)) {
            keep[i] = true;
            need.extend(n.strs(1).into_iter().filter(|s| !s.is_empty()));
        }
    }
    for o in outs {
        if !nodes.iter().any(|n| n.strs(2).iter().any(|x| x == o)) {
            return Err(format!("출력 {o} 을 내는 노드 없음"));
        }
    }
    // 원래 그래프의 모양 정보에서 출력 모양을 찾음(없으면 이름·실수만)
    let infos: Vec<Msg> = g.msgs(13).into_iter().chain(g.msgs(12)).collect();
    let mut ng = Msg::default();
    let (mut ni, mut k) = (0usize, 0usize);
    for (no, v) in &g.0 {
        match no {
            1 => {
                if keep[ni] {
                    ng.0.push((1, v.clone()));
                    k += 1;
                }
                ni += 1;
            }
            5 => {
                let Val::Bytes(b) = v else { continue };
                if need.contains(&Msg::parse(b)?.str(8).unwrap_or_default()) {
                    ng.0.push((5, v.clone()));
                }
            }
            12 | 13 => {}
            _ => ng.0.push((*no, v.clone())),
        }
    }
    for o in outs {
        match infos.iter().find(|m| m.str(1).as_deref() == Some(*o)) {
            Some(m) => ng.push_msg(12, m),
            None => {
                // 실수 텐서, 모양은 비워 둠(컴파일러가 모양을 다시 구함)
                let mut tt = Msg::default();
                tt.push_int(1, 1);
                let mut ty = Msg::default();
                ty.push_msg(1, &tt);
                let mut vi = Msg::default();
                vi.push_str(1, o);
                vi.push_msg(2, &ty);
                ng.push_msg(12, &vi);
            }
        }
    }
    model.0[gi].1 = Val::Bytes(ng.bytes());
    std::fs::write(dst, model.bytes()).map_err(|e| format!("{dst}: {e}"))?;
    println!("{dst}: 노드 {k} / {} 개, 출력 {} 개", nodes.len(), outs.len());
    Ok(())
}

/// embed.rs preprocess_into(patches = true) 와 같은 식
fn patchify(rgb: &[u8], w: usize, h: usize, out: &mut Vec<f32>) {
    const S: usize = 256;
    let axis = |dst: usize, src: usize| -> (usize, usize, f32) {
        let f = ((dst as f32 + 0.5) * src as f32 / S as f32 - 0.5).max(0.0);
        let a = (f as usize).min(src - 1);
        (a, (a + 1).min(src - 1), f - a as f32)
    };
    let base = out.len();
    out.resize(base + 3 * S * S, 0.0);
    for y in 0..S {
        let (y0, y1, fy) = axis(y, h);
        for x in 0..S {
            let (x0, x1, fx) = axis(x, w);
            for c in 0..3 {
                let p = |yy: usize, xx: usize| rgb[(yy * w + xx) * 3 + c] as f32;
                let v = ((p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx) * (1.0 - fy) + (p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx) * fy) / 127.5 - 1.0;
                out[base + ((y / 32) * 8 + x / 32) * 3072 + c * 1024 + (y % 32) * 32 + x % 32] = v;
            }
        }
    }
}

fn patches_calib(dir: &str, dst: &str, n: usize) -> Result<(), String> {
    let mut files: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("{dir}: {e}"))?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg"))).collect();
    files.sort();
    files.truncate(n);
    let mut all = vec![];
    for f in &files {
        let im = image::open(f).map_err(|e| format!("{}: {e}", f.display()))?.to_rgb8();
        patchify(im.as_raw(), im.width() as usize, im.height() as usize, &mut all);
    }
    let k = files.len();
    // .npy v1: 머리글은 16 바이트 단위로 맞춤
    let mut hdr = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({k}, 64, 3072), }}");
    while (10 + hdr.len() + 1) % 16 != 0 {
        hdr.push(' ');
    }
    hdr.push('\n');
    let mut o = b"\x93NUMPY\x01\x00".to_vec();
    o.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    o.extend_from_slice(hdr.as_bytes());
    o.extend(all.iter().flat_map(|x| x.to_le_bytes()));
    std::fs::write(dst, o).map_err(|e| format!("{dst}: {e}"))?;
    println!("{dst}: 사진 {k} 장 → [{k}, 64, 3072]");
    Ok(())
}
