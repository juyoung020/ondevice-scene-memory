//! Text search over the objects of a memory folder.
//!
//! memmap (`--embed`) leaves one SigLIP2 image vector per object (`objects/O<id>_emb.bin`, 768 f32, unit length). A query is turned
//! into a vector by the SigLIP2 text encoder and compared with two things per object:
//!   name  — the object's class name (also a text vector): "의자" lands next to "chair", so stored English names answer Korean queries
//!   look  — the object's image vector: free descriptions ("something to drink from") match by appearance
//! The "vector DB" is those files read into one flat scan; a memory holds tens to hundreds of objects.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

pub const DIM: usize = 768;
/// SigLIP2 base patch32-256: logits = SCALE · cos (logit_scale = e^4.717 in the checkpoint)
const SCALE: f32 = 111.83;
/// Reading a query as a class name (constants read off 36 Korean / English queries against the 1,225 names of vocab/names.voc, docs/performance_log.md):
/// closest-name cosine minus the median cosine below which the query is no class name, and from which it surely is
const NAME_GAP_LO: f32 = 0.06;
const NAME_GAP_HI: f32 = 0.10;
/// cosine distance from the closest name over which a name's share drops by e
const NAME_FALLOFF: f32 = 0.035;
/// Texts that describe no object: a picture these fit as well as the query is no hit for the query
pub const NULLS: [&str; 5] = ["a photo", "an image", "a blurry photo", "nothing", "background"];
/// Token count the text encoder was trained with
const TOKENS: usize = 64;

/// a · b (unit vectors → cosine): [`crate::objtable::dot`] (NEON on the board, the plain loop elsewhere)
pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    crate::objtable::dot(a, b)
}

/// One object vector file → unit vector (None when missing or not DIM floats): the per-query reading the memory table replaced (test reference)
#[cfg(test)]
pub fn read_emb(p: &Path) -> Option<Vec<f32>> {
    let b = crate::memfs::read(p)?;
    if b.len() != DIM * 4 {
        return None;
    }
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(feature = "search")]
pub struct TextEnc {
    net: Net,
    tok: tokenizers::Tokenizer,
    cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// word vectors read from a file (`--vocab`): never dropped, never computed
    known: std::sync::OnceLock<HashMap<String, Arc<Vec<f32>>>>,
}

/// The text encoder runner: plain Rust (`textnet`, weights from `text_model_fp16.onnx`), or ONNX Runtime (`search-onnx` builds)
#[cfg(feature = "search")]
enum Net {
    Rust(crate::textnet::TextNet),
    #[cfg(feature = "search-onnx")]
    Onnx(Mutex<ort::session::Session>),
}

#[cfg(feature = "search")]
impl TextEnc {
    /// `dir` holds `tokenizer.json` and `text_model_fp16.onnx` (or `text_model.onnx` for ONNX Runtime).
    /// `onnx`: run it with ONNX Runtime instead of the Rust encoder (`search-onnx` builds)
    pub fn load(dir: &Path) -> Result<TextEnc, String> {
        // PC(search-onnx): ONNX Runtime 이 Rust 인코더보다 2.2 배 빠르다(docs/npu_models.md "글 인코더")
        Self::load_with(dir, cfg!(feature = "search-onnx"))
    }

    pub fn load_with(dir: &Path, onnx: bool) -> Result<TextEnc, String> {
        let net = Self::open_net(dir, onnx)?;
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: TOKENS, ..Default::default() })).map_err(|e| e.to_string())?;
        tok.with_padding(Some(tokenizers::PaddingParams { strategy: tokenizers::PaddingStrategy::Fixed(TOKENS), pad_id: 0, pad_token: "<pad>".into(), ..Default::default() }));
        Ok(TextEnc { net, tok, cache: Mutex::new(HashMap::new()), known: std::sync::OnceLock::new() })
    }

    fn open_net(dir: &Path, onnx: bool) -> Result<Net, String> {
        #[cfg(feature = "search-onnx")]
        if onnx {
            let model = ["text_model_fp16.onnx", "text_model.onnx"].iter().map(|n| dir.join(n)).find(|p| p.exists())
                .ok_or_else(|| format!("no text_model*.onnx in {}", dir.display()))?;
            let sess = ort::session::Session::builder(common::ort::env()).map_err(|e| e.to_string())?
                .with_intra_threads(4).map_err(|e| e.to_string())?
                .commit_from_file(&model).map_err(|e| format!("{}: {e}", model.display()))?;
            return Ok(Net::Onnx(Mutex::new(sess)));
        }
        let _ = onnx;
        Ok(Net::Rust(crate::textnet::TextNet::load(&dir.join("text_model_fp16.onnx"))?))
    }

    /// Hand over word vectors made ahead of time (`read_vocab`): those words are answered without running the model
    pub fn know(&self, vocab: Vec<(String, Arc<Vec<f32>>)>) {
        let _ = self.known.set(vocab.into_iter().map(|(n, v)| (n.trim().to_lowercase(), v)).collect());
    }

    /// Text → unit vector. SigLIP2 was trained on lower-cased text; results are kept (class names repeat on every query).
    pub fn encode(&self, text: &str) -> Result<Arc<Vec<f32>>, String> {
        let key = text.trim().to_lowercase();
        if let Some(v) = self.known.get().and_then(|k| k.get(&key)) {
            return Ok(v.clone());
        }
        if let Some(v) = self.cache.lock().unwrap().get(&key) {
            return Ok(v.clone());
        }
        let mut v = self.run(&key)?;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        let v = Arc::new(v);
        let mut c = self.cache.lock().unwrap();
        if c.len() > 4096 {
            c.clear();
        }
        c.insert(key, v.clone());
        Ok(v)
    }

    /// Text (already lower-cased) → encoder output, not normalised
    pub fn run(&self, text: &str) -> Result<Vec<f32>, String> {
        let enc = self.tok.encode(text, true).map_err(|e| e.to_string())?;
        let ids = enc.get_ids();
        match &self.net {
            Net::Rust(n) => n.run(ids),
            #[cfg(feature = "search-onnx")]
            Net::Onnx(sess) => {
                let ids: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
                let inp = ort::value::TensorRef::from_array_view(([1usize, ids.len()], &ids[..])).map_err(|e| e.to_string())?;
                let mut sess = sess.lock().unwrap();
                let out = sess.run(ort::inputs![inp]).map_err(|e| e.to_string())?;
                let (_, v) = out["pooler_output"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                Ok(v.to_vec())
            }
        }
    }
}

/// Rust encoder ↔ ONNX Runtime on the same texts: SGVIEW_TEXT_CHECK=<model dir> SGVIEW_TEXT_LIST=<texts, one a line>
#[cfg(all(test, feature = "search-onnx"))]
mod rust_vs_onnx {
    #[test]
    fn same_vectors() {
        let (Ok(dir), Ok(list)) = (std::env::var("SGVIEW_TEXT_CHECK"), std::env::var("SGVIEW_TEXT_LIST")) else { return };
        let dir = std::path::Path::new(&dir);
        let r = super::TextEnc::load_with(dir, false).unwrap();
        let o = super::TextEnc::load_with(dir, true).unwrap();
        let texts = std::fs::read_to_string(list).unwrap();
        let texts: Vec<String> = texts.lines().map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty()).collect();
        let (mut tr, mut to, mut worst, mut maxd) = (0f64, 0f64, 1f32, 0f32);
        for t in &texts {
            let t0 = std::time::Instant::now();
            let a = r.run(t).unwrap();
            tr += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let b = o.run(t).unwrap();
            to += t0.elapsed().as_secs_f64();
            let (na, nb) = (a.iter().map(|x| x * x).sum::<f32>().sqrt(), b.iter().map(|x| x * x).sum::<f32>().sqrt());
            let c = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>() / (na * nb);
            worst = worst.min(c);
            maxd = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(maxd, f32::max);
        }
        let n = texts.len() as f64;
        println!("texts {}: min cosine {worst:.7}, max |diff| {maxd:.2e}, rust {:.1} ms, onnx {:.1} ms per text", texts.len(), tr / n * 1e3, to / n * 1e3);
        assert!(worst > 0.9999);
    }
}

/// Word vectors made ahead of time (`sgview --encode-vocab`, read by `--vocab` and by memmap `--vocab`):
/// one text line `SGVOC 1 <n> <dim>`, n lines of names, then n · dim f32 (little endian, unit length)
pub fn read_vocab(p: &Path) -> Result<Vec<(String, Arc<Vec<f32>>)>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let bad = || format!("{}: not a word vector file", p.display());
    let mut pos = 0;
    let mut line = || -> Option<&str> {
        let end = pos + b[pos..].iter().position(|c| *c == b'\n')?;
        let s = std::str::from_utf8(&b[pos..end]).ok()?;
        pos = end + 1;
        Some(s)
    };
    let head: Vec<&str> = line().ok_or_else(bad)?.split(' ').collect();
    let (n, dim) = match head[..] {
        ["SGVOC", "1", n, d] => (n.parse::<usize>().map_err(|_| bad())?, d.parse::<usize>().map_err(|_| bad())?),
        _ => return Err(bad()),
    };
    let names: Vec<String> = (0..n).map(|_| line().map(str::to_string)).collect::<Option<_>>().ok_or_else(bad)?;
    if dim != DIM || b.len() != pos + n * dim * 4 {
        return Err(bad());
    }
    let f: Vec<f32> = b[pos..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(names.into_iter().zip(f.chunks(dim)).map(|(n, v)| (n, Arc::new(v.to_vec()))).collect())
}

/// `--encode-vocab <names.txt> <out>`: one name a line → word vector file (done once on a PC; the file is then used as it is).
/// A line may add descriptions after a colon, `monitor: a black computer screen; a turned-off monitor`. Each name's vector is the
/// mean of the name in several sentences ("a photo of a {name}.", "a blurry photo of a {name}." …), plus the mean of its descriptions
/// ("a photo of {description}.") when given — sentences match photos better than a bare word, and dark monitors stop looking like "webcam"
#[cfg(feature = "search")]
pub fn encode_vocab(model: &Path, list: &Path, out: &Path) -> Result<usize, String> {
    const TEMPLATES: [&str; 6] = ["a photo of a {}.", "a photo of the {}.", "a blurry photo of a {}.", "a dark photo of a {}.", "a close-up photo of a {}.", "a photo of a {} in an office."];
    let enc = TextEnc::load(model)?;
    let text = std::fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
    let lines: Vec<(&str, Vec<&str>)> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(|l| match l.split_once(':') {
        Some((n, d)) => (n.trim(), d.split(';').map(str::trim).filter(|x| !x.is_empty()).collect()),
        None => (l, vec![]),
    }).collect();
    let unit = |mut v: Vec<f32>| {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    };
    let mean = |texts: &[String]| -> Result<Vec<f32>, String> {
        let mut s = vec![0f32; DIM];
        for t in texts {
            s.iter_mut().zip(enc.encode(t)?.iter()).for_each(|(a, b)| *a += b);
        }
        Ok(unit(s))
    };
    let mut b = format!("SGVOC 1 {} {DIM}\n", lines.len()).into_bytes();
    for (n, _) in &lines {
        b.extend(n.as_bytes());
        b.push(b'\n');
    }
    for (n, d) in &lines {
        let mut v = mean(&TEMPLATES.iter().map(|t| t.replace("{}", n)).collect::<Vec<_>>())?;
        if !d.is_empty() {
            let w = mean(&d.iter().map(|x| format!("a photo of {x}.")).collect::<Vec<_>>())?;
            v = unit(v.iter().zip(&w).map(|(a, b)| a + b).collect());
        }
        b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    }
    std::fs::write(out, b).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(lines.len())
}

/// Text → unit vector, or None when no encoder is loaded (then only the letters of the name are compared)
pub type Encode<'a> = &'a dyn Fn(&str) -> Option<Arc<Vec<f32>>>;

/// The objects' picture vectors as one table in memory (by vector file), with SCALE · cos(name, picture) kept for every
/// vocabulary name then every NULLS text: a query reads no vector file and computes no name × picture product
static TABLE: Mutex<Option<crate::objtable::ObjTable<std::path::PathBuf>>> = Mutex::new(None);

/// `--search-direct <logit_bias>`: search in one step, query vector × the object table (see [`search_direct`])
static DIRECT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
pub fn set_direct(logit_bias: f32) {
    let _ = DIRECT.set(logit_bias);
}

/// One-step search: p = sigmoid(SCALE · cos(query, picture) + logit_bias) for every object of the table (no vocabulary, no NULLS),
/// an object whose name holds the query letters (or the query holds the name) scores 1. Same reply shape as [`search`]
/// (`names` empty, `p_name` 1 for a letter match else 0)
pub fn search_direct(root: &Path, view: &serde_json::Value, q: &str, enc: Encode, k: usize, min: f32, bias: f32) -> serde_json::Value {
    let ql = q.trim().to_lowercase();
    let objs = view["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let tq = if ql.is_empty() { None } else { enc(&ql) };
    let mut table = TABLE.lock().unwrap();
    let table = table.get_or_insert_with(|| crate::objtable::ObjTable::new(DIM, SCALE));
    let paths: Vec<Option<std::path::PathBuf>> = objs.iter().map(|o| o["emb"].as_str().map(|p| root.join(p))).collect();
    if tq.is_some() {
        let items: Vec<(std::path::PathBuf, Arc<Vec<u8>>)> = paths.iter().flatten().filter_map(|p| crate::memfs::read(p).map(|b| (p.clone(), b))).collect();
        table.sync_bytes(&items);
    }
    let cosines = tq.as_ref().map(|t| table.scores(t)).unwrap_or_default();
    let mut hits: Vec<(f32, serde_json::Value)> = vec![];
    for (o, p) in objs.iter().zip(&paths) {
        if ql.is_empty() {
            break;
        }
        let nl = o["name"].as_str().unwrap_or("").to_lowercase();
        let exact = !nl.is_empty() && (nl.contains(&ql) || ql.contains(nl.as_str()));
        let p_look = match p.as_ref().and_then(|p| table.row(p)) {
            Some(r) if !cosines.is_empty() => 1.0 / (1.0 + (-(SCALE * cosines[r] + bias)).exp()),
            _ => 0.0,
        };
        let score = if exact { 1.0 } else { p_look };
        if score >= min {
            let r3 = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
            hits.push((score, serde_json::json!({"id": o["id"], "name": o["name"], "state": o["state"], "pos": o["pos"],
                "score": r3(score), "by": if exact { "name" } else { "look" }, "p_name": if exact { 1.0 } else { 0.0 }, "p_look": r3(p_look)})));
        }
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1["p_look"].as_f64().unwrap_or(0.0).total_cmp(&a.1["p_look"].as_f64().unwrap_or(0.0))));
    hits.truncate(k);
    serde_json::json!({"q": q, "names": [], "semantic": tq.is_some(), "results": hits.into_iter().map(|h| h.1).collect::<Vec<_>>()})
}

/// log(Σ e^x)
fn lse(xs: impl Iterator<Item = f32>) -> f32 {
    let v: Vec<f32> = xs.collect();
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY { m } else { m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() }
}

/// Class names the query competes against: the detector's vocabulary memmap leaves in `vocab.json`, plus every name in the memory
fn vocabulary(root: &Path, present: &[String]) -> Vec<String> {
    let mut v: Vec<String> = crate::memfs::read(&root.join("vocab.json"))
        .and_then(|t| serde_json::from_slice::<Vec<String>>(&t).ok()).unwrap_or_default()
        .into_iter().map(|s| s.to_lowercase()).collect();
    v.extend(present.iter().cloned());
    v.sort();
    v.dedup();
    v
}

/// Rank the objects of `view` (view.json) against `q`; vectors are read from `root`. Returns the reply of /api/search:
/// `{"q", "semantic", "names": [{"name", "p"}] (class names the query was read as, p ≥ 0.2, at most 3), "results": [{"id", "name", "state", "pos", "score", "by", "p_name", "p_look"}]}`, best first, score ≥ `min`, at most `k` (equal scores: better look first).
///
///   p_name  the query means this object's class name. Text vectors of this model all sit close together (cos ≈ 0.8 between
///           unrelated words), so the cosines to the vocabulary are read relative to each other:
///             conf   = how far the closest name stands out from the median name, 0 at NAME_GAP_LO .. 1 at NAME_GAP_HI
///                      ("의자" → chair stands out by 0.15, "화면" → nothing stands out, 0.07)
///             p_name = conf · e^((cos − closest cos) / NAME_FALLOFF): the closest name gets conf, near-synonyms
///                      ("가방": handbag, suitcase, backpack) stay in, a class that is not in the memory ("고양이" → cat) leaves
///                      every stored name near 0
///           1 when the letters match (query inside the name or the name inside the query)
///   p_look  the query describes this object's picture better than any class name does:
///           softmax over {query} ∪ vocabulary ∪ NULLS of SCALE · cos(text, picture); names that mean what the query means
///           (NAME_GAP_LO above the median name and within e^−1.6 of the closest name) are left out
///   score   1 − (1 − p_name)(1 − p_look);  by = "name" or "look", whichever is larger
pub fn search(root: &Path, view: &serde_json::Value, q: &str, enc: Encode, k: usize, min: f32) -> serde_json::Value {
    if let Some(&bias) = DIRECT.get() {
        return search_direct(root, view, q, enc, k, min, bias);
    }
    let ql = q.trim().to_lowercase();
    let objs = view["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let names: Vec<String> = objs.iter().map(|o| o["name"].as_str().unwrap_or("").to_lowercase()).collect();
    let tq = if ql.is_empty() { None } else { enc(&ql) };
    // class-name vectors, and how well the query matches each name
    let vocab: Vec<(String, Arc<Vec<f32>>)> = match &tq {
        Some(_) => vocabulary(root, &names).into_iter().filter(|n| !n.is_empty()).filter_map(|n| enc(&n).map(|v| (n, v))).collect(),
        None => vec![],
    };
    let nulls: Vec<Arc<Vec<f32>>> = if tq.is_some() { NULLS.iter().filter_map(|n| enc(n)).collect() } else { vec![] };
    let (p_names, alike): (HashMap<&str, f32>, std::collections::HashSet<&str>) = match &tq {
        Some(t) => {
            let c: Vec<f32> = vocab.iter().map(|(_, v)| cos(t, v)).collect();
            let mut sorted = c.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let (best, median) = (sorted.first().copied().unwrap_or(0.0), sorted.get(sorted.len() / 2).copied().unwrap_or(0.0));
            let conf = ((best - median - NAME_GAP_LO) / (NAME_GAP_HI - NAME_GAP_LO)).clamp(0.0, 1.0);
            let near: HashMap<&str, f32> = vocab.iter().zip(&c).map(|((n, _), x)| (n.as_str(), ((x - best) / NAME_FALLOFF).exp())).collect();
            let alike: std::collections::HashSet<&str> = vocab.iter().zip(&c).filter(|(_, x)| *x - median >= NAME_GAP_LO && ((*x - best) / NAME_FALLOFF).exp() >= 0.2).map(|((n, _), _)| n.as_str()).collect();
            (near.into_iter().map(|(n, r)| (n, conf * r)).collect(), alike)
        }
        None => (HashMap::new(), Default::default()),
    };
    // names that stand out for the query and sit close to its closest name say the same thing as the query: they do not compete with it for a picture
    let same = |n: &str| n == ql || alike.contains(n);
    // picture logits against every vocabulary name and NULLS, kept per object until its vector or the vocabulary changes;
    // a query then needs one dot product per object (query · picture) and the log-sum-exp over the kept logits
    // the memory table: vocabulary words then NULLS, picture vectors synced to the vector files (only new or changed ones computed)
    let mut table = TABLE.lock().unwrap();
    let table = table.get_or_insert_with(|| crate::objtable::ObjTable::new(DIM, SCALE));
    let paths: Vec<Option<std::path::PathBuf>> = objs.iter().map(|o| o["emb"].as_str().map(|p| root.join(p))).collect();
    if tq.is_some() && !ql.is_empty() {
        table.set_words(vocab.iter().map(|(_, v)| v.clone()).chain(nulls.iter().cloned()).collect());
        let items: Vec<(std::path::PathBuf, Arc<Vec<u8>>)> = paths.iter().flatten().filter_map(|p| crate::memfs::read(p).map(|b| (p.clone(), b))).collect();
        table.sync_bytes(&items);
    }
    let rows: Vec<Option<usize>> = paths.iter().map(|p| p.as_ref().and_then(|p| table.row(p))).collect();
    let keep: Vec<usize> = (0..vocab.len()).filter(|&i| !same(&vocab[i].0)).collect();
    // p_look per object: one dot product (query · picture) and the log-sum-exp over the kept logits, objects in parallel
    let table = &*table;
    let look = |row: &Option<usize>| -> f32 {
        match (&tq, row) {
            (Some(t), Some(r)) => {
                let lq = SCALE * cos(t, table.vector(*r));
                let (logit, nv) = (table.logits(*r), vocab.len());
                let z = lse(std::iter::once(lq).chain(keep.iter().map(|&i| logit[i])).chain(logit[nv..].iter().copied()));
                (lq - z).exp()
            }
            _ => 0.0,
        }
    };
    #[cfg(feature = "search")]
    let looks: Vec<f32> = { use rayon::prelude::*; rows.par_iter().with_min_len(16).map(look).collect() };
    #[cfg(not(feature = "search"))]
    let looks: Vec<f32> = rows.iter().map(look).collect();
    let mut hits: Vec<(f32, serde_json::Value)> = vec![];
    for ((o, nl), p_look) in objs.iter().zip(&names).zip(looks) {
        if ql.is_empty() {
            break;
        }
        let exact = !nl.is_empty() && (nl.contains(&ql) || ql.contains(nl.as_str()));
        let p_name = if exact { 1.0 } else { p_names.get(nl.as_str()).copied().unwrap_or(0.0) };
        let score = 1.0 - (1.0 - p_name) * (1.0 - p_look);
        if score >= min {
            let r3 = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
            hits.push((score, serde_json::json!({"id": o["id"], "name": o["name"], "state": o["state"], "pos": o["pos"],
                "score": r3(score), "by": if p_name >= p_look { "name" } else { "look" }, "p_name": r3(p_name), "p_look": r3(p_look)})));
        }
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1["p_look"].as_f64().unwrap_or(0.0).total_cmp(&a.1["p_look"].as_f64().unwrap_or(0.0))));
    hits.truncate(k);
    // class names the query was read as, best first
    let mut read: Vec<(&str, f32)> = p_names.iter().filter(|(_, p)| **p >= 0.2).map(|(n, p)| (*n, *p)).collect();
    read.sort_by(|a, b| b.1.total_cmp(&a.1));
    read.truncate(3);
    serde_json::json!({"q": q, "names": read.iter().map(|(n, p)| serde_json::json!({"name": n, "p": (*p as f64 * 100.0).round() / 100.0})).collect::<Vec<_>>(), "semantic": tq.is_some(), "results": hits.into_iter().map(|h| h.1).collect::<Vec<_>>()})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// unit vector along axis `i`, tilted a little towards axis `j`
    fn axis(i: usize, j: usize, tilt: f32) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[i] = 1.0;
        v[j] += tilt;
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// memory folder with a chair (0), a cup (1) and a tv (2); pictures point along the axis of their class
    fn memory(tag: &str) -> (std::path::PathBuf, serde_json::Value) {
        let root = std::env::temp_dir().join(format!("sgview_search_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(root.join("objects")).unwrap();
        std::fs::write(root.join("vocab.json"), r#"["chair","cup","tv","cat"]"#).unwrap();
        let mut objs = vec![];
        for (id, (name, ax)) in [("chair", 0), ("cup", 1), ("tv", 2)].into_iter().enumerate() {
            let rel = format!("objects/O{id}_emb.bin");
            let bytes: Vec<u8> = axis(ax, 9, 0.0).iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(root.join(&rel), bytes).unwrap();
            objs.push(serde_json::json!({"id": id, "name": name, "state": "present", "pos": [0, 0, 0], "emb": rel}));
        }
        (root, serde_json::json!({"objects": objs}))
    }

    /// stand-in text encoder: a class name and its Korean word share an axis; "마실 것" is no class name but points at the cup's picture.
    /// Text and picture cosines are small in the real model, so texts sit mostly on their own axes (30+) with a small picture part.
    fn enc(t: &str) -> Option<Arc<Vec<f32>>> {
        Some(Arc::new(match t {
            "chair" => axis(30, 0, 0.12),
            "의자" => axis(30, 0, 0.11),
            "cup" => axis(31, 1, 0.12),
            "tv" => axis(32, 2, 0.12),
            "cat" => axis(33, 3, 0.12),
            "고양이" => axis(33, 3, 0.11),
            "마실 것" => axis(40, 1, 0.16),
            _ => axis(50, 9, 0.0),
        }))
    }

    fn ids(r: &serde_json::Value) -> Vec<(u64, String)> {
        r["results"].as_array().unwrap().iter().map(|h| (h["id"].as_u64().unwrap(), h["by"].as_str().unwrap().to_string())).collect()
    }

    #[test]
    fn word_vector_file_round_trip() {
        let p = std::env::temp_dir().join(format!("sgview_voc_{}", std::process::id()));
        let mut b = format!("SGVOC 1 2 {DIM}\nchair\ncomputer monitor\n").into_bytes();
        b.extend(axis(0, 1, 0.0).iter().chain(axis(2, 3, 0.5).iter()).flat_map(|x| x.to_le_bytes()));
        std::fs::write(&p, &b).unwrap();
        let v = read_vocab(&p).unwrap();
        assert_eq!(v.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), ["chair", "computer monitor"]);
        assert_eq!(*v[1].1, axis(2, 3, 0.5));
        // a vector short
        std::fs::write(&p, &b[..b.len() - 4]).unwrap();
        assert!(read_vocab(&p).is_err());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn letters_only_without_encoder() {
        let (root, view) = memory("letters");
        let none = |_: &str| None;
        let r = search(&root, &view, "Chair", &none, 10, 0.1);
        assert_eq!(ids(&r), vec![(0, "name".to_string())]);
        assert_eq!(r["semantic"], false);
        assert!(search(&root, &view, "의자", &none, 10, 0.1)["results"].as_array().unwrap().is_empty());
        assert!(search(&root, &view, "  ", &none, 10, 0.1)["results"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn other_language_finds_the_stored_name() {
        let (root, view) = memory("korean");
        let r = search(&root, &view, "의자", &enc, 10, 0.1);
        assert_eq!(ids(&r), vec![(0, "name".to_string())], "{r}");
        assert!(r["results"][0]["score"].as_f64().unwrap() > 0.9);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn absent_class_matches_nothing() {
        // "고양이" means "cat": the vocabulary holds it, the memory does not
        let (root, view) = memory("absent");
        let r = search(&root, &view, "고양이", &enc, 10, 0.1);
        assert!(r["results"].as_array().unwrap().is_empty(), "{r}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn description_matches_by_look() {
        let (root, view) = memory("look");
        let r = search(&root, &view, "마실 것", &enc, 10, 0.1);
        assert_eq!(ids(&r), vec![(1, "look".to_string())], "{r}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn direct_one_step() {
        let (root, view) = memory("direct");
        // letters: the chair scores 1 by name
        let r = search_direct(&root, &view, "chair", &enc, 5, 0.5, -10.0);
        assert_eq!(ids(&r)[0], (0, "name".to_string()));
        // a description: the picture closest to the query vector comes first, by look
        let r = search_direct(&root, &view, "마실 것", &enc, 5, 0.0, -10.0);
        assert_eq!(ids(&r)[0], (1, "look".to_string()));
        assert_eq!(r["names"], serde_json::json!([]));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_vector_file_is_skipped() {
        let (root, view) = memory("missing");
        std::fs::remove_file(root.join("objects/O0_emb.bin")).unwrap();
        let r = search(&root, &view, "의자", &enc, 10, 0.1);
        assert_eq!(ids(&r), vec![(0, "name".to_string())]);
        assert_eq!(r["results"][0]["p_look"], 0.0);
        let _ = std::fs::remove_dir_all(root);
    }
}


/// The viewer's encoder (first search): model dir + word vectors made ahead of time (`--vocab`) for the class names
#[cfg(feature = "search")]
pub fn load_encoder(dir: &str, vocab: Option<&str>) -> Option<TextEnc> {
    match TextEnc::load(Path::new(dir)) {
        Ok(e) => {
            match vocab.map(|f| read_vocab(Path::new(f))) {
                Some(Ok(v)) => e.know(v),
                Some(Err(err)) => eprintln!("search: {err}"),
                None => {}
            }
            eprintln!("search: text encoder ready ({dir})");
            Some(e)
        }
        Err(e) => {
            eprintln!("search: {e} — searching by name only");
            None
        }
    }
}

/// Same ranking and scores as the per-query computation it replaced (kept here as the reference), and query time by object count
#[cfg(test)]
mod pre_tests {
    use super::*;
    fn search_ref(root: &Path, view: &serde_json::Value, q: &str, enc: Encode, k: usize, min: f32) -> serde_json::Value {
        let ql = q.trim().to_lowercase();
        let objs = view["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        let names: Vec<String> = objs.iter().map(|o| o["name"].as_str().unwrap_or("").to_lowercase()).collect();
        let tq = if ql.is_empty() { None } else { enc(&ql) };
        // class-name vectors, and how well the query matches each name
        let vocab: Vec<(String, Arc<Vec<f32>>)> = match &tq {
            Some(_) => vocabulary(root, &names).into_iter().filter(|n| !n.is_empty()).filter_map(|n| enc(&n).map(|v| (n, v))).collect(),
            None => vec![],
        };
        let nulls: Vec<Arc<Vec<f32>>> = if tq.is_some() { NULLS.iter().filter_map(|n| enc(n)).collect() } else { vec![] };
        let (p_names, alike): (HashMap<&str, f32>, std::collections::HashSet<&str>) = match &tq {
            Some(t) => {
                let c: Vec<f32> = vocab.iter().map(|(_, v)| cos(t, v)).collect();
                let mut sorted = c.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let (best, median) = (sorted.first().copied().unwrap_or(0.0), sorted.get(sorted.len() / 2).copied().unwrap_or(0.0));
                let conf = ((best - median - NAME_GAP_LO) / (NAME_GAP_HI - NAME_GAP_LO)).clamp(0.0, 1.0);
                let near: HashMap<&str, f32> = vocab.iter().zip(&c).map(|((n, _), x)| (n.as_str(), ((x - best) / NAME_FALLOFF).exp())).collect();
                let alike: std::collections::HashSet<&str> = vocab.iter().zip(&c).filter(|(_, x)| *x - median >= NAME_GAP_LO && ((*x - best) / NAME_FALLOFF).exp() >= 0.2).map(|((n, _), _)| n.as_str()).collect();
                (near.into_iter().map(|(n, r)| (n, conf * r)).collect(), alike)
            }
            None => (HashMap::new(), Default::default()),
        };
        // names that stand out for the query and sit close to its closest name say the same thing as the query: they do not compete with it for a picture
        let same = |n: &str| n == ql || alike.contains(n);
        let mut hits: Vec<(f32, serde_json::Value)> = vec![];
        for (o, nl) in objs.iter().zip(&names) {
            if ql.is_empty() {
                break;
            }
            let exact = !nl.is_empty() && (nl.contains(&ql) || ql.contains(nl.as_str()));
            let p_name = if exact { 1.0 } else { p_names.get(nl.as_str()).copied().unwrap_or(0.0) };
            let p_look = match (&tq, o["emb"].as_str().and_then(|p| read_emb(&root.join(p)))) {
                (Some(t), Some(e)) => {
                    let lq = SCALE * cos(t, &e);
                    let z = lse(std::iter::once(lq).chain(vocab.iter().filter(|(n, _)| !same(n)).map(|(_, v)| SCALE * cos(v, &e))).chain(nulls.iter().map(|v| SCALE * cos(v, &e))));
                    (lq - z).exp()
                }
                _ => 0.0,
            };
            let score = 1.0 - (1.0 - p_name) * (1.0 - p_look);
            if score >= min {
                let r3 = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
                hits.push((score, serde_json::json!({"id": o["id"], "name": o["name"], "state": o["state"], "pos": o["pos"],
                    "score": r3(score), "by": if p_name >= p_look { "name" } else { "look" }, "p_name": r3(p_name), "p_look": r3(p_look)})));
            }
        }
        hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1["p_look"].as_f64().unwrap_or(0.0).total_cmp(&a.1["p_look"].as_f64().unwrap_or(0.0))));
        hits.truncate(k);
        // class names the query was read as, best first
        let mut read: Vec<(&str, f32)> = p_names.iter().filter(|(_, p)| **p >= 0.2).map(|(n, p)| (*n, *p)).collect();
        read.sort_by(|a, b| b.1.total_cmp(&a.1));
        read.truncate(3);
        serde_json::json!({"q": q, "names": read.iter().map(|(n, p)| serde_json::json!({"name": n, "p": (*p as f64 * 100.0).round() / 100.0})).collect::<Vec<_>>(), "semantic": tq.is_some(), "results": hits.into_iter().map(|h| h.1).collect::<Vec<_>>()})
    }

    fn rnd_unit(s: &mut u64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..DIM).map(|_| { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 40) as f32) / (1u64 << 24) as f32 - 0.5 }).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    /// memory with `n` objects and a vocabulary of `nv` names: (root, view, encoder table)
    fn world(tag: &str, n: usize, nv: usize) -> (std::path::PathBuf, serde_json::Value, HashMap<String, Arc<Vec<f32>>>) {
        let root = std::env::temp_dir().join(format!("sgview_pre_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(root.join("objects")).unwrap();
        let mut s = 11u64;
        let mut table: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
        let names: Vec<String> = (0..nv).map(|i| format!("name{i}")).collect();
        for nm in &names {
            table.insert(nm.clone(), Arc::new(rnd_unit(&mut s)));
        }
        for t in NULLS.iter().chain(["chair-ish", "something"].iter()) {
            table.insert(t.to_string(), Arc::new(rnd_unit(&mut s)));
        }
        std::fs::write(root.join("vocab.json"), serde_json::to_vec(&names).unwrap()).unwrap();
        crate::memfs::publish(&root.join("vocab.json"), Arc::new(serde_json::to_vec(&names).unwrap()));
        let objs: Vec<serde_json::Value> = (0..n).map(|id| {
            let rel = format!("objects/O{id}_emb.bin");
            let bytes: Vec<u8> = rnd_unit(&mut s).iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(root.join(&rel), &bytes).unwrap();
            crate::memfs::publish(&root.join(&rel), Arc::new(bytes)); // --serve: vector files live in memory
            serde_json::json!({"id": id, "name": names[id % nv], "state": "present", "pos": [0, 0, 0], "emb": rel})
        }).collect();
        (root, serde_json::json!({"objects": objs}), table)
    }

    #[test]
    fn same_as_per_query_scan() {
        let (root, view, table) = world("same", 60, 200);
        let enc = |t: &str| table.get(t).cloned().or_else(|| Some(Arc::new(vec![0.03; DIM])));
        for q in ["name3", "chair-ish", "something", "name17 thing"] {
            for _ in 0..2 {
                // twice: the second query reads the memory table
                assert_eq!(search(&root, &view, q, &enc, 50, 0.0), search_ref(&root, &view, q, &enc, 50, 0.0), "{q}");
            }
        }
        // a changed vector file is computed again
        let bytes: Vec<u8> = vec![0.05f32; DIM].iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(root.join("objects/O5_emb.bin"), bytes).unwrap();
        assert_eq!(search(&root, &view, "something", &enc, 50, 0.0), search_ref(&root, &view, "something", &enc, 50, 0.0));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `cargo test -p sgview --release -- --ignored --nocapture query_time`: ms per query, objects 10 · 100 · 1000, vocabulary 1,225
    #[test]
    #[ignore]
    fn query_time() {
        crate::memfs::enable(); // the board runs the viewer in the same process (--serve): files from memory
        for n in [10usize, 100, 1000] {
            let (root, view, table) = world(&format!("t{n}"), n, 1225);
            let enc = |t: &str| table.get(t).cloned().or_else(|| Some(Arc::new(vec![0.03; DIM])));
            let time = |f: &dyn Fn() -> serde_json::Value| {
                let c = std::time::Instant::now();
                for _ in 0..5 {
                    std::hint::black_box(f());
                }
                c.elapsed().as_secs_f64() * 1e3 / 5.0
            };
            let old = time(&|| search_ref(&root, &view, "something", &enc, 20, 0.0));
            let first = { let c = std::time::Instant::now(); search(&root, &view, "chair-ish", &enc, 20, 0.0); c.elapsed().as_secs_f64() * 1e3 };
            let new = time(&|| search(&root, &view, "something", &enc, 20, 0.0));
            println!("objects {n}: per-query scan {old:.2} ms, memory table {new:.2} ms (first query after all vectors changed {first:.2} ms)");
            // no vocabulary: query vector × object table → sigmoid(scale · cos + bias) → threshold · top-k
            let mut t: crate::objtable::ObjTable<u32> = crate::objtable::ObjTable::new(DIM, SCALE);
            let mut s = 77u64;
            for id in 0..n as u32 {
                t.upsert(id, &rnd_unit(&mut s));
            }
            let q = rnd_unit(&mut s);
            let direct = time(&|| {
                let p: Vec<f32> = t.scores(&q).iter().map(|c| 1.0 / (1.0 + (-(SCALE * c - 16.77)).exp())).collect();
                let mut o: Vec<usize> = (0..p.len()).filter(|&i| p[i] >= 0.0).collect();
                o.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
                o.truncate(20);
                serde_json::json!(o)
            });
            println!("objects {n}: no vocabulary (scores + sigmoid + top-k) {direct:.3} ms");
            let _ = std::fs::remove_dir_all(&root);
        }
    }
}
