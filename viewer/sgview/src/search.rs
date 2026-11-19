//! Text search over the objects of a memory folder, in one step (docs/search_direct.md).
//!
//! memmap (`--embed`) leaves one SigLIP2 image vector per object (`objects/O<id>_emb.bin`, 768 f32, unit length), kept in memory as one
//! table ([`crate::objtable`]). A query is turned into a vector by the SigLIP2 text encoder and every object gets
//! p = sigmoid(SCALE · cos(query, picture) + LOGIT_BIAS), the pair probability the model was trained to give; objects with p ≥ the
//! threshold come back. The query is also read as class names (query text × name text against the `--vocab` word vectors, made ahead of
//! time): an object whose stored name the query means ("마우스" → mouse) is a hit even when its picture alone scores low.
//! score = 1 − (1 − p_name)(1 − p_look). An object whose stored name holds the query letters (or the query holds the name) scores 1.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

pub const DIM: usize = 768;
/// SigLIP2 base patch32-256: logits = SCALE · cos + LOGIT_BIAS (logit_scale = e^4.717 and logit_bias in the checkpoint)
const SCALE: f32 = 111.83;
pub const LOGIT_BIAS: f32 = -16.766876;
/// Default threshold on p (cosine 0.1006); a query with Hangul gets DIRECT_MIN_KO (cosine 0.0964): Korean queries score lower (docs/search_direct.md)
pub const DIRECT_MIN: f32 = 0.004;
pub const DIRECT_MIN_KO: f32 = 0.0025;
/// Token count the text encoder was trained with
const TOKENS: usize = 64;
/// Reading the query as a class name (text × text, no picture): constants read off 36 Korean / English queries against the 1,225 names of
/// vocab/names.voc (docs/performance_log.md). Closest-name cosine minus the median cosine below which the query is no class name, and from which it surely is
const NAME_GAP_LO: f32 = 0.06;
const NAME_GAP_HI: f32 = 0.10;
/// cosine distance from the closest name over which a name's share drops by e
const NAME_FALLOFF: f32 = 0.035;
/// A stored name the query was read as with at least this share is a hit ("마우스" → mouse, "티비" → tv · monitor)
pub const NAME_MIN_P: f32 = 0.2;
/// Word vectors the query is read against (`--vocab`, made ahead of time): set when the encoder loads
static READ_VOCAB: std::sync::OnceLock<Vec<(String, Arc<Vec<f32>>)>> = std::sync::OnceLock::new();

/// How strongly the query means each name: conf · e^((cos − best) / NAME_FALLOFF), conf = how far the closest name stands out from the median
/// name (text vectors of this model all sit close together, so cosines are read relative to each other). Names: `vocab` ∪ `present`
fn read_names(tq: &[f32], vocab: &[(String, Arc<Vec<f32>>)], present: &[(String, Arc<Vec<f32>>)]) -> HashMap<String, f32> {
    let mut all: Vec<(&str, f32)> = vocab.iter().chain(present.iter()).map(|(n, v)| (n.as_str(), cos(tq, v))).collect();
    if all.is_empty() {
        return HashMap::new();
    }
    let mut sorted: Vec<f32> = all.iter().map(|x| x.1).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (best, median) = (sorted[0], sorted[sorted.len() / 2]);
    let conf = ((best - median - NAME_GAP_LO) / (NAME_GAP_HI - NAME_GAP_LO)).clamp(0.0, 1.0);
    all.dedup_by(|a, b| a.0 == b.0);
    all.into_iter().map(|(n, c)| (n.to_string(), conf * ((c - best) / NAME_FALLOFF).exp())).collect()
}

/// a · b (unit vectors → cosine): [`crate::objtable::dot`] (NEON on the board, the plain loop elsewhere)
pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    crate::objtable::dot(a, b)
}

/// p = sigmoid(SCALE · cos + LOGIT_BIAS)
pub fn prob(c: f32) -> f32 {
    1.0 / (1.0 + (-(SCALE * c + LOGIT_BIAS)).exp())
}

/// The threshold for this query: DIRECT_MIN, DIRECT_MIN_KO when it holds Hangul
pub fn default_min(q: &str) -> f32 {
    if q.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c) || ('\u{3131}'..='\u{318E}').contains(&c)) { DIRECT_MIN_KO } else { DIRECT_MIN }
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

/// The objects' picture vectors as one table in memory (by vector file): a query reads no vector file
static TABLE: Mutex<Option<crate::objtable::ObjTable<std::path::PathBuf>>> = Mutex::new(None);

/// Rank the objects of `view` (view.json) against `q`; vectors are read from `root` (memory files under `--serve`). Returns the reply of /api/search:
/// `{"q", "semantic", "min", "results": [{"id", "name", "state", "pos", "score", "cos", "by"}]}`: objects with p ≥ `min` (None: [`default_min`]),
/// letter matches first (score 1, by "name"), then by cosine (score = p, by "look"), at most `k`.
pub fn search(root: &Path, view: &serde_json::Value, q: &str, enc: Encode, k: usize, min: Option<f32>) -> serde_json::Value {
    let ql = q.trim().to_lowercase();
    let min = min.unwrap_or_else(|| default_min(&ql));
    let objs = view["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    // text encodings first, with no lock held: the encoder runs rayon, and a lock held across it let a pool worker that stole another
    // search block on that same lock (all three workers deadlocked on the board, 26-11-19)
    let tq = if ql.is_empty() { None } else { enc(&ql) };
    // the query read as class names (text × text): "마우스" → mouse even when the picture alone is too weak (Korean queries score low on pictures)
    let p_names: HashMap<String, f32> = match &tq {
        Some(t) => {
            let mut present: Vec<String> = objs.iter().map(|o| o["name"].as_str().unwrap_or("").to_lowercase()).filter(|n| !n.is_empty()).collect();
            present.sort();
            present.dedup();
            let pv: Vec<(String, Arc<Vec<f32>>)> = present.into_iter().filter_map(|n| enc(&n).map(|v| (n, v))).collect();
            read_names(t, READ_VOCAB.get().map(|v| v.as_slice()).unwrap_or(&[]), &pv)
        }
        None => HashMap::new(),
    };
    let paths: Vec<Option<std::path::PathBuf>> = objs.iter().map(|o| o["emb"].as_str().map(|p| root.join(p))).collect();
    // picture cosine per object: the table lock covers only the sync and one dot product per object (no rayon under the lock)
    let cos_of: Vec<Option<f32>> = match &tq {
        Some(t) => {
            let items: Vec<(std::path::PathBuf, Arc<Vec<u8>>)> = paths.iter().flatten().filter_map(|p| crate::memfs::read(p).map(|b| (p.clone(), b))).collect();
            let mut table = TABLE.lock().unwrap();
            let table = table.get_or_insert_with(|| crate::objtable::ObjTable::new(DIM));
            table.sync_bytes(&items);
            paths.iter().map(|p| p.as_ref().and_then(|p| table.row(p)).map(|r| crate::objtable::dot(t, table.vector(r)))).collect()
        }
        None => vec![None; paths.len()],
    };
    // (letter match, score, cosine, reply row)
    let mut hits: Vec<(bool, f32, f32, serde_json::Value)> = vec![];
    for (o, c) in objs.iter().zip(&cos_of) {
        if ql.is_empty() {
            break;
        }
        let nl = o["name"].as_str().unwrap_or("").to_lowercase();
        let exact = !nl.is_empty() && (nl.contains(&ql) || ql.contains(nl.as_str()));
        let c = *c;
        let pl = c.map_or(0.0, prob);
        let pn = if exact { 1.0 } else { p_names.get(&nl).copied().unwrap_or(0.0) };
        if exact || pn >= NAME_MIN_P || pl >= min {
            let score = 1.0 - (1.0 - pn) * (1.0 - pl);
            let r4 = |x: f32| (x as f64 * 10000.0).round() / 10000.0;
            hits.push((exact, score, c.unwrap_or(-1.0), serde_json::json!({"id": o["id"], "name": o["name"], "state": o["state"], "pos": o["pos"],
                "score": r4(score), "cos": r4(c.unwrap_or(0.0)), "p_name": r4(pn), "p_look": r4(pl), "by": if pn >= pl { "name" } else { "look" }})));
        }
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(b.2.total_cmp(&a.2)));
    hits.truncate(k);
    let mut read: Vec<(&String, &f32)> = p_names.iter().filter(|(_, p)| **p >= NAME_MIN_P).collect();
    read.sort_by(|a, b| b.1.total_cmp(a.1));
    read.truncate(3);
    serde_json::json!({"q": q, "semantic": tq.is_some(), "min": min,
        "names": read.iter().map(|(n, p)| serde_json::json!({"name": n, "p": (**p as f64 * 100.0).round() / 100.0})).collect::<Vec<_>>(),
        "results": hits.into_iter().map(|h| h.3).collect::<Vec<_>>()})
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
        let mut objs = vec![];
        for (id, (name, ax)) in [("chair", 0), ("cup", 1), ("tv", 2)].into_iter().enumerate() {
            let rel = format!("objects/O{id}_emb.bin");
            let bytes: Vec<u8> = axis(ax, 9, 0.0).iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(root.join(&rel), bytes).unwrap();
            objs.push(serde_json::json!({"id": id, "name": name, "state": "present", "pos": [0, 0, 0], "emb": rel}));
        }
        (root, serde_json::json!({"objects": objs}))
    }

    /// stand-in text encoder: a text sits on its own axis (30+) with a small part along the picture it describes
    /// (real text · picture cosines are small too: p = 0.5 at cosine 0.150)
    fn enc(t: &str) -> Option<Arc<Vec<f32>>> {
        Some(Arc::new(match t {
            "의자" => axis(30, 0, 0.17),
            "마실 것" => axis(40, 1, 0.16),
            "고양이" => axis(33, 3, 0.17),
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
        let r = search(&root, &view, "Chair", &none, 10, None);
        assert_eq!(ids(&r), vec![(0, "name".to_string())]);
        assert_eq!(r["semantic"], false);
        assert!(search(&root, &view, "의자", &none, 10, None)["results"].as_array().unwrap().is_empty());
        assert!(search(&root, &view, "  ", &none, 10, None)["results"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn query_finds_the_picture_in_any_language() {
        let (root, view) = memory("korean");
        let r = search(&root, &view, "의자", &enc, 10, None);
        assert_eq!(ids(&r), vec![(0, "look".to_string())], "{r}");
        assert_eq!(r["min"].as_f64().unwrap() as f32, DIRECT_MIN_KO);
        let r = search(&root, &view, "마실 것", &enc, 10, None);
        assert_eq!(ids(&r), vec![(1, "look".to_string())], "{r}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn absent_thing_matches_nothing() {
        // "고양이" (cat): no picture in the memory points its way
        let (root, view) = memory("absent");
        let r = search(&root, &view, "고양이", &enc, 10, None);
        assert!(r["results"].as_array().unwrap().is_empty(), "{r}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn letters_first_then_cosine() {
        let (root, view) = memory("order");
        // every picture passes a zero threshold; the cup holds the letters "cup" and comes first
        let r = search(&root, &view, "cup", &enc, 10, Some(0.0));
        assert_eq!(ids(&r)[0], (1, "name".to_string()));
        assert_eq!(r["results"].as_array().unwrap().len(), 3);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn threshold_by_language() {
        assert_eq!(default_min("chair"), DIRECT_MIN);
        assert_eq!(default_min("빨간 의자"), DIRECT_MIN_KO);
        // p = 0.5 at cosine 0.150, p = 0.004 near cosine 0.1006
        assert!((prob(0.15) - 0.5).abs() < 0.01);
        assert!((prob(0.1006) - 0.004).abs() < 0.0003);
    }

    #[test]
    fn missing_vector_file_is_skipped() {
        let (root, view) = memory("missing");
        std::fs::remove_file(root.join("objects/O0_emb.bin")).unwrap();
        assert!(search(&root, &view, "의자", &enc, 10, None)["results"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    /// `cargo test -p sgview --release -- --ignored --nocapture query_time`: ms per query, objects 10 · 100 · 1,000, vector files in memory
    #[test]
    #[ignore]
    fn query_time() {
        crate::memfs::enable(); // the board runs the viewer in the same process (--serve): files from memory
        let mut s = 11u64;
        let mut rnd_unit = || {
            let mut v: Vec<f32> = (0..DIM).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5 }).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            v
        };
        for n in [10usize, 100, 1000] {
            let root = std::env::temp_dir().join(format!("sgview_qt{n}_{}", std::process::id()));
            let objs: Vec<serde_json::Value> = (0..n).map(|id| {
                let rel = format!("objects/O{id}_emb.bin");
                crate::memfs::publish(&root.join(&rel), Arc::new(rnd_unit().iter().flat_map(|x| x.to_le_bytes()).collect()));
                serde_json::json!({"id": id, "name": "object", "state": "present", "pos": [0, 0, 0], "emb": rel})
            }).collect();
            let view = serde_json::json!({"objects": objs});
            let q = Arc::new(rnd_unit());
            let enc = |_: &str| Some(q.clone());
            let c = std::time::Instant::now();
            search(&root, &view, "something", &enc, 20, None);
            let first = c.elapsed().as_secs_f64() * 1e3;
            let c = std::time::Instant::now();
            for _ in 0..5 {
                std::hint::black_box(search(&root, &view, "something", &enc, 20, None));
            }
            println!("objects {n}: query {:.3} ms (first, vectors into the table: {first:.2} ms)", c.elapsed().as_secs_f64() * 1e3 / 5.0);
        }
    }
}

/// The viewer's encoder (first search): model dir + word vectors made ahead of time (`--vocab`): queries that are one of these words skip the encoder
#[cfg(feature = "search")]
pub fn load_encoder(dir: &str, vocab: Option<&str>) -> Option<TextEnc> {
    match TextEnc::load(Path::new(dir)) {
        Ok(e) => {
            match vocab.map(|f| read_vocab(Path::new(f))) {
                Some(Ok(v)) => {
                    let _ = READ_VOCAB.set(v.clone());
                    e.know(v)
                }
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
