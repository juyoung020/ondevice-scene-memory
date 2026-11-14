//! Text search over the objects of a memory folder.
//!
//! memmap (`--embed`) leaves one SigLIP2 image vector per object (`objects/O<id>_emb.bin`, 768 f32, unit length). A query is turned
//! into a vector by the SigLIP2 text encoder and compared with two things per object:
//!   name  — the object's class name (also a text vector): "의자" lands next to "chair", so stored English names answer Korean queries
//!   look  — the object's image vector: free descriptions ("something to drink from") match by appearance
//! The "vector DB" is those files read into one flat scan; a memory holds tens to hundreds of objects.
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

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
const NULLS: [&str; 5] = ["a photo", "an image", "a blurry photo", "nothing", "background"];
/// Token count the text encoder was trained with
const TOKENS: usize = 64;

pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One object vector file → unit vector (None when missing or not DIM floats)
pub fn read_emb(p: &Path) -> Option<Vec<f32>> {
    let b = std::fs::read(p).ok()?;
    if b.len() != DIM * 4 {
        return None;
    }
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(feature = "search")]
pub struct TextEnc {
    sess: Mutex<ort::session::Session>,
    tok: tokenizers::Tokenizer,
    cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// word vectors read from a file (`--vocab`): never dropped, never computed
    known: std::sync::OnceLock<HashMap<String, Arc<Vec<f32>>>>,
}

#[cfg(feature = "search")]
impl TextEnc {
    /// `dir` holds `tokenizer.json` and `text_model_fp16.onnx` (or `text_model.onnx`)
    pub fn load(dir: &Path) -> Result<TextEnc, String> {
        let model = ["text_model_fp16.onnx", "text_model.onnx"].iter().map(|n| dir.join(n)).find(|p| p.exists())
            .ok_or_else(|| format!("no text_model*.onnx in {}", dir.display()))?;
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: TOKENS, ..Default::default() })).map_err(|e| e.to_string())?;
        tok.with_padding(Some(tokenizers::PaddingParams { strategy: tokenizers::PaddingStrategy::Fixed(TOKENS), pad_id: 0, pad_token: "<pad>".into(), ..Default::default() }));
        let env: &'static ort::environment::Environment = Box::leak(Box::new(ort::init().build().map_err(|e| e.to_string())?));
        let sess = ort::session::Session::builder(env).map_err(|e| e.to_string())?
            .with_intra_threads(4).map_err(|e| e.to_string())?
            .commit_from_file(&model).map_err(|e| format!("{}: {e}", model.display()))?;
        Ok(TextEnc { sess: Mutex::new(sess), tok, cache: Mutex::new(HashMap::new()), known: std::sync::OnceLock::new() })
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
        let enc = self.tok.encode(key.as_str(), true).map_err(|e| e.to_string())?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mut v = {
            let mut sess = self.sess.lock().unwrap();
            let inp = ort::value::TensorRef::from_array_view(([1usize, ids.len()], &ids[..])).map_err(|e| e.to_string())?;
            let out = sess.run(ort::inputs![inp]).map_err(|e| e.to_string())?;
            let (_, v) = out["pooler_output"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
            v.to_vec()
        };
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

/// `--encode-vocab <names.txt> <out>`: one name a line → word vector file (done once on a PC; the file is then used as it is)
#[cfg(feature = "search")]
pub fn encode_vocab(model: &Path, list: &Path, out: &Path) -> Result<usize, String> {
    let enc = TextEnc::load(model)?;
    let text = std::fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
    let names: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut b = format!("SGVOC 1 {} {DIM}\n", names.len()).into_bytes();
    for n in &names {
        b.extend(n.as_bytes());
        b.push(b'\n');
    }
    for n in &names {
        b.extend(enc.encode(n)?.iter().flat_map(|x| x.to_le_bytes()));
    }
    std::fs::write(out, b).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(names.len())
}

/// Text → unit vector, or None when no encoder is loaded (then only the letters of the name are compared)
pub type Encode<'a> = &'a dyn Fn(&str) -> Option<Arc<Vec<f32>>>;

/// log(Σ e^x)
fn lse(xs: impl Iterator<Item = f32>) -> f32 {
    let v: Vec<f32> = xs.collect();
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY { m } else { m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() }
}

/// Class names the query competes against: the detector's vocabulary memmap leaves in `vocab.json`, plus every name in the memory
fn vocabulary(root: &Path, present: &[String]) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_to_string(root.join("vocab.json")).ok()
        .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok()).unwrap_or_default()
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
    fn missing_vector_file_is_skipped() {
        let (root, view) = memory("missing");
        std::fs::remove_file(root.join("objects/O0_emb.bin")).unwrap();
        let r = search(&root, &view, "의자", &enc, 10, 0.1);
        assert_eq!(ids(&r), vec![(0, "name".to_string())]);
        assert_eq!(r["results"][0]["p_look"], 0.0);
        let _ = std::fs::remove_dir_all(root);
    }
}
