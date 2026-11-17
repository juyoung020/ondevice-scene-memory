//! 물체 벡터 메모리 표: 물체마다 벡터 하나(예: SigLIP2 사진 벡터 768 개)를 이어 붙인 f32 행렬 하나에 두고,
//! 단어 목록(글 벡터)과의 점수 `scale · (단어 · 물체)` 를 물체가 들어오거나 바뀔 때만 미리 계산해 둔다.
//! 질의는 파일을 읽지 않고 이 표만 본다: 질의 벡터와 모든 물체의 내적 한 번(top-k), 미리 계산한 단어 점수는 [`ObjTable::logits`].
//!
//! ```ignore
//! use sgview::objtable::ObjTable;
//! let mut t: ObjTable<u32> = ObjTable::new(768, 111.83);
//! t.set_words(word_vectors);          // 단어 목록(바뀌면 모든 물체의 단어 점수를 다시 계산)
//! t.upsert(7, &picture_vector);       // 추가 · 갱신(같은 열쇠면 바꿔 씀)
//! t.remove(&7);                       // 삭제
//! let best = t.top_k(&query_vector, 5); // [(열쇠, 내적)] 큰 순
//! ```
//! 벡터 파일 바이트(f32 리틀 엔디언)로 맞추기는 [`ObjTable::sync_bytes`]: 바뀐 것만 다시 계산하고 목록에 없는 물체는 지운다.
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// a · b. aarch64(보드): NEON 4 칸 × 4 묶음(더하는 순서가 달라 차례대로 더한 값과 상대 1e-7 안팎 차), 그 밖: 차례대로
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON 은 모든 aarch64 대상에 있음
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_plain(a, b)
}

/// 차례대로 더한 내적(기준)
pub fn dot_plain(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 16 <= n {
        for (k, s) in acc.iter_mut().enumerate() {
            *s = vfmaq_f32(*s, vld1q_f32(pa.add(i + 4 * k)), vld1q_f32(pb.add(i + 4 * k)));
        }
        i += 16;
    }
    let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
    while i < n {
        s += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    s
}

/// 여러 줄을 나눠 돌림(rayon 이 있으면 코어 여럿, 없으면 차례대로). 결과 순서는 입력 순서
fn par_map<T: Sync, R: Send>(v: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "search")]
    {
        use rayon::prelude::*;
        v.par_iter().with_min_len(8).map(f).collect()
    }
    #[cfg(not(feature = "search"))]
    v.iter().map(f).collect()
}

/// 물체 벡터 표(열쇠 K: 물체 번호 · 파일 경로 등)
pub struct ObjTable<K> {
    dim: usize,
    scale: f32,
    keys: Vec<K>,
    idx: HashMap<K, usize>,
    /// 줄을 만든 벡터 파일 바이트(sync_bytes 의 바뀜 확인용, upsert 로 넣은 줄은 None)
    src: Vec<Option<Arc<Vec<u8>>>>,
    /// 줄 수 × dim
    vecs: Vec<f32>,
    words: Vec<Arc<Vec<f32>>>,
    /// 줄 수 × 단어 수: scale · (단어 · 물체)
    logits: Vec<f32>,
}

impl<K: Hash + Eq + Clone + Send + Sync> ObjTable<K> {
    pub fn new(dim: usize, scale: f32) -> ObjTable<K> {
        ObjTable { dim, scale, keys: vec![], idx: HashMap::new(), src: vec![], vecs: vec![], words: vec![], logits: vec![] }
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn words(&self) -> &[Arc<Vec<f32>>] {
        &self.words
    }

    fn row_logits(words: &[Arc<Vec<f32>>], scale: f32, e: &[f32]) -> Vec<f32> {
        words.iter().map(|w| scale * dot(w, e)).collect()
    }

    /// 단어 목록. 전과 같으면(같은 벡터) 그대로, 다르면 모든 물체의 단어 점수를 다시 계산. 돌려줌: 다시 계산했는지
    pub fn set_words(&mut self, words: Vec<Arc<Vec<f32>>>) -> bool {
        let same = self.words.len() == words.len() && self.words.iter().zip(&words).all(|(a, b)| Arc::ptr_eq(a, b) || a == b);
        if same {
            return false;
        }
        self.words = words;
        let (d, nw, s) = (self.dim, self.words.len(), self.scale);
        let rows: Vec<usize> = (0..self.keys.len()).collect();
        let w = &self.words;
        let vecs = &self.vecs;
        let l = par_map(&rows, |&r| Self::row_logits(w, s, &vecs[r * d..(r + 1) * d]));
        self.logits = Vec::with_capacity(rows.len() * nw);
        for x in l {
            self.logits.extend(x);
        }
        true
    }

    fn put_row(&mut self, key: K, v: &[f32], logit: Vec<f32>, src: Option<Arc<Vec<u8>>>) {
        assert_eq!(v.len(), self.dim, "물체 벡터 길이");
        let (d, nw) = (self.dim, self.words.len());
        match self.idx.get(&key) {
            Some(&r) => {
                self.vecs[r * d..(r + 1) * d].copy_from_slice(v);
                self.logits[r * nw..(r + 1) * nw].copy_from_slice(&logit);
                self.src[r] = src;
            }
            None => {
                self.idx.insert(key.clone(), self.keys.len());
                self.keys.push(key);
                self.vecs.extend_from_slice(v);
                self.logits.extend(logit);
                self.src.push(src);
            }
        }
    }

    /// 추가 · 갱신: 그 물체의 벡터를 넣고 단어 점수를 계산
    pub fn upsert(&mut self, key: K, v: &[f32]) {
        let l = Self::row_logits(&self.words, self.scale, v);
        self.put_row(key, v, l, None);
    }

    /// 삭제(마지막 줄을 그 자리로 옮김). 돌려줌: 있었는지
    pub fn remove(&mut self, key: &K) -> bool {
        let Some(r) = self.idx.remove(key) else { return false };
        let (d, nw, last) = (self.dim, self.words.len(), self.keys.len() - 1);
        if r != last {
            self.vecs.copy_within(last * d..(last + 1) * d, r * d);
            self.logits.copy_within(last * nw..(last + 1) * nw, r * nw);
            let k = self.keys[last].clone();
            self.idx.insert(k, r);
        }
        self.keys.swap_remove(r);
        self.src.swap_remove(r);
        self.vecs.truncate(last * d);
        self.logits.truncate(last * nw);
        true
    }

    /// 벡터 파일 바이트(f32 리틀 엔디언, dim 개)로 맞추기: 바이트가 바뀐 물체만 다시 계산(코어 여럿), items 에 없는 물체는 지움.
    /// 길이가 맞지 않는 바이트는 건너뜀
    pub fn sync_bytes(&mut self, items: &[(K, Arc<Vec<u8>>)]) {
        let live: std::collections::HashSet<&K> = items.iter().map(|x| &x.0).collect();
        let gone: Vec<K> = self.keys.iter().filter(|k| !live.contains(k)).cloned().collect();
        for k in &gone {
            self.remove(k);
        }
        let todo: Vec<&(K, Arc<Vec<u8>>)> = items.iter().filter(|(k, b)| b.len() == self.dim * 4 && match self.idx.get(k) {
            Some(&r) => self.src[r].as_ref().is_none_or(|s| !Arc::ptr_eq(s, b) && s != b),
            None => true,
        }).collect();
        let (w, s) = (&self.words, self.scale);
        let made = par_map(&todo, |(k, b)| {
            let e: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let l = Self::row_logits(w, s, &e);
            (k.clone(), e, l, b.clone())
        });
        for (k, e, l, b) in made {
            self.put_row(k, &e, l, Some(b));
        }
    }

    /// 그 물체의 줄 번호
    pub fn row(&self, key: &K) -> Option<usize> {
        self.idx.get(key).copied()
    }
    pub fn key(&self, row: usize) -> &K {
        &self.keys[row]
    }
    /// 줄의 물체 벡터
    pub fn vector(&self, row: usize) -> &[f32] {
        &self.vecs[row * self.dim..(row + 1) * self.dim]
    }
    /// 줄의 단어 점수(단어 목록 순서)
    pub fn logits(&self, row: usize) -> &[f32] {
        let nw = self.words.len();
        &self.logits[row * nw..(row + 1) * nw]
    }

    /// 질의 벡터와 모든 물체의 내적(줄 순서)
    pub fn scores(&self, q: &[f32]) -> Vec<f32> {
        let rows: Vec<usize> = (0..self.keys.len()).collect();
        par_map(&rows, |&r| dot(q, self.vector(r)))
    }

    /// 질의 벡터와 내적이 큰 물체 k 개: [(열쇠, 내적)] 큰 순
    pub fn top_k(&self, q: &[f32], k: usize) -> Vec<(K, f32)> {
        let s = self.scores(q);
        let mut o: Vec<usize> = (0..s.len()).collect();
        let k = k.min(o.len());
        if k < o.len() && k > 0 {
            o.select_nth_unstable_by(k - 1, |a, b| s[*b].total_cmp(&s[*a]));
        }
        o.truncate(k);
        o.sort_by(|a, b| s[*b].total_cmp(&s[*a]));
        o.into_iter().map(|r| (self.keys[r].clone(), s[r])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(s: &mut u64, d: usize) -> Vec<f32> {
        (0..d).map(|_| { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((*s >> 40) as f32) / (1u64 << 24) as f32 - 0.5 }).collect()
    }

    #[test]
    fn add_update_remove_topk() {
        let mut s = 3u64;
        let words: Vec<Arc<Vec<f32>>> = (0..5).map(|_| Arc::new(rnd(&mut s, 16))).collect();
        let mut t: ObjTable<u32> = ObjTable::new(16, 2.0);
        t.set_words(words.clone());
        let vs: Vec<Vec<f32>> = (0..6).map(|_| rnd(&mut s, 16)).collect();
        for (i, v) in vs.iter().enumerate() {
            t.upsert(i as u32, v);
        }
        assert_eq!(t.len(), 6);
        let r = t.row(&3).unwrap();
        assert_eq!(t.vector(r), &vs[3][..]);
        assert_eq!(t.logits(r)[2], 2.0 * dot(&words[2], &vs[3]));
        // 질의 = 물체 4 의 벡터: 1 등은 물체 4
        assert_eq!(t.top_k(&vs[4], 1)[0].0, 4);
        // 갱신 · 삭제(마지막 줄이 빈 자리로)
        t.upsert(4, &vs[0]);
        assert_eq!(t.vector(t.row(&4).unwrap()), &vs[0][..]);
        assert!(t.remove(&1) && !t.remove(&1));
        assert_eq!(t.len(), 5);
        for k in [0u32, 2, 3, 4, 5] {
            let r = t.row(&k).unwrap();
            let want = if k == 4 { &vs[0] } else { &vs[k as usize] };
            assert_eq!(t.vector(r), &want[..]);
            assert_eq!(t.logits(r)[1], 2.0 * dot(&words[1], want));
        }
        // 단어 목록이 바뀌면 다시 계산
        let w2: Vec<Arc<Vec<f32>>> = (0..3).map(|_| Arc::new(rnd(&mut s, 16))).collect();
        assert!(t.set_words(w2.clone()) && !t.set_words(w2.clone()));
        let r = t.row(&5).unwrap();
        assert_eq!(t.logits(r), &[2.0 * dot(&w2[0], &vs[5]), 2.0 * dot(&w2[1], &vs[5]), 2.0 * dot(&w2[2], &vs[5])][..]);
        let top = t.top_k(&vs[2], 10);
        assert_eq!(top.len(), 5);
        assert!(top.windows(2).all(|p| p[0].1 >= p[1].1));
    }

    #[test]
    fn sync_with_file_bytes() {
        let mut s = 9u64;
        let bytes = |v: &[f32]| Arc::new(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let mut t: ObjTable<String> = ObjTable::new(8, 1.0);
        t.set_words(vec![Arc::new(rnd(&mut s, 8))]);
        let (a, b) = (rnd(&mut s, 8), rnd(&mut s, 8));
        t.sync_bytes(&[("a".into(), bytes(&a)), ("b".into(), bytes(&b)), ("bad".into(), Arc::new(vec![1, 2, 3]))]);
        assert_eq!(t.len(), 2);
        t.sync_bytes(&[("b".into(), bytes(&a))]);
        assert_eq!(t.len(), 1);
        assert_eq!(t.vector(t.row(&"b".to_string()).unwrap()), &a[..]);
    }

    #[test]
    fn neon_dot_close() {
        let mut s = 5u64;
        for _ in 0..200 {
            let (a, b) = (rnd(&mut s, 768), rnd(&mut s, 768));
            let (x, y) = (dot(&a, &b), dot_plain(&a, &b));
            assert!((x - y).abs() <= 1e-5 * (1.0 + y.abs()), "{x} {y}");
        }
    }
}
