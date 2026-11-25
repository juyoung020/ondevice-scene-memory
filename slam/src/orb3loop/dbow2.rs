//! DBoW2(ORB-SLAM3 Thirdparty/DBoW2) ORB 어휘 트리 · 변환 · L1 점수를 비트 동일하게 옮긴 것.
//! 원본: TemplatedVocabulary.h(loadFromTextFile · transform(features, BowVector, FeatureVector, levelsup) ·
//! transform(feature, ..)), BowVector.cpp(addWeight · normalize), FeatureVector.cpp(addFeature),
//! ScoringObject.cpp(L1Scoring::score), FORB.cpp(distance). ORBvoc.txt 는 L1 점수 · TF_IDF 무게.
//!
//! 원본 텍스트 로더의 버릇까지 그대로: 파일 끝 줄바꿈 뒤 빈 줄도 노드 하나로 만든다(부모 · 잎 표시는
//! 바로 앞 줄 값이 남고, 무게 0 · 설명자 0). 이 노드는 단어 971814 로 끝 부모의 마지막 자식이 되며,
//! 설명자가 거기로 떨어지면 무게 0 이라 BoW 에서 빠진다(원본 덤프로 확인).
//! 이진 캐시(`<txt>.rsbin`)는 같은 노드 표를 그대로 저장해 결과가 같다.

use std::io::{Read, Write};

pub type WordId = u32;
pub type NodeId = u32;
pub type Desc = [u8; 32];

/// std::map<WordId, double> 과 같은 순서(단어 번호 오름차순)
pub type BowVector = Vec<(WordId, f64)>;
/// std::map<NodeId, std::vector<unsigned>> 과 같은 순서
pub type FeatureVector = Vec<(NodeId, Vec<u32>)>;

#[inline]
pub fn distance(a: &Desc, b: &Desc) -> i32 {
    let mut d = 0u32;
    for i in 0..4 {
        let x = u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        d += x.count_ones();
    }
    d as i32
}

pub struct Vocabulary {
    pub k: u32,
    pub l: u32,
    pub scoring: u32,
    pub weighting: u32,
    pub parent: Vec<NodeId>,
    pub word_id: Vec<WordId>,
    pub weight: Vec<f64>,
    pub desc: Vec<Desc>,
    pub leaf: Vec<bool>,
    /// 자식 목록(CSR): 원본 children 과 같은 순서(노드 번호 순으로 push)
    pub child_off: Vec<u32>,
    pub child: Vec<NodeId>,
    /// 단어 → 노드
    pub words: Vec<NodeId>,
}

const MAGIC: &[u8; 8] = b"ORBVRS01";

impl Vocabulary {
    pub fn size(&self) -> usize {
        self.words.len()
    }

    fn empty(k: u32, l: u32, scoring: u32, weighting: u32, n: usize) -> Self {
        Vocabulary {
            k,
            l,
            scoring,
            weighting,
            parent: Vec::with_capacity(n),
            word_id: Vec::with_capacity(n),
            weight: Vec::with_capacity(n),
            desc: Vec::with_capacity(n),
            leaf: Vec::with_capacity(n),
            child_off: vec![],
            child: vec![],
            words: vec![],
        }
    }

    fn push(&mut self, parent: u32, leaf: bool, w: f64, d: Desc) {
        let nid = self.parent.len() as u32;
        self.parent.push(parent);
        self.weight.push(w);
        self.desc.push(d);
        self.leaf.push(leaf);
        if leaf {
            self.word_id.push(self.words.len() as u32);
            self.words.push(nid);
        } else {
            self.word_id.push(0);
        }
    }

    fn build_children(&mut self) {
        let n = self.parent.len();
        let mut off = vec![0u32; n + 1];
        for i in 1..n {
            off[self.parent[i] as usize + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let mut fill = off.clone();
        let mut child = vec![0u32; n.saturating_sub(1)];
        for i in 1..n {
            let p = self.parent[i] as usize;
            child[fill[p] as usize] = i as u32;
            fill[p] += 1;
        }
        self.child_off = off;
        self.child = child;
    }

    /// loadFromTextFile 그대로
    pub fn load_text(path: &str) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let mut lines = s.split('\n');
        let head: Vec<i64> = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .map(|t| t.parse().unwrap_or(-1))
            .collect();
        if head.len() < 4 {
            return Err(std::io::Error::other("Vocabulary loading failure"));
        }
        let (k, l, n1, n2) = (head[0], head[1], head[2], head[3]);
        if !(0..=20).contains(&k) || !(1..=10).contains(&l) || !(0..=5).contains(&n1) || !(0..=3).contains(&n2) {
            return Err(std::io::Error::other("Vocabulary loading failure"));
        }
        let mut v = Self::empty(k as u32, l as u32, n1 as u32, n2 as u32, 1 << 20);
        v.push(0, false, 0.0, [0; 32]);
        v.leaf[0] = false;
        let (mut prev_pid, mut prev_leaf) = (0u32, 0i64);
        for line in lines {
            let mut it = line.split_whitespace();
            let (pid, isleaf, d, w) = match it.next() {
                // 빈 줄: 원본은 읽기 실패로 지역 변수가 앞 줄 값 그대로, 무게 0, 설명자 0
                None => (prev_pid, prev_leaf, [0u8; 32], 0.0),
                Some(t) => {
                    let pid: u32 = t.parse().unwrap();
                    let isleaf: i64 = it.next().unwrap().parse().unwrap();
                    let mut d = [0u8; 32];
                    for b in d.iter_mut() {
                        *b = it.next().unwrap().parse::<i64>().unwrap() as u8;
                    }
                    let w: f64 = it.next().unwrap().parse().unwrap();
                    (pid, isleaf, d, w)
                }
            };
            prev_pid = pid;
            prev_leaf = isleaf;
            v.push(pid, isleaf > 0, w, d);
        }
        v.build_children();
        Ok(v)
    }

    pub fn save_bin(&self, path: &str) -> std::io::Result<()> {
        let n = self.parent.len();
        let mut b = Vec::with_capacity(28 + n * 45);
        b.extend_from_slice(MAGIC);
        for x in [self.k, self.l, self.scoring, self.weighting, n as u32] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for i in 0..n {
            b.extend_from_slice(&self.parent[i].to_le_bytes());
            b.push(self.leaf[i] as u8);
            b.extend_from_slice(&self.weight[i].to_le_bytes());
            b.extend_from_slice(&self.desc[i]);
        }
        let tmp = format!("{path}.tmp");
        std::fs::File::create(&tmp)?.write_all(&b)?;
        std::fs::rename(tmp, path)
    }

    pub fn load_bin(path: &str) -> std::io::Result<Self> {
        let mut b = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut b)?;
        if b.len() < 28 || &b[..8] != MAGIC {
            return Err(std::io::Error::other("not ORBVRS01"));
        }
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (k, l, sc, wt, n) = (u(8), u(12), u(16), u(20), u(24) as usize);
        if b.len() != 28 + n * 45 {
            return Err(std::io::Error::other("size"));
        }
        let mut v = Self::empty(k, l, sc, wt, n);
        let mut o = 28;
        for _ in 0..n {
            let w = f64::from_le_bytes(b[o + 5..o + 13].try_into().unwrap());
            v.push(u(o), b[o + 4] != 0, w, b[o + 13..o + 45].try_into().unwrap());
            o += 45;
        }
        v.build_children();
        Ok(v)
    }

    /// 이진 캐시가 있으면 그걸, 없으면 텍스트를 읽고 캐시를 만든다(결과 같음)
    pub fn load(path_txt: &str) -> std::io::Result<Self> {
        let bin = format!("{path_txt}.rsbin");
        if let Ok(v) = Self::load_bin(&bin) {
            return Ok(v);
        }
        let v = Self::load_text(path_txt)?;
        let _ = v.save_bin(&bin);
        Ok(v)
    }

    #[inline]
    fn is_leaf(&self, n: usize) -> bool {
        // 원본 Node::isLeaf() = children.empty()
        self.child_off[n] == self.child_off[n + 1]
    }

    /// transform(feature, word_id, weight, nid, levelsup) → (단어, 무게, levelsup 위 노드)
    pub fn transform_one(&self, f: &Desc, levelsup: i32) -> (WordId, f64, NodeId) {
        let nid_level = self.l as i32 - levelsup;
        let mut nid = 0u32;
        let mut final_id = 0usize;
        let mut level = 0;
        loop {
            level += 1;
            let (a, b) = (self.child_off[final_id] as usize, self.child_off[final_id + 1] as usize);
            let ch = &self.child[a..b];
            final_id = ch[0] as usize;
            let mut best = distance(f, &self.desc[final_id]);
            for &c in &ch[1..] {
                let d = distance(f, &self.desc[c as usize]);
                if d < best {
                    best = d;
                    final_id = c as usize;
                }
            }
            if level == nid_level {
                nid = final_id as u32;
            }
            if self.is_leaf(final_id) {
                break;
            }
        }
        (self.word_id[final_id], self.weight[final_id], nid)
    }

    /// transform(features, BowVector, FeatureVector, levelsup) — L1 정규화 · TF_IDF(ORBvoc)
    pub fn transform(&self, feats: &[Desc], levelsup: i32) -> (BowVector, FeatureVector) {
        assert!(self.weighting <= 1 && self.scoring == 0, "ORBvoc 설정(L1 · TF_IDF)만");
        let mut bw: Vec<(WordId, f64)> = Vec::with_capacity(feats.len());
        let mut fw: Vec<(NodeId, u32)> = Vec::with_capacity(feats.len());
        for (i, f) in feats.iter().enumerate() {
            let (id, w, nid) = self.transform_one(f, levelsup);
            if w > 0.0 {
                bw.push((id, w));
                fw.push((nid, i as u32));
            }
        }
        // addWeight: 같은 단어는 특징 순서대로 더함(안정 정렬 후 접기)
        bw.sort_by_key(|e| e.0);
        let mut bv: BowVector = Vec::with_capacity(bw.len());
        for (id, w) in bw {
            match bv.last_mut() {
                Some(l) if l.0 == id => l.1 += w,
                _ => bv.push((id, w)),
            }
        }
        fw.sort_by_key(|e| e.0);
        let mut fv: FeatureVector = Vec::new();
        for (nid, i) in fw {
            match fv.last_mut() {
                Some(l) if l.0 == nid => l.1.push(i),
                _ => fv.push((nid, vec![i])),
            }
        }
        // BowVector::normalize(L1)
        let mut norm = 0.0f64;
        for e in &bv {
            norm += e.1.abs();
        }
        if norm > 0.0 {
            for e in bv.iter_mut() {
                e.1 /= norm;
            }
        }
        (bv, fv)
    }

    /// L1Scoring::score
    pub fn score(&self, v1: &BowVector, v2: &BowVector) -> f64 {
        score_l1(v1, v2)
    }
}

/// L1Scoring::score (lower_bound 건너뛰기 = 오름차순 합집합 걷기, 더하는 순서 같음)
pub fn score_l1(v1: &BowVector, v2: &BowVector) -> f64 {
    let (mut i, mut j) = (0, 0);
    let mut s = 0.0f64;
    while i < v1.len() && j < v2.len() {
        let (a, b) = (v1[i], v2[j]);
        if a.0 == b.0 {
            s += (a.1 - b.1).abs() - a.1.abs() - b.1.abs();
            i += 1;
            j += 1;
        } else if a.0 < b.0 {
            i += v1[i..].partition_point(|e| e.0 < b.0);
        } else {
            j += v2[j..].partition_point(|e| e.0 < a.0);
        }
    }
    -s / 2.0
}
