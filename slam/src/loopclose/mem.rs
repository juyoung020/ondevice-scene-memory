//! 지도 노드 기억(Memory) · 노드(Signature) · 링크(Link) · 단어 사전(VWDictionary, 찾기 색인은 WordIndex 에 맡김).
//! 원본: corelib/src/Memory.cpp(update · createSignature 의 위치 추적 특징 경로 · addSignatureToStm · rehearsal · rehearsalMerge ·
//!       moveSignatureToWMFromSTM · computeLikelihood · getNeighborsId · addLink · removeLink · moveToTrash · getMetricConstraints),
//!       Signature.cpp(compareTo · addLink · removeLink), Link.cpp(inverse · merge · transVariance · rotVariance),
//!       VWDictionary.cpp(update · addNewWords · removeWords · convertBinTo32F), VisualWord.cpp, Features2d.cpp(limitKeypoints, 응답 순).
//! 우리 설정(Mem/IncrementalMemory=true, Mem/STMSize=10, Mem/RehearsalSimilarity=0.6, Mem/UseOdomFeatures=true, Kp/MaxFeatures=300,
//! Vis/MaxFeatures=400, Kp/NndrRatio=0.8, Kp/NewWordsComparedTogether=true, Mem/CovOffDiagIgnored=true, Mem/GenerateIds=true)이 지나가는 길만 옮겼다.
use super::rtf::Tf;
use super::WordIndex;
use std::collections::{BTreeMap, BTreeSet};

/// 6x6 정보·공분산 행렬(행 우선, double)
pub type Mat6 = [f64; 36];

pub fn eye6(s: f64) -> Mat6 {
    let mut m = [0.0; 36];
    for i in 0..6 {
        m[i * 7] = s;
    }
    m
}

/// cv::Mat::inv()(DECOMP_LU, 6x6 double): OpenCV LUImpl 의 부분 피벗 가우스 소거를 같은 순서로
pub fn inv6(a: &Mat6) -> Mat6 {
    let mut a = *a;
    let mut b = eye6(1.0);
    let m = 6;
    for i in 0..m {
        let mut k = i;
        for j in i + 1..m {
            if a[j * 6 + i].abs() > a[k * 6 + i].abs() {
                k = j;
            }
        }
        if a[k * 6 + i].abs() < f64::EPSILON * 10.0 {
            return [0.0; 36]; // 원본: 특이 행렬이면 0 행렬
        }
        if k != i {
            for j in i..m {
                a.swap(i * 6 + j, k * 6 + j);
            }
            for j in 0..m {
                b.swap(i * 6 + j, k * 6 + j);
            }
        }
        let d = -1.0 / a[i * 6 + i];
        for j in i + 1..m {
            let alpha = a[j * 6 + i] * d;
            for c in i + 1..m {
                a[j * 6 + c] += alpha * a[i * 6 + c];
            }
            for c in 0..m {
                b[j * 6 + c] += alpha * b[i * 6 + c];
            }
        }
    }
    for i in (0..m).rev() {
        for j in 0..m {
            let mut s = b[i * 6 + j];
            for k in i + 1..m {
                s -= a[i * 6 + k] * b[k * 6 + j];
            }
            b[i * 6 + j] = s / a[i * 6 + i];
        }
    }
    b
}

/// uMax3(a,b,c): 중간값을 float 에 담는 원본 그대로
pub fn umax3(a: f64, b: f64, c: f64) -> f64 {
    let m = if a > b { a } else { b } as f32 as f64;
    if m > c { m } else { c }
}
fn umin3(a: f64, b: f64, c: f64) -> f64 {
    let m = if a < b { a } else { b } as f32 as f64;
    if m < c { m } else { c }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LinkType {
    Neighbor = 0,
    GlobalClosure = 1,
    LocalSpaceClosure = 2,
    LocalTimeClosure = 3,
    UserClosure = 4,
    VirtualClosure = 5,
    NeighborMerged = 6,
    PosePrior = 7,
    Landmark = 8,
    Gravity = 9,
    Undef = 99,
}

impl LinkType {
    pub fn from_i32(v: i32) -> LinkType {
        use LinkType::*;
        match v {
            0 => Neighbor,
            1 => GlobalClosure,
            2 => LocalSpaceClosure,
            3 => LocalTimeClosure,
            4 => UserClosure,
            5 => VirtualClosure,
            6 => NeighborMerged,
            7 => PosePrior,
            8 => Landmark,
            9 => Gravity,
            _ => Undef,
        }
    }
}

/// 두 노드 사이 제약: from 에서 본 to 의 위치 + 정보 행렬
#[derive(Clone, Debug)]
pub struct Link {
    pub from: i32,
    pub to: i32,
    pub ty: LinkType,
    pub t: Tf,
    pub inf: Mat6,
}

impl Link {
    pub fn new(from: i32, to: i32, ty: LinkType, t: Tf, inf: Mat6) -> Link {
        Link { from, to, ty, t, inf }
    }
    /// Link::inverse: 정보 행렬은 그대로
    pub fn inverse(&self) -> Link {
        if self.t.is_null() {
            Link::new(self.to, self.from, self.ty, Tf::NULL, eye6(1.0))
        } else {
            Link::new(self.to, self.from, self.ty, self.t.inverse(), self.inf)
        }
    }
    /// Link::merge(link, outputType) (NeighborMerged 아님: 정보 행렬은 x 정보가 작은 쪽)
    pub fn merge(&self, l: &Link, ty: LinkType) -> Link {
        if self.t.is_null() {
            return Link::new(self.from, l.to, ty, Tf::NULL, eye6(1.0));
        }
        if ty == LinkType::NeighborMerged {
            let a = inv6(&self.inf);
            let b = inv6(&l.inf);
            let mut s = [0.0; 36];
            for i in 0..36 {
                s[i] = a[i] + b[i];
            }
            return Link::new(self.from, l.to, ty, self.t.mul(&l.t), inv6(&s));
        }
        let inf = if self.inf[0] < l.inf[0] { self.inf } else { l.inf };
        Link::new(self.from, l.to, ty, self.t.mul(&l.t), inf)
    }
    pub fn is_valid(&self) -> bool {
        self.from != 0 && self.to != 0 && !self.t.is_null() && self.ty != LinkType::Undef
    }
    /// transVariance(minimum=false)
    pub fn trans_variance(&self) -> f64 {
        let f = |v: f64| if v <= 0.0001 { 9999999.0 } else { v };
        let mut v = umin3(f(self.inf[0]), f(self.inf[7]), f(self.inf[14]));
        if v == 9999999.0 {
            v = 0.0001;
        }
        1.0 / v
    }
    /// rotVariance(minimum=false)
    pub fn rot_variance(&self) -> f64 {
        let f = |v: f64| if v <= 0.0001 { 9999999.0 } else { v };
        let mut v = umin3(f(self.inf[21]), f(self.inf[28]), f(self.inf[35]));
        if v == 9999999.0 {
            v = 0.0001;
        }
        1.0 / v
    }
}

/// std::multimap<int, Link>: 같은 키는 넣은 순서
#[derive(Clone, Debug, Default)]
pub struct LinkMap(pub BTreeMap<i32, Vec<Link>>);

impl LinkMap {
    pub fn insert(&mut self, k: i32, l: Link) {
        self.0.entry(k).or_default().push(l);
    }
    pub fn iter(&self) -> impl Iterator<Item = (i32, &Link)> {
        self.0.iter().flat_map(|(k, v)| v.iter().map(move |l| (*k, l)))
    }
    pub fn len(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear();
    }
    /// graph::findLink(links, from, to, checkBothWays, type): (키, 키 안 순번)
    pub fn find(&self, from: i32, to: i32, both: bool, ty: LinkType) -> Option<(i32, usize)> {
        let ok = |l: &Link, t: i32| l.to == t && (ty == LinkType::Undef || ty == l.ty);
        if let Some(v) = self.0.get(&from) {
            if let Some(i) = v.iter().position(|l| ok(l, to)) {
                return Some((from, i));
            }
        }
        if both {
            if let Some(v) = self.0.get(&to) {
                if let Some(i) = v.iter().position(|l| ok(l, from)) {
                    return Some((to, i));
                }
            }
        }
        None
    }
    pub fn get(&self, at: (i32, usize)) -> &Link {
        &self.0[&at.0][at.1]
    }
    pub fn remove_at(&mut self, at: (i32, usize)) {
        let v = self.0.get_mut(&at.0).unwrap();
        v.remove(at.1);
        if v.is_empty() {
            self.0.remove(&at.0);
        }
    }
    pub fn remove_key(&mut self, k: i32) {
        self.0.remove(&k);
    }
}

/// 특징점 하나(위치 추적이 준 것)
#[derive(Clone, Copy, Debug, Default)]
pub struct Kp {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
}

/// 지도 노드
#[derive(Clone, Debug)]
pub struct Signature {
    pub id: i32,
    pub map_id: i32,
    pub weight: i32,
    pub stamp: f64,
    pub pose: Tf,
    /// 이웃 id → 링크(원본 multimap, 두 노드 사이 링크는 하나)
    pub links: BTreeMap<i32, Link>,
    /// 단어 id → 특징점 순번들(multimap: 같은 단어는 넣은 순서)
    pub words: BTreeMap<i32, Vec<i32>>,
    pub words_count: usize,
    pub invalid_words: usize,
    pub kpts: Vec<Kp>,
    pub pts3d: Vec<[f32; 3]>,
    pub desc: Vec<[u8; 32]>,
    pub enabled: bool,
    pub velocity: Vec<f32>,
}

impl Signature {
    pub fn is_bad(&self) -> bool {
        self.words_count as i64 - self.invalid_words as i64 <= 0
    }
    pub fn has_link(&self, id: i32) -> bool {
        self.links.contains_key(&id)
    }
    pub fn add_link(&mut self, l: Link) {
        debug_assert!(l.from == self.id);
        self.links.insert(l.to, l);
    }
    /// 원본 multimap 순서의 (단어, 순번) 쌍
    pub fn word_pairs(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.words.iter().flat_map(|(w, v)| v.iter().map(move |i| (*w, *i)))
    }
    /// Signature::compareTo(전역 설명자 없음): EpipolarGeometry::findPairs(multimap) 짝 수 / 큰 쪽 유효 단어 수
    pub fn compare_to(&self, s: &Signature) -> f32 {
        if s.is_bad() || self.is_bad() {
            return 0.0;
        }
        let a = self.words_count as i32 - self.invalid_words as i32;
        let b = s.words_count as i32 - s.invalid_words as i32;
        let total = if a > b { a } else { b };
        // findPairs(words(=s), _words(=self)): s 의 고유 단어마다 min(개수)
        let mut pairs = 0usize;
        for (w, va) in &s.words {
            if *w >= 0 {
                if let Some(vb) = self.words.get(w) {
                    pairs += va.len().min(vb.len());
                }
            }
        }
        pairs as f32 / total as f32
    }
}

/// 사전의 단어 하나
#[derive(Clone, Debug)]
pub struct VisualWord {
    pub id: i32,
    pub desc: [u8; 32],
    /// 노드 id → 그 노드가 이 단어를 가진 수
    pub refs: BTreeMap<i32, i32>,
}

/// VWDictionary(Kp/IncrementalDictionary=true, Kp/IncrementalFlann=true, Kp/NNStrategy=1). 찾기 색인은 WordIndex
pub struct Dictionary {
    pub words: BTreeMap<i32, VisualWord>,
    pub not_indexed: BTreeSet<i32>,
    pub removed_indexed: BTreeSet<i32>,
    pub unused: BTreeSet<i32>,
    pub last_word_id: i32,
    pub map_index_id: BTreeMap<i32, i32>,
    pub map_id_index: BTreeMap<i32, i32>,
    pub index: Box<dyn WordIndex>,
    pub nndr: f32,
}

/// VWDictionary::convertBinTo32F(byteToFloat=false) 뒤 L1 거리 = 서로 다른 비트 수
#[inline]
fn hamming(a: &[u8; 32], b: &[u8; 32]) -> f32 {
    let mut s = 0u32;
    for i in 0..32 {
        s += (a[i] ^ b[i]).count_ones();
    }
    s as f32
}

impl Dictionary {
    pub fn new(index: Box<dyn WordIndex>, nndr: f32) -> Dictionary {
        Dictionary {
            words: BTreeMap::new(),
            not_indexed: BTreeSet::new(),
            removed_indexed: BTreeSet::new(),
            unused: BTreeSet::new(),
            last_word_id: 0,
            map_index_id: BTreeMap::new(),
            map_id_index: BTreeMap::new(),
            index,
            nndr,
        }
    }

    pub fn clear(&mut self) {
        self.words.clear();
        self.not_indexed.clear();
        self.removed_indexed.clear();
        self.unused.clear();
        self.last_word_id = 0;
        self.map_index_id.clear();
        self.map_id_index.clear();
        self.index.release();
    }

    pub fn indexed_count(&self) -> usize {
        self.map_index_id.len()
    }

    /// VWDictionary::update: 처음에는 색인을 한 번에 짓고, 그 뒤로는 지운 단어를 빼고 새 단어를 하나씩 더한다
    pub fn update(&mut self) {
        if self.not_indexed.is_empty() && !self.words.is_empty() && self.removed_indexed.is_empty() {
            return;
        }
        let first = self.removed_indexed.is_empty() && self.words.len() == self.not_indexed.len();
        if !first && !self.words.is_empty() {
            for id in std::mem::take(&mut self.removed_indexed) {
                let idx = self.map_id_index[&id];
                self.index.remove_point(idx as u32);
                self.map_index_id.remove(&idx);
                self.map_id_index.remove(&id);
            }
            for id in self.not_indexed.clone() {
                let d = self.words[&id].desc;
                let index = if !self.index.is_built() {
                    self.index.build(&[d]);
                    0
                } else {
                    self.index.add_points(&[d])[0] as i32
                };
                assert!(self.map_index_id.insert(index, id).is_none());
                assert!(self.map_id_index.insert(id, index).is_none());
            }
        } else {
            self.map_index_id.clear();
            self.map_id_index.clear();
            self.index.release();
            if !self.words.is_empty() {
                let mut rows = Vec::with_capacity(self.words.len());
                for (i, (id, w)) in self.words.iter().enumerate() {
                    rows.push(w.desc);
                    self.map_index_id.insert(i as i32, *id);
                    self.map_id_index.insert(*id, i as i32);
                }
                self.index.build(&rows);
            }
        }
        self.not_indexed.clear();
        self.removed_indexed.clear();
    }

    fn add_word_ref(&mut self, word: i32, sig: i32) {
        if let Some(w) = self.words.get_mut(&word) {
            *w.refs.entry(sig).or_insert(0) += 1;
            self.unused.remove(&word);
        }
    }

    pub fn remove_all_word_ref(&mut self, word: i32, sig: i32) {
        if let Some(w) = self.words.get_mut(&word) {
            w.refs.remove(&sig);
            if w.refs.is_empty() {
                self.unused.insert(word);
            }
        }
    }

    /// VWDictionary::removeWords
    pub fn remove_words(&mut self, ids: &[i32]) {
        for id in ids {
            self.words.remove(id);
            self.unused.remove(id);
            if !self.not_indexed.remove(id) {
                self.removed_indexed.insert(*id);
            }
        }
    }

    /// VWDictionary::addNewWords: 색인에서 가까운 2 개 + 이번 노드의 새 단어끼리 비교, NNDR 통과면 기존 단어, 아니면 새 단어
    pub fn add_new_words(&mut self, desc: &[[u8; 32]], sig: i32) -> Vec<i32> {
        let mut ids = Vec::with_capacity(desc.len());
        if desc.is_empty() {
            return ids;
        }
        let k = 2usize;
        let (res, dists) = if self.index.is_built() { self.index.knn_search(desc, k, 32) } else { (Vec::new(), Vec::new()) };
        let searched = self.index.is_built();
        let mut new_words: Vec<[u8; 32]> = Vec::new();
        let mut new_ids: Vec<i32> = Vec::new();
        for i in 0..desc.len() {
            // multimap<float,int>: 같은 거리는 넣은 순서
            let mut full: Vec<(f32, i32)> = Vec::with_capacity(4);
            let put = |full: &mut Vec<(f32, i32)>, d: f32, id: i32| {
                let pos = full.iter().position(|(x, _)| *x > d).unwrap_or(full.len());
                full.insert(pos, (d, id));
            };
            if searched {
                for j in 0..k {
                    let d = dists[i * k + j];
                    let index = res[i * k + j];
                    if index < 0 {
                        continue;
                    }
                    let id = self.map_index_id.get(&index).copied().unwrap_or(0);
                    if d >= 0.0 && id != 0 {
                        put(&mut full, d, id);
                    } else {
                        break;
                    }
                }
            }
            if !new_words.is_empty() {
                // cv::BFMatcher(NORM_L1) knnMatch k = 2(새 단어가 1 개면 1): 같은 거리는 앞 순번
                let kk = if new_words.len() > 1 { 2 } else { 1 };
                let mut best: [(f32, i32); 2] = [(f32::MAX, -1), (f32::MAX, -1)];
                for (j, w) in new_words.iter().enumerate() {
                    let d = hamming(&desc[i], w);
                    if d < best[kk - 1].0 {
                        let mut p = kk - 1;
                        while p > 0 && best[p - 1].0 > d {
                            best[p] = best[p - 1];
                            p -= 1;
                        }
                        best[p] = (d, j as i32);
                    }
                }
                for b in best.iter().take(kk) {
                    if b.1 < 0 {
                        break;
                    }
                    let id = new_ids[b.1 as usize];
                    if b.0 >= 0.0 && id != 0 {
                        put(&mut full, b.0, id);
                    } else {
                        break;
                    }
                }
            }
            let bad = full.len() < 2 || full[0].0 > self.nndr * full[1].0;
            if bad {
                self.last_word_id += 1;
                let id = self.last_word_id;
                let mut refs = BTreeMap::new();
                refs.insert(sig, 1);
                self.words.insert(id, VisualWord { id, desc: desc[i], refs });
                self.not_indexed.insert(id);
                new_words.push(desc[i]);
                new_ids.push(id);
                ids.push(id);
            } else {
                let id = full[0].1;
                self.add_word_ref(id, sig);
                ids.push(id);
            }
        }
        ids
    }
}

/// Feature2D::limitKeypoints(inliers 판): |응답| 큰 순(같으면 뒤 순번 먼저, multimap 역순)으로 max 개
pub fn limit_inliers(kps: &[Kp], max: usize) -> Vec<bool> {
    if max == 0 || kps.len() <= max {
        return vec![true; kps.len()];
    }
    let order = response_order(kps);
    let mut inl = vec![false; kps.len()];
    for &i in order.iter().take(max) {
        inl[i] = true;
    }
    inl
}

/// multimap<float,int>(|응답|) 을 거꾸로 돈 순서
fn response_order(kps: &[Kp]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..kps.len()).collect();
    idx.sort_by(|&a, &b| {
        let ra = kps[a].response.abs();
        let rb = kps[b].response.abs();
        rb.partial_cmp(&ra).unwrap().then(b.cmp(&a))
    });
    idx
}

/// 위치 추적이 넘긴 한 노드의 특징
#[derive(Clone, Debug, Default)]
pub struct NodeFeatures {
    pub kps: Vec<Kp>,
    pub pts3d: Vec<[f32; 3]>,
    pub desc: Vec<[u8; 32]>,
}

/// Memory 의 설정값
#[derive(Clone, Debug)]
pub struct MemParams {
    pub stm_size: usize,
    pub rehearsal_similarity: f32,
    pub kp_max_features: i32,
    pub vis_max_features: i32,
    pub linear_update: f32,
    pub angular_update: f32,
    pub nndr: f32,
}

pub struct Memory {
    pub p: MemParams,
    pub signatures: BTreeMap<i32, Signature>,
    /// 작업 기억(-1 = 가상 장소 포함)
    pub wm: BTreeSet<i32>,
    pub stm: BTreeSet<i32>,
    pub dict: Dictionary,
    pub id_count: i32,
    pub id_map_count: i32,
    pub last_signature: Option<i32>,
    pub last_global_loop_closure_id: i32,
    pub all_nodes_in_wm: bool,
    pub odom_max_inf: Vec<f64>,
    /// 지난 update 의 되새김(rehearsal) 결과: (합친 노드, 닮음, 닮은 노드)
    pub rehearsal: (i32, f32, i32),
}

impl Memory {
    pub fn new(p: MemParams, index: Box<dyn WordIndex>) -> Memory {
        let nndr = p.nndr;
        let mut m = Memory {
            p,
            signatures: BTreeMap::new(),
            wm: BTreeSet::new(),
            stm: BTreeSet::new(),
            dict: Dictionary::new(index, nndr),
            id_count: 0,
            id_map_count: 0,
            last_signature: None,
            last_global_loop_closure_id: 0,
            all_nodes_in_wm: true,
            odom_max_inf: Vec::new(),
            rehearsal: (0, 0.0, 0),
        };
        m.wm.insert(-1);
        m
    }

    /// Memory::init(빈 데이터베이스로 다시)
    pub fn reset(&mut self) {
        self.signatures.clear();
        self.wm.clear();
        self.wm.insert(-1);
        self.stm.clear();
        self.dict.clear();
        self.id_count = 0;
        self.id_map_count = 0;
        self.last_signature = None;
        self.last_global_loop_closure_id = 0;
        self.all_nodes_in_wm = true;
        self.odom_max_inf.clear();
        self.rehearsal = (0, 0.0, 0);
    }

    pub fn sig(&self, id: i32) -> Option<&Signature> {
        self.signatures.get(&id)
    }

    /// getWorkingMemSize(ignoreIntermediateNodes): 가상 장소 뺌
    pub fn wm_size(&self, ignore_intermediate: bool) -> usize {
        let n = self.wm.len() - 1;
        if ignore_intermediate {
            n - self.wm.iter().filter(|id| **id > 0 && self.signatures.get(id).map(|s| s.weight == -1).unwrap_or(false)).count()
        } else {
            n
        }
    }

    pub fn last_working_signature(&self) -> Option<&Signature> {
        self.last_signature.and_then(|id| self.signatures.get(&id))
    }

    /// Memory::cleanUnusedWords(preUpdate)
    fn clean_unused_words(&mut self) {
        let ids: Vec<i32> = self.dict.unused.iter().copied().collect();
        if !ids.is_empty() {
            self.dict.remove_words(&ids);
        }
    }

    /// Memory::update(data, pose, covariance, velocity)
    pub fn update(&mut self, f: &NodeFeatures, stamp: f64, pose: Tf, cov: &Mat6, velocity: &[f32], image_size: (u32, u32)) -> i32 {
        self.rehearsal = (0, 0.0, 0);
        self.clean_unused_words();
        let id = self.create_signature(f, stamp, pose, image_size);
        if velocity.len() == 6 {
            self.signatures.get_mut(&id).unwrap().velocity = velocity.to_vec();
        }
        self.add_signature_to_stm(id, cov);
        self.last_signature = Some(id);
        if self.p.rehearsal_similarity < 1.0 {
            self.do_rehearsal(id);
        }
        let mut not_intermediate = self.stm.iter().filter(|i| self.signatures[i].weight >= 0).count();
        while !self.stm.is_empty() && self.p.stm_size > 0 && not_intermediate > self.p.stm_size {
            let first = *self.stm.iter().next().unwrap();
            if self.signatures[&first].weight >= 0 {
                not_intermediate -= 1;
            }
            self.wm.insert(first);
            self.stm.remove(&first);
        }
        id
    }

    /// createSignature: 위치 추적 특징 경로(Mem/UseOdomFeatures)
    fn create_signature(&mut self, f: &NodeFeatures, stamp: f64, pose: Tf, image_size: (u32, u32)) -> i32 {
        self.id_count += 1;
        let id = self.id_count;
        let mut kps = f.kps.clone();
        let mut p3 = f.pts3d.clone();
        let mut desc = f.desc.clone();
        let kmax = self.p.kp_max_features;
        let vmax = self.p.vis_max_features;
        // Mem/RawDescriptorsKept 이고 위치가 있으면 Vis/MaxFeatures 까지 둔다
        let max_features = if !pose.is_null() && kmax > 0 && kmax < vmax { vmax } else { kmax };
        let _ = image_size;
        if max_features > 0 && kps.len() > max_features as usize {
            let order = response_order(&kps);
            let keep: Vec<usize> = order.into_iter().take(max_features as usize).collect();
            kps = keep.iter().map(|&i| kps[i]).collect();
            if !p3.is_empty() {
                p3 = keep.iter().map(|&i| p3[i]).collect();
            }
            desc = keep.iter().map(|&i| desc[i]).collect();
        }
        // 사전 갱신(preUpdate 스레드, 단어 더하기 전에 끝남)
        self.dict.update();
        let mut word_ids: Vec<i32> = Vec::new();
        if !kps.is_empty() {
            if !desc.is_empty() {
                // meanWordsPerLocation = 0(위치 추적 특징) 이라 나쁜 노드 거르기 없음
                let mut q: Vec<[u8; 32]> = desc.clone();
                let mut q2raw: Vec<usize> = Vec::new();
                if kmax > 0 && desc.len() > kmax as usize {
                    let inl = limit_inliers(&kps, kmax as usize);
                    q.clear();
                    for (k, ok) in inl.iter().enumerate() {
                        if *ok {
                            q.push(desc[k]);
                            q2raw.push(k);
                        }
                    }
                }
                let ids = self.dict.add_new_words(&q, id);
                if ids.len() < kps.len() {
                    let mut all = vec![-1i32; kps.len()];
                    for (i, w) in ids.iter().enumerate() {
                        all[q2raw[i]] = *w;
                    }
                    let mut neg = -1;
                    for w in all.iter_mut() {
                        if *w < 0 {
                            *w = neg;
                            neg -= 1;
                        }
                    }
                    word_ids = all;
                } else {
                    word_ids = ids;
                }
            } else {
                word_ids = (0..kps.len() as i32).map(|i| -1 - i).collect();
            }
        }
        let mut words: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let mut invalid = 0;
        for (i, w) in word_ids.iter().enumerate() {
            words.entry(*w).or_default().push(i as i32);
            if *w <= 0 {
                invalid += 1;
            }
        }
        let s = Signature {
            id,
            map_id: self.id_map_count,
            weight: 0,
            stamp,
            pose,
            links: BTreeMap::new(),
            words,
            words_count: word_ids.len(),
            invalid_words: invalid,
            kpts: kps,
            pts3d: p3,
            desc,
            enabled: !word_ids.is_empty(),
            velocity: Vec::new(),
        };
        self.signatures.insert(id, s);
        id
    }

    /// addSignatureToStm: 앞 노드와 이웃 링크(정보 = 공분산 대각의 역수)
    fn add_signature_to_stm(&mut self, id: i32, cov: &Mat6) {
        if let Some(&last) = self.stm.iter().next_back() {
            let (lmap, lpose) = {
                let l = &self.signatures[&last];
                (l.map_id, l.pose)
            };
            let (smap, spose) = {
                let s = &self.signatures[&id];
                (s.map_id, s.pose)
            };
            if lmap == smap {
                if !spose.is_null() && !lpose.is_null() {
                    let mut inf = [0.0; 36];
                    for i in 0..6 {
                        inf[i * 7] = 1.0 / cov[i * 7];
                    }
                    if cov[0].is_finite() && cov[0] > 0.0 && !(inf[0].is_finite() && inf[0] > 0.0) {
                        inf = eye6(1.0);
                    }
                    if self.odom_max_inf.is_empty() {
                        self.odom_max_inf = vec![0.0; 6];
                    }
                    for i in 0..6 {
                        if self.odom_max_inf[i] < inf[i * 7] {
                            self.odom_max_inf[i] = inf[i * 7];
                        }
                    }
                    let motion = lpose.inverse().mul(&spose);
                    self.signatures.get_mut(&last).unwrap().add_link(Link::new(last, id, LinkType::Neighbor, motion, inf));
                    self.signatures.get_mut(&id).unwrap().add_link(Link::new(id, last, LinkType::Neighbor, motion.inverse(), inf));
                } else {
                    self.signatures.get_mut(&last).unwrap().add_link(Link::new(last, id, LinkType::Neighbor, Tf::NULL, eye6(1.0)));
                    self.signatures.get_mut(&id).unwrap().add_link(Link::new(id, last, LinkType::Neighbor, Tf::NULL, eye6(1.0)));
                }
            }
        }
        self.stm.insert(id);
        let s = self.signatures.get_mut(&id).unwrap();
        if s.words_count > 0 {
            s.enabled = true;
        }
    }

    /// Memory::rehearsal: STM 의 바로 앞 노드와 닮음이 0.6 이상이면 합침(또는 무게만)
    fn do_rehearsal(&mut self, id: i32) {
        if self.signatures[&id].is_bad() {
            return;
        }
        let mut sb = None;
        for i in self.stm.iter().rev() {
            let s = &self.signatures[i];
            if s.weight >= 0 && s.id != id {
                sb = Some(*i);
                break;
            }
        }
        let Some(old) = sb else {
            return;
        };
        let sim = self.signatures[&id].compare_to(&self.signatures[&old]);
        let mut merged = 0;
        if sim >= self.p.rehearsal_similarity && self.rehearsal_merge(old, id) {
            merged = old;
        }
        self.rehearsal = (merged, sim, if sim >= self.p.rehearsal_similarity { old } else { 0 });
    }

    /// rehearsalMerge(Mem/RehearsalIdUpdatedToNewOne=false)
    fn rehearsal_merge(&mut self, old: i32, new: i32) -> bool {
        if let Some(l) = self.signatures[&old].links.get(&new) {
            if l.ty != LinkType::Neighbor && l.ty != LinkType::NeighborMerged && l.from != l.to {
                return false;
            }
        }
        let (full, intermediate) = {
            let ns = &self.signatures[&new];
            if let Some((_, l)) = ns.links.iter().next().filter(|(_, l)| !l.t.is_null()) {
                let r = l.t.xyz_rpy();
                let d = self.p.linear_update;
                let a = self.p.angular_update;
                let moving = r[0].abs() > d || r[1].abs() > d || r[2].abs() > d || r[3].abs() > a || r[4].abs() > a || r[5].abs() > a;
                (!moving && ns.has_link(old), !moving && !ns.has_link(old))
            } else {
                let first_null = ns.links.values().next().map(|l| l.t.is_null()).unwrap_or(false);
                (ns.has_link(old) && first_null, false)
            }
        };
        if full {
            self.signatures.get_mut(&old).unwrap().links.remove(&new);
            self.signatures.get_mut(&new).unwrap().links.remove(&old);
            self.signatures.get_mut(&new).unwrap().add_link(Link::new(new, old, LinkType::GlobalClosure, Tf::NULL, eye6(1.0)));
            let nw = self.signatures[&new].weight;
            let ow = self.signatures[&old].weight;
            self.signatures.get_mut(&old).unwrap().weight = nw + 1 + ow;
            if self.last_signature == Some(new) {
                self.last_signature = Some(old);
            }
            self.signatures.get_mut(&new).unwrap().weight = -9;
            // moveToTrash(newS, keepLinkedToGraph = Mem/NotLinkedNodesKept = true): 데이터베이스로 감
            self.move_to_trash(new, true);
            return true;
        }
        let w = self.signatures[&new].weight.max(0);
        let o = self.signatures.get_mut(&old).unwrap();
        o.weight = w + o.weight + 1;
        if intermediate {
            // convertToIntermediate: 우리 경로에서 나온 적 없음(덤프 0 회)
            let s = self.signatures.get_mut(&new).unwrap();
            s.weight = -1;
        } else {
            self.signatures.get_mut(&new).unwrap().weight = 0;
        }
        false
    }

    /// disableWordsRef
    fn disable_words_ref(&mut self, id: i32) {
        let s = self.signatures.get_mut(&id).unwrap();
        if !s.enabled {
            return;
        }
        let keys: Vec<i32> = s.words.keys().copied().collect();
        s.enabled = false;
        for k in keys {
            self.dict.remove_all_word_ref(k, id);
        }
    }

    /// moveToTrash(s, keepLinkedToGraph)
    pub fn move_to_trash(&mut self, id: i32, keep_linked: bool) {
        if !keep_linked {
            let links: Vec<Link> = self.signatures[&id].links.values().cloned().collect();
            let w = self.signatures[&id].weight;
            for l in links {
                if l.from != l.to && l.to > 0 {
                    let to = self.signatures.get_mut(&l.to).unwrap();
                    if l.ty == LinkType::GlobalClosure && w > 0 {
                        to.weight += w;
                    }
                    to.links.remove(&id);
                }
            }
            let s = self.signatures.get_mut(&id).unwrap();
            s.links.retain(|_, l| l.from == l.to);
            s.weight = -9;
        }
        self.disable_words_ref(id);
        if !keep_linked {
            let keys: Vec<i32> = self.signatures[&id].words.keys().copied().collect();
            for k in keys {
                if self.dict.unused.contains(&k) {
                    self.dict.remove_words(&[k]);
                }
            }
        }
        self.wm.remove(&id);
        self.stm.remove(&id);
        self.signatures.remove(&id);
        if self.last_signature == Some(id) {
            self.last_signature = self.stm.iter().next_back().copied().or_else(|| self.wm.iter().next_back().copied().filter(|i| *i > 0));
        }
        if self.last_global_loop_closure_id == id {
            self.last_global_loop_closure_id = 0;
        }
        if keep_linked {
            // 데이터베이스에 남긴 노드가 생김
            self.all_nodes_in_wm = false;
        }
    }

    /// computeLikelihood(tf-idf)
    pub fn compute_likelihood(&self, id: i32, ids: &[i32]) -> BTreeMap<i32, f32> {
        let mut lik: BTreeMap<i32, f32> = ids.iter().map(|i| (*i, 0.0f32)).collect();
        let s = &self.signatures[&id];
        let n = self.signatures.len() as f32;
        if n != 0.0 {
            for w in s.words.keys() {
                if *w <= 0 {
                    continue;
                }
                let vw = &self.dict.words[w];
                let nw = vw.refs.len() as f32;
                if nw != 0.0 {
                    let lognnw = (n / nw).log10();
                    if lognnw != 0.0 {
                        for (sid, cnt) in &vw.refs {
                            if let Some(v) = lik.get_mut(sid) {
                                let nwi = *cnt as f32;
                                let ni = self.signatures.get(sid).map(|s| s.words_count).unwrap_or(0) as f32;
                                if ni != 0.0 {
                                    *v += (nwi * lognnw) / ni;
                                }
                            }
                        }
                    }
                }
            }
        }
        lik
    }

    /// getNeighborsId(id, maxDepth, 0, incrementMarginOnLoop, ignoreLoopIds, ignoreIntermediateNodes, ignoreLocalSpaceLoopIds, nodesSet)
    pub fn neighbors_id(&self, id: i32, max_depth: i32, inc_margin_on_loop: bool, ignore_loop: bool, ignore_inter: bool, ignore_local_space: bool, nodes: Option<&BTreeSet<i32>>) -> BTreeMap<i32, i32> {
        let mut ids: BTreeMap<i32, i32> = BTreeMap::new();
        if id <= 0 {
            return ids;
        }
        let mut next: BTreeSet<i32> = BTreeSet::new();
        next.insert(id);
        let mut current: BTreeSet<i32> = BTreeSet::new();
        let mut ignored: BTreeSet<i32> = BTreeSet::new();
        let mut m = 0;
        while (max_depth == 0 || m < max_depth) && !next.is_empty() {
            let mut list: Vec<i32> = next.iter().rev().copied().collect();
            next.clear();
            let mut k = 0;
            while k < list.len() {
                let j = list[k];
                k += 1;
                if ids.contains_key(&j) || nodes.map(|n| !n.contains(&j)).unwrap_or(false) {
                    continue;
                }
                let Some(s) = self.signatures.get(&j) else {
                    continue;
                };
                let inter = s.weight == -1;
                if !ignore_inter || !inter {
                    let eff = if m > 0 && inter { m - 1 } else { m };
                    ids.insert(j, eff);
                } else {
                    ignored.insert(j);
                }
                for (to, l) in &s.links {
                    if ids.contains_key(to) || ignored.contains(to) {
                        continue;
                    }
                    if l.ty == LinkType::Neighbor || l.ty == LinkType::NeighborMerged {
                        if inter {
                            if current.insert(*to) {
                                list.push(*to);
                            }
                        } else {
                            next.insert(*to);
                        }
                    } else if !ignore_loop && (!ignore_local_space || l.ty != LinkType::LocalSpaceClosure) {
                        if inc_margin_on_loop {
                            next.insert(*to);
                        } else if current.insert(*to) {
                            list.push(*to);
                        }
                    }
                }
            }
            m += 1;
        }
        ids
    }

    /// addLink(루프·근접): 새 노드 쪽으로 무게를 몰아줌(Mem/ReduceGraph=false)
    pub fn add_link(&mut self, link: Link) -> bool {
        if !self.signatures.contains_key(&link.to) || !self.signatures.contains_key(&link.from) {
            return false;
        }
        if self.signatures[&link.to].has_link(link.from) {
            return true;
        }
        let (from, to, ty) = (link.from, link.to, link.ty);
        self.signatures.get_mut(&to).unwrap().add_link(link.inverse());
        self.signatures.get_mut(&from).unwrap().add_link(link);
        if ty != LinkType::VirtualClosure {
            self.last_global_loop_closure_id = from.max(to);
            let fw = self.signatures[&from].weight;
            let tw = self.signatures[&to].weight;
            if from > to {
                self.signatures.get_mut(&from).unwrap().weight = fw + tw;
                self.signatures.get_mut(&to).unwrap().weight = 0;
            } else {
                self.signatures.get_mut(&to).unwrap().weight = tw + fw;
                self.signatures.get_mut(&from).unwrap().weight = 0;
            }
        }
        true
    }

    /// removeLink(oldId, newId)
    pub fn remove_link(&mut self, a: i32, b: i32) {
        let (o, n) = if a < b { (a, b) } else { (b, a) };
        if !self.signatures.contains_key(&o) || !self.signatures.contains_key(&n) {
            return;
        }
        if self.signatures[&o].has_link(n) && self.signatures[&n].has_link(o) {
            let ty = self.signatures[&o].links[&n].ty;
            if ty == LinkType::GlobalClosure && self.signatures[&n].weight > 0 {
                self.signatures.get_mut(&o).unwrap().weight += 1;
                let ns = self.signatures.get_mut(&n).unwrap();
                ns.weight = if ns.weight > 0 { ns.weight - 1 } else { 0 };
            }
            self.signatures.get_mut(&o).unwrap().links.remove(&n);
            self.signatures.get_mut(&n).unwrap().links.remove(&o);
            let ns = &self.signatures[&n];
            let no_children = !ns.links.iter().any(|(k, l)| l.ty != LinkType::Neighbor && l.ty != LinkType::NeighborMerged && l.from != l.to && *k < n);
            if no_children && n == self.last_global_loop_closure_id {
                self.last_global_loop_closure_id = 0;
            }
        }
    }

    /// getLoopClosureLinks 가 비었는지
    pub fn has_loop_closure_links(&self, id: i32) -> bool {
        self.signatures
            .get(&id)
            .map(|s| s.links.values().any(|l| l.ty != LinkType::Neighbor && l.ty != LinkType::NeighborMerged && l.from != l.to && l.ty != LinkType::Undef))
            .unwrap_or(false)
    }

    /// getMetricConstraints(ids, lookInDatabase=false, landmarksAdded=true): 중간 노드 없는 경로
    pub fn metric_constraints(&self, ids: &BTreeSet<i32>) -> (BTreeMap<i32, Tf>, LinkMap) {
        let mut poses = BTreeMap::new();
        for id in ids {
            if let Some(s) = self.signatures.get(id) {
                if !s.pose.is_null() {
                    poses.insert(*id, s.pose);
                }
            }
        }
        let mut links = LinkMap::default();
        for id in ids {
            if !poses.contains_key(id) {
                continue;
            }
            let s = &self.signatures[id];
            if s.weight == -1 {
                poses.remove(id);
                continue;
            }
            for (to, l) in &s.links {
                let found = links.find(*id, *to, true, LinkType::Undef);
                let free = match found {
                    None => true,
                    Some(at) => {
                        let f = links.get(at);
                        f.from == f.to
                    }
                };
                if l.is_valid() && free && poses.contains_key(to) && l.ty != LinkType::Landmark {
                    links.insert(*id, l.clone());
                }
            }
        }
        (poses, links)
    }

    /// incrementMapId: 지금 지도에 노드가 있으면 STM 을 모두 WM 으로 옮기고 지도 번호 +1
    pub fn increment_map_id(&mut self) -> i32 {
        if let Some(s) = self.last_working_signature() {
            if s.map_id == self.id_map_count {
                while let Some(&first) = self.stm.iter().next() {
                    self.wm.insert(first);
                    self.stm.remove(&first);
                }
                self.id_map_count += 1;
                return self.id_map_count;
            }
        }
        self.id_map_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv6_diagonal_is_reciprocal() {
        let mut c = [0.0; 36];
        for i in 0..6 {
            c[i * 7] = 0.001 * (i + 1) as f64;
        }
        let r = inv6(&c);
        for i in 0..6 {
            assert_eq!(r[i * 7], 1.0 / c[i * 7]);
        }
    }

    #[test]
    fn response_order_ties() {
        let k = |r: f32| Kp { response: r, ..Default::default() };
        let kps = vec![k(1.0), k(3.0), k(3.0), k(2.0)];
        assert_eq!(response_order(&kps), vec![2, 1, 3, 0]);
    }
}
