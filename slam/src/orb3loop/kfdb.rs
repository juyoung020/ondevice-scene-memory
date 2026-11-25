//! ORB-SLAM3 KeyFrameDatabase(역색인 · 루프/합치기 후보 · 재위치 후보)를 비트 동일하게 옮긴 것.
//! 원본: src/KeyFrameDatabase.cc(add · erase · clear · clearMap · DetectNBestCandidates ·
//! DetectRelocalizationCandidates) + ORB_SLAM3.patch(재위치: 부를 때마다 새 표지 1<<40 부터, 다른 지도는 단어 세기부터 뺌).
//!
//! 원본은 표지 · 단어 수 · 점수를 KeyFrame 멤버(mnPlaceRecognitionQuery/Words/Score, mnRelocQuery/Words/Score)에
//! 남기고 다음 질의에서도 읽는다(공시성 이웃이 이번 질의 목록엔 있으나 점수 문턱을 못 넘으면 지난 질의 점수가 더해짐).
//! 같은 결과를 내려고 그 상태를 키프레임 번호별로 여기 보관한다. 원본 mRelocScore 는 생성자에서 초기화하지 않아
//! 처음 값이 정의되지 않음 — 여기선 0(하네스 가짜 KeyFrame 도 0).
//!
//! 지도 · 공시성(covisibility) 그래프는 다른 몫(공통 자료 구조)이라 `KfGraph` 트레이트로 받는다.
//! GetBestCovisibilityKeyFrames 순서(무게 내림차순, 같은 무게는 원본이 포인터 순)도 그쪽이 준 그대로 쓴다.

use super::dbow2::{score_l1, BowVector, Vocabulary};

pub type KfId = u32;
pub type MapId = u32;

/// 키프레임 · 지도 상태 조회(원본 KeyFrame/Map 메서드)
pub trait KfGraph {
    fn bow(&self, kf: KfId) -> &BowVector;
    fn map_of(&self, kf: KfId) -> MapId;
    fn is_bad(&self, kf: KfId) -> bool;
    fn map_is_bad(&self, m: MapId) -> bool;
    /// GetConnectedKeyFrames (집합: 포함 여부만 씀)
    fn is_connected(&self, kf: KfId, other: KfId) -> bool;
    /// GetBestCovisibilityKeyFrames(n)
    fn best_covis(&self, kf: KfId, n: usize) -> &[KfId];
    /// 원본 KeyFrame::mnId (질의 표지 = 질의 키프레임 mnId)
    fn mn_id(&self, kf: KfId) -> u64 {
        kf as u64
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct QueryState {
    pub query: u64,
    pub words: i32,
    pub score: f32,
}

pub struct KeyFrameDatabase {
    inv: Vec<Vec<KfId>>,
    /// mnPlaceRecognitionQuery/Words/Score
    pub place: Vec<QueryState>,
    /// mnRelocQuery/Words/Score
    pub reloc: Vec<QueryState>,
    reloc_q: u64,
}

fn st(v: &mut Vec<QueryState>, k: KfId) -> &mut QueryState {
    let k = k as usize;
    if v.len() <= k {
        v.resize(k + 1, QueryState::default());
    }
    &mut v[k]
}

fn get(v: &[QueryState], k: KfId) -> QueryState {
    v.get(k as usize).copied().unwrap_or_default()
}

impl KeyFrameDatabase {
    pub fn new(voc: &Vocabulary) -> Self {
        Self::with_words(voc.size())
    }

    pub fn with_words(n: usize) -> Self {
        KeyFrameDatabase { inv: vec![Vec::new(); n], place: vec![], reloc: vec![], reloc_q: 1u64 << 40 }
    }

    pub fn add(&mut self, kf: KfId, bow: &BowVector) {
        for &(w, _) in bow {
            self.inv[w as usize].push(kf);
        }
    }

    /// erase: 단어마다 처음 만난 것 하나만 지움
    pub fn erase(&mut self, kf: KfId, bow: &BowVector) {
        for &(w, _) in bow {
            let l = &mut self.inv[w as usize];
            if let Some(p) = l.iter().position(|&x| x == kf) {
                l.remove(p);
            }
        }
    }

    pub fn clear(&mut self) {
        let n = self.inv.len();
        self.inv = vec![Vec::new(); n];
    }

    pub fn clear_map<G: KfGraph>(&mut self, g: &G, m: MapId) {
        for l in self.inv.iter_mut() {
            l.retain(|&k| g.map_of(k) != m);
        }
    }

    /// DetectNBestCandidates(pKF, vpLoopCand, vpMergeCand, nNumCandidates) → (루프 후보, 합치기 후보)
    pub fn detect_n_best<G: KfGraph>(&mut self, g: &G, kf: KfId, n_cand: usize) -> (Vec<KfId>, Vec<KfId>) {
        let qid = g.mn_id(kf);
        let mut sharing: Vec<KfId> = Vec::new();
        for &(w, _) in g.bow(kf) {
            for &ki in &self.inv[w as usize] {
                let s = st(&mut self.place, ki);
                if s.query != qid {
                    s.words = 0;
                    if !g.is_connected(kf, ki) {
                        s.query = qid;
                        // 결정성 고침(원본 버그): 이번 질의에서 점수를 안 매긴 이웃의 지난 점수를 더하지 않게 0 으로
                        s.score = 0.0;
                        sharing.push(ki);
                    }
                }
                s.words += 1;
            }
        }
        let (mut lc, mut mc) = (Vec::new(), Vec::new());
        if sharing.is_empty() {
            return (lc, mc);
        }
        let mut max_common = 0i32;
        for &k in &sharing {
            max_common = max_common.max(get(&self.place, k).words);
        }
        let min_common = (max_common as f32 * 0.8f32) as i32;
        let mut score_match: Vec<(f32, KfId)> = Vec::new();
        for &k in &sharing {
            if get(&self.place, k).words > min_common {
                let si = score_l1(g.bow(kf), g.bow(k)) as f32;
                st(&mut self.place, k).score = si;
                score_match.push((si, k));
            }
        }
        if score_match.is_empty() {
            return (lc, mc);
        }
        let mut acc: Vec<(f32, KfId)> = Vec::with_capacity(score_match.len());
        for &(s0, k) in &score_match {
            let (mut best, mut accs, mut best_kf) = (s0, s0, k);
            for &k2 in g.best_covis(k, 10) {
                let s2 = get(&self.place, k2);
                if s2.query != qid {
                    continue;
                }
                accs += s2.score;
                if s2.score > best {
                    best_kf = k2;
                    best = s2.score;
                }
            }
            acc.push((accs, best_kf));
        }
        // list::sort(compFirst): 안정 정렬, 큰 점수 먼저
        acc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let my_map = g.map_of(kf);
        let mut added: Vec<KfId> = Vec::new();
        for &(_, ki) in &acc {
            if !(lc.len() < n_cand || mc.len() < n_cand) {
                break;
            }
            // 원본은 isBad 면 i · it 를 올리지 않고 continue(무한 반복). 역색인엔 나쁜 키프레임이 없어 닿지 않는 길
            assert!(!g.is_bad(ki), "원본에선 무한 반복이 되는 경우");
            if !added.contains(&ki) {
                let m = g.map_of(ki);
                if m == my_map && lc.len() < n_cand {
                    lc.push(ki);
                } else if m != my_map && mc.len() < n_cand && !g.map_is_bad(m) {
                    mc.push(ki);
                }
                added.push(ki);
            }
        }
        (lc, mc)
    }

    /// DetectRelocalizationCandidates(F, pMap) (패치 적용본)
    pub fn detect_reloc<G: KfGraph>(&mut self, g: &G, bow: &BowVector, map: MapId) -> Vec<KfId> {
        let q = self.reloc_q;
        self.reloc_q += 1;
        let mut sharing: Vec<KfId> = Vec::new();
        for &(w, _) in bow {
            for &ki in &self.inv[w as usize] {
                let s = st(&mut self.reloc, ki);
                if s.query != q {
                    if g.map_of(ki) != map {
                        continue;
                    }
                    s.words = 0;
                    s.query = q;
                    s.score = 0.0; // 결정성 고침(원본 버그, detect_n_best 와 같음)
                    sharing.push(ki);
                }
                s.words += 1;
            }
        }
        if sharing.is_empty() {
            return vec![];
        }
        let mut max_common = 0i32;
        for &k in &sharing {
            max_common = max_common.max(get(&self.reloc, k).words);
        }
        let min_common = (max_common as f32 * 0.8f32) as i32;
        let mut score_match: Vec<(f32, KfId)> = Vec::new();
        for &k in &sharing {
            if get(&self.reloc, k).words > min_common {
                let si = score_l1(bow, g.bow(k)) as f32;
                st(&mut self.reloc, k).score = si;
                score_match.push((si, k));
            }
        }
        if score_match.is_empty() {
            return vec![];
        }
        let mut acc: Vec<(f32, KfId)> = Vec::with_capacity(score_match.len());
        let mut best_acc = 0f32;
        for &(s0, k) in &score_match {
            let (mut best, mut accs, mut best_kf) = (s0, s0, k);
            for &k2 in g.best_covis(k, 10) {
                let s2 = get(&self.reloc, k2);
                if s2.query != q {
                    continue;
                }
                accs += s2.score;
                if s2.score > best {
                    best_kf = k2;
                    best = s2.score;
                }
            }
            acc.push((accs, best_kf));
            if accs > best_acc {
                best_acc = accs;
            }
        }
        let min_keep = 0.75f32 * best_acc;
        let mut out: Vec<KfId> = Vec::new();
        for &(s, k) in &acc {
            if s > min_keep {
                if g.map_of(k) != map {
                    continue;
                }
                if !out.contains(&k) {
                    out.push(k);
                }
            }
        }
        out
    }
}
