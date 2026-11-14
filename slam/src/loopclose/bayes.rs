//! 루프 클로저 가설의 베이즈 필터. 원본: corelib/src/BayesFilter.cpp(computePosterior · updatePosterior),
//! bayes/PredictionModel.cpp(set · addNeighborProb · normalize · fillVirtualPlaceColumn · resolveNeighbors · cachedNeighbors),
//! bayes/SparsePrediction.cpp(generate · update(Appended/Remapped) · multiply).
//! Bayes/PredictionLC 기본값의 합이 float 로 정확히 1 이라 Bayes/SparsePrediction=true 의 희소 예측만 쓴다(조밀 판은 합이 1 미만일 때만).
//! 희소 예측의 열 저장 방식(용량·압축)은 곱 결과에 영향이 없어 열마다 (행, 값) 목록으로 둔다. 곱 순서(열 순서대로 더함)는 원본 그대로.
use super::mem::Memory;
use std::collections::{BTreeMap, BTreeSet};

pub struct PredictionModel {
    pub values: Vec<f64>,
    pub total: f32,
    pub epsilon: f32,
    pub virtual_place_prior: f32,
}

impl PredictionModel {
    /// PredictionModel::set: 값은 uStr2Float(float) 로 읽어 double 에 담음
    pub fn new(prediction: &str, virtual_place_prior: f32) -> PredictionModel {
        let values: Vec<f64> = prediction.split_whitespace().map(|s| s.parse::<f32>().unwrap_or(0.0) as f64).collect();
        let mut total = 0.0f32;
        let mut eps = 0.0f32;
        for (j, v) in values.iter().enumerate() {
            total = (total as f64 + v) as f32;
            if j == 0 || *v < eps as f64 {
                eps = *v as f32;
            }
        }
        PredictionModel { values, total, epsilon: eps, virtual_place_prior }
    }
    pub fn depth(&self) -> i32 {
        self.values.len() as i32 - 1
    }

    fn add_neighbor_prob(&self, col: &mut [f32], neighbors: &BTreeMap<i32, i32>, idx: &BTreeMap<i32, usize>) -> f32 {
        let mut sum = 0.0f32;
        for (id, margin) in neighbors {
            if *id >= 0 {
                if let Some(&j) = idx.get(id) {
                    let v = self.values[(*margin + 1) as usize] as f32;
                    col[j] = v;
                    sum += v;
                }
            }
        }
        sum
    }

    fn normalize(&self, col: &mut [f32], index: usize, mut added: f32, virtual_used: bool) {
        let cols = col.len();
        if (added as f64) < self.total as f64 - self.values[0] {
            let delta = (self.total as f64 - self.values[0] - added as f64) as f32;
            col[index] += delta;
            added += delta;
        }
        let mut all_other = 0.0f32;
        if self.total < 1.0 {
            all_other = 1.0 - self.total;
        }
        if all_other > 0.0 && cols > 1 {
            let value = all_other / (cols - 1) as f32;
            for j in (if virtual_used { 1 } else { 0 })..cols {
                if col[j] == 0.0 {
                    col[j] = value;
                    added += col[j];
                }
            }
        }
        let max_norm = (1.0 - if virtual_used { self.values[0] } else { 0.0 }) as f32;
        if (added as f64) < max_norm as f64 - 0.0001 || (added as f64) > max_norm as f64 + 0.0001 {
            for j in (if virtual_used { 1 } else { 0 })..cols {
                col[j] *= max_norm / added;
                if col[j] < self.epsilon {
                    col[j] = 0.0;
                }
            }
        }
        if virtual_used {
            col[0] = self.values[0] as f32;
        }
    }

    fn fill_virtual(&self, col: &mut [f32]) {
        let size = col.len();
        if self.virtual_place_prior > 0.0 {
            if size > 1 {
                col[0] = self.virtual_place_prior;
                let val = ((1.0 - self.virtual_place_prior as f64) / (size - 1) as f64) as f32;
                for c in col.iter_mut().skip(1) {
                    *c = val;
                }
            } else if size > 0 {
                col[0] = 1.0;
            }
        } else if size > 1 {
            let val = (1.0 / size as f64) as f32;
            for c in col.iter_mut() {
                *c = val;
            }
        } else if size > 0 {
            col[0] = 1.0;
        }
    }
}

type Cache = BTreeMap<i32, BTreeMap<i32, i32>>;

fn neighbors_of(mem: &Memory, id: i32, depth: i32) -> BTreeMap<i32, i32> {
    mem.neighbors_id(id, depth, false, false, true, true, None)
}

/// cachedNeighbors
fn cached_neighbors(mem: &Memory, id: i32, depth: i32, cache: &mut Cache) -> BTreeMap<i32, i32> {
    if !cache.contains_key(&id) {
        let n = neighbors_of(mem, id, depth);
        for (k, m) in &n {
            if let Some(c) = cache.get_mut(k) {
                c.insert(id, *m);
            }
        }
        cache.insert(id, n);
    }
    cache[&id].clone()
}

pub struct BayesFilter {
    pub model: PredictionModel,
    /// 희소 예측: 열마다 0 아닌 (행, 값)
    columns: Vec<Vec<(usize, f32)>>,
    sparse_ids: Vec<i32>,
    cache: Cache,
    prediction_changed: bool,
    pub posterior_ids: Vec<i32>,
    pub posterior_values: Vec<f32>,
}

impl BayesFilter {
    pub fn new(model: PredictionModel) -> BayesFilter {
        assert!(model.total >= 1.0, "Bayes/PredictionLC 합이 1 미만(조밀 예측)은 옮기지 않았다");
        BayesFilter { model, columns: Vec::new(), sparse_ids: Vec::new(), cache: Cache::new(), prediction_changed: true, posterior_ids: Vec::new(), posterior_values: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.posterior_ids.clear();
        self.posterior_values.clear();
        self.columns.clear();
        self.sparse_ids.clear();
        self.prediction_changed = true;
        self.cache.clear();
    }

    fn take_column(&mut self, col: &mut [f32], index: usize) {
        let mut v = Vec::new();
        for (r, c) in col.iter_mut().enumerate() {
            if *c != 0.0 {
                v.push((r, *c));
                *c = 0.0;
            }
        }
        self.columns[index] = v;
    }

    fn build_column(&mut self, mem: &Memory, id: i32, index: usize, ids: &[i32], idx: &BTreeMap<i32, usize>, buf: &mut [f32]) {
        let n = cached_neighbors(mem, id, self.model.depth(), &mut self.cache);
        let sum = self.model.add_neighbor_prob(buf, &n, idx);
        self.model.normalize(buf, index, sum, ids[0] < 0);
        self.take_column(buf, index);
    }

    /// SparsePrediction::generate
    fn generate(&mut self, mem: &Memory, ids: &[i32]) {
        let size = ids.len();
        let idx: BTreeMap<i32, usize> = ids.iter().enumerate().filter(|(_, id)| **id > 0).map(|(i, id)| (*id, i)).collect();
        self.columns = vec![Vec::new(); size];
        let mut col = vec![0.0f32; size];
        let mut done: BTreeSet<i32> = BTreeSet::new();
        for i in 0..size {
            if done.contains(&ids[i]) {
                continue;
            }
            if ids[i] > 0 {
                // resolveNeighbors
                let full = neighbors_of(mem, ids[i], self.model.depth());
                self.cache.insert(ids[i], full.clone());
                let mut n = full.clone();
                let mut margin0 = Vec::new();
                n.retain(|k, m| {
                    if mem.stm.contains(k) {
                        false
                    } else {
                        if *m == 0 && idx.contains_key(k) {
                            margin0.push(*k);
                        }
                        true
                    }
                });
                assert!(!margin0.is_empty(), "No 0 margin neighbor for signature {}", ids[i]);
                for id in margin0 {
                    self.cache.insert(id, full.clone());
                    let index = idx[&id];
                    let sum = self.model.add_neighbor_prob(&mut col, &n, &idx);
                    self.model.normalize(&mut col, index, sum, ids[0] < 0);
                    self.take_column(&mut col, index);
                    done.insert(id);
                }
            } else {
                self.model.fill_virtual(&mut col);
                self.take_column(&mut col, i);
            }
        }
        self.sparse_ids = ids.to_vec();
    }

    fn update(&mut self, mem: &Memory, new_ids: &[i32]) -> bool {
        if self.sparse_ids.is_empty() || new_ids.is_empty() || self.columns.len() != self.sparse_ids.len() {
            return false;
        }
        let old = self.sparse_ids.clone();
        let appended = new_ids.len() > old.len() && new_ids[..old.len()] == old[..];
        let size = new_ids.len();
        let idx: BTreeMap<i32, usize> = new_ids.iter().enumerate().filter(|(_, id)| **id > 0).map(|(i, id)| (*id, i)).collect();
        let mut col = vec![0.0f32; size];
        if appended {
            self.columns.resize(size, Vec::new());
            let mut to_update: BTreeSet<i32> = BTreeSet::new();
            for i in old.len()..size {
                assert!(new_ids[i] > 0);
                let n = cached_neighbors(mem, new_ids[i], self.model.depth(), &mut self.cache);
                let sum = self.model.add_neighbor_prob(&mut col, &n, &idx);
                self.model.normalize(&mut col, i, sum, new_ids[0] < 0);
                self.take_column(&mut col, i);
                for k in n.keys() {
                    if let Some(&j) = idx.get(k) {
                        if j < old.len() {
                            to_update.insert(*k);
                        }
                    }
                }
            }
            for id in to_update {
                self.build_column(mem, id, idx[&id], new_ids, &idx, &mut col);
            }
        } else {
            if (old[0] < 0) != (new_ids[0] < 0) {
                return false;
            }
            let old_idx: BTreeMap<i32, usize> = old.iter().enumerate().filter(|(_, id)| **id > 0).map(|(i, id)| (*id, i)).collect();
            let mut old_to_new: Vec<i64> = vec![-1; old.len()];
            let mut removed = 0;
            for i in 0..old.len() {
                if old[i] <= 0 {
                    old_to_new[i] = 0;
                    continue;
                }
                match idx.get(&old[i]) {
                    None => {
                        self.cache.remove(&old[i]);
                        removed += 1;
                    }
                    Some(&j) => old_to_new[i] = j as i64,
                }
            }
            let mut to_build: BTreeSet<i32> = BTreeSet::new();
            for i in 0..size {
                if new_ids[i] <= 0 || old_idx.contains_key(&new_ids[i]) {
                    continue;
                }
                to_build.insert(new_ids[i]);
                let n = cached_neighbors(mem, new_ids[i], self.model.depth(), &mut self.cache);
                for k in n.keys() {
                    if *k > 0 && idx.contains_key(k) && old_idx.contains_key(k) {
                        to_build.insert(*k);
                    }
                }
            }
            if removed > 0 {
                for i in 0..old.len() {
                    if old[i] <= 0 || old_to_new[i] < 0 || to_build.contains(&old[i]) {
                        continue;
                    }
                    if self.columns[i].iter().any(|(r, _)| old_to_new[*r] < 0) {
                        to_build.insert(old[i]);
                    }
                }
            }
            let mut kept: Vec<Vec<(usize, f32)>> = vec![Vec::new(); size];
            for i in 0..old.len() {
                let index = old_to_new[i];
                if index < 0 || old[i] <= 0 || to_build.contains(&old[i]) {
                    continue;
                }
                kept[index as usize] = self.columns[i].iter().map(|(r, v)| (old_to_new[*r] as usize, *v)).collect();
            }
            self.columns = kept;
            for id in to_build {
                self.build_column(mem, id, idx[&id], new_ids, &idx, &mut col);
            }
        }
        if new_ids[0] < 0 {
            self.model.fill_virtual(&mut col);
            self.take_column(&mut col, 0);
        }
        self.sparse_ids = new_ids.to_vec();
        true
    }

    /// BayesFilter::computePosterior
    pub fn compute_posterior(&mut self, mem: &Memory, likelihood: &BTreeMap<i32, f32>) {
        if likelihood.is_empty() {
            return;
        }
        let ids: Vec<i32> = likelihood.keys().copied().collect();
        let lik: Vec<f32> = likelihood.values().copied().collect();
        let same = self.posterior_ids == ids;
        if !same {
            self.prediction_changed = true;
        }
        if self.prediction_changed || self.sparse_ids != ids {
            if !self.update(mem, &ids) {
                self.generate(mem, &ids);
            }
        }
        self.prediction_changed = false;
        if !same {
            // updatePosterior: 새 id 는 0(처음이면 1)
            let was_empty = self.posterior_ids.is_empty();
            let mut vals = Vec::with_capacity(ids.len());
            let mut k = 0;
            for id in &ids {
                while k < self.posterior_ids.len() && self.posterior_ids[k] < *id {
                    k += 1;
                }
                let mut v = if was_empty { 1.0 } else { 0.0 };
                if k < self.posterior_ids.len() && self.posterior_ids[k] == *id {
                    v = self.posterior_values[k];
                }
                vals.push(v);
            }
            self.posterior_ids = ids.clone();
            self.posterior_values = vals;
        }
        // SparsePrediction::multiply: 열 순서대로 prior[행] += 값 * 사후
        let mut prior = vec![0.0f32; ids.len()];
        for (c, col) in self.columns.iter().enumerate() {
            let w = self.posterior_values[c];
            if w == 0.0 {
                continue;
            }
            for (r, v) in col {
                prior[*r] += *v * w;
            }
        }
        let mut sum = 0.0f32;
        for k in 0..ids.len() {
            self.posterior_values[k] = lik[k] * prior[k];
            sum += self.posterior_values[k];
        }
        if sum != 0.0 {
            for v in self.posterior_values.iter_mut() {
                *v /= sum;
            }
        }
    }
}
