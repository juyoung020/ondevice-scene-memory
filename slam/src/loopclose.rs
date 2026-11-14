//! 지도 갱신·루프 클로저: RTAB-Map 0.24.2 Rtabmap::process 의 우리 경로(Mem/IncrementalMemory=true, RGBD/Enabled=true,
//! Rtabmap/CreateIntermediateNodes=false, RGBD/ProximityBySpace=true, RGBD/NeighborLinkRefining=false, 메모리 관리 없음).
//! 원본: corelib/src/Rtabmap.cpp(process · adjustLikelihood · getPaths · optimizeCurrentMap · optimizeGraph · triggerNewMap · resetMemory · getInformation),
//!       Graph.cpp(findNearestNodes · computePathDepths · computeMaxGraphErrors), Optimizer.cpp(getConnectedGraph).
//! 단어 찾기 색인(FLANN)·그래프 최적화(GTSAM)·두 노드 정합(RegistrationVis)은 trait 로 받는다(원본과 같은 결과를 내는 구현을 끼움).
//! 위치는 RTAB-Map Transform 과 같은 3x4 float(rtf::Tf) 로 다룬다: 곱·역행렬 비트가 원본과 같아야 하기 때문.
pub mod bayes;
pub mod flann;
pub mod mem;
pub mod nnkd;
pub use crate::register::rtf;

use crate::Pose;
use bayes::{BayesFilter, PredictionModel};
use mem::{eye6, inv6, umax3, Link, LinkMap, LinkType, Mat6, MemParams, Memory, NodeFeatures, Signature};
use rtf::Tf;
use std::collections::{BTreeMap, BTreeSet};

/// 단어 찾기 색인(rtflann KD-tree 4 그루, L1, checks 32): 설명자는 BRIEF 32 바이트(색인 안에서 256 개 0/1 float 로 펼침)
pub trait WordIndex {
    fn is_built(&self) -> bool;
    /// buildIndex(첫 갱신)
    fn build(&mut self, rows: &[[u8; 32]]);
    /// addPoints: 새 점의 색인 번호
    fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32>;
    fn remove_point(&mut self, idx: u32);
    /// knnSearch(k, checks, eps=0, sorted): 질의마다 k 개(색인, 거리), 못 찾으면 -1
    fn knn_search(&mut self, q: &[[u8; 32]], k: usize, checks: i32) -> (Vec<i32>, Vec<f32>);
    fn release(&mut self);
}

impl WordIndex for flann::KdTreeIndex {
    fn is_built(&self) -> bool {
        flann::KdTreeIndex::is_built(self)
    }
    fn build(&mut self, rows: &[[u8; 32]]) {
        flann::KdTreeIndex::build(self, rows)
    }
    fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32> {
        flann::KdTreeIndex::add_points(self, rows)
    }
    fn remove_point(&mut self, idx: u32) {
        flann::KdTreeIndex::remove_point(self, idx)
    }
    /// 질의끼리 서로 상관없고 나무는 읽기만 하므로 덩어리로 나눠 여러 스레드에서 찾아도 결과 비트가 같다
    fn knn_search(&mut self, q: &[[u8; 32]], k: usize, checks: i32) -> (Vec<i32>, Vec<f32>) {
        use rayon::prelude::*;
        const CHUNK: usize = 64;
        if q.len() <= CHUNK {
            return flann::KdTreeIndex::knn_search(self, q, k, checks);
        }
        let me: &flann::KdTreeIndex = self;
        let parts: Vec<(Vec<i32>, Vec<f32>)> = q.par_chunks(CHUNK).map(|c| me.knn_search(c, k, checks)).collect();
        let mut idx = Vec::with_capacity(q.len() * k);
        let mut dist = Vec::with_capacity(q.len() * k);
        for (i, d) in parts {
            idx.extend(i);
            dist.extend(d);
        }
        (idx, dist)
    }
    fn release(&mut self) {
        flann::KdTreeIndex::release(self)
    }
}

/// 그래프 최적화 결과(실패면 poses 가 빔)
#[derive(Clone, Debug, Default)]
pub struct OptOut {
    pub poses: BTreeMap<i32, Tf>,
    pub error: f64,
    pub iterations: i32,
}

/// Optimizer::optimize(root, poses, links(multimap 순서)) 와 같은 결과
pub trait PoseGraphOptimizer {
    fn optimize(&mut self, root: i32, poses: &BTreeMap<i32, Tf>, links: &LinkMap) -> OptOut;
    /// Optimizer/Iterations
    fn iterations(&self) -> i32 {
        20
    }
}

/// graph.rs(GTSAM 가우스-뉴턴) 를 끼운 것
pub struct GtsamOptimizer(pub crate::graph::OptimizerParams);

impl PoseGraphOptimizer for GtsamOptimizer {
    fn optimize(&mut self, root: i32, poses: &BTreeMap<i32, Tf>, links: &LinkMap) -> OptOut {
        let p: BTreeMap<i32, [f32; 12]> = poses.iter().map(|(k, v)| (*k, v.0)).collect();
        let l: Vec<crate::graph::Link> = links.iter().map(|(_, l)| crate::graph::Link { from: l.from, to: l.to, kind: l.ty as i32, transform: l.t.0, inf: l.inf }).collect();
        match crate::graph::optimize(root, &p, &l, &self.0) {
            Ok(o) => OptOut { poses: o.poses.into_iter().map(|(k, v)| (k, Tf(v))).collect(), error: o.final_error, iterations: o.iterations },
            Err(e) => panic!("그래프 최적화 설정 밖: {e}"),
        }
    }
    fn iterations(&self) -> i32 {
        self.0.iterations
    }
}

/// 두 노드 정합(Memory::computeTransform → RegistrationVis) 결과. t 가 NULL 이면 거절
#[derive(Clone, Debug)]
pub struct RegOut {
    pub t: Tf,
    pub cov: Mat6,
    pub inliers: i32,
    pub matches: i32,
}

/// Memory::computeTransform(fromS, toS, guess): from 에서 본 to 의 위치
pub trait LoopRegistration {
    fn compute(&mut self, from: &Signature, to: &Signature, guess: Tf) -> RegOut;
}

/// process 한 번의 입력(위치 추적이 준 그대로)
#[derive(Clone, Debug)]
pub struct NodeInput {
    pub features: NodeFeatures,
    pub stamp: f64,
    /// 위치 추적 위치(NULL = 놓침)
    pub pose: Tf,
    /// 위치 추적 공분산(6x6, 행 우선)
    pub cov: Mat6,
    pub velocity: Vec<f32>,
    pub image_size: (u32, u32),
}

/// 한 번의 결과 요약(검증·통계용)
#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    pub likelihood: BTreeMap<i32, f32>,
    pub raw_likelihood: BTreeMap<i32, f32>,
    pub posterior: BTreeMap<i32, f32>,
    pub highest: (i32, f32),
    pub proximity_id: i32,
    pub small_displacement: bool,
    pub rejected: bool,
}

/// 설정(우리 ini 와 Parameters.h 기본값)
#[derive(Clone, Debug)]
pub struct SlamParams {
    pub loop_thr: f32,
    pub aggressive_loop_thr: f32,
    pub local_radius: f32,
    pub proximity_max_graph_depth: i32,
    pub proximity_max_paths: i32,
    pub proximity_filtering_radius: f32,
    pub proximity_angle: f32,
    pub optimize_max_error: f32,
    pub linear_update: f32,
    pub angular_update: f32,
    pub max_odom_cache: usize,
}

impl SlamParams {
    pub fn from_params(p: &crate::params::Params) -> SlamParams {
        SlamParams {
            loop_thr: p.get("Rtabmap/LoopThr", 0.11f32),
            aggressive_loop_thr: p.get("RGBD/AggressiveLoopThr", 0.05f32),
            local_radius: p.get("RGBD/LocalRadius", 10.0f32),
            proximity_max_graph_depth: p.get("RGBD/ProximityMaxGraphDepth", 50),
            proximity_max_paths: p.get("RGBD/ProximityMaxPaths", 3),
            proximity_filtering_radius: p.get("RGBD/ProximityPathFilteringRadius", 1.0f32),
            // _proximityAngle *= M_PI/180.0f (float 변수)
            proximity_angle: (p.get("RGBD/ProximityAngle", 45.0f32) as f64 * (std::f64::consts::PI / 180.0f32 as f64)) as f32,
            optimize_max_error: p.get("RGBD/OptimizeMaxError", 3.0f32),
            linear_update: p.get("RGBD/LinearUpdate", 0.1f32),
            angular_update: p.get("RGBD/AngularUpdate", 0.1f32),
            max_odom_cache: p.get("RGBD/MaxOdomCacheSize", 10usize),
        }
    }
}

pub struct Slam {
    pub p: SlamParams,
    pub mem: Memory,
    pub bayes: BayesFilter,
    pub opt: Box<dyn PoseGraphOptimizer>,
    pub reg: Box<dyn LoopRegistration>,
    optimized: BTreeMap<i32, Tf>,
    constraints: LinkMap,
    map_correction: Tf,
    loop_hyp: (i32, f32),
    highest: (i32, f32),
    odom_cache_poses: BTreeMap<i32, Tf>,
    odom_cache_links: LinkMap,
    pub info: StepInfo,
    /// getConnectedGraph 를 거친 최적화 수(검증용)
    pub connected_graph_calls: usize,
}

impl Slam {
    pub fn new(params: &crate::params::Params, index: Box<dyn WordIndex>, opt: Box<dyn PoseGraphOptimizer>, reg: Box<dyn LoopRegistration>) -> Slam {
        let p = SlamParams::from_params(params);
        let mp = MemParams {
            stm_size: params.get("Mem/STMSize", 10usize),
            rehearsal_similarity: params.get("Mem/RehearsalSimilarity", 0.6f32),
            kp_max_features: params.get("Kp/MaxFeatures", 500),
            vis_max_features: params.get("Vis/MaxFeatures", 1000),
            linear_update: p.linear_update,
            angular_update: p.angular_update,
            nndr: params.get("Kp/NndrRatio", 0.8f32),
        };
        let lc = params
            .0
            .get("Bayes/PredictionLC")
            .cloned()
            .unwrap_or_else(|| "0.1 0.36 0.30 0.16 0.062 0.0151 0.00255 0.000324 2.5e-05 1e-06 4.8e-08 1.2e-09 1.9e-11 2.2e-13 1.7e-15 8.5e-18 2.9e-20 6.9e-23".into());
        let model = PredictionModel::new(&lc, params.get("Bayes/VirtualPlacePriorThr", 0.9f32));
        let mut mem = Memory::new(mp, index);
        mem.dict.checks = params.get("Rust/DictChecks", 32);
        Slam {
            p,
            mem,
            bayes: BayesFilter::new(model),
            opt,
            reg,
            optimized: BTreeMap::new(),
            constraints: LinkMap::default(),
            map_correction: Tf::IDENTITY,
            loop_hyp: (0, 0.0),
            highest: (0, 0.0),
            odom_cache_poses: BTreeMap::new(),
            odom_cache_links: LinkMap::default(),
            info: StepInfo::default(),
            connected_graph_calls: 0,
        }
    }

    pub fn loop_closure_id(&self) -> i32 {
        self.loop_hyp.0
    }
    /// getLastLocationId = Memory 가 마지막으로 만든 id
    pub fn last_location_id(&self) -> i32 {
        self.mem.id_count
    }
    pub fn map_correction(&self) -> Tf {
        self.map_correction
    }
    pub fn local_optimized_tf(&self) -> &BTreeMap<i32, Tf> {
        &self.optimized
    }
    pub fn local_optimized_poses(&self) -> BTreeMap<i32, Pose> {
        self.optimized.iter().map(|(k, v)| (*k, v.to_pose())).collect()
    }
    pub fn constraints(&self) -> &LinkMap {
        &self.constraints
    }

    /// triggerNewMap
    pub fn trigger_new_map(&mut self) -> i32 {
        self.odom_cache_poses.clear();
        self.odom_cache_links.clear();
        let id = self.mem.increment_map_id();
        self.optimized.clear();
        self.constraints.clear();
        self.bayes.reset();
        id
    }

    /// resetMemory
    pub fn reset(&mut self) {
        self.highest = (0, 0.0);
        self.loop_hyp = (0, 0.0);
        self.optimized.clear();
        self.constraints.clear();
        self.map_correction = Tf::IDENTITY;
        self.odom_cache_poses.clear();
        self.odom_cache_links.clear();
        self.mem.reset();
        self.bayes.reset();
    }

    fn compute_transform(&mut self, from: i32, to: i32, guess: Tf) -> RegOut {
        match (self.mem.sig(from), self.mem.sig(to)) {
            (Some(a), Some(b)) => self.reg.compute(a, b, guess),
            _ => RegOut { t: Tf::NULL, cov: eye6(1.0), inliers: 0, matches: 0 },
        }
    }

    /// Rtabmap::process(data, odomPose, covariance, velocity). false = 노드를 만들지 않음(위치 없음)
    pub fn process(&mut self, inp: &NodeInput) -> bool {
        self.loop_hyp = (0, 0.0);
        let last_highest = self.highest;
        self.highest = (0, 0.0);
        self.info = StepInfo::default();
        let odom_pose = inp.pose;
        if odom_pose.is_null() {
            return false;
        }
        if let Some(last) = self.mem.last_working_signature() {
            if !last.pose.is_identity() && odom_pose.is_identity() {
                self.trigger_new_map();
            }
        }
        self.mem.update(&inp.features, inp.stamp, odom_pose, &inp.cov, &inp.velocity, inp.image_size);
        let sid = self.mem.last_signature.unwrap();
        let rehearsed = self.mem.rehearsal.0;
        let mut small = false;
        if rehearsed > 0 {
            self.optimized.remove(&rehearsed);
        } else {
            let s = self.mem.sig(sid).unwrap();
            let mut t = Tf::NULL;
            if s.weight >= 0 {
                let n = self.mem.neighbors_id(sid, 2, true, true, true, true, None);
                if n.len() == 2 {
                    let nid = *n.keys().next().unwrap();
                    if let Some(l) = s.links.get(&nid) {
                        t = l.t;
                    } else if let Some(p) = self.optimized.get(&nid) {
                        t = p.inverse().mul(&self.map_correction).mul(&s.pose);
                    }
                }
            }
            if (self.p.linear_update > 0.0 || self.p.angular_update > 0.0) && !t.is_null() {
                let r = t.xyz_rpy();
                let (lu, au) = (self.p.linear_update, self.p.angular_update);
                let moving = (lu > 0.0 && (r[0].abs() > lu || r[1].abs() > lu || r[2].abs() > lu)) || (au > 0.0 && (r[3].abs() > au || r[4].abs() > au || r[5].abs() > au));
                small = !moving;
            }
        }
        self.info.small_displacement = small;
        let s = self.mem.sig(sid).unwrap().clone();
        let new_pose = self.map_correction.mul(&s.pose);
        self.optimized.entry(sid).or_insert(new_pose);
        if let Some((_, l)) = s.links.iter().next() {
            if l.ty == LinkType::Neighbor {
                let tmp = l.inverse();
                self.constraints.insert(tmp.from, tmp);
            }
        }
        // 위치 추적 캐시(최근 10 노드): 근접 찾기의 그래프 깊이 계산에 들어감
        if s.weight >= 0 {
            if !self.odom_cache_poses.is_empty() {
                while !self.odom_cache_poses.is_empty() && self.odom_cache_poses.len() > self.p.max_odom_cache {
                    let first = *self.odom_cache_poses.keys().next().unwrap();
                    self.odom_cache_links.remove_key(first);
                    self.odom_cache_poses.remove(&first);
                }
                if let Some((&last, lp)) = self.odom_cache_poses.iter().next_back() {
                    let l = Link::new(last, sid, LinkType::Neighbor, lp.inverse().mul(&s.pose), inv6(&inp.cov));
                    self.odom_cache_links.insert(last, l);
                }
            }
            self.odom_cache_poses.insert(sid, s.pose);
        }
        let loc_prev = s.links.iter().next().map(|(k, _)| *k != sid && self.mem.has_loop_closure_links(*k)).unwrap_or(false);
        let mut rejected = false;
        let mut likelihood = BTreeMap::new();
        if !s.is_bad() && s.weight >= 0 && (!small || !loc_prev) {
            if self.mem.wm_size(true) > 0 {
                let mut cmp = Vec::new();
                for id in &self.mem.wm {
                    if *id > 0 {
                        if self.mem.sig(*id).unwrap().weight != -1 {
                            cmp.push(*id);
                        }
                    } else {
                        cmp.push(*id);
                    }
                }
                let raw = self.mem.compute_likelihood(sid, &cmp);
                likelihood = raw.clone();
                adjust_likelihood(&mut likelihood);
                self.bayes.compute_posterior(&self.mem, &likelihood);
                self.info.raw_likelihood = raw;
                self.info.likelihood = likelihood.clone();
                let ids = &self.bayes.posterior_ids;
                let vals = &self.bayes.posterior_values;
                self.info.posterior = ids.iter().copied().zip(vals.iter().copied()).collect();
                if !ids.is_empty() {
                    // 큰 id 부터: 같은 확률이면 큰 id
                    for i in (0..ids.len()).rev() {
                        if ids[i] > 0 && vals[i] > self.highest.1 {
                            self.highest = (ids[i], vals[i]);
                        }
                    }
                    self.highest.1 = 1.0 - vals[0];
                }
                if self.highest.0 > 0 {
                    let mut thr = self.p.loop_thr;
                    if !self.optimized.contains_key(&self.highest.0) && self.mem.wm_size(true) > 0 && thr > self.p.aggressive_loop_thr {
                        thr = self.p.aggressive_loop_thr;
                    }
                    if self.highest.1 >= thr {
                        rejected = true;
                        if !(self.bayes.posterior_ids.len() <= 2 && thr > 0.0) {
                            self.loop_hyp = self.highest;
                            rejected = false;
                        }
                    }
                }
            }
        } else if !s.is_bad() && s.weight >= 0 && small {
            self.highest = last_highest;
        }
        self.info.highest = self.highest;
        let mut added: Vec<(i32, i32)> = Vec::new();
        let mut last_prox = 0;
        let mut suppressed = 0;
        if self.p.local_radius > 0.0 && s.weight >= 0 && self.opt.iterations() != 0 && (!small || !loc_prev) {
            let cur = self.optimized[&sid];
            let nearest_ids = find_nearest_nodes_id(sid, &self.optimized, self.p.local_radius, 0.0);
            let mut links: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
            let push = |links: &mut BTreeMap<i32, Vec<i32>>, a: i32, b: i32| links.entry(a).or_default().push(b);
            if self.p.proximity_max_graph_depth > 0 {
                for (_, l) in self.constraints.iter() {
                    if self.optimized.contains_key(&l.from) && self.optimized.contains_key(&l.to) {
                        push(&mut links, l.from, l.to);
                        push(&mut links, l.to, l.from);
                    }
                }
                if self.odom_cache_poses.len() > 1 {
                    let mut with = self.optimized.clone();
                    for (k, v) in &self.odom_cache_poses {
                        with.entry(*k).or_insert(*v);
                    }
                    for (_, l) in self.odom_cache_links.iter() {
                        if with.contains_key(&l.from) && with.contains_key(&l.to) && l.from != l.to {
                            push(&mut links, l.from, l.to);
                            push(&mut links, l.to, l.from);
                        }
                    }
                }
            }
            let depths = if self.p.proximity_max_graph_depth > 0 { compute_path_depths(&links, sid, self.p.proximity_max_graph_depth) } else { BTreeMap::new() };
            let mut nearest_poses = BTreeMap::new();
            for (id, _) in nearest_ids.range(1..) {
                if !self.mem.stm.contains(id) {
                    if self.p.proximity_max_graph_depth > 0 && !depths.contains_key(id) {
                        continue;
                    }
                    nearest_poses.insert(*id, self.optimized[id]);
                }
            }
            let paths = self.get_paths(&nearest_poses, &cur, self.p.proximity_max_graph_depth);
            let cur_inv = cur.inverse();
            // NearestPathKey(가능도, id, 거리²): std::map 이라 같은 키는 처음 것만
            let less = |a: &(f32, i32, f32), b: &(f32, i32, f32)| a.0 < b.0 || (a.0 == b.0 && (a.2 > b.2 || (a.2 == b.2 && a.1 < b.1)));
            let mut keyed: Vec<((f32, i32, f32), BTreeMap<i32, Tf>)> = Vec::new();
            for (pid, path) in paths {
                let mut hl = 0.0f32;
                let mut hid = pid;
                let mut sd = -1.0f32;
                for (j, p) in &path {
                    let v = likelihood.get(j).copied().unwrap_or(0.0);
                    let d = cur_inv.mul(p).norm_sq();
                    if v > hl || (v == hl && (sd < 0.0 || d < sd)) {
                        hl = v;
                        hid = *j;
                        sd = d;
                    }
                }
                let k = (hl, hid, sd);
                if !keyed.iter().any(|(x, _)| !less(x, &k) && !less(&k, x)) {
                    keyed.push((k, path));
                }
            }
            keyed.sort_by(|a, b| {
                if less(&a.0, &b.0) {
                    std::cmp::Ordering::Less
                } else if less(&b.0, &a.0) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            let fr = self.p.proximity_filtering_radius;
            let mut checked = 0;
            for (key, path0) in keyed.iter().rev() {
                if !(self.p.proximity_max_paths <= 0 || checked < self.p.proximity_max_paths) {
                    break;
                }
                let mut path = path0.clone();
                path.insert(sid, cur);
                let path = find_nearest_poses_id(sid, &path, self.p.local_radius, self.p.proximity_angle);
                let nearest = if key.0 > 0.0 && path.contains_key(&key.1) { key.1 } else { find_nearest_node(&path, &cur).0 };
                if nearest > 0 {
                    let has = self.mem.sig(sid).unwrap().has_link(nearest);
                    if !has && (fr <= 0.0 || cur.dist_sq(&self.optimized[&nearest]) < fr * fr) {
                        checked += 1;
                        let r = self.compute_transform(nearest, sid, Tf::NULL);
                        if !r.t.is_null() {
                            let t = r.t.inverse();
                            if fr <= 0.0 || t.norm_sq() <= fr * fr {
                                last_prox = nearest;
                                let mut ty = LinkType::LocalSpaceClosure;
                                if self.loop_hyp.0 > 0 && nearest_ids.contains_key(&self.loop_hyp.0) {
                                    if nearest == self.loop_hyp.0 {
                                        ty = LinkType::GlobalClosure;
                                        suppressed = nearest;
                                    } else if suppressed == 0 {
                                        suppressed = nearest;
                                    }
                                }
                                self.mem.add_link(Link::new(sid, nearest, ty, t, inv6(&r.cov)));
                                added.push((sid, nearest));
                            }
                        }
                    }
                }
            }
        }
        if self.loop_hyp.0 > 0 {
            if suppressed == 0 {
                let r = self.compute_transform(self.loop_hyp.0, sid, Tf::NULL);
                rejected = r.t.is_null();
                if !rejected {
                    let t = r.t.inverse();
                    rejected = !self.mem.add_link(Link::new(sid, self.loop_hyp.0, LinkType::GlobalClosure, t, inv6(&r.cov)));
                    if !rejected {
                        added.push((sid, self.loop_hyp.0));
                    }
                }
                if rejected {
                    self.loop_hyp.0 = 0;
                }
            } else if suppressed != self.loop_hyp.0 {
                self.loop_hyp.0 = 0;
            }
        }
        let s_weight = self.mem.sig(sid).map(|s| s.weight).unwrap_or(-9);
        if s_weight != -1 && (self.loop_hyp.0 > 0 || last_prox > 0) {
            let mut poses = self.optimized.clone();
            let (ok, cons, iters) = self.optimize_current_map(sid, &mut poses);
            let mut update = true;
            let mut reject = !ok;
            if ok && !added.is_empty() && iters > 0 && !cons.is_empty() {
                let e = compute_max_graph_errors(&poses, &cons);
                let max = self.p.optimize_max_error;
                reject = (e.linear_valid && max > 0.0 && e.linear_ratio > max) || (e.angular_valid && max > 0.0 && e.angular_ratio > max);
            }
            if reject {
                for (a, b) in &added {
                    self.mem.remove_link(*a, *b);
                }
                update = false;
                self.loop_hyp.0 = 0;
                last_prox = 0;
                rejected = true;
            }
            if update {
                self.optimized = poses;
                self.constraints = cons;
            }
            let pose = self.mem.sig(sid).unwrap().pose;
            self.map_correction = self.optimized[&sid].mul(&pose.inverse());
        }
        // 통계의 사후 확률 = 베이즈 필터의 지금 상태(이번에 다시 계산하지 않았어도)
        self.info.posterior = self.bayes.posterior_ids.iter().copied().zip(self.bayes.posterior_values.iter().copied()).collect();
        self.info.proximity_id = last_prox;
        self.info.rejected = rejected;
        let w = self.mem.sig(sid).map(|s| s.weight).unwrap_or(-9);
        if w >= 0 && small && self.loop_hyp.0 == 0 && last_prox == 0 {
            self.mem.move_to_trash(sid, false);
            self.odom_cache_poses.remove(&sid);
            let keys: Vec<i32> = self.odom_cache_links.0.keys().copied().collect();
            for k in keys {
                let v = self.odom_cache_links.0.get_mut(&k).unwrap();
                v.retain(|l| l.from != sid && l.to != sid);
                if v.is_empty() {
                    self.odom_cache_links.0.remove(&k);
                }
            }
            if !self.optimized.is_empty() || !self.constraints.is_empty() {
                if self.mem.last_working_signature().is_some() && !self.optimized.is_empty() {
                    self.optimized.remove(&sid);
                    // 원본은 sid 를 키로 가진 제약만 지운다(이웃 제약은 앞 노드 키라 남음)
                    if let Some(v) = self.constraints.0.get(&sid).cloned() {
                        for l in v {
                            if l.to != l.from {
                                if let Some(at) = self.constraints.find(l.to, l.from, false, LinkType::Undef) {
                                    self.constraints.remove_at(at);
                                }
                            }
                        }
                    }
                    self.constraints.remove_key(sid);
                } else {
                    self.optimized.clear();
                    self.constraints.clear();
                }
            }
        }
        true
    }

    /// Rtabmap::getGraph(poses, constraints, optimized, global). 데이터베이스가 없으므로 global 은 마지막 노드를 고를 때
    /// 중간 노드를 건너뛰지 않는다는 것만 다르다(지운 노드는 원본에서도 링크가 끊겨 닿지 않음, 우리 설정엔 중간 노드가 없음).
    /// optimized: 지금 지역 최적 위치를 추측으로 다시 최적화(global=false 이고 지역 위치가 있으면 위치는 지역 최적 위치 그대로)
    pub fn get_graph(&mut self, optimized: bool, global: bool) -> (BTreeMap<i32, Tf>, LinkMap) {
        let last = if global {
            self.mem.last_working_signature().map(|s| s.id)
        } else {
            match self.mem.last_working_signature() {
                Some(s) if s.weight == -1 => self.mem.signatures.values().rev().find(|s| s.weight != -1).map(|s| s.id),
                o => o.map(|s| s.id),
            }
        };
        let Some(id) = last else {
            return (BTreeMap::new(), LinkMap::default());
        };
        if optimized {
            let mut poses = self.optimized.clone();
            let (_, cons, _) = self.optimize_current_map(id, &mut poses);
            if !global && !self.optimized.is_empty() {
                poses = self.optimized.clone();
            }
            (poses, cons)
        } else {
            let ids = self.mem.neighbors_id(id, 0, true, false, false, false, None);
            self.mem.metric_constraints(&ids.keys().copied().collect())
        }
    }

    /// Rtabmap::getPaths
    fn get_paths(&self, poses_in: &BTreeMap<i32, Tf>, target: &Tf, max_depth: i32) -> BTreeMap<i32, BTreeMap<i32, Tf>> {
        let mut paths = BTreeMap::new();
        let mut poses: BTreeMap<i32, Tf> = poses_in.range(1..).map(|(k, v)| (*k, *v)).collect();
        let nodes: BTreeSet<i32> = poses.keys().copied().collect();
        if nodes.is_empty() || target.is_null() {
            return paths;
        }
        while !poses.is_empty() {
            let nearest = find_nearest_node(&poses, target).0;
            if nearest == 0 {
                break;
            }
            let ids = self.mem.neighbors_id(nearest, max_depth, true, true, true, true, Some(&nodes));
            let mut path: BTreeMap<i32, Tf> = BTreeMap::new();
            for id in ids.keys() {
                if let Some(p) = poses.get(id).copied() {
                    let mut valid = path.is_empty();
                    if !valid {
                        if let Some(s) = self.mem.sig(*id) {
                            valid = s.links.iter().any(|(k, l)| (l.ty == LinkType::Neighbor || l.ty == LinkType::NeighborMerged) && path.contains_key(k));
                        }
                    }
                    if valid {
                        path.insert(*id, p);
                        poses.remove(id);
                    }
                }
            }
            if path.is_empty() {
                break;
            }
            paths.insert(nearest, path);
        }
        paths
    }

    /// optimizeCurrentMap(id, lookInDatabase=false) → (성공, 제약, 반복 수)
    fn optimize_current_map(&mut self, id: i32, poses: &mut BTreeMap<i32, Tf>) -> (bool, LinkMap, i32) {
        let ids = self.mem.neighbors_id(id, 0, true, false, false, false, None);
        let mut root = id;
        if ids.len() > 1 {
            for k in ids.keys() {
                if let Some(s) = self.mem.sig(*k) {
                    if s.weight != -1 {
                        root = *k;
                        break;
                    }
                }
            }
        }
        let set: BTreeSet<i32> = ids.keys().copied().collect();
        let (mut mp, links) = self.mem.metric_constraints(&set);
        if self.opt.iterations() > 0 {
            for (k, v) in mp.iter_mut() {
                if *k != root {
                    if let Some(g) = poses.get(k) {
                        *v = *g;
                    }
                }
            }
        }
        let (out, cons) = if mp.len() != poses.len() {
            self.connected_graph_calls += 1;
            let (po, lo) = connected_graph(root, &mp, &links);
            (self.opt.optimize(root, &po, &lo), lo)
        } else {
            (self.opt.optimize(root, &mp, &links), links)
        };
        if out.poses.is_empty() {
            poses.clear();
            return (false, LinkMap::default(), 0);
        }
        *poses = out.poses;
        (true, cons, out.iterations)
    }
}

/// adjustLikelihood(Rtabmap/VirtualPlaceLikelihoodRatio=0): uMean · uVariance 는 float
fn adjust_likelihood(lik: &mut BTreeMap<i32, f32>) {
    if lik.is_empty() {
        return;
    }
    let vals: Vec<f32> = lik.values().skip(1).copied().filter(|v| *v > 0.0).collect();
    let mut mean = 0.0f32;
    if !vals.is_empty() {
        for v in &vals {
            mean += *v;
        }
        mean /= vals.len() as f32;
    }
    let mut var = 0.0f32;
    if vals.len() > 1 {
        let mut sum = 0.0f32;
        for v in &vals {
            sum += (*v - mean) * (*v - mean);
        }
        var = sum / (vals.len() - 1) as f32;
    }
    let std = var.sqrt();
    let eps = 0.0001f32;
    let mut max = 0.0f32;
    for (_, v) in lik.iter_mut().skip(1) {
        let value = *v;
        *v = 1.0;
        if value > mean + std && mean != 0.0 {
            *v = (value - (std - eps)) / mean;
        }
        if value > max {
            max = value;
        }
    }
    let first = lik.values_mut().next().unwrap();
    *first = if std > eps && max != 0.0 { mean / std + 1.0 } else { 2.0 };
}

impl Tf {
    /// getNormSquared = uNormSquared(x,y,z)
    pub fn norm_sq(&self) -> f32 {
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    }
    pub fn dist_sq(&self, o: &Tf) -> f32 {
        let (a, b, c) = (self.x() - o.x(), self.y() - o.y(), self.z() - o.z());
        a * a + b * b + c * c
    }
    /// getAngle = getQuaternionf().angularDistance(o)
    pub fn angle_to(&self, o: &Tf) -> f32 {
        let q = |t: &Tf| rtf::quat_normalized_f(rtf::quat_from_mat3f(&[t.0[0], t.0[1], t.0[2], t.0[4], t.0[5], t.0[6], t.0[8], t.0[9], t.0[10]]));
        let a = q(self);
        let b = q(o);
        let (x1, y1, z1, w1) = (-a[0], -a[1], -a[2], a[3]);
        let (x2, y2, z2, w2) = (b[0], b[1], b[2], b[3]);
        let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        let y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2;
        let z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2;
        2.0 * (x * x + y * y + z * z).sqrt().atan2(w.abs())
    }
}

/// graph::findNearestNodes(target, poses, radius, angle, k): PCL KdTreeFLANN(nnkd) 로 찾고 각도로 거름
fn find_nearest_nodes(target: &Tf, poses: &BTreeMap<i32, Tf>, radius: f32, angle: f32, k: usize) -> BTreeMap<i32, f32> {
    let mut found = BTreeMap::new();
    if poses.is_empty() {
        return found;
    }
    let ids: Vec<i32> = poses.keys().copied().collect();
    let tree = nnkd::KdSingle::new(poses.values().map(|p| [p.x(), p.y(), p.z()]).collect());
    let q = [target.x(), target.y(), target.z()];
    let cand: Vec<(i32, f32)> = if radius > 0.0 {
        assert!(k == 0, "반지름과 k 를 함께 쓰는 찾기(KNNRadiusResultSet)는 옮기지 않았다");
        tree.radius(&q, radius as f64)
    } else {
        tree.knn(&q, k)
    }
    .into_iter()
    .map(|(i, d)| (ids[i], d))
    .collect();
    let va = [target.0[0], target.0[4], target.0[8]];
    for (id, d) in cand {
        if angle > 0.0 {
            let p = &poses[&id];
            let vb = [p.0[0], p.0[4], p.0[8]];
            // pcl::getAngle3D(Vector4f, w=0): 각각 normalized() 뒤 dot, [-1,1] 로 자르고 float acos. 4 칸 SSE 합 = (0+2)+(1+3)
            let nrm = |v: [f32; 3]| {
                let n = (v[0] * v[0] + v[2] * v[2]) + (v[1] * v[1] + 0.0);
                let s = n.sqrt();
                [v[0] / s, v[1] / s, v[2] / s]
            };
            let (a1, b1) = (nrm(va), nrm(vb));
            let mut rad = (a1[0] * b1[0] + a1[2] * b1[2]) + (a1[1] * b1[1] + 0.0);
            rad = rad.clamp(-1.0, 1.0);
            let rad = rad;
            let a = (rad as f64).acos(); // PCL: double rad, std::acos(double)
            if a <= angle as f64 {
                found.insert(id, d);
            }
        } else {
            found.insert(id, d);
        }
    }
    found
}

fn find_nearest_nodes_id(id: i32, poses: &BTreeMap<i32, Tf>, radius: f32, angle: f32) -> BTreeMap<i32, f32> {
    let mut m = poses.clone();
    let t = m.remove(&id).unwrap();
    find_nearest_nodes(&t, &m, radius, angle, 0)
}

fn find_nearest_poses_id(id: i32, poses: &BTreeMap<i32, Tf>, radius: f32, angle: f32) -> BTreeMap<i32, Tf> {
    find_nearest_nodes_id(id, poses, radius, angle).keys().map(|k| (*k, poses[k])).collect()
}

fn find_nearest_node(poses: &BTreeMap<i32, Tf>, target: &Tf) -> (i32, f32) {
    find_nearest_nodes(target, poses, 0.0, 0.0, 1).into_iter().next().unwrap_or((0, 0.0))
}

/// graph::computePathDepths(너비 우선)
fn compute_path_depths(links: &BTreeMap<i32, Vec<i32>>, from: i32, max: i32) -> BTreeMap<i32, i32> {
    let mut d = BTreeMap::new();
    d.insert(from, 0);
    let mut q = std::collections::VecDeque::new();
    q.push_back(from);
    while let Some(c) = q.pop_front() {
        let cd = d[&c];
        if max > 0 && cd >= max {
            continue;
        }
        if let Some(v) = links.get(&c) {
            for n in v {
                if !d.contains_key(n) {
                    d.insert(*n, cd + 1);
                    q.push_back(*n);
                }
            }
        }
    }
    d
}

pub struct MaxErrors {
    pub linear_ratio: f32,
    pub angular_ratio: f32,
    pub linear_valid: bool,
    pub angular_valid: bool,
}

/// graph::computeMaxGraphErrors(force3DoF=false)
fn compute_max_graph_errors(poses: &BTreeMap<i32, Tf>, links: &LinkMap) -> MaxErrors {
    let none = MaxErrors { linear_ratio: 0.0, angular_ratio: 0.0, linear_valid: false, angular_valid: false };
    let mut e = MaxErrors { linear_ratio: 0.0, angular_ratio: 0.0, linear_valid: false, angular_valid: false };
    for (_, l) in links.iter() {
        if l.from == l.to {
            continue;
        }
        let (Some(t1), Some(t2)) = (poses.get(&l.from), poses.get(&l.to)) else {
            return none;
        };
        let (t, lt) = if l.from < 0 { (t2.inverse().mul(t1), l.t.inverse()) } else { (t1.inverse().mul(t2), l.t) };
        let lin = umax3((lt.x() - t.x()).abs() as f64, (lt.y() - t.y()).abs() as f64, (lt.z() - t.z()).abs() as f64) as f32;
        let sl = l.trans_variance().sqrt() as f32;
        let r = lin / sl;
        if r > e.linear_ratio {
            e.linear_ratio = r;
            e.linear_valid = true;
        }
        if l.ty != LinkType::Landmark || 1.0 / l.inf[35] < 9999.0 {
            let ang = t.angle_to(&lt);
            let sa = l.rot_variance().sqrt() as f32;
            let r = ang / sa;
            if r > e.angular_ratio {
                e.angular_ratio = r;
                e.angular_valid = true;
            }
        }
    }
    e
}

/// Optimizer::getConnectedGraph(뿌리에서 닿는 노드·링크만, 위치는 링크로 이어 붙임). LinkIdKey: 이웃 링크 먼저, 그다음 id 순
fn connected_graph(root: i32, poses_in: &BTreeMap<i32, Tf>, links_in: &LinkMap) -> (BTreeMap<i32, Tf>, LinkMap) {
    let key = |id: i32, ty: LinkType| -> (i32, i32) {
        if id < 0 {
            return (3, -id);
        }
        let cls = match ty {
            LinkType::Neighbor => 0,
            LinkType::NeighborMerged => 1,
            _ => 2,
        };
        (cls, id)
    };
    let mut next: BTreeMap<(i32, i32), (i32, Tf)> = BTreeMap::new();
    next.insert(key(root, LinkType::Undef), (root, poses_in[&root]));
    let mut bi: BTreeMap<i32, Vec<(i32, LinkType)>> = BTreeMap::new();
    for (_, l) in links_in.iter() {
        if l.from != l.to {
            let has = |bi: &BTreeMap<i32, Vec<(i32, LinkType)>>, a: i32, b: i32| bi.get(&a).map(|v| v.iter().any(|(t, ty)| *t == b && *ty == l.ty)).unwrap_or(false);
            if !has(&bi, l.from, l.to) && !has(&bi, l.to, l.from) {
                bi.entry(l.from).or_default().push((l.to, l.ty));
                bi.entry(l.to).or_default().push((l.from, l.ty));
            }
        }
    }
    let mut out: BTreeMap<i32, Tf> = BTreeMap::new();
    let mut lout = LinkMap::default();
    while let Some((&k, &(cur, pose))) = next.iter().next() {
        next.remove(&k);
        if out.contains_key(&cur) {
            continue;
        }
        out.insert(cur, pose);
        if let Some(v) = links_in.0.get(&cur) {
            for l in v {
                if l.from == l.to && l.ty != LinkType::PosePrior {
                    lout.insert(cur, l.clone());
                }
            }
        }
        for (to, ty) in bi.get(&cur).cloned().unwrap_or_default() {
            if !poses_in.contains_key(&to) {
                continue;
            }
            let kl = links_in.get(links_in.find(cur, to, true, ty).unwrap()).clone();
            if !out.contains_key(&to) {
                let t = if kl.from == cur { kl.t } else { kl.t.inverse() };
                let p = out[&cur].mul(&t);
                next.entry(key(to, ty)).or_insert((to, p));
            }
            if lout.find(cur, to, true, kl.ty).is_none() {
                if kl.to < 0 {
                    lout.insert(kl.to, kl.inverse());
                } else {
                    lout.insert(kl.from, kl);
                }
            }
        }
    }
    (out, lout)
}

#[cfg(test)]
mod refdump;

impl Default for NodeInput {
    fn default() -> Self {
        NodeInput { features: NodeFeatures::default(), stamp: 0.0, pose: Tf::NULL, cov: [0.0; 36], velocity: Vec::new(), image_size: (0, 0) }
    }
}
