//! 시각 단어 사전의 검색 색인 (Kp/NNStrategy=1, Kp/IncrementalFlann=true, Kp/ByteToFloat=false).
//! 원본: rtabmap 0.24.2 corelib/src/FlannIndex.cpp(FLANN_INDEX_KDTREE, useDistanceL1=true) 와
//!       그 안에 들어 있는 rtflann(corelib/src/rtflann: algorithms/kdtree_index.h · nn_index.h,
//!       util/random.h · heap.h · result_set.h, algorithms/dist.h 의 L1<float>). OpenCV cv::flann 은 쓰지 않는다.
//!
//! VWDictionary 는 BRIEF 32 바이트를 비트마다 0/1 float 256 개로 펼쳐(convertBinTo32F) L1 거리로 넣는다.
//! 여기서는 펼치지 않고 비트로 들고 있다. 값이 0/1 이라 L1 합은 정수(≤256)이고 float 로 정확하므로
//! 원본의 4 개씩 묶은 float 합과 popcount(xor) 는 비트까지 같다. 나머지(평균·분산·자르는 값)는 원본 float 계산 그대로.
//!
//! 무작위: 원본은 나무마다 std::random_device 로 mt19937 씨앗을 받아 std::shuffle(libstdc++) 하고,
//! 자를 축은 glibc rand() 로 고른다. random_device 는 실행마다 달라 원본 자체가 재현되지 않는다.
//! 그래서 씨앗은 밖에서 넣는다(seed_source, 기본은 k 번째 = (k+1)*2654435761). 같은 씨앗·같은 rand 상태면 결과가 비트까지 같다.

/// 나무 수(FlannIndex::buildIndex 의 KDTreeIndexParams(4))
pub const TREES: usize = 4;
/// 평균·분산을 낼 때 보는 앞쪽 점 수 - 1 (kdtree_index.h SAMPLE_MEAN)
const SAMPLE_MEAN: usize = 100;
/// 분산 큰 축 몇 개 중에서 고를지 (RAND_DIM)
const RAND_DIM: usize = 5;
/// 설명자 비트 수(= 원본 veclen)
const DIM: usize = 256;
const NONE: u32 = u32::MAX;

/// glibc rand() (TYPE_3, 덧셈 되먹임). srand(seed) 와 같은 상태에서 시작
#[derive(Clone)]
pub struct GlibcRand {
    r: [u32; 34],
    k: usize,
}

impl GlibcRand {
    pub fn new(seed: u32) -> Self {
        let mut v = vec![0u32; 344];
        let seed = if seed == 0 { 1 } else { seed };
        v[0] = seed;
        let mut w = seed as i32 as i64; // glibc 는 int32_t word
        for i in 1..31 {
            let (hi, lo) = (w / 127773, w % 127773);
            w = 16807 * lo - 2836 * hi;
            if w < 0 {
                w += 2147483647;
            }
            v[i] = w as u32;
        }
        for i in 31..34 {
            v[i] = v[i - 31];
        }
        for i in 34..344 {
            v[i] = v[i - 31].wrapping_add(v[i - 3]);
        }
        let mut r = [0u32; 34];
        for i in 310..344 {
            r[i % 34] = v[i];
        }
        GlibcRand { r, k: 344 }
    }

    /// rand(): 0..=RAND_MAX(2^31-1)
    pub fn next(&mut self) -> i32 {
        let k = self.k;
        let x = self.r[(k + 3) % 34].wrapping_add(self.r[(k + 31) % 34]);
        self.r[k % 34] = x;
        self.k = k + 1;
        (x >> 1) as i32
    }

    /// rtflann rand_int(high, 0)
    fn rand_int(&mut self, high: i32) -> i32 {
        (high as f64 * (self.next() as f64 / (2147483647.0 + 1.0))) as i32
    }
}

/// std::mt19937
struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// libstdc++ uniform_int_distribution<unsigned long>{a, b}(mt19937) — 줄이는 경우(Lemire, 64 비트 곱)
    fn uniform(&mut self, a: u64, b: u64) -> u64 {
        let urange = b - a;
        let ret = if urange < u32::MAX as u64 {
            let range = (urange + 1) as u32;
            let mut p = self.next() as u64 * range as u64;
            let mut low = p as u32;
            if low < range {
                let threshold = range.wrapping_neg() % range;
                while low < threshold {
                    p = self.next() as u64 * range as u64;
                    low = p as u32;
                }
            }
            p >> 32
        } else if urange == u32::MAX as u64 {
            self.next() as u64
        } else {
            // 2^32 개 넘는 점(쓰이지 않음)
            unimplemented!("uniform_int_distribution 늘리는 경우")
        };
        ret + a
    }
}

/// libstdc++ 13 std::shuffle(mt19937): 범위가 작으면 한 번 뽑아 두 칸씩 바꾼다
fn shuffle(v: &mut [i32], g: &mut Mt19937) {
    let n = v.len() as u64;
    if n == 0 {
        return;
    }
    if u32::MAX as u64 / n >= n {
        let mut i = 1usize;
        if n % 2 == 0 {
            let j = g.uniform(0, 1) as usize;
            v.swap(i, j);
            i += 1;
        }
        while i != v.len() {
            let r = i as u64 + 1;
            let x = g.uniform(0, r * (r + 1) - 1);
            v.swap(i, (x / (r + 1)) as usize);
            i += 1;
            v.swap(i, (x % (r + 1)) as usize);
            i += 1;
        }
        return;
    }
    for i in 1..v.len() {
        let j = g.uniform(0, i as u64) as usize;
        v.swap(i, j);
    }
}

#[inline]
fn bit(p: &[u64; 4], k: usize) -> f32 {
    ((p[k >> 6] >> (k & 63)) & 1) as f32
}

/// BRIEF 32 바이트 → 비트 묶음(바이트 j 의 비트 b = 원본 float 칸 j*8+b)
#[inline]
fn pack(d: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, c) in d.chunks_exact(8).enumerate() {
        w[i] = u64::from_le_bytes(c.try_into().unwrap());
    }
    w
}

#[inline]
fn pack_bytes(p: &[u64; 4]) -> [u8; 32] {
    let mut o = [0u8; 32];
    for (i, w) in p.iter().enumerate() {
        o[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    o
}

/// 바이트 값마다 8 비트를 칸별 가림(0 또는 모두 1)으로 편 표
static BYTE_MASK: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut v = 0;
    while v < 256 {
        let mut b = 0;
        while b < 8 {
            if (v >> b) & 1 != 0 {
                t[v][b] = u32::MAX;
            }
            b += 1;
        }
        v += 1;
    }
    t
};

/// L1<float> 거리: 0/1 값이라 popcount(xor) 와 비트까지 같다
#[inline]
fn l1(a: &[u64; 4], b: &[u64; 4]) -> f32 {
    ((a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones() + (a[2] ^ b[2]).count_ones() + (a[3] ^ b[3]).count_ones())
        as f32
}

/// 나무 마디. 잎이면 divfeat = 점 번호
#[derive(Clone, Copy)]
struct Node {
    divfeat: i32,
    divval: f32,
    child1: u32,
    child2: u32,
}

#[derive(Clone, Copy)]
struct Branch {
    node: u32,
    mindist: f32,
}

/// rtflann Heap<BranchSt>: libstdc++ push_heap/pop_heap 에 mindist 작은 것이 위(같은 값 순서도 그대로)
struct Heap {
    v: Vec<Branch>,
    length: usize,
}

impl Heap {
    // comp(a, b) = b.mindist < a.mindist
    #[inline]
    fn comp(a: &Branch, b: &Branch) -> bool {
        b.mindist < a.mindist
    }

    fn push_up(v: &mut [Branch], mut hole: usize, top: usize, value: Branch) {
        let mut parent = hole.wrapping_sub(1) / 2;
        while hole > top && Self::comp(&v[parent], &value) {
            v[hole] = v[parent];
            hole = parent;
            parent = hole.wrapping_sub(1) / 2;
        }
        v[hole] = value;
    }

    fn insert(&mut self, b: Branch) {
        if self.v.len() == self.length {
            return;
        }
        self.v.push(b);
        let n = self.v.len();
        Self::push_up(&mut self.v, n - 1, 0, b);
    }

    fn pop_min(&mut self) -> Option<Branch> {
        let n = self.v.len();
        if n == 0 {
            return None;
        }
        let out = self.v[0];
        if n > 1 {
            // __pop_heap: 끝 값을 빼 두고 맨 위 자리를 아래로 내린 뒤 다시 올린다(__adjust_heap)
            let value = self.v[n - 1];
            self.v[n - 1] = self.v[0];
            let len = n - 1;
            let v = &mut self.v[..len];
            let mut hole = 0usize;
            let mut second = 0usize;
            while len > 0 && second < (len - 1) / 2 {
                second = 2 * (second + 1);
                if Self::comp(&v[second], &v[second - 1]) {
                    second -= 1;
                }
                v[hole] = v[second];
                hole = second;
            }
            if len % 2 == 0 && len >= 2 && second == (len - 2) / 2 {
                second = 2 * (second + 1);
                v[hole] = v[second - 1];
                hole = second - 1;
            }
            Self::push_up(v, hole, 0, value);
        }
        self.v.pop();
        Some(out)
    }
}

/// KNNSimpleResultSet (FLANN_FIRST_MATCH 없음)
struct Knn {
    dist: Vec<f32>,
    index: Vec<usize>,
    count: usize,
    worst: f32,
}

impl Knn {
    fn new(k: usize) -> Self {
        Knn { dist: vec![f32::MAX; k], index: vec![usize::MAX; k], count: 0, worst: f32::MAX }
    }
    #[inline]
    fn full(&self) -> bool {
        self.count == self.dist.len()
    }
    #[inline]
    fn add(&mut self, d: f32, idx: usize) {
        if d >= self.worst {
            return;
        }
        let cap = self.dist.len();
        if self.count < cap {
            self.count += 1;
        }
        let mut i = self.count - 1;
        while i > 0 {
            if self.dist[i - 1] > d {
                self.dist[i] = self.dist[i - 1];
                self.index[i] = self.index[i - 1];
            } else {
                break;
            }
            i -= 1;
        }
        self.dist[i] = d;
        self.index[i] = idx;
        self.worst = self.dist[cap - 1];
    }
}

/// FlannIndex(FLANN_INDEX_KDTREE, L1, 재균형) + rtflann KDTreeIndex<L1<float>>
pub struct KdTreeIndex {
    // NNIndex
    pts: Vec<[u64; 4]>,
    size_at_build: usize,
    removed: bool,
    removed_points: Vec<bool>,
    removed_count: usize,
    ids: Vec<usize>,
    last_id: usize,
    // KDTreeIndex
    nodes: Vec<Node>,
    roots: Vec<u32>,
    built: bool,
    // FlannIndex
    next_index: u32,
    /// 지운 점 비율이 이 값을 넘으면 다시 짓는다((factor-1)/factor, factor≤1 이면 1 = 안 지음)
    removed_ratio: f32,
    /// glibc rand() 상태(원본은 프로세스 전체 공유). 기본 srand(1)
    pub rand: GlibcRand,
    seed_source: Box<dyn FnMut() -> u32 + Send + Sync>,
}

impl KdTreeIndex {
    /// rebalancing_factor = Kp/FlannRebalancingFactor(기본 2)
    pub fn new(rebalancing_factor: f32) -> Self {
        let mut k: u32 = 0;
        KdTreeIndex {
            pts: Vec::new(),
            size_at_build: 0,
            removed: false,
            removed_points: Vec::new(),
            removed_count: 0,
            ids: Vec::new(),
            last_id: 0,
            nodes: Vec::new(),
            roots: Vec::new(),
            built: false,
            next_index: 0,
            removed_ratio: if rebalancing_factor <= 1.0 { 1.0 } else { (rebalancing_factor - 1.0) / rebalancing_factor },
            rand: GlibcRand::new(1),
            seed_source: Box::new(move || {
                k = k.wrapping_add(1);
                k.wrapping_mul(2654435761)
            }),
        }
    }

    pub fn from_params(p: &crate::params::Params) -> Self {
        Self::new(p.get("Kp/FlannRebalancingFactor", 2.0f32))
    }

    /// 나무마다 부르는 std::random_device() 값을 바꾼다
    pub fn set_seed_source(&mut self, f: impl FnMut() -> u32 + Send + Sync + 'static) {
        self.seed_source = Box::new(f);
    }

    pub fn is_built(&self) -> bool {
        self.built
    }

    /// 살아 있는 점 수(NNIndex::size)
    pub fn size(&self) -> usize {
        self.pts.len() - self.removed_count
    }

    pub fn removed_count(&self) -> usize {
        self.removed_count
    }

    /// FlannIndex::release: 처음 상태로(rand·씨앗 상태는 그대로)
    pub fn release(&mut self) {
        self.pts.clear();
        self.size_at_build = 0;
        self.removed = false;
        self.removed_points.clear();
        self.removed_count = 0;
        self.ids.clear();
        self.last_id = 0;
        self.nodes.clear();
        self.roots.clear();
        self.built = false;
        self.next_index = 0;
    }

    /// FlannIndex::buildIndex: 새로 만든다. 번호는 0 부터 행 순서
    pub fn build(&mut self, rows: &[[u8; 32]]) {
        self.release();
        self.pts = rows.iter().map(pack).collect();
        self.build_index();
        self.built = true;
        self.next_index = rows.len() as u32;
    }

    /// FlannIndex::addPoints: 나무에 더하고(addPoints(points, 0) 라 자체 재구성 없음),
    /// 지운 점이 너무 많으면 다시 짓는다. 새 점들의 번호를 돌려준다
    pub fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32> {
        if !self.built {
            return Vec::new();
        }
        let old = self.pts.len();
        // extendDataset
        for r in rows {
            self.pts.push(pack(r));
            if self.removed {
                self.removed_points.push(false);
                self.ids.push(self.last_id);
                self.last_id += 1;
            }
        }
        for i in old..self.pts.len() {
            for t in 0..self.roots.len() {
                self.add_point_to_tree(self.roots[t], i);
            }
        }
        // needsRebuild
        let total = self.size() + self.removed_count;
        if self.removed_ratio < 1.0 && total > 0 && (self.removed_count as f32) > self.removed_ratio * total as f32 {
            self.build_index();
        }
        let ids: Vec<u32> = (0..rows.len() as u32).map(|i| self.next_index + i).collect();
        self.next_index += rows.len() as u32;
        ids
    }

    /// FlannIndex::removePoint(번호): 표시만 하고 다음 재구성 때 빠진다
    pub fn remove_point(&mut self, id: u32) {
        if !self.built {
            return;
        }
        if !self.removed {
            self.ids = (0..self.pts.len()).collect();
            self.removed_points = vec![false; self.pts.len()];
            self.last_id = self.pts.len();
            self.removed = true;
        }
        if let Some(i) = self.id_to_index(id as usize) {
            if !self.removed_points[i] {
                self.removed_points[i] = true;
                self.removed_count += 1;
            }
        }
    }

    fn id_to_index(&self, id: usize) -> Option<usize> {
        if self.ids.is_empty() {
            return Some(id);
        }
        if id < self.ids.len() && self.ids[id] == id {
            return Some(id);
        }
        let (mut s, mut e) = (0usize, self.ids.len());
        while s < e {
            let m = (s + e) / 2;
            if self.ids[m] == id {
                return Some(m);
            } else if self.ids[m] < id {
                s = m + 1;
            } else {
                e = m;
            }
        }
        None
    }

    /// NNIndex::buildIndex: 지운 점 정리 → 나무 4 개 새로
    fn build_index(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        // cleanRemovedPoints
        if self.removed {
            let mut last = 0;
            for i in 0..self.pts.len() {
                if !self.removed_points[i] {
                    self.pts[last] = self.pts[i];
                    self.ids[last] = self.ids[i];
                    self.removed_points[last] = false;
                    last += 1;
                }
            }
            self.pts.truncate(last);
            self.ids.truncate(last);
            self.removed_points.truncate(last);
            self.removed_count = 0;
        }
        let n = self.pts.len();
        let mut ind: Vec<i32> = (0..n as i32).collect();
        let mut mean = vec![0f32; DIM];
        let mut var = vec![0f32; DIM];
        for _ in 0..TREES {
            let mut g = Mt19937::new((self.seed_source)());
            shuffle(&mut ind, &mut g);
            let root = self.divide_tree(&mut ind, 0, n, &mut mean, &mut var);
            self.roots.push(root);
        }
        self.size_at_build = n;
    }

    fn divide_tree(&mut self, ind: &mut [i32], s: usize, count: usize, mean: &mut [f32], var: &mut [f32]) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { divfeat: 0, divval: 0.0, child1: NONE, child2: NONE });
        if count == 1 {
            self.nodes[id as usize].divfeat = ind[s];
        } else {
            let (idx, cutfeat, cutval) = self.mean_split(&mut ind[s..s + count], mean, var);
            self.nodes[id as usize].divfeat = cutfeat;
            self.nodes[id as usize].divval = cutval;
            let c1 = self.divide_tree(ind, s, idx, mean, var);
            let c2 = self.divide_tree(ind, s + idx, count - idx, mean, var);
            self.nodes[id as usize].child1 = c1;
            self.nodes[id as usize].child2 = c2;
        }
        id
    }

    /// 앞쪽 최대 101 점의 평균·분산(원본 float 순서 그대로) → 분산 큰 축 5 개 중 rand 로 하나 → 평균에서 자른다
    fn mean_split(&mut self, ind: &mut [i32], mean: &mut [f32], var: &mut [f32]) -> (usize, i32, f32) {
        let count = ind.len();
        let cnt = (SAMPLE_MEAN + 1).min(count);
        // 원본은 0/1 float 를 차례로 더한다: 합은 1 의 개수와 정확히 같으므로 개수를 세어 float 로 바꾼다
        // 바이트 값 → 8 칸 비트 가림(0 / 0xFFFFFFFF). 칸 k = 바이트 k/8 의 비트 k%8
        let mut ones = [0u32; DIM];
        for &j in &ind[..cnt] {
            let bytes = pack_bytes(&self.pts[j as usize]);
            for (c, &by) in bytes.iter().enumerate() {
                let m = &BYTE_MASK[by as usize];
                let o = &mut ones[c * 8..c * 8 + 8];
                for l in 0..8 {
                    o[l] += m[l] & 1;
                }
            }
        }
        let div = 1.0f32 / cnt as f32;
        // 분산: 칸마다 더하는 값은 (0-평균)² 또는 (1-평균)² 둘뿐. 원본과 같은 점 순서로 그 값을 더한다
        let mut sq0 = [0u32; DIM];
        let mut sqx = [0u32; DIM];
        for k in 0..DIM {
            let m = ones[k] as f32 * div;
            mean[k] = m;
            let (d0, d1) = (0.0f32 - m, 1.0f32 - m);
            sq0[k] = (d0 * d0).to_bits();
            sqx[k] = sq0[k] ^ (d1 * d1).to_bits();
        }
        let mut acc = [0f32; DIM];
        for &j in &ind[..cnt] {
            let bytes = pack_bytes(&self.pts[j as usize]);
            for (c, &by) in bytes.iter().enumerate() {
                let m = &BYTE_MASK[by as usize];
                let r = c * 8..c * 8 + 8;
                let (v, a, x) = (&mut acc[r.clone()], &sq0[r.clone()], &sqx[r]);
                for l in 0..8 {
                    v[l] += f32::from_bits(a[l] ^ (x[l] & m[l]));
                }
            }
        }
        var[..DIM].copy_from_slice(&acc);
        // selectDivision
        let mut num = 0usize;
        let mut top = [0usize; RAND_DIM];
        for i in 0..DIM {
            if num < RAND_DIM || var[i] > var[top[num - 1]] {
                if num < RAND_DIM {
                    top[num] = i;
                    num += 1;
                } else {
                    top[num - 1] = i;
                }
                let mut j = num - 1;
                while j > 0 && var[top[j]] > var[top[j - 1]] {
                    top.swap(j, j - 1);
                    j -= 1;
                }
            }
        }
        let cutfeat = top[self.rand.rand_int(num as i32) as usize];
        let cutval = mean[cutfeat];
        // planeSplit
        let v = |pts: &Vec<[u64; 4]>, i: i32| bit(&pts[i as usize], cutfeat);
        let mut left = 0isize;
        let mut right = count as isize - 1;
        loop {
            while left <= right && v(&self.pts, ind[left as usize]) < cutval {
                left += 1;
            }
            while left <= right && v(&self.pts, ind[right as usize]) >= cutval {
                right -= 1;
            }
            if left > right {
                break;
            }
            ind.swap(left as usize, right as usize);
            left += 1;
            right -= 1;
        }
        let lim1 = left as usize;
        right = count as isize - 1;
        loop {
            while left <= right && v(&self.pts, ind[left as usize]) <= cutval {
                left += 1;
            }
            while left <= right && v(&self.pts, ind[right as usize]) > cutval {
                right -= 1;
            }
            if left > right {
                break;
            }
            ind.swap(left as usize, right as usize);
            left += 1;
            right -= 1;
        }
        let lim2 = left as usize;
        let mut index = if lim1 > count / 2 {
            lim1
        } else if lim2 < count / 2 {
            lim2
        } else {
            count / 2
        };
        if lim1 == count || lim2 == 0 {
            index = count / 2;
        }
        (index, cutfeat as i32, cutval)
    }

    /// 잎까지 내려가 그 잎을 둘로 나눈다(가장 처음 다른 축에서 반으로)
    fn add_point_to_tree(&mut self, mut node: u32, ind: usize) {
        let p = self.pts[ind];
        loop {
            let n = self.nodes[node as usize];
            if n.child1 == NONE && n.child2 == NONE {
                let leaf = self.pts[n.divfeat as usize];
                let mut max_span = 0f32;
                let mut div_feat = 0usize;
                for i in 0..DIM {
                    let span = (bit(&p, i) - bit(&leaf, i)).abs();
                    if span > max_span {
                        max_span = span;
                        div_feat = i;
                    }
                }
                let (a, b) = (bit(&p, div_feat), bit(&leaf, div_feat));
                let (l, r) = if a < b { (ind as i32, n.divfeat) } else { (n.divfeat, ind as i32) };
                let left = self.nodes.len() as u32;
                self.nodes.push(Node { divfeat: l, divval: 0.0, child1: NONE, child2: NONE });
                self.nodes.push(Node { divfeat: r, divval: 0.0, child1: NONE, child2: NONE });
                let m = &mut self.nodes[node as usize];
                m.divfeat = div_feat as i32;
                m.divval = (a + b) / 2.0;
                m.child1 = left;
                m.child2 = left + 1;
                return;
            }
            node = if bit(&p, n.divfeat as usize) < n.divval { n.child1 } else { n.child2 };
        }
    }

    /// FlannIndex::knnSearch(query, k, checks, eps=0, sorted). 행마다 k 칸: 번호(없으면 -1), 거리(없으면 -1)
    pub fn knn_search(&self, query: &[[u8; 32]], k: usize, checks: i32) -> (Vec<i32>, Vec<f32>) {
        let mut idx = vec![-1i32; query.len() * k];
        let mut dist = vec![-1f32; query.len() * k];
        if !self.built || k == 0 {
            return (idx, dist);
        }
        // 원본은 질의마다 힙·확인표를 새로 잡는다. 여기서는 비워서 다시 쓴다(결과 같음)
        let n = self.pts.len();
        let mut rs = Knn::new(k);
        let mut heap = Heap { v: Vec::with_capacity(256), length: n };
        let mut checked = vec![0u64; n.div_ceil(64)];
        for (qi, q) in query.iter().enumerate() {
            heap.v.clear();
            checked.fill(0);
            self.search_one(&pack(q), &mut rs, checks, &mut heap, &mut checked, &mut idx[qi * k..(qi + 1) * k], &mut dist[qi * k..(qi + 1) * k]);
        }
        (idx, dist)
    }

    #[allow(clippy::too_many_arguments)]
    fn search_one(&self, q: &[u64; 4], rs: &mut Knn, checks: i32, heap: &mut Heap, checked: &mut [u64], out_i: &mut [i32], out_d: &mut [f32]) {
        rs.count = 0;
        rs.worst = f32::MAX;
        let cap = rs.dist.len();
        rs.dist[cap - 1] = f32::MAX;
        let mut cc = 0i32;
        let eps = 1.0f32 + 0.0;
        for &r in &self.roots {
            self.search_level(rs, q, r, 0.0, &mut cc, checks, eps, heap, checked);
        }
        while let Some(b) = heap.pop_min() {
            if !(cc < checks || !rs.full()) {
                break;
            }
            self.search_level(rs, q, b.node, b.mindist, &mut cc, checks, eps, heap, checked);
        }
        for i in 0..rs.count.min(cap) {
            let id = if self.removed { self.ids[rs.index[i]] } else { rs.index[i] };
            out_i[i] = id as i32;
            out_d[i] = rs.dist[i];
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn search_level(
        &self,
        rs: &mut Knn,
        q: &[u64; 4],
        mut node: u32,
        mindist: f32,
        cc: &mut i32,
        max_check: i32,
        eps: f32,
        heap: &mut Heap,
        checked: &mut [u64],
    ) {
        loop {
            if rs.worst < mindist {
                return;
            }
            let n = &self.nodes[node as usize];
            if n.child1 == NONE && n.child2 == NONE {
                let index = n.divfeat as usize;
                if self.removed && self.removed_points[index] {
                    return;
                }
                if (checked[index >> 6] >> (index & 63)) & 1 != 0 || (*cc >= max_check && rs.full()) {
                    return;
                }
                checked[index >> 6] |= 1 << (index & 63);
                *cc += 1;
                rs.add(l1(&self.pts[index], q), index);
                return;
            }
            let val = bit(q, n.divfeat as usize);
            let diff = val - n.divval;
            let (best, other) = if diff < 0.0 { (n.child1, n.child2) } else { (n.child2, n.child1) };
            let new_distsq = mindist + (val - n.divval).abs();
            if new_distsq * eps < rs.worst || !rs.full() {
                heap.insert(Branch { node: other, mindist: new_distsq });
            }
            node = best;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glibc_rand_srand1() {
        // glibc srand(1) 뒤 rand() 첫 값들
        let mut g = GlibcRand::new(1);
        assert_eq!([g.next(), g.next(), g.next()], [1804289383, 846930886, 1681692777]);
    }

    #[test]
    fn mt19937_10000th() {
        // C++ 표준이 정한 값: 기본 씨앗 5489 의 10000 번째 = 4123659995
        let mut g = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next();
        }
        assert_eq!(v, 4123659995);
    }

    /// 짓기 결과(마디·검색 결과)의 지문과 시간. 지문은 원본 그대로 옮긴 판(점마다 float 칸 더하기)에서 잰 값
    #[test]
    fn build_fingerprint() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let rows: Vec<[u8; 32]> = (0..1000).map(|_| {
            let mut d = [0u8; 32];
            for c in d.chunks_exact_mut(8) {
                c.copy_from_slice(&next().to_le_bytes());
            }
            d
        }).collect();
        let mut h = 0xcbf29ce484222325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        let t0 = std::time::Instant::now();
        let reps = 20;
        let mut idx = KdTreeIndex::new(1.0);
        for _ in 0..reps {
            idx.build(&rows);
        }
        let tb = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        for n in &idx.nodes {
            mix(n.divfeat as u64);
            mix(n.divval.to_bits() as u64);
            mix(n.child1 as u64);
            mix(n.child2 as u64);
        }
        let t1 = std::time::Instant::now();
        let (i, d) = idx.knn_search(&rows[..500], 2, 32);
        let ts = t1.elapsed().as_secs_f64() * 1e3;
        for (a, b) in i.iter().zip(&d) {
            mix(*a as u64);
            mix(b.to_bits() as u64);
        }
        eprintln!("짓기 {tb:.2} ms, 찾기(500) {ts:.2} ms, 지문 {h:016x}");
        assert_eq!(h, 0xe2629005165aef3d, "지문 {h:016x}");
    }

    #[test]
    fn exact_self_match() {
        // 넣은 점 자신은 거리 0 으로 첫 칸에 나온다
        let rows: Vec<[u8; 32]> = (0..500u32).map(|i| {
            let mut d = [0u8; 32];
            for (j, b) in d.iter_mut().enumerate() {
                *b = (i.wrapping_mul(2654435761).rotate_left(j as u32) >> 3) as u8;
            }
            d
        }).collect();
        let mut idx = KdTreeIndex::new(2.0);
        idx.build(&rows[..100]);
        idx.add_points(&rows[100..]);
        let (i, d) = idx.knn_search(&rows, 2, 32);
        let hits = (0..rows.len()).filter(|&q| i[2 * q] == q as i32 && d[2 * q] == 0.0).count();
        assert!(hits > 400, "{hits}");
    }
}
