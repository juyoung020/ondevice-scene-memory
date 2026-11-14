//! graph::findNearestNodes 가 쓰는 pcl::search::KdTree<PointXYZ>(KdTreeFLANN) = FLANN 1.9 KDTreeSingleIndex<L2_Simple<float>>(잎 15).
//! 원본: flann/algorithms/kdtree_single_index.h(buildIndexImpl · divideTree · middleSplit · planeSplit · computeInitialDistances · searchLevel),
//!       flann/util/result_set.h(KNNSimpleResultSet · RadiusResultSet), pcl/kdtree/impl/kdtree_flann.hpp(nearestKSearch · radiusSearch).
//! 같은 거리일 때 어느 점이 남는지는 나무를 도는 순서로 정해지므로 나무를 그대로 옮겼다.
//! 우리 경로의 두 경우만: k 개 찾기(반지름 0) · 반지름 안 전부(k = 0). 반지름과 k 를 함께 주는 경우(KNNRadiusResultSet)는 쓰지 않는다.

struct Node {
    left: usize,
    right: usize,
    divfeat: usize,
    divlow: f32,
    divhigh: f32,
    child: Option<(usize, usize)>,
}

pub struct KdSingle {
    pts: Vec<[f32; 3]>,
    vind: Vec<usize>,
    nodes: Vec<Node>,
    root: usize,
    bbox: [(f32, f32); 3],
}

const LEAF: usize = 15;

impl KdSingle {
    pub fn new(pts: Vec<[f32; 3]>) -> KdSingle {
        let n = pts.len();
        let mut t = KdSingle { pts, vind: (0..n).collect(), nodes: Vec::new(), root: 0, bbox: [(0.0, 0.0); 3] };
        let mut bb = [(t.pts[0][0], t.pts[0][0]), (t.pts[0][1], t.pts[0][1]), (t.pts[0][2], t.pts[0][2])];
        for p in &t.pts[1..] {
            for i in 0..3 {
                if p[i] < bb[i].0 {
                    bb[i].0 = p[i];
                }
                if p[i] > bb[i].1 {
                    bb[i].1 = p[i];
                }
            }
        }
        t.bbox = bb;
        let mut b = bb;
        t.root = t.divide(0, n, &mut b);
        t
    }

    fn divide(&mut self, left: usize, right: usize, bbox: &mut [(f32, f32); 3]) -> usize {
        if right - left <= LEAF {
            let p0 = self.pts[self.vind[left]];
            for i in 0..3 {
                bbox[i] = (p0[i], p0[i]);
            }
            for k in left + 1..right {
                let p = self.pts[self.vind[k]];
                for i in 0..3 {
                    if bbox[i].0 > p[i] {
                        bbox[i].0 = p[i];
                    }
                    if bbox[i].1 < p[i] {
                        bbox[i].1 = p[i];
                    }
                }
            }
            self.nodes.push(Node { left, right, divfeat: 0, divlow: 0.0, divhigh: 0.0, child: None });
            return self.nodes.len() - 1;
        }
        let (idx, cutfeat, cutval) = self.middle_split(left, right - left, bbox);
        let me = self.nodes.len();
        self.nodes.push(Node { left: 0, right: 0, divfeat: cutfeat, divlow: 0.0, divhigh: 0.0, child: None });
        let mut lb = *bbox;
        lb[cutfeat].1 = cutval;
        let c1 = self.divide(left, left + idx, &mut lb);
        let mut rb = *bbox;
        rb[cutfeat].0 = cutval;
        let c2 = self.divide(left + idx, right, &mut rb);
        let n = &mut self.nodes[me];
        n.divlow = lb[cutfeat].1;
        n.divhigh = rb[cutfeat].0;
        n.child = Some((c1, c2));
        for i in 0..3 {
            bbox[i].0 = lb[i].0.min(rb[i].0);
            bbox[i].1 = lb[i].1.max(rb[i].1);
        }
        me
    }

    fn min_max(&self, off: usize, count: usize, dim: usize) -> (f32, f32) {
        let mut mn = self.pts[self.vind[off]][dim];
        let mut mx = mn;
        for i in 1..count {
            let v = self.pts[self.vind[off + i]][dim];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        (mn, mx)
    }

    fn middle_split(&mut self, off: usize, count: usize, bbox: &[(f32, f32); 3]) -> (usize, usize, f32) {
        let mut max_span = bbox[0].1 - bbox[0].0;
        let mut cutfeat = 0;
        for i in 1..3 {
            let span = bbox[i].1 - bbox[i].0;
            if span > max_span {
                max_span = span;
                cutfeat = i;
            }
        }
        let (mn, mx) = self.min_max(off, count, cutfeat);
        let mut cutval = (mn + mx) / 2.0;
        max_span = mx - mn;
        let k = cutfeat;
        for i in 0..3 {
            if i == k {
                continue;
            }
            let span = bbox[i].1 - bbox[i].0;
            if span > max_span {
                let (mn, mx) = self.min_max(off, count, i);
                let span = mx - mn;
                if span > max_span {
                    max_span = span;
                    cutfeat = i;
                    cutval = (mn + mx) / 2.0;
                }
            }
        }
        let (lim1, lim2) = self.plane_split(off, count, cutfeat, cutval);
        let index = if lim1 > count / 2 {
            lim1
        } else if lim2 < count / 2 {
            lim2
        } else {
            count / 2
        };
        (index, cutfeat, cutval)
    }

    fn plane_split(&mut self, off: usize, count: usize, f: usize, cv: f32) -> (usize, usize) {
        let v = |s: &Self, i: isize| s.pts[s.vind[off + i as usize]][f];
        let mut left: isize = 0;
        let mut right: isize = count as isize - 1;
        loop {
            while left <= right && v(self, left) < cv {
                left += 1;
            }
            while left <= right && v(self, right) >= cv {
                right -= 1;
            }
            if left > right {
                break;
            }
            self.vind.swap(off + left as usize, off + right as usize);
            left += 1;
            right -= 1;
        }
        let lim1 = left as usize;
        right = count as isize - 1;
        loop {
            while left <= right && v(self, left) <= cv {
                left += 1;
            }
            while left <= right && v(self, right) > cv {
                right -= 1;
            }
            if left > right {
                break;
            }
            self.vind.swap(off + left as usize, off + right as usize);
            left += 1;
            right -= 1;
        }
        (lim1, left as usize)
    }

    fn search(&self, rs: &mut Rs, q: &[f32; 3]) {
        let mut dists = [0f32; 3];
        let mut d = 0f32;
        for i in 0..3 {
            if q[i] < self.bbox[i].0 {
                dists[i] = (q[i] - self.bbox[i].0) * (q[i] - self.bbox[i].0);
                d += dists[i];
            }
            if q[i] > self.bbox[i].1 {
                dists[i] = (q[i] - self.bbox[i].1) * (q[i] - self.bbox[i].1);
                d += dists[i];
            }
        }
        self.level(rs, q, self.root, d, &mut dists);
    }

    fn level(&self, rs: &mut Rs, q: &[f32; 3], node: usize, mindist: f32, dists: &mut [f32; 3]) {
        let n = &self.nodes[node];
        let Some((c1, c2)) = n.child else {
            let worst = rs.worst();
            for i in n.left..n.right {
                let p = self.pts[self.vind[i]];
                let mut dist = 0f32;
                for k in 0..3 {
                    let df = q[k] - p[k];
                    dist += df * df;
                }
                if dist < worst {
                    rs.add(dist, self.vind[i]);
                }
            }
            return;
        };
        let idx = n.divfeat;
        let val = q[idx];
        let d1 = val - n.divlow;
        let d2 = val - n.divhigh;
        let (best, other, cut) = if d1 + d2 < 0.0 { (c1, c2, (val - n.divhigh) * (val - n.divhigh)) } else { (c2, c1, (val - n.divlow) * (val - n.divlow)) };
        self.level(rs, q, best, mindist, dists);
        let dst = dists[idx];
        let mindist = mindist + cut - dst;
        dists[idx] = cut;
        // epsError = 1 + eps(0)
        if mindist * 1.0 <= rs.worst() {
            self.level(rs, q, other, mindist, dists);
        }
        dists[idx] = dst;
    }

    /// nearestKSearch: (점 순번, 거리²) 가까운 순
    pub fn knn(&self, q: &[f32; 3], k: usize) -> Vec<(usize, f32)> {
        let k = k.min(self.pts.len());
        if k == 0 {
            return Vec::new();
        }
        let mut rs = Rs::Knn { cap: k, v: Vec::new(), worst: f32::MAX };
        self.search(&mut rs, q);
        match rs {
            Rs::Knn { v, .. } => v,
            _ => unreachable!(),
        }
    }

    /// radiusSearch(max_nn = 0): 거리² < r² 인 점 전부(가까운 순)
    pub fn radius(&self, q: &[f32; 3], radius: f64) -> Vec<(usize, f32)> {
        let r2 = (radius * radius) as f32;
        let mut rs = Rs::Radius { r: r2, v: Vec::new() };
        self.search(&mut rs, q);
        let Rs::Radius { mut v, .. } = rs else { unreachable!() };
        // DistanceIndex operator<: 거리, 같으면 순번
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        v
    }
}

enum Rs {
    /// KNNSimpleResultSet
    Knn { cap: usize, v: Vec<(usize, f32)>, worst: f32 },
    /// RadiusResultSet
    Radius { r: f32, v: Vec<(usize, f32)> },
}

impl Rs {
    fn worst(&self) -> f32 {
        match self {
            Rs::Knn { worst, .. } => *worst,
            Rs::Radius { r, .. } => *r,
        }
    }
    fn add(&mut self, dist: f32, idx: usize) {
        match self {
            Rs::Knn { cap, v, worst } => {
                if dist >= *worst {
                    return;
                }
                if v.len() < *cap {
                    v.push((usize::MAX, f32::MAX));
                }
                let mut i = v.len() - 1;
                while i > 0 && v[i - 1].1 > dist {
                    v[i] = v[i - 1];
                    i -= 1;
                }
                v[i] = (idx, dist);
                if v.len() == *cap {
                    *worst = v[*cap - 1].1;
                }
            }
            Rs::Radius { r, v } => {
                if dist < *r {
                    v.push((idx, dist));
                }
            }
        }
    }
}
