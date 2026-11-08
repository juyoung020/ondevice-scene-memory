//! 서브맵 표면(Voxblox·voxgraph·cblox 방식): 표면을 짧은 구간마다 따로 쌓은 TSDF(서브맵)로 나누고, 서브맵을 RTAB-Map 노드에 붙인다.
//! 루프 클로저로 노드가 움직이면 장면을 다시 넣지 않고 서브맵 자세만 바꿔 전체 표면을 다시 합친다(cblox getProjectedMap).
//! - 서브맵 좌표: 만들 때 지도 좌표와 같은 방향(회전 없음), 원점은 카메라 위치를 블록 격자에 맞춘 곳.
//!   그래서 노드가 안 움직인 서브맵은 칸이 전체 표면 칸과 그대로 겹쳐 보간 없이 더해진다.
//! - 서브맵 자세 = 기준 노드의 지금 자세 × (만들 때 기준 노드에서 본 서브맵 원점).
//! - 지금 쌓는 서브맵(활성)은 바뀐 블록만 전체 표면에 옮긴다: 지난번 옮긴 값(그림자)을 빼고 지금 값을 더함.
use crate::geom::*;
use crate::tsdf::{BKey, Pose, Tsdf};
use std::sync::Arc;

/// 서브맵 자세가 이만큼 넘게 바뀌면 전체 표면을 다시 합칠 거리(m)·각도(°): 장면 보관(kfstore)과 같은 문턱
const MOVE_T: f64 = 0.02;
const MOVE_R: f64 = 1.0;

pub fn mul(a: &Pose, b: &Pose) -> Pose {
    (mm(&a.0, &b.0), add(&mv(&a.0, &b.1), &a.1))
}

/// a⁻¹ · b
pub fn inv_mul(a: &Pose, b: &Pose) -> Pose {
    let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| a.0[j][i]));
    (mm(&rt, &b.0), mv(&rt, &sub(&b.1, &a.1)))
}

pub fn angle(a: &M3, b: &M3) -> f64 {
    let mut tr = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            tr += a[j][i] * b[j][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

fn moved(a: &Pose, b: &Pose) -> (f64, f64) {
    (norm(&sub(&a.1, &b.1)), angle(&a.0, &b.0))
}

const ID: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// 다 쌓은 서브맵
pub struct Submap {
    anchor: u32, // 기준 노드(0 = 없음: 자세 고정)
    rel: Pose,   // 기준 노드 → 서브맵 원점(만들 때 고정). 기준 노드가 없으면 서브맵 자세 그대로
    pub placed: Pose, // 전체 표면에 지금 놓인 자세
    pub tsdf: Arc<Tsdf>,
}

impl Submap {
    /// 지금 노드 위치로 본 자세(node: 노드 id → 지도 좌표 자세)
    pub fn pose(&self, node: &dyn Fn(u32) -> Option<Pose>) -> Pose {
        match (self.anchor != 0).then(|| node(self.anchor)).flatten() {
            Some(n) => mul(&n, &self.rel),
            None => if self.anchor == 0 { self.rel } else { self.placed },
        }
    }
}

struct Active {
    sm: Submap,      // tsdf 는 쌓는 동안 비워 둠(아래 tsdf 를 씀)
    tsdf: Tsdf,
    shadow: Tsdf,    // 전체 표면에 마지막으로 옮긴 값
    t0: f64,
    origin: Pose,    // 만들 때 카메라 자세(지도 좌표)
}

/// 새 서브맵을 여는 조건: 기준 노드가 바뀌었고, 지금 서브맵이 secs 초 넘었거나 카메라가 dist m·deg ° 넘게 움직였을 때
#[derive(Clone, Copy)]
pub struct Policy {
    pub secs: f64,
    pub dist: f64,
    pub deg: f64,
}

pub struct Submaps {
    pub vox: f64,
    pub policy: Policy,
    pub list: Vec<Submap>,
    active: Option<Active>,
    pub global: Tsdf, // 전체 표면(면 파일은 여기서 뽑음)
    pub dirty: bool,  // 서브맵 자세가 바뀜 → 전체 표면을 다시 합쳐야 함
    pub moved: (f64, f64),
}

/// 작업 스레드에 넘길 다시 합치기 일감: 다 쌓은 서브맵과 놓을 자세
pub struct Job {
    vox: f64,
    items: Vec<(Arc<Tsdf>, Pose, Option<Pose>)>, // (서브맵, 놓을 자세, 움직였으면 옛 자세)
}

pub struct Done {
    global: Tsdf,
    poses: Vec<Pose>,
    dirty: FastSet<BKey>,
    pub n: usize,
    pub n_moved: usize,
    pub secs: f64,
}

impl Submaps {
    pub fn new(vox: f64, policy: Policy) -> Submaps {
        Submaps { vox, policy, list: vec![], active: None, global: Tsdf::new(vox), dirty: false, moved: (0.0, 0.0) }
    }

    /// 서브맵 수(활성 포함)
    pub fn len(&self) -> usize {
        self.list.len() + self.active.is_some() as usize
    }

    /// 서브맵 칸 배열 바이트 합(전체 표면 제외)
    pub fn bytes(&self) -> usize {
        self.list.iter().map(|s| s.tsdf.bytes()).sum::<usize>() + self.active.as_ref().map_or(0, |a| a.tsdf.bytes() + a.shadow.bytes())
    }

    /// 한 장면 넣기: cam = 카메라 → 지도(바닥 높이 맞춘 뒤), anchor: 이 장면의 기준 노드
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(&mut self, t: f64, anchor: u32, node: &dyn Fn(u32) -> Option<Pose>, cam: &Pose, depth: &[f32], rgb: &[u8], w: usize, h: usize, k: [f64; 4], max_depth: f64) {
        let p = self.policy;
        if let Some(a) = &self.active {
            let (dt, dr) = moved(&a.origin, cam);
            if anchor != 0 && anchor != a.sm.anchor && (t - a.t0 >= p.secs || dt >= p.dist || dr >= p.deg) {
                self.close();
            }
        }
        if self.active.is_none() {
            let bs = self.global.block_size();
            let s0: Pose = (ID, cam.1.map(|v| (v / bs).round() * bs));
            let (anc, rel) = match (anchor != 0).then(|| node(anchor)).flatten() {
                Some(n) => (anchor, inv_mul(&n, &s0)),
                None => (0, s0),
            };
            self.active = Some(Active {
                sm: Submap { anchor: anc, rel, placed: s0, tsdf: Arc::new(Tsdf::new(self.vox)) },
                tsdf: Tsdf::new(self.vox), shadow: Tsdf::new(self.vox), t0: t, origin: *cam,
            });
        }
        let a = self.active.as_mut().unwrap();
        let loc = inv_mul(&a.sm.placed, cam);
        a.tsdf.integrate(depth, rgb, w, h, k, &loc.0, &loc.1, max_depth);
    }

    /// 활성 서브맵의 바뀐 블록을 전체 표면에 옮긴다(면 파일을 쓰기 전에)
    pub fn flush(&mut self) {
        let Some(a) = self.active.as_mut() else { return };
        let regs = a.tsdf.drain_dirty_regions();
        if regs.is_empty() {
            return;
        }
        let keys = a.tsdf.blocks_in_regions(&regs);
        self.global.merge(&a.shadow, &a.sm.placed, -1.0, Some(&keys));
        self.global.merge(&a.tsdf, &a.sm.placed, 1.0, Some(&keys));
        a.shadow.copy_blocks_from(&a.tsdf, &keys);
    }

    /// 활성 서브맵을 닫아 다 쌓은 목록으로
    pub fn close(&mut self) {
        self.flush();
        if let Some(a) = self.active.take() {
            let mut sm = a.sm;
            sm.tsdf = Arc::new(a.tsdf);
            self.list.push(sm);
        }
    }

    /// 그래프 블록 뒤: 놓인 자세에서 문턱 넘게 바뀐 서브맵이 있으면 다시 합치기 표시. 활성 서브맵이 움직였으면 닫는다(새 서브맵은 다시 격자에 맞춰 열림).
    pub fn check(&mut self, node: &dyn Fn(u32) -> Option<Pose>) {
        if let Some(a) = &self.active {
            let (dt, dr) = moved(&a.sm.pose(node), &a.sm.placed);
            if dt > MOVE_T || dr > MOVE_R {
                self.close();
            }
        }
        for s in &self.list {
            let (dt, dr) = moved(&s.pose(node), &s.placed);
            if dt > MOVE_T || dr > MOVE_R {
                self.dirty = true;
                self.moved = (self.moved.0.max(dt), self.moved.1.max(dr));
            }
        }
    }

    /// 다시 합치기 일감: 활성 서브맵을 닫고, 다 쌓은 서브맵을 문턱 넘게 움직였으면 새 자세로, 아니면 놓인 자세 그대로
    pub fn plan(&mut self, node: &dyn Fn(u32) -> Option<Pose>) -> Job {
        self.close();
        self.dirty = false;
        self.moved = (0.0, 0.0);
        let items = self
            .list
            .iter()
            .map(|s| {
                let now = s.pose(node);
                let (dt, dr) = moved(&now, &s.placed);
                if dt > MOVE_T || dr > MOVE_R { (s.tsdf.clone(), now, Some(s.placed)) } else { (s.tsdf.clone(), s.placed, None) }
            })
            .collect();
        Job { vox: self.vox, items }
    }

    /// 다시 합친 전체 표면으로 갈아 끼운다. 일감 뒤에 닫힌 서브맵·활성 서브맵(그림자)은 놓인 자세로 더한다.
    /// 판 번호는 이어받고, 움직인 서브맵이 덮던 곳·덮는 곳만 면을 다시 뽑는다.
    pub fn install(&mut self, mut d: Done) {
        for (s, p) in self.list.iter_mut().zip(&d.poses) {
            s.placed = *p;
        }
        for s in &self.list[d.n..] {
            d.global.merge(&s.tsdf, &s.placed, 1.0, None);
        }
        if let Some(a) = &self.active {
            d.global.merge(&a.shadow, &a.sm.placed, 1.0, None);
        }
        d.global.adopt_versions(&mut self.global, d.dirty);
        self.global = d.global;
    }
}

impl Job {
    /// 빈 전체 표면에 서브맵을 하나씩 옮겨 더한다(작업 스레드)
    pub fn run(self) -> Done {
        let c0 = std::time::Instant::now();
        let mut global = Tsdf::new(self.vox);
        let mut dirty = FastSet::default();
        let mut n_moved = 0;
        for (ts, pose, old) in &self.items {
            global.merge(ts, pose, 1.0, None);
            if let Some(o) = old {
                n_moved += 1;
                dirty.extend(Tsdf::footprint(ts, o));
                dirty.extend(Tsdf::footprint(ts, pose));
            }
        }
        Done { global, poses: self.items.iter().map(|x| x.1).collect(), dirty, n: self.items.len(), n_moved, secs: c0.elapsed().as_secs_f64() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rz(a: f64) -> M3 {
        let (s, c) = a.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }
    fn ry(a: f64) -> M3 {
        let (s, c) = a.sin_cos();
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    }

    /// 카메라 앞 z = 1 m 의 평평한 벽(80×60, 보정값 k)
    fn wall() -> (Vec<f32>, Vec<u8>, usize, usize, [f64; 4]) {
        let (w, h) = (80, 60);
        (vec![1.0; w * h], vec![128; w * h * 3], w, h, [60.0, 60.0, 40.0, 30.0])
    }

    #[test]
    fn pose_follows_anchor() {
        // 노드 7 이 (1, 0, 0) 에 있을 때 서브맵을 열고, 노드가 원점 기준 90° 돌아 (0, 1, 0) 으로 가면 서브맵도 같이 돈다
        let mut node_p: Pose = (ID, [1.0, 0.0, 0.0]);
        let mut sm = Submaps::new(0.05, Policy { secs: 5.0, dist: 1.0, deg: 30.0 });
        let (d, c, w, h, k) = wall();
        let cam: Pose = (ID, [1.5, 0.0, 0.0]);
        {
            let np = node_p;
            sm.integrate(0.0, 7, &|id| (id == 7).then_some(np), &cam, &d, &c, w, h, k, 3.0);
        }
        sm.close();
        let s0 = sm.list[0].placed;
        assert!(sm.global.aligned(&s0).is_some());
        node_p = (rz(std::f64::consts::FRAC_PI_2), [0.0, 1.0, 0.0]);
        let np = node_p;
        let node = move |id: u32| (id == 7).then_some(np);
        sm.check(&node);
        assert!(sm.dirty);
        let now = sm.list[0].pose(&node);
        // 서브맵 원점은 노드에서 (x0 − 1, 0, 0) 앞 → 돌린 뒤 (0, 1 + x0 − 1, 0)
        let x0 = s0.1[0];
        assert!((now.1[0]).abs() < 1e-9 && (now.1[1] - x0).abs() < 1e-9, "{:?}", now.1);
        assert!(angle(&now.0, &rz(std::f64::consts::FRAC_PI_2)) < 1e-6);
    }

    #[test]
    fn new_submap_needs_anchor_change() {
        let mut sm = Submaps::new(0.05, Policy { secs: 5.0, dist: 1.0, deg: 30.0 });
        let (d, c, w, h, k) = wall();
        let node = |_id: u32| Some((ID, [0.0; 3]));
        for i in 0..20 {
            let cam: Pose = (ID, [0.0; 3]);
            sm.integrate(i as f64, 1, &node, &cam, &d, &c, w, h, k, 3.0);
        }
        assert_eq!(sm.len(), 1); // 오래됐어도 기준 노드가 같으면 그대로
        sm.integrate(20.0, 2, &node, &(ID, [0.0; 3]), &d, &c, w, h, k, 3.0);
        assert_eq!(sm.len(), 2);
        sm.integrate(21.0, 3, &node, &(ID, [0.0; 3]), &d, &c, w, h, k, 3.0);
        assert_eq!(sm.len(), 2); // 노드는 바뀌었지만 1 초·제자리
        sm.integrate(22.0, 4, &node, &(ID, [1.2, 0.0, 0.0]), &d, &c, w, h, k, 3.0);
        assert_eq!(sm.len(), 3); // 1 m 넘게 움직임
    }

    #[test]
    fn flush_equals_direct() {
        // 움직임 없으면: 서브맵 여러 개를 합친 전체 표면 = 한 TSDF 에 바로 쌓은 것(무게 상한 전까지)
        let (d, c, w, h, k) = wall();
        let mut direct = Tsdf::new(0.05);
        let mut sm = Submaps::new(0.05, Policy { secs: 1.0, dist: 1.0, deg: 30.0 });
        let node = |_id: u32| Some((ID, [0.0; 3]));
        for i in 0..6 {
            let cam: Pose = (ry(0.05 * i as f64), [0.1 * i as f64, 0.02 * i as f64, 0.0]);
            direct.integrate(&d, &c, w, h, k, &cam.0, &cam.1, 3.0);
            sm.integrate(i as f64, i + 1, &node, &cam, &d, &c, w, h, k, 3.0);
            if i % 2 == 0 {
                sm.flush();
            }
        }
        sm.flush();
        assert!(sm.len() >= 3);
        let a = direct.voxels();
        assert!(!a.is_empty());
        for (x, dd, ww) in &a {
            let (gd, gw) = sm.global.at(x).expect("전체 표면에 칸 있음");
            assert!((gw - ww).abs() < 1e-4 && (gd - dd).abs() < 1e-4, "{x:?}: {dd} {ww} 대 {gd} {gw}");
        }
        assert_eq!(sm.global.voxels().len(), a.len());
    }

    #[test]
    fn merge_rotated_and_remove() {
        // 벽을 서브맵 좌표에 쌓고 30° 돌려 옮김: 옮긴 칸의 거리 값이 돌린 벽까지 거리와 맞고, 같은 자세로 빼면 비어야 한다
        let (d, c, w, h, k) = wall();
        let mut src = Tsdf::new(0.03);
        src.integrate(&d, &c, w, h, k, &ID, &[0.0; 3], 3.0);
        let pose: Pose = (mm(&rz(0.4), &ry(0.5)), [0.37, -0.21, 0.13]);
        let mut g = Tsdf::new(0.03);
        g.merge(&src, &pose, 1.0, None);
        let tr = g.trunc();
        let (mut n, mut bad) = (0, 0);
        for (x, dd, ww) in g.voxels() {
            let xs = inv_mul(&pose, &(ID, x)).1;
            let sdf = 1.0 - xs[2];
            if ww >= 0.99 && sdf.abs() < 0.6 * tr && xs[0].abs() < 0.5 && xs[1].abs() < 0.35 {
                n += 1;
                if (dd as f64 * tr - sdf).abs() > 0.3 * 0.03 {
                    bad += 1;
                }
            }
        }
        assert!(n > 200 && bad == 0, "{n} 칸 중 {bad} 칸 어긋남");
        g.merge(&src, &pose, -1.0, None);
        assert!(g.voxels().is_empty());
    }

    #[test]
    fn rebuild_moves_only_moved() {
        let (d, c, w, h, k) = wall();
        let mut sm = Submaps::new(0.05, Policy { secs: 0.0, dist: 0.0, deg: 0.0 });
        let n1 = (ID, [0.0; 3]);
        let n2 = (ID, [3.0, 0.0, 0.0]);
        let node0 = move |id: u32| Some(if id == 1 { n1 } else { n2 });
        sm.integrate(0.0, 1, &node0, &(ID, [0.0; 3]), &d, &c, w, h, k, 3.0);
        sm.integrate(1.0, 2, &node0, &(ID, [3.0, 0.0, 0.0]), &d, &c, w, h, k, 3.0);
        sm.flush();
        let before = sm.global.voxels().len();
        // 노드 2 만 위로 0.5 m
        let n2b = (ID, [3.0, 0.0, 0.5]);
        let node1 = move |id: u32| Some(if id == 1 { n1 } else { n2b });
        sm.check(&node1); // 활성(노드 2)이 움직임 → 닫힘
        assert_eq!(sm.list.len(), 2);
        assert!(sm.dirty);
        let job = sm.plan(&node1);
        let done = job.run();
        assert_eq!((done.n, done.n_moved), (2, 1));
        sm.install(done);
        assert_eq!(sm.global.voxels().len(), before);
        // 옮긴 벽: (3, 0, 1.5) 근처 칸이 표면(거리 ~0)
        let (dd, ww) = sm.global.at(&[3.0 + 0.025, 0.025, 0.5 + 1.0 + 0.025]).unwrap();
        assert!(ww > 0.0 && dd.abs() < 0.5, "{dd} {ww}");
    }
}
