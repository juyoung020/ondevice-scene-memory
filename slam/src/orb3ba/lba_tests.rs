//! LocalBundleAdjustment 전체(그래프 만들기 · 최적화 · 이상치 지우기 · 되써 넣기)를 원본과 비트 비교.
//! 입력: 저장소 밖 하네스 `~/opt/orb3_lba`(`run.sh 이름 녹화 장수 호출수`)가 만든 `이름_w.bin`(지도 상태 앞 · 뒤) ·
//! `이름_ba.bin`(같은 호출의 g2o 그래프 · 결과). `ORB3LBA=경로/이름[:경로/이름...]`, 없으면 실패.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::lba::{build_graph_ord, local_bundle_adjustment_ord, ObsOrder};
use super::tests::read_calls;
use super::{Est, Kind};
use crate::orb3rs::track::TrackExt;
use crate::orb3rs::types::{Atlas, FrameCtx, KeyFrame, KeyPoint, Map, MapPoint, Se3};

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Rd<'a> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let a: [u8; N] = self.b[self.i..self.i + N].try_into().unwrap();
        self.i += N;
        a
    }
    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take())
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.take())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take())
    }
    fn fv<const N: usize>(&mut self) -> [f32; N] {
        let mut a = [0.0; N];
        for v in a.iter_mut() {
            *v = self.f32();
        }
        a
    }
    fn se3(&mut self) -> Se3 {
        Se3 { q: self.fv::<4>(), t: self.fv::<3>() }
    }
    fn mps(&mut self) -> Vec<Option<u64>> {
        let n = self.u32();
        (0..n).map(|_| { let v = self.i64(); (v >= 0).then_some(v as u64) }).collect()
    }
    fn obs(&mut self) -> BTreeMap<u64, (i32, i32)> {
        self.obs_list().into_iter().collect()
    }
    /// 원본 순회 순서(포인터 순) 그대로
    fn obs_list(&mut self) -> Vec<(u64, (i32, i32))> {
        let n = self.u32();
        (0..n).map(|_| (self.u64(), (self.i32(), self.i32()))).collect()
    }
}

struct Before {
    kf: u64,
    atlas: Atlas,
    ext: TrackExt,
    inertial: bool,
    order: HashMap<u64, Vec<u64>>,
}

struct AfterMp {
    bad: bool,
    pos: [f32; 3],
    normal: [f32; 3],
    min: f32,
    max: f32,
    nobs: i32,
    ref_kf: i64,
    obs: BTreeMap<u64, (i32, i32)>,
}

struct After {
    poses: Vec<(u64, Se3)>,
    kf_mps: Vec<(u64, Vec<Option<u64>>)>,
    erased: Vec<(u64, u64)>,
    mps: Vec<(u64, AfterMp)>,
}

fn read_world(path: &str) -> BTreeMap<i32, (Before, After)> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut r = Rd { b: &b, i: 0 };
    let mut befores: HashMap<i32, Before> = HashMap::new();
    let mut out = BTreeMap::new();
    while r.i < b.len() {
        let magic = r.u32();
        let idx = r.i32();
        if magic == 0x5742414c {
            let kf = r.u64();
            let init = r.u64();
            let inertial = r.u8() != 0;
            let nl = r.i32() as usize;
            let scale: Vec<f32> = (0..nl).map(|_| r.f32()).collect();
            let inv_sigma2: Vec<f32> = (0..nl).map(|_| r.f32()).collect();
            let [fx, fy, cx, cy, bf] = r.fv::<5>();
            let mut ctx = FrameCtx::default();
            ctx.cam.fx = fx;
            ctx.cam.fy = fy;
            ctx.cam.cx = cx;
            ctx.cam.cy = cy;
            ctx.cam.bf = bf;
            ctx.pyr.nlevels = nl as i32;
            ctx.pyr.scale = scale;
            ctx.pyr.inv_sigma2 = inv_sigma2;
            let ctx = Arc::new(ctx);
            let nc = r.u32();
            let conn: Vec<u64> = (0..nc).map(|_| r.u64()).collect();
            let mut atlas = Atlas::default();
            let mut ext = TrackExt::default();
            let mut order = HashMap::new();
            let nk = r.u32();
            let mut map_id = 0;
            for _ in 0..nk {
                let id = r.u64();
                let bad = r.u8() != 0;
                let map = r.u64();
                let tcw = r.se3();
                let has_up = r.u8() != 0;
                let up = r.fv::<3>();
                let n = r.u32();
                let mut keys = Vec::new();
                let mut u_right = Vec::new();
                for _ in 0..n {
                    let (x, y, octave) = (r.f32(), r.f32(), r.i32());
                    keys.push(KeyPoint { x, y, octave, ..Default::default() });
                    u_right.push(r.f32());
                }
                let map_points = r.mps();
                let mut k = KeyFrame { id, map, bad, keys, u_right, map_points, ctx: ctx.clone(), up_cam: has_up.then_some(up), ..Default::default() };
                k.set_pose(tcw);
                if id == kf {
                    map_id = map;
                    k.ordered_conn = conn.iter().map(|&c| (c, 0)).collect();
                }
                atlas.kfs.insert(id, k);
            }
            let nm = r.u32();
            for _ in 0..nm {
                let id = r.u64();
                let bad = r.u8() != 0;
                let map = r.u64();
                let pos = r.fv::<3>();
                let nobs = r.i32();
                let rk = r.i64();
                let ol = r.obs_list();
                order.insert(id, ol.iter().map(|o| o.0).collect());
                let obs = ol.into_iter().collect();
                atlas.mps.insert(id, MapPoint { id, map, bad, pos, obs, n_obs: nobs, ref_kf: rk.max(0) as u64, ..Default::default() });
            }
            atlas.maps.push(Map { id: map_id, init_kf_id: init, ..Default::default() });
            befores.insert(idx, Before { kf, atlas, ext, inertial, order });
        } else if magic == 0x4142414d {
            let n = r.u32();
            let poses = (0..n).map(|_| (r.u64(), r.se3())).collect();
            let n = r.u32();
            let kf_mps = (0..n).map(|_| (r.u64(), r.mps())).collect();
            let n = r.u32();
            let erased = (0..n).map(|_| (r.u64(), r.u64())).collect();
            let n = r.u32();
            let mps = (0..n)
                .map(|_| {
                    let id = r.u64();
                    let bad = r.u8() != 0;
                    let pos = r.fv::<3>();
                    let normal = r.fv::<3>();
                    let (min, max, nobs, ref_kf) = (r.f32(), r.f32(), r.i32(), r.i64());
                    (id, AfterMp { bad, pos, normal, min, max, nobs, ref_kf, obs: r.obs() })
                })
                .collect();
            let b4 = befores.remove(&idx).expect("앞 기록");
            out.insert(idx, (b4, After { poses, kf_mps, erased, mps }));
        } else {
            panic!("모르는 기록 {magic:x}");
        }
    }
    out
}

fn bits3(a: &[f32; 3]) -> [u32; 3] {
    a.map(f32::to_bits)
}

fn check(base: &str) -> usize {
    let world = read_world(&format!("{base}_w.bin"));
    let calls = read_calls(&format!("{base}_ba.bin"));
    assert!(!world.is_empty(), "{base}: 호출 없음");
    let mut bad = 0;
    let mut n = 0;
    for c in &calls {
        let Some((b4, af)) = world.get(&c.idx) else { panic!("{base}: 호출 {} 의 지도 상태 없음", c.idx) };
        n += 1;
        // 1) 그래프
        let ord = ObsOrder(Some(&b4.order));
        let g = build_graph_ord(b4.kf, &b4.atlas, &b4.ext, b4.inertial, ord);
        let p = &g.problem;
        let mut diffs = Vec::new();
        let mut mine: Vec<_> = p.vertices.iter().collect();
        mine.sort_by_key(|v| v.id);
        if mine.len() != c.problem.vertices.len() {
            diffs.push(format!("정점 수 {} != {}", mine.len(), c.problem.vertices.len()));
        } else {
            for (a, o) in mine.iter().zip(&c.problem.vertices) {
                let same = a.id == o.id
                    && a.fixed == o.fixed
                    && match (&a.est, &o.est) {
                        (Est::Pose { q, t }, Est::Pose { q: q2, t: t2 }) => q.iter().chain(t).zip(q2.iter().chain(t2)).all(|(x, y)| x.to_bits() == y.to_bits()),
                        (Est::Point(x), Est::Point(y)) => x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits()),
                        _ => false,
                    };
                if !same {
                    if diffs.is_empty() {
                        eprintln!("정점 차이 예: {:?} / 원본 {:?}", a, o);
                    }
                    diffs.push(format!("정점 {}", a.id));
                }
            }
        }
        if p.edges.len() != c.problem.edges.len() {
            diffs.push(format!("간선 수 {} != {}", p.edges.len(), c.problem.edges.len()));
        } else {
            for (k, (a, o)) in p.edges.iter().zip(&c.problem.edges).enumerate() {
                let ids = |e: &super::Edge, pp: &super::Problem| (pp.vertices[e.v0].id, if matches!(e.kind, Kind::Gravity) { -1 } else { pp.vertices[e.v1].id });
                let d = a.dim();
                let same = a.kind == o.kind
                    && ids(a, p) == ids(o, &c.problem)
                    && a.meas[..d].iter().zip(&o.meas[..d]).all(|(x, y)| x.to_bits() == y.to_bits())
                    && a.info[..d * d].iter().zip(&o.info[..d * d]).all(|(x, y)| x.to_bits() == y.to_bits())
                    && a.delta.map(f64::to_bits) == o.delta.map(f64::to_bits);
                if !same {
                    if diffs.is_empty() {
                        eprintln!("간선 차이 예 {k}: {:?} {:?} / 원본 {:?} {:?}", a, ids(a, p), o, ids(o, &c.problem));
                    }
                    diffs.push(format!("간선 {k}"));
                }
            }
        }
        // 2) 전체
        let mut atlas = b4.atlas.clone();
        let mut ext = b4.ext.clone();
        let st = local_bundle_adjustment_ord(b4.kf, &mut atlas, &mut ext, b4.inertial, &|| false, ord);
        if st.iterations as i32 != c.iterations {
            diffs.push(format!("반복 {} != {}", st.iterations, c.iterations));
        }
        if st.erased != af.erased {
            diffs.push(format!("지운 관측 {} != {}", st.erased.len(), af.erased.len()));
        }
        for (id, s) in &af.poses {
            let m = atlas.kfs[id].tcw;
            if m.q.map(f32::to_bits) != s.q.map(f32::to_bits) || bits3(&m.t) != bits3(&s.t) {
                diffs.push(format!("키프레임 {id} 자세"));
            }
        }
        for (id, v) in &af.kf_mps {
            if &atlas.kfs[id].map_points != v {
                diffs.push(format!("키프레임 {id} 점 칸"));
            }
        }
        for (id, o) in &af.mps {
            let m = &atlas.mps[id];
            let same = m.bad == o.bad
                && bits3(&m.pos) == bits3(&o.pos)
                && (o.bad || (bits3(&m.normal) == bits3(&o.normal) && m.min_dist.to_bits() == o.min.to_bits() && m.max_dist.to_bits() == o.max.to_bits()))
                && m.n_obs == o.nobs
                && (o.ref_kf < 0 || m.ref_kf == o.ref_kf as u64)
                && m.obs == o.obs;
            if !same {
                eprintln!(
                    "점 {id}: 나쁨 {}/{} 위치 {:?}/{:?} 법선 {:?}/{:?} 거리 {} {}/{} {} 관측수 {}/{} 기준 {}/{} 관측 {:?}/{:?}",
                    m.bad, o.bad, m.pos, o.pos, m.normal, o.normal, m.min_dist, m.max_dist, o.min, o.max, m.n_obs, o.nobs, m.ref_kf, o.ref_kf, m.obs, o.obs
                );
                let b = &b4.atlas.mps[id];
                eprintln!("  앞: 기준 {} 관측수 {} 관측 {:?} 순서 {:?} 지움 {:?}", b.ref_kf, b.n_obs, b.obs, b4.order.get(id), af.erased.iter().filter(|e| e.1 == *id).collect::<Vec<_>>());
                diffs.push(format!("점 {id}"));
            }
        }
        if !diffs.is_empty() {
            bad += 1;
            eprintln!("{base} 호출 {}: 다름 {} — {:?}", c.idx, diffs.len(), &diffs[..diffs.len().min(8)]);
        }
    }
    eprintln!("{base}: LBA {n} 호출 중 {} 비트 동일", n - bad);
    bad
}

#[test]
fn lba_full_eq_original() {
    let v = std::env::var("ORB3LBA").expect("ORB3LBA 없음: 비교 입력이 있어야 한다");
    let bad: usize = v.split(':').map(check).sum();
    assert_eq!(bad, 0);
}
