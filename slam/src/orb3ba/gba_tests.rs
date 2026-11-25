//! 전역 BA(BundleAdjustment) 전체를 원본과 비트 비교.
//! 입력: 저장소 밖 하네스 `~/opt/orb3_gba` 가 만든 `이름_w.bin`(지도 상태 앞 · 뒤) · `이름.bin`(같은 호출의 g2o 그래프 · 결과).
//! `ORB3GBA=경로/이름[:...]`, 없으면 실패.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::gba::{build_graph, bundle_adjustment_ord};
use super::lba::ObsOrder;
use super::tests::read_calls;
use super::{Est, Kind};
use crate::orb3rs::types::{Atlas, FrameCtx, KeyFrame, KeyPoint, Map, MapPoint, Se3};

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}
impl Rd<'_> {
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
        std::array::from_fn(|_| self.f32())
    }
    fn se3(&mut self) -> Se3 {
        Se3 { q: self.fv::<4>(), t: self.fv::<3>() }
    }
}

struct Before {
    loop_kf: u64,
    init: u64,
    origin: u64,
    robust: bool,
    n_iter: usize,
    atlas: Atlas,
    kfs: Vec<u64>,
    mps: Vec<u64>,
    order: HashMap<u64, Vec<u64>>,
    /// 그래프가 실제로 읽은 점(나머지는 그 순간 나쁨이었음)
    used: std::collections::HashSet<u64>,
}
struct AfterKf {
    pose: Se3,
    gba: Se3,
    for_kf: u64,
}
struct AfterMp {
    pos: [f32; 3],
    gba: [f32; 3],
    for_kf: u64,
    normal: [f32; 3],
    min: f32,
    max: f32,
}

fn read_world(path: &str) -> BTreeMap<i32, (Before, Vec<(u64, AfterKf)>, Vec<(u64, AfterMp)>)> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut r = Rd { b: &b, i: 0 };
    let mut befores: HashMap<i32, Before> = HashMap::new();
    let mut out = BTreeMap::new();
    while r.i < b.len() {
        let magic = r.u32();
        let idx = r.i32();
        if magic == 0x57424247 {
            let (loop_kf, init, origin, map_id) = (r.u64(), r.u64(), r.u64(), r.u64());
            let robust = r.u8() != 0;
            let n_iter = r.i32() as usize;
            let nl = r.i32() as usize;
            let mut ctx = FrameCtx::default();
            ctx.pyr.scale = (0..nl).map(|_| r.f32()).collect();
            ctx.pyr.inv_sigma2 = (0..nl).map(|_| r.f32()).collect();
            ctx.pyr.nlevels = nl as i32;
            let [fx, fy, cx, cy, bf] = r.fv::<5>();
            (ctx.cam.fx, ctx.cam.fy, ctx.cam.cx, ctx.cam.cy, ctx.cam.bf) = (fx, fy, cx, cy, bf);
            let ctx = Arc::new(ctx);
            let mut atlas = Atlas::default();
            let nk = r.u32();
            let mut kfs = Vec::new();
            for _ in 0..nk {
                let id = r.u64();
                let bad = r.u8() != 0;
                let tcw = r.se3();
                let n = r.u32();
                let mut keys = Vec::new();
                let mut u_right = Vec::new();
                for _ in 0..n {
                    let (x, y, octave) = (r.f32(), r.f32(), r.i32());
                    keys.push(KeyPoint { x, y, octave, ..Default::default() });
                    u_right.push(r.f32());
                }
                let mut k = KeyFrame { id, map: map_id, bad, keys, u_right, ctx: ctx.clone(), ..Default::default() };
                k.set_pose(tcw);
                atlas.kfs.insert(id, k);
                kfs.push(id);
            }
            let nm = r.u32();
            let mut mps = Vec::new();
            let mut order = HashMap::new();
            for _ in 0..nm {
                let id = r.u64();
                let bad = r.u8() != 0;
                let pos = r.fv::<3>();
                let rk = r.i64();
                let n = r.u32();
                let ol: Vec<(u64, (i32, i32))> = (0..n).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
                order.insert(id, ol.iter().map(|o| o.0).collect());
                atlas.mps.insert(id, MapPoint { id, map: map_id, bad, pos, obs: ol.into_iter().collect(), ref_kf: rk.max(0) as u64, ..Default::default() });
                mps.push(id);
            }
            atlas.maps.push(Map { id: map_id, kfs: kfs.clone(), mps: mps.clone(), origin_kf: Some(origin), ..Default::default() });
            befores.insert(idx, Before { loop_kf, init, origin, robust, n_iter, atlas, kfs, mps, order, used: Default::default() });
        } else if magic == 0x55424247 {
            // 그래프가 실제로 읽은 점 위치 · 관측으로 앞 기록을 덮음
            let id = r.u64();
            let pos: [f32; 3] = std::array::from_fn(|_| f64::from_le_bytes(r.take()) as f32);
            let n = r.u32();
            let ol: Vec<(u64, (i32, i32))> = (0..n).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
            let b4 = befores.get_mut(&idx).expect("앞 기록");
            b4.order.insert(id, ol.iter().map(|o| o.0).collect());
            b4.used.insert(id);
            let m = b4.atlas.mps.get_mut(&id).unwrap();
            m.pos = pos;
            m.obs = ol.into_iter().collect();
        } else if magic == 0x41424248 {
            let n = r.u32();
            let kfs = (0..n).map(|_| (r.u64(), AfterKf { pose: r.se3(), gba: r.se3(), for_kf: r.u64() })).collect();
            let n = r.u32();
            let mps = (0..n)
                .map(|_| {
                    let id = r.u64();
                    (id, AfterMp { pos: r.fv::<3>(), gba: r.fv::<3>(), for_kf: r.u64(), normal: r.fv::<3>(), min: r.f32(), max: r.f32() })
                })
                .collect();
            let mut b4 = befores.remove(&idx).expect("앞 기록");
            // 원본이 그래프를 만들 때 나쁨이던 점(지도 스레드가 앞 기록 뒤에 지움): 읽은 기록이 없음
            for (id, m) in b4.atlas.mps.iter_mut() {
                if !b4.used.contains(id) {
                    m.bad = true;
                }
            }
            out.insert(idx, (b4, kfs, mps));
        } else {
            panic!("모르는 기록 {magic:x}");
        }
    }
    out
}

fn b3(a: &[f32; 3]) -> [u32; 3] {
    a.map(f32::to_bits)
}
fn bse3(s: &Se3) -> ([u32; 4], [u32; 3]) {
    (s.q.map(f32::to_bits), b3(&s.t))
}

fn check(base: &str) -> usize {
    let world = read_world(&format!("{base}_w.bin"));
    let calls = read_calls(&format!("{base}.bin"));
    assert!(!world.is_empty(), "{base}: 호출 없음");
    let mut bad = 0;
    for c in &calls {
        let Some((b4, akf, amp)) = world.get(&c.idx) else { panic!("{base}: 호출 {} 의 지도 상태 없음", c.idx) };
        let ord = ObsOrder(Some(&b4.order));
        let mut diffs = Vec::new();
        // 1) 그래프
        let g = build_graph(&b4.atlas, Some(b4.init), &b4.kfs, &b4.mps, b4.robust, ord);
        let p = &g.problem;
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
                    diffs.push(format!("정점 {}", a.id));
                }
            }
        }
        if p.edges.len() != c.problem.edges.len() {
            let ids = |e: &super::Edge, pp: &super::Problem| (pp.vertices[e.v0].id, pp.vertices[e.v1].id);
            if let Some(k) = (0..p.edges.len().min(c.problem.edges.len())).find(|&k| ids(&p.edges[k], p) != ids(&c.problem.edges[k], &c.problem)) {
                eprintln!("  처음 다른 간선 {k}: 우리 {:?} 원본 {:?}", ids(&p.edges[k], p), ids(&c.problem.edges[k], &c.problem));
                let (pv, kv) = ids(&c.problem.edges[k], &c.problem);
                let mut mk = 0u64;
                for k in &b4.kfs { mk = mk.max(*k); }
                let mp = (pv as u64) - mk - 1;
                let kf = kv as u64;
                eprintln!("  점 {mp}: 관측 {:?}, 키프레임 {kf} 있음 {} 나쁨 {:?}", b4.atlas.mps.get(&mp).map(|m| &m.obs), b4.atlas.kfs.contains_key(&kf), b4.atlas.kfs.get(&kf).map(|k| k.bad));
            }
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
                    diffs.push(format!("간선 {k}"));
                }
            }
        }
        // 2) 전체
        let mut atlas = b4.atlas.clone();
        let out = bundle_adjustment_ord(&mut atlas, Some(b4.init), Some(b4.origin), &b4.kfs, &b4.mps, b4.n_iter, &|| false, b4.loop_kf, b4.robust, ord);
        if out.iterations as i32 != c.iterations {
            diffs.push(format!("반복 {} != {}", out.iterations, c.iterations));
        }
        let kg: HashMap<u64, Se3> = out.kf_gba.iter().copied().collect();
        let mg: HashMap<u64, [f32; 3]> = out.mp_gba.iter().copied().collect();
        for (id, o) in akf {
            if bse3(&atlas.kfs[id].tcw) != bse3(&o.pose) {
                diffs.push(format!("키프레임 {id} 자세"));
            }
            let mine = kg.get(id);
            if (o.for_kf == b4.loop_kf && !out.direct) != mine.is_some() || mine.is_some_and(|s| bse3(s) != bse3(&o.gba)) {
                diffs.push(format!("키프레임 {id} TcwGBA"));
            }
        }
        for (id, o) in amp {
            let m = &atlas.mps[id];
            if b3(&m.pos) != b3(&o.pos) || (out.direct && (b3(&m.normal) != b3(&o.normal) || m.min_dist.to_bits() != o.min.to_bits() || m.max_dist.to_bits() != o.max.to_bits())) {
                diffs.push(format!("점 {id}"));
            }
            let mine = mg.get(id);
            if (o.for_kf == b4.loop_kf && !out.direct) != mine.is_some() || mine.is_some_and(|v| b3(v) != b3(&o.gba)) {
                diffs.push(format!("점 {id} PosGBA"));
            }
        }
        eprintln!("{base} 호출 {}: 키프레임 {} 점 {} 간선 {} 반복 {} 바로쓰기 {} — 다름 {}", c.idx, akf.len(), amp.len(), p.edges.len(), out.iterations, out.direct, diffs.len());
        if !diffs.is_empty() {
            bad += 1;
            eprintln!("  {:?}", &diffs[..diffs.len().min(8)]);
        }
    }
    bad
}

#[test]
fn gba_full_eq_original() {
    let v = std::env::var("ORB3GBA").expect("ORB3GBA 없음: 비교 입력이 있어야 한다");
    let bad: usize = v.split(':').map(check).sum();
    assert_eq!(bad, 0);
}
