//! 합치기용 지역 BA 전체를 원본(orb3_refid: 집합 순서 = 번호 순)과 비트 비교.
//! 입력: `~/opt/orb3_gba` 가 만든 `이름_mw.bin`(지도 상태 앞 · 뒤) · `이름_mlba.bin`(g2o 그래프 · 결과, 호출마다 두 기록: 5 번 · 10 번).
//! `ORB3MLBA=경로/이름[:...]`, 없으면 실패.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::lba::ObsOrder;
use super::mlba::{build_graph, merge_local_ba, KfMpOrder, MergeMarks};
use super::tests::read_calls;
use super::Est;
use crate::orb3rs::track::TrackExt;
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
    fn ids(&mut self) -> Vec<u64> {
        let n = self.u32();
        (0..n).map(|_| self.u64()).collect()
    }
    fn mps(&mut self) -> Vec<Option<u64>> {
        let n = self.u32();
        (0..n).map(|_| { let v = self.i64(); (v >= 0).then_some(v as u64) }).collect()
    }
    fn obs(&mut self) -> BTreeMap<u64, (i32, i32)> {
        let n = self.u32();
        (0..n).map(|_| (self.u64(), (self.i32(), self.i32()))).collect()
    }
}

struct Before {
    main: u64,
    adjust: Vec<u64>,
    fixed: Vec<u64>,
    atlas: Atlas,
    ext: TrackExt,
    marks: MergeMarks,
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
    kfs: Vec<(u64, Se3, Vec<Option<u64>>)>,
    mps: Vec<(u64, AfterMp)>,
}

fn read_world(path: &str) -> BTreeMap<i32, (Before, Option<After>)> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut r = Rd { b: &b, i: 0 };
    let mut out: BTreeMap<i32, (Before, Option<After>)> = BTreeMap::new();
    while r.i < b.len() {
        let magic = r.u32();
        let idx = r.i32();
        if magic == 0x574d4c42 {
            let main = r.u64();
            let adjust = r.ids();
            let fixed = r.ids();
            let nl = r.i32() as usize;
            let mut ctx = FrameCtx::default();
            ctx.pyr.scale = (0..nl).map(|_| r.f32()).collect();
            ctx.pyr.inv_sigma2 = (0..nl).map(|_| r.f32()).collect();
            ctx.pyr.nlevels = nl as i32;
            let [fx, fy, cx, cy, bf] = r.fv::<5>();
            (ctx.cam.fx, ctx.cam.fy, ctx.cam.cx, ctx.cam.cy, ctx.cam.bf) = (fx, fy, cx, cy, bf);
            let ctx = Arc::new(ctx);
            let mut atlas = Atlas::default();
            let mut ext = TrackExt::default();
            let mut marks = MergeMarks::default();
            let nk = r.u32();
            for _ in 0..nk {
                let id = r.u64();
                let bad = r.u8() != 0;
                let map = r.u64();
                let tcw = r.se3();
                let mark = r.u64();
                let n = r.u32();
                let mut keys = Vec::new();
                let mut u_right = Vec::new();
                for _ in 0..n {
                    let (x, y, octave) = (r.f32(), r.f32(), r.i32());
                    keys.push(KeyPoint { x, y, octave, ..Default::default() });
                    u_right.push(r.f32());
                }
                let map_points = r.mps();
                let mut k = KeyFrame { id, map, bad, keys, u_right, map_points, ctx: ctx.clone(), ..Default::default() };
                k.set_pose(tcw);
                atlas.kfs.insert(id, k);
                marks.kf.insert(id, mark);
            }
            let nm = r.u32();
            for _ in 0..nm {
                let id = r.u64();
                let bad = r.u8() != 0;
                let map = r.u64();
                let pos = r.fv::<3>();
                let nobs = r.i32();
                let mark = r.u64();
                let rk = r.i64();
                let obs = r.obs();
                ext.mp_nobs.insert(id, nobs);
                marks.mp.insert(id, mark);
                atlas.mps.insert(id, MapPoint { id, map, bad, pos, obs, ref_kf: rk.max(0) as u64, ..Default::default() });
            }
            let map_id = atlas.kfs[&main].map;
            atlas.maps.push(Map { id: map_id, ..Default::default() });
            out.insert(idx, (Before { main, adjust, fixed, atlas, ext, marks }, None));
        } else if magic == 0x414d4c42 {
            let n = r.u32();
            let kfs = (0..n).map(|_| (r.u64(), r.se3(), r.mps())).collect();
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
            out.get_mut(&idx).expect("앞 기록").1 = Some(After { kfs, mps });
        } else {
            panic!("모르는 기록 {magic:x}");
        }
    }
    out
}

fn b3(a: &[f32; 3]) -> [u32; 3] {
    a.map(f32::to_bits)
}

fn check(base: &str) -> usize {
    let world = read_world(&format!("{base}_mw.bin"));
    let calls = read_calls(&format!("{base}_mlba.bin"));
    assert!(!world.is_empty(), "{base}: 호출 없음");
    assert_eq!(calls.len(), world.len() * 2, "{base}: 그래프 기록은 호출마다 두 개(5 번 · 10 번)");
    let mut bad = 0;
    for (n, (idx, (b4, af))) in world.iter().enumerate() {
        let af = af.as_ref().expect("뒤 기록(멈춤으로 일찍 끝난 호출은 아직 다루지 않음)");
        let mut diffs = Vec::new();
        // 1) 첫 그래프(5 번)
        let mut marks = b4.marks.clone();
        let g = build_graph(b4.main, &b4.adjust, &b4.fixed, &b4.atlas, &mut marks, ObsOrder::default(), KfMpOrder::default());
        let c1 = &calls[2 * n];
        let mut mine: Vec<_> = g.problem.vertices.iter().collect();
        mine.sort_by_key(|v| v.id);
        let vsame = mine.len() == c1.problem.vertices.len()
            && mine.iter().zip(&c1.problem.vertices).all(|(a, o)| {
                a.id == o.id
                    && a.fixed == o.fixed
                    && match (&a.est, &o.est) {
                        (Est::Pose { q, t }, Est::Pose { q: q2, t: t2 }) => q.iter().chain(t).zip(q2.iter().chain(t2)).all(|(x, y)| x.to_bits() == y.to_bits()),
                        (Est::Point(x), Est::Point(y)) => x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits()),
                        _ => false,
                    }
            });
        if !vsame {
            diffs.push(format!("정점 {} / {}", mine.len(), c1.problem.vertices.len()));
        }
        let ids = |e: &super::Edge, pp: &super::Problem| (pp.vertices[e.v0].id, pp.vertices[e.v1].id);
        let esame = g.problem.edges.len() == c1.problem.edges.len()
            && g.problem.edges.iter().zip(&c1.problem.edges).all(|(a, o)| {
                let d = a.dim();
                a.kind == o.kind
                    && ids(a, &g.problem) == ids(o, &c1.problem)
                    && a.meas[..d].iter().zip(&o.meas[..d]).all(|(x, y)| x.to_bits() == y.to_bits())
                    && a.info[..d * d].iter().zip(&o.info[..d * d]).all(|(x, y)| x.to_bits() == y.to_bits())
                    && a.delta.map(f64::to_bits) == o.delta.map(f64::to_bits)
            });
        if !esame {
            diffs.push(format!("간선 {} / {}", g.problem.edges.len(), c1.problem.edges.len()));
        }
        // 2) 전체
        let mut atlas = b4.atlas.clone();
        let mut ext = b4.ext.clone();
        let mut marks = b4.marks.clone();
        let st = merge_local_ba(b4.main, &b4.adjust, &b4.fixed, &mut atlas, &mut ext, &mut marks, &|| false, ObsOrder::default(), KfMpOrder::default());
        let c2 = &calls[2 * n + 1];
        if st.iterations != [c1.iterations as usize, c2.iterations as usize] {
            diffs.push(format!("반복 {:?} / {} {}", st.iterations, c1.iterations, c2.iterations));
        }
        for (id, pose, slots) in &af.kfs {
            let k = &atlas.kfs[id];
            if k.tcw.q.map(f32::to_bits) != pose.q.map(f32::to_bits) || b3(&k.tcw.t) != b3(&pose.t) {
                diffs.push(format!("키프레임 {id} 자세"));
            }
            if &k.map_points != slots {
                diffs.push(format!("키프레임 {id} 점 칸"));
            }
        }
        for (id, o) in &af.mps {
            let m = &atlas.mps[id];
            let same = m.bad == o.bad
                && b3(&m.pos) == b3(&o.pos)
                && (o.bad || (b3(&m.normal) == b3(&o.normal) && m.min_dist.to_bits() == o.min.to_bits() && m.max_dist.to_bits() == o.max.to_bits()))
                && ext.nobs(*id) == o.nobs
                && (o.ref_kf < 0 || m.ref_kf == o.ref_kf as u64)
                && m.obs == o.obs;
            if !same {
                diffs.push(format!("점 {id}"));
            }
        }
        eprintln!(
            "{base} 호출 {idx}: 조정 {} 고정 {} 정점 {} 간선 {} 반복 {:?} 지운 관측 {} — 다름 {}",
            b4.adjust.len(),
            b4.fixed.len(),
            g.problem.vertices.len(),
            g.problem.edges.len(),
            st.iterations,
            st.erased.len(),
            diffs.len()
        );
        if !diffs.is_empty() {
            bad += 1;
            eprintln!("  {:?}", &diffs[..diffs.len().min(8)]);
        }
    }
    let _ = HashMap::<u8, u8>::new();
    bad
}

#[test]
fn mlba_full_eq_original() {
    let v = std::env::var("ORB3MLBA").expect("ORB3MLBA 없음: 비교 입력이 있어야 한다");
    let bad: usize = v.split(':').map(check).sum();
    assert_eq!(bad, 0);
}
