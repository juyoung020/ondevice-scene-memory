//! CorrectLoop(`orb3loop::correct`) 원본 대조. 원본은 번호 순 비교자 기준 빌드(`~/opt/orb3_refid`)를 돌린 덤프
//! (`~/opt/orb3_loop_bg/reloc/correct_dump`): 루프 닫기마다 지도 스레드 멈춘 뒤 지도 통째 · 입력, OptimizeEssentialGraph
//! 들어갈 때(입력 세 표 · 지도) · 나올 때 지도, 끝 지도. OptimizeEssentialGraph 는 원본 결과(자세 · 점 위치)를 넣고,
//! 들어갈 때 Rust 상태가 원본과 같은지 · 끝 상태가 같은지 본다.
//! ORBLOOP_CORRECT=덤프1:덤프2 cargo test --release -p slam --test orb3loop_correct
use slam::orb3loop::correct::*;
use slam::orb3loop::detect::G2oSim3;
use slam::orb3rs::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;

struct Rd {
    f: std::io::BufReader<std::fs::File>,
}
impl Rd {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0u8; N];
        self.f.read_exact(&mut a).unwrap();
        a
    }
    fn more(&mut self) -> bool {
        use std::io::BufRead;
        !self.f.fill_buf().unwrap().is_empty()
    }
    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take())
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
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take())
    }
    fn fv(&mut self) -> Vec<f32> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.f32()).collect()
    }
    fn keys(&mut self) -> Vec<KeyPoint> {
        let n = self.u32() as usize;
        (0..n)
            .map(|_| KeyPoint { x: self.f32(), y: self.f32(), size: self.f32(), angle: self.f32(), response: self.f32(), octave: self.i32(), class_id: self.i32() })
            .collect()
    }
    fn feat(&mut self) -> BTreeMap<u32, Vec<u32>> {
        let n = self.u32();
        let mut m = BTreeMap::new();
        for _ in 0..n {
            let k = self.u32();
            let c = self.u32();
            m.insert(k, (0..c).map(|_| self.u32()).collect());
        }
        m
    }
    fn se3(&mut self) -> Se3 {
        Se3 { q: [self.f32(), self.f32(), self.f32(), self.f32()], t: [self.f32(), self.f32(), self.f32()] }
    }
    fn sim(&mut self) -> G2oSim3 {
        G2oSim3 { r: [self.f64(), self.f64(), self.f64(), self.f64()], t: [self.f64(), self.f64(), self.f64()], s: self.f64() }
    }
    fn opt(&mut self) -> Option<u64> {
        let v = self.i64();
        (v >= 0).then_some(v as u64)
    }
    fn mps(&mut self) -> Vec<Option<MpId>> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.opt()).collect()
    }
    fn ids(&mut self) -> Vec<KfId> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.u64()).collect()
    }
}

struct DMap {
    kfs: BTreeMap<KfId, KeyFrame>,
    mps: BTreeMap<MpId, MapPoint>,
    not_erase: BTreeMap<KfId, bool>,
    corrected_by: BTreeMap<MpId, (KfId, KfId)>,
}

fn read_map(r: &mut Rd) -> DMap {
    let map_id = r.u64();
    let (fx, fy, cx, cy, invfx, invfy, bf, bb, thd) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let (min_x, max_x, min_y, max_y, giw, gih) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let nlevels = r.i32();
    let scale_factor = r.f32();
    let log_scale_factor = r.f32();
    let (scale, inv_scale, sigma2, inv_sigma2) = (r.fv(), r.fv(), r.fv(), r.fv());
    let cam = Camera { fx, fy, cx, cy, invfx, invfy, bf, b: bb, th_depth: thd, width: 0, height: 0 };
    let pyr = Pyramid { nfeatures: 0, scale_factor, nlevels, ini_th_fast: 0, min_th_fast: 0, scale, inv_scale, sigma2, inv_sigma2, log_scale_factor };
    let ctx = Arc::new(FrameCtx { cam, pyr, min_x, max_x, min_y, max_y, grid_inv_w: giw, grid_inv_h: gih });
    let _ = map_id;
    let nk = r.u32();
    let mut kfs = BTreeMap::new();
    let mut not_erase = BTreeMap::new();
    for _ in 0..nk {
        let id = r.u64();
        let frame_id = r.u64();
        let bad = r.u8() != 0;
        not_erase.insert(id, r.u8() != 0);
        let map = r.opt().unwrap_or(u64::MAX);
        let tcw = r.se3();
        let twc = r.se3();
        let keys = r.keys();
        let desc = (0..keys.len()).map(|_| r.take::<32>()).collect();
        let (u_right, depth) = (r.fv(), r.fv());
        let map_points = r.mps();
        let feat = r.feat();
        let mut grid = vec![Vec::new(); GRID_COLS * GRID_ROWS];
        for c in 0..GRID_COLS {
            for w in 0..GRID_ROWS {
                let k = r.u32();
                grid[c * GRID_ROWS + w] = (0..k).map(|_| r.u32()).collect();
            }
        }
        let nc = r.u32();
        let conn_weights = (0..nc).map(|_| (r.u64(), r.i32())).collect();
        let no = r.u32();
        let ordered_conn = (0..no).map(|_| (r.u64(), r.i32())).collect();
        let parent = r.opt();
        let children = r.ids();
        let loop_edges = r.ids();
        let merge_edges = r.ids();
        kfs.insert(
            id,
            KeyFrame { id, frame_id, map, bad, tcw, twc, keys, desc, u_right, depth, map_points, feat, grid, conn_weights, ordered_conn, parent, children, loop_edges, merge_edges, ctx: ctx.clone(), ..Default::default() },
        );
    }
    let nm = r.u32();
    let mut mps = BTreeMap::new();
    let mut corrected_by = BTreeMap::new();
    for _ in 0..nm {
        let id = r.u64();
        let bad = r.u8() != 0;
        let map = r.opt().unwrap_or(u64::MAX);
        let pos = [r.f32(), r.f32(), r.f32()];
        let normal = [r.f32(), r.f32(), r.f32()];
        let desc = r.take::<32>();
        let (min_dist, max_dist) = (r.f32(), r.f32());
        let track_scale_level = r.i32();
        let ref_kf = r.opt().unwrap_or(u64::MAX);
        let replaced = r.opt();
        let n_obs = r.i32();
        let visible = r.i32();
        let found = r.i32();
        let cb = (r.u64(), r.u64());
        if cb.0 != 0 || cb.1 != 0 {
            corrected_by.insert(id, cb);
        }
        let no = r.u32();
        let obs = (0..no).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
        mps.insert(id, MapPoint { id, map, bad, pos, normal, desc, min_dist, max_dist, track_scale_level, ref_kf, replaced, n_obs, visible, found, obs, ..Default::default() });
    }
    DMap { kfs, mps, not_erase, corrected_by }
}

fn b3(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn bse3(a: &Se3, b: &Se3) -> bool {
    a.q.iter().zip(&b.q).all(|(x, y)| x.to_bits() == y.to_bits()) && b3(&a.t, &b.t)
}
fn sim_bits(a: &G2oSim3, b: &G2oSim3) -> bool {
    a.r.iter().chain(a.t.iter()).chain(std::iter::once(&a.s)).zip(b.r.iter().chain(b.t.iter()).chain(std::iter::once(&b.s))).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// 원본 지도(d)와 Rust 상태 비교: 다른 칸 이름별 수
fn diff(kfs: &BTreeMap<KfId, KeyFrame>, mps: &BTreeMap<MpId, MapPoint>, d: &DMap) -> BTreeMap<&'static str, usize> {
    let mut c: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut add = |k: &'static str, bad: bool| {
        if bad {
            *c.entry(k).or_insert(0) += 1;
        }
    };
    for (id, o) in &d.kfs {
        let Some(k) = kfs.get(id) else {
            add("키프레임 없음", true);
            continue;
        };
        add("KF 자세", !bse3(&k.tcw, &o.tcw) || !bse3(&k.twc, &o.twc));
        add("KF 나쁨", k.bad != o.bad);
        add("KF 지도 점", k.map_points != o.map_points);
        add("KF 연결 무게", k.conn_weights != o.conn_weights);
        add("KF 연결 순서", k.ordered_conn != o.ordered_conn);
        add("KF 루프 가장자리", k.loop_edges != o.loop_edges);
    }
    for (id, o) in &d.mps {
        let Some(p) = mps.get(id) else {
            add("점 없음", true);
            continue;
        };
        add("점 위치", !b3(&p.pos, &o.pos));
        add("점 법선", !b3(&p.normal, &o.normal));
        add("점 거리", p.min_dist.to_bits() != o.min_dist.to_bits() || p.max_dist.to_bits() != o.max_dist.to_bits());
        add("점 기술자", p.desc != o.desc);
        add("점 관측", p.obs != o.obs);
        add("점 나쁨 · 바뀜", p.bad != o.bad || p.replaced != o.replaced);
        add("점 nObs · 보임 · 찾음", p.n_obs != o.n_obs || p.visible != o.visible || p.found != o.found);
    }
    c
}

struct Replay {
    before: Option<(KfId, KfId, bool, BTreeMap<KfId, G2oSim3>, BTreeMap<KfId, G2oSim3>, BTreeMap<KfId, BTreeSet<KfId>>, DMap)>,
    after: Option<DMap>,
    why: Vec<String>,
}

impl EssentialGraph for Replay {
    fn optimize(
        &mut self,
        kfs: &mut BTreeMap<KfId, KeyFrame>,
        mps: &mut BTreeMap<MpId, MapPoint>,
        _cb: &BTreeMap<MpId, (KfId, KfId)>,
        loop_kf: KfId,
        cur: KfId,
        non: &BTreeMap<KfId, G2oSim3>,
        cor: &BTreeMap<KfId, G2oSim3>,
        lc: &BTreeMap<KfId, BTreeSet<KfId>>,
        fix: bool,
    ) {
        let (ol, oc, of, onon, ocor, olc, dm) = self.before.take().expect("원본 ESSB 없음");
        if (ol, oc, of) != (loop_kf, cur, fix) {
            self.why.push("Essential 입력 KF".into());
        }
        let same_map = |a: &BTreeMap<KfId, G2oSim3>, b: &BTreeMap<KfId, G2oSim3>| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0 && sim_bits(x.1, y.1));
        if !same_map(non, &onon) {
            self.why.push("NonCorrectedSim3".into());
        }
        if !same_map(cor, &ocor) {
            self.why.push("CorrectedSim3".into());
        }
        if lc != &olc {
            self.why.push("LoopConnections".into());
        }
        let d = diff(kfs, mps, &dm);
        if !d.is_empty() {
            self.why.push(format!("Essential 앞 지도 {d:?}"));
        }
        let a = self.after.take().expect("원본 ESSA 없음");
        for (id, o) in &a.kfs {
            if let Some(k) = kfs.get_mut(id) {
                k.tcw = o.tcw;
                k.twc = o.twc;
            }
        }
        for (id, o) in &a.mps {
            if let Some(p) = mps.get_mut(id) {
                p.pos = o.pos;
                p.normal = o.normal;
                p.min_dist = o.min_dist;
                p.max_dist = o.max_dist;
            }
        }
    }
}

#[test]
fn correct_loop_matches_original() {
    let dumps = std::env::var("ORBLOOP_CORRECT").expect("입력 없음: ORBLOOP_CORRECT(원본 correct_dump 덤프, : 로 여럿)");
    let (mut n, mut same) = (0, 0);
    let mut bad = Vec::new();
    for path in dumps.split(':') {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
        assert!(f.metadata().unwrap().len() > 0, "빈 덤프 {path}");
        let mut r = Rd { f: std::io::BufReader::with_capacity(1 << 20, f) };
        while r.more() {
            assert_eq!(r.u32(), 0x504f4f4c, "LOOP 자리 아님");
            let cur = r.u64();
            let matched = r.u64();
            let fix = r.u8() != 0;
            let scw = r.sim();
            let loop_mps = r.mps();
            let loop_matched = r.mps();
            let _nmaps = r.u32();
            let m0 = read_map(&mut r);
            assert_eq!(r.u32(), 0x42535345, "ESSB 자리 아님");
            let (el, ec, ef) = (r.u64(), r.u64(), r.u8() != 0);
            let rd_map = |r: &mut Rd| -> BTreeMap<KfId, G2oSim3> {
                let n = r.u32();
                (0..n).map(|_| (r.u64(), r.sim())).collect()
            };
            let non = rd_map(&mut r);
            let cor = rd_map(&mut r);
            let nl = r.u32();
            let lc: BTreeMap<KfId, BTreeSet<KfId>> = (0..nl).map(|_| (r.u64(), r.ids().into_iter().collect())).collect();
            let mb = read_map(&mut r);
            assert_eq!(r.u32(), 0x41535345, "ESSA 자리 아님");
            let ma = read_map(&mut r);
            assert_eq!(r.u32(), 0x454f4f4c, "LOOE 자리 아님");
            let gba = r.u8() != 0;
            let me = read_map(&mut r);

            let DMap { mut kfs, mut mps, mut corrected_by, .. } = m0;
            let mut rep = Replay { before: Some((el, ec, ef, non, cor, lc, mb)), after: Some(ma), why: Vec::new() };
            let inp = LoopIn { cur, matched, scw, loop_mps: &loop_mps, loop_matched: &loop_matched, fix_scale: fix, imu_init: false };
            let out = correct_loop(&inp, &mut kfs, &mut mps, &mut corrected_by, &mut rep);
            let mut why = rep.why;
            if rep.before.is_some() {
                why.push("Essential 안 불림".into());
            }
            if out.run_gba != gba {
                why.push("GBA".into());
            }
            let d = diff(&kfs, &mps, &me);
            if !d.is_empty() {
                why.push(format!("끝 지도 {d:?}"));
            }
            for k in out.not_erase {
                if !me.not_erase[&k] {
                    why.push("mbNotErase".into());
                }
            }
            n += 1;
            if why.is_empty() {
                same += 1;
            } else {
                bad.push(format!("{path} 루프 {n}(KF {cur}↔{matched}, 지도 KF {} 개): {}", kfs.len(), why.join(" / ")));
            }
        }
    }
    eprintln!("루프 닫기 {n} 번 중 같음 {same}");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(n > 0, "루프 닫기가 없는 입력");
    assert_eq!(same, n);
}
