//! LoopClosing 공통 영역 찾기(`orb3loop::detect`) 원본 대조. 원본 기준 빌드 실행 덤프(`~/opt/orb3_loop_bg/reloc/loop_dump`):
//! DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF 호출마다 들어갈 때 상태 · 쓴 난수 · OptimizeSim3 입출력 · 나올 때 출력.
//! OptimizeSim3 는 원본 FFI 도우미(`mlp_srv` 명령 4, 기준 빌드의 Optimizer::OptimizeSim3 를 가짜 키프레임으로 부름).
//! ORBLOOP_LOOPDUMP=덤프1:덤프2 ORBLOOP_MLPSRV=.../mlp_srv cargo test --release -p slam --test orb3loop_detect
use slam::orb3loop::detect::*;
use slam::orb3rs::*;
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
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
    fn mp(&mut self) -> Option<MpId> {
        let v = self.i64();
        (v >= 0).then_some(v as u64)
    }
    fn kf(&mut self) -> Option<KfId> {
        self.mp()
    }
    fn mps(&mut self) -> Vec<Option<MpId>> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.mp()).collect()
    }
}

fn sim_bits(a: &G2oSim3, b: &G2oSim3) -> bool {
    a.r.iter().chain(a.t.iter()).chain(std::iter::once(&a.s)).zip(b.r.iter().chain(b.t.iter()).chain(std::iter::once(&b.s))).all(|(x, y)| x.to_bits() == y.to_bits())
}

struct World2 {
    kfs: BTreeMap<KfId, KeyFrame>,
    mps: BTreeMap<MpId, MapPoint>,
}

fn read_world(r: &mut Rd) -> World2 {
    let (fx, fy, cx, cy, invfx, invfy, bf, bb, thd) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let (min_x, max_x, min_y, max_y, giw, gih) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let nlevels = r.i32();
    let scale_factor = r.f32();
    let log_scale_factor = r.f32();
    let (scale, inv_scale, sigma2, inv_sigma2) = (r.fv(), r.fv(), r.fv(), r.fv());
    let cam = Camera { fx, fy, cx, cy, invfx, invfy, bf, b: bb, th_depth: thd, width: 0, height: 0 };
    let pyr = Pyramid { nfeatures: 0, scale_factor, nlevels, ini_th_fast: 0, min_th_fast: 0, scale, inv_scale, sigma2, inv_sigma2, log_scale_factor };
    let ctx = Arc::new(FrameCtx { cam, pyr, min_x, max_x, min_y, max_y, grid_inv_w: giw, grid_inv_h: gih });
    let nk = r.u32();
    let mut kfs = BTreeMap::new();
    for _ in 0..nk {
        let id = r.u64();
        let frame_id = r.u64();
        let bad = r.u8() != 0;
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
        kfs.insert(id, KeyFrame { id, frame_id, bad, tcw, twc, keys, desc, u_right, depth, map_points, feat, grid, conn_weights, ordered_conn, ctx: ctx.clone(), ..Default::default() });
    }
    let nm = r.u32();
    let mut mps = BTreeMap::new();
    for _ in 0..nm {
        let id = r.u64();
        let bad = r.u8() != 0;
        let pos = [r.f32(), r.f32(), r.f32()];
        let normal = [r.f32(), r.f32(), r.f32()];
        let desc = r.take::<32>();
        let (min_dist, max_dist) = (r.f32(), r.f32());
        let track_scale_level = r.i32();
        let no = r.u32();
        let obs = (0..no).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
        mps.insert(id, MapPoint { id, bad, pos, normal, desc, min_dist, max_dist, track_scale_level, obs, ..Default::default() });
    }
    World2 { kfs, mps }
}

struct OsimRec {
    kf1: KfId,
    kf2: KfId,
    min: Vec<Option<MpId>>,
    sin: G2oSim3,
    ret: i32,
    mout: Vec<Option<MpId>>,
    sout: G2oSim3,
}

struct Srv {
    _c: Child,
    i: ChildStdin,
    o: ChildStdout,
}
impl Srv {
    fn rd<const N: usize>(&mut self) -> [u8; N] {
        let mut a = [0u8; N];
        self.o.read_exact(&mut a).expect("mlp_srv 응답 없음");
        a
    }
}

#[derive(Default)]
struct Tally {
    n: u64,
    same: u64,
    ok: u64,
    osim_n: u64,
    osim_same: u64,
    bad: Vec<String>,
}

/// 원본 OptimizeSim3 FFI + 원본 실행 기록과 입력 · 출력 비교
struct Ffi<'a> {
    srv: &'a mut Srv,
    recs: VecDeque<OsimRec>,
    t: &'a mut Tally,
}

impl OptimizeSim3 for Ffi<'_> {
    fn optimize_sim3(&mut self, kf1: &KeyFrame, kf2: &KeyFrame, matches: &mut [Option<MpId>], s12: &mut G2oSim3, th2: f32, fix: bool, all: bool, mps: &BTreeMap<MpId, MapPoint>) -> i32 {
        let rec = self.recs.pop_front();
        let mut b: Vec<u8> = vec![4];
        for kf in [kf1, kf2] {
            for v in kf.tcw.q.iter().chain(kf.tcw.t.iter()) {
                b.extend(v.to_le_bytes());
            }
            let c = &kf.ctx.cam;
            for v in [c.fx, c.fy, c.cx, c.cy] {
                b.extend(v.to_le_bytes());
            }
            b.extend((kf.ctx.pyr.inv_sigma2.len() as u32).to_le_bytes());
            for v in &kf.ctx.pyr.inv_sigma2 {
                b.extend(v.to_le_bytes());
            }
            b.extend((kf.keys.len() as u32).to_le_bytes());
            for k in &kf.keys {
                b.extend(k.x.to_le_bytes());
                b.extend(k.y.to_le_bytes());
                b.extend(k.octave.to_le_bytes());
            }
        }
        b.extend((matches.len() as u32).to_le_bytes());
        let put = |b: &mut Vec<u8>, m: Option<MpId>| match m {
            None => b.push(0),
            Some(id) => {
                let p = &mps[&id];
                b.push(if p.bad { 2 } else { 1 });
                for v in p.pos {
                    b.extend(v.to_le_bytes());
                }
            }
        };
        for (i, m) in matches.iter().enumerate() {
            put(&mut b, kf1.map_points.get(i).copied().flatten());
            put(&mut b, *m);
            if let Some(id) = m {
                let p = &mps[id];
                b.extend(p.obs.get(&kf2.id).map_or(-1, |o| o.0).to_le_bytes());
                b.extend(p.track_scale_level.to_le_bytes());
            }
        }
        for v in s12.r.iter().chain(s12.t.iter()).chain(std::iter::once(&s12.s)) {
            b.extend(v.to_le_bytes());
        }
        b.extend(th2.to_le_bytes());
        b.push(fix as u8);
        b.push(all as u8);
        self.srv.i.write_all(&b).unwrap();
        self.srv.i.flush().unwrap();
        let r = i32::from_le_bytes(self.srv.rd());
        let keep: Vec<bool> = (0..matches.len()).map(|_| self.srv.rd::<1>()[0] != 0).collect();
        let s: [f64; 8] = std::array::from_fn(|_| f64::from_le_bytes(self.srv.rd()));
        let min = matches.to_vec();
        let sin = *s12;
        for (m, k) in matches.iter_mut().zip(keep) {
            if !k {
                *m = None;
            }
        }
        *s12 = G2oSim3 { r: [s[0], s[1], s[2], s[3]], t: [s[4], s[5], s[6]], s: s[7] };
        self.t.osim_n += 1;
        match rec {
            Some(rec) => {
                let same = rec.kf1 == kf1.id && rec.kf2 == kf2.id && rec.min == min && sim_bits(&rec.sin, &sin) && rec.ret == r && rec.mout == matches && sim_bits(&rec.sout, s12);
                self.t.osim_same += same as u64;
                if !same && self.t.bad.len() < 10 {
                    self.t.bad.push(format!(
                        "OptimizeSim3 kf {}/{}·{}/{}: 짝 입력 {} S 입력 {} 결과 {r}/{} 짝 {} S {}",
                        kf1.id,
                        rec.kf1,
                        kf2.id,
                        rec.kf2,
                        rec.min == min,
                        sim_bits(&rec.sin, &sin),
                        rec.ret,
                        rec.mout == matches,
                        sim_bits(&rec.sout, s12)
                    ));
                }
            }
            None => {
                if self.t.bad.len() < 10 {
                    self.t.bad.push("원본보다 OptimizeSim3 가 많음".into());
                }
            }
        }
        r
    }
}

fn read_inner(r: &mut Rd) -> (VecDeque<OsimRec>, Vec<i32>) {
    let mut recs = VecDeque::new();
    loop {
        let m = r.u32();
        if m == 0x444e4152 {
            let n = r.u32() as usize;
            return (recs, (0..n).map(|_| r.i32()).collect());
        }
        assert_eq!(m, 0x4d495350, "OSIM 자리 아님");
        let kf1 = r.u64();
        let kf2 = r.u64();
        let min = r.mps();
        let sin = r.sim();
        let _th2 = r.f32();
        let _fix = r.u8();
        let _all = r.u8();
        let ret = r.i32();
        let mout = r.mps();
        let sout = r.sim();
        recs.push_back(OsimRec { kf1, kf2, min, sin, ret, mout, sout });
    }
}

fn run(path: &str, srv: &mut Srv, t: &mut Tally) {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
    assert!(f.metadata().unwrap().len() > 0, "빈 덤프 {path}");
    let mut r = Rd { f: std::io::BufReader::with_capacity(1 << 20, f) };
    while r.more() {
        let kind = r.u32();
        let k = t.n;
        let mut why: Vec<String> = Vec::new();
        let ok;
        if kind == 0x42524344 {
            let cur = r.u64();
            let fix = r.u8() != 0;
            let nc = r.u32() as usize;
            let cands: Vec<Option<KfId>> = (0..nc).map(|_| r.kf()).collect();
            let mut out = BowOut { matched_kf2: r.kf(), last_current_kf: r.kf(), scw: r.sim(), n_coincidences: r.i32(), mps: r.mps(), matched_mps: r.mps(), set_not_erase: None };
            let wd = read_world(&mut r);
            let (recs, rand) = read_inner(&mut r);
            assert_eq!(r.u32(), 0x45524344, "DCRE 자리 아님");
            let ret = r.u8() != 0;
            let (o_kf2, o_last, o_scw, o_nc, o_mps, o_mm) = (r.kf(), r.kf(), r.sim(), r.i32(), r.mps(), r.mps());
            let w = World { kfs: &wd.kfs, mps: &wd.mps };
            let mut feed: VecDeque<i32> = rand.into_iter().collect();
            let mut bad_rand = false;
            let mut first_bad = String::new();
            let tot = feed.len() / 3;
            let mut rf = |a: i32, b: i32| -> i32 {
                if feed.len() < 3 {
                    bad_rand = true;
                    return a;
                }
                let (ea, eb, v) = (feed.pop_front().unwrap(), feed.pop_front().unwrap(), feed.pop_front().unwrap());
                if (ea, eb) != (a, b) {
                    if !bad_rand {
                        first_bad = format!("{} 번째 ({a},{b}) 원본 ({ea},{eb})", tot - feed.len() / 3);
                    }
                    bad_rand = true;
                    return a;
                }
                v
            };
            let mut ffi = Ffi { srv: &mut *srv, recs, t: &mut *t };
            let got = detect_common_regions_from_bow(&w, cur, fix, &cands, &mut ffi, &mut rf, &mut out);
            if !ffi.recs.is_empty() {
                why.push(format!("원본 OptimizeSim3 {} 개 남음", ffi.recs.len()));
            }
            if bad_rand || !feed.is_empty() {
                why.push(format!("난수 어긋남(전체 {tot}, 남음 {}, {first_bad})", feed.len() / 3));
            }
            if got != ret {
                why.push(format!("결과 {got}/{ret}"));
            }
            if out.matched_kf2 != o_kf2 || out.last_current_kf != o_last || out.n_coincidences != o_nc {
                why.push(format!("KF {:?}/{o_kf2:?} 일치 수 {}/{o_nc}", out.matched_kf2, out.n_coincidences));
            }
            if !sim_bits(&out.scw, &o_scw) {
                why.push("gScw 비트".into());
            }
            if out.mps != o_mps || out.matched_mps != o_mm {
                why.push("점 목록".into());
            }
            ok = ret;
            if why.is_empty() || k < 3 {
                // 자리 표시
            }
            let _ = cur;
        } else {
            assert_eq!(kind, 0x4c524344, "DCRL 자리 아님");
            let cur = r.u64();
            let member = r.u64();
            let matched = r.u64();
            let fix = r.u8() != 0;
            let scw = r.sim();
            let _n = r.i32();
            let _a = r.mps();
            let _b = r.mps();
            let wd = read_world(&mut r);
            let (recs, rand) = read_inner(&mut r);
            assert!(rand.is_empty());
            assert_eq!(r.u32(), 0x454c5244, "DRLE 자리 아님");
            let ret = r.u8() != 0;
            let (o_scw, o_n, o_mps, o_mm) = (r.sim(), r.i32(), r.mps(), r.mps());
            let w = World { kfs: &wd.kfs, mps: &wd.mps };
            let mut ffi = Ffi { srv: &mut *srv, recs, t: &mut *t };
            let o = detect_and_refine_sim3_from_last_kf(&w, cur, member, matched, &scw, fix, &mut ffi);
            if !ffi.recs.is_empty() {
                why.push("원본 OptimizeSim3 남음".into());
            }
            if o.ok != ret || o.n_proj != o_n {
                why.push(format!("결과 {}/{ret} 짝 수 {}/{o_n}", o.ok, o.n_proj));
            }
            if !sim_bits(&o.scw, &o_scw) {
                why.push("gScw 비트".into());
            }
            if o.mps.iter().map(|&x| Some(x)).collect::<Vec<_>>() != o_mps || o.matched != o_mm {
                why.push("점 목록".into());
            }
            ok = ret;
        }
        t.n += 1;
        t.ok += ok as u64;
        if why.is_empty() {
            t.same += 1;
        } else if t.bad.len() < 10 {
            t.bad.push(format!("{path} 호출 {k}({}): {}", if kind == 0x42524344 { "BoW" } else { "LastKF" }, why.join(", ")));
        }
    }
}

#[test]
fn detect_common_regions_match_original() {
    let dumps = std::env::var("ORBLOOP_LOOPDUMP").expect("입력 없음: ORBLOOP_LOOPDUMP(원본 loop_dump 덤프, : 로 여럿)");
    let exe = std::env::var("ORBLOOP_MLPSRV").expect("입력 없음: ORBLOOP_MLPSRV(원본 FFI 도우미 mlp_srv)");
    let mut c = Command::new(&exe).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap_or_else(|e| panic!("{exe}: {e}"));
    let (i, o) = (c.stdin.take().unwrap(), c.stdout.take().unwrap());
    let mut srv = Srv { _c: c, i, o };
    let mut t = Tally::default();
    for p in dumps.split(':') {
        run(p, &mut srv, &mut t);
    }
    eprintln!("호출 {}(참 {}) 중 같음 {}, OptimizeSim3 {} 중 같음 {}", t.n, t.ok, t.same, t.osim_n, t.osim_same);
    for s in &t.bad {
        eprintln!("  {s}");
    }
    assert!(t.n > 0 && t.osim_n > 0, "OptimizeSim3 까지 가는 호출이 없는 입력");
    assert_eq!(t.same, t.n);
    assert_eq!(t.osim_same, t.osim_n);
}
