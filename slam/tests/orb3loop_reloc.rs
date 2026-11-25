//! 재위치 흐름(`orb3loop::reloc::relocalize`) 원본 대조. 원본 기준 빌드를 추적을 잃게 하는 순서(되감기 · 건너뛰기)로 돌린
//! 덤프(`~/opt/orb3_loop_bg/reloc/reloc_dump`)에서 재위치 시도마다 후보 질의 직후 상태를 Rust 자료로 옮겨 같은 흐름을 돌리고,
//! PnP 는 원본 MLPnPsolver FFI 도우미(`mlp_srv`, 기준 빌드에 링크, 원본 실행의 난수를 그대로 받음)로 부른다.
//! 비교: iterate 마다 짝 목록 · 결과, 끝 결과(성공 · 자세 · 지도 점 · 이상치).
//! ORBLOOP_RELOC=덤프1:덤프2 ORBLOOP_MLPSRV=.../mlp_srv cargo test --release -p slam --test orb3loop_reloc
use slam::orb3loop::reloc::*;
use slam::orb3rs::*;
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

struct Rd {
    b: Vec<u8>,
    o: usize,
}
impl Rd {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let a: [u8; N] = self.b[self.o..self.o + N].try_into().unwrap();
        self.o += N;
        a
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
    fn descs(&mut self, n: usize) -> Vec<Desc> {
        (0..n).map(|_| self.take::<32>()).collect()
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
    fn mp(&mut self) -> Option<MpId> {
        let v = self.i64();
        (v >= 0).then_some(v as u64)
    }
}

struct PnpRec {
    solver: i32,
    matches: Vec<Option<MpId>>,
    niter: i32,
    rand: Vec<i32>,
    ret: bool,
    nomore: bool,
    ninl: i32,
    inl: Vec<bool>,
    t: [f32; 16],
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
    iter_same: u64,
    iter_n: u64,
    match_same: u64,
    rel_same: u64,
    rel_n: u64,
    ok_n: u64,
    pnp_ok: u64,
    bad: Vec<String>,
}

/// 원본 MLPnP(FFI) + 원본 실행 기록 대조
struct Ffi<'a> {
    srv: &'a mut Srv,
    recs: VecDeque<PnpRec>,
    cam: [f32; 4],
    sigma2: Vec<f32>,
    nmade: i32,
    t: &'a mut Tally,
    rel: u64,
}

struct Sol<'a, 'b> {
    ffi: *mut Ffi<'a>,
    h: u32,
    idx: i32,
    matches: Vec<Option<MpId>>,
    _p: std::marker::PhantomData<&'b ()>,
}

impl<'a> PnpFactory for Ffi<'a> {
    type S = Sol<'a, 'a>;
    fn make(&mut self, f: &Frame, matches: &[Option<MpId>], mps: &BTreeMap<MpId, MapPoint>) -> Self::S {
        let mut b: Vec<u8> = vec![1];
        for v in self.cam {
            b.extend(v.to_le_bytes());
        }
        b.extend((self.sigma2.len() as u32).to_le_bytes());
        for v in &self.sigma2 {
            b.extend(v.to_le_bytes());
        }
        b.extend((matches.len() as u32).to_le_bytes());
        for (i, m) in matches.iter().enumerate() {
            let (x, y, o) = if i < f.keys.len() { (f.keys[i].x, f.keys[i].y, f.keys[i].octave) } else { (0.0, 0.0, 0) };
            b.extend(x.to_le_bytes());
            b.extend(y.to_le_bytes());
            b.extend(o.to_le_bytes());
            match m.map(|id| &mps[&id]).filter(|p| !p.bad) {
                Some(p) => {
                    b.push(1);
                    for v in p.pos {
                        b.extend(v.to_le_bytes());
                    }
                }
                None => b.push(0),
            }
        }
        self.srv.i.write_all(&b).unwrap();
        self.srv.i.flush().unwrap();
        let h = u32::from_le_bytes(self.srv.rd());
        let idx = self.nmade;
        self.nmade += 1;
        Sol { ffi: self as *mut _, h, idx, matches: matches.to_vec(), _p: Default::default() }
    }
}

impl<'a, 'b> PnpSolver for Sol<'a, 'b> {
    fn iterate(&mut self, n: i32) -> PnpStep {
        // 안전: Ffi 는 relocalize 가 끝날 때까지 살아 있고 iterate 는 한 스레드에서 차례로 불린다
        let ffi = unsafe { &mut *self.ffi };
        let rel = ffi.rel;
        let rec = ffi.recs.pop_front().unwrap_or_else(|| panic!("재위치 {rel}: 원본보다 iterate 가 많음"));
        assert_eq!(rec.solver, self.idx, "재위치 {rel}: iterate 순서(풀이기 번호) 다름");
        assert_eq!(rec.niter, n);
        let same_m = rec.matches == self.matches;
        ffi.t.match_same += same_m as u64;
        let mut b: Vec<u8> = vec![2];
        b.extend(self.h.to_le_bytes());
        b.extend(n.to_le_bytes());
        b.extend((rec.rand.len() as u32).to_le_bytes());
        for v in &rec.rand {
            b.extend(v.to_le_bytes());
        }
        ffi.srv.i.write_all(&b).unwrap();
        ffi.srv.i.flush().unwrap();
        let ret = ffi.srv.rd::<1>()[0] != 0;
        let nomore = ffi.srv.rd::<1>()[0] != 0;
        let ninl = i32::from_le_bytes(ffi.srv.rd());
        let ni = u32::from_le_bytes(ffi.srv.rd()) as usize;
        let inl: Vec<bool> = (0..ni).map(|_| ffi.srv.rd::<1>()[0] != 0).collect();
        let t: [f32; 16] = std::array::from_fn(|_| f32::from_le_bytes(ffi.srv.rd()));
        let used = u32::from_le_bytes(ffi.srv.rd()) as usize;
        let q: [f32; 4] = std::array::from_fn(|_| f32::from_le_bytes(ffi.srv.rd()));
        let tt: [f32; 3] = std::array::from_fn(|_| f32::from_le_bytes(ffi.srv.rd()));
        let same = same_m
            && ret == rec.ret
            && nomore == rec.nomore
            && ninl == rec.ninl
            && inl == rec.inl
            && used == rec.rand.len()
            && (!ret || t.iter().zip(rec.t.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
        ffi.t.iter_n += 1;
        ffi.t.iter_same += same as u64;
        ffi.t.pnp_ok += ret as u64;
        if !same && ffi.t.bad.len() < 8 {
            ffi.t.bad.push(format!("재위치 {rel} iterate 풀이기 {}: 짝 같음 {same_m} ret {ret}/{} nomore {nomore}/{} ninl {ninl}/{} 난수 {used}/{}", self.idx, rec.ret, rec.nomore, rec.ninl, rec.rand.len()));
        }
        PnpStep { pose: ret.then_some(Se3 { q, t: tt }), no_more: nomore, inliers: inl }
    }
}

/// 원본 실행 난수를 차례로 내줌(범위가 다르면 실패)
struct Feed(VecDeque<i32>);
impl RandomInt for Feed {
    fn random_int(&mut self, a: i32, b: i32) -> i32 {
        assert!(self.0.len() >= 3, "난수 모자람");
        let (ea, eb, r) = (self.0.pop_front().unwrap(), self.0.pop_front().unwrap(), self.0.pop_front().unwrap());
        assert_eq!((ea, eb), (a, b), "난수 범위 다름");
        r
    }
}

/// Rust MLPnP 를 감싸 iterate 마다 원본 기록과 비교
struct Chk<'a, 'r> {
    inner: RustPnp<'r, Feed>,
    recs: VecDeque<PnpRec>,
    t: &'a mut Tally,
    nmade: i32,
}
struct ChkS<'a, 'r> {
    s: RustPnpSolver<'r, Feed>,
    c: *mut Chk<'a, 'r>,
    idx: i32,
    matches: Vec<Option<MpId>>,
}
impl<'a, 'r> PnpFactory for Chk<'a, 'r> {
    type S = ChkS<'a, 'r>;
    fn make(&mut self, f: &Frame, matches: &[Option<MpId>], mps: &BTreeMap<MpId, MapPoint>) -> Self::S {
        let s = self.inner.make(f, matches, mps);
        self.nmade += 1;
        ChkS { s, c: self as *mut _, idx: self.nmade - 1, matches: matches.to_vec() }
    }
}
impl PnpSolver for ChkS<'_, '_> {
    fn iterate(&mut self, n: i32) -> PnpStep {
        let c = unsafe { &mut *self.c };
        let rec = c.recs.pop_front().expect("원본보다 iterate 가 많음");
        assert_eq!(rec.solver, self.idx, "iterate 순서 다름");
        let st = self.s.iterate(n);
        let same = rec.matches == self.matches && st.pose.is_some() == rec.ret && st.no_more == rec.nomore && st.inliers == rec.inl;
        c.t.iter_n += 1;
        c.t.iter_same += same as u64;
        c.t.match_same += (rec.matches == self.matches) as u64;
        c.t.pnp_ok += rec.ret as u64;
        st
    }
}

fn run(path: &str, srv: &mut Srv, t: &mut Tally, rust: bool) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
    assert!(!b.is_empty(), "빈 덤프 {path}");
    let mut r = Rd { b, o: 0 };
    while r.o < r.b.len() {
        assert_eq!(r.u32(), 0x434c4552, "RELC 자리 아님");
        let fid = r.u64();
        let ts = r.f64();
        let _map = r.u64();
        let (fx, fy, cx, cy, invfx, invfy, bf, bb, thd) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (min_x, max_x, min_y, max_y, giw, gih) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let nlevels = r.i32();
        let scale_factor = r.f32();
        let log_scale_factor = r.f32();
        let (scale, inv_scale, sigma2, inv_sigma2) = (r.fv(), r.fv(), r.fv(), r.fv());
        let cam = Camera { fx, fy, cx, cy, invfx, invfy, bf, b: bb, th_depth: thd, width: 0, height: 0 };
        let pyr = Pyramid { nfeatures: 0, scale_factor, nlevels, ini_th_fast: 0, min_th_fast: 0, scale, inv_scale, sigma2: sigma2.clone(), inv_sigma2, log_scale_factor };
        let ctx = Arc::new(FrameCtx { cam, pyr, min_x, max_x, min_y, max_y, grid_inv_w: giw, grid_inv_h: gih });
        let keys = r.keys();
        let n = keys.len();
        let desc = r.descs(n);
        let (u_right, depth) = (r.fv(), r.fv());
        let mut grid = vec![Vec::new(); GRID_COLS * GRID_ROWS];
        for c in 0..GRID_COLS {
            for w in 0..GRID_ROWS {
                let k = r.u32();
                grid[c * GRID_ROWS + w] = (0..k).map(|_| r.u32()).collect();
            }
        }
        let nb = r.u32();
        let bow: BTreeMap<u32, f64> = (0..nb).map(|_| (r.u32(), r.f64())).collect();
        let feat = r.feat();
        let has0 = r.u8() != 0;
        let pose0 = r.se3();
        let mp0: Vec<Option<MpId>> = (0..n).map(|_| r.mp()).collect();
        let out0: Vec<bool> = (0..n).map(|_| r.u8() != 0).collect();
        let mut f = Frame {
            id: fid,
            timestamp: ts,
            keys,
            desc,
            u_right,
            depth,
            map_points: mp0,
            outlier: out0,
            grid,
            ctx: ctx.clone(),
            bow,
            feat,
            ..Default::default()
        };
        if has0 {
            f.set_pose(pose0);
        }
        let nc = r.u32() as usize;
        let mut cands = Vec::new();
        let mut kfs = BTreeMap::new();
        for _ in 0..nc {
            let id = r.u64();
            let frame_id = r.u64();
            let bad = r.u8() != 0;
            let tcw = r.se3();
            let twc = r.se3();
            let keys = r.keys();
            let desc = r.descs(keys.len());
            let (u_right, depth) = (r.fv(), r.fv());
            let nm = r.u32() as usize;
            let map_points = (0..nm).map(|_| r.mp()).collect();
            let feat = r.feat();
            cands.push(id);
            kfs.insert(id, KeyFrame { id, frame_id, bad, tcw, twc, keys, desc, u_right, depth, map_points, feat, ctx: ctx.clone(), ..Default::default() });
        }
        let nmp = r.u32();
        let mut mps = BTreeMap::new();
        for _ in 0..nmp {
            let id = r.u64();
            let bad = r.u8() != 0;
            let pos = [r.f32(), r.f32(), r.f32()];
            let normal = [r.f32(), r.f32(), r.f32()];
            let desc = r.take::<32>();
            let (min_dist, max_dist) = (r.f32(), r.f32());
            mps.insert(id, MapPoint { id, bad, pos, normal, desc, min_dist, max_dist, ..Default::default() });
        }
        let mut recs = VecDeque::new();
        loop {
            let m = r.u32();
            if m == 0x454c4552 {
                break;
            }
            assert_eq!(m, 0x49504e50, "PNPI 자리 아님");
            let solver = r.i32();
            let nm = r.u32() as usize;
            let matches = (0..nm).map(|_| r.mp()).collect();
            let niter = r.i32();
            let nr = r.u32() as usize;
            let rand = (0..nr).map(|_| r.i32()).collect();
            let ret = r.u8() != 0;
            let nomore = r.u8() != 0;
            let ninl = r.i32();
            let ni = r.u32() as usize;
            let inl = (0..ni).map(|_| r.u8() != 0).collect();
            let tm: [f32; 16] = std::array::from_fn(|_| r.f32());
            recs.push_back(PnpRec { solver, matches, niter, rand, ret, nomore, ninl, inl, t: tm });
        }
        let ret = r.u8() != 0;
        let has_pose = r.u8() != 0;
        let pose = r.se3();
        let nn = r.u32() as usize;
        assert_eq!(nn, n);
        let omp: Vec<Option<MpId>> = (0..n).map(|_| r.mp()).collect();
        let oout: Vec<bool> = (0..n).map(|_| r.u8() != 0).collect();

        let rel = t.rel_n;
        let (ok, left) = if rust {
            let feed = std::cell::RefCell::new(Feed(recs.iter().flat_map(|r| r.rand.iter().copied()).collect()));
            let mut c = Chk { inner: RustPnp { rand: &feed }, recs, t: &mut *t, nmade: 0 };
            let ok = relocalize(&mut f, &cands, &kfs, &mps, &mut c);
            (ok, c.recs.len() + feed.borrow().0.len())
        } else {
            let mut ffi = Ffi { srv: &mut *srv, recs, cam: [fx, fy, cx, cy], sigma2, nmade: 0, t: &mut *t, rel };
            let ok = relocalize(&mut f, &cands, &kfs, &mps, &mut ffi);
            let left = ffi.recs.len();
            srv.i.write_all(&[3]).unwrap();
            (ok, left)
        };
        let mut why = Vec::new();
        if left != 0 {
            why.push(format!("원본 iterate {left} 개 남음"));
        }
        if ok != ret {
            why.push(format!("성공 {ok}/{ret}"));
        }
        if has_pose {
            let p = f.tcw.unwrap_or_default();
            if p.q.iter().chain(p.t.iter()).zip(pose.q.iter().chain(pose.t.iter())).any(|(a, b)| a.to_bits() != b.to_bits()) {
                why.push("자세 비트".into());
            }
        }
        if f.map_points != omp {
            why.push(format!("지도 점 {}/{} 칸 다름", f.map_points.iter().zip(&omp).filter(|(a, b)| a != b).count(), n));
        }
        if f.outlier != oout {
            why.push("이상치 표시".into());
        }
        t.rel_n += 1;
        t.ok_n += ret as u64;
        if why.is_empty() {
            t.rel_same += 1;
        } else if t.bad.len() < 8 {
            t.bad.push(format!("{path} 재위치 {rel}(장면 {fid}, 후보 {nc}): {}", why.join(", ")));
        }
    }
}

fn check(rust: bool) {
    let dumps = std::env::var("ORBLOOP_RELOC").expect("입력 없음: ORBLOOP_RELOC(원본 reloc_dump 덤프, : 로 여럿)");
    let exe = std::env::var("ORBLOOP_MLPSRV").expect("입력 없음: ORBLOOP_MLPSRV(원본 MLPnP FFI 도우미 mlp_srv)");
    let mut c = Command::new(&exe).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap_or_else(|e| panic!("{exe}: {e}"));
    let (i, o) = (c.stdin.take().unwrap(), c.stdout.take().unwrap());
    let mut srv = Srv { _c: c, i, o };
    let mut t = Tally::default();
    for p in dumps.split(':') {
        run(p, &mut srv, &mut t, rust);
    }
    eprintln!(
        "[{}] 재위치 {} 번(성공 {}) 중 같음 {}, iterate {} 번(자세 나옴 {}) 중 같음 {}, 짝 목록 같음 {}",
        if rust { "Rust MLPnP" } else { "원본 MLPnP FFI" },
        t.rel_n, t.ok_n, t.rel_same, t.iter_n, t.pnp_ok, t.iter_same, t.match_same
    );
    for s in &t.bad {
        eprintln!("  {s}");
    }
    assert!(t.rel_n > 0 && t.ok_n > 0, "재위치 성공이 없는 입력");
    assert_eq!(t.rel_same, t.rel_n);
    assert_eq!(t.iter_same, t.iter_n);
}

#[test]
fn relocalization_with_original_mlpnp_ffi() {
    check(false);
}

#[test]
fn relocalization_with_rust_mlpnp() {
    check(true);
}
