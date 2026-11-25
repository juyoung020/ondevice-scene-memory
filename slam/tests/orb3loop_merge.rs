//! MergeLocal(`orb3loop::merge`) 원본 대조(번호 순 비교자 기준 빌드 `~/opt/orb3_refid`, 덤프 `merge_dump`).
//! 덤프: 들어갈 때 Atlas 통째(지도 목록 · 모든 지도 키프레임 · 점), 맞춤 BA 앞 · 뒤, 본질 그래프 앞 · 뒤, 끝 Release 직전, 끝 지도 목록 · 합침 가장자리.
//! BA 두 개는 원본 결과(자세 · 점 · 관측)를 넣고 들어갈 때 Rust 상태를 비교. 원래 CorrectLoop 머리말:
//! CorrectLoop(`orb3loop::correct`) 원본 대조. 원본은 번호 순 비교자 기준 빌드(`~/opt/orb3_refid`)를 돌린 덤프
//! (`~/opt/orb3_loop_bg/reloc/correct_dump`): 루프 닫기마다 지도 스레드 멈춘 뒤 지도 통째 · 입력, OptimizeEssentialGraph
//! 들어갈 때(입력 세 표 · 지도) · 나올 때 지도, 끝 지도. OptimizeEssentialGraph 는 원본 결과(자세 · 점 위치)를 넣고,
//! 들어갈 때 Rust 상태가 원본과 같은지 · 끝 상태가 같은지 본다.
//! ORBLOOP_MERGE=덤프1:덤프2 cargo test --release -p slam --test orb3loop_merge
use slam::orb3loop::merge::*;
use slam::orb3loop::detect::G2oSim3;
use slam::orb3rs::*;
use std::collections::BTreeMap;
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

struct DAtlasMap {
    id: u64,
    bad: bool,
    current: bool,
    kfs: Vec<KfId>,
    mps: Vec<MpId>,
}

fn read_map(r: &mut Rd) -> (Vec<DAtlasMap>, DMap) {
    let nmaps = r.u32();
    let mut maps = Vec::new();
    for _ in 0..nmaps {
        let id = r.u64();
        let bad = r.u8() != 0;
        let current = r.u8() != 0;
        let _init = r.opt();
        let _lower = r.opt();
        let _ini = r.u64();
        let _max = r.u64();
        let kfs = r.ids();
        let mps = r.ids();
        maps.push(DAtlasMap { id, bad, current, kfs, mps });
    }
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
    let mut not_erase = BTreeMap::new();
    for _ in 0..nk {
        let id = r.u64();
        let frame_id = r.u64();
        let bad = r.u8() != 0;
        not_erase.insert(id, r.u8() != 0);
        let first_conn_done = r.u8() == 0;
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
            KeyFrame { id, frame_id, map, bad, first_conn_done, tcw, twc, keys, desc, u_right, depth, map_points, feat, grid, conn_weights, ordered_conn, parent, children, loop_edges, merge_edges, ctx: ctx.clone(), ..Default::default() },
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
    (maps, DMap { kfs, mps, not_erase, corrected_by })
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
fn diff(kfs: &BTreeMap<KfId, KeyFrame>, mps: &BTreeMap<MpId, MapPoint>, d: &DMap, tr: &dyn Fn(u64) -> u64) -> BTreeMap<&'static str, usize> {
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
        add("KF 지도", tr(k.map) != o.map);
        add("KF 부모 · 자식", k.parent != o.parent || k.children != o.children);
        add("KF 첫 연결", k.first_conn_done != o.first_conn_done);
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
        add("점 nObs", p.n_obs != o.n_obs);
        add("점 지도", tr(p.map) != o.map);
        // mnVisible · mnFound 는 추적 스레드가 CorrectLoop 중에도 늘림(원본도 같이 돎) — 따로 셈
        add("점 보임 · 찾음(추적 스레드)", p.visible != o.visible || p.found != o.found);
    }
    c
}

/// 추적 스레드가 같이 바꾸는 칸(mnVisible · mnFound)은 판정에서 뺌(원본 실행 중 추적이 멈추지 않음)
fn strip(mut d: BTreeMap<&'static str, usize>) -> BTreeMap<&'static str, usize> {
    d.remove("점 보임 · 찾음(추적 스레드)");
    d
}


/// BA 결과 넣기: 자세 · 지도 점 칸 · 연결(키프레임), 위치 · 법선 · 거리 · 관측 · 나쁨(점)
fn apply(a: &mut Atlas, d: &DMap) {
    for (id, o) in &d.kfs {
        if let Some(k) = a.kfs.get_mut(id) {
            k.tcw = o.tcw;
            k.twc = o.twc;
            k.map_points = o.map_points.clone();
            k.conn_weights = o.conn_weights.clone();
            k.ordered_conn = o.ordered_conn.clone();
        }
    }
    for (id, o) in &d.mps {
        if let Some(p) = a.mps.get_mut(id) {
            p.pos = o.pos;
            p.normal = o.normal;
            p.min_dist = o.min_dist;
            p.max_dist = o.max_dist;
            p.obs = o.obs.clone();
            p.bad = o.bad;
            p.n_obs = o.n_obs;
            p.ref_kf = o.ref_kf;
            p.replaced = o.replaced;
            p.desc = o.desc;
        }
    }
}

type Tr = std::rc::Rc<std::cell::Cell<(u64, u64)>>;
fn trf(t: &Tr) -> impl Fn(u64) -> u64 + '_ {
    move |m| {
        let (from, to) = t.get();
        if m == from {
            to
        } else {
            m
        }
    }
}

struct Lba {
    rec: Option<(KfId, Vec<KfId>, Vec<KfId>, DMap, DMap)>,
    why: Vec<String>,
    tr: Tr,
}
impl MergeLba for Lba {
    fn run(&mut self, a: &mut Atlas, main: KfId, adjust: &[KfId], fixed: &[KfId]) {
        let (m, ad, fx, b, af) = self.rec.take().expect("원본 맞춤 BA 없음");
        if (m, ad.as_slice(), fx.as_slice()) != (main, adjust, fixed) {
            self.why.push(format!("맞춤 BA 입력 창 {}/{} 고정 {}/{}", adjust.len(), ad.len(), fixed.len(), fx.len()));
        }
        let d = strip(diff(&a.kfs, &a.mps, &b, &trf(&self.tr)));
        if !d.is_empty() {
            self.why.push(format!("맞춤 BA 앞 {d:?}"));
        }
        if rust_mode() {
            // Rust 맞춤 BA(orb3ba::mlba)로 풀고 원본 뒤 상태와 비교 — 다음 단계는 원본 값으로 이어감
            let (mut ext, mut marks) = (Default::default(), Default::default());
            slam::orb3sys::mergeops::RsMergeLba { ext: &mut ext, marks: &mut marks }.run(a, main, adjust, fixed);
            let d = strip(diff(&a.kfs, &a.mps, &af, &trf(&self.tr)));
            if !d.is_empty() {
                self.why.push(format!("Rust 맞춤 BA 뒤 {d:?}"));
            }
        }
        apply(a, &af);
    }
}
/// ORBLOOP_RUST_MERGE=1: 맞춤 BA · 본질 그래프도 Rust 로 풀어 원본 결과와 비교
fn rust_mode() -> bool {
    std::env::var_os("ORBLOOP_RUST_MERGE").is_some()
}

struct Ess {
    bef: BTreeMap<KfId, Se3>,
    rec: Option<(KfId, [Vec<KfId>; 3], Vec<Option<MpId>>, DMap, DMap)>,
    why: Vec<String>,
    tr: Tr,
}
impl MergeEssential for Ess {
    fn run(&mut self, a: &mut Atlas, cur: KfId, f: &[KfId], fc: &[KfId], nf: &[KfId], nmp: &[MpId]) {
        let (c, lists, mps, b, af) = self.rec.take().expect("원본 본질 그래프 없음");
        let nmp2: Vec<Option<MpId>> = nmp.iter().map(|&x| Some(x)).collect();
        if c != cur || lists[0] != f || lists[1] != fc || lists[2] != nf || mps != nmp2 {
            self.why.push(format!(
                "본질 그래프 입력 목록 cur {} 고정 {}/{} {} 보정 {}/{} {} 나머지 {}/{} {} 점 {}/{} {}",
                c == cur,
                f.len(), lists[0].len(), lists[0] == f,
                fc.len(), lists[1].len(), lists[1] == fc,
                nf.len(), lists[2].len(), lists[2] == nf,
                nmp2.len(), mps.len(), mps == nmp2
            ));
        }
        let d = strip(diff(&a.kfs, &a.mps, &b, &trf(&self.tr)));
        if !d.is_empty() {
            self.why.push(format!("본질 그래프 앞 {d:?}"));
        }
        if rust_mode() {
            let mut e = slam::orb3sys::mergeops::RsMergeEss { bef: std::mem::take(&mut self.bef) };
            e.run(a, cur, f, fc, nf, nmp);
            let d = strip(diff(&a.kfs, &a.mps, &af, &trf(&self.tr)));
            if !d.is_empty() {
                self.why.push(format!("Rust 본질 그래프 뒤 {d:?}"));
            }
        }
        apply(a, &af);
    }
}

fn to_atlas(maps: Vec<DAtlasMap>, d: DMap) -> Atlas {
    let mut a = Atlas { kfs: d.kfs, mps: d.mps, ..Default::default() };
    for m in maps {
        // 원본 Map 칸(첫 · 가장 낮은 키프레임 등)은 add_keyframe 순서로 다시 세움: 번호 순으로 넣으면 첫 = 가장 낮은 번호
        let mut mm = Map { id: m.id, bad: m.bad, ..Default::default() };
        for k in &m.kfs {
            mm.add_keyframe(*k);
        }
        mm.mps = m.mps;
        if m.current {
            a.current = a.maps.len();
        }
        a.maps.push(mm);
    }
    a
}

#[test]
fn merge_local_matches_original() {
    let dumps = std::env::var("ORBLOOP_MERGE").expect("입력 없음: ORBLOOP_MERGE(원본 merge_dump 덤프, : 로 여럿)");
    let (mut n, mut same) = (0, 0);
    let mut bad = Vec::new();
    for path in dumps.split(':') {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
        assert!(f.metadata().unwrap().len() > 0, "빈 덤프 {path}");
        let mut r = Rd { f: std::io::BufReader::with_capacity(1 << 20, f) };
        while r.more() {
            assert_eq!(r.u32(), 0x4752454d, "MERG 자리 아님");
            let cur = r.u64();
            let matched = r.u64();
            let _fix = r.u8();
            let gba = r.u8() != 0;
            let scw = r.sim();
            let (m0, d0) = read_map(&mut r);
            assert_eq!(r.u32(), 0x4241424c, "LBAB 자리 아님");
            let lm = r.u64();
            let la: Vec<KfId> = r.mps().into_iter().map(|x| x.unwrap()).collect();
            let lf: Vec<KfId> = r.mps().into_iter().map(|x| x.unwrap()).collect();
            let (_, lb) = read_map(&mut r);
            assert_eq!(r.u32(), 0x4141424c, "LBAA 자리 아님");
            let (_, laf) = read_map(&mut r);
            let mut ess_rec = None;
            let mut m = r.u32();
            if m == 0x42325345 {
                let c = r.u64();
                let l0: Vec<KfId> = r.mps().into_iter().map(|x| x.unwrap()).collect();
                let l1: Vec<KfId> = r.mps().into_iter().map(|x| x.unwrap()).collect();
                let l2: Vec<KfId> = r.mps().into_iter().map(|x| x.unwrap()).collect();
                let mp = r.mps();
                let (_, eb) = read_map(&mut r);
                assert_eq!(r.u32(), 0x41325345, "ES2A 자리 아님");
                let (_, ea) = read_map(&mut r);
                ess_rec = Some((c, [l0, l1, l2], mp, eb, ea));
                m = r.u32();
            }
            assert_eq!(m, 0x4c45524d, "MREL 자리 아님");
            let (_, rel) = read_map(&mut r);
            assert_eq!(r.u32(), 0x45475245, "ERGE 자리 아님");
            let ogba = r.u8() != 0;
            let nm = r.u32();
            let omaps: Vec<u64> = (0..nm).map(|_| r.u64()).collect();
            let ocur = r.u64();
            let oe1 = r.ids();
            let oe2 = r.ids();
            let _ = (r.u8(), r.u8());

            let mut a = to_atlas(m0, d0);
            for k in a.kfs.values_mut() {
                // 덤프의 mbFirstConnection 그대로
                let _ = k;
            }
            let tr: Tr = Default::default();
            let cur_map = a.kfs[&cur].map;
            let merge_map = a.kfs[&matched].map;
            // 원본은 합치기 덤프가 ChangeId 뒤라 합칠 지도 번호가 지금 지도 번호로 보임
            tr.set((merge_map, cur_map));
            let mut lba = Lba { rec: Some((lm, la, lf, lb, laf)), why: Vec::new(), tr: tr.clone() };
            let bef = a.kfs.iter().filter(|(_, k)| k.map == cur_map).map(|(id, k)| (*id, k.tcw)).collect();
            let mut ess = Ess { bef, rec: ess_rec, why: Vec::new(), tr: tr.clone() };
            let mut why_rel = Vec::new();
            let inp = MergeIn { cur, matched, scw, gba_running: gba };
            let out = merge_local(&inp, &mut a, &mut lba, &mut ess, &mut |a: &Atlas, _mm, _cm| {
                let d = strip(diff(&a.kfs, &a.mps, &rel, &trf(&tr)));
                if !d.is_empty() {
                    why_rel.push(format!("끝(Release 앞) {d:?}"));
                }
            });
            let mut why = lba.why;
            why.extend(ess.why);
            if lba.rec.is_some() {
                why.push("맞춤 BA 안 불림".into());
            }
            if ess.rec.is_some() {
                why.push("본질 그래프 안 불림".into());
            }
            why.extend(why_rel);
            if out.relaunch_gba != ogba {
                why.push("GBA".into());
            }
            let rmaps: Vec<u64> = a.maps.iter().map(|m| m.id).collect();
            if rmaps != omaps || a.maps[a.current].id != ocur {
                why.push(format!("지도 목록 {rmaps:?}/{omaps:?}"));
            }
            if a.kfs[&cur].merge_edges != oe1 || a.kfs[&matched].merge_edges != oe2 {
                why.push("합침 가장자리".into());
            }
            n += 1;
            if why.is_empty() {
                same += 1;
            } else {
                bad.push(format!("{path} 합치기 {n}(KF {cur}↔{matched}): {}", why.join(" / ")));
            }
        }
    }
    eprintln!("합치기 {n} 번 중 같음 {same}");
    for b in &bad {
        eprintln!("  {b}");
    }
    assert!(n > 0, "합치기가 없는 입력");
    assert_eq!(same, n);
}
