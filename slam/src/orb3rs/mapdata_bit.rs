//! 원본 덤프 대조(`slam/cpp/orb3_mapdump.cc` 가 기준 빌드 원본을 돌리며 남긴 호출 앞뒤 상태).
//! `ORBMAP_DUMPS=a.bin:b.bin cargo test --release -p slam --lib orb3rs::mapdata` — 입력이 없으면 실패.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

struct R<'a> {
    b: &'a [u8],
    p: usize,
    /// 덤프 끝이 잘림(원본 실행이 교착으로 시간 초과 종료될 때 마지막 기록)
    eof: bool,
}
impl R<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        if self.p + N > self.b.len() {
            self.eof = true;
            self.p = self.b.len();
            return [0; N];
        }
        let v: [u8; N] = self.b[self.p..self.p + N].try_into().unwrap();
        self.p += N;
        v
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
    fn se3(&mut self) -> Se3 {
        Se3 { q: [self.f32(), self.f32(), self.f32(), self.f32()], t: [self.f32(), self.f32(), self.f32()] }
    }
    fn ids(&mut self) -> Vec<u64> {
        let n = if self.eof { 0 } else { self.u32() as usize };
        if self.eof {
            return Vec::new();
        }
        (0..n).map(|_| self.u64()).collect()
    }
}

struct DMap {
    id: u64,
    init: u64,
    max: u64,
    lower: i64,
    initial: i64,
    big: i32,
    change: i32,
    nmp: u64,
    nref: u32,
    kfs: Vec<u64>,
}
struct DKf {
    id: u64,
    bad: bool,
    map: i64,
    first: bool,
    not_erase: bool,
    to_be: bool,
    tcw: Se3,
    twc: Se3,
    tcp: Se3,
    tcw_bm: Se3,
    twc_bm: Se3,
    conn: BTreeMap<u64, i32>,
    ordered: Vec<(u64, i32)>,
    parent: i64,
    children: Vec<u64>,
    loops: Vec<u64>,
    merges: Vec<u64>,
}
struct DMp {
    id: u64,
    bad: bool,
    map: i64,
    inmap: bool,
    pos: V3,
    normal: V3,
    min: f32,
    max: f32,
    desc: Desc,
    nobs: i32,
    ref_kf: i64,
    visible: i32,
    found: i32,
    replaced: i64,
    first_kf: i64,
    obs: BTreeMap<u64, (i32, i32)>,
}
#[derive(Default)]
struct State {
    maps: Vec<DMap>,
    g: Vec<DKf>,
    l: Vec<(u64, BTreeMap<u32, u64>)>,
    m: Vec<DMp>,
}
struct Static {
    _addr: u64,
    oct: Vec<i32>,
    ur: Vec<f32>,
    desc: Vec<Desc>,
}

fn rd_state(r: &mut R) -> State {
    let mut s = State::default();
    for _ in 0..r.u32() {
        s.maps.push(DMap { id: r.u64(), init: r.u64(), max: r.u64(), lower: r.i64(), initial: r.i64(), big: r.i32(), change: r.i32(), nmp: r.u64(), nref: r.u32(), kfs: r.ids() });
    }
    for _ in 0..r.u32() {
        let (id, bad, map, first, not_erase, to_be) = (r.u64(), r.u8() != 0, r.i64(), r.u8() != 0, r.u8() != 0, r.u8() != 0);
        let (tcw, twc, tcp, tcw_bm, twc_bm) = (r.se3(), r.se3(), r.se3(), r.se3(), r.se3());
        let n = r.u32();
        let conn = (0..n).map(|_| (r.u64(), r.i32())).collect();
        let n = r.u32();
        let ordered = (0..n).map(|_| (r.u64(), r.i32())).collect();
        let parent = r.i64();
        let (children, loops, merges) = (r.ids(), r.ids(), r.ids());
        s.g.push(DKf { id, bad, map, first, not_erase, to_be, tcw, twc, tcp, tcw_bm, twc_bm, conn, ordered, parent, children, loops, merges });
    }
    for _ in 0..r.u32() {
        let id = r.u64();
        let n = r.u32();
        s.l.push((id, (0..n).map(|_| (r.u32(), r.u64())).collect()));
    }
    for _ in 0..r.u32() {
        let (id, bad, map, inmap) = (r.u64(), r.u8() != 0, r.i64(), r.u8() != 0);
        let pos = [r.f32(), r.f32(), r.f32()];
        let normal = [r.f32(), r.f32(), r.f32()];
        let (min, max) = (r.f32(), r.f32());
        let desc: Desc = r.take();
        let (nobs, ref_kf, visible, found, replaced, first_kf) = (r.i32(), r.i64(), r.i32(), r.i32(), r.i64(), r.i64());
        let n = r.u32();
        let obs = (0..n).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
        s.m.push(DMp { id, bad, map, inmap, pos, normal, min, max, desc, nobs, ref_kf, visible, found, replaced, first_kf, obs });
    }
    s
}

fn opt(x: i64) -> Option<u64> {
    (x >= 0).then_some(x as u64)
}

type StOf<'a> = dyn Fn(u64) -> Option<&'a Static> + 'a;

/// 원본 std::map/set<KeyFrame*> 은 주소 순이라, 기록 안 키프레임들의 주소 순이 id 순과 다르면 id 를 주소 순위로 바꿔 대조한다
/// (Rust 판은 id 순으로 돌므로 같은 순서가 됨. 키프레임 id 는 같음 비교로만 쓰이고, 크기 비교는 Map 의 최대 · 최저 id 뿐 — 그 둘은 이때 빼고 비교).
fn relabel(s: &mut State, f: &dyn Fn(u64) -> Option<u64>) {
    // 주소를 모르는 키프레임(덤프에 정적 기록이 없음: 지도 첫 kf 등 같음 비교로만 쓰이는 것)은 겹치지 않는 큰 번호로
    let f = |x: u64| f(x).unwrap_or(u64::MAX - x);
    let fi = |x: i64| if x >= 0 { f(x as u64) as i64 } else { x };
    for m in &mut s.maps {
        m.init = f(m.init);
        m.lower = -1;
        m.max = 0; // 크기 비교 칸이라 바꿔 대조할 때는 비교하지 않음
        m.initial = fi(m.initial);
        m.kfs = m.kfs.iter().map(|&k| f(k)).collect();
    }
    for k in &mut s.g {
        k.id = f(k.id);
        k.conn = k.conn.iter().map(|(&a, &w)| (f(a), w)).collect();
        k.ordered = k.ordered.iter().map(|&(a, w)| (f(a), w)).collect();
        k.parent = fi(k.parent);
        for v in [&mut k.children, &mut k.loops, &mut k.merges] {
            *v = v.iter().map(|&x| f(x)).collect();
        }
    }
    for l in &mut s.l {
        l.0 = f(l.0);
    }
    for p in &mut s.m {
        p.ref_kf = fi(p.ref_kf);
        p.obs = p.obs.iter().map(|(&k, &v)| (f(k), v)).collect();
    }
}
fn state_kfs(s: &State, out: &mut BTreeSet<u64>) {
    for m in &s.maps {
        out.extend(m.kfs.iter().copied().chain(opt(m.initial)));
    }
    for k in &s.g {
        out.insert(k.id);
        out.extend(k.conn.keys().copied().chain(k.ordered.iter().map(|x| x.0)).chain(opt(k.parent)));
        out.extend(k.children.iter().chain(&k.loops).chain(&k.merges));
    }
    out.extend(s.l.iter().map(|x| x.0));
    for p in &s.m {
        out.extend(p.obs.keys().copied().chain(opt(p.ref_kf)));
    }
}

fn kf_from_static(id: u64, st: &StOf, ctx: &Arc<FrameCtx>) -> KeyFrame {
    let s = st(id).unwrap();
    let keys = s.oct.iter().map(|&o| KeyPoint { octave: o, ..Default::default() }).collect();
    KeyFrame { id, keys, u_right: s.ur.clone(), desc: s.desc.clone(), map_points: vec![None; s.oct.len()], ctx: ctx.clone(), ..Default::default() }
}

fn build(s: &State, st: &StOf, ctx: &Arc<FrameCtx>) -> Atlas {
    let mut a = Atlas::default();
    for m in &s.maps {
        let mut mps: Vec<u64> = s.m.iter().filter(|p| p.inmap && p.map == m.id as i64).map(|p| p.id).collect();
        mps.sort();
        a.maps.push(Map {
            id: m.id,
            init_kf_id: m.init,
            max_kf_id: m.max,
            lower_kf: opt(m.lower),
            origin_kf: opt(m.initial),
            big_change_idx: m.big,
            change_id: m.change,
            kfs: sorted(&m.kfs),
            mps,
            ref_mps: vec![0; m.nref as usize],
            ..Default::default()
        });
    }
    let mut want: BTreeSet<u64> = BTreeSet::new();
    for k in &s.g {
        want.insert(k.id);
        want.extend(k.conn.keys());
        want.extend(k.ordered.iter().map(|x| x.0));
        want.extend(opt(k.parent));
        want.extend(k.children.iter().chain(&k.loops).chain(&k.merges));
    }
    for (k, _) in &s.l {
        want.insert(*k);
    }
    for p in &s.m {
        want.extend(p.obs.keys());
        want.extend(opt(p.ref_kf));
    }
    for id in want {
        let k = if st(id).is_some() { kf_from_static(id, st, ctx) } else { KeyFrame { id, ctx: ctx.clone(), ..Default::default() } };
        a.kfs.insert(id, k);
    }
    for d in &s.g {
        let k = a.kfs.get_mut(&d.id).unwrap();
        k.bad = d.bad;
        k.map = d.map.max(0) as u64;
        k.first_conn_done = !d.first;
        k.not_erase = d.not_erase;
        k.to_be_erased = d.to_be;
        k.tcw = d.tcw;
        k.twc = d.twc;
        k.tcp = d.tcp;
        k.tcw_bef_merge = d.tcw_bm;
        k.twc_bef_merge = d.twc_bm;
        k.conn_weights = d.conn.clone();
        k.ordered_conn = d.ordered.clone();
        k.parent = opt(d.parent);
        k.children = sorted(&d.children);
        k.loop_edges = sorted(&d.loops);
        k.merge_edges = sorted(&d.merges);
    }
    for (id, l) in &s.l {
        let k = a.kfs.get_mut(id).unwrap();
        for (&i, &mp) in l {
            k.map_points[i as usize] = Some(mp);
        }
    }
    for d in &s.m {
        a.mps.insert(
            d.id,
            MapPoint {
                id: d.id,
                map: d.map.max(0) as u64,
                pos: d.pos,
                normal: d.normal,
                desc: d.desc,
                obs: d.obs.clone(),
                n_obs: d.nobs,
                ref_kf: d.ref_kf.max(0) as u64,
                first_kf: d.first_kf as u64,
                visible: d.visible,
                found: d.found,
                min_dist: d.min,
                max_dist: d.max,
                bad: d.bad,
                replaced: opt(d.replaced),
                ..Default::default()
            },
        );
    }
    a
}

fn sorted(v: &[u64]) -> Vec<u64> {
    let mut v = v.to_vec();
    v.sort();
    v
}
fn bits3(v: &V3) -> [u32; 3] {
    v.map(f32::to_bits)
}
fn se3b(s: &Se3) -> [u32; 7] {
    [s.q[0], s.q[1], s.q[2], s.q[3], s.t[0], s.t[1], s.t[2]].map(f32::to_bits)
}
fn inmap_count(s: &State, map: u64) -> usize {
    s.m.iter().filter(|p| p.inmap && p.map == map as i64).count()
}

/// 덤프 뒤 상태와 Rust 상태 비교, 다른 칸 이름들
fn compare(a: &Atlas, before: &State, after: &State, addr_ok: bool, base_mps: usize) -> Vec<String> {
    let mut bad = Vec::new();
    let tag = |s: &str| s.to_string();
    for (m, m0) in after.maps.iter().zip(&before.maps) {
        let r = a.map_ref(m.id).unwrap();
        // 빈 지도의 mpKFinitial 은 덤프가 -1 로 적음(첫 키프레임 전 미초기화 포인터를 피하려고) — 비교 안 함
        let rorig = if m.kfs.is_empty() { None } else { r.origin_kf };
        let morig = if m.kfs.is_empty() { None } else { opt(m.initial) };
        let (rmax, rlow, dmax, dlow) = if addr_ok { (r.max_kf_id, r.lower_kf, m.max, opt(m.lower)) } else { (0, None, 0, None) };
        if (r.init_kf_id, rmax, rlow, rorig, r.big_change_idx, r.change_id) != (m.init, dmax, dlow, morig, m.big, m.change) {
            bad.push(format!("map{} 칸 rs {:?} 원본 {:?}", m.id, (r.init_kf_id, rmax, rlow, r.origin_kf, r.big_change_idx, r.change_id), (m.init, dmax, dlow, opt(m.initial), m.big, m.change)));
        }
        if r.kfs != sorted(&m.kfs) {
            bad.push(format!("map{} kfs", m.id));
        }
        let base = if base_mps != usize::MAX { base_mps } else { inmap_count(before, m.id) };
        if r.mps.len() as i64 - base as i64 != m.nmp as i64 - m0.nmp as i64 || r.ref_mps.len() != m.nref as usize {
            bad.push(format!("map{} 점 수", m.id));
        }
    }
    for d in &after.g {
        let k = &a.kfs[&d.id];
        let mut c = |name: String, ok: bool| {
            if !ok {
                bad.push(format!("kf{} {}", d.id, name));
            }
        };
        c("bad".into(), k.bad == d.bad);
        c("flags".into(), (k.first_conn_done, k.not_erase, k.to_be_erased) == (!d.first, d.not_erase, d.to_be));
        c("conn".into(), k.conn_weights == d.conn);
        c(tag("ordered"), k.ordered_conn == d.ordered);
        c(tag("parent"), k.parent == opt(d.parent));
        c(tag("children"), k.children == sorted(&d.children));
        c("loops".into(), k.loop_edges == sorted(&d.loops) && k.merge_edges == sorted(&d.merges));
        c("tcp".into(), se3b(&k.tcp) == se3b(&d.tcp));
        c("pose".into(), se3b(&k.tcw) == se3b(&d.tcw) && se3b(&k.twc) == se3b(&d.twc));
        c("bef_merge".into(), se3b(&k.tcw_bef_merge) == se3b(&d.tcw_bm) && se3b(&k.twc_bef_merge) == se3b(&d.twc_bm));
    }
    for (id, l) in &after.l {
        let k = &a.kfs[id];
        let r: BTreeMap<u32, u64> = k.map_points.iter().enumerate().filter_map(|(i, m)| m.map(|m| (i as u32, m))).collect();
        if &r != l {
            bad.push(format!("kf{} 점 칸", id));
        }
    }
    for d in &after.m {
        let p = &a.mps[&d.id];
        let has_map = after.maps.iter().any(|m| m.id as i64 == d.map);
        let inmap = !has_map || a.map_ref(p.map).is_some_and(|m| m.mps.binary_search(&p.id).is_ok()) == d.inmap;
        let mut c = |name: String, ok: bool| {
            if !ok {
                bad.push(format!("mp{} {}", d.id, name));
            }
        };
        c("bad".into(), p.bad == d.bad);
        c("inmap".into(), inmap);
        c("pos".into(), bits3(&p.pos) == bits3(&d.pos));
        c("normal".into(), bits3(&p.normal) == bits3(&d.normal));
        c("dist".into(), (p.min_dist.to_bits(), p.max_dist.to_bits()) == (d.min.to_bits(), d.max.to_bits()));
        c(tag("desc"), p.desc == d.desc);
        c("nobs".into(), p.n_obs == d.nobs);
        if !d.obs.is_empty() {
            c(tag("ref"), p.ref_kf as i64 == d.ref_kf);
        }
        c("count".into(), (p.visible, p.found) == (d.visible, d.found));
        c("replaced".into(), p.replaced == opt(d.replaced));
        c("first_kf".into(), p.first_kf as i64 == d.first_kf);
        c("obs".into(), p.obs == d.obs);
    }
    bad
}

/// 덤프에 나온 키프레임들의 주소 순 = id 순인지(원본 std::map/set 순서가 Rust 와 같은지)
fn addr_monotone(addrs: &BTreeMap<u64, u64>) -> bool {
    addrs.values().zip(addrs.values().skip(1)).all(|(a, b)| a < b)
}

const NAMES: [&str; 33] = [
    "", "UpdateConnections", "KF SetBadFlag", "KF SetErase", "AddConnection", "EraseConnection", "AddChild", "EraseChild", "ChangeParent",
    "AddLoopEdge", "AddMergeEdge", "SetNotErase", "KF AddMapPoint", "EraseMapPointMatch(idx)", "EraseMapPointMatch(mp)", "ReplaceMapPointMatch",
    "AddObservation", "EraseObservation", "MP SetBadFlag", "Replace", "ComputeDistinctiveDescriptors", "UpdateNormalAndDepth",
    "PredictScale", "", "Map AddKeyFrame", "Map EraseKeyFrame", "Map AddMapPoint", "Map EraseMapPoint",
    "InformNewBigChange", "IncreaseChangeIndex", "SetReferenceMapPoints", "UpdateBestCovisibles", "OptimizeEssentialGraph(합치기)",
];

#[test]
fn original_dumps() {
    let dumps = std::env::var("ORBMAP_DUMPS").expect("ORBMAP_DUMPS(원본 덤프 경로, : 로 구분) 가 필요함");
    let mut total_bad = 0usize;
    for path in dumps.split(':').filter(|s| !s.is_empty()) {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(!b.is_empty(), "{path}: 빈 덤프");
        let mut r = R { b: &b, p: 0, eof: false };
        let mut st: BTreeMap<u64, Static> = BTreeMap::new();
        let mut addrs: BTreeMap<u64, u64> = BTreeMap::new();
        let mut ctx = Arc::new(FrameCtx::default());
        let (mut n, mut nbad, mut nadd) = ([0usize; 33], [0usize; 33], [0usize; 33]);
        let mut first: Vec<String> = Vec::new();
        let mut shown = [0usize; 33];
        let mut nkill = 0usize;
        let mut nmeg_addr = 0usize;
        while r.p < b.len() {
            match r.u32() {
                0x4448444d => {
                    let nl = r.i32();
                    let lsf = r.f32();
                    let scale: Vec<f32> = (0..nl).map(|_| r.f32()).collect();
                    let pyr = Pyramid { nlevels: nl, log_scale_factor: lsf, scale, ..Default::default() };
                    ctx = Arc::new(FrameCtx { pyr, ..Default::default() });
                }
                0x5453444d => {
                    let id = r.u64();
                    let addr = r.u64();
                    addrs.insert(id, addr);
                    let nk = r.u32() as usize;
                    let mut s = Static {
                        _addr: addr, oct: Vec::with_capacity(nk), ur: Vec::with_capacity(nk), desc: Vec::with_capacity(nk) };
                    for _ in 0..nk {
                        s.oct.push(r.i32());
                        s.ur.push(r.f32());
                        s.desc.push(r.take());
                    }
                    st.insert(id, s);
                }
                0x5350444d => {
                    let (mx, dist, lsf, nl, res) = (r.f32(), r.f32(), r.f32(), r.i32(), r.i32());
                    n[22] += 1;
                    let p = MapPoint { max_dist: mx, ..Default::default() };
                    let pyr = Pyramid { nlevels: nl, log_scale_factor: lsf, ..Default::default() };
                    if p.predict_scale(dist, &pyr) != res {
                        nbad[22] += 1;
                        if first.len() < 40 {
                            first.push(format!("PredictScale max {mx} dist {dist} → 원본 {res}"));
                        }
                    }
                }
                0x4352444d => {
                    let op = r.u8() as usize;
                    let (x, y, z) = (r.i64(), r.i64(), r.i64());
                    let lists: Vec<Vec<u64>> = if op == 32 { (0..4).map(|_| r.ids()).collect() } else { Vec::new() };
                    let before = rd_state(&mut r);
                    let after = rd_state(&mut r);
                    if r.eof {
                        eprintln!("  (끝 기록 잘림: {} — 원본 실행이 중간에 끝남)", NAMES[op]);
                        break;
                    }
                    let mut before = before;
                    let mut after = after;
                    let mut ids = BTreeSet::new();
                    state_kfs(&before, &mut ids);
                    state_kfs(&after, &mut ids);
                    let kf_args: &[bool; 3] = match op {
                        1..=3 | 11..=15 | 31 => &[true, false, false],
                        4..=10 => &[true, true, false],
                        16 | 17 | 24 | 25 => &[false, true, false],
                        _ => &[false, false, false],
                    };
                    let mut args = [x, y, z];
                    for (v, &is) in args.iter().zip(kf_args) {
                        if is && *v >= 0 {
                            ids.insert(*v as u64);
                        }
                    }
                    let by_addr: Vec<u64> = {
                        let mut v: Vec<u64> = ids.iter().copied().collect();
                        v.retain(|i| addrs.contains_key(i));
                        v.sort_by_key(|i| addrs[i]);
                        v
                    };
                    // Map 넣기 · 빼기는 순서에 안 닿고 최대 · 최저 id 를 크기로 비교하므로 바꾸지 않음
                    let addr_mono = by_addr.iter().copied().eq(ids.iter().copied().filter(|i| addrs.contains_key(i)));
                    // 합치기 본질 그래프는 id 크기 비교(l < i)를 쓰므로 바꾸지 않고 대조(주소 순이 달라도 결과에 닿는 집합은 루프 간선 · 관측뿐)
                    let addr_ok = matches!(op, 24 | 25 | 32) || addr_mono;
                    let fwd: BTreeMap<u64, u64> = by_addr.iter().enumerate().map(|(r, &i)| (i, r as u64)).collect();
                    let inv: BTreeMap<u64, u64> = fwd.iter().map(|(&a, &b)| (b, a)).collect();
                    if !addr_ok {
                        let f = |i: u64| fwd.get(&i).copied();
                        relabel(&mut before, &f);
                        relabel(&mut after, &f);
                        for (v, &is) in args.iter_mut().zip(kf_args) {
                            if is && *v >= 0 {
                                *v = fwd.get(&(*v as u64)).copied().unwrap_or(u64::MAX - *v as u64) as i64;
                            }
                        }
                    }
                    let (x, y, z) = (args[0], args[1], args[2]);
                    let st_of = |i: u64| -> Option<&Static> { st.get(&if addr_ok { i } else { *inv.get(&i)? }) };
                    let mut a = build(&before, &st_of, &ctx);
                    let mut base_mps = usize::MAX;
                    let id = x as u64;
                    match op {
                        1 => kf_update_connections(&mut a, id, y != 0),
                        2 => {
                            kf_set_bad_flag(&mut a, id);
                        }
                        3 => {
                            kf_set_erase(&mut a, id);
                        }
                        4 => kf_add_connection(&mut a, id, y as u64, z as i32),
                        5 => kf_erase_connection(&mut a, id, y as u64),
                        6 => a.kfs.get_mut(&id).unwrap().add_child(y as u64),
                        7 => a.kfs.get_mut(&id).unwrap().erase_child(y as u64),
                        8 => kf_change_parent(&mut a, id, y as u64),
                        9 => a.kfs.get_mut(&id).unwrap().add_loop_edge(y as u64),
                        10 => a.kfs.get_mut(&id).unwrap().add_merge_edge(y as u64),
                        11 => a.kfs.get_mut(&id).unwrap().set_not_erase(),
                        12 | 15 => a.kfs.get_mut(&id).unwrap().map_points[z as usize] = opt(y),
                        13 => a.kfs.get_mut(&id).unwrap().erase_map_point_match_idx(z as usize),
                        14 => kf_erase_map_point_match(&mut a, id, y as u64),
                        16 => {
                            let k = a.kfs.get(&(y as u64)).cloned().unwrap_or_else(|| kf_from_static(y as u64, &st_of, &ctx));
                            a.mps.get_mut(&id).unwrap().add_observation(&k, z as usize);
                        }
                        17 => mp_erase_observation(&mut a, id, y as u64),
                        18 => mp_set_bad_flag(&mut a, id),
                        19 => mp_replace(&mut a, id, y as u64),
                        20 => {
                            let mut p = a.mps.remove(&id).unwrap();
                            p.compute_distinctive_descriptors(&a.kfs);
                            a.mps.insert(id, p);
                        }
                        21 => mp_update_normal_and_depth(&mut a, id),
                        24 => a.map_mut(id).unwrap().add_keyframe(y as u64),
                        25 => a.map_mut(id).unwrap().erase_keyframe(y as u64),
                        26 | 27 => {
                            // 점의 지도가 이 지도가 아닐 수 있어 넣기 전 이 지도 소속은 덤프 값(z)으로
                            let m = a.map_mut(id).unwrap();
                            if z != 0 { set_insert(&mut m.mps, y as u64); } else { set_remove(&mut m.mps, y as u64); }
                            base_mps = m.mps.len();
                            if op == 26 { m.add_map_point(y as u64) } else { m.erase_map_point(y as u64) }
                        }
                        28 => a.map_mut(id).unwrap().inform_new_big_change(),
                        29 => a.map_mut(id).unwrap().increase_change_index(),
                        31 => kf_update_best_covisibles(&mut a, id),
                        32 => super::super::essential_merge::optimize_essential_graph_merge(&mut a, id, &lists[0], &lists[1], &lists[2], &lists[3]),
                        30 => a.map_mut(id).unwrap().set_reference_map_points(vec![0; y as usize]),
                        _ => panic!("모르는 op {op}"),
                    }
                    if !addr_ok || (op == 32 && !addr_mono) {
                        nadd[op] += 1;
                    }
                    let mut diffs = compare(&a, &before, &after, addr_ok, base_mps);
                    if op == 1 {
                        let k = &a.kfs[&id];
                        for nn in [5, 10, 20, 30] {
                            let v: Vec<u64> = r.ids().into_iter().map(|i| if addr_ok { i } else { fwd.get(&i).copied().unwrap_or(u64::MAX - i) }).collect();
                            if v != k.best_covisibility(nn) {
                                diffs.push(format!("best_covisibility({nn})"));
                            }
                        }
                        for w in [0, 15, 30, 60, 100] {
                            let v: Vec<u64> = r.ids().into_iter().map(|i| if addr_ok { i } else { fwd.get(&i).copied().unwrap_or(u64::MAX - i) }).collect();
                            if v != k.covisibles_by_weight(w) {
                                diffs.push(format!("covisibles_by_weight({w})"));
                            }
                        }
                    }
                    if op == 32 {
                        // 법선 합은 관측 순서(원본 주소 순)에 닿음: 그 점 관측 kf 의 주소 순 = id 순일 때만 비교
                        // 관측 순서(원본 주소 순)에 닿는 것: 법선 합, 나쁜 기준 kf 를 지운 뒤 새 기준(→ 위치 · 거리).
                        // 그 점 관측 kf 의 주소 순 = id 순인 점만 비교(주소 순 자체는 UpdateNormalAndDepth · EraseObservation 시험에서 확인)
                        diffs.retain(|d| {
                            let Some(rest) = d.strip_prefix("mp") else { return true };
                            let id: u64 = rest.split(' ').next().unwrap().parse().unwrap();
                            let p = before.m.iter().find(|m| m.id == id).unwrap();
                            let ad: Vec<u64> = p.obs.keys().map(|k| addrs[k]).collect();
                            let mono = ad.windows(2).all(|w| w[0] < w[1]);
                            if !mono {
                                nmeg_addr += 1;
                            }
                            mono
                        });
                    }
                    n[op] += 1;
                    if matches!(op, 2 | 3) && after.g.iter().any(|k| k.id == x as u64 && k.bad) {
                        nkill += 1;
                    }
                    if !diffs.is_empty() {
                        nbad[op] += 1;
                        shown[op] += 1;
                        if shown[op] <= 3 {
                            first.push(format!("{} #{} (id {x}): {}", NAMES[op], n[op], diffs[..diffs.len().min(6)].join(", ")));
                        }
                    }
                }
                m => panic!("{path}: 모르는 기록 {m:#x} @{}", r.p),
            }
        }
        eprintln!("== {path}: 키프레임 {} 개, 주소 순 = id 순: {}", addrs.len(), addr_monotone(&addrs));
        for op in 1..33 {
            if n[op] > 0 {
                eprintln!("  {:32} {:6} 호출, 다름 {:4} (주소 순위로 바꿔 대조 {})", NAMES[op], n[op], nbad[op], nadd[op]);
            }
        }
        eprintln!("  (SetBadFlag 이 실제로 키프레임을 지운 호출 {nkill}, 합치기 본질 그래프에서 관측 주소 순≠id 순이라 법선 비교를 뺀 점 {nmeg_addr})");
        for f in &first {
            eprintln!("  ! {f}");
        }
        total_bad += nbad.iter().sum::<usize>();
    }
    assert_eq!(total_bad, 0, "원본과 다른 호출이 있음");
}
