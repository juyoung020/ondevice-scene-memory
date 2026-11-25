//! 포터 TRACK 비트 비교: 원본 ORB-SLAM3 를 돌린 하네스(저장소 밖 ~/opt/orb3_track_bg/track_dump)가 쓴 덤프를 읽어
//! 같은 입력으로 Rust(slam::orb3rs::track · localmap)를 돌리고 원본 출력과 비교한다.
//! cargo run --release --example orb3rs_track_replay -- dump.bin
use slam::orb3rs::localmap::*;
use slam::orb3rs::track::*;
use slam::orb3rs::*;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};

struct R<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> R<'a> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let a: [u8; N] = self.b[self.o..self.o + N].try_into().unwrap();
        self.o += N;
        a
    }
    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }
    fn b(&mut self) -> bool {
        self.u8() != 0
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
}

#[derive(Default, Debug)]
struct Tally {
    n: u64,
    same: u64,
    examples: Vec<String>,
}
impl Tally {
    fn add(&mut self, ok: bool, why: impl FnOnce() -> String) {
        self.n += 1;
        if ok {
            self.same += 1;
        } else if self.examples.len() < 5 {
            self.examples.push(why());
        }
    }
}

struct Lm {
    stopped: bool,
    stop_req: bool,
    accept: bool,
    init: bool,
    queue: i32,
}
impl MapperStatus for Lm {
    fn is_stopped(&self) -> bool {
        self.stopped
    }
    fn stop_requested(&self) -> bool {
        self.stop_req
    }
    fn accept_keyframes(&self) -> bool {
        self.accept
    }
    fn is_initializing(&self) -> bool {
        self.init
    }
    fn keyframes_in_queue(&self) -> i32 {
        self.queue
    }
    fn interrupt_ba(&self) {}
}

fn kf_entry(a: &mut Atlas, id: KfId) -> &mut KeyFrame {
    a.kfs.entry(id).or_insert_with(|| KeyFrame { id, ..Default::default() })
}

fn nnkf(r: &mut R, t: &mut Tally, t_racy: &mut Tally) {
    let id = r.u64();
    let mut tr = Tracker { sensor: r.i32(), state: r.i32(), only_tracking: r.b(), ..Default::default() };
    let mut lm = Lm { stopped: r.b(), stop_req: r.b(), accept: false, init: false, queue: 0 };
    let n_kfs = r.i32();
    tr.last_reloc_frame_id = r.u64();
    tr.max_frames = r.i32();
    tr.min_frames = r.i32();
    tr.last_kf_frame_id = r.u64();
    let nref = r.u32();
    let refs: Vec<(bool, i32)> = (0..nref).map(|_| (r.b(), r.i32())).collect();
    lm.accept = r.b();
    lm.init = r.b();
    lm.queue = r.i32();
    let n = r.u32() as usize;
    let mut f = Frame { id, ..Default::default() };
    for i in 0..n {
        f.depth.push(r.f32());
        let fl = r.u8();
        f.map_points.push(if fl & 1 != 0 { Some(i as u64) } else { None });
        f.outlier.push(fl & 2 != 0);
        f.keys.push(KeyPoint::default());
    }
    tr.th_depth = r.f32();
    tr.matches_inliers = r.i32();
    tr.has_camera2 = r.b();
    f.timestamp = r.f64();
    let lts = r.f64();
    let has_last = r.b();
    let want = r.b();
    let racy = r.b();
    let ctx = NeedKfCtx {
        n_kfs,
        ref_tracked: Box::new(move |min_obs| refs.iter().filter(|(bad, o)| !bad && *o >= min_obs).count() as i32),
        imu_initialized: false,
        last_kf_ts: if has_last { Some(lts) } else { None },
    };
    let got = need_new_keyframe(&tr, &f, &lm, &ctx);
    if racy {
        // 원본이 부르는 도중 다른 스레드가 입력을 바꿈: 덤프 값으로는 원본 판단을 다시 만들 수 없어 따로 셈
        t_racy.add(got == want, || format!("장면 {id}: 원본 {want} rust {got}"));
    } else {
        let nr3 = (ctx.ref_tracked)(if n_kfs <= 2 { 2 } else { 3 });
        t.add(got == want, || format!("장면 {id}: 원본 {want} rust {got} (상태 {} kfs {n_kfs} 기준 {nr3} 짝 {} 받음 {} 시작 {} 대기 {} 멈춤 {} {} 마지막KF {} reloc {})", tr.state, tr.matches_inliers, lm.accept, lm.init, lm.queue, lm.stopped, lm.stop_req, tr.last_kf_frame_id, tr.last_reloc_frame_id));
    }
}

fn read_kf(r: &mut R, a: &mut Atlas, ext: &mut TrackExt, addr: &mut HashMap<KfId, u64>) {
    let id = r.u64();
    addr.insert(id, r.u64());
    let bad = r.b();
    ext.kf_track_ref.insert(id, r.u64());
    let nb = r.u32();
    let mut best = Vec::new();
    for _ in 0..nb {
        let b = r.u64();
        let bb = r.b();
        ext.kf_track_ref.insert(b, r.u64());
        kf_entry(a, b).bad = bb;
        best.push((b, 0));
    }
    let nc = r.u32();
    let mut ch = Vec::new();
    for _ in 0..nc {
        let c = r.u64();
        let cb = r.b();
        ext.kf_track_ref.insert(c, r.u64());
        addr.insert(c, r.u64());
        kf_entry(a, c).bad = cb;
        ch.push(c);
    }
    let p = r.u64();
    let parent = if p == u64::MAX {
        None
    } else {
        let pb = r.b();
        ext.kf_track_ref.insert(p, r.u64());
        kf_entry(a, p).bad = pb;
        Some(p)
    };
    let k = kf_entry(a, id);
    k.bad = bad;
    k.ordered_conn = best;
    k.children = ch;
    k.parent = parent;
}

fn ulkf(r: &mut R, t: &mut Tally, tref: &mut Tally) {
    let cur = r.u64();
    let last_reloc = r.u64();
    let _imu = r.b();
    let sensor = r.i32();
    let n = r.u32() as usize;
    let mut a = Atlas::default();
    let mut ext = TrackExt::default();
    let mut f = Frame::default();
    let mut next_mp = 1u64;
    for _ in 0..n {
        match r.u8() {
            0 => f.map_points.push(None),
            1 => {
                a.mps.insert(next_mp, MapPoint { id: next_mp, bad: true, ..Default::default() });
                f.map_points.push(Some(next_mp));
                next_mp += 1;
            }
            _ => {
                let no = r.u32();
                let mut m = MapPoint { id: next_mp, ..Default::default() };
                for _ in 0..no {
                    m.obs.insert(r.u64(), (0, -1));
                }
                a.mps.insert(next_mp, m);
                f.map_points.push(Some(next_mp));
                next_mp += 1;
            }
        }
    }
    let nk = r.u32();
    let mut addr = HashMap::new();
    for _ in 0..nk {
        read_kf(r, &mut a, &mut ext, &mut addr);
    }
    let ref0 = r.u64();
    let nl = r.u32();
    let want: Vec<u64> = (0..nl).map(|_| r.u64()).collect();
    let want_ref = r.u64();
    let want_slots: Vec<bool> = (0..n).map(|_| r.b()).collect();
    let mut tr = Tracker { sensor, last_reloc_frame_id: last_reloc, reference_kf: if ref0 == u64::MAX { None } else { Some(ref0) }, ..Default::default() };
    let mut cref = None;
    // 기준 원본은 결정성 패치판(std::set/map<KeyFrame*> 이 번호 순) — 주소 대신 id 순
    let _ = &addr;
    let order = |k: KfId| k;
    update_local_keyframes(&mut tr, &mut f, cur, &mut cref, &a, &mut ext, &order);
    let got_slots: Vec<bool> = f.map_points.iter().map(|s| s.is_some()).collect();
    let got_ref = tr.reference_kf.unwrap_or(u64::MAX);
    t.add(tr.local_kfs == want && got_slots == want_slots, || format!("장면 {cur}: 원본 {:?}\n   rust {:?}", want, tr.local_kfs));
    tref.add(got_ref == want_ref, || format!("장면 {cur}: 기준 원본 {want_ref} rust {got_ref}"));
}

fn ulmp(r: &mut R, t: &mut Tally) {
    let cur = r.u64();
    let mut a = Atlas::default();
    let mut ext = TrackExt::default();
    let nk = r.u32();
    let mut tr = Tracker::default();
    for _ in 0..nk {
        let id = r.u64();
        let sz = r.u32() as usize;
        let n = r.u32();
        let k = kf_entry(&mut a, id);
        k.map_points = vec![None; sz];
        for _ in 0..n {
            let i = r.u32() as usize;
            k.map_points[i] = Some(r.u64());
        }
        tr.local_kfs.push(id);
    }
    let nu = r.u32();
    for _ in 0..nu {
        let id = r.u64();
        let bad = r.b();
        ext.mp_track_ref.insert(id, r.u64());
        a.mps.insert(id, MapPoint { id, bad, ..Default::default() });
    }
    let nl = r.u32();
    let want: Vec<u64> = (0..nl).map(|_| r.u64()).collect();
    update_local_points(&mut tr, cur, &a, &mut ext);
    t.add(tr.local_mps == want, || format!("장면 {cur}: 원본 {} 개 rust {} 개", want.len(), tr.local_mps.len()));
}

fn cnkf(r: &mut R, tpose: &mut Tally, tpts: &mut Tally, stats: &mut [u64; 2]) {
    let id = r.u64();
    let _sensor = r.i32();
    let q = [r.f32(), r.f32(), r.f32(), r.f32()];
    let tt = [r.f32(), r.f32(), r.f32()];
    let (cx, cy, invfx, invfy) = (r.f32(), r.f32(), r.f32(), r.f32());
    let cam = Camera { fx: 0.0, fy: 0.0, cx, cy, invfx, invfy, bf: 0.0, b: 0.0, th_depth: 0.0, width: 0, height: 0 };
    let th = r.f32();
    let _nleft = r.i32();
    let n = r.u32() as usize;
    let mut f = Frame { id, ..Default::default() };
    let mut nobs = HashMap::new();
    for i in 0..n {
        let x = r.f32();
        let y = r.f32();
        f.keys.push(KeyPoint { x, y, ..Default::default() });
        f.depth.push(r.f32());
        let o = r.i32();
        if o >= 0 {
            f.map_points.push(Some(i as u64));
            nobs.insert(i as u64, o);
        } else {
            f.map_points.push(None);
        }
    }
    let made = r.b();
    let mut rwc = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rwc[i][j] = r.f32();
        }
    }
    let ow = [r.f32(), r.f32(), r.f32()];
    let nc = r.u32();
    let created: Vec<(i32, [f32; 3])> = (0..nc).map(|_| (r.i32(), [r.f32(), r.f32(), r.f32()])).collect();
    let ncl = r.u32();
    let _cleared: Vec<i32> = (0..ncl).map(|_| r.i32()).collect();
    if !made {
        return;
    }
    let (grwc, gow) = pose_inverse_matrices(&Se3 { q, t: tt });
    let bits = |a: &[f32]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let flat = |m: &[[f32; 3]; 3]| m.iter().flatten().copied().collect::<Vec<_>>();
    tpose.add(bits(&flat(&grwc)) == bits(&flat(&rwc)) && bits(&gow) == bits(&ow), || format!("장면 {id}: Rwc {:?} vs {:?} Ow {:?} vs {:?}", flat(&grwc), flat(&rwc), gow, ow));
    // 점 위치는 원본 자세 행렬로(자세 행렬 비교와 따로)
    let (mut got, _cl) = new_keyframe_points(&f, &rwc, &ow, &cam, th, 100, &|m| nobs[&m]);
    // 덤프는 점 번호 순
    got.sort_by_key(|g| g.0);

    stats[0] += created.len() as u64;
    let same_idx = got.len() == created.len() && got.iter().zip(&created).all(|(g, c)| g.0 as i32 == c.0);
    let mut nbit = 0;
    if same_idx {
        for (g, c) in got.iter().zip(&created) {
            if bits(&g.1) == bits(&c.1) {
                nbit += 1;
            }
        }
    }
    stats[1] += nbit;
    tpts.add(same_idx && nbit as usize == created.len(), || {
        let d = got.iter().zip(&created).find(|(g, c)| bits(&g.1) != bits(&c.1));
        format!("장면 {id}: 점 수 원본 {} rust {} 같은 번호 {same_idx} 비트 같은 {nbit} 예 {:?}", created.len(), got.len(), d)
    });
}


fn mpcl(r: &mut R, t: &mut Tally, tbad: &mut Tally) {
    let cur = r.u64();
    let mono = r.b();
    let n = r.u32();
    let mut a = Atlas::default();
    let mut ext = TrackExt::default();
    let mut recent = Vec::new();
    let mut bad0 = Vec::new();
    for _ in 0..n {
        let id = r.u64();
        let bad = r.b();
        let found = r.i32();
        let visible = r.i32();
        let first = r.i64();
        let o = r.i32();
        a.mps.insert(id, MapPoint { id, bad, found, visible, first_kf: first as u64, n_obs: o, ..Default::default() });
        recent.push(id);
        bad0.push(bad);
    }
    let nr = r.u32();
    let want: Vec<u64> = (0..nr).map(|_| r.u64()).collect();
    let _bad1: Vec<bool> = (0..n).map(|_| r.b()).collect();
    let nc = r.u32();
    // 원본이 SetBadFlag 를 부른 점(연쇄로 불린 것 포함 — 컬링 속 MapPoint::SetBadFlag 는 다른 점을 부르지 않음)
    let want_bad: Vec<u64> = (0..nc).map(|_| r.u64()).collect();
    let _ = bad0;
    let set_bad = map_point_culling(&mut recent, cur, mono, &mut a, &mut ext);
    t.add(recent == want, || format!("키프레임 {cur}: 남김 원본 {} rust {}", want.len(), recent.len()));
    tbad.add(set_bad == want_bad, || format!("키프레임 {cur}: 나쁨 원본 {:?} rust {:?}", want_bad, set_bad));
}

fn kfcl(r: &mut R, t: &mut Tally) {
    let cur = r.u64();
    let mono = r.b();
    let _inertial = r.b();
    let abort0 = r.b();
    let nl = r.u32();
    let mut a = Atlas::default();
    let mut ext = TrackExt::default();
    let mut local = Vec::new();
    let mut thd = HashMap::new();
    for _ in 0..nl {
        let id = r.u64();
        let init = r.b();
        let bad = r.b();
        let ne = r.b();
        thd.insert(id, r.f32());
        let _nleft = r.i32();
        let sz = r.u32() as usize;
        let n = r.u32();
        let k = kf_entry(&mut a, id);
        k.bad = bad;
        k.map = 0;
        k.map_points = vec![None; sz];
        k.depth = vec![-1.0; sz];
        k.u_right = vec![-1.0; sz];
        k.keys = vec![KeyPoint::default(); sz];
        for _ in 0..n {
            let i = r.u32() as usize;
            k.map_points[i] = Some(r.u64());
            k.depth[i] = r.f32();
            k.keys[i].octave = r.i32();
            k.u_right[i] = r.f32();
        }
        k.not_erase = ne;
        if init {
            match a.maps.iter_mut().find(|m| m.id == 0) {
                Some(m) => m.init_kf_id = id,
                None => a.maps.push(Map { id: 0, init_kf_id: id, ..Default::default() }),
            }
        }
        local.push(id);
    }
    let nu = r.u32();
    for _ in 0..nu {
        let id = r.u64();
        let bad = r.b();
        let n_obs = r.i32();
        let no = r.u32();
        let mut m = MapPoint { id, bad, n_obs, ..Default::default() };
        for _ in 0..no {
            let k = r.u64();
            let li = r.i32();
            let ri = r.i32();
            let oct = r.i32();
            let ur = r.f32();
            m.obs.insert(k, (li, ri));
            let kk = kf_entry(&mut a, k);
            if li >= 0 {
                let l = li as usize;
                if kk.keys.len() <= l {
                    kk.keys.resize(l + 1, KeyPoint::default());
                    kk.u_right.resize(l + 1, -1.0);
                    kk.depth.resize(l + 1, -1.0);
                    kk.map_points.resize(l + 1, None);
                }
                kk.keys[l].octave = oct;
                kk.u_right[l] = ur;
                if kk.map_points[l].is_none() {
                    kk.map_points[l] = Some(id);
                }
            }
        }
        a.mps.insert(id, m);
    }
    let nc = r.u32();
    let want: Vec<u64> = (0..nc).map(|_| r.u64()).collect();
    let abort1 = r.b();
    let ab = Cell::new(abort0);
    let _ = abort1;
    let got = keyframe_culling(&local, mono, &|k| thd[&k], &|| ab.get(), &mut a, &mut ext);
    t.add(got == want, || format!("키프레임 {cur}: 원본 {:?} rust {:?} (abort {abort0}->{abort1})", want, got));
}

/// TRAK: 기록된 하위 호출 결과로 흐름을 다시 돌려, 같은 차례로 같은 것을 부르는지 · 끝 상태가 같은지
struct Script {
    evs: Vec<(i32, i64, i32)>,
    p: usize,
    ok: bool,
    why: String,
    maps: Vec<(u64, i32)>,
    cur_map: u64,
    kfs_cur: i32,
    last_set: bool,
    cur_set: bool,
    imu: bool,
}
impl Script {
    fn next(&mut self, code: i32, a: Option<i64>) -> i32 {
        if self.p >= self.evs.len() {
            if self.ok {
                self.ok = false;
                self.why = format!("rust 가 더 부름: {code}");
            }
            return 0;
        }
        let e = self.evs[self.p];
        if e.0 != code || a.map_or(false, |a| a != e.1) {
            if self.ok {
                self.ok = false;
                self.why = format!("{} 번째: 원본 {:?} rust ({code},{a:?})", self.p, e);
            }
        }
        self.p += 1;
        e.2
    }
}
impl TrackOps for Script {
    fn stereo_initialization(&mut self) -> i32 {
        self.next(5, None)
    }
    fn check_replaced_in_last_frame(&mut self) {}
    fn track_reference_keyframe(&mut self) -> bool {
        self.next(1, None) != 0
    }
    fn track_with_motion_model(&mut self) -> bool {
        self.next(2, None) != 0
    }
    fn relocalization(&mut self, map: Option<MapId>) -> bool {
        self.next(3, Some(map.map(|m| m as i64).unwrap_or(-1))) != 0
    }
    fn track_local_map(&mut self) -> bool {
        self.next(4, None) != 0
    }
    fn pose_finite(&mut self) -> bool {
        self.evs[self.p - 1].1 != 0
    }
    fn create_map_in_atlas(&mut self) {
        self.next(6, None);
    }
    fn reset_active_map(&mut self) {
        self.next(7, None);
    }
    fn change_map(&mut self, map: MapId) {
        self.next(13, Some(map as i64));
    }
    fn need_new_keyframe(&mut self, _t: &Tracker) -> bool {
        self.next(8, None) != 0
    }
    fn create_new_keyframe(&mut self) -> Option<KfId> {
        self.next(9, None);
        Some(0)
    }
    fn try_graft(&mut self) {
        // 붙이기 시도 속 재위치들(TryGraft 는 PLT 를 안 거쳐 사건이 안 묶임): 끝까지 남은 재위치 · 지도 바꾸기를 먹음
        while self.p < self.evs.len() && (self.evs[self.p].0 == 3) {
            self.p += 1;
        }
    }
    fn all_maps(&mut self) -> Vec<(MapId, i32)> {
        self.maps.clone()
    }
    fn current_map(&mut self) -> MapId {
        self.cur_map
    }
    fn kfs_in_current_map(&mut self) -> i32 {
        self.kfs_cur
    }
    fn imu_initialized(&mut self) -> bool {
        self.imu
    }
    fn last_frame_set(&mut self) -> bool {
        self.last_set
    }
    fn cur_frame_set(&mut self) -> bool {
        self.cur_set
    }
    fn post_track_cleanup(&mut self, _v: bool) {}
}

fn trak(r: &mut R, t: &mut Tally, states: &mut BTreeMap<i32, u64>) {
    let id = r.u64();
    let ts = r.f64();
    let _lts = r.f64();
    let st = r.i32();
    let only = r.b();
    let vel = r.b();
    let reloc = r.u64();
    let sensor = r.i32();
    let tsl = r.f64();
    let cur_map = r.i64();
    let imu = r.b();
    let nm = r.u32();
    let maps: Vec<(u64, i32)> = (0..nm).map(|_| (r.i64() as u64, r.i32())).collect();
    let has_last_kf = r.b();
    let last_set = r.b();
    let ne = r.u32();
    let evs: Vec<(i32, i64, i32)> = (0..ne).map(|_| (r.i32(), r.i64(), r.i32())).collect();
    let st1 = r.i32();
    let tsl1 = r.f64();
    let vel1 = r.b();
    let has_last_kf1 = r.b();
    let cur_set = r.b();
    let kfs_cur = maps.iter().find(|m| m.0 as i64 == cur_map).map(|m| m.1).unwrap_or(0);
    let mut tr = Tracker {
        sensor,
        state: st,
        only_tracking: only,
        velocity: vel,
        last_reloc_frame_id: reloc,
        time_stamp_lost: tsl,
        last_kf: if has_last_kf { Some(0) } else { None },
        ..Default::default()
    };
    let mut s = Script { evs, p: 0, ok: true, why: String::new(), maps, cur_map: cur_map as u64, kfs_cur, last_set, cur_set, imu };
    track_flow(&mut tr, id, ts, &mut s);
    if s.ok && s.p != s.evs.len() {
        s.ok = false;
        s.why = format!("원본이 더 부름: {:?}", &s.evs[s.p..]);
    }
    *states.entry(st1).or_default() += 1;
    let same_end = tr.state == st1 && tr.time_stamp_lost.to_bits() == tsl1.to_bits() && tr.velocity == vel1 && tr.last_kf.is_some() == has_last_kf1;
    t.add(s.ok && same_end, || {
        format!("장면 {id}: {} 끝 상태 원본 {st1} rust {} lost {tsl1} vs {} vel {vel1} vs {} lastkf {has_last_kf1} vs {}", s.why, tr.state, tr.time_stamp_lost, tr.velocity, tr.last_kf.is_some())
    });
}



fn ctx_with_scale(scale: Vec<f32>) -> std::sync::Arc<FrameCtx> {
    let nl = scale.len() as i32;
    std::sync::Arc::new(FrameCtx { pyr: Pyramid { scale, nlevels: nl, ..Default::default() }, ..Default::default() })
}

/// UNDP: MapPoint::UpdateNormalAndDepth. 원본 관측 순서(주소 순)가 id 순과 다르면 따로 셈
fn undp(r: &mut R, t: &mut Tally, t_addr: &mut Tally) {
    let bad = r.b();
    let pos = [r.f32(), r.f32(), r.f32()];
    let rid = r.u64();
    let _raddr = r.u64();
    let n = r.u32();
    let mut a = Atlas::default();
    let mut m = MapPoint { id: 1, pos, bad, ref_kf: rid, ..Default::default() };
    let mut ids = Vec::new();
    for _ in 0..n {
        let id = r.u64();
        let _ad = r.u64();
        let li = r.i32();
        let c = [r.f32(), r.f32(), r.f32()];
        let oct = r.i32();
        ids.push(id);
        m.obs.insert(id, (li, -1));
        let k = kf_entry(&mut a, id);
        k.twc.t = c;
        if li >= 0 {
            if k.keys.len() <= li as usize {
                k.keys.resize(li as usize + 1, KeyPoint::default());
            }
            k.keys[li as usize].octave = oct;
        }
    }
    if rid != u64::MAX {
        let c = [r.f32(), r.f32(), r.f32()];
        let ns = r.u32();
        let sc: Vec<f32> = (0..ns).map(|_| r.f32()).collect();
        let k = kf_entry(&mut a, rid);
        k.twc.t = c;
        if k.keys.is_empty() {
            k.keys.push(KeyPoint::default());
        }
        k.ctx = ctx_with_scale(sc);
    }
    let want_n = [r.f32(), r.f32(), r.f32()];
    let want_min = r.f32();
    let want_max = r.f32();
    if rid == u64::MAX || bad || n == 0 {
        return;
    }
    let before = (m.normal, m.min_dist, m.max_dist);
    let _ = before;
    a.mps.insert(1, m);
    slam::orb3rs::mapdata::mp_update_normal_and_depth(&mut a, 1);
    let g = &a.mps[&1];
    let bits = |v: &[f32; 3]| v.map(|x| x.to_bits());
    let ok = bits(&g.normal) == bits(&want_n) && g.min_dist.to_bits() == want_min.to_bits() && g.max_dist.to_bits() == want_max.to_bits();
    let sorted = ids.windows(2).all(|w| w[0] < w[1]);
    let tt = if sorted { t } else { t_addr };
    tt.add(ok, || format!("점 기준 {rid}: 법선 원본 {want_n:?} rust {:?} 거리 {want_min} {want_max} vs {} {}", g.normal, g.min_dist, g.max_dist));
}

fn read_conn(r: &mut R) -> (BTreeMap<KfId, i32>, Vec<(KfId, i32)>, bool, Vec<(KfId, bool)>) {
    let n = r.u32();
    let mut w = BTreeMap::new();
    let mut ids = Vec::new();
    for _ in 0..n {
        let id = r.u64();
        let _ad = r.u64();
        let wt = r.i32();
        let bad = r.b();
        w.insert(id, wt);
        ids.push((id, bad));
    }
    let sorted = ids.windows(2).all(|p| p[0].0 < p[1].0);
    let m = r.u32();
    let ord = (0..m).map(|_| (r.u64(), r.i32())).collect();
    (w, ord, sorted, ids)
}

/// UCON: KeyFrame::UpdateConnections
fn ucon(r: &mut R, t: &mut Tally, t_addr: &mut Tally) {
    let id = r.u64();
    let up = r.b();
    let first = r.b();
    let init = r.u64();
    let par0 = r.u64();
    let n = r.u32();
    let mut a = Atlas::default();
    kf_entry(&mut a, id);
    let mut addr_sorted = true;
    let mut slots = Vec::new();
    for i in 0..n {
        let bad = r.b();
        let no = r.u32();
        let mut m = MapPoint { id: i as u64 + 1, bad, ..Default::default() };
        let mut ids = Vec::new();
        for _ in 0..no {
            let k = r.u64();
            let _ad = r.u64();
            let kb = r.b();
            let same = r.b();
            ids.push(k);
            m.obs.insert(k, (0, -1));
            let e = kf_entry(&mut a, k);
            e.bad = kb;
            if !same {
                e.map = 999;
            }
        }
        addr_sorted &= ids.windows(2).all(|w| w[0] < w[1]);
        slots.push(Some(m.id));
        a.mps.insert(m.id, m);
    }
    let no = r.u32();
    let mut others = Vec::new();
    for _ in 0..no {
        let k = r.u64();
        let kb = r.b();
        let (w, ord, srt, ids) = read_conn(r);
        addr_sorted &= srt;
        for (c, cb) in ids {
            kf_entry(&mut a, c).bad = cb;
        }
        let e = kf_entry(&mut a, k);
        e.bad = kb;
        e.conn_weights = w;
        e.ordered_conn = ord;
        others.push(k);
    }
    {
        let e = a.kfs.get_mut(&id).unwrap();
        e.map_points = slots;
        e.parent = if first { None } else { Some(if par0 == u64::MAX { u64::MAX - 1 } else { par0 }) };
    }
    let (w1, ord1, srt1, _) = read_conn(r);
    addr_sorted &= srt1;
    let par1 = r.u64();
    let after: Vec<(BTreeMap<KfId, i32>, Vec<(KfId, i32)>)> = others.iter().map(|_| { let (w, o, s, _) = read_conn(r); addr_sorted &= s; (w, o) }).collect();
    if !up {
        return;
    }
    // 공통 판(mapdata): 지도 mnInitKFid · mbFirstConnection 으로 부모 정함
    a.kfs.get_mut(&id).unwrap().first_conn_done = !first;
    let map = a.kfs[&id].map;
    match a.maps.iter_mut().find(|m| m.id == map) {
        Some(m) => m.init_kf_id = init,
        None => a.maps.push(Map { id: map, init_kf_id: init, ..Default::default() }),
    }
    slam::orb3rs::mapdata::kf_update_connections(&mut a, id, true);
    let k = &a.kfs[&id];
    let mut ok = k.ordered_conn == ord1 && (k.conn_weights == w1 || w1.is_empty() && k.conn_weights.is_empty());
    let gp = k.parent.map(|p| if p == u64::MAX - 1 { u64::MAX } else { p }).unwrap_or(u64::MAX);
    ok &= gp == par1;
    let mut why = String::new();
    for (i, o) in others.iter().enumerate() {
        let ko = &a.kfs[o];
        if ko.ordered_conn != after[i].1 || ko.conn_weights != after[i].0 {
            ok = false;
            why = format!("이웃 {o}: 원본 {:?} rust {:?}", after[i].1, ko.ordered_conn);
        }
    }
    let tt = if addr_sorted { t } else { t_addr };
    tt.add(ok, || format!("키프레임 {id}: 원본 {ord1:?} 부모 {par1} rust {:?} 부모 {gp} {why}", k.ordered_conn));
}

static POSE_DIFF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn read_kfdata(r: &mut R, keys_same: &mut u64) -> KeyFrame {
    let id = r.u64();
    let q = [r.f32(), r.f32(), r.f32(), r.f32()];
    let t = [r.f32(), r.f32(), r.f32()];
    // 원본 KeyFrame 이 들고 있는 mRwc · mTwc.translation()(SetPose 때 계산) — Rust kf_pose 와 비트 비교
    let mut rwc0 = [[0f32; 3]; 3];
    for row in rwc0.iter_mut() {
        for v in row.iter_mut() {
            *v = r.f32();
        }
    }
    let ow0 = [r.f32(), r.f32(), r.f32()];
    let kp = kf_pose(&Se3 { q, t });
    if kp.rwc_stored != rwc0 || kp.ow != ow0 {
        POSE_DIFF.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let _nleft = r.i32();
    let _cam2 = r.b();
    let n = r.u32() as usize;
    let mut k = KeyFrame { id, tcw: Se3 { q, t }, ..Default::default() };
    for _ in 0..n {
        let x = r.f32();
        let y = r.f32();
        let octave = r.i32();
        let (kx, ky) = (r.f32(), r.f32());
        if kx.to_bits() != x.to_bits() || ky.to_bits() != y.to_bits() {
            *keys_same += 1;
        }
        k.keys.push(KeyPoint { x, y, octave, ..Default::default() });
        k.keys_raw.push([kx, ky]);
        k.u_right.push(r.f32());
        k.depth.push(r.f32());
    }
    k
}

fn cnmp(r: &mut R, t: &mut Tally, stats: &mut [u64; 3]) {
    let mono = r.b();
    let inertial = r.b();
    let far = r.b();
    let th_far = r.f32();
    let (fx, fy, cx, cy, invfx, invfy, mb, mbf, sf) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let par = [r.f32(), r.f32(), r.f32(), r.f32()];
    assert!(par == [fx, fy, cx, cy]);
    let nl = r.u32();
    let mut scale = Vec::new();
    let mut sigma2 = Vec::new();
    for _ in 0..nl {
        scale.push(r.f32());
        sigma2.push(r.f32());
    }
    let mut diff_keys = 0;
    let cur = read_kfdata(r, &mut diff_keys);
    let nn = r.u32();
    let mut nb = Vec::new();
    for _ in 0..nn {
        let k = read_kfdata(r, &mut diff_keys);
        let _cam = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        nb.push(k);
    }
    let ntri = r.u32();
    let mut tri: Vec<(u64, Vec<(usize, usize)>)> = Vec::new();
    for _ in 0..ntri {
        let k = r.u64();
        let n = r.u32();
        tri.push((k, (0..n).map(|_| (r.u32() as usize, r.u32() as usize)).collect()));
    }
    let nchk = r.u32();
    let chk: Vec<bool> = (0..nchk).map(|_| r.b()).collect();
    let nadd = r.u32();
    let want: Vec<NewMp> = (0..nadd)
        .map(|_| {
            let pos = [r.f32(), r.f32(), r.f32()];
            let i1 = r.i32();
            let k2 = r.u64();
            let i2 = r.i32();
            NewMp { kf2: k2, idx1: i1 as usize, idx2: i2 as usize, pos }
        })
        .collect();
    let prm = TriParams {
        cam: Camera { fx, fy, cx, cy, invfx, invfy, bf: mbf, b: mb, th_depth: 0.0, width: 0, height: 0 },
        scale,
        sigma2,
        scale_factor: sf,
        monocular: mono,
        inertial,
        far_points: far,
        th_far_points: th_far,
    };
    let refs: Vec<&KeyFrame> = nb.iter().collect();
    let mut ti = 0;
    let mut ci = 0;
    let mut got = Vec::new();
    let mut order_ok = true;
    create_new_map_points(
        &cur,
        &refs,
        &prm,
        &mut |k2| {
            if ti >= tri.len() || tri[ti].0 != k2 {
                order_ok = false;
                return Vec::new();
            }
            ti += 1;
            tri[ti - 1].1.clone()
        },
        &mut || {
            ci += 1;
            chk.get(ci - 1).copied().unwrap_or(false)
        },
        &mut |m| got.push(m),
    );
    stats[0] += want.len() as u64;
    let bits = |p: &[f32; 3]| p.map(|v| v.to_bits());
    let same_pts = got.iter().zip(&want).filter(|(g, w)| g.kf2 == w.kf2 && g.idx1 == w.idx1 && g.idx2 == w.idx2 && bits(&g.pos) == bits(&w.pos)).count();
    stats[1] += same_pts as u64;
    stats[2] += diff_keys;
    let ok = order_ok && ti == tri.len() && got.len() == want.len() && same_pts == want.len();
    t.add(ok, || {
        let d = got.iter().zip(&want).find(|(g, w)| !(g.kf2 == w.kf2 && g.idx1 == w.idx1 && g.idx2 == w.idx2 && bits(&g.pos) == bits(&w.pos)));
        format!("키프레임 {}: 새 점 원본 {} rust {} 같은 {same_pts} 짝 순서 {order_ok} 처음 다른 {:?}", cur.id, want.len(), got.len(), d)
    });
}

fn tria(r: &mut R, t: &mut Tally) {
    let x1 = [r.f32(), r.f32(), r.f32()];
    let x2 = [r.f32(), r.f32(), r.f32()];
    let mut t1 = [[0f32; 4]; 3];
    let mut t2 = [[0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            t1[i][j] = r.f32();
        }
    }
    for i in 0..3 {
        for j in 0..4 {
            t2[i][j] = r.f32();
        }
    }
    let ok = r.b();
    let x = [r.f32(), r.f32(), r.f32()];
    let got = triangulate(&x1, &x2, &t1, &t2);
    let same = match got {
        Some(g) => ok && g.map(|v| v.to_bits()) == x.map(|v| v.to_bits()),
        None => !ok,
    };
    t.add(same, || format!("원본 {ok} {x:?} rust {got:?}"));
}

// ---------------- Tracking 단계(CTXF · CRLF · TRKF · TWMM · SLPT · TLMP · SINI) ----------------

thread_local! {
    static CTX: std::cell::RefCell<Option<std::sync::Arc<FrameCtx>>> = const { std::cell::RefCell::new(None) };
}

/// CTXF: 장면 공통 값(카메라 · 층 표 · 경계 · 격자) + 지역 지도 mbFarPoints · mThFarPoints
fn ctxf(r: &mut R, far: &mut (bool, f32)) {
    let (fx, fy, cx, cy, invfx, invfy, bf, b, th_depth) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    let nl = r.i32();
    let (sf, lsf) = (r.f32(), r.f32());
    let mut pyr = Pyramid { nlevels: nl, scale_factor: sf, log_scale_factor: lsf, ..Default::default() };
    for _ in 0..nl {
        pyr.scale.push(r.f32());
        pyr.inv_scale.push(r.f32());
        pyr.sigma2.push(r.f32());
        pyr.inv_sigma2.push(r.f32());
    }
    let (min_x, max_x, min_y, max_y, gw, gh) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
    *far = (r.b(), r.f32());
    let cam = Camera { fx, fy, cx, cy, invfx, invfy, bf, b, th_depth, width: max_x as i32, height: max_y as i32 };
    let c = FrameCtx { cam, pyr, min_x, max_x, min_y, max_y, grid_inv_w: gw, grid_inv_h: gh };
    CTX.with(|x| *x.borrow_mut() = Some(std::sync::Arc::new(c)));
}
fn ctx() -> std::sync::Arc<FrameCtx> {
    CTX.with(|x| x.borrow().clone().expect("CTXF 먼저"))
}
fn opt_id(v: u64) -> Option<u64> {
    if v == u64::MAX { None } else { Some(v) }
}
fn read_pose(r: &mut R) -> Se3 {
    Se3 { q: [r.f32(), r.f32(), r.f32(), r.f32()], t: [r.f32(), r.f32(), r.f32()] }
}
fn read_desc(r: &mut R) -> Desc {
    r.take::<32>()
}
fn read_feat(r: &mut R) -> BTreeMap<u32, Vec<u32>> {
    let n = r.u32();
    (0..n)
        .map(|_| {
            let id = r.u32();
            let k = r.u32();
            (id, (0..k).map(|_| r.u32()).collect())
        })
        .collect()
}
fn read_frame(r: &mut R) -> Frame {
    let id = r.u64();
    let ts = r.f64();
    let pose = if r.b() { Some(read_pose(r)) } else { None };
    let n = r.u32() as usize;
    let mut f = Frame { id, timestamp: ts, ctx: ctx(), ..Default::default() };
    for _ in 0..n {
        f.keys.push(KeyPoint { x: r.f32(), y: r.f32(), size: r.f32(), angle: r.f32(), response: r.f32(), octave: r.i32(), class_id: r.i32() });
        f.desc.push(read_desc(r));
        f.u_right.push(r.f32());
        f.depth.push(r.f32());
        f.map_points.push(opt_id(r.u64()));
        f.outlier.push(r.b());
    }
    f.ref_kf = opt_id(r.u64());
    if r.b() {
        f.feat = read_feat(r);
    }
    f.assign_grid();
    if let Some(p) = pose {
        f.set_pose(p);
    }
    f
}
/// 장면 결과(자세 있음 · 자세 · 점 · 바깥점)
type FrameOut = (Option<[u32; 7]>, Vec<(Option<u64>, bool)>);
fn read_frame_out(r: &mut R, n: usize) -> FrameOut {
    let p = if r.b() { Some(pose_bits(&read_pose(r))) } else { None };
    (p, (0..n).map(|_| (opt_id(r.u64()), r.b())).collect())
}
fn pose_bits(p: &Se3) -> [u32; 7] {
    [p.q[0], p.q[1], p.q[2], p.q[3], p.t[0], p.t[1], p.t[2]].map(|v| v.to_bits())
}
fn frame_out(f: &Frame) -> FrameOut {
    (f.tcw.as_ref().map(pose_bits), f.map_points.iter().copied().zip(f.outlier.iter().copied()).collect())
}
fn read_mp(r: &mut R) -> MapPoint {
    let id = r.u64();
    let bad = r.b();
    let pos = [r.f32(), r.f32(), r.f32()];
    let normal = [r.f32(), r.f32(), r.f32()];
    let desc = read_desc(r);
    let n_obs = r.i32();
    let (min_dist, max_dist) = (r.f32(), r.f32());
    let replaced = opt_id(r.u64());
    let (visible, found) = (r.i32(), r.i32());
    let last_frame_seen = r.u64();
    let track_in_view = r.b();
    let (track_proj_x, track_proj_y, track_proj_xr, track_scale_level, track_view_cos, track_depth) = (r.f32(), r.f32(), r.f32(), r.i32(), r.f32(), r.f32());
    MapPoint {
        id,
        bad,
        pos,
        normal,
        desc,
        n_obs,
        min_dist,
        max_dist,
        replaced,
        visible,
        found,
        last_frame_seen,
        track_in_view,
        track_proj_x,
        track_proj_y,
        track_proj_xr,
        track_scale_level,
        track_view_cos,
        track_depth,
        ..Default::default()
    }
}
fn read_mps(r: &mut R) -> BTreeMap<MpId, MapPoint> {
    let n = r.u32();
    (0..n).map(|_| read_mp(r)).map(|m| (m.id, m)).collect()
}
/// 지도 점 결과 칸: (보임, 찾음, 마지막 본 장면, 시야 칸 비트)
type MpOut = Vec<(u64, i32, i32, u64, Option<[u32; 6]>)>;
fn read_mp_out(r: &mut R) -> MpOut {
    let n = r.u32();
    (0..n)
        .map(|_| {
            let id = r.u64();
            let (v, f) = (r.i32(), r.i32());
            let ls = r.u64();
            let tv = if r.b() { Some([r.f32().to_bits(), r.f32().to_bits(), r.f32().to_bits(), r.i32() as u32, r.f32().to_bits(), r.f32().to_bits()]) } else { None };
            (id, v, f, ls, tv)
        })
        .collect()
}
fn mp_out(want: &MpOut, mps: &BTreeMap<MpId, MapPoint>) -> MpOut {
    want.iter()
        .map(|w| {
            let m = &mps[&w.0];
            let tv = m.track_in_view.then(|| {
                [m.track_proj_x.to_bits(), m.track_proj_y.to_bits(), m.track_proj_xr.to_bits(), m.track_scale_level as u32, m.track_view_cos.to_bits(), m.track_depth.to_bits()]
            });
            (m.id, m.visible, m.found, m.last_frame_seen, tv)
        })
        .collect()
}
fn diff_frame(a: &FrameOut, b: &FrameOut) -> String {
    if a.0 != b.0 {
        return format!("자세 원본 {:?} rust {:?}", a.0, b.0);
    }
    let k = a.1.iter().zip(&b.1).position(|(x, y)| x != y);
    format!("점 {k:?}: 원본 {:?} rust {:?}", k.map(|i| a.1[i]), k.map(|i| b.1[i]))
}
fn diff_mp(a: &MpOut, b: &MpOut) -> String {
    let k = a.iter().zip(b).position(|(x, y)| x != y);
    format!("지도 점 {k:?}: 원본 {:?} rust {:?}", k.map(|i| a[i]), k.map(|i| b[i]))
}

/// CRLF: CheckReplacedInLastFrame
fn crlf(r: &mut R, t: &mut Tally) {
    let n = r.u32() as usize;
    let mut last = Frame::default();
    let mut mps = BTreeMap::new();
    for _ in 0..n {
        let (m, rep) = (opt_id(r.u64()), opt_id(r.u64()));
        last.map_points.push(m);
        if let Some(m) = m {
            mps.insert(m, MapPoint { id: m, replaced: rep, ..Default::default() });
        }
    }
    let want: Vec<Option<u64>> = (0..n).map(|_| opt_id(r.u64())).collect();
    check_replaced_in_last_frame(&mut last, &mps);
    t.add(last.map_points == want, || "점 다름".into());
}

/// TRKF: TrackReferenceKeyFrame
fn trkf(r: &mut R, t: &mut Tally, tmp: &mut Tally) {
    let mut cur = read_frame(r);
    let _last_set = r.b();
    let last_pose = read_pose(r);
    let kid = r.u64();
    let n = r.u32() as usize;
    let mut kf = KeyFrame { id: kid, ctx: ctx(), ..Default::default() };
    for _ in 0..n {
        kf.keys.push(KeyPoint { x: r.f32(), y: r.f32(), angle: r.f32(), octave: r.i32(), ..Default::default() });
        kf.desc.push(read_desc(r));
        kf.u_right.push(r.f32());
        kf.map_points.push(opt_id(r.u64()));
    }
    kf.feat = read_feat(r);
    let mut mps = read_mps(r);
    let want_r = r.b();
    let want = read_frame_out(r, cur.keys.len());
    let want_mp = read_mp_out(r);
    let got_r = track_reference_keyframe(&mut cur, last_pose, &kf, &mut mps);
    let got = frame_out(&cur);
    t.add(got_r == want_r && (got == want || !want_r && got.1 == want.1 && want.0.is_none() == got.0.is_none()), || {
        format!("장면 {} 결과 원본 {want_r} rust {got_r}; {}", cur.id, diff_frame(&want, &got))
    });
    let gm = mp_out(&want_mp, &mps);
    tmp.add(gm == want_mp, || diff_mp(&want_mp, &gm));
}

/// TWMM: UpdateLastFrame + TrackWithMotionModel
fn twmm(r: &mut R, t: &mut Tally, tmp: &mut Tally) {
    let sensor = r.i32();
    let mut cur = read_frame(r);
    let mut last = read_frame(r);
    let ref_pose = read_pose(r);
    let tlr = read_pose(r);
    let vel = read_pose(r);
    let mut mps = read_mps(r);
    let want_r = r.b();
    let _ = r.b();
    let want_last = pose_bits(&read_pose(r));
    let want = read_frame_out(r, cur.keys.len());
    let want_mp = read_mp_out(r);
    let got_r = track_with_motion_model(&mut cur, &mut last, ref_pose, tlr, vel, sensor, &mut mps);
    let got = frame_out(&cur);
    let got_last = pose_bits(last.tcw.as_ref().unwrap());
    t.add(got_r == want_r && got == want && got_last == want_last, || {
        format!("장면 {} 결과 원본 {want_r} rust {got_r}; 앞 자세 같음 {}; {}", cur.id, got_last == want_last, diff_frame(&want, &got))
    });
    let gm = mp_out(&want_mp, &mps);
    tmp.add(gm == want_mp, || diff_mp(&want_mp, &gm));
}

/// SLPT: SearchLocalPoints
fn slpt(r: &mut R, far: (bool, f32), t: &mut Tally, tmp: &mut Tally) {
    let sensor = r.i32();
    let state = r.i32();
    let last_reloc = r.u64();
    let imu = r.b();
    assert!(!imu);
    let mut cur = read_frame(r);
    let n = r.u32();
    let local: Vec<MpId> = (0..n).map(|_| r.u64()).collect();
    let mut mps = read_mps(r);
    let want = read_frame_out(r, cur.keys.len());
    let want_mp = read_mp_out(r);
    SLP_FRAME.with(|s| *s.borrow_mut() = Some(cur.clone()));
    let tk = Tracker { sensor, state, last_reloc_frame_id: last_reloc, local_mps: local, ..Default::default() };
    search_local_points(&tk, &mut cur, &mut mps, far);
    let got = frame_out(&cur);
    t.add(got == want, || format!("장면 {}: {}", cur.id, diff_frame(&want, &got)));
    let gm = mp_out(&want_mp, &mps);
    tmp.add(gm == want_mp, || diff_mp(&want_mp, &gm));
}

/// TLMP: TrackLocalMap 의 SearchLocalPoints 뒤(PoseOptimization · 점 통계 · 판단)
fn tlmp(r: &mut R, t: &mut Tally, tmp: &mut Tally) {
    let sensor = r.i32();
    let state = r.i32();
    let last_reloc = r.u64();
    let max_frames = r.i32();
    let only = r.b();
    let id = r.u64();
    let len = r.u32() as usize;
    let after = r.b[r.o..r.o + len].to_vec();
    r.o += len;
    let mut mps = read_mps(r);
    let want_r = r.b();
    let want_inl = r.i32();
    // 원본 SearchLocalPoints 뒤 장면 + 점들의 결과 전 찾음 수
    let mut ra = R { b: &after, o: 0 };
    let n = (after.len() - 1 - if after[0] != 0 { 28 } else { 0 }) / 9;
    let a = read_frame_out(&mut ra, n);
    let want = read_frame_out(r, n);
    let want_mp = read_mp_out(r);
    let mut cur = Frame { id, ctx: ctx(), ..Default::default() };
    cur.keys = vec![KeyPoint::default(); n];
    cur.u_right = vec![-1.0; n];
    cur.map_points = a.1.iter().map(|x| x.0).collect();
    cur.outlier = a.1.iter().map(|x| x.1).collect();
    // 자세 최적화에 필요한 특징점 · 오른쪽 좌표가 이 기록에 없음 → SLPT 기록과 같은 장면에서 채움
    TLM_PEND.with(|p| p.borrow_mut().push((id, cur, mps, a.0, want, want_mp, want_r, want_inl, Tracker { sensor, state, last_reloc_frame_id: last_reloc, max_frames, only_tracking: only, ..Default::default() })));
    tlmp_run(t, tmp);
}

type TlmPend = (u64, Frame, BTreeMap<MpId, MapPoint>, Option<[u32; 7]>, FrameOut, MpOut, bool, i32, Tracker);
thread_local! {
    static TLM_PEND: std::cell::RefCell<Vec<TlmPend>> = const { std::cell::RefCell::new(Vec::new()) };
    static SLP_FRAME: std::cell::RefCell<Option<Frame>> = const { std::cell::RefCell::new(None) };
}
/// 같은 장면 SLPT 의 특징점으로 TLMP 를 돌림(SLPT 가 TLMP 보다 먼저 기록됨)
fn tlmp_run(t: &mut Tally, tmp: &mut Tally) {
    let pend = TLM_PEND.with(|p| std::mem::take(&mut *p.borrow_mut()));
    for (id, mut cur, mut mps, pose_after_slp, want, want_mp, want_r, want_inl, mut tk) in pend {
        let Some(sf) = SLP_FRAME.with(|s| s.borrow_mut().take()).filter(|f| f.id == id) else {
            t.add(false, || format!("장면 {id}: SLPT 기록 없음"));
            continue;
        };
        cur.keys = sf.keys;
        cur.u_right = sf.u_right;
        cur.depth = sf.depth;
        let p = pose_after_slp.expect("자세");
        let f = |i: usize| f32::from_bits(p[i]);
        cur.set_pose(Se3 { q: [f(0), f(1), f(2), f(3)], t: [f(4), f(5), f(6)] });
        slam::orb3loop::reloc::pose_optimization_frame(&mut cur, &mps);
        let got_r = track_local_map_finish(&mut tk, &mut cur, &mut mps);
        let got = frame_out(&cur);
        t.add(got_r == want_r && tk.matches_inliers == want_inl && got == want, || {
            format!("장면 {id} 결과 원본 {want_r}/{want_inl} rust {got_r}/{}; {}", tk.matches_inliers, diff_frame(&want, &got))
        });
        // 점 칸은 원본 실행 뒤 값을 넣어 돌리므로(찾음 수가 이미 올라감) 비교는 바깥점 · 맞은 수로만
        let _ = (&want_mp, &mut *tmp);
    }
}

/// SINI: StereoInitialization
fn sini(r: &mut R, t: &mut Tally, tpts: &mut Tally) {
    let mut cur = read_frame(r);
    let want_state = r.i32();
    let want = read_frame_out(r, cur.keys.len());
    let n = r.u32();
    let made: Vec<(i32, MapPoint)> = (0..n).map(|_| (r.i32(), read_mp(r))).collect();
    let nl = r.u32();
    let _local: Vec<u64> = (0..nl).map(|_| r.u64()).collect();
    let mut atlas = Atlas { maps: vec![Map::default()], ..Default::default() };
    let mut tk = Tracker { state: state::NOT_INITIALIZED, ..Default::default() };
    let kf = stereo_initialization(&mut tk, &mut cur, &mut atlas);
    let got_state = if kf.is_some() { state::OK } else { state::NOT_INITIALIZED };
    let same_slots = cur.map_points.iter().map(|m| m.is_some()).eq(want.1.iter().map(|x| x.0.is_some()));
    t.add(got_state == want_state && same_slots && cur.tcw.as_ref().map(pose_bits) == want.0, || {
        format!("장면 {} 상태 원본 {want_state} rust {got_state}, 칸 같음 {same_slots}", cur.id)
    });
    for (i, w) in &made {
        let Some(g) = cur.map_points[*i as usize].map(|m| &atlas.mps[&m]) else {
            tpts.add(false, || format!("칸 {i} 없음"));
            continue;
        };
        let bits = |m: &MapPoint| (m.pos.map(f32::to_bits), m.normal.map(f32::to_bits), m.desc, m.n_obs, m.min_dist.to_bits(), m.max_dist.to_bits());
        tpts.add(bits(g) == bits(w), || format!("칸 {i}: 원본 {:?} rust {:?}", bits(w), bits(g)));
    }
}

// ---------------- LocalMapping 단계 통째(PNKF · SINB) ----------------

fn map_id(v: i64) -> MapId {
    v as u64
}
fn read_kf_full(r: &mut R) -> KeyFrame {
    let id = r.u64();
    let frame_id = r.u64();
    let map = map_id(r.i64());
    let timestamp = r.f64();
    let bad = r.b();
    let not_erase = r.b();
    let first_conn_done = !r.b();
    let tcw = read_pose(r);
    let twc = read_pose(r);
    let n = r.u32() as usize;
    let mut k = KeyFrame { id, frame_id, map, timestamp, bad, not_erase, first_conn_done, tcw, twc, ctx: ctx(), ..Default::default() };
    for _ in 0..n {
        k.keys.push(KeyPoint { x: r.f32(), y: r.f32(), size: r.f32(), angle: r.f32(), response: r.f32(), octave: r.i32(), class_id: -1 });
        k.keys_raw.push([r.f32(), r.f32()]);
        k.desc.push(read_desc(r));
        k.u_right.push(r.f32());
        k.depth.push(r.f32());
        k.map_points.push(opt_id(r.u64()));
    }
    let nb = r.u32();
    k.bow = (0..nb).map(|_| (r.u32(), r.f64())).collect();
    k.feat = read_feat(r);
    let nc = r.u32();
    k.conn_weights = (0..nc).map(|_| (r.u64(), r.i32())).collect();
    let no = r.u32();
    k.ordered_conn = (0..no).map(|_| (r.u64(), r.i32())).collect();
    k.parent = opt_id(r.u64());
    let nch = r.u32();
    k.children = (0..nch).map(|_| r.u64()).collect();
    let mut f = Frame { keys: k.keys.clone(), ctx: ctx(), ..Default::default() };
    f.assign_grid();
    k.grid = f.grid;
    k
}
fn read_mp_full(r: &mut R) -> MapPoint {
    let id = r.u64();
    let map = map_id(r.i64());
    let bad = r.b();
    let pos = [r.f32(), r.f32(), r.f32()];
    let normal = [r.f32(), r.f32(), r.f32()];
    let desc = read_desc(r);
    let n_obs = r.i32();
    let (min_dist, max_dist) = (r.f32(), r.f32());
    let ref_kf = r.u64();
    let first_kf = r.u64();
    let first_frame = r.u64();
    let replaced = opt_id(r.u64());
    let (visible, found) = (r.i32(), r.i32());
    let no = r.u32();
    let obs = (0..no).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
    MapPoint { id, map, bad, pos, normal, desc, n_obs, min_dist, max_dist, ref_kf, first_kf, first_frame, replaced, visible, found, obs, ..Default::default() }
}
/// 지도 조각 → Atlas. 돌려줌: (통째 키프레임 번호들, 점 번호들)
fn read_region(r: &mut R, a: &mut Atlas) -> (Vec<KfId>, Vec<MpId>) {
    let nk = r.u32();
    let mut ks = Vec::new();
    for _ in 0..nk {
        let k = read_kf_full(r);
        ks.push(k.id);
        a.kfs.insert(k.id, k);
    }
    let nm = r.u32();
    let mut ms = Vec::new();
    for _ in 0..nm {
        let m = read_mp_full(r);
        ms.push(m.id);
        a.mps.insert(m.id, m);
    }
    let ns = r.u32();
    for _ in 0..ns {
        let id = r.u64();
        let bad = r.b();
        let map = map_id(r.i64());
        a.kfs.insert(id, KeyFrame { id, bad, map, ctx: ctx(), ..Default::default() });
    }
    (ks, ms)
}
type KfCmp = (Vec<Option<u64>>, Vec<(u64, i32)>, Vec<(u64, i32)>, Option<u64>, Vec<u64>, bool);
fn kf_cmp(k: &KeyFrame) -> KfCmp {
    let mut ch = k.children.clone();
    ch.sort_unstable();
    (k.map_points.clone(), k.conn_weights.iter().map(|(a, b)| (*a, *b)).collect(), k.ordered_conn.clone(), k.parent, ch, k.bad)
}
type MpCmp = (bool, [u32; 3], [u32; 3], Desc, i32, u32, u32, u64, Option<u64>, i32, i32, Vec<(u64, (i32, i32))>);
fn mp_cmp(m: &MapPoint) -> MpCmp {
    (
        m.bad,
        m.pos.map(f32::to_bits),
        m.normal.map(f32::to_bits),
        m.desc,
        m.n_obs,
        m.min_dist.to_bits(),
        m.max_dist.to_bits(),
        m.ref_kf,
        m.replaced,
        m.visible,
        m.found,
        m.obs.iter().map(|(a, b)| (*a, *b)).collect(),
    )
}
/// 단계 뒤 조각을 원본과 비교: (키프레임 같음, 점 같음, 첫 차이)
fn cmp_region(r: &mut R, a: &Atlas, ks: &[KfId], ms: &[MpId]) -> (bool, bool, String) {
    let mut why = String::new();
    let mut kok = true;
    for &k in ks {
        let w = read_kf_full(r);
        let (x, y) = (kf_cmp(&w), kf_cmp(&a.kfs[&k]));
        if x != y && kok {
            kok = false;
            let f = |i: usize| match i {
                0 => format!("점 {:?}", x.0.iter().zip(&y.0).enumerate().find(|(_, (p, q))| p != q)),
                1 => format!("무게 {:?} / {:?}", x.1, y.1),
                2 => format!("순서 {:?} / {:?}", x.2, y.2),
                _ => format!("부모 {:?}/{:?} 자식 {:?}/{:?} 나쁨 {}/{}", x.3, y.3, x.4, y.4, x.5, y.5),
            };
            let i = if x.0 != y.0 { 0 } else if x.1 != y.1 { 1 } else if x.2 != y.2 { 2 } else { 3 };
            why += &format!("키프레임 {k}: {} ", f(i));
        }
    }
    let mut mok = true;
    for &m in ms {
        let w = read_mp_full(r);
        let (x, y) = (mp_cmp(&w), mp_cmp(&a.mps[&m]));
        if x != y && mok {
            mok = false;
            why += &format!("점 {m}: 원본 {x:?} rust {y:?}");
        }
    }
    (kok, mok, why)
}
fn read_map(r: &mut R) -> (MapId, KfId, bool) {
    (map_id(r.i64()), r.u64(), r.b())
}

/// PNKF: ProcessNewKeyFrame 통째
fn pnkf(r: &mut R, tk: &mut Tally, tm: &mut Tally, tr: &mut Tally) {
    let kf = r.u64();
    let (mid_, init, had) = read_map(r);
    let mut a = Atlas::default();
    let (ks, ms) = read_region(r, &mut a);
    a.maps.push(Map { id: mid_, init_kf_id: init, kfs: if had { vec![kf] } else { vec![] }, ..Default::default() });
    let mut recent = Vec::new();
    process_new_keyframe(&mut a, kf, &mut recent);
    let (_, _, has) = read_map(r);
    let (kok, mok, why) = cmp_region(r, &a, &ks, &ms);
    let n = r.u32();
    let want: Vec<u64> = (0..n).map(|_| r.u64()).collect();
    tk.add(kok && has == a.maps[0].kfs.contains(&kf), || format!("키프레임 {kf}: {why}"));
    tm.add(mok, || format!("키프레임 {kf}: {why}"));
    tr.add(want == recent, || format!("키프레임 {kf}: 최근 점 원본 {} rust {}", want.len(), recent.len()));
}

/// SINB: SearchInNeighbors 통째. 원본 도중 mbAbortBA 가 서면 따로 셈
fn sinb(r: &mut R, tk: &mut Tally, tm: &mut Tally, racy: &mut Tally) {
    let kf = r.u64();
    let ab0 = r.b();
    let mut a = Atlas::default();
    let (ks, ms) = read_region(r, &mut a);
    let map = a.kfs[&kf].map;
    a.maps.push(Map { id: map, ..Default::default() });
    let ab1 = r.b();
    search_in_neighbors(&mut a, kf, &|| ab1);
    let (kok, mok, why) = cmp_region(r, &a, &ks, &ms);
    if ab0 || ab1 {
        racy.add(kok && mok, || format!("키프레임 {kf}: {why}"));
    } else {
        tk.add(kok, || format!("키프레임 {kf}: {why}"));
        tm.add(mok, || format!("키프레임 {kf}: {why}"));
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("dump.bin");
    let d = std::fs::read(&path).unwrap();
    let mut o = 0;
    let mut tal: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut cn_stats = [0u64; 2];
    let mut cnmp_stats = [0u64; 3];
    let mut states = BTreeMap::new();
    let mut lost_frames = 0u64;
    let mut frames = 0u64;
    let mut far = (false, 0f32);
    while o + 8 <= d.len() {
        let tag = std::str::from_utf8(&d[o..o + 4]).unwrap().to_string();
        let n = u32::from_le_bytes(d[o + 4..o + 8].try_into().unwrap()) as usize;
        let body = &d[o + 8..o + 8 + n];
        o += 8 + n;
        let mut r = R { b: body, o: 0 };
        match tag.as_str() {
            "NNKF" => {
                let mut racy = std::mem::take(tal.entry("NeedNewKeyFrame(도중 입력 바뀜 · 참고)").or_default());
                nnkf(&mut r, tal.entry("NeedNewKeyFrame").or_default(), &mut racy);
                *tal.entry("NeedNewKeyFrame(도중 입력 바뀜 · 참고)").or_default() = racy;
            }
            "ULKF" => {
                let mut a = std::mem::take(tal.entry("UpdateLocalKeyFrames").or_default());
                let mut b = std::mem::take(tal.entry("UpdateLocalKeyFrames 기준 KF").or_default());
                ulkf(&mut r, &mut a, &mut b);
                tal.insert("UpdateLocalKeyFrames", a);
                tal.insert("UpdateLocalKeyFrames 기준 KF", b);
            }
            "ULMP" => ulmp(&mut r, tal.entry("UpdateLocalPoints").or_default()),
            "CNKF" => {
                let mut a = std::mem::take(tal.entry("UpdatePoseMatrices(Rwc·Ow)").or_default());
                let mut b = std::mem::take(tal.entry("CreateNewKeyFrame 새 점").or_default());
                cnkf(&mut r, &mut a, &mut b, &mut cn_stats);
                tal.insert("UpdatePoseMatrices(Rwc·Ow)", a);
                tal.insert("CreateNewKeyFrame 새 점", b);
            }
            "MPCL" => {
                let mut a = std::mem::take(tal.entry("MapPointCulling 남김").or_default());
                let mut b = std::mem::take(tal.entry("MapPointCulling 나쁨").or_default());
                mpcl(&mut r, &mut a, &mut b);
                tal.insert("MapPointCulling 남김", a);
                tal.insert("MapPointCulling 나쁨", b);
            }
            "CNMP" => cnmp(&mut r, tal.entry("CreateNewMapPoints").or_default(), &mut cnmp_stats),
            "UNDP" => {
                let mut b = std::mem::take(tal.entry("UpdateNormalAndDepth(주소 순 ≠ id 순)").or_default());
                undp(&mut r, tal.entry("UpdateNormalAndDepth").or_default(), &mut b);
                *tal.entry("UpdateNormalAndDepth(주소 순 ≠ id 순)").or_default() = b;
            }
            "UCON" => {
                let mut b = std::mem::take(tal.entry("UpdateConnections(주소 순 ≠ id 순)").or_default());
                ucon(&mut r, tal.entry("UpdateConnections").or_default(), &mut b);
                *tal.entry("UpdateConnections(주소 순 ≠ id 순)").or_default() = b;
            }
            "TRIA" => tria(&mut r, tal.entry("Triangulate(JacobiSVD)").or_default()),
            "KFCL" => kfcl(&mut r, tal.entry("KeyFrameCulling").or_default()),
            "TRAK" => trak(&mut r, tal.entry("Track 흐름").or_default(), &mut states),
            "CTXF" => ctxf(&mut r, &mut far),
            "CRLF" => crlf(&mut r, tal.entry("CheckReplacedInLastFrame").or_default()),
            "TRKF" => {
                let mut b = std::mem::take(tal.entry("TrackReferenceKeyFrame 점 칸").or_default());
                trkf(&mut r, tal.entry("TrackReferenceKeyFrame").or_default(), &mut b);
                tal.insert("TrackReferenceKeyFrame 점 칸", b);
            }
            "TWMM" => {
                let mut b = std::mem::take(tal.entry("TrackWithMotionModel 점 칸").or_default());
                twmm(&mut r, tal.entry("TrackWithMotionModel").or_default(), &mut b);
                tal.insert("TrackWithMotionModel 점 칸", b);
            }
            "SLPT" => {
                let mut b = std::mem::take(tal.entry("SearchLocalPoints 점 칸").or_default());
                slpt(&mut r, far, tal.entry("SearchLocalPoints").or_default(), &mut b);
                tal.insert("SearchLocalPoints 점 칸", b);
            }
            "TLMP" => {
                let mut b = Tally::default();
                tlmp(&mut r, tal.entry("TrackLocalMap(자세 최적화 · 판단)").or_default(), &mut b);
            }
            "SINI" => {
                let mut b = std::mem::take(tal.entry("StereoInitialization 새 점").or_default());
                sini(&mut r, tal.entry("StereoInitialization").or_default(), &mut b);
                tal.insert("StereoInitialization 새 점", b);
            }
            "PNKF" => {
                let mut a = std::mem::take(tal.entry("ProcessNewKeyFrame 키프레임").or_default());
                let mut b = std::mem::take(tal.entry("ProcessNewKeyFrame 점").or_default());
                let mut c = std::mem::take(tal.entry("ProcessNewKeyFrame 최근 점").or_default());
                pnkf(&mut r, &mut a, &mut b, &mut c);
                tal.insert("ProcessNewKeyFrame 키프레임", a);
                tal.insert("ProcessNewKeyFrame 점", b);
                tal.insert("ProcessNewKeyFrame 최근 점", c);
            }
            "SINB" => {
                let mut a = std::mem::take(tal.entry("SearchInNeighbors 키프레임").or_default());
                let mut b = std::mem::take(tal.entry("SearchInNeighbors 점").or_default());
                let mut c = std::mem::take(tal.entry("SearchInNeighbors(도중 중단 깃발 · 참고)").or_default());
                sinb(&mut r, &mut a, &mut b, &mut c);
                tal.insert("SearchInNeighbors 키프레임", a);
                tal.insert("SearchInNeighbors 점", b);
                tal.insert("SearchInNeighbors(도중 중단 깃발 · 참고)", c);
            }
            "FRAM" => {
                frames += 1;
                let _ts = r.f64();
                if r.i32() != state::OK {
                    lost_frames += 1;
                }
            }
            _ => {}
        }
        assert!(tag == "FRAM" || r.o == r.b.len(), "{tag} 길이 어긋남 {} vs {}", r.o, r.b.len());
    }
    println!("{path}: 장면 {frames}, OK 아닌 장면 {lost_frames}, 끝 상태 {:?}", states);
    println!("| 함수 | 비교 | 비트 같음 |");
    for (k, v) in &tal {
        println!("| {k} | {} | {} |", v.n, v.same);
    }
    println!("새 지도 점 {} 개 중 위치 비트 같음 {}", cn_stats[0], cn_stats[1]);
    println!("삼각측량 새 점 {} 개 중 짝 · 위치 비트 같음 {} (mvKeys≠mvKeysUn {})", cnmp_stats[0], cnmp_stats[1], cnmp_stats[2]);
    println!("키프레임 mRwc · Ow 가 Rust kf_pose 와 다른 경우 {}", POSE_DIFF.load(std::sync::atomic::Ordering::Relaxed));
    for (k, v) in &tal {
        for e in &v.examples {
            println!("  [{k}] {e}");
        }
    }
}
