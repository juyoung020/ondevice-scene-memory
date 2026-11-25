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
        t.add(got == want, || format!("장면 {id}: 원본 {want} rust {got}"));
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
    let order = |k: KfId| *addr.get(&k).unwrap_or(&u64::MAX);
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
    fn create_new_keyframe(&mut self) -> KfId {
        self.next(9, None);
        0
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
            "TRIA" => tria(&mut r, tal.entry("Triangulate(JacobiSVD)").or_default()),
            "KFCL" => kfcl(&mut r, tal.entry("KeyFrameCulling").or_default()),
            "TRAK" => trak(&mut r, tal.entry("Track 흐름").or_default(), &mut states),
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
