//! 전역 BA 결과 넣기(`orb3sys::gbaops::apply_gba`) 원본 대조. 원본 = 번호 순 비교자 기준 빌드(`~/opt/orb3_refid`)를 돌린
//! `~/opt/orb3_sys/gba/gba_dump`(LoopClosing::RunGlobalBundleAdjustment: GlobalBundleAdjustemnt 직후 지도 · mTcwGBA · mPosGBA ·
//! mnBAGlobalForKF · 첫 키프레임(GBAB), 결과 넣은 뒤 Release 직전 지도(GBAA)).
//! ORBSYS_GBA=덤프1:덤프2 cargo test --release -p slam --test orb3sys_gba_apply
use slam::orb3loop::correct::*;
use slam::orb3loop::detect::G2oSim3;
use slam::orb3rs::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;

use slam::orb3sys::gbaops::apply_gba;
use slam::orb3sys::mapping::MapState;
use slam::orb3ba::gba::GbaOut;

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
    map_id: u64,
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
    DMap { map_id, kfs, mps, not_erase, corrected_by }
}


#[test]
fn gba_apply_matches_original() {
    let dumps = std::env::var("ORBSYS_GBA").expect("입력 없음: ORBSYS_GBA(gba_dump 덤프, : 로 여럿)");
    let (mut n, mut same) = (0, 0);
    for path in dumps.split(':') {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
        assert!(f.metadata().unwrap().len() > 0, "빈 덤프 {path}");
        let mut r = Rd { f: std::io::BufReader::with_capacity(1 << 20, f) };
        let mut pending: Option<(MapState, GbaOut, u64, u64)> = None;
        while r.more() {
            let magic = r.u32();
            if magic == 0x42414247 {
                let loop_kf = r.u64();
                let _stop = r.u8();
                let d = read_map(&mut r);
                let pos_map = d.map_id;
                let nk = r.u32();
                let mut kf_gba = Vec::new();
                for _ in 0..nk {
                    let id = r.u64();
                    let ba = r.u64();
                    let t = r.se3();
                    if ba == loop_kf {
                        kf_gba.push((id, t));
                    }
                }
                let nm = r.u32();
                let mut mp_gba = Vec::new();
                for _ in 0..nm {
                    let id = r.u64();
                    let ba = r.u64();
                    let p = [r.f32(), r.f32(), r.f32()];
                    if ba == loop_kf {
                        mp_gba.push((id, p));
                    }
                }
                let origins = r.ids();
                let mut st = MapState::default();
                let mut map = Map { id: pos_map, origin_kf: origins.first().copied(), ..Default::default() };
                for (id, k) in &d.kfs {
                    if k.map == pos_map && !k.bad {
                        map.kfs.push(*id);
                    }
                }
                for (id, p) in &d.mps {
                    if p.map == pos_map && !p.bad {
                        map.mps.push(*id);
                    }
                }
                st.atlas.kfs = d.kfs;
                st.atlas.mps = d.mps;
                st.atlas.maps.push(map);
                pending = Some((st, GbaOut { direct: false, iterations: 0, kf_gba, mp_gba }, pos_map, loop_kf));
            } else if magic == 0x41414247 {
                let loop_kf = r.u64();
                let d = read_map(&mut r);
                let Some((mut st, out, map_id, lk)) = pending.take() else { panic!("GBAA 앞에 GBAB 없음") };
                assert_eq!(lk, loop_kf);
                apply_gba(&mut st, &out, map_id);
                n += 1;
                let mut bad = Vec::new();
                for (id, k) in &d.kfs {
                    if k.map != map_id {
                        continue;
                    }
                    match st.atlas.kfs.get(id) {
                        Some(m) if bse3(&m.tcw, &k.tcw) && bse3(&m.twc, &k.twc) => {}
                        Some(m) => bad.push(format!("KF {id}: {:?}/{:?}", m.tcw, k.tcw)),
                        None => bad.push(format!("KF {id} 없음")),
                    }
                }
                for (id, p) in &d.mps {
                    if p.map != map_id || p.bad {
                        continue;
                    }
                    match st.atlas.mps.get(id) {
                        Some(m) if b3(&m.pos, &p.pos) => {}
                        Some(m) => bad.push(format!("점 {id}: {:?}/{:?}", m.pos, p.pos)),
                        None => bad.push(format!("점 {id} 없음")),
                    }
                }
                eprintln!("{path} GBA(루프 KF {loop_kf}): 다름 {} {:?}", bad.len(), bad.iter().take(4).collect::<Vec<_>>());
                same += bad.is_empty() as usize;
            } else {
                panic!("모르는 기록 {magic:x}");
            }
        }
    }
    eprintln!("결과 넣기 {n} 번 중 같음 {same}");
    assert!(n > 0);
    assert_eq!(same, n);
}

fn b3(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn bse3(a: &Se3, b: &Se3) -> bool {
    a.q.iter().chain(&a.t).zip(b.q.iter().chain(&b.t)).all(|(x, y)| x.to_bits() == y.to_bits())
}
