//! 새 점 넣기(`orb3sys::rsys::create_new_map_points_into` = 포터 TRACK 삼각측량 + 통합 넣기) 원본 대조. 원본 = 번호 순 비교자 기준 빌드
//! `~/opt/orb3_sys/gba/gba_dump`(ORBSYS_CNMP_EVERY=n: n 번째 CreateNewMapPoints 마다 들어갈 때 지도 · MapPoint::nNextId(CNMB),
//! CheckNewKeyFrames 답 · 새 점 전부 · 최근 점 목록 끝(CNMA)).
//! ORBSYS_CNMP=덤프 cargo test --release -p slam --test orb3sys_cnmp
use slam::orb3loop::correct::*;
use slam::orb3loop::detect::G2oSim3;
use slam::orb3rs::*;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;


use slam::orb3rs::localmap::TriParams;
use slam::orb3sys::rsys::create_new_map_points_into;

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
fn cnmp_insert_matches_original() {
    let dumps = std::env::var("ORBSYS_CNMP").expect("입력 없음: ORBSYS_CNMP(gba_dump CNMP 덤프, : 로 여럿)");
    let (mut n, mut same, mut npts, mut n_struct) = (0, 0, 0usize, 0usize);
    for path in dumps.split(':') {
        let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("덤프 {path}: {e}"));
        assert!(f.metadata().unwrap().len() > 0, "빈 덤프 {path}");
        let mut r = Rd { f: std::io::BufReader::with_capacity(1 << 20, f) };
        while r.more() {
            let magic = r.u32();
            assert_eq!(magic, 0x424d4e43, "CNMB 기대");
            let kf = r.u64();
            let n0 = r.u64();
            let far = r.u8() != 0;
            let th_far = r.f32();
            let mut d = read_map(&mut r);
            let nk = r.u32();
            for _ in 0..nk {
                let id = r.u64();
                let nkp = r.u32();
                let raw: Vec<[f32; 2]> = (0..nkp).map(|_| [r.f32(), r.f32()]).collect();
                if let Some(k) = d.kfs.get_mut(&id) {
                    k.keys_raw = raw;
                }
            }
            assert_eq!(r.u32(), 0x414d4e43, "CNMA 기대");
            let nc = r.u32();
            let mut chk: std::collections::VecDeque<bool> = (0..nc).map(|_| r.u8() != 0).collect();
            let nn = r.u32();
            struct P {
                id: u64,
                bad: bool,
                pos: [f32; 3],
                normal: [f32; 3],
                desc: [u8; 32],
                min: f32,
                max: f32,
                rk: i64,
                nobs: i32,
                vis: i32,
                fnd: i32,
                fkf: i64,
                ffr: i64,
                obs: BTreeMap<u64, (i32, i32)>,
            }
            let want: Vec<P> = (0..nn)
                .map(|_| {
                    let id = r.u64();
                    let bad = r.u8() != 0;
                    let pos = [r.f32(), r.f32(), r.f32()];
                    let normal = [r.f32(), r.f32(), r.f32()];
                    let desc = r.take::<32>();
                    let (min, max) = (r.f32(), r.f32());
                    let rk = r.i64();
                    let (nobs, vis, fnd) = (r.i32(), r.i32(), r.i32());
                    let (fkf, ffr) = (r.i64(), r.i64());
                    let no = r.u32();
                    let obs = (0..no).map(|_| (r.u64(), (r.i32(), r.i32()))).collect();
                    P { id, bad, pos, normal, desc, min, max, rk, nobs, vis, fnd, fkf, ffr, obs }
                })
                .collect();
            let nr = r.u32();
            let want_recent: Vec<u64> = (0..nr).map(|_| r.i64() as u64).collect();
            // Rust 쪽
            let ctx = d.kfs[&kf].ctx.clone();
            let tri = TriParams {
                cam: ctx.cam.clone(),
                scale: ctx.pyr.scale.clone(),
                sigma2: ctx.pyr.sigma2.clone(),
                scale_factor: ctx.pyr.scale_factor,
                monocular: false,
                inertial: false,
                far_points: far,
                th_far_points: th_far,
            };
            let map_id = d.map_id;
            let mut a = Atlas::default();
            let mut m = Map { id: map_id, ..Default::default() };
            for (id, k) in &d.kfs {
                if k.map == map_id && !k.bad {
                    m.kfs.push(*id);
                }
            }
            for (id, p) in &d.mps {
                if p.map == map_id && !p.bad {
                    m.mps.push(*id);
                }
            }
            a.kfs = d.kfs;
            a.mps = d.mps;
            a.maps.push(m);
            a.next_mp = n0;
            let mut recent = Vec::new();
            let got = create_new_map_points_into(&mut a, kf, &tri, &mut recent, &mut || chk.pop_front().unwrap_or(true));
            n += 1;
            npts += want.len();
            let mut bad = Vec::new();
            if got != want.iter().map(|p| p.id).collect::<Vec<_>>() {
                bad.push(format!("새 점 id {} 개 / 원본 {} 개", got.len(), want.len()));
            }
            if recent != want_recent {
                bad.push("최근 점 목록".to_string());
            }
            let mut struct_bad = 0;
            for w in &want {
                let Some(p) = a.mps.get(&w.id) else { continue };
                // 구조(삼각측량 값과 무관한 칸): 관측 · nObs · 보임 · 찾음 · 첫 KF · 첫 장면 · 기준 KF · 나쁨
                if !(p.bad == w.bad && p.ref_kf as i64 == w.rk && (p.n_obs, p.visible, p.found) == (w.nobs, w.vis, w.fnd) && (p.first_kf as i64, p.first_frame as i64) == (w.fkf, w.ffr) && p.obs == w.obs) {
                    struct_bad += 1;
                }
                let b3 = |x: &[f32; 3], y: &[f32; 3]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
                let ok = p.bad == w.bad
                    && b3(&p.pos, &w.pos)
                    && b3(&p.normal, &w.normal)
                    && p.desc == w.desc
                    && p.min_dist.to_bits() == w.min.to_bits()
                    && p.max_dist.to_bits() == w.max.to_bits()
                    && p.ref_kf as i64 == w.rk
                    && (p.n_obs, p.visible, p.found) == (w.nobs, w.vis, w.fnd)
                    && (p.first_kf as i64, p.first_frame as i64) == (w.fkf, w.ffr)
                    && p.obs == w.obs;
                if !ok && bad.len() < 5 {
                    bad.push(format!("점 {}: 위치 {:?}/{:?} 법선 {:?}/{:?} 거리 {} {}/{} {} 관측수 {}/{}", w.id, p.pos, w.pos, p.normal, w.normal, p.min_dist, p.max_dist, w.min, w.max, p.n_obs, w.nobs));
                }
            }
            eprintln!("{path} KF {kf}: 새 점 {} 개, 구조 다름 {struct_bad}, 다름 {:?}", want.len(), bad.iter().take(2).collect::<Vec<_>>());
            same += bad.is_empty() as usize;
            n_struct += (struct_bad == 0 && got.len() == want.len() && recent == want_recent) as usize;
        }
    }
    eprintln!("CreateNewMapPoints {n} 번(새 점 {npts} 개) 중 비트 같음 {same}, 구조(id · 관측 · 수 · 최근 목록) 같음 {n_struct}");
    assert!(n > 0);
    assert_eq!(same, n);
}
