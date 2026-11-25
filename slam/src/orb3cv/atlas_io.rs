//! Atlas 저장 · 불러오기(원본 System::SaveAtlas · LoadAtlas: Atlas/Map/KeyFrame/MapPoint 의 PreSave · serialize · PostLoad).
//! Boost 직렬화 대신 우리 이진 형식(작은 끝 순서, 머리 "ORB3RSAT" + 판 1). 무엇을 남기고 무엇을 되돌리는지는 원본과 같다:
//! - 저장: id 순으로 지도마다 나쁘지 않은 키프레임 · 지도 점만. 다른 지도 · 나쁜 키프레임을 가리키는 연결 · 자식 · 루프 · 합침 가장자리 ·
//!   점 번호는 빼거나 −1(원본 PreSave 의 `spKF.find` · `spMP.find`). 키프레임이 하나도 없는 지도는 버림(SetMapBad).
//! - 불러오기: 저장된 칸은 비트 그대로, 저장 안 되는 칸은 원본 기본값(MapPoint 의 mnVisible · mnFound = 1, 추적 임시 칸 0),
//!   ordered_conn 은 UpdateBestCovisibles(무게 · id 오름차순 정렬 뒤 뒤집기), 자식 · 루프 · 합침 가장자리는 std::set 처럼 id 순.
//!   그 뒤 System 처럼 빈 지도를 하나 새로 만들어 현재 지도로 둔다. 키프레임 DB 에 넣기(pKFDB->add)는 부르는 쪽이 한다.
//! - 원본 Map::PreSave 는 다른 지도 · 나쁜 키프레임 관측을 EraseObservation 으로 지우며(nObs 줄이기 · 기준 키프레임 바꾸기 ·
//!   점 나쁨 표시까지) 살아 있는 지도를 바꾼다 — [`pre_save`](저장 전에 부름). `save` 는 그런 관측이 남아 있으면 Err.

use crate::orb3rs::types::{Atlas, Desc, FrameCtx, KeyFrame, KeyPoint, Map, MapPoint, Se3};
use std::collections::BTreeSet;
use std::io::{self, Read, Write};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"ORB3RSAT";
const VERSION: u32 = 1;

struct W<'a, T: Write>(&'a mut T);
impl<T: Write> W<'_, T> {
    fn u8(&mut self, v: u8) -> io::Result<()> {
        self.0.write_all(&[v])
    }
    fn u32(&mut self, v: u32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn i32(&mut self, v: i32) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn i64(&mut self, v: i64) -> io::Result<()> {
        self.0.write_all(&v.to_le_bytes())
    }
    fn f32(&mut self, v: f32) -> io::Result<()> {
        self.u32(v.to_bits())
    }
    fn f64(&mut self, v: f64) -> io::Result<()> {
        self.u64(v.to_bits())
    }
    fn len(&mut self, n: usize) -> io::Result<()> {
        self.u64(n as u64)
    }
    fn f32s(&mut self, v: &[f32]) -> io::Result<()> {
        self.len(v.len())?;
        v.iter().try_for_each(|&x| self.f32(x))
    }
    fn se3(&mut self, s: &Se3) -> io::Result<()> {
        s.q.iter().chain(&s.t).try_for_each(|&x| self.f32(x))
    }
    fn desc(&mut self, d: &Desc) -> io::Result<()> {
        self.0.write_all(d)
    }
    fn ids(&mut self, v: &[u64]) -> io::Result<()> {
        self.len(v.len())?;
        v.iter().try_for_each(|&x| self.u64(x))
    }
}

struct R<'a, T: Read>(&'a mut T);
impl<T: Read> R<'_, T> {
    fn arr<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.0.read_exact(&mut b)?;
        Ok(b)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.arr::<1>()?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn len(&mut self) -> io::Result<usize> {
        let n = self.u64()?;
        if n > (1 << 32) {
            return Err(bad("길이가 너무 큼"));
        }
        Ok(n as usize)
    }
    fn f32s(&mut self) -> io::Result<Vec<f32>> {
        let n = self.len()?;
        (0..n).map(|_| self.f32()).collect()
    }
    fn se3(&mut self) -> io::Result<Se3> {
        let mut v = [0f32; 7];
        for x in v.iter_mut() {
            *x = self.f32()?;
        }
        Ok(Se3 { q: [v[0], v[1], v[2], v[3]], t: [v[4], v[5], v[6]] })
    }
    fn desc(&mut self) -> io::Result<Desc> {
        self.arr()
    }
    fn ids(&mut self) -> io::Result<Vec<u64>> {
        let n = self.len()?;
        (0..n).map(|_| self.u64()).collect()
    }
}

fn bad(m: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.to_string())
}

fn ctx_eq(a: &FrameCtx, b: &FrameCtx) -> bool {
    a.cam == b.cam && a.pyr == b.pyr && [a.min_x, a.max_x, a.min_y, a.max_y, a.grid_inv_w, a.grid_inv_h].map(f32::to_bits) == [b.min_x, b.max_x, b.min_y, b.max_y, b.grid_inv_w, b.grid_inv_h].map(f32::to_bits)
}


/// Atlas::PreSave 중 살아 있는 Atlas 를 바꾸는 부분: id 순 지도마다(나쁘지 않고 키프레임이 있는 것) Map::PreSave 의 첫 고리 —
/// 나쁘지 않은 점마다 관측 사본을 돌며 다른 지도 · 나쁜 키프레임 관측을 EraseObservation. 키프레임 없는 지도는 SetMapBad 처럼 나쁨 표시.
/// (mnLastInitKFidMap 등 저장 형식에 안 담기는 칸은 다루지 않음.)
pub fn pre_save(a: &mut Atlas) {
    let mut ids: Vec<u64> = a.maps.iter().map(|m| m.id).collect();
    ids.sort();
    for mid in ids {
        let Some(mi) = a.maps.iter().position(|m| m.id == mid) else { continue };
        if a.maps[mi].bad {
            continue;
        }
        if a.maps[mi].kfs.is_empty() {
            a.maps[mi].bad = true;
            continue;
        }
        let mut pts = a.maps[mi].mps.clone();
        pts.sort();
        pts.dedup();
        for mp in pts {
            let Some(p) = a.mps.get(&mp) else { continue };
            if p.bad {
                continue;
            }
            let obs: Vec<u64> = p.obs.keys().copied().collect();
            for kf in obs {
                let foreign = a.kfs.get(&kf).map_or(true, |k| k.map != mid || k.bad);
                if foreign {
                    crate::orb3rs::mapdata::mp_erase_observation(a, mp, kf);
                }
            }
        }
    }
}

/// System::SaveAtlas. next_frame = Frame::nNextId, next_map = Map::nNextId(원본이 함께 저장)
pub fn save<T: Write>(a: &Atlas, next_frame: u64, next_map: u64, out: &mut T) -> io::Result<()> {
    // 저장할 지도: id 순, 나쁘지 않고 키프레임이 있는 것
    let mut maps: Vec<&Map> = a.maps.iter().filter(|m| !m.bad && !m.kfs.is_empty()).collect();
    maps.sort_by_key(|m| m.id);
    // 모든 키프레임이 같은 카메라 · 층 표를 쓴다(원본은 키프레임마다 저장 — 다르면 우리 형식으로는 못 담음)
    let ctx = a.kfs.values().next().map(|k| k.ctx.clone());
    if let Some(c) = &ctx {
        if a.kfs.values().any(|k| !ctx_eq(&k.ctx, c)) {
            return Err(bad("키프레임마다 카메라 값이 다름"));
        }
    }
    let mut w = W(out);
    w.0.write_all(MAGIC)?;
    w.u32(VERSION)?;
    w.u64(a.next_kf)?;
    w.u64(a.next_mp)?;
    w.u64(next_frame)?;
    w.u64(next_map)?;
    // 카메라 · 층 표
    w.u8(ctx.is_some() as u8)?;
    if let Some(c) = &ctx {
        let cm = &c.cam;
        for v in [cm.fx, cm.fy, cm.cx, cm.cy, cm.invfx, cm.invfy, cm.bf, cm.b, cm.th_depth] {
            w.f32(v)?;
        }
        w.i32(cm.width)?;
        w.i32(cm.height)?;
        let p = &c.pyr;
        w.i32(p.nfeatures)?;
        w.f32(p.scale_factor)?;
        w.i32(p.nlevels)?;
        w.i32(p.ini_th_fast)?;
        w.i32(p.min_th_fast)?;
        for v in [&p.scale, &p.inv_scale, &p.sigma2, &p.inv_sigma2] {
            w.f32s(v)?;
        }
        w.f32(p.log_scale_factor)?;
        for v in [c.min_x, c.max_x, c.min_y, c.max_y, c.grid_inv_w, c.grid_inv_h] {
            w.f32(v)?;
        }
    }
    w.len(maps.len())?;
    for m in maps {
        let kset: BTreeSet<u64> = m.kfs.iter().copied().filter(|id| a.kfs.get(id).is_some_and(|k| !k.bad)).collect();
        let mset: BTreeSet<u64> = m.mps.iter().copied().filter(|id| a.mps.get(id).is_some_and(|p| !p.bad)).collect();
        for id in &mset {
            let p = &a.mps[id];
            if p.obs.keys().any(|k| !kset.contains(k)) {
                return Err(bad(&format!("지도 점 {id} 가 다른 지도 · 나쁜 키프레임 관측을 가짐(EraseObservation 먼저)")));
            }
        }
        w.u64(m.id)?;
        w.i32(m.change_id)?;
        w.u8(m.origin_kf.is_some() as u8)?;
        w.u64(m.origin_kf.unwrap_or(0))?;
        w.len(mset.len())?;
        for id in &mset {
            let p = &a.mps[id];
            w.u64(p.id)?;
            w.u64(p.first_kf)?;
            w.u64(p.first_frame)?;
            w.i32(p.n_obs)?;
            p.pos.iter().chain(&p.normal).try_for_each(|&x| w.f32(x))?;
            w.len(p.obs.len())?;
            for (k, (l, r)) in &p.obs {
                w.u64(*k)?;
                w.i32(*l)?;
                w.i32(*r)?;
            }
            w.desc(&p.desc)?;
            w.u64(p.ref_kf)?;
            // mBackupReplacedId: 같은 지도의 점일 때만
            w.i64(p.replaced.filter(|r| mset.contains(r)).map_or(-1, |r| r as i64))?;
            w.f32(p.min_dist)?;
            w.f32(p.max_dist)?;
        }
        w.len(kset.len())?;
        for id in &kset {
            let k = &a.kfs[id];
            w.u64(k.id)?;
            w.u64(k.frame_id)?;
            w.f64(k.timestamp)?;
            w.len(k.keys.len())?;
            for kp in &k.keys {
                for v in [kp.x, kp.y, kp.size, kp.angle, kp.response] {
                    w.f32(v)?;
                }
                w.i32(kp.octave)?;
                w.i32(kp.class_id)?;
            }
            w.f32s(&k.u_right)?;
            w.f32s(&k.depth)?;
            k.desc.iter().try_for_each(|d| w.desc(d))?;
            w.len(k.bow.len())?;
            for (word, v) in &k.bow {
                w.u32(*word)?;
                w.f64(*v)?;
            }
            w.len(k.feat.len())?;
            for (node, v) in &k.feat {
                w.u32(*node)?;
                w.len(v.len())?;
                v.iter().try_for_each(|&x| w.u32(x))?;
            }
            w.se3(&k.tcp)?;
            w.se3(&k.tcw)?;
            // mvBackupMapPointsId: 같은 지도 점만, 아니면 −1
            w.len(k.map_points.len())?;
            for mp in &k.map_points {
                w.i64(mp.filter(|m| mset.contains(m)).map_or(-1, |m| m as i64))?;
            }
            w.len(k.grid.len())?;
            for c in &k.grid {
                w.len(c.len())?;
                c.iter().try_for_each(|&x| w.u32(x))?;
            }
            let conn: Vec<(u64, i32)> = k.conn_weights.iter().filter(|(id, _)| kset.contains(id)).map(|(a, b)| (*a, *b)).collect();
            w.len(conn.len())?;
            for (id, wt) in conn {
                w.u64(id)?;
                w.i32(wt)?;
            }
            w.i64(k.parent.filter(|p| kset.contains(p)).map_or(-1, |p| p as i64))?;
            for v in [&k.children, &k.loop_edges, &k.merge_edges] {
                let s: BTreeSet<u64> = v.iter().copied().filter(|x| kset.contains(x)).collect();
                w.ids(&s.into_iter().collect::<Vec<_>>())?;
            }
            w.u8(k.bad as u8)?;
        }
    }
    Ok(())
}

/// System::LoadAtlas + Atlas · Map · KeyFrame · MapPoint PostLoad + System 의 CreateNewMap. 돌려줌: (Atlas, Frame::nNextId)
pub fn load<T: Read>(input: &mut T) -> io::Result<(Atlas, u64)> {
    let mut r = R(input);
    if &r.arr::<8>()? != MAGIC {
        return Err(bad("Atlas 파일 아님"));
    }
    if r.u32()? != VERSION {
        return Err(bad("판이 다름"));
    }
    let mut a = Atlas { next_kf: r.u64()?, next_mp: r.u64()?, ..Default::default() };
    let next_frame = r.u64()?;
    let next_map = r.u64()?;
    let ctx = if r.u8()? != 0 {
        let mut c = FrameCtx::default();
        let cm = &mut c.cam;
        for v in [&mut cm.fx, &mut cm.fy, &mut cm.cx, &mut cm.cy, &mut cm.invfx, &mut cm.invfy, &mut cm.bf, &mut cm.b, &mut cm.th_depth] {
            *v = r.f32()?;
        }
        cm.width = r.i32()?;
        cm.height = r.i32()?;
        let p = &mut c.pyr;
        p.nfeatures = r.i32()?;
        p.scale_factor = r.f32()?;
        p.nlevels = r.i32()?;
        p.ini_th_fast = r.i32()?;
        p.min_th_fast = r.i32()?;
        p.scale = r.f32s()?;
        p.inv_scale = r.f32s()?;
        p.sigma2 = r.f32s()?;
        p.inv_sigma2 = r.f32s()?;
        p.log_scale_factor = r.f32()?;
        for v in [&mut c.min_x, &mut c.max_x, &mut c.min_y, &mut c.max_y, &mut c.grid_inv_w, &mut c.grid_inv_h] {
            *v = r.f32()?;
        }
        Arc::new(c)
    } else {
        Arc::new(FrameCtx::default())
    };
    let nmaps = r.len()?;
    for _ in 0..nmaps {
        let mut m = Map { id: r.u64()?, change_id: r.i32()?, ..Default::default() };
        let has_origin = r.u8()? != 0;
        let origin = r.u64()?;
        let nmp = r.len()?;
        let mut mps = Vec::with_capacity(nmp);
        for _ in 0..nmp {
            let mut p = MapPoint { visible: 1, found: 1, map: m.id, ..Default::default() };
            p.id = r.u64()?;
            p.first_kf = r.u64()?;
            p.first_frame = r.u64()?;
            p.n_obs = r.i32()?;
            for x in p.pos.iter_mut().chain(p.normal.iter_mut()) {
                *x = r.f32()?;
            }
            let no = r.len()?;
            for _ in 0..no {
                let k = r.u64()?;
                let l = r.i32()?;
                let rr = r.i32()?;
                p.obs.insert(k, (l, rr));
            }
            p.desc = r.desc()?;
            p.ref_kf = r.u64()?;
            let rep = r.i64()?;
            p.replaced = (rep >= 0).then_some(rep as u64);
            p.min_dist = r.f32()?;
            p.max_dist = r.f32()?;
            mps.push(p);
        }
        let nkf = r.len()?;
        let mut kfs = Vec::with_capacity(nkf);
        for _ in 0..nkf {
            let mut k = KeyFrame { map: m.id, ctx: ctx.clone(), ..Default::default() };
            k.id = r.u64()?;
            k.frame_id = r.u64()?;
            k.timestamp = r.f64()?;
            let n = r.len()?;
            k.keys = (0..n)
                .map(|_| -> io::Result<KeyPoint> {
                    Ok(KeyPoint { x: r.f32()?, y: r.f32()?, size: r.f32()?, angle: r.f32()?, response: r.f32()?, octave: r.i32()?, class_id: r.i32()? })
                })
                .collect::<io::Result<_>>()?;
            k.u_right = r.f32s()?;
            k.depth = r.f32s()?;
            k.desc = (0..n).map(|_| r.desc()).collect::<io::Result<_>>()?;
            let nb = r.len()?;
            for _ in 0..nb {
                let word = r.u32()?;
                let v = r.f64()?;
                k.bow.insert(word, v);
            }
            let nf = r.len()?;
            for _ in 0..nf {
                let node = r.u32()?;
                let c = r.len()?;
                let v = (0..c).map(|_| r.u32()).collect::<io::Result<Vec<_>>>()?;
                k.feat.insert(node, v);
            }
            k.tcp = r.se3()?;
            // SetPose: mTcw · mTwc
            k.set_pose(r.se3()?);
            let nm = r.len()?;
            k.map_points = (0..nm).map(|_| r.i64().map(|v| (v >= 0).then_some(v as u64))).collect::<io::Result<_>>()?;
            let ng = r.len()?;
            k.grid = (0..ng)
                .map(|_| {
                    let c = r.len()?;
                    (0..c).map(|_| r.u32()).collect::<io::Result<Vec<_>>>()
                })
                .collect::<io::Result<_>>()?;
            let nc = r.len()?;
            for _ in 0..nc {
                let id = r.u64()?;
                let wt = r.i32()?;
                k.conn_weights.insert(id, wt);
            }
            // UpdateBestCovisibles: (무게, 키프레임) 오름차순 정렬 → 앞에 끼워 넣기 = 내림차순
            let mut pairs: Vec<(i32, u64)> = k.conn_weights.iter().map(|(id, wt)| (*wt, *id)).collect();
            pairs.sort();
            k.ordered_conn = pairs.into_iter().rev().map(|(wt, id)| (id, wt)).collect();
            let par = r.i64()?;
            k.parent = (par >= 0).then_some(par as u64);
            k.children = r.ids()?;
            k.loop_edges = r.ids()?;
            k.merge_edges = r.ids()?;
            k.bad = r.u8()? != 0;
            kfs.push(k);
        }
        // PostLoad: 같은 지도에 없는 번호는 지움(원본 map[id] 가 null)
        let kset: BTreeSet<u64> = kfs.iter().map(|k| k.id).collect();
        let mset: BTreeSet<u64> = mps.iter().map(|p| p.id).collect();
        for p in &mut mps {
            p.obs.retain(|k, _| kset.contains(k));
            if p.replaced.is_some_and(|x| !mset.contains(&x)) {
                p.replaced = None;
            }
        }
        for k in &mut kfs {
            for mp in k.map_points.iter_mut() {
                if mp.is_some_and(|x| !mset.contains(&x)) {
                    *mp = None;
                }
            }
        }
        m.kfs = kfs.iter().map(|k| k.id).collect();
        m.mps = mps.iter().map(|p| p.id).collect();
        m.origin_kf = if has_origin && kset.contains(&origin) { Some(origin) } else { None };
        for k in kfs {
            a.kfs.insert(k.id, k);
        }
        for p in mps {
            a.mps.insert(p.id, p);
        }
        a.maps.push(m);
    }
    // System: 불러온 뒤 새 지도
    a.maps.push(Map { id: next_map, ..Default::default() });
    a.current = a.maps.len() - 1;
    Ok((a, next_frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orb3rs::types::{Camera, Pyramid};

    struct Rng(u64);
    impl Rng {
        fn n(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            f32::from_bits((self.n() as u32 & 0x3fff_ffff) | 0x0080_0000)
        }
    }

    fn sample() -> Atlas {
        let mut g = Rng(0x1234_5678_9abc);
        let cam = Camera { fx: 600.5, fy: 601.25, cx: 320.1, cy: 240.7, invfx: 1.0 / 600.5, invfy: 1.0 / 601.25, bf: 30.0, b: 0.05, th_depth: 2.0, width: 640, height: 480 };
        let pyr = Pyramid { nfeatures: 1250, scale_factor: 1.2, nlevels: 8, ini_th_fast: 20, min_th_fast: 7, scale: vec![1.0, 1.2], inv_scale: vec![1.0, 0.8333], sigma2: vec![1.0, 1.44], inv_sigma2: vec![1.0, 0.69], log_scale_factor: 0.18 };
        let ctx = Arc::new(FrameCtx { cam, pyr, min_x: 0.0, max_x: 640.0, min_y: 0.0, max_y: 480.0, grid_inv_w: 0.1, grid_inv_h: 0.1 });
        let mut a = Atlas { next_kf: 40, next_mp: 400, ..Default::default() };
        for mid in 0..3u64 {
            let mut m = Map { id: mid, change_id: mid as i32 * 3, origin_kf: Some(mid * 10), bad: mid == 2, ..Default::default() };
            for kid in mid * 10..mid * 10 + 8 {
                let n = 20;
                let mut k = KeyFrame { id: kid, frame_id: kid * 3, map: mid, timestamp: kid as f64 * 0.033 + 1e9, ctx: ctx.clone(), bad: kid % 7 == 6, ..Default::default() };
                k.keys = (0..n).map(|i| KeyPoint { x: g.f(), y: g.f(), size: 31.0, angle: g.f(), response: g.f(), octave: i % 8, class_id: -1 }).collect();
                k.desc = (0..n).map(|_| std::array::from_fn(|_| g.n() as u8)).collect();
                k.u_right = (0..n).map(|_| g.f()).collect();
                k.depth = (0..n).map(|_| g.f()).collect();
                k.map_points = (0..n).map(|i| if i % 3 == 0 { None } else { Some(mid * 100 + (g.n() % 30)) }).collect();
                k.grid = (0..12).map(|i| (0..i).map(|j| j * 2).collect()).collect();
                k.set_pose(Se3 { q: [0.1, 0.2, 0.3, 0.927], t: [g.f(), g.f(), g.f()] });
                k.tcp = Se3 { q: [0.0, 0.0, 0.0, 1.0], t: [g.f(), 1.0, 2.0] };
                k.bow = (0..5).map(|i| (i * 100 + g.n() as u32 % 50, g.f() as f64 / 3.0)).collect();
                k.feat = (0..3).map(|i| (i, vec![1, 2, 3])).collect();
                for o in mid * 10..mid * 10 + 8 {
                    if o != kid && g.n() % 2 == 0 {
                        k.conn_weights.insert(o, (g.n() % 5) as i32 + 15);
                    }
                }
                k.conn_weights.insert(99, 50); // 다른 지도: 빠져야 함
                k.parent = if kid > mid * 10 { Some(kid - 1) } else { None };
                k.children = vec![kid + 1, 999];
                k.loop_edges = vec![mid * 10 + 3, mid * 10 + 1];
                m.kfs.push(kid);
                a.kfs.insert(kid, k);
            }
            for pid in mid * 100..mid * 100 + 30 {
                let mut p = MapPoint { id: pid, map: mid, pos: [g.f(), g.f(), g.f()], normal: [g.f(), 0.0, 1.0], desc: std::array::from_fn(|_| g.n() as u8), n_obs: 2, first_frame: 5, ref_kf: mid * 10, first_kf: mid * 10, visible: 9, found: 4, min_dist: g.f(), max_dist: g.f(), bad: pid % 11 == 10, last_frame_seen: 77, track_in_view: true, ..Default::default() };
                for k in mid * 10..mid * 10 + 6 {
                    if (pid + k) % 3 == 0 && k % 7 != 6 {
                        p.obs.insert(k, (k as i32, -1));
                    }
                }
                p.replaced = if pid % 5 == 0 { Some(pid + 1) } else { None };
                m.mps.push(pid);
                a.mps.insert(pid, p);
            }
            a.maps.push(m);
        }
        a
    }

    #[test]
    fn round_trip() {
        let a = sample();
        let mut buf = Vec::new();
        save(&a, 123, 3, &mut buf).unwrap();
        let (b, nf) = load(&mut buf.as_slice()).unwrap();
        assert_eq!(nf, 123);
        assert_eq!((b.next_kf, b.next_mp), (40, 400));
        // 나쁜 지도(2) 빠지고 새 빈 지도(3)가 현재
        assert_eq!(b.maps.iter().map(|m| m.id).collect::<Vec<_>>(), vec![0, 1, 3]);
        assert_eq!(b.current, 2);
        for (id, k) in &b.kfs {
            let o = &a.kfs[id];
            assert!(!o.bad);
            assert_eq!(k.keys.len(), o.keys.len());
            assert!(k.keys.iter().zip(&o.keys).all(|(x, y)| x.x.to_bits() == y.x.to_bits() && x.angle.to_bits() == y.angle.to_bits() && x.octave == y.octave));
            assert_eq!(k.desc, o.desc);
            assert_eq!(k.u_right.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), o.u_right.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(k.tcw.q, o.tcw.q);
            assert_eq!(k.twc.t, o.twc.t);
            assert_eq!(k.bow, o.bow);
            assert_eq!(k.feat, o.feat);
            assert_eq!(k.grid, o.grid);
            assert!(!k.conn_weights.contains_key(&99));
            assert!(!k.children.contains(&999));
            assert!(k.loop_edges.windows(2).all(|w| w[0] < w[1]));
            assert!(k.ordered_conn.windows(2).all(|w| w[0].1 >= w[1].1));
            for (x, y) in k.map_points.iter().zip(&o.map_points) {
                match x {
                    Some(m) => assert_eq!(Some(*m), *y),
                    None => assert!(y.is_none() || a.mps[&y.unwrap()].bad || !a.mps.contains_key(&y.unwrap())),
                }
            }
        }
        assert!(a.kfs.values().filter(|k| !k.bad && k.map != 2).all(|k| b.kfs.contains_key(&k.id)));
        for (id, p) in &b.mps {
            let o = &a.mps[id];
            assert!(!o.bad);
            assert_eq!(p.pos.map(f32::to_bits), o.pos.map(f32::to_bits));
            assert_eq!(p.desc, o.desc);
            assert_eq!((p.visible, p.found, p.last_frame_seen, p.track_in_view), (1, 1, 0, false));
            assert!(p.obs.keys().all(|k| b.kfs.contains_key(k)));
        }
        // 다시 저장하면 같은 바이트(새 빈 지도는 키프레임이 없어 빠짐)
        let mut buf2 = Vec::new();
        save(&b, 123, 3, &mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn pre_save_erases_foreign_obs() {
        let mut a = sample();
        // 점 1: 다른 지도(10) · 나쁜 키프레임(6) 관측 섞기, nObs 를 넉넉히
        let k0 = a.kfs[&0].u_right[3];
        a.kfs.get_mut(&0).unwrap().u_right[3] = 1.0; // 입체
        a.kfs.get_mut(&10).unwrap().u_right[4] = -1.0; // 단안
        a.kfs.get_mut(&10).unwrap().map_points[4] = Some(1);
        {
            let p = a.mps.get_mut(&1).unwrap();
            p.obs.clear();
            p.obs.insert(0, (3, -1));
            p.obs.insert(6, (2, -1));
            p.obs.insert(10, (4, -1));
            p.n_obs = 6;
            p.ref_kf = 10;
            p.bad = false;
        }
        // 점 2: 바깥 관측을 지우면 nObs ≤ 2 → 나쁨
        {
            let p = a.mps.get_mut(&2).unwrap();
            p.obs.clear();
            p.obs.insert(1, (5, -1));
            p.obs.insert(11, (6, -1));
            p.n_obs = 3;
            p.bad = false;
        }
        a.kfs.get_mut(&1).unwrap().map_points[5] = Some(2);
        assert!(save(&a, 1, 3, &mut Vec::new()).is_err());
        pre_save(&mut a);
        let p = &a.mps[&1];
        assert_eq!(p.obs.keys().copied().collect::<Vec<_>>(), vec![0]);
        // 6(나쁨, u_right 값에 따라 1 또는 2) · 10(단안 1)
        let six = if a.kfs[&6].u_right[2] >= 0.0 { 2 } else { 1 };
        assert_eq!(p.n_obs, 6 - six - 1);
        assert_eq!(p.ref_kf, 0);
        assert_eq!(p.bad, p.n_obs <= 2);
        let q = &a.mps[&2];
        assert!(q.bad && q.obs.is_empty());
        assert_eq!(a.kfs[&1].map_points[5], None);
        assert!(!a.maps[0].mps.contains(&2));
        // 다른 지도 키프레임의 번호는 EraseObservation 이 비우지 않음(원본 그대로)
        assert_eq!(a.kfs[&10].map_points[4], Some(1));
        let _ = k0;
        save(&a, 1, 3, &mut Vec::new()).unwrap();
    }
}
