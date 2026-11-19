//! /bstream: 바뀐 몫만 담은 이진 판(binary delta). 서버는 열린 페이지마다 보낸 판을 기억하고(`Client`), 새 판이 오면
//! 그 페이지가 아직 못 받은 것만 한 덩어리(frame)로 보낸다. 페이지는 Web Worker 에서 풀고 메인 스레드는 적용만.
//!
//! 덩어리 = u32 길이(뒤 바이트 수) + 칸(section) 들. 칸 = u8 종류 + u32 길이 + 몸. 숫자는 모두 little-endian.
//!   1 VER   u16 n + 판 글자, f64 wall(ms, 0 = 없음)
//!   2 HEAD  view.json 의 나머지 키 중 바뀐 것만 담은 JSON 객체(objects · graph.nodes · wall 제외, 없어진 키는 "__del":[…])
//!   3 OBJS  u32 n + n × u32 id(지금 차례, 없는 id 는 지움), u32 m + m × (u32 길이 + 물체 JSON)(새로 · 바뀐 것만)
//!   4 TRAJ  f64 로그 id, u32 from(0 = 새 길), u32 n + n × (f64 t, f64 x, y, z)(덧붙은 점만)
//!   5 MAP   u16 n + 판 글자, u32 w, h, u8 full, u32 k + k × (u32 x0, y0, tw, th + tw·th 바이트)(PGM 줄 차례, 바뀐 32 칸 조각만)
//!   6 OCT   f64 run, u8 reset, u32 n + n × (u8 l, i32 x, y, z, u32 점 수, u32 판)(점이 늘어난 노드만; reset 이면 전부)
//!   7 POSE  pose.json 바이트
//!   8 SUB   f64 run, u8 reset, u32 n × (u32 id, u32 점 판, u32 점 수, f32 × 12 자세(r 행 차례 + p)), u32 m × u32 없어진 id
//!           (청크 점 지도 `--pts-map chunk`: 자세가 바뀌었거나 점이 는 청크만 — 루프 클로저는 자세만 오고 점은 그대로)
//!   9 PTS   u32 k × (u32 id, u32 점 판, u32 길이 + 청크 점 파일 바이트(u32 n + n × i16 xyz + n × u8 rgb)): 칸 8 로 알린 청크 중
//!           이 페이지가 아직 그 판의 점을 못 받은 것, 새 청크부터. 한 덩어리에 PTS_BUDGET 바이트까지, 페이지 쪽 보낼 바이트가 쌓여 있으면
//!           줄여서(느린 링크에서 밀린 바이트가 끝없이 늘지 않게) — 남은 것은 다음 덩어리로. 점은 페이지마다 파일 요청 없이 이 연결로만 간다
//! 빈 덩어리(길이 0)는 살아 있음 알림.
use crate::live::View;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const TILE: usize = 32;

/// 한 판을 물체 · 머리로 나눈 것(판마다 한 번, 모든 페이지가 나눠 씀)
pub struct Parts {
    ver: String,
    head: Vec<(String, u64, Vec<u8>)>,
    order: Vec<u32>,
    objs: Vec<(u32, u64, Vec<u8>)>,
    wall: f64,
    oct: Option<(u64, String, i64)>, // (run, octree.json 경로, 판)
    sub: Option<(u64, String, Vec<SubChunk>)>, // 청크 점 지도(판, 점 파일 폴더, 청크 목록): 머리에서는 뺀다
}

/// 청크 하나: id, 점 판, 점 수, 자세(r 행 차례 9 + p 3)
#[derive(Clone, PartialEq)]
pub struct SubChunk {
    id: u32,
    ver: u32,
    n: u32,
    pose: [f32; 12],
}

/// 옥트리 노드 목록(octree.json 판마다 한 번 읽음)
pub struct OctIdx {
    key: String,
    run: u64,
    nodes: Vec<[i64; 6]>,
}

#[derive(Default)]
pub struct Client {
    vv: String,
    head: HashMap<String, u64>,
    objs: HashMap<u32, u64>,
    order_h: u64,
    traj: (u64, usize),
    map_v: String,
    oct_run: Option<u64>,
    oct: HashMap<(i64, i64, i64, i64), i64>,
    sub_run: Option<u64>,
    sub: HashMap<u32, SubChunk>,
    /// 청크마다 보낸 점 판
    sub_pts: HashMap<u32, u32>,
    /// 아직 못 보낸 청크 점이 있음(보낼 바이트가 줄면 다시 짬)
    pub pts_pending: bool,
    pose_v: String,
}

/// 한 덩어리에 담는 청크 점 바이트 상한(Tailscale 중계 약 1 MB/s 에서 0.25 초 몫)
pub const PTS_BUDGET: usize = 256 << 10;

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h = (h ^ x as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl Parts {
    pub fn oct_ref(&self) -> Option<(u64, String, i64)> {
        self.oct.clone()
    }
}

pub fn parts(cache: &Mutex<Option<Arc<Parts>>>, v: &View) -> Option<Arc<Parts>> {
    if let Some(p) = cache.lock().unwrap().as_ref() {
        if p.ver == v.ver {
            return Some(p.clone());
        }
    }
    let doc = v.doc()?;
    let m = doc.as_object()?;
    let mut head = Vec::new();
    let mut order = Vec::new();
    let mut objs = Vec::new();
    let mut sub = None;
    for (k, x) in m {
        match k.as_str() {
            "objects" => {
                for (i, o) in x.as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter().enumerate() {
                    let id = o["id"].as_u64().unwrap_or(i as u64) as u32;
                    let b = serde_json::to_vec(o).unwrap_or_default();
                    order.push(id);
                    objs.push((id, fnv(&b), b));
                }
            }
            "wall" => {}
            "scene_points" if x["submap"]["list"].is_array() => {
                // 청크 목록은 칸 8 로 바뀐 것만(자세가 바뀐 청크 · 점이 는 청크), 머리에는 나머지만
                let s = &x["submap"];
                let list = s["list"].as_array().unwrap().iter().filter_map(|e| {
                    let a = e.as_array()?;
                    let g = |i: usize| a.get(i).and_then(|v| v.as_f64());
                    let mut pose = [0f32; 12];
                    for (i, q) in pose.iter_mut().enumerate() {
                        *q = g(3 + i)? as f32;
                    }
                    Some(SubChunk { id: g(0)? as u32, ver: g(1)? as u32, n: g(2)? as u32, pose })
                }).collect();
                sub = Some((s["run"].as_u64().unwrap_or(0), s["dir"].as_str().unwrap_or("").to_string(), list));
                let mut y = x.clone();
                if let Some(o) = y["submap"].as_object_mut() {
                    o.remove("list");
                }
                let b = serde_json::to_vec(&y).unwrap_or_default();
                head.push((k.clone(), fnv(&b), b));
            }
            _ => {
                let b = if k == "graph" { serde_json::to_vec(&serde_json::json!({"edges": x["edges"]})) } else { serde_json::to_vec(x) }.unwrap_or_default();
                head.push((k.clone(), fnv(&b), b));
            }
        }
    }
    let wall = doc["wall"].as_f64().unwrap_or(0.0);
    let o = &doc["scene_points"]["octree"];
    let oct = o["path"].as_str().map(|p| (o["run"].as_u64().unwrap_or(0), p.to_string(), o["ver"].as_i64().unwrap_or(0)));
    let p = Arc::new(Parts { ver: v.ver.clone(), head, order, objs, wall, oct, sub });
    *cache.lock().unwrap() = Some(p.clone());
    Some(p)
}

pub fn oct_idx(cache: &Mutex<Option<Arc<OctIdx>>>, dir: &std::path::Path, run: u64, path: &str, ver: i64) -> Option<Arc<OctIdx>> {
    let key = format!("{run}|{path}|{ver}");
    if let Some(c) = cache.lock().unwrap().as_ref() {
        if c.key == key {
            return Some(c.clone());
        }
    }
    let b = crate::memfs::read(&dir.join(path))?;
    let j: Value = serde_json::from_slice(&b).ok()?;
    if j["run"].as_u64() != Some(run) {
        return None; // 노드 목록이 아직 앞 판(다음 판에서 다시)
    }
    let nodes = j["nodes"].as_array()?.iter().filter_map(|n| {
        let a = n.as_array()?;
        let g = |i: usize| a.get(i).and_then(|x| x.as_i64());
        Some([g(0)?, g(1)?, g(2)?, g(3)?, g(4)?, g(5)?])
    }).collect();
    let o = Arc::new(OctIdx { key, run, nodes });
    *cache.lock().unwrap() = Some(o.clone());
    Some(o)
}

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) { self.0.push(v); }
    fn u16(&mut self, v: u16) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn i32(&mut self, v: i32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn f64(&mut self, v: f64) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn str16(&mut self, s: &str) { self.u16(s.len() as u16); self.0.extend_from_slice(s.as_bytes()); }
    /// 칸 열기: 길이 자리를 비워 두고 돌려줌
    fn open(&mut self, tag: u8) -> usize { self.u8(tag); self.u32(0); self.0.len() }
    fn close(&mut self, at: usize) { let n = (self.0.len() - at) as u32; self.0[at - 4..at].copy_from_slice(&n.to_le_bytes()); }
}

/// 지도 조각: old(이 페이지가 가진 판)과 다른 32 칸 조각들. 크기가 다르면 None(전체)
fn map_tiles(old: &crate::Pgm, new: &crate::Pgm) -> Option<Vec<(usize, usize, usize, usize)>> {
    if old.w != new.w || old.h != new.h {
        return None;
    }
    let (w, h) = (new.w, new.h);
    let (a, b) = (old.px(), new.px());
    let mut out = Vec::new();
    for ty in (0..h).step_by(TILE) {
        let th = TILE.min(h - ty);
        for tx in (0..w).step_by(TILE) {
            let tw = TILE.min(w - tx);
            if (ty..ty + th).any(|y| a[y * w + tx..y * w + tx + tw] != b[y * w + tx..y * w + tx + tw]) {
                out.push((tx, ty, tw, th));
            }
        }
    }
    Some(out)
}

pub struct Now<'a> {
    pub view: Option<(Arc<View>, Arc<Parts>)>,
    pub map: Option<(String, Arc<crate::Pgm>)>,
    pub map_hist: &'a [(String, Arc<crate::Pgm>)],
    pub oct: Option<Arc<OctIdx>>,
    pub pose: Option<(String, Arc<Vec<u8>>)>,
    /// 기억 폴더(청크 점 파일을 읽을 곳)
    pub root: &'a std::path::Path,
    /// 이번 덩어리에 담을 청크 점 바이트(0 = 담지 않음)
    pub pts_budget: usize,
}

/// 위치만 담은 덩어리(바뀌었을 때): 판 · 지도 · 노드를 짜기 전에 먼저 보낸다(무거운 판이 위치를 늦추지 않게)
pub fn pose_frame(c: &mut Client, pose: &Option<(String, Arc<Vec<u8>>)>) -> Option<Vec<u8>> {
    let (v, b) = pose.as_ref()?;
    if *v == c.pose_v {
        return None;
    }
    let mut w = W(vec![0; 4]);
    let at = w.open(7);
    w.0.extend_from_slice(b);
    w.close(at);
    c.pose_v = v.clone();
    let n = (w.0.len() - 4) as u32;
    w.0[..4].copy_from_slice(&n.to_le_bytes());
    Some(w.0)
}

/// 이 페이지가 아직 못 받은 것만 담은 덩어리(없으면 None)
pub fn frame(c: &mut Client, now: &Now) -> Option<Vec<u8>> {
    let mut w = W(vec![0; 4]);
    if let Some((v, p)) = &now.view {
        if v.ver != c.vv {
            let at = w.open(1);
            w.str16(&p.ver);
            w.f64(p.wall);
            w.close(at);
            c.vv = v.ver.clone();
            let ch: Vec<_> = p.head.iter().filter(|(k, h, _)| c.head.get(k) != Some(h)).collect();
            let gone: Vec<_> = c.head.keys().filter(|k| !p.head.iter().any(|(q, _, _)| q == *k)).cloned().collect();
            if !ch.is_empty() || !gone.is_empty() {
                let at = w.open(2);
                w.u8(b'{');
                for (i, (k, _, b)) in ch.iter().enumerate() {
                    if i > 0 {
                        w.u8(b',');
                    }
                    w.0.extend_from_slice(&serde_json::to_vec(k).unwrap_or_default());
                    w.u8(b':');
                    w.0.extend_from_slice(b);
                }
                if !gone.is_empty() {
                    if !ch.is_empty() {
                        w.u8(b',');
                    }
                    w.0.extend_from_slice(b"\"__del\":");
                    w.0.extend_from_slice(&serde_json::to_vec(&gone).unwrap_or_default());
                }
                w.u8(b'}');
                w.close(at);
                c.head = p.head.iter().map(|(k, h, _)| (k.clone(), *h)).collect();
            }
            let oh = fnv(&p.order.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>());
            let changed: Vec<_> = p.objs.iter().filter(|(id, h, _)| c.objs.get(id) != Some(h)).collect();
            if !changed.is_empty() || oh != c.order_h {
                let at = w.open(3);
                w.u32(p.order.len() as u32);
                for id in &p.order {
                    w.u32(*id);
                }
                w.u32(changed.len() as u32);
                for (_, _, b) in &changed {
                    w.u32(b.len() as u32);
                    w.0.extend_from_slice(b);
                }
                w.close(at);
                c.objs = p.objs.iter().map(|(id, h, _)| (*id, *h)).collect();
                c.order_h = oh;
            }
            let (id, n, _) = v.traj_from(usize::MAX);
            let same = id != 0 && id == c.traj.0 && c.traj.1 <= n;
            if !same || n != c.traj.1 {
                let from = if same { c.traj.1 } else { 0 };
                let (_, _, pts) = v.traj_from(from);
                let at = w.open(4);
                w.f64(id as f64);
                w.u32(from as u32);
                w.u32(pts.len() as u32);
                for (t, q) in &pts {
                    w.f64(*t);
                    for x in q {
                        w.f64(*x);
                    }
                }
                w.close(at);
                c.traj = (id, n);
            }
            if let Some((run, _, list)) = &p.sub {
                let reset = c.sub_run != Some(*run);
                if reset {
                    c.sub.clear();
                    c.sub_pts.clear();
                }
                let ch: Vec<&SubChunk> = list.iter().filter(|s| c.sub.get(&s.id) != Some(s)).collect();
                let gone: Vec<u32> = c.sub.keys().filter(|id| !list.iter().any(|s| s.id == **id)).copied().collect();
                if reset || !ch.is_empty() || !gone.is_empty() {
                    let at = w.open(8);
                    w.f64(*run as f64);
                    w.u8(reset as u8);
                    w.u32(ch.len() as u32);
                    for s in &ch {
                        w.u32(s.id);
                        w.u32(s.ver);
                        w.u32(s.n);
                        for q in s.pose {
                            w.0.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                    w.u32(gone.len() as u32);
                    for id in &gone {
                        w.u32(*id);
                        c.sub.remove(id);
                        c.sub_pts.remove(id);
                    }
                    w.close(at);
                    for s in ch {
                        c.sub.insert(s.id, s.clone());
                    }
                    c.sub_run = Some(*run);
                }
            }
        }
    }
    if let Some((ver, cur)) = &now.map {
        if *ver != c.map_v {
            let old = now.map_hist.iter().find(|(v, _)| *v == c.map_v).map(|(_, p)| p.clone());
            let tiles = old.and_then(|o| map_tiles(&o, cur));
            let full = tiles.is_none();
            let tiles = tiles.unwrap_or_else(|| vec![(0, 0, cur.w, cur.h)]);
            let at = w.open(5);
            w.str16(ver);
            w.u32(cur.w as u32);
            w.u32(cur.h as u32);
            w.u8(full as u8);
            w.u32(tiles.len() as u32);
            let px = cur.px();
            for (x0, y0, tw, th) in tiles {
                w.u32(x0 as u32);
                w.u32(y0 as u32);
                w.u32(tw as u32);
                w.u32(th as u32);
                for y in y0..y0 + th {
                    w.0.extend_from_slice(&px[y * cur.w + x0..y * cur.w + x0 + tw]);
                }
            }
            w.close(at);
            c.map_v = ver.clone();
        }
    }
    if let Some(o) = &now.oct {
        let reset = c.oct_run != Some(o.run);
        if reset {
            c.oct.clear();
        }
        let ch: Vec<_> = o.nodes.iter().filter(|n| c.oct.get(&(n[0], n[1], n[2], n[3])) != Some(&n[5])).collect();
        if reset || !ch.is_empty() {
            let at = w.open(6);
            w.f64(o.run as f64);
            w.u8(reset as u8);
            w.u32(ch.len() as u32);
            for n in &ch {
                w.u8(n[0] as u8);
                w.i32(n[1] as i32);
                w.i32(n[2] as i32);
                w.i32(n[3] as i32);
                w.u32(n[4] as u32);
                w.u32(n[5] as u32);
                c.oct.insert((n[0], n[1], n[2], n[3]), n[5]);
            }
            w.close(at);
            c.oct_run = Some(o.run);
        }
    }
    // 청크 점: 이 페이지가 아는(칸 8) 청크 중 그 판의 점을 아직 못 받은 것, 새 청크(큰 id)부터 예산까지
    c.pts_pending = false;
    if let Some((_, p)) = &now.view {
        if let Some((_, dir, list)) = &p.sub {
            let mut want: Vec<&SubChunk> = list.iter().filter(|s| s.n > 0 && c.sub.contains_key(&s.id) && c.sub_pts.get(&s.id) != Some(&s.ver)).collect();
            want.sort_unstable_by(|a, b| b.id.cmp(&a.id));
            if !want.is_empty() {
                let mut sent = 0usize;
                let mut at = None;
                let mut k = 0u32;
                for s in want {
                    if sent >= now.pts_budget {
                        c.pts_pending = true;
                        break;
                    }
                    let Some(b) = crate::read_file(&now.root.join(dir).join(format!("c{}.bin", s.id))) else { continue };
                    let n = b.get(..4).map_or(0, |h| u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize);
                    if b.len() != 4 + 9 * n {
                        continue; // 쓰는 중(다음 판에서)
                    }
                    if at.is_none() {
                        let a = w.open(9);
                        w.u32(0);
                        at = Some(a);
                    }
                    w.u32(s.id);
                    w.u32(s.ver);
                    w.u32(b.len() as u32);
                    w.0.extend_from_slice(&b);
                    c.sub_pts.insert(s.id, s.ver);
                    sent += b.len();
                    k += 1;
                }
                if let Some(a) = at {
                    w.0[a..a + 4].copy_from_slice(&k.to_le_bytes());
                    w.close(a);
                }
            }
        }
    }
    if let Some((v, b)) = &now.pose {
        if *v != c.pose_v {
            let at = w.open(7);
            w.0.extend_from_slice(b);
            w.close(at);
            c.pose_v = v.clone();
        }
    }
    if w.0.len() == 4 {
        return None;
    }
    let n = (w.0.len() - 4) as u32;
    w.0[..4].copy_from_slice(&n.to_le_bytes());
    Some(w.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn objects_only_changed() {
        let mk = |ver: u64, a: i64| {
            let head = serde_json::json!({"stamp": 1.0, "objects": [{"id": 1, "a": a}, {"id": 2, "a": 0}], "graph": {"edges": []}});
            Arc::new(View::live(ver, head, crate::live::TrajLog::new(), 0))
        };
        let cache = Mutex::new(None);
        let mut c = Client::default();
        let v1 = mk(1, 0);
        let p1 = parts(&cache, &v1).unwrap();
        let f1 = frame(&mut c, &Now { view: Some((v1.clone(), p1)), map: None, map_hist: &[], oct: None, pose: None, root: std::path::Path::new("."), pts_budget: 0 }).unwrap();
        let v2 = mk(2, 5);
        let p2 = parts(&cache, &v2).unwrap();
        let f2 = frame(&mut c, &Now { view: Some((v2.clone(), p2.clone())), map: None, map_hist: &[], oct: None, pose: None, root: std::path::Path::new("."), pts_budget: 0 }).unwrap();
        assert!(f2.len() < f1.len());
        // 물체 칸: 바뀐 것 하나만
        let s = String::from_utf8_lossy(&f2);
        assert!(s.contains("\"a\":5") && !s.contains("\"a\":0"));
        assert!(frame(&mut c, &Now { view: Some((v2, p2)), map: None, map_hist: &[], oct: None, pose: None, root: std::path::Path::new("."), pts_budget: 0 }).is_none());
    }

    #[test]
    fn submap_sends_changed_chunks_only() {
        let e = |id: u32, px: f64| serde_json::json!([id, 1, 10, 1, 0, 0, 0, 1, 0, 0, 0, 1, px, 0, 0]);
        let mk = |ver: u64, px: f64| {
            let head = serde_json::json!({"stamp": 1.0, "objects": [], "graph": {"edges": []},
                "scene_points": {"run": 1, "submap": {"run": 9, "dir": "chunks/9", "list": [e(0, 0.0), e(1, px)]}}});
            Arc::new(View::live(ver, head, crate::live::TrajLog::new(), 0))
        };
        let cache = Mutex::new(None);
        let mut c = Client::default();
        let v1 = mk(1, 0.0);
        let p1 = parts(&cache, &v1).unwrap();
        frame(&mut c, &Now { view: Some((v1, p1)), map: None, map_hist: &[], oct: None, pose: None, root: std::path::Path::new("."), pts_budget: 0 }).unwrap();
        assert_eq!(c.sub.len(), 2);
        // 청크 1 만 자세가 바뀜: 칸 8 에 하나, 머리(scene_points)는 그대로라 칸 2 없음
        let v2 = mk(2, 0.5);
        let p2 = parts(&cache, &v2).unwrap();
        let f = frame(&mut c, &Now { view: Some((v2, p2)), map: None, map_hist: &[], oct: None, pose: None, root: std::path::Path::new("."), pts_budget: 0 }).unwrap();
        let mut o = 4;
        let mut tags = vec![];
        while o + 5 <= f.len() {
            let len = u32::from_le_bytes(f[o + 1..o + 5].try_into().unwrap()) as usize;
            tags.push(f[o]);
            if f[o] == 8 {
                assert_eq!(u32::from_le_bytes(f[o + 14..o + 18].try_into().unwrap()), 1);
            }
            o += 5 + len;
        }
        assert!(tags.contains(&8) && !tags.contains(&2), "{tags:?}");
    }
}
