//! sgview — scene-graph viewer server (Rust). Replaces the Python/viser viewer.
//!
//! Reads the folder scenemap writes (`view.json`, `map.pgm` + `map.yaml`, `objects/O<id>_*`) and serves it to a three.js page.
//! Everything the page needs comes from `view.json`, which already carries
//! objects, rooms, the compact graph, robot pose and grid info. Walls as 2D lines / numbers come from `walls.rs`.
//!
//!   sgview <memory_dir> [--port 8080] [--bind 0.0.0.0] [--search-model <dir with text_model*.onnx + tokenizer.json>]
//!
//! GET /                      the page
//! GET /api/view?v=<ver>      {"v": ver, "map_v": ver, "view": <view.json>} or {"unchanged": true} (poll; ver = mtime+size)
//! GET /api/map[?since=<ver>] PGM pixels (row 0 = max y); with since only the changed rows (X-Y0/X-Y1/X-Full)
//! GET /api/level             the camera is level now: turn the map so the current camera pose is horizontal
//! GET /api/reset             ask the running pipeline to start over
//! GET /api/save              write the in-memory memory dir to disk now (`ondevice --serve`)
//!                            (one program: common::control flags; a separate viewer leaves level.req / reset.req in the dir)
//! GET /api/walls?x=&y=&yaw=  {"segments": [[ax,ay,bx,by]..], "state": [56 floats], "pose": [x,y,yaw]}
//! GET /api/depth?path=<rel>  16-bit grey PNG → u16 little-endian pixels (X-W / X-H)
//! GET /file/<relative path>  any file below the memory dir (PNG crops, PLY points)
//! GET /api/search?q=<text>   objects ranked against the text (see search.rs); optional k (20) and min score (0.2)
//! GET /api/files?p=a,b,..   many files below the memory dir in one reply (see pack_files)
//! GET /stream               SSE: vv (view version) / fpose (pose.json), pushed when they change
//! GET /bstream              binary deltas per page (changed objects, new trajectory points, changed map tiles, new octree nodes, pose; see delta.rs)
//!
//! In one program (`ondevice --serve`) the view comes from [`live`] (structure handed over by the pipeline, JSON built on
//! request) and every other file from [`memfs`]; a separate viewer reads the folder.
mod delta;
pub mod live;
pub mod memfs;
mod search;
#[cfg(feature = "search")]
mod textnet;
mod walls;

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::UNIX_EPOCH;

const INDEX_HTML: &str = include_str!("../assets/index.html");
const THREE_JS: &[u8] = include_bytes!("../assets/three.min.js");
const ORBIT_JS: &[u8] = include_bytes!("../assets/OrbitControls.js");
const GLTF_JS: &[u8] = include_bytes!("../assets/GLTFLoader.js");
include!(concat!(env!("OUT_DIR"), "/robot_assets.rs"));   // ROBOT_FILES: the URDF model (GLB meshes + robot.json)

const WALL_STATE_LEN: usize = 56;

/// Non-interlaced 8/16-bit grey or 8-bit RGB(A) PNG → (w, h, channels, bytes per sample, raw samples big-endian as stored).
fn decode_png(b: &[u8]) -> Option<(usize, usize, usize, usize, Vec<u8>)> {
    use std::io::Read;
    if b.len() < 33 || &b[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let (mut w, mut h, mut depth, mut ctype, mut interlace) = (0usize, 0usize, 0u8, 0u8, 0u8);
    let mut idat = Vec::new();
    let mut i = 8;
    while i + 8 <= b.len() {
        let len = u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        let ty = &b[i + 4..i + 8];
        let data = b.get(i + 8..i + 8 + len)?;
        match ty {
            b"IHDR" => {
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                depth = data[8];
                ctype = data[9];
                interlace = data[12];
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        i += 12 + len;
    }
    let ch = match ctype {
        0 => 1,
        2 => 3,
        6 => 4,
        _ => return None,
    };
    if interlace != 0 || (depth != 8 && depth != 16) || w == 0 || h == 0 {
        return None;
    }
    let bps = (depth / 8) as usize;
    let bpp = ch * bps;
    let stride = w * bpp;
    let mut raw = Vec::with_capacity((stride + 1) * h);
    flate2::read::ZlibDecoder::new(&idat[..]).read_to_end(&mut raw).ok()?;
    if raw.len() < (stride + 1) * h {
        return None;
    }
    let mut out = vec![0u8; stride * h];
    for y in 0..h {
        let f = raw[y * (stride + 1)];
        let line = &raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        for x in 0..stride {
            let a = if x >= bpp { out[y * stride + x - bpp] as i32 } else { 0 };
            let up = if y > 0 { out[(y - 1) * stride + x] as i32 } else { 0 };
            let c = if x >= bpp && y > 0 { out[(y - 1) * stride + x - bpp] as i32 } else { 0 };
            let p = match f {
                0 => 0,
                1 => a,
                2 => up,
                3 => (a + up) / 2,
                4 => {
                    let pp = a + up - c;
                    let (pa, pb, pc) = ((pp - a).abs(), (pp - up).abs(), (pp - c).abs());
                    if pa <= pb && pa <= pc { a } else if pb <= pc { up } else { c }
                }
                _ => return None,
            };
            out[y * stride + x] = (line[x] as i32 + p) as u8;
        }
    }
    Some((w, h, ch, bps, out))
}

/// Occupancy grid in scenemap layout (row 0 = min y; −1 unknown, 0..100 occupied %), plus its wall segments.
struct WallMap {
    ver: String,
    ig_key: String,
    w: i32,
    h: i32,
    res: f64,
    ox: f64,
    oy: f64,
    cells: Vec<i8>,
    segs: Vec<f64>, // ax ay bx by …
    theta: f64,     // wall direction [rad] (slam maps are in the start-pose frame, so walls can be tilted); ~0 when axis-aligned
}

/// Recent map.pgm versions (newest last): a client that has version V gets only the rows that changed since V.
struct MapHist {
    vers: Vec<(String, Arc<Pgm>)>,
}
const MAP_HIST: usize = 8;

struct State {
    dir: PathBuf,
    walls: Mutex<Option<Arc<WallMap>>>,
    view: Mutex<Option<Arc<live::View>>>, // view.json read from the folder, by version (a separate viewer)
    maps: Mutex<MapHist>,
    streams: Mutex<Vec<StreamClient>>,    // open /stream pages (one thread pushes to all of them)
    parts: Mutex<Option<Arc<delta::Parts>>>, // the current view split into objects + head (once per version, shared by every /bstream page)
    octidx: Mutex<Option<Arc<delta::OctIdx>>>, // octree node list of the current view (read once per version)
    // --search-model (dir, --vocab): the encoder and tokenizer are loaded on the first search, so a viewer nobody searches holds neither
    #[cfg(feature = "search")]
    text_src: Option<(String, Option<String>)>,
    #[cfg(feature = "search")]
    text_enc: std::sync::OnceLock<Option<search::TextEnc>>,
}

/// An open /stream page and the versions it was last sent
struct StreamClient {
    s: TcpStream,
    vv: String,
    pv: String,
    bin: Option<delta::Client>, // /bstream: what this page has (sent only what is new for it)
}

/// 파일 바이트: memmap 이 같은 프로세스에서 걸어 둔 것(memfs)이 있으면 그것, 아니면 디스크
/// (메모리 바이트는 복사 없이 나눠 씀)
fn read_file(p: &Path) -> Option<Arc<Vec<u8>>> {
    match memfs::get(p) {
        Some((_, b)) => Some(b),
        None => fs::read(p).ok().map(Arc::new),
    }
}

/// 나눠 쓰는 파일 바이트(응답 몸)
struct Shared(Arc<Vec<u8>>);
impl AsRef<[u8]> for Shared {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

fn read_text(p: &Path) -> Option<String> {
    String::from_utf8(Arc::try_unwrap(read_file(p)?).unwrap_or_else(|a| (*a).clone())).ok()
}

fn file_ver(p: &Path) -> Option<String> {
    if let Some((v, _)) = memfs::get(p) {
        return Some(v);
    }
    let m = fs::metadata(p).ok()?;
    let t = m.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("{}-{}", t, m.len()))
}

/// map.pgm pixels: the file bytes shared as they are (no copy), pixels from `off`
pub struct Pgm {
    w: usize,
    h: usize,
    bytes: Arc<Vec<u8>>,
    off: usize,
}

impl Pgm {
    fn px(&self) -> &[u8] {
        &self.bytes[self.off..self.off + self.w * self.h]
    }
}

fn read_pgm(p: &Path) -> Option<Pgm> {
    let b = read_file(p)?;
    let mut i = 0;
    let mut tok = Vec::new();
    while tok.len() < 4 {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if s == i {
            return None;
        }
        tok.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    i += 1; // one whitespace after maxval
    if tok[0] != "P5" || tok[3] != "255" {
        return None;
    }
    let (w, h): (usize, usize) = (tok[1].parse().ok()?, tok[2].parse().ok()?);
    if b.len() < i + w * h {
        return None;
    }
    Some(Pgm { w, h, bytes: b, off: i })
}

/// resolution and origin (x, y) of a ROS map.yaml
fn read_yaml(p: &Path) -> (f64, f64, f64) {
    let (mut res, mut ox, mut oy) = (0.05, 0.0, 0.0);
    if let Some(s) = read_text(p) {
        for line in s.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(v) = line.strip_prefix("resolution:") {
                res = v.trim().parse().unwrap_or(res);
            } else if let Some(v) = line.strip_prefix("origin:") {
                let n: Vec<f64> = v.trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() >= 2 {
                    ox = n[0];
                    oy = n[1];
                }
            }
        }
    }
    (res, ox, oy)
}

/// The current view: the one the pipeline in this process handed over, else view.json from the folder
/// (cached by version; `None` while the file is missing or mid-write).
fn load_view(st: &State) -> Option<Arc<live::View>> {
    if let Some(v) = live::current() {
        return Some(v);
    }
    let vp = st.dir.join("view.json");
    let ver = file_ver(&vp)?;
    let mut c = st.view.lock().unwrap();
    if c.as_ref().is_none_or(|v| v.ver != ver) {
        let b = read_file(&vp)?;
        let t = b.trim_ascii();
        if !(t.starts_with(b"{") && t.ends_with(b"}")) {
            return None;
        }
        *c = Some(Arc::new(live::View::file(ver, b)));
    }
    c.clone()
}

/// Footprints (x0 y0 x1 y1) of free-standing objects from the view: bottom below 0.4 m, not room-sized, not gone, +0.1 m.
/// Their occupied cells are furniture (sofa, table …), not walls. Wall-mounted things (frames, lamps) start higher and are left alone.
fn furniture_rects(st: &State) -> Vec<f64> {
    match load_view(st).as_ref().and_then(|v| v.doc()) {
        Some(v) => furniture_rects_of(v),
        None => Vec::new(),
    }
}

fn furniture_rects_of(v: &serde_json::Value) -> Vec<f64> {
    let mut out = Vec::new();
    for o in v["objects"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        if o["state"] == "gone" {
            continue;
        }
        let (p, e) = (&o["pos"], &o["extent"]);
        let g = |a: &serde_json::Value, i: usize| a[i].as_f64().unwrap_or(0.0);
        let (hx, hy, hz) = (g(e, 0) / 2.0, g(e, 1) / 2.0, g(e, 2) / 2.0);
        if g(p, 2) - hz > 0.4 || hx * 2.0 > 5.0 || hy * 2.0 > 5.0 {
            continue;
        }
        out.extend_from_slice(&[g(p, 0) - hx - 0.1, g(p, 1) - hy - 0.1, g(p, 0) + hx + 0.1, g(p, 1) + hy + 0.1]);
    }
    out
}

/// Which robot made the run, for the trajectory label: `robot_footprint.json` next to the memory dir (run dir) or inside it.
/// An explicit `"robot"` field wins; older files only carry the simulator AABB (LIMO + OMX-F ≈ 0.35 × 0.36 × 0.38 m).
/// Unknown → null (the page says "robot").
fn robot_kind(st: &State) -> Option<String> {
    let mut cands = vec![st.dir.join("robot_footprint.json")];
    if let Some(p) = st.dir.parent() {
        cands.push(p.join("robot_footprint.json"));
    }
    for p in cands {
        let v: serde_json::Value = match std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str(&t).ok()) {
            Some(v) => v,
            None => continue,
        };
        if let Some(r) = v["robot"].as_str() {
            return Some(r.to_string());
        }
        let e: Vec<f64> = v["robot_aabb_extent"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default();
        if e.len() == 3 {
            let xy = e[0].max(e[1]);
            if xy < 0.5 && e[2] < 0.6 {
                return Some("limo_omx".into());
            }
        }
        return None;
    }
    None
}

/// The wall map of the current `map.pgm` (cached by mtime+size; segments are recomputed only when the map file changes).
fn wall_map(st: &State) -> Option<Arc<WallMap>> {
    let pgm_path = st.dir.join("map.pgm");
    let ver = file_ver(&pgm_path)?;
    let ig = furniture_rects(st);
    let ig_key = format!("{:?}", ig);
    let mut g = st.walls.lock().unwrap();
    if let Some(w) = g.as_ref() {
        if w.ver == ver && w.ig_key == ig_key {
            return Some(w.clone());
        }
    }
    let pgm = read_pgm(&pgm_path)?;
    let (res, ox, oy) = read_yaml(&st.dir.join("map.yaml"));
    let (w, h) = (pgm.w, pgm.h);
    // PGM: row 0 = max y, grey ≤ 90 occupied, 205 unknown, else free  →  scenemap layout (row 0 = min y)
    let mut cells = vec![0i8; w * h];
    for y in 0..h {
        let src = &pgm.px()[(h - 1 - y) * w..(h - y) * w];
        let dst = &mut cells[y * w..(y + 1) * w];
        for x in 0..w {
            dst[x] = if src[x] <= 90 { 100 } else if src[x] == 205 { -1 } else { 0 };
        }
    }
    let (buf, theta) = walls::wall_segments(&walls::Grid { cells: &cells, w: w as i32, h: h as i32, res, ox, oy }, &ig);
    let wm = Arc::new(WallMap { ver, ig_key, w: w as i32, h: h as i32, res, ox, oy, cells, segs: buf, theta });
    *g = Some(wm.clone());
    Some(wm)
}

/// Current map pixels (version-cached; history kept for delta updates).
fn current_map(st: &State) -> Option<(String, Arc<Pgm>)> {
    let ver = file_ver(&st.dir.join("map.pgm"))?;
    let mut h = st.maps.lock().unwrap();
    if let Some((v, p)) = h.vers.last() {
        if *v == ver {
            return Some((v.clone(), p.clone()));
        }
    }
    let p = Arc::new(read_pgm(&st.dir.join("map.pgm"))?);
    h.vers.push((ver.clone(), p.clone()));
    if h.vers.len() > MAP_HIST {
        h.vers.remove(0);
    }
    Some((ver, p))
}

/// Rows [y0, y1] (PGM row order) that differ between `old` and `new`, None if the shapes differ or nothing changed.
fn changed_rows(old: &Pgm, new: &Pgm) -> Option<(usize, usize)> {
    if old.w != new.w || old.h != new.h {
        return None;
    }
    let w = new.w;
    let mut first = None;
    let mut last = 0;
    for y in 0..new.h {
        if old.px()[y * w..(y + 1) * w] != new.px()[y * w..(y + 1) * w] {
            if first.is_none() {
                first = Some(y);
            }
            last = y;
        }
    }
    first.map(|f| (f, last))
}

fn json_f64s(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{:.4}", x));
    }
    s.push(']');
    s
}

fn walls_json(st: &State, q: &HashMap<String, String>) -> Option<String> {
    let wm = wall_map(st)?;
    let get = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let pose = [get("x"), get("y"), get("yaw")];
    let out: [f32; WALL_STATE_LEN] = walls::wall_state(&walls::Grid { cells: &wm.cells, w: wm.w, h: wm.h, res: wm.res, ox: wm.ox, oy: wm.oy }, &wm.segs, &pose);
    let segs: Vec<String> = wm.segs.chunks(4).map(|c| json_f64s(c)).collect();
    let state: Vec<String> = out.iter().map(|x| format!("{:.5}", x)).collect();
    Some(format!(
        "{{\"segments\":[{}],\"theta\":{:.6},\"state\":[{}],\"pose\":{},\"map_v\":\"{}\",\"n_sectors\":16,\"k_segments\":8,\"max_range\":4.0}}",
        segs.join(","),
        wm.theta,
        state.join(","),
        json_f64s(&pose),
        wm.ver
    ))
}

fn content_type(p: &str) -> &'static str {
    match p.rsplit('.').next().unwrap_or("") {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "json" => "application/json",
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "ply" | "pgm" => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

/// What the current request on this connection asked for (one connection = one thread, so a thread-local is per request).
#[derive(Default, Clone)]
struct ReqCtx {
    gzip: bool,                  // Accept-Encoding has gzip
    keep: bool,                  // keep the connection open after the reply (HTTP/1.1 default)
    if_none_match: Option<String>,
}
thread_local! { static REQ: std::cell::RefCell<ReqCtx> = std::cell::RefCell::new(ReqCtx::default()); }

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h = (h ^ x as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// gzip of text bodies, cached by content hash: view.json is sent to every client once per version and the page's
/// scripts (three.min.js 603 KB → 150 KB) once per load, so each distinct body is compressed only once.
fn gzip_cached(body: &[u8]) -> Arc<Vec<u8>> {
    static CACHE: Mutex<Vec<(u64, usize, Arc<Vec<u8>>)>> = Mutex::new(Vec::new());
    let key = (fnv(body), body.len());
    if let Some((_, _, z)) = CACHE.lock().unwrap().iter().find(|(h, n, _)| (*h, *n) == key) {
        return z.clone();
    }
    let mut e = flate2::write::GzEncoder::new(Vec::with_capacity(body.len() / 4), flate2::Compression::new(6));
    let _ = e.write_all(body);
    let z = Arc::new(e.finish().unwrap_or_default());
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 16 {
        c.remove(0);
    }
    c.push((key.0, key.1, z.clone()));
    z
}

fn respond(s: &mut TcpStream, code: u16, ctype: &str, extra: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        304 => "Not Modified",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let ctx = REQ.with(|r| r.borrow().clone());
    let texty = ctype.starts_with("application/json") || ctype.starts_with("text/") || ctype.starts_with("application/javascript");
    let z = (code == 200 && ctx.gzip && texty && body.len() > 1024).then(|| gzip_cached(body));
    let (body, enc): (&[u8], &str) = match &z {
        Some(z) => (z, "Content-Encoding: gzip\r\nVary: Accept-Encoding\r\n"),
        None => (body, ""),
    };
    // no-cache (not no-store): the browser may keep a copy but must revalidate; /file/ and the page answer 304 by ETag
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: {}\r\n{}{}\r\n",
        code,
        reason,
        ctype,
        body.len(),
        if ctx.keep { "keep-alive" } else { "close" },
        enc,
        extra
    );
    let mut out = Vec::with_capacity(head.len() + body.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(body);
    let _ = s.write_all(&out);
}

/// 200 with an ETag, or 304 when the browser already has this version
fn respond_tagged<B: AsRef<[u8]>>(s: &mut TcpStream, ctype: &str, etag: &str, body: impl FnOnce() -> Option<B>) {
    let tag = format!("\"{}\"", etag);
    if REQ.with(|r| r.borrow().if_none_match.as_deref() == Some(tag.as_str())) {
        return respond(s, 304, ctype, &format!("ETag: {}\r\n", tag), b"");
    }
    match body() {
        Some(b) => respond(s, 200, ctype, &format!("ETag: {}\r\n", tag), b.as_ref()),
        None => respond(s, 404, "text/plain", "", b"not found"),
    }
}

/// Built-in page files: tag = content hash (computed once per asset)
fn respond_static(s: &mut TcpStream, ctype: &str, body: &'static [u8]) {
    static TAGS: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());
    let id = body.as_ptr() as usize;
    let tag = {
        let mut t = TAGS.lock().unwrap();
        match t.iter().find(|(p, _)| *p == id) {
            Some((_, g)) => g.clone(),
            None => {
                let g = format!("s{:x}-{}", fnv(body), body.len());
                t.push((id, g.clone()));
                g
            }
        }
    };
    respond_tagged(s, ctype, &tag, || Some(std::borrow::Cow::Borrowed(body)))
}

/// `/api/files` body: per file `u32 name length, name, u32 data length (u32::MAX = missing), data` (little-endian), in request order.
fn pack_files(root: &Path, rels: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in rels {
        out.extend_from_slice(&(r.len() as u32).to_le_bytes());
        out.extend_from_slice(r.as_bytes());
        match safe_join(root, r).and_then(|p| read_file(&p)) {
            Some(b) => {
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(&b);
            }
            None => out.extend_from_slice(&u32::MAX.to_le_bytes()),
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_query(q: &str) -> HashMap<String, String> {
    q.split('&')
        .filter_map(|kv| {
            let mut it = kv.splitn(2, '=');
            Some((it.next()?.to_string(), percent_decode(it.next().unwrap_or(""))))
        })
        .collect()
}

/// `rel` below `root`, no `..`, no absolute parts
fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    let mut p = root.to_path_buf();
    for c in Path::new(rel).components() {
        match c {
            Component::Normal(x) => p.push(x),
            _ => return None,
        }
    }
    Some(p)
}

/// One connection: HTTP/1.1 keep-alive, so the page's requests (view on change, batched point files, object files) at
/// load reuse one socket and one thread instead of a TCP handshake + thread spawn per request. Idle connections close after 5 s (a pool thread serves one connection at a time).
fn handle(s: TcpStream, st: Arc<State>) {
    let mut s = s;
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let mut rd = BufReader::new(s.try_clone().unwrap());
    loop {
        let mut line = String::new();
        if rd.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut ctx = ReqCtx { keep: line.trim_end().ends_with("HTTP/1.1"), ..Default::default() };
        loop {
            let mut h = String::new();
            if rd.read_line(&mut h).unwrap_or(0) <= 2 {
                break;
            }
            let Some((k, v)) = h.split_once(':') else { continue };
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            match k.as_str() {
                "accept-encoding" => ctx.gzip = v.split(',').any(|e| e.trim().split(';').next() == Some("gzip")),
                "if-none-match" => ctx.if_none_match = Some(v.to_string()),
                "connection" => {
                    let v = v.to_ascii_lowercase();
                    if v.contains("close") {
                        ctx.keep = false;
                    } else if v.contains("keep-alive") {
                        ctx.keep = true;
                    }
                }
                _ => {}
            }
        }
        REQ.with(|r| *r.borrow_mut() = ctx);
        handle_one(&mut s, &st, &line);
        if !REQ.with(|r| r.borrow().keep) {
            return;
        }
    }
}

fn handle_one(s: &mut TcpStream, st: &Arc<State>, line: &str) {
    let mut s = s;
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    if method != "GET" {
        return respond(&mut s, 400, "text/plain", "", b"GET only");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let q = parse_query(query);
    match path {
        "/" | "/index.html" => respond_static(&mut s, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        "/three.min.js" => respond_static(&mut s, "application/javascript", THREE_JS),
        "/GLTFLoader.js" => respond_static(&mut s, "application/javascript", GLTF_JS),
        _ if path.starts_with("/robot/") => match ROBOT_FILES.iter().find(|(n, _)| *n == &path["/robot/".len()..]) {
            Some((n, b)) => respond(&mut s, 200, if n.ends_with(".json") { "application/json" } else { "model/gltf-binary" }, "", b),
            None if &path["/robot/".len()..] == "robot.json" => respond(&mut s, 200, "application/json", "", b"{}"),   // no model in this build: the page keeps its box stand-in
            None => respond(&mut s, 404, "text/plain", "", b"no such model file"),
        },
        "/OrbitControls.js" => respond_static(&mut s, "application/javascript", ORBIT_JS),
        "/favicon.ico" => respond(&mut s, 204, "image/x-icon", "", b""),
        // Buttons: in one program (ondevice --serve) a shared flag the pipeline / tracker read every frame;
        // a separate viewer leaves a request file in the memory dir (slam --reset-file, memmap level.req)
        "/api/reset" => signal(&mut s, &st.dir, "reset.req", common::control::request_reset),
        // Level button: the camera is held level now; the pipeline takes the current camera pose as horizontal and re-stacks the map with that rotation
        "/api/level" => signal(&mut s, &st.dir, "level.req", common::control::request_level),
        // Save button: write the in-memory memory dir to disk now (one program only)
        "/api/save" => {
            if common::control::wired() {
                common::control::request_save();
                respond(&mut s, 200, "application/json", "", b"{\"ok\":true}")
            } else {
                respond(&mut s, 200, "application/json", "", b"{\"ok\":false,\"why\":\"the folder is on disk already\"}")
            }
        }
        "/api/search" => {
            let Some(v) = load_view(&st) else { return respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}") };
            let Some(view) = v.doc() else { return respond(&mut s, 500, "text/plain", "", b"view.json") };
            #[cfg(feature = "search")]
            let te = st.text_enc.get_or_init(|| st.text_src.as_ref().and_then(|(m, vocab)| search::load_encoder(m, vocab.as_deref())));
            #[cfg(feature = "search")]
            let enc = |t: &str| te.as_ref().and_then(|e| e.encode(t).map_err(|e| eprintln!("search: {e}")).ok());
            #[cfg(not(feature = "search"))]
            let enc = |_: &str| None;
            let k = q.get("k").and_then(|v| v.parse().ok()).unwrap_or(20);
            let min = q.get("min").and_then(|v| v.parse().ok()).unwrap_or(0.2);
            let body = search::search(&st.dir, view, q.get("q").map_or("", |v| v.as_str()), &enc, k, min).to_string();
            respond(&mut s, 200, "application/json", "", body.as_bytes())
        }
        // ?metrics=1 page: frames drawn and pose → screen latency every 2 s, logged here (measuring on the board screen)
        "/api/metric" => {
            let g = |k: &str| q.get(k).map_or("-", |v| v.as_str());
            // MemAvailable of this machine (board): run boundaries are logged with it to tell memory pressure from viewer work
            let avail = fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).map_or("-".to_string(), |kb| (kb / 1024).to_string());
            if let Some(e) = q.get("event") {
                eprintln!("viewer event: {e}, js heap {} MB, MemAvailable {avail} MB", g("jsmb"));
                return respond(&mut s, 204, "text/plain", "", b"");
            }
            eprintln!("viewer: {} fps (render {} ms), pose→screen p50 {} p95 {} ms (n {}), map→screen p50 {} p95 {} ms (n {}), view {} × fetch {} ms + apply {} ms, {} KB, max frame gap {} ms, long tasks {} (max {} ms), js heap {} MB, MemAvailable {} MB", g("fps"), g("rms"), g("lat50"), g("lat95"), g("n"), g("vlat50"), g("vlat95"), g("vn"), g("nv"), g("fms"), g("ams"), g("kb"), g("gap"), g("lt"), g("ltmax"), g("jsmb"), avail);
            respond(&mut s, 204, "text/plain", "", b"")
        }
        "/api/mode" =>respond(&mut s, 200, "application/json", "", b"{\"live\":false}"),
        "/api/robot" => {
            let body = match robot_kind(&st) {
                Some(r) => format!("{{\"robot\":{}}}", serde_json::Value::String(r)),
                None => "{\"robot\":null}".to_string(),
            };
            respond(&mut s, 200, "application/json", "", body.as_bytes())
        }
        "/bstream" => {
            REQ.with(|r| r.borrow_mut().keep = false);
            if let Ok(c) = s.try_clone() {
                open_bstream(c, st);
            }
        }
        "/stream" => {
            REQ.with(|r| r.borrow_mut().keep = false); // the event stream owns this connection until the page goes away
            if let Ok(c) = s.try_clone() {
                open_stream(c, st);
            }
        }
        "/api/view" => {
            let Some(v) = load_view(&st) else { return respond(&mut s, 200, "application/json", "", b"{\"waiting\":true}") }; // none yet or mid-write: next poll
            if q.get("v").is_some_and(|x| *x == v.ver) {
                return respond(&mut s, 200, "application/json", "", b"{\"unchanged\":true}");
            }
            let map_v = file_ver(&st.dir.join("map.pgm")).unwrap_or_default();
            // tr/tf: the page already has the first tf trajectory points of log tr → only the appended ones
            let delta = q.get("tr").and_then(|t| t.parse().ok()).zip(q.get("tf").and_then(|t| t.parse().ok())).and_then(|(tr, tf)| v.api_delta(&map_v, tr, tf));
            match delta {
                Some(b) => respond(&mut s, 200, "application/json", "", &b),
                None => respond(&mut s, 200, "application/json", "", &v.api_body(&map_v)),
            }
        }
        // full map, or with ?since=<map_v> only the rows changed since that version (X-Y0..X-Y1, PGM row order; X-Full: 0)
        "/api/map" => match current_map(&st) {
            Some((ver, cur)) => {
                let mut y0 = 0;
                let mut y1 = cur.h.saturating_sub(1);
                let mut full = 1;
                if let Some(since) = q.get("since") {
                    let old = st.maps.lock().unwrap().vers.iter().find(|(v, _)| v == since).map(|(_, p)| p.clone());
                    if let Some(old) = old {
                        match changed_rows(&old, &cur) {
                            Some((a, b)) => { y0 = a; y1 = b; full = 0; }
                            None if old.w == cur.w && old.h == cur.h => { full = 0; y0 = 1; y1 = 0; } // identical pixels
                            None => {}
                        }
                    }
                }
                let hdr = format!("X-W: {}\r\nX-H: {}\r\nX-Y0: {}\r\nX-Y1: {}\r\nX-Full: {}\r\nX-Ver: {}\r\n", cur.w, cur.h, y0, y1, full, ver);
                let body: &[u8] = if y0 > y1 { &[] } else { &cur.px()[y0 * cur.w..(y1 + 1) * cur.w] };
                respond(&mut s, 200, "application/octet-stream", &hdr, body)
            }
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        // 16-bit grey PNG (depth in mm) → little-endian u16 pixels; the page can't read 16-bit PNGs from a canvas
        "/api/depth" => {
            let rel = q.get("path").map(|s| s.as_str()).unwrap_or("");
            match safe_join(&st.dir, rel).and_then(|p| memfs::read(&p)).and_then(|b| decode_png(&b)) {
                Some((w, h, 1, bps, px)) => {
                    let mut out = Vec::with_capacity(w * h * 2);
                    for i in 0..w * h {
                        let v = if bps == 2 { u16::from_be_bytes([px[2 * i], px[2 * i + 1]]) } else { px[i] as u16 };
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    respond(&mut s, 200, "application/octet-stream", &format!("X-W: {}\r\nX-H: {}\r\n", w, h), &out)
                }
                _ => respond(&mut s, 404, "text/plain", "", b"not a grey png"),
            }
        }
        // many files in one reply (first load: dozens of point nodes): see pack_files
        "/api/files" => {
            let rels: Vec<&str> = q.get("p").map(|s| s.split(',').filter(|r| !r.is_empty()).take(256).collect()).unwrap_or_default();
            let vers: Vec<String> = rels.iter().map(|r| safe_join(&st.dir, r).and_then(|p| file_ver(&p)).unwrap_or_default()).collect();
            let tag = format!("b{:x}", fnv(format!("{:?}{:?}", rels, vers).as_bytes()));
            let dir = st.dir.clone();
            respond_tagged(&mut s, "application/octet-stream", &tag, || Some(std::borrow::Cow::<[u8]>::Owned(pack_files(&dir, &rels))))
        }
        "/api/walls" => match walls_json(&st, &q) {
            Some(j) => respond(&mut s, 200, "application/json", "", j.as_bytes()),
            None => respond(&mut s, 404, "text/plain", "", b"no map"),
        },
        _ if path.starts_with("/file/") => {
            let rel = &path["/file/".len()..];
            // the handed-over view (one program): its bytes, built once per version
            if rel == "view.json" {
                if let Some(v) = live::current() {
                    return respond_tagged(&mut s, "application/json", &v.ver.clone(), || Some(Shared(v.bytes())));
                }
            }
            // ETag = mtime + size: a page reload (or a second viewer) revalidates the map/objects it already has with 304
            match safe_join(&st.dir, rel).and_then(|p| file_ver(&p).map(|v| (p, v))) {
                Some((p, ver)) => respond_tagged(&mut s, content_type(rel), &ver, || read_file(&p).map(Shared)),
                None => respond(&mut s, 404, "text/plain", "", b"not found"),
            }
        }
        _ => respond(&mut s, 404, "text/plain", "", b"not found"),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// /stream: server-sent events pushed on change. One thread serves every open page: it sleeps until a file or the view
// changes (memfs::bump: the pipeline in this process publishes, or a 50 ms folder check for a separate viewer), then
// sends each page what is new for it. An idle page gets a keepalive comment every 10 s.
// ---------------------------------------------------------------------------------------------------------------------
fn sse(event: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, data)
}

/// Button press: shared flag in one program, else a request file in the memory dir
fn signal(s: &mut TcpStream, dir: &Path, req: &str, flag: fn()) {
    if common::control::wired() {
        flag();
        return respond(s, 200, "application/json", "", b"{\"ok\":true}");
    }
    match std::fs::write(dir.join(req), b"") {
        Ok(()) => respond(s, 200, "application/json", "", b"{\"ok\":true}"),
        Err(e) => respond(s, 500, "text/plain", "", e.to_string().as_bytes()),
    }
}

fn open_stream(mut s: TcpStream, st: &Arc<State>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = s.set_nodelay(true);
    let _ = s.set_write_timeout(Some(Duration::from_secs(5))); // a stuck page is dropped instead of holding the others
    st.streams.lock().unwrap().push(StreamClient { s, vv: String::new(), pv: String::new(), bin: None });
    memfs::bump(); // the pusher sends the new page what it has now
}

/// Binary delta stream: no length, the body runs until the page goes away (read by the page's worker with fetch)
fn open_bstream(mut s: TcpStream, st: &Arc<State>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nCache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n";
    if s.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = s.set_nodelay(true);
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
    st.streams.lock().unwrap().push(StreamClient { s, vv: String::new(), pv: String::new(), bin: Some(delta::Client::default()) });
    memfs::bump();
}

/// The current view version (the handed-over view, else view.json's)
fn view_ver(st: &State) -> Option<String> {
    match live::current() {
        Some(v) => Some(v.ver.clone()),
        None => file_ver(&st.dir.join("view.json")),
    }
}

fn stream_pusher(st: Arc<State>) {
    let pp = st.dir.join("pose.json");
    let mut seen = 0u64;
    let mut quiet = std::time::Instant::now();
    loop {
        // in one program every change bumps; a separate viewer also looks at the folder every 50 ms
        let wait = if memfs::enabled() { Duration::from_secs(10) } else { Duration::from_millis(50) };
        seen = memfs::wait_change(seen, wait);
        let mut cl = st.streams.lock().unwrap();
        if cl.is_empty() {
            continue;
        }
        let vv = view_ver(&st);
        let pose = file_ver(&pp).and_then(|v| {
            let t = read_file(&pp)?;
            let t = std::str::from_utf8(t.trim_ascii()).ok()?;
            // a half-written file is skipped; the next look sees it again
            (t.starts_with('{') && t.ends_with('}')).then(|| (v, sse("fpose", &t.replace('\n', " "))))
        });
        let keep = quiet.elapsed() > Duration::from_secs(10);
        let mut any = false;
        // binary pages: the camera pose goes out first on its own (the view split / map / octree list below can take
        // tens of ms when a new view arrives), then the view split once, the map and octree list read once, then a frame per page with only what it lacks
        if cl.iter().any(|c| c.bin.is_some()) {
            let pose = file_ver(&pp).and_then(|v| {
                let t = read_file(&pp)?;
                let b = t.trim_ascii();
                (b.starts_with(b"{") && b.ends_with(b"}")).then(|| (v, Arc::new(b.to_vec())))
            });
            cl.retain_mut(|c| match c.bin.as_mut().and_then(|b| delta::pose_frame(b, &pose)) {
                Some(f) => c.s.write_all(&f).is_ok(),
                None => true,
            });
        }
        let now = cl.iter().any(|c| c.bin.is_some()).then(|| {
            let view = load_view(&st).and_then(|v| delta::parts(&st.parts, &v).map(|p| (v, p)));
            let map = current_map(&st);
            let oct = view.as_ref().and_then(|(_, p)| p.oct_ref()).and_then(|(run, path, ver)| delta::oct_idx(&st.octidx, &st.dir, run, &path, ver));
            let pose = file_ver(&pp).and_then(|v| {
                let t = read_file(&pp)?;
                let b = t.trim_ascii();
                (b.starts_with(b"{") && b.ends_with(b"}")).then(|| (v, Arc::new(b.to_vec())))
            });
            (view, map, oct, pose)
        });
        let hist = now.as_ref().map(|_| st.maps.lock().unwrap().vers.clone()).unwrap_or_default();
        cl.retain_mut(|c| {
            if let Some(b) = c.bin.as_mut() {
                let (view, map, oct, pose) = now.as_ref().unwrap();
                let n = delta::Now { view: view.clone(), map: map.clone(), map_hist: &hist, oct: oct.clone(), pose: pose.clone() };
                let f = delta::frame(b, &n).or_else(|| keep.then(|| vec![0u8; 4]));
                any |= f.is_some();
                return f.is_none_or(|f| c.s.write_all(&f).is_ok());
            }
            let mut out = String::new();
            if let Some(v) = &vv {
                if *v != c.vv {
                    out.push_str(&sse("vv", &serde_json::Value::String(v.clone()).to_string()));
                    c.vv = v.clone();
                }
            }
            if let Some((v, e)) = &pose {
                if *v != c.pv {
                    out.push_str(e);
                    c.pv = v.clone();
                }
            }
            if out.is_empty() && keep {
                out.push_str(": keepalive\n\n");
            }
            any |= !out.is_empty();
            out.is_empty() || c.s.write_all(out.as_bytes()).is_ok()
        });
        if any {
            quiet = std::time::Instant::now();
        }
    }
}

/// 뷰어 서버 설정(명령줄을 `Config::parse` 로 한 번 읽은 것)
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub dir: Option<String>,
    pub port: u16,
    pub bind: String,
    pub search_model: Option<String>,
    pub vocab: Option<String>,
    pub encode_vocab: Option<(String, String)>,
}

impl Config {
    /// 뷰어 인자(프로그램 이름 뺀 것): `<기억 폴더> [--port 8080] [--bind 0.0.0.0] [--search-model <dir>] [--vocab <파일>]`
    pub fn parse(args: Vec<String>) -> Config {
        let mut c = Config { port: 8080, bind: "0.0.0.0".into(), ..Default::default() };
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--port" => c.port = it.next().and_then(|v| v.parse().ok()).unwrap_or(c.port),
                "--bind" => c.bind = it.next().unwrap_or(c.bind),
                "--search-model" => c.search_model = it.next(),
                "--vocab" => c.vocab = it.next(),
                "--encode-vocab" => c.encode_vocab = it.next().zip(it.next()),
                _ => c.dir = Some(a),
            }
        }
        c
    }
}

/// HTTP connection threads (the page opens up to 6 sockets per host)
const HTTP_THREADS: usize = 8;

/// 서버 실행(막힘): 뷰어 명령줄 인자(프로그램 이름 뺀 것, `ondevice view …`)
pub fn run(args: Vec<String>) {
    serve(Config::parse(args))
}

/// 서버 실행(막힘): 읽어 둔 설정으로. `ondevice --serve` 는 이것을 스레드로 부른다
pub fn serve(c: Config) {
    let Config { dir, port, bind, search_model, vocab, encode_vocab } = c;
    #[cfg(not(feature = "search"))]
    let _ = &vocab;
    // word vectors for a name list, made once (on a PC): sgview --search-model <dir> --encode-vocab <names.txt> <out>
    if let Some((list, out)) = &encode_vocab {
        #[cfg(feature = "search")]
        match search_model.as_deref().ok_or("--encode-vocab needs --search-model <dir>".to_string()).and_then(|m| search::encode_vocab(Path::new(m), Path::new(list), Path::new(out))) {
            Ok(n) => return eprintln!("{n} names → {out}"),
            Err(e) => {
                eprintln!("encode-vocab: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(not(feature = "search"))]
        {
            eprintln!("encode-vocab {list} {out}: built without the search feature");
            std::process::exit(1);
        }
    }
    let dir = match dir {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: sgview <memory_dir> [--port 8080] [--bind 0.0.0.0] [--search-model <dir>] [--vocab <word vectors>]");
            std::process::exit(2);
        }
    };
    let st = Arc::new(State { dir, walls: Mutex::new(None), view: Mutex::new(None), maps: Mutex::new(MapHist { vers: Vec::new() }), streams: Mutex::new(Vec::new()),
        parts: Mutex::new(None), octidx: Mutex::new(None),
        #[cfg(feature = "search")]
        text_src: search_model.clone().map(|m| (m, vocab.clone())),
        #[cfg(feature = "search")]
        text_enc: std::sync::OnceLock::new() });
    let l = TcpListener::bind((bind.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("cannot listen on {}:{}: {}", bind, port, e);
        std::process::exit(1);
    });
    #[cfg(not(feature = "search"))]
    if search_model.is_some() {
        eprintln!("search: built without the search feature — searching by name only");
    }
    eprintln!("sgview: http://localhost:{}  (memory dir {})", port, st.dir.display());
    {
        let st2 = st.clone();
        std::thread::Builder::new().name("sgview-stream".into()).spawn(move || stream_pusher(st2)).unwrap();
    }
    // a small fixed pool of connection threads: the page keeps a few keep-alive sockets (idle ones close after 5 s);
    // /stream sockets go to the one pusher thread
    let (tx, rx) = std::sync::mpsc::channel::<TcpStream>();
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..HTTP_THREADS {
        let (st, rx) = (st.clone(), rx.clone());
        std::thread::Builder::new().name(format!("sgview-http{i}")).spawn(move || loop {
            let c = rx.lock().unwrap().recv();
            match c {
                Ok(c) => handle(c, st.clone()),
                Err(_) => return,
            }
        }).unwrap();
    }
    for c in l.incoming().flatten() {
        let _ = c.set_nodelay(true);
        let _ = tx.send(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_files_layout() {
        let d = std::env::temp_dir().join(format!("sgview_pack_{}", std::process::id()));
        fs::create_dir_all(d.join("data")).unwrap();
        fs::write(d.join("data/a.ply"), b"abc").unwrap();
        let b = pack_files(&d, &["data/a.ply", "data/none.ply", "../x"]);
        let mut i = 0;
        let mut rd = |n: usize| { let s = &b[i..i + n]; i += n; s.to_vec() };
        let u = |v: Vec<u8>| u32::from_le_bytes([v[0], v[1], v[2], v[3]]);
        assert_eq!(u(rd(4)), 10);
        assert_eq!(rd(10), b"data/a.ply");
        assert_eq!(u(rd(4)), 3);
        assert_eq!(rd(3), b"abc");
        assert_eq!(u(rd(4)), 13);
        rd(13);
        assert_eq!(u(rd(4)), u32::MAX);
        assert_eq!(u(rd(4)), 4);
        rd(4);
        assert_eq!(u(rd(4)), u32::MAX);   // outside the folder: refused like a missing file
        assert_eq!(i, b.len());
        let _ = fs::remove_dir_all(&d);
    }
}
