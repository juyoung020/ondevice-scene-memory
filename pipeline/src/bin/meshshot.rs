//! 표면 지도(mesh/*.ply)를 PNG 로 그린다: 위에서 내려다본 그림 + 네 모서리에서 비스듬히 본 그림 3 장(2×2).
//! 뷰어 없이 지도 품질을 눈으로 견줄 때 쓴다(천장은 바닥 위 --ceil m 에서 자름).
//! view.json 의 표면 목록에 텍스처 번호가 있는 구역은 텍스처 면(mesh/t_x_y_z_n.ply + .jpg)을, 나머지는 색 면(r_x_y_z.ply)을 그린다.
//! 빛은 뷰어와 같은 반구 빛 + 해 빛(꼭짓점 법선).
//!   meshshot <기억 폴더> <출력.png> [--ceil 2.2] [--size 800x600] [--plain] [--eye x,y,z --at x,y,z]
//!   --plain: 텍스처 없이 색 면만, --eye/--at: 그 자리에서 본 그림 한 장(가까이 보기)
use std::io::Read;

type V3 = [f32; 3];

struct Vert {
    p: V3,
    n: V3,
    uv: [f32; 2],
    c: [u8; 3],
}

struct Tex {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

/// PLY 읽기(색 면: 위치·색, 텍스처 면: 위치·법선·UV·색). tex: 이 면의 텍스처 번호
fn load_mesh(path: &std::path::Path, verts: &mut Vec<Vert>, tris: &mut Vec<([u32; 3], Option<usize>)>, tex: Option<usize>) {
    let mut b = vec![];
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    let he = b.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
    let head = std::str::from_utf8(&b[..he]).unwrap();
    let num = |key: &str| head.lines().find(|l| l.starts_with(key)).map_or(0, |l| l.split_whitespace().last().unwrap().parse::<usize>().unwrap());
    let (nv, nf) = (num("element vertex"), num("element face"));
    let textured = head.contains("property float s");
    let stride = if textured { 35 } else { 15 };
    let base = verts.len() as u32;
    let mut o = he;
    for _ in 0..nv {
        let f = |k: usize| f32::from_le_bytes(b[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
        let c = o + stride - 3;
        verts.push(if textured {
            Vert { p: [f(0), f(1), f(2)], n: [f(3), f(4), f(5)], uv: [f(6), f(7)], c: [b[c], b[c + 1], b[c + 2]] }
        } else {
            Vert { p: [f(0), f(1), f(2)], n: [0.0; 3], uv: [0.0; 2], c: [b[c], b[c + 1], b[c + 2]] }
        });
        o += stride;
    }
    let first = tris.len();
    for _ in 0..nf {
        let i = |k: usize| i32::from_le_bytes(b[o + 1 + 4 * k..o + 5 + 4 * k].try_into().unwrap()) as u32 + base;
        tris.push(([i(0), i(1), i(2)], tex));
        o += 13;
    }
    if !textured {
        // 색 면: 꼭짓점 법선(넓이 무게)
        for (t, _) in &tris[first..] {
            let n = cross(sub(verts[t[1] as usize].p, verts[t[0] as usize].p), sub(verts[t[2] as usize].p, verts[t[0] as usize].p));
            for &i in t {
                let v = &mut verts[i as usize].n;
                *v = [v[0] + n[0], v[1] + n[1], v[2] + n[2]];
            }
        }
        for v in &mut verts[base as usize..] {
            v.n = norm(v.n);
        }
    }
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt().max(1e-9);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// 한 화면: 세계 점 → (화면 x, y, 깊이). None = 화면 뒤
struct Cam {
    eye: V3,
    r: V3,
    u: V3,
    f: V3,
    ortho: Option<f32>, // Some(화소당 m)
    focal: f32,
    w: usize,
    h: usize,
    c: V3,
}

impl Cam {
    fn proj(&self, p: V3) -> Option<(f32, f32, f32)> {
        let d = sub(p, if self.ortho.is_some() { self.c } else { self.eye });
        let (x, y, z) = (dot(d, self.r), dot(d, self.u), dot(d, self.f));
        match self.ortho {
            Some(s) => Some((self.w as f32 / 2.0 + x / s, self.h as f32 / 2.0 - y / s, -z)),
            None => (z > 0.05).then(|| (self.w as f32 / 2.0 + self.focal * x / z, self.h as f32 / 2.0 - self.focal * y / z, z)),
        }
    }
    fn look(eye: V3, at: V3, w: usize, h: usize, focal: f32) -> Cam {
        let f = norm(sub(at, eye));
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        let u = cross(r, f);
        Cam { eye, r, u, f, ortho: None, focal, w, h, c: at }
    }
}

fn sample(t: &Tex, uv: [f32; 2]) -> [f32; 3] {
    let x = (uv[0] * t.w as f32 - 0.5).clamp(0.0, t.w as f32 - 1.001);
    let y = ((1.0 - uv[1]) * t.h as f32 - 0.5).clamp(0.0, t.h as f32 - 1.001);
    let (xi, yi) = (x as usize, y as usize);
    let (fx, fy) = (x - xi as f32, y - yi as f32);
    let mut o = [0f32; 3];
    for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let i = ((yi + dy) * t.w + xi + dx) * 3;
        for q in 0..3 {
            o[q] += wt * t.px[i + q] as f32;
        }
    }
    o
}

/// 반구 빛(위 0.85, 아래 0.55) + 해 빛 0.25: 뷰어 표면 셰이더와 같은 식
fn light(n: V3) -> f32 {
    let sun = norm([0.4, -0.6, 1.0]);
    let up = 0.5 + 0.5 * n[2];
    0.55 + 0.30 * up + 0.25 * dot(n, sun).max(0.0)
}

fn render(cam: &Cam, verts: &[Vert], tris: &[([u32; 3], Option<usize>)], texs: &[Tex], out: &mut [u8], stride: usize, ox: usize, oy: usize) {
    let mut zb = vec![f32::INFINITY; cam.w * cam.h];
    for (t, ti) in tris {
        let (a, b, c) = (&verts[t[0] as usize], &verts[t[1] as usize], &verts[t[2] as usize]);
        let (Some(pa), Some(pb), Some(pc)) = (cam.proj(a.p), cam.proj(b.p), cam.proj(c.p)) else { continue };
        let x0 = pa.0.min(pb.0).min(pc.0).floor().max(0.0) as i32;
        let x1 = pa.0.max(pb.0).max(pc.0).ceil().min(cam.w as f32 - 1.0) as i32;
        let y0 = pa.1.min(pb.1).min(pc.1).floor().max(0.0) as i32;
        let y1 = pa.1.max(pb.1).max(pc.1).ceil().min(cam.h as f32 - 1.0) as i32;
        let area = (pb.0 - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (pb.1 - pa.1);
        if area.abs() < 1e-6 || x1 < x0 || y1 < y0 || (x1 - x0) > 600 {
            continue;
        }
        // 원근 보정 보간(1/z)
        let persp = cam.ortho.is_none();
        let iz = |p: (f32, f32, f32)| if persp { 1.0 / p.2 } else { 1.0 };
        let (za, zb_, zc) = (iz(pa), iz(pb), iz(pc));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((pb.0 - px) * (pc.1 - py) - (pc.0 - px) * (pb.1 - py)) / area;
                let w1 = ((pc.0 - px) * (pa.1 - py) - (pa.0 - px) * (pc.1 - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * pa.2 + w1 * pb.2 + w2 * pc.2;
                let i = y as usize * cam.w + x as usize;
                if z >= zb[i] {
                    continue;
                }
                zb[i] = z;
                let (q0, q1, q2) = (w0 * za, w1 * zb_, w2 * zc);
                let s = q0 + q1 + q2;
                let (q0, q1, q2) = (q0 / s, q1 / s, q2 / s);
                let n = norm([0, 1, 2].map(|k| q0 * a.n[k] + q1 * b.n[k] + q2 * c.n[k]));
                // 뒷면에서 보면 법선을 뒤집음(양면 그리기)
                let n = if dot(n, sub(cam.eye, a.p)) < 0.0 && persp { [-n[0], -n[1], -n[2]] } else { n };
                let sh = light(n);
                let base = match ti {
                    Some(k) => sample(&texs[*k], [q0 * a.uv[0] + q1 * b.uv[0] + q2 * c.uv[0], q0 * a.uv[1] + q1 * b.uv[1] + q2 * c.uv[1]]),
                    None => [255.0; 3],
                };
                let o = ((oy + y as usize) * stride + ox + x as usize) * 3;
                for k in 0..3 {
                    let col = (q0 * a.c[k] as f32 + q1 * b.c[k] as f32 + q2 * c.c[k] as f32) / if ti.is_some() { 127.5 } else { 255.0 };
                    out[o + k] = (base[k] * col * sh).min(255.0) as u8;
                }
            }
        }
    }
}

fn parse3(s: &str) -> V3 {
    let v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("meshshot <기억 폴더> <출력.png> [--ceil 2.2] [--size 800x600] [--plain] [--eye x,y,z --at x,y,z]");
        std::process::exit(2);
    }
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let ceil: f32 = get("--ceil").map_or(2.2, |s| s.parse().unwrap());
    let plain = a.iter().any(|x| x == "--plain");
    let (w, h) = get("--size").map_or((800, 600), |s| {
        let v: Vec<usize> = s.split('x').map(|x| x.parse().unwrap()).collect();
        (v[0], v[1])
    });
    let dir = std::path::Path::new(&a[1]);
    // 구역 목록(view.json): 텍스처 번호가 있으면 텍스처 면
    let regions: Vec<Vec<i64>> = std::fs::read_to_string(dir.join("view.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| serde_json::from_value(v["scene_points"]["surface"]["regions"].clone()).ok())
        .unwrap_or_default();
    let (mut verts, mut tris, mut texs) = (vec![], vec![], vec![]);
    let mut n_tex = 0;
    if regions.is_empty() {
        for e in std::fs::read_dir(dir.join("mesh")).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("r_") && name.ends_with(".ply") {
                load_mesh(&e.path(), &mut verts, &mut tris, None);
            }
        }
    }
    for r in &regions {
        let key = format!("{}_{}_{}", r[0], r[1], r[2]);
        if r.len() > 4 && !plain {
            let base = dir.join(format!("mesh/t_{key}_{}", r[4]));
            let im = image::open(base.with_extension("jpg")).unwrap().into_rgb8();
            texs.push(Tex { w: im.width() as usize, h: im.height() as usize, px: im.into_raw() });
            load_mesh(&base.with_extension("ply"), &mut verts, &mut tris, Some(texs.len() - 1));
            n_tex += 1;
        } else {
            let p = dir.join(format!("mesh/r_{key}.ply"));
            if p.exists() {
                load_mesh(&p, &mut verts, &mut tris, None);
            }
        }
    }
    // 천장 위는 버림(위에서 내려다볼 때 가림)
    tris.retain(|(t, _)| t.iter().all(|&i| verts[i as usize].p[2] < ceil));
    if let (Some(e), Some(at)) = (get("--eye"), get("--at")) {
        let mut img = vec![30u8; w * h * 3];
        render(&Cam::look(parse3(&e), parse3(&at), w, h, w as f32 * 0.8), &verts, &tris, &texs, &mut img, w, 0, 0);
        image::save_buffer(&a[2], &img, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
        eprintln!("삼각형 {} 개(텍스처 구역 {n_tex} 개) → {}", tris.len(), a[2]);
        return;
    }
    // 범위: 축마다 꼭짓점 위치의 3–97 % (먼 바닥 몇 조각에 시점이 끌려가지 않게)
    let (mut lo, mut hi) = ([0f32; 3], [0f32; 3]);
    for k in 0..3 {
        let mut v: Vec<f32> = tris.iter().flat_map(|(t, _)| t.iter().map(|&i| verts[i as usize].p[k])).step_by(7).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (lo[k], hi[k]) = (v[v.len() * 3 / 100], v[v.len() * 97 / 100]);
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(0.5);
    let (sw, sh) = (w * 2, h * 2);
    let mut img = vec![30u8; sw * sh * 3];
    // 위에서
    let top = Cam { eye: [c[0], c[1], hi[2] + 1.0], r: [1.0, 0.0, 0.0], u: [0.0, 1.0, 0.0], f: [0.0, 0.0, -1.0], ortho: Some(span * 1.05 / h.min(w) as f32), focal: 0.0, w, h, c: [c[0], c[1], hi[2] + 1.0] };
    render(&top, &verts, &tris, &texs, &mut img, sw, 0, 0);
    // 비스듬히: 세 방향(높이 45°)
    for (k, ang) in [45f32, 165.0, 285.0].iter().enumerate() {
        let th = ang.to_radians();
        let dist = span * 0.75;
        let eye = [c[0] + dist * th.cos(), c[1] + dist * th.sin(), c[2] + dist * 0.9];
        let cam = Cam::look(eye, c, w, h, w as f32 * 0.75);
        let (ox, oy) = [(w, 0), (0, h), (w, h)][k];
        render(&cam, &verts, &tris, &texs, &mut img, sw, ox, oy);
    }
    image::save_buffer(&a[2], &img, sw as u32, sh as u32, image::ColorType::Rgb8).unwrap();
    eprintln!("삼각형 {} 개(텍스처 구역 {n_tex} 개), 범위 {:.1}×{:.1}×{:.1} m → {}", tris.len(), hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], a[2]);
}
