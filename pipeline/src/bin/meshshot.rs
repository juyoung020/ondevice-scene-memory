//! 표면 지도(mesh/*.ply)를 PNG 로 그린다: 위에서 내려다본 그림 + 네 모서리에서 비스듬히 본 그림 3 장(2×2).
//! 뷰어 없이 지도 품질을 눈으로 견줄 때 쓴다(천장은 바닥 위 --ceil m 에서 자름).
//!   meshshot <기억 폴더> <출력.png> [--ceil 2.2] [--size 800x600]
use std::io::Read;

type V3 = [f32; 3];

fn load_mesh(path: &std::path::Path, verts: &mut Vec<(V3, [u8; 3])>, tris: &mut Vec<[u32; 3]>) {
    let mut b = vec![];
    std::fs::File::open(path).unwrap().read_to_end(&mut b).unwrap();
    let he = b.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
    let head = std::str::from_utf8(&b[..he]).unwrap();
    let num = |key: &str| head.lines().find(|l| l.starts_with(key)).map_or(0, |l| l.split_whitespace().last().unwrap().parse::<usize>().unwrap());
    let (nv, nf) = (num("element vertex"), num("element face"));
    let base = verts.len() as u32;
    let mut o = he;
    for _ in 0..nv {
        let f = |k: usize| f32::from_le_bytes(b[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
        verts.push(([f(0), f(1), f(2)], [b[o + 12], b[o + 13], b[o + 14]]));
        o += 15;
    }
    for _ in 0..nf {
        let i = |k: usize| i32::from_le_bytes(b[o + 1 + 4 * k..o + 5 + 4 * k].try_into().unwrap()) as u32 + base;
        tris.push([i(0), i(1), i(2)]);
        o += 13;
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
}

fn render(cam: &Cam, verts: &[(V3, [u8; 3])], tris: &[[u32; 3]], out: &mut [u8], stride: usize, ox: usize, oy: usize) {
    let mut zb = vec![f32::INFINITY; cam.w * cam.h];
    let light = norm([0.3, 0.5, 1.0]);
    for t in tris {
        let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
        let n = norm(cross(sub(b.0, a.0), sub(c.0, a.0)));
        let shade = 0.45 + 0.55 * dot(n, light).abs();
        let (Some(pa), Some(pb), Some(pc)) = (cam.proj(a.0), cam.proj(b.0), cam.proj(c.0)) else { continue };
        let x0 = pa.0.min(pb.0).min(pc.0).floor().max(0.0) as i32;
        let x1 = pa.0.max(pb.0).max(pc.0).ceil().min(cam.w as f32 - 1.0) as i32;
        let y0 = pa.1.min(pb.1).min(pc.1).floor().max(0.0) as i32;
        let y1 = pa.1.max(pb.1).max(pc.1).ceil().min(cam.h as f32 - 1.0) as i32;
        let area = (pb.0 - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (pb.1 - pa.1);
        if area.abs() < 1e-6 || x1 < x0 || y1 < y0 || (x1 - x0) > 400 {
            continue;
        }
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
                if z < zb[i] {
                    zb[i] = z;
                    let o = ((oy + y as usize) * stride + ox + x as usize) * 3;
                    for k in 0..3 {
                        let col = w0 * a.1[k] as f32 + w1 * b.1[k] as f32 + w2 * c.1[k] as f32;
                        out[o + k] = (col * shade).min(255.0) as u8;
                    }
                }
            }
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("meshshot <기억 폴더> <출력.png> [--ceil 2.2] [--size 800x600]");
        std::process::exit(2);
    }
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let ceil: f32 = get("--ceil").map_or(2.2, |s| s.parse().unwrap());
    let (w, h) = get("--size").map_or((800, 600), |s| {
        let v: Vec<usize> = s.split('x').map(|x| x.parse().unwrap()).collect();
        (v[0], v[1])
    });
    let (mut verts, mut tris) = (vec![], vec![]);
    for e in std::fs::read_dir(std::path::Path::new(&a[1]).join("mesh")).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "ply") {
            load_mesh(&e.path(), &mut verts, &mut tris);
        }
    }
    // 천장 위는 버림(위에서 내려다볼 때 가림)
    tris.retain(|t| t.iter().all(|&i| verts[i as usize].0[2] < ceil));
    // 범위: 축마다 꼭짓점 위치의 3–97 % (먼 바닥 몇 조각에 시점이 끌려가지 않게)
    let (mut lo, mut hi) = ([0f32; 3], [0f32; 3]);
    for k in 0..3 {
        let mut v: Vec<f32> = tris.iter().flat_map(|t| t.iter().map(|&i| verts[i as usize].0[k])).step_by(7).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (lo[k], hi[k]) = (v[v.len() * 3 / 100], v[v.len() * 97 / 100]);
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(0.5);
    let (sw, sh) = (w * 2, h * 2);
    let mut img = vec![30u8; sw * sh * 3];
    // 위에서
    let top = Cam { eye: c, r: [1.0, 0.0, 0.0], u: [0.0, 1.0, 0.0], f: [0.0, 0.0, -1.0], ortho: Some(span * 1.05 / h.min(w) as f32), focal: 0.0, w, h, c: [c[0], c[1], hi[2] + 1.0] };
    render(&top, &verts, &tris, &mut img, sw, 0, 0);
    // 비스듬히: 세 방향(높이 45°)
    for (k, ang) in [45f32, 165.0, 285.0].iter().enumerate() {
        let th = ang.to_radians();
        let dist = span * 0.75;
        let eye = [c[0] + dist * th.cos(), c[1] + dist * th.sin(), c[2] + dist * 0.9];
        let f = norm(sub(c, eye));
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        let u = cross(r, f);
        let cam = Cam { eye, r, u, f, ortho: None, focal: w as f32 * 0.75, w, h, c };
        let (ox, oy) = [(w, 0), (0, h), (w, h)][k];
        render(&cam, &verts, &tris, &mut img, sw, ox, oy);
    }
    image::save_buffer(&a[2], &img, sw as u32, sh as u32, image::ColorType::Rgb8).unwrap();
    eprintln!("삼각형 {} 개, 범위 {:.1}×{:.1}×{:.1} m → {}", tris.len(), hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], a[2]);
}
