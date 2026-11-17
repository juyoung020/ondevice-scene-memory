//! 물체 지도 견주기: 기억 폴더 하나의 중복(과분할) 짝, 두 폴더(전 · 후)면 물체끼리 짝지어 더 합쳐진 것 · 더 갈라진 것.
//!   objcmp <기억 폴더> [<다른 기억 폴더>]
//! 점: objects/O<id>_points.ply. 칸 = 4 cm.
//! 중복 짝(이름 무관): 작은 쪽 점의 30 % 이상이 큰 쪽 점 칸(이웃 포함) 안이거나, 두 상자(점 5–95 %) 겹침이 작은 상자의 절반 이상.
//! 전후 짝: 작은 쪽 칸의 50 % 이상이 겹치면 같은 자리. 후 물체 하나가 서로 중복이 아닌 전 물체 둘 이상을 덮으면 "더 합침"(잘못 합쳤을 수 있는 것),
//! 반대면 "더 갈라짐"
use std::collections::HashSet;
use std::path::Path;

type K = (i32, i32, i32);

struct O {
    id: u64,
    name: String,
    pts: Vec<[f32; 3]>,
    set: HashSet<K>,
    bbox: ([f32; 3], [f32; 3]),
}

fn kk(p: &[f32; 3], v: f32) -> K {
    ((p[0] / v).floor() as i32, (p[1] / v).floor() as i32, (p[2] / v).floor() as i32)
}

const VOX: f32 = 0.04;

fn load(dir: &Path) -> Vec<O> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("view.json")).expect("view.json")).unwrap();
    v["objects"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| {
            let b = std::fs::read(dir.join(o["points"]["path"].as_str()?)).ok()?;
            let end = b.windows(11).position(|w| w == b"end_header\n")?;
            let head = String::from_utf8_lossy(&b[..end]);
            let nv: usize = head.lines().find_map(|l| l.strip_prefix("element vertex ").map(|x| x.trim().parse().unwrap_or(0)))?;
            let pts: Vec<[f32; 3]> = b[end + 11..]
                .chunks_exact(15)
                .take(nv)
                .map(|c| [0, 1, 2].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap())))
                .collect();
            if pts.len() < 10 {
                return None;
            }
            let mut lo = [0.0; 3];
            let mut hi = [0.0; 3];
            for k in 0..3 {
                let mut v: Vec<f32> = pts.iter().map(|x| x[k]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                lo[k] = v[v.len() / 20];
                hi[k] = v[v.len() * 19 / 20];
            }
            let set = pts.iter().map(|p| kk(p, VOX)).collect();
            Some(O { id: o["id"].as_u64()?, name: o["name"].as_str()?.to_string(), pts, set, bbox: (lo, hi) })
        })
        .collect()
}

/// a 의 점 중 b 의 점 칸(이웃 포함) 안 비율
fn near(a: &O, b: &O) -> f32 {
    a.pts
        .iter()
        .filter(|p| {
            let c = kk(p, VOX);
            (-1..=1).any(|x| (-1..=1).any(|y| (-1..=1).any(|z| b.set.contains(&(c.0 + x, c.1 + y, c.2 + z)))))
        })
        .count() as f32
        / a.pts.len() as f32
}

fn box_ovl(a: &O, b: &O) -> f32 {
    let vol = |b: &([f32; 3], [f32; 3])| (0..3).map(|k| (b.1[k] - b.0[k]).max(0.01)).product::<f32>();
    let inter: f32 = (0..3).map(|k| (a.bbox.1[k].min(b.bbox.1[k]) - a.bbox.0[k].max(b.bbox.0[k])).max(0.0)).product();
    inter / vol(&a.bbox).min(vol(&b.bbox))
}

fn dup(a: &O, b: &O) -> bool {
    let (s, l) = if a.pts.len() <= b.pts.len() { (a, b) } else { (b, a) };
    near(s, l) >= 0.3 || box_ovl(a, b) >= 0.5
}

fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

/// (중복 짝 수, 중복 묶음을 하나로 센 물체 수)
fn dups(objs: &[O], verbose: bool) -> (usize, usize) {
    let mut par: Vec<usize> = (0..objs.len()).collect();
    let mut n = 0;
    for i in 0..objs.len() {
        for j in i + 1..objs.len() {
            if dup(&objs[i], &objs[j]) {
                n += 1;
                if verbose {
                    println!("  중복 O{} {} ({} 점) · O{} {} ({} 점)", objs[i].id, objs[i].name, objs[i].pts.len(), objs[j].id, objs[j].name, objs[j].pts.len());
                }
                let (a, b) = (root(&mut par, i), root(&mut par, j));
                par[a] = b;
            }
        }
    }
    (n, (0..objs.len()).filter(|&i| root(&mut par, i) == i).count())
}

/// 작은 쪽 칸 중 겹친 비율
fn same_place(a: &O, b: &O) -> bool {
    let (s, l) = if a.set.len() <= b.set.len() { (a, b) } else { (b, a) };
    s.set.iter().filter(|k| l.set.contains(k)).count() as f32 >= 0.5 * s.set.len() as f32
}

/// 후 물체 하나가 덮는 전 물체 중 서로 중복이 아닌 것이 둘 이상인 경우
fn more_merged(new: &[O], old: &[O], tag: &str) -> usize {
    let mut n = 0;
    for b in new {
        let cov: Vec<&O> = old.iter().filter(|a| same_place(a, b)).collect();
        let distinct = (0..cov.len()).any(|i| (i + 1..cov.len()).any(|j| !dup(cov[i], cov[j])));
        if distinct {
            n += 1;
            println!("  {tag} O{} {} ({} 점) ← {}", b.id, b.name, b.pts.len(), cov.iter().map(|a| format!("O{} {} ({} 점)", a.id, a.name, a.pts.len())).collect::<Vec<_>>().join(" + "));
        }
    }
    n
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x = load(Path::new(&a[1]));
    let (dx, ux) = dups(&x, a.len() == 2);
    println!("{}: 물체 {} 개, 중복 짝 {dx} 개, 중복 묶음을 하나로 센 물체 {ux} 개", a[1], x.len());
    if let Some(b) = a.get(2) {
        let y = load(Path::new(b));
        let (dy, uy) = dups(&y, true);
        println!("{b}: 물체 {} 개, 중복 짝 {dy} 개, 중복 묶음을 하나로 센 물체 {uy} 개", y.len());
        let same = x.iter().filter(|o| y.iter().any(|p| same_place(o, p) && o.set.len().abs_diff(p.set.len()) * 10 <= o.set.len().max(p.set.len()))).count();
        let m = more_merged(&y, &x, "후가 더 합침");
        let s = more_merged(&x, &y, "후가 더 갈라짐");
        println!("전후: 거의 같은 물체 {same} / {}, 후가 더 합친 물체 {m}, 후가 더 갈라진 물체 {s}", x.len());
    }
}
