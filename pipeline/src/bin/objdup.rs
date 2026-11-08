//! 물체 지도의 중복(과분할) 짝 세기: 같은 이름 두 물체의 점이 얼마나 겹치는지.
//!   objdup <기억 폴더> [--all]
//! 짝마다: 중심 거리, 작은 쪽 점 중 큰 쪽 점 4 cm 안 비율(겹침), 두 상자(점 5–95 % 범위) 겹침 부피 / 작은 상자 부피.
//! 중복 = 겹침 ≥ 0.3 또는 상자 겹침 ≥ 0.5. --all: 같은 이름이고 중심 1.5 m 안인 짝 모두 출력.
use std::collections::HashSet;
use std::path::Path;

fn load(p: &Path) -> Vec<[f32; 3]> {
    let b = std::fs::read(p).unwrap_or_default();
    let Some(end) = b.windows(11).position(|w| w == b"end_header\n") else { return vec![] };
    let head = String::from_utf8_lossy(&b[..end]);
    let nv: usize = head.lines().find_map(|l| l.strip_prefix("element vertex ").map(|x| x.trim().parse().unwrap_or(0))).unwrap_or(0);
    let mut o = end + 11;
    let mut out = Vec::with_capacity(nv);
    for _ in 0..nv {
        if o + 15 > b.len() {
            break;
        }
        let f = |k: usize| f32::from_le_bytes(b[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
        out.push([f(0), f(1), f(2)]);
        o += 15;
    }
    out
}

fn kk(p: &[f32; 3], v: f32) -> (i32, i32, i32) {
    ((p[0] / v).floor() as i32, (p[1] / v).floor() as i32, (p[2] / v).floor() as i32)
}

fn bbox(p: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for k in 0..3 {
        let mut v: Vec<f32> = p.iter().map(|x| x[k]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lo[k] = v[v.len() / 20];
        hi[k] = v[v.len() * 19 / 20];
    }
    (lo, hi)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = Path::new(&a[1]);
    let all = a.iter().any(|x| x == "--all");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("view.json")).expect("view.json")).unwrap();
    let objs: Vec<(u64, String, Vec<[f32; 3]>)> = v["objects"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| {
            let pts = load(&dir.join(o["points"]["path"].as_str()?));
            (pts.len() >= 10).then(|| (o["id"].as_u64().unwrap(), o["name"].as_str().unwrap().to_string(), pts))
        })
        .collect();
    let vox = 0.04f32;
    let sets: Vec<HashSet<(i32, i32, i32)>> = objs.iter().map(|o| o.2.iter().map(|p| kk(p, vox)).collect()).collect();
    let boxes: Vec<_> = objs.iter().map(|o| bbox(&o.2)).collect();
    let cen = |b: &([f32; 3], [f32; 3])| [0, 1, 2].map(|k| 0.5 * (b.0[k] + b.1[k]));
    let mut ndup = 0;
    for i in 0..objs.len() {
        for j in i + 1..objs.len() {
            if objs[i].1 != objs[j].1 {
                continue;
            }
            let (ci, cj) = (cen(&boxes[i]), cen(&boxes[j]));
            let d = ((0..3).map(|k| (ci[k] - cj[k]).powi(2)).sum::<f32>()).sqrt();
            // 겹침: 작은 쪽 점 중 큰 쪽 점(4 cm 칸 + 이웃 칸) 안
            let (l, ps) = if objs[i].2.len() <= objs[j].2.len() { (j, &objs[i].2) } else { (i, &objs[j].2) };
            let near = ps
                .iter()
                .filter(|p| {
                    let c = kk(p, vox);
                    (-1..=1).any(|x| (-1..=1).any(|y| (-1..=1).any(|z| sets[l].contains(&(c.0 + x, c.1 + y, c.2 + z)))))
                })
                .count() as f32
                / ps.len() as f32;
            let vol = |b: &([f32; 3], [f32; 3])| (0..3).map(|k| (b.1[k] - b.0[k]).max(0.01)).product::<f32>();
            let (bi, bj) = (&boxes[i], &boxes[j]);
            let inter: f32 = (0..3).map(|k| (bi.1[k].min(bj.1[k]) - bi.0[k].max(bj.0[k])).max(0.0)).product();
            let bov = inter / vol(bi).min(vol(bj));
            let dup = near >= 0.3 || bov >= 0.5;
            if dup {
                ndup += 1;
            }
            if dup || (all && d < 1.5) {
                println!(
                    "{} {:<12} O{} ({} 점) · O{} ({} 점): 중심 {:.2} m, 겹침 {:.2}, 상자 {:.2}",
                    if dup { "중복" } else { "    " },
                    objs[i].1, objs[i].0, objs[i].2.len(), objs[j].0, objs[j].2.len(), d, near, bov
                );
            }
        }
    }
    println!("물체 {} 개, 중복 짝 {} 개", objs.len(), ndup);
}
