//! 두 표면 지도(mesh/r_*.ply 폴더)의 거리: A 의 꼭짓점마다 B 의 가장 가까운 꼭짓점까지 거리.
//!   surfcmp <A 기억 폴더> <B(기준) 기억 폴더> [--align]
//! 출력: 평균·중앙값·90 %·5 cm 안 비율·10 cm 안 비율(10 cm 밖 꼭짓점은 '기준에 없는 면'으로 따로 셈).
//! --align: 두 폴더의 poses_map.txt(지도에 넣은 카메라 위치)를 시각으로 짝지어 A 를 B 좌표로 옮긴 뒤 잼
//!          (세계 좌표 잡는 법이 다른 실행끼리: 예 스트림 입력(바닥 합의) 대 오프라인 위치 파일(경로 평면)).
#[path = "../horn.rs"]
mod horn;
use std::collections::HashMap;
use std::path::Path;

fn load(dir: &Path) -> Vec<[f32; 3]> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(dir.join("mesh")) else { return out };
    for e in rd.flatten() {
        let b = std::fs::read(e.path()).unwrap_or_default();
        let Some(end) = b.windows(11).position(|w| w == b"end_header\n") else { continue };
        let head = String::from_utf8_lossy(&b[..end]);
        let nv: usize = head.lines().find_map(|l| l.strip_prefix("element vertex ").map(|x| x.trim().parse().unwrap_or(0))).unwrap_or(0);
        let mut o = end + 11;
        for _ in 0..nv {
            let f = |k: usize| f32::from_le_bytes(b[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
            out.push([f(0), f(1), f(2)]);
            o += 15;
        }
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mut pa, pb) = (load(Path::new(&a[1])), load(Path::new(&a[2])));
    if a.iter().any(|x| x == "--align") {
        let tr = |d: &str| horn::read_tum(&Path::new(d).join("poses_map.txt").to_string_lossy());
        let (ea, eb) = (tr(&a[1]), tr(&a[2]));
        let pairs: Vec<_> = horn::pair_by_time(&ea, &eb, 0.02).into_iter().map(|(i, j)| (ea[i].1, eb[j].1)).collect();
        assert!(pairs.len() >= 3, "--align: 시각이 맞는 위치 짝이 3 개 미만");
        let (r, ce, cg) = horn::horn(&pairs);
        let rms = (pairs.iter().map(|(e, g)| { let d = horn::apply(&r, &ce, &cg, e); (0..3).map(|k| (d[k] - g[k]).powi(2)).sum::<f64>() }).sum::<f64>() / pairs.len() as f64).sqrt();
        eprintln!("맞춤: 카메라 위치 짝 {} 개, 맞춘 뒤 위치 차이 RMSE {:.4} m", pairs.len(), rms);
        for p in pa.iter_mut() {
            let q = horn::apply(&r, &ce, &cg, &[p[0] as f64, p[1] as f64, p[2] as f64]);
            *p = [q[0] as f32, q[1] as f32, q[2] as f32];
        }
    }
    let cell = 0.05f32;
    let key = |p: &[f32; 3]| ((p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32, (p[2] / cell).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<[f32; 3]>> = HashMap::new();
    for p in &pb {
        grid.entry(key(p)).or_default().push(*p);
    }
    let mut ds: Vec<f32> = pa
        .iter()
        .map(|p| {
            let k = key(p);
            let mut best = f32::MAX;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        if let Some(v) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for q in v {
                                let d = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                                best = best.min(d);
                            }
                        }
                    }
                }
            }
            best.sqrt()
        })
        .collect();
    ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = ds.len().max(1) as f32;
    let near: Vec<f32> = ds.iter().copied().filter(|d| *d <= 0.10).collect();
    let pct = |q: f32| near.get(((near.len().max(1) - 1) as f32 * q) as usize).copied().unwrap_or(f32::NAN);
    println!(
        "{{\"verts_a\":{},\"verts_b\":{},\"within_5cm\":{:.3},\"within_10cm\":{:.3},\"median_cm\":{:.2},\"p90_cm\":{:.2},\"mean_cm\":{:.2}}}",
        pa.len(),
        pb.len(),
        ds.iter().filter(|d| **d <= 0.05).count() as f32 / n,
        near.len() as f32 / n,
        pct(0.5) * 100.0,
        pct(0.9) * 100.0,
        near.iter().sum::<f32>() / near.len().max(1) as f32 * 100.0
    );
}
