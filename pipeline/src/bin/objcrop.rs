//! 물체 점을 스트림 컬러 장면에 다시 투영해 크롭으로 저장: 합친(또는 안 합친) 물체가 실제로 몇 개인지 눈으로 확인.
//!   objcrop <기억 폴더> <스트림.mmf> <id,id,...> <출력 접두어> [장면 수=4]
//! 물체 점(objects/O<id>_points.ply, 메쉬 좌표)을 poses_map.txt 의 같은 시각 카메라 위치로 투영. 깊이보다 10 cm 넘게 뒤인 점(가려짐)은 뺌.
//! 점이 가장 많이 보이는 장면부터 시각이 2 초 넘게 떨어진 장면을 고르고, 물체별 투영 상자(빨강·초록·파랑·노랑 차례)를 그린 크롭(여백 포함, 2 배)을 저장.
#[path = "../geom.rs"]
#[allow(dead_code)]
mod geom;
#[path = "../stream.rs"]
#[allow(dead_code)]
mod stream;
use geom::*;
use std::path::Path;

fn load_ply(p: &Path) -> Vec<V3> {
    let b = std::fs::read(p).unwrap_or_default();
    let Some(end) = b.windows(11).position(|w| w == b"end_header\n") else { return vec![] };
    let head = String::from_utf8_lossy(&b[..end]);
    let nv: usize = head.lines().find_map(|l| l.strip_prefix("element vertex ").map(|x| x.trim().parse().unwrap_or(0))).unwrap_or(0);
    let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
    (0..nv).map(|i| end + 11 + 15 * i).filter(|o| o + 15 <= b.len()).map(|o| [f(o), f(o + 4), f(o + 8)]).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = Path::new(&a[1]);
    let ids: Vec<&str> = a[3].split(',').collect();
    let pre = &a[4];
    let nshow: usize = a.get(5).map_or(4, |s| s.parse().unwrap());
    let objs: Vec<Vec<V3>> = ids.iter().map(|i| load_ply(&dir.join(format!("objects/O{i}_points.ply")))).collect();
    // 시각 → (자세, 위치): 카메라(OpenCV) → 메쉬
    let mut poses: Vec<(f64, M3, V3)> = vec![];
    for l in std::fs::read_to_string(dir.join("poses_map.txt")).unwrap().lines().filter(|l| !l.starts_with('#')) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        poses.push((v[0], quat_to_m([v[4], v[5], v[6], v[7]]), [v[1], v[2], v[3]]));
    }
    // 1 차: 장면마다 물체별 보이는 점 수와 투영 상자
    struct Hit {
        seq: usize,
        t: f64,
        boxes: Vec<Option<[f64; 4]>>,
        score: usize,
    }
    let project = |f: &stream::StreamFrame, r: &M3, p: &V3, pts: &[V3]| -> (usize, Option<[f64; 4]>) {
        let d = f.depth_m();
        let mut bb = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        let mut n = 0;
        for q in pts {
            let w = sub(q, p);
            let c = [0, 1, 2].map(|k| r[0][k] * w[0] + r[1][k] * w[1] + r[2][k] * w[2]); // Rᵀ(q − p)
            if c[2] < 0.1 {
                continue;
            }
            let (u, v) = (f.k[0] * c[0] / c[2] + f.k[2], f.k[1] * c[1] / c[2] + f.k[3]);
            if u < 0.0 || v < 0.0 || u >= f.w as f64 || v >= f.h as f64 {
                continue;
            }
            let dz = d[v as usize * f.w + u as usize] as f64;
            if dz > 0.0 && c[2] > dz + 0.10 {
                continue;
            }
            n += 1;
            bb = [bb[0].min(u), bb[1].min(v), bb[2].max(u), bb[3].max(v)];
        }
        (n, (n >= 20).then_some(bb))
    };
    let mut hits = vec![];
    let mut rd = stream::StreamReader::open(&a[2]);
    let mut seq = 0usize;
    while let Some(f) = rd.next() {
        seq += 1;
        let Some(ps) = poses.iter().find(|x| (x.0 - f.t).abs() < 1e-4) else { continue };
        let mut boxes = vec![];
        let mut score = usize::MAX;
        for o in &objs {
            let (n, b) = project(&f, &ps.1, &ps.2, o);
            // 모든 물체가 보이는 장면을 우선(가장 덜 보이는 물체의 점 비율)
            score = score.min(n * 1000 / o.len().max(1));
            boxes.push(b);
        }
        if boxes.iter().all(|b| b.is_some()) {
            hits.push(Hit { seq, t: f.t, boxes, score });
        }
    }
    hits.sort_by(|a, b| b.score.cmp(&a.score));
    let mut pick: Vec<&Hit> = vec![];
    for h in &hits {
        if pick.len() < nshow && pick.iter().all(|p| (p.t - h.t).abs() > 2.0) {
            pick.push(h);
        }
    }
    println!("물체 {:?}: 점 {:?}, 모두 보이는 장면 {} 개", ids, objs.iter().map(|o| o.len()).collect::<Vec<_>>(), hits.len());
    // 2 차: 고른 장면 다시 읽어 크롭 저장
    let want: std::collections::HashMap<usize, &Hit> = pick.iter().map(|h| (h.seq, *h)).collect();
    let mut rd = stream::StreamReader::open(&a[2]);
    let mut seq = 0usize;
    let cols = [[255u8, 0, 0], [0, 255, 0], [0, 128, 255], [255, 255, 0]];
    while let Some(f) = rd.next() {
        seq += 1;
        let Some(h) = want.get(&seq) else { continue };
        let mut im = image::RgbImage::from_raw(f.w as u32, f.h as u32, f.rgb.clone()).unwrap();
        let mut u = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (bi, b) in h.boxes.iter().enumerate() {
            let b = b.unwrap();
            u = [u[0].min(b[0]), u[1].min(b[1]), u[2].max(b[2]), u[3].max(b[3])];
            let c = image::Rgb(cols[bi % 4]);
            let (x0, y0, x1, y1) = (b[0] as u32, b[1] as u32, (b[2] as u32).min(f.w as u32 - 1), (b[3] as u32).min(f.h as u32 - 1));
            for x in x0..=x1 {
                im.put_pixel(x, y0, c);
                im.put_pixel(x, y1, c);
            }
            for y in y0..=y1 {
                im.put_pixel(x0, y, c);
                im.put_pixel(x1, y, c);
            }
        }
        let m = 0.6 * (u[2] - u[0]).max(u[3] - u[1]) + 20.0;
        let (x0, y0) = ((u[0] - m).max(0.0) as u32, (u[1] - m).max(0.0) as u32);
        let (x1, y1) = ((u[2] + m).min(f.w as f64 - 1.0) as u32, (u[3] + m).min(f.h as f64 - 1.0) as u32);
        let cr = image::imageops::crop_imm(&im, x0, y0, x1 - x0, y1 - y0).to_image();
        let cr = image::imageops::resize(&cr, cr.width() * 2, cr.height() * 2, image::imageops::FilterType::Triangle);
        let out = format!("{pre}_t{:.1}.png", h.t % 1000.0);
        cr.save(&out).unwrap();
        println!("  {out}  (장면 {seq}, 가장 덜 보인 물체 점 {:.1} %)", h.score as f64 / 10.0);
    }
}
