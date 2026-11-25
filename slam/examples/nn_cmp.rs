//! MegaLoc · XFeat: Rust 실행기 출력과 ONNX Runtime 출력(파이썬 덤프)의 최종 결과 비교.
//!   nn_cmp dump <rgb폴더> <장수> <폴더>                       TUM 사진 → 입력(megaloc_in.f32 · xfeat_in.u8 / xfeat_in.f32) 저장
//!   nn_cmp cmp  <폴더> <megaloc.onnx> <xfeat.onnx> <head.bin>   Rust 실행 → ort 출력(<폴더>/ort_<모델>_<출력>.f32)과 날값 · 기술자 · 장소 1 위 · 점 · 짝 비교
use slam::place::{self, Head};
use slam::xfeat::{self, XFeatures};

fn rd(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn cos(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut d, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
        mx = mx.max((x - y).abs());
    }
    (d / (na.sqrt() * nb.sqrt()).max(1e-30), mx)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "dump" => {
            let n: usize = a[3].parse().unwrap();
            let mut files: Vec<_> = std::fs::read_dir(&a[2]).unwrap().map(|e| e.unwrap().path()).collect();
            files.sort();
            let step = (files.len() / n).max(1);
            let (mut ml, mut xu, mut xf) = (vec![], vec![], vec![]);
            let (mut small, mut patches, mut gray, mut q8) = (vec![], vec![], vec![], vec![]);
            for p in files.iter().step_by(step).take(n) {
                let im = image::open(p).unwrap().to_rgb8();
                let (w, h) = (im.width() as usize, im.height() as usize);
                let rgb = im.into_raw();
                place::prepare(&rgb, w, h, &mut small, &mut patches);
                ml.extend_from_slice(&patches);
                xfeat::normalize_rgb(&rgb, &mut gray);
                xfeat::quantize_input(&gray, &mut q8);
                xu.extend_from_slice(&q8);
                xf.extend_from_slice(&gray);
            }
            let o = &a[4];
            std::fs::create_dir_all(o).unwrap();
            std::fs::write(format!("{o}/megaloc_in.f32"), slam::nnrun::f32_bytes(&ml)).unwrap();
            std::fs::write(format!("{o}/xfeat_in.u8"), &xu).unwrap();
            std::fs::write(format!("{o}/xfeat_in.f32"), slam::nnrun::f32_bytes(&xf)).unwrap();
            println!("{} 장", ml.len() / patches.len());
        }
        "cmp" => {
            let o = &a[2];
            let head = Head::load(&a[5]).unwrap();
            let mut ml = slam::nnrun::Net::open(&a[3], 8, 0).unwrap();
            let mut xn = slam::nnrun::Net::open(&a[4], 8, 0).unwrap();
            let min = rd(&format!("{o}/megaloc_in.f32"));
            let per = 192 * 588;
            let n = min.len() / per;
            let xin: Vec<u8> = if xn.wants_u8() { std::fs::read(format!("{o}/xfeat_in.u8")).unwrap() } else { std::fs::read(format!("{o}/xfeat_in.f32")).unwrap() };
            let xper = xin.len() / n;
            let ref_of = |m: &str, name: &str| rd(&format!("{o}/ort_{m}_{name}.f32"));
            let (rf, rs, rt) = (ref_of("megaloc", "f"), ref_of("megaloc", "s"), ref_of("megaloc", "t"));
            let (xfe, xkp, xhe) = (ref_of("xfeat", "feats"), ref_of("xfeat", "kpts"), ref_of("xfeat", "heat"));
            let (mut out, mut dr, mut dq) = (vec![], vec![], vec![]);
            let (mut raw_ml, mut raw_xf) = ((1f64, 0f32), (1f64, 0f32));
            let (mut d_rust, mut d_ort) = (vec![], vec![]);
            let (mut f_rust, mut f_ort) = (vec![], vec![]);
            let mut sc = xfeat::Scratch::default();
            let (mut tm, mut tx) = (vec![], vec![]);
            for i in 0..n {
                let t0 = std::time::Instant::now();
                ml.run(slam::nnrun::f32_bytes(&min[i * per..(i + 1) * per]), &mut out);
                tm.push(t0.elapsed().as_secs_f64() * 1e3);
                let (f, s, t) = (ml.out(&out, "f"), ml.out(&out, "s"), ml.out(&out, "t"));
                let (fl, sl, tl) = (f.len(), s.len(), t.len());
                for (x, y) in [(f, &rf[i * fl..(i + 1) * fl]), (s, &rs[i * sl..(i + 1) * sl]), (t, &rt[i * tl..(i + 1) * tl])] {
                    let (c, m) = cos(x, y);
                    raw_ml = (raw_ml.0.min(c), raw_ml.1.max(m));
                }
                head.describe(f, s, t, &mut dr);
                head.describe(&rf[i * fl..(i + 1) * fl], &rs[i * sl..(i + 1) * sl], &rt[i * tl..(i + 1) * tl], &mut dq);
                d_rust.push(dr.clone());
                d_ort.push(dq.clone());
                let t0 = std::time::Instant::now();
                xn.run(&xin[i * xper..(i + 1) * xper], &mut out);
                tx.push(t0.elapsed().as_secs_f64() * 1e3);
                let (fe, kp, he) = (xn.out(&out, "feats"), xn.out(&out, "kpts"), xn.out(&out, "heat"));
                let (a1, a2, a3) = (fe.len(), kp.len(), he.len());
                for (x, y) in [(fe, &xfe[i * a1..(i + 1) * a1]), (kp, &xkp[i * a2..(i + 1) * a2]), (he, &xhe[i * a3..(i + 1) * a3])] {
                    let (c, m) = cos(x, y);
                    raw_xf = (raw_xf.0.min(c), raw_xf.1.max(m));
                }
                let mut q = XFeatures::default();
                xfeat::detect(&xfeat::Maps { feats: fe, kpts: kp, heat: he }, 1000, 0.05, &mut sc, &mut q);
                f_rust.push(q);
                let mut q = XFeatures::default();
                xfeat::detect(&xfeat::Maps { feats: &xfe[i * a1..(i + 1) * a1], kpts: &xkp[i * a2..(i + 1) * a2], heat: &xhe[i * a3..(i + 1) * a3] }, 1000, 0.05, &mut sc, &mut q);
                f_ort.push(q);
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            println!("{n} 장");
            println!("MegaLoc 날값: 최소 코사인 {:.7}, 최대 |차| {:.3e}; Rust 중앙값 {:.1} ms", raw_ml.0, raw_ml.1, med(&mut tm));
            let dmin = d_rust.iter().zip(&d_ort).map(|(x, y)| cos(x, y).0).fold(1.0, f64::min);
            // 장소 1 위: 자기 자신 빼고 가장 가까운 장
            let top1 = |d: &Vec<Vec<f32>>, i: usize| (0..n).filter(|&j| j != i).max_by(|&x, &y| place::dot(&d[i], &d[x]).total_cmp(&place::dot(&d[i], &d[y]))).unwrap();
            let same = (0..n).filter(|&i| top1(&d_rust, i) == top1(&d_ort, i)).count();
            println!("MegaLoc 기술자: 최소 코사인 {dmin:.7}, 장소 1 위 같음 {same} / {n}");
            println!("XFeat 날값: 최소 코사인 {:.7}, 최대 |차| {:.3e}; Rust 중앙값 {:.1} ms", raw_xf.0, raw_xf.1, med(&mut tx));
            let (mut kp_same, mut kp_all, mut dmax) = (0usize, 0usize, 0f32);
            for (r, q) in f_rust.iter().zip(&f_ort) {
                kp_all += q.len();
                for (i, p) in r.kps.iter().enumerate() {
                    if let Some(j) = q.kps.iter().position(|k| k == p) {
                        kp_same += 1;
                        let (_, m) = cos(r.d(i), q.d(j));
                        dmax = dmax.max(m);
                    }
                }
            }
            let (mut sim, mut pr, mut pq) = (vec![], vec![], vec![]);
            let (mut m_r, mut m_q, mut m_both) = (0usize, 0usize, 0usize);
            for i in 1..n {
                xfeat::match_mnn(&f_rust[i - 1], &f_rust[i], -1.0, &mut sim, &mut pr);
                xfeat::match_mnn(&f_ort[i - 1], &f_ort[i], -1.0, &mut sim, &mut pq);
                m_r += pr.len();
                m_q += pq.len();
                // 같은 짝: 점 자리로
                let key = |f: &[XFeatures], a: u32, b: u32| (f[i - 1].kps[a as usize].map(f32::to_bits), f[i].kps[b as usize].map(f32::to_bits));
                let set: std::collections::HashSet<_> = pq.iter().map(|&(a, b)| key(&f_ort, a, b)).collect();
                m_both += pr.iter().filter(|&&(a, b)| set.contains(&key(&f_rust, a, b))).count();
            }
            println!("XFeat 점: 같은 자리 {kp_same} / {kp_all}, 같은 점 기술자 최대 |차| {dmax:.2e}; 이웃 장 짝 Rust {m_r} · ort {m_q} · 공통 {m_both}");
        }
        _ => eprintln!("명령: dump | cmp"),
    }
}
