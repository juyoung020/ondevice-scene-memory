//! 루프 · 재위치 신경망 경로 확인 · 벤치(PC). 파이썬 기준 자료(저장소 밖 일회성 스크립트가 만든 .npy · .tsv)와
//! Rust 결과를 견주고, TUM 재생으로 후보 찾기 Recall · 검증 성공률을 잰다. 만드는 법은 docs/loop_closure_npu.md.
//!
//! cargo run --release -p slam --features onnx --example lc_bench -- <명령> …
//!   xfeat-post <vb>                 원본 신경망 출력(maps/*.npy) → Rust 후처리 · 짝 = 파이썬 원본과 비교
//!   xfeat-e2e  <vb> <xfeat.onnx>     사진 → Rust 전처리 → ONNX → Rust 후처리 · 짝 = 파이썬 원본과 비교
//!   verify     <vb> [xfeat.onnx]     흐림 0 · 5 · 10 · 15 검증 성공률(지금 BRIEF · 방향 BRIEF · ORB · XFeat 실수 · 8 비트 흉내) + 파이썬 기준 기하와 비트 비교
//!   place      <pl> <megaloc.onnx> <head.bin>   전처리 · 기술자 = 파이썬 원본과 비교, Recall@1/@5
use slam::lcverify::{self, Intr, Params as VParams, V3};
use slam::xfeat::{self, XFeatures};
use std::time::Instant;

// ---------- .npy 읽기(리틀 엔디언, C 순서) ----------
struct Npy {
    shape: Vec<usize>,
    descr: String,
    data: Vec<u8>,
}
fn npy(path: &str) -> Npy {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(&b[..6] == b"\x93NUMPY", "{path}: npy 아님");
    let (hl, start) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12) };
    let h = std::str::from_utf8(&b[start..start + hl]).unwrap();
    let descr = h.split("'descr':").nth(1).unwrap().split('\'').nth(1).unwrap().to_string();
    assert!(!h.contains("'fortran_order': True"));
    let sh = h.split("'shape':").nth(1).unwrap();
    let sh = &sh[sh.find('(').unwrap() + 1..sh.find(')').unwrap()];
    let shape = sh.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
    Npy { shape, descr, data: b[start + hl..].to_vec() }
}
impl Npy {
    fn f32(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        self.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn f64(&self) -> Vec<f64> {
        assert_eq!(self.descr, "<f8");
        self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn i64(&self) -> Vec<i64> {
        assert_eq!(self.descr, "<i8");
        self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

fn rgb(path: &str) -> (Vec<u8>, usize, usize) {
    let im = image::open(path).unwrap_or_else(|e| panic!("{path}: {e}")).to_rgb8();
    let (w, h) = im.dimensions();
    (im.into_raw(), w as usize, h as usize)
}
fn depth16(path: &str) -> Vec<u16> {
    match image::open(path).unwrap_or_else(|e| panic!("{path}: {e}")) {
        image::DynamicImage::ImageLuma16(b) => b.into_raw(),
        other => panic!("{path}: 16 비트 깊이 아님 {:?}", other.color()),
    }
}

const K: Intr = Intr { fx: 517.3, fy: 516.5, cx: 318.6, cy: 255.3 };
const BL: [u32; 4] = [0, 5, 10, 15];

/// pairs.tsv 한 줄
struct Pair {
    seq: String,
    i: usize,
    q_rgb: String,
    d_rgb: String,
    q_depth: String,
    d_depth: String,
    tq: [f64; 16],
    td: [f64; 16],
    gq: V3,
    gd: V3,
}
fn pairs(vb: &str) -> Vec<Pair> {
    std::fs::read_to_string(format!("{vb}/pairs.tsv"))
        .unwrap()
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let f = |k: usize| c[k].parse::<f64>().unwrap();
            Pair {
                seq: c[0].into(),
                i: c[1].parse().unwrap(),
                q_rgb: c[3].into(),
                d_rgb: c[4].into(),
                q_depth: c[5].into(),
                d_depth: c[6].into(),
                tq: std::array::from_fn(|k| f(7 + k)),
                td: std::array::from_fn(|k| f(23 + k)),
                gq: [f(39), f(40), f(41)],
                gd: [f(42), f(43), f(44)],
            }
        })
        .collect()
}
fn query_png(vb: &str, p: &Pair, l: u32) -> String {
    if l == 0 { p.q_rgb.clone() } else { format!("{vb}/img/{}_{}_L{l}.png", p.seq, p.i) }
}

/// 정답 db 카메라 → 질의 카메라(lcsel: inv(Tq) · Td)
fn gt_rel(p: &Pair) -> ([[f64; 3]; 3], V3) {
    let r = |t: &[f64; 16]| -> [[f64; 3]; 3] { [[t[0], t[1], t[2]], [t[4], t[5], t[6]], [t[8], t[9], t[10]]] };
    let (rq, rd) = (r(&p.tq), r(&p.td));
    let (tq, td) = ([p.tq[3], p.tq[7], p.tq[11]], [p.td[3], p.td[7], p.td[11]]);
    // inv(Tq)·Td: R = Rqᵀ Rd, t = Rqᵀ (td − tq)
    let mut rr = [[0.0; 3]; 3];
    let mut t = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            rr[i][j] = rq[0][i] * rd[0][j] + rq[1][i] * rd[1][j] + rq[2][i] * rd[2][j];
        }
        t[i] = rq[0][i] * (td[0] - tq[0]) + rq[1][i] * (td[1] - tq[1]) + rq[2][i] * (td[2] - tq[2]);
    }
    (rr, t)
}
fn correct(p: &Pair, r: &[[f64; 3]; 3], t: &V3) -> bool {
    let (gr, gt) = gt_rel(p);
    let (dt, da) = lcverify::pose_error(&gr, &gt, r, t);
    dt < 0.10 && da < 5.0
}

fn backproj(depth: &[u16], k: &[[f32; 2]]) -> Vec<Option<V3>> {
    k.iter().map(|&p| lcverify::backproject(depth, 640, 480, 5000.0, &K, p, 0.1, 4.0)).collect()
}

/// 검증 설정(시험용 환경 변수 LC_ITERS · LC_TOL · LC_BASE 로 바꿈, 기본 = lcverify 기본)
fn vparams() -> VParams {
    let mut p = VParams::default();
    let e = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    if let Some(v) = e("LC_ITERS") { p.iters = v as u32; }
    if let Some(v) = e("LC_TOL") { p.len_tol = v; }
    if let Some(v) = e("LC_BASE") { p.min_base = v; }
    p
}

/// 짝 + 깊이 → (4 자유도 검증 성공, 6 자유도 PnP(RTAB-Map 이식) 성공)
fn judge(p: &Pair, kq: &[[f32; 2]], kd: &[[f32; 2]], dq: &[u16], dd: &[u16]) -> (bool, bool, Option<lcverify::Verdict>) {
    let xq = backproj(dq, kq);
    let xd = backproj(dd, kd);
    let v = lcverify::verify(kq, &xq, &xd, p.gq, p.gd, &K, &vparams());
    let ok4 = v.as_ref().is_some_and(|v| v.accepted && correct(p, &v.r, &v.t));
    // 6 자유도 PnP RANSAC(100 회, 2 화소), 대상 3D → 질의 2D
    let (mut op, mut ip) = (Vec::new(), Vec::new());
    for (i, x) in xd.iter().enumerate() {
        if let Some(x) = x {
            op.push([x[0] as f32, x[1] as f32, x[2] as f32]);
            ip.push(kq[i]);
        }
    }
    let mut ok6 = false;
    if op.len() >= 6
        && let Some((tf, inl)) = slam::geom::pnp_ransac_k(&op, &ip, &[K.fx, K.fy, K.cx, K.cy], 100, 2.0)
        && inl.len() >= 20
    {
        let m = tf.0;
        let r = [[m[0] as f64, m[1] as f64, m[2] as f64], [m[4] as f64, m[5] as f64, m[6] as f64], [m[8] as f64, m[9] as f64, m[10] as f64]];
        ok6 = correct(p, &r, &[m[3] as f64, m[7] as f64, m[11] as f64]);
    }
    (ok4, ok6, v)
}

fn matches(vb: &str, name: &str) -> (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<i64>) {
    let to2 = |v: Vec<f32>| v.chunks_exact(2).map(|c| [c[0], c[1]]).collect::<Vec<_>>();
    (to2(npy(&format!("{vb}/{name}_kq.npy")).f32()), to2(npy(&format!("{vb}/{name}_kd.npy")).f32()), npy(&format!("{vb}/{name}_off.npy")).i64())
}

fn pct(v: &[bool]) -> f64 {
    100.0 * v.iter().filter(|&&b| b).count() as f64 / v.len().max(1) as f64
}

// ---------- XFeat ----------
fn xfeat_from_maps(m: &[f32]) -> XFeatures {
    let plane = xfeat::HC * xfeat::WC;
    let maps = xfeat::Maps { feats: &m[..64 * plane], kpts: &m[64 * plane..129 * plane], heat: &m[129 * plane..130 * plane] };
    let (mut sc, mut out) = (xfeat::Scratch::default(), XFeatures::default());
    xfeat::detect(&maps, 1000, 0.05, &mut sc, &mut out);
    out
}

fn cmp_feats(tag: &str, r: &XFeatures, kp: &[f32], desc: &[f32]) -> (usize, usize, f32) {
    let n = kp.len() / 2;
    let same = (0..n.min(r.len())).filter(|&i| r.kps[i] == [kp[2 * i], kp[2 * i + 1]]).count();
    let mut dmax = 0f32;
    if same == n && r.len() == n {
        for i in 0..r.desc.len() {
            dmax = dmax.max((r.desc[i] - desc[i]).abs());
        }
    } else {
        let set: std::collections::HashSet<(u32, u32)> = kp.chunks_exact(2).map(|c| (c[0].to_bits(), c[1].to_bits())).collect();
        let inset = r.kps.iter().filter(|k| set.contains(&(k[0].to_bits(), k[1].to_bits()))).count();
        let diff: Vec<usize> = (0..n.min(r.len())).filter(|&i| r.kps[i] != [kp[2 * i], kp[2 * i + 1]]).collect();
        let sc: Vec<f32> = diff.iter().map(|&i| r.scores[i]).collect();
        eprintln!("{tag}: 점 수 Rust {} 파이썬 {n}, 같은 자리 {same}, 집합으로 같은 점 {inset}, 다른 자리 {diff:?} 점수 {sc:?}", r.len());
    }
    (same, n, dmax)
}

fn match_set(a: &XFeatures, b: &XFeatures) -> Vec<[f32; 4]> {
    let (mut sim, mut mm) = (Vec::new(), Vec::new());
    xfeat::match_mnn(a, b, -1.0, &mut sim, &mut mm);
    mm.iter().map(|&(i, j)| [a.kps[i as usize][0], a.kps[i as usize][1], b.kps[j as usize][0], b.kps[j as usize][1]]).collect()
}
fn overlap(r: &[[f32; 4]], py: &[f32]) -> (usize, usize, usize) {
    let p: std::collections::HashSet<[u32; 4]> = py.chunks_exact(4).map(|c| [c[0].to_bits(), c[1].to_bits(), c[2].to_bits(), c[3].to_bits()]).collect();
    let both = r.iter().filter(|m| p.contains(&[m[0].to_bits(), m[1].to_bits(), m[2].to_bits(), m[3].to_bits()])).count();
    (both, r.len(), p.len())
}

fn map_files(vb: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(format!("{vb}/maps"))
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with("_m.npy"))
        .map(|n| n.trim_end_matches("_m.npy").to_string())
        .collect();
    v.sort();
    v
}

fn xfeat_post(vb: &str) {
    let (mut kp_same, mut kp_all, mut dmax, mut mb, mut mr, mut mp) = (0, 0, 0f32, 0, 0, 0);
    let mut t_det = 0f64;
    let mut n_img = 0;
    for stem in map_files(vb) {
        let mut f = Vec::new();
        for tag in ["q", "d"] {
            let m = npy(&format!("{vb}/maps/{stem}_{tag}.npy")).f32();
            let t0 = Instant::now();
            let r = xfeat_from_maps(&m);
            t_det += t0.elapsed().as_secs_f64();
            n_img += 1;
            let kp = npy(&format!("{vb}/maps/{stem}_{tag}_kp.npy")).f32();
            let de = npy(&format!("{vb}/maps/{stem}_{tag}_desc.npy")).f32();
            let (s, n, d) = cmp_feats(&format!("{stem}_{tag}"), &r, &kp, &de);
            kp_same += s;
            kp_all += n;
            dmax = dmax.max(d);
            f.push(r);
        }
        let ms = match_set(&f[0], &f[1]);
        let (b, r, p) = overlap(&ms, &npy(&format!("{vb}/maps/{stem}_m.npy")).f32());
        mb += b;
        mr += r;
        mp += p;
    }
    println!("xfeat-post: 점 같은 자리 {kp_same}/{kp_all}, 기술자 최대 차 {dmax:.2e}, 짝 공통 {mb} (Rust {mr}, 파이썬 {mp}), 후처리(점 · 기술자) {:.2} ms/장(PC 1 스레드)", t_det * 1e3 / n_img as f64);
}

#[cfg(feature = "onnx")]
fn xfeat_run(net: &mut slam::nnrun::Net, rgbv: &[u8], buf: &mut Vec<f32>, tm: &mut [f64; 3]) -> XFeatures {
    let t0 = Instant::now();
    let mut x = Vec::new();
    xfeat::normalize_rgb(rgbv, &mut x);
    let t1 = Instant::now();
    if net.wants_u8() {
        let mut q = Vec::new();
        xfeat::quantize_input(&x, &mut q);
        net.run(&q, buf);
    } else {
        net.run(slam::nnrun::f32_bytes(&x), buf);
    }
    let t2 = Instant::now();
    let maps = xfeat::Maps { feats: net.out(buf, "feats"), kpts: net.out(buf, "kpts"), heat: net.out(buf, "heat") };
    let (mut sc, mut out) = (xfeat::Scratch::default(), XFeatures::default());
    xfeat::detect(&maps, 1000, 0.05, &mut sc, &mut out);
    tm[0] += (t1 - t0).as_secs_f64();
    tm[1] += (t2 - t1).as_secs_f64();
    tm[2] += t2.elapsed().as_secs_f64();
    out
}

#[cfg(feature = "onnx")]
fn xfeat_e2e(vb: &str, onnx: &str) {
    let mut net = slam::nnrun::Net::open(onnx, 8, 0).unwrap();
    let ps = pairs(vb);
    let mut buf = Vec::new();
    let mut tm = [0f64; 3];
    let (mut ks, mut ka, mut mb, mut mr, mut mp, mut gmax) = (0, 0, 0, 0, 0, 0f32);
    for stem in map_files(vb) {
        let pid: usize = stem.split('_').next().unwrap().parse().unwrap();
        let l: u32 = stem.split("_L").nth(1).unwrap().parse().unwrap();
        let p = &ps[pid];
        let mut f = Vec::new();
        for (tag, path) in [("q", query_png(vb, p, l)), ("d", p.d_rgb.clone())] {
            let (im, _, _) = rgb(&path);
            let mut x = Vec::new();
            xfeat::normalize_rgb(&im, &mut x);
            let ng = npy(&format!("{vb}/maps/{stem}_{tag}_ng.npy")).f32();
            gmax = gmax.max(x.iter().zip(&ng).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
            let r = xfeat_run(&mut net, &im, &mut buf, &mut tm);
            let kp = npy(&format!("{vb}/maps/{stem}_{tag}_kp.npy")).f32();
            let de = npy(&format!("{vb}/maps/{stem}_{tag}_desc.npy")).f32();
            let (s, n, _) = cmp_feats(&format!("e2e {stem}_{tag}"), &r, &kp, &de);
            ks += s;
            ka += n;
            f.push(r);
        }
        let ms = match_set(&f[0], &f[1]);
        let (b, r, pp) = overlap(&ms, &npy(&format!("{vb}/maps/{stem}_m.npy")).f32());
        mb += b;
        mr += r;
        mp += pp;
    }
    let n = (2 * map_files(vb).len()) as f64;
    println!(
        "xfeat-e2e: 정규화 회색 최대 차 {gmax:.2e}, 점 같은 자리 {ks}/{ka}, 짝 공통 {mb} (Rust {mr}, 파이썬 {mp}); ms/장 전처리 {:.2} · ONNX {:.1} · 후처리 {:.2}",
        tm[0] * 1e3 / n,
        tm[1] * 1e3 / n,
        tm[2] * 1e3 / n
    );
}

// ---------- 검증 벤치 ----------
fn brief_frame(path: &str, depth: &str, oriented: bool) -> slam::Frame {
    let (c, w, h) = rgb(path);
    let gray = slam::io::to_gray(&c, true);
    let raw = depth16(depth);
    let (dmm, draw) = slam::io::split_depth(raw, 5000.0);
    slam::Frame {
        id: 0,
        t: 0.0,
        gray,
        color: c,
        color_is_rgb: true,
        depth_mm: dmm,
        depth_raw: draw,
        depth_per_m: 5000.0,
        cam: slam::Camera { fx: K.fx, fy: K.fy, cx: K.cx, cy: K.cy, width: w as u32, height: h as u32, local: slam::io::optical_rotation(), oriented },
    }
}

fn verify(vb: &str, onnx: Option<&str>) {
    let ps = pairs(vb);
    let ext = slam::features::Extractor::new();
    let local = slam::io::optical_rotation();
    #[cfg(feature = "onnx")]
    let mut net = onnx.map(|o| slam::nnrun::Net::open(o, 8, 0).unwrap());
    #[cfg(not(feature = "onnx"))]
    let _ = onnx;
    let mut rows: Vec<(String, u32, f64, f64)> = Vec::new();
    let mut bit = (0usize, 0usize, 0f64, 0usize);
    let mut times: std::collections::BTreeMap<&str, (f64, usize)> = Default::default();
    let bl: Vec<u32> = std::env::var("LC_BLUR").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(BL.to_vec());
    for &l in &bl {
        let mut res: std::collections::BTreeMap<&str, (Vec<bool>, Vec<bool>)> = Default::default();
        let dm: Vec<(&str, (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<i64>))> = ["orb", "xf32", "xf8"].iter().map(|m| (*m, matches(vb, &format!("{m}_L{l}")))).collect();
        let refp = npy(&format!("{vb}/ref_xf32_L{l}.npy")).f64();
        #[cfg(feature = "onnx")]
        let mut xbuf = Vec::new();
        for (pi, p) in ps.iter().enumerate() {
            let dq = depth16(&p.q_depth);
            let dd = depth16(&p.d_depth);
            for (m, (kq, kd, off)) in &dm {
                let (a, b) = (off[pi] as usize, off[pi + 1] as usize);
                let t0 = Instant::now();
                let (o4, o6, v) = judge(p, &kq[a..b], &kd[a..b], &dq, &dd);
                let e = times.entry(if *m == "orb" { "verify4+pnp" } else { "verify4+pnp " }).or_default();
                e.0 += t0.elapsed().as_secs_f64();
                e.1 += 1;
                let r = res.entry(m).or_default();
                r.0.push(o4);
                r.1.push(o6);
                if *m == "xf32" {
                    // 파이썬 기준 기하와 비트 비교: 맞은 점 수 · 회전 · 이동
                    let rr = &refp[pi * 15..(pi + 1) * 15];
                    bit.1 += 1;
                    match &v {
                        None => bit.0 += (rr[0] < 0.0) as usize,
                        Some(v) => {
                            let mut d = (v.inliers as f64 - rr[0]).abs() + (v.hyp_inliers as f64 - rr[1]).abs();
                            for i in 0..3 {
                                for j in 0..3 {
                                    d = d.max((v.r[i][j] - rr[2 + i * 3 + j]).abs());
                                }
                                d = d.max((v.t[i] - rr[11 + i]).abs());
                            }
                            if d == 0.0 {
                                bit.0 += 1;
                            } else {
                                bit.2 = bit.2.max(d);
                                bit.3 += 1;
                            }
                        }
                    }
                }
            }
            // BRIEF · 방향 BRIEF: GFTT + BRIEF → 상호 최근접 짝 → PnP(klt::brief_pnp, 위치 추적 되찾기와 같은 것)(LC_DUMPS_ONLY=1 이면 건너뜀)
            let dumps_only = std::env::var("LC_DUMPS_ONLY").is_ok();
            for (name, ori) in [("brief", false), ("brief_ori", true)] {
                if dumps_only {
                    continue;
                }
                let t0 = Instant::now();
                let fq = brief_frame(&query_png(vb, p, l), &p.q_depth, ori);
                let fd = brief_frame(&p.d_rgb, &p.d_depth, ori);
                let (a, b) = (ext.extract(&fd), ext.extract(&fq));
                let t1 = Instant::now();
                let info = slam::klt::brief_pnp(&a, &b, &fq.cam, 20, 100, 2.0);
                let t2 = Instant::now();
                let e = times.entry(if ori { "brief_ori 특징(두 장)" } else { "brief 특징(두 장)" }).or_default();
                e.0 += (t1 - t0).as_secs_f64();
                e.1 += 1;
                let e = times.entry("brief 짝+PnP").or_default();
                e.0 += (t2 - t1).as_secs_f64();
                e.1 += 1;
                let ok = info.is_some_and(|(tf, _)| {
                    let tr = tf.to_pose();
                    // 몸체 축 (질의의 대상 기준 위치) → 광학 축 db → 질의 변환
                    let tdq = local.inverse() * tr * local;
                    let tqd = tdq.inverse();
                    let m = tqd.rotation.to_rotation_matrix();
                    let mm = m.matrix();
                    let r = [[mm[(0, 0)], mm[(0, 1)], mm[(0, 2)]], [mm[(1, 0)], mm[(1, 1)], mm[(1, 2)]], [mm[(2, 0)], mm[(2, 1)], mm[(2, 2)]]];
                    correct(p, &r, &[tqd.translation.x, tqd.translation.y, tqd.translation.z])
                });
                let r = res.entry(name).or_default();
                r.1.push(ok);
            }
            #[cfg(feature = "onnx")]
            if let Some(net) = net.as_mut() {
                let mut tm = [0f64; 3];
                let (iq, _, _) = rgb(&query_png(vb, p, l));
                let (id, _, _) = rgb(&p.d_rgb);
                let fq = xfeat_run(net, &iq, &mut xbuf, &mut tm);
                let fd = xfeat_run(net, &id, &mut xbuf, &mut tm);
                let t0 = Instant::now();
                let ms = match_set(&fq, &fd);
                let e = times.entry("xfeat 짝(1000×1000)").or_default();
                e.0 += t0.elapsed().as_secs_f64();
                e.1 += 1;
                for (k, v) in [("xfeat 전처리", tm[0]), ("xfeat ONNX", tm[1]), ("xfeat 후처리", tm[2])] {
                    let e = times.entry(k).or_default();
                    e.0 += v / 2.0;
                    e.1 += 1;
                }
                let kq: Vec<[f32; 2]> = ms.iter().map(|m| [m[0], m[1]]).collect();
                let kd: Vec<[f32; 2]> = ms.iter().map(|m| [m[2], m[3]]).collect();
                let (o4, o6, _) = judge(p, &kq, &kd, &dq, &dd);
                let r = res.entry("xf32_rust").or_default();
                r.0.push(o4);
                r.1.push(o6);
            }
        }
        for (m, (a, b)) in &res {
            rows.push((m.to_string(), l, if a.is_empty() { f64::NAN } else { pct(a) }, pct(b)));
        }
        eprintln!("흐림 {l} 끝");
    }
    println!("검증 성공률(%) — 4DoF = 중력 고정 RANSAC + 다듬기(lcverify), PnP = 6 자유도 PnP RANSAC(BRIEF 는 RegistrationVis 그대로)");
    println!("방식\t흐림\t4DoF\tPnP");
    for (m, l, a, b) in &rows {
        println!("{m}\t{l}\t{a:.1}\t{b:.1}");
    }
    println!("파이썬 기준 기하와 비트 같음: {}/{} (다른 것 {}, 최대 차 {:.2e})", bit.0, bit.1, bit.3, bit.2);
    for (k, (s, n)) in &times {
        println!("시간 {k}: {:.2} ms(PC 1 스레드, {n} 번 평균)", s * 1e3 / *n as f64);
    }
}

// ---------- 장소 인식 ----------
#[cfg(feature = "onnx")]
fn place(pl: &str, onnx: &str, head: &str) {
    use slam::place::*;
    let head = Head::load(head).unwrap();
    let mut net = slam::nnrun::Net::open(onnx, 8, 0).unwrap();
    // 1) 전처리: TUM 원본 → INTER_AREA → 파이썬(cv2) 결과와 바이트 비교
    let list: Vec<String> = std::fs::read_to_string(format!("{pl}/frames.txt")).unwrap().lines().map(|s| s.to_string()).collect();
    let small_py = std::fs::read(format!("{pl}/small_L0.u8")).unwrap();
    let sz = IN_W * IN_H * 3;
    let (mut diff_px, mut diff_max) = (0usize, 0i32);
    let mut small = Vec::new();
    let mut patches = Vec::new();
    let t0 = Instant::now();
    for (k, path) in list.iter().enumerate() {
        let (im, w, h) = rgb(path.split('\t').nth(1).unwrap());
        resize_area(&im, w, h, 3, IN_W, IN_H, &mut small);
        for (a, b) in small.iter().zip(&small_py[k * sz..(k + 1) * sz]) {
            if a != b {
                diff_px += 1;
                diff_max = diff_max.max((*a as i32 - *b as i32).abs());
            }
        }
    }
    let t_resize = t0.elapsed().as_secs_f64() * 1e3 / list.len() as f64;
    println!("place 전처리: 줄인 사진 {} 장, 다른 바이트 {diff_px} / {} (최대 차 {diff_max}), {t_resize:.2} ms/장", list.len(), list.len() * sz);
    // 2) 흐림마다 기술자: 파이썬이 만든 줄인 사진 → Rust 조각 → ONNX → Rust 집계 · 투영, 파이썬 기술자와 코사인
    let gt = npy(&format!("{pl}/gt.npy")); // [n, n] u8: 1 = 정답, 2 = 옛 장면(후보), 3 = 둘 다
    let n = gt.shape[0];
    let mut buf = Vec::new();
    let mut desc0: Vec<Vec<f32>> = Vec::new();
    let mut tm = [0f64; 3];
    for l in [0u32, 10, 25, 50] {
        let sm = std::fs::read(format!("{pl}/small_L{l}.u8")).unwrap();
        let py = npy(&format!("{pl}/desc_fp32_L{l}.npy"));
        let pyd = py.f32();
        let dim = py.shape[1];
        assert_eq!(dim, head.dim);
        let mut cmin = 1f64;
        let mut descs = Vec::with_capacity(n);
        for k in 0..n {
            let t0 = Instant::now();
            to_patches(&sm[k * sz..(k + 1) * sz], &mut patches);
            let t1 = Instant::now();
            net.run(slam::nnrun::f32_bytes(&patches), &mut buf);
            let t2 = Instant::now();
            let mut d = Vec::new();
            head.describe(net.out(&buf, "f"), net.out(&buf, "s"), net.out(&buf, "t"), &mut d);
            tm[0] += (t1 - t0).as_secs_f64();
            tm[1] += (t2 - t1).as_secs_f64();
            tm[2] += t2.elapsed().as_secs_f64();
            let c: f64 = d.iter().zip(&pyd[k * dim..(k + 1) * dim]).map(|(a, b)| *a as f64 * *b as f64).sum();
            cmin = cmin.min(c);
            descs.push(d);
        }
        if l == 0 {
            desc0 = descs.clone();
        }
        // Recall@1 · @5: 질의 = 흐림 l, 후보 = 흐림 0 의 옛 장면
        let (mut r1, mut r5, mut nq) = (0, 0, 0);
        let mut seq_start = 0;
        let seqs: Vec<usize> = npy(&format!("{pl}/seq_len.npy")).i64().iter().map(|&x| x as usize).collect();
        for &sl in &seqs {
            for qi in seq_start..seq_start + sl {
                let row = &gt.data[qi * n..(qi + 1) * n];
                if !row.iter().any(|&g| g & 1 == 1) {
                    continue;
                }
                let mut db = PlaceDb::new(head.dim);
                for di in seq_start..seq_start + sl {
                    if row[di] & 2 == 2 {
                        db.add(di as i64, &desc0[di]);
                    }
                }
                let top = db.topk(&descs[qi], 5, |_| true);
                nq += 1;
                r1 += top.first().is_some_and(|t| row[t.0 as usize] & 1 == 1) as usize;
                r5 += top.iter().any(|t| row[t.0 as usize] & 1 == 1) as usize;
            }
            seq_start += sl;
        }
        println!("place 흐림 {l}: 파이썬 기술자와 코사인 최소 {cmin:.7}, R@1 {:.1} · R@5 {:.1} % (질의 {nq})", 100.0 * r1 as f64 / nq as f64, 100.0 * r5 as f64 / nq as f64);
    }
    let k = (4 * n) as f64;
    println!("place ms/장(PC): 조각 {:.2} · ONNX {:.1} · 집계+투영 {:.2}", tm[0] * 1e3 / k, tm[1] * 1e3 / k, tm[2] * 1e3 / k);
}

// ---------- 보드 확인(scripts/board_lc_check.sh 가 부름) ----------
/// 사진 → 모델 입력 바이트: megaloc = 조각 f32, xfeat8 = u8 회색(NPU 판 · u8 ONNX)
fn prep_in(kind: &str, img: &str, out: &str) {
    let (im, w, h) = rgb(img);
    let bytes: Vec<u8> = match kind {
        "megaloc" => {
            let (mut s, mut p) = (Vec::new(), Vec::new());
            slam::place::prepare(&im, w, h, &mut s, &mut p);
            p.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        "xfeat8" => {
            let (mut x, mut q) = (Vec::new(), Vec::new());
            xfeat::normalize_rgb(&im, &mut x);
            xfeat::quantize_input(&x, &mut q);
            q
        }
        _ => panic!("prep-in megaloc|xfeat8"),
    };
    std::fs::write(out, bytes).unwrap();
}

#[cfg(any(feature = "onnx", feature = "dxrt"))]
fn read_f32(path: &str) -> Option<Vec<f32>> {
    std::fs::read(path).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// 모델 하나를 입력 하나로: 출력마다 기준(ref_dir/<이름>.f32)과 코사인 · 최대 차, 그리고 runs 번 올리기 → 받기 평균 시간.
/// --save 면 출력을 기준으로 저장(PC ONNX 에서). --head <MLH1> 이면 MegaLoc 기술자 코사인도
#[cfg(any(feature = "onnx", feature = "dxrt"))]
fn npu_check(a: &[String]) {
    let (model, inp, refd) = (&a[2], &a[3], &a[4]);
    let save = a.iter().any(|s| s == "--save");
    let runs: usize = a.iter().position(|s| s == "--runs").map(|i| a[i + 1].parse().unwrap()).unwrap_or(50);
    let head = a.iter().position(|s| s == "--head").map(|i| slam::place::Head::load(&a[i + 1]).unwrap());
    let mut net = slam::nnrun::Net::open(model, 4, 0).unwrap_or_else(|e| panic!("{e}"));
    let input = std::fs::read(inp).unwrap();
    let mut buf = Vec::new();
    net.run(&input, &mut buf);
    std::fs::create_dir_all(refd).ok();
    let names: Vec<&str> = ["f", "s", "t", "feats", "kpts", "heat"].into_iter().filter(|n| net.has(n)).collect();
    for n in &names {
        let o = net.out(&buf, n);
        let path = format!("{refd}/{n}.f32");
        if save {
            std::fs::write(&path, o.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        } else if let Some(r) = read_f32(&path) {
            let (mut d, mut ab, mut aa, mut bb) = (0f32, 0f64, 0f64, 0f64);
            for (x, y) in o.iter().zip(&r) {
                d = d.max((x - y).abs());
                ab += (*x as f64) * (*y as f64);
                aa += (*x as f64) * (*x as f64);
                bb += (*y as f64) * (*y as f64);
            }
            println!("출력 {n}: 코사인 {:.5}, 최대 차 {d:.3e}", ab / (aa.sqrt() * bb.sqrt()).max(1e-30));
        }
    }
    if let Some(h) = head
        && !save
        && let (Some(f), Some(s), Some(t)) = (read_f32(&format!("{refd}/f.f32")), read_f32(&format!("{refd}/s.f32")), read_f32(&format!("{refd}/t.f32")))
    {
        let (mut d0, mut d1) = (Vec::new(), Vec::new());
        h.describe(&f, &s, &t, &mut d0);
        h.describe(net.out(&buf, "f"), net.out(&buf, "s"), net.out(&buf, "t"), &mut d1);
        println!("MegaLoc 기술자 코사인(기준 대비) {:.5}", slam::place::dot(&d0, &d1));
    }
    let t0 = Instant::now();
    for _ in 0..runs {
        net.run(&input, &mut buf);
    }
    println!("{model}: 올리기 → 받기 {:.2} ms/번({runs} 번, 출력 f32 변환 포함)", t0.elapsed().as_secs_f64() * 1e3 / runs as f64);
}

/// 여러 모델을 함께 올려 secs 초 동안 스레드마다 계속 돌림: 모델별 횟수 · 평균 ms · 오류. 인자: <초> <모델:입력> …
#[cfg(any(feature = "onnx", feature = "dxrt"))]
fn npu_stress(a: &[String]) {
    let secs: f64 = a[2].parse().unwrap();
    let specs: Vec<(String, String)> = a[3..].iter().map(|s| { let (m, i) = s.split_once(':').unwrap(); (m.to_string(), i.to_string()) }).collect();
    // 먼저 모두 연다(적재 실패를 따로 적음)
    let mut nets = Vec::new();
    for (m, i) in &specs {
        match slam::nnrun::Net::open(m, 2, 0) {
            Ok(n) => {
                println!("적재 성공 {m}");
                nets.push((m.clone(), n, std::fs::read(i).unwrap()));
            }
            Err(e) => println!("적재 실패 {m}: {e}"),
        }
    }
    let hs: Vec<_> = nets
        .into_iter()
        .map(|(m, mut n, input)| {
            std::thread::spawn(move || {
                let t0 = Instant::now();
                let (mut k, mut err) = (0usize, None);
                let mut buf = Vec::new();
                while t0.elapsed().as_secs_f64() < secs {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| n.run(&input, &mut buf))) {
                        Ok(()) => k += 1,
                        Err(e) => {
                            err = Some(e.downcast_ref::<String>().cloned().unwrap_or_else(|| "패닉".into()));
                            break;
                        }
                    }
                }
                (m, k, t0.elapsed().as_secs_f64(), err)
            })
        })
        .collect();
    for h in hs {
        let (m, k, s, e) = h.join().unwrap();
        println!("{m}: {k} 번, {:.1} 번/초, 평균 {:.2} ms{}", k as f64 / s, s * 1e3 / k.max(1) as f64, e.map(|e| format!(", 오류: {e}")).unwrap_or_default());
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "prep-in" => prep_in(&a[2], &a[3], &a[4]),
        #[cfg(any(feature = "onnx", feature = "dxrt"))]
        "npu-check" => npu_check(&a),
        #[cfg(any(feature = "onnx", feature = "dxrt"))]
        "npu-stress" => npu_stress(&a),
        "xfeat-post" => xfeat_post(&a[2]),
        #[cfg(feature = "onnx")]
        "xfeat-e2e" => xfeat_e2e(&a[2], &a[3]),
        "verify" => verify(&a[2], a.get(3).map(String::as_str)),
        #[cfg(feature = "onnx")]
        "place" => place(&a[2], &a[3], &a[4]),
        _ => eprintln!("명령: xfeat-post <vb> | xfeat-e2e <vb> <onnx> | verify <vb> [onnx] | place <pl> <onnx> <head.bin>  (ONNX 쓰는 명령은 --features onnx)"),
    }
}
