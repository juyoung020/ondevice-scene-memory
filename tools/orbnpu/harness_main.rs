//! PC 비교 하네스(tools/orbnpu/rs_harness.sh 가 ~/orbnpu_rs 로 복사해 빌드). 모자이크(orbnpu_bench mosaic)의 층을
//! orb3rs::extract 에 넣고, FAST 자리를 (a) 원본 fast 또는 (b) ONNX(nnrun) S + orb3npu::fast_npu 로 돌려 덤프(orbnpu_bench 형식).
//! 쓰기: orbnpu_rs <mosdir> <n> <out.bin> [onnx 모델] — 모델이 없으면 CPU fast. NOISE=1: S 를 화소마다 0 또는 1 낮춤(8 비트 흉내 최악)
#[allow(dead_code)]
mod orb3rs {
    pub mod extract;
    pub mod stdsort;
    pub mod types;
}
#[allow(dead_code)]
#[path = "/mnt/c/ondevice-orb-npu/slam/src/nnrun.rs"]
mod nnrun;
#[path = "/mnt/c/ondevice-orb-npu/slam/src/orb3npu.rs"]
mod orb3npu;

mod hook {
    use super::orb3rs::types::KeyPoint;
    use std::sync::RwLock;
    /// 층마다 NPU S(층 크기 u8). 비어 있으면 원본 fast
    pub static S: RwLock<Vec<Vec<u8>>> = RwLock::new(Vec::new());
    #[allow(clippy::too_many_arguments)]
    pub fn fast(level: usize, img: &[u8], iw: usize, x0: usize, y0: usize, cw: usize, ch: usize, th: i32, out: &mut Vec<KeyPoint>) {
        let s = S.read().unwrap();
        if s.is_empty() {
            return super::orb3rs::extract::fast(img, iw, x0, y0, cw, ch, th, out);
        }
        let mut v = Vec::new();
        super::orb3npu::fast_npu(img, iw, x0, y0, cw, ch, th, &s[level], super::orb3npu::MARGIN, &mut v);
        out.extend(v.iter().map(|k| KeyPoint { x: k.x, y: k.y, size: 7.0, angle: -1.0, response: k.response, octave: 0, class_id: -1 }));
    }
}

use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (md, n, outp) = (&a[1], a[2].parse::<usize>().unwrap(), &a[3]);
    let noise = std::env::var("NOISE").is_ok();
    let lay = std::fs::read_to_string(format!("{md}/layout.txt")).unwrap();
    let mut it = lay.lines();
    let hd: Vec<usize> = it.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (mw, mh) = (hd[0], hd[1]);
    let slots: Vec<Vec<usize>> = it.filter(|l| !l.trim().is_empty()).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect();
    let sizes: Vec<(usize, usize)> = slots.iter().map(|s| (s[2] - 38, s[3] - 38)).collect();
    let mut mos = orb3npu::Mosaic::new(&sizes);
    assert_eq!((mos.h, orb3npu::MOS_W), (mh, mw), "배치가 모델과 다름");
    for (s, q) in slots.iter().zip(&mos.slots) {
        assert_eq!((s[0], s[1], s[2], s[3]), *q);
    }
    let mut net = a.get(4).map(|p| nnrun::Net::open(p, 8, 0).unwrap());
    let mut ext = orb3rs::extract::Extractor::new(orb3rs::types::Pyramid::new(1000, 1.2, 8, 20, 7));
    let mut o = std::io::BufWriter::new(std::fs::File::create(outp).unwrap());
    let (mut t_npu, mut t_cpu) = (0f64, 0f64);
    let mut outbuf = Vec::new();
    for i in 0..n {
        let raw = std::fs::read(format!("{md}/f{i:04}.raw")).unwrap();
        // 층 영상 = 모자이크 칸의 테두리 안쪽(C++ 피라미드와 같은 바이트)
        ext.levels = slots
            .iter()
            .zip(&sizes)
            .map(|(s, &(lw, lh))| {
                let mut v = Vec::with_capacity(lw * lh);
                for r in 0..lh {
                    let off = (s[1] + 19 + r) * mw + s[0] + 19;
                    v.extend_from_slice(&raw[off..off + lw]);
                }
                (v, lw, lh)
            })
            .collect();
        if let Some(net) = net.as_mut() {
            let t0 = std::time::Instant::now();
            let lv: Vec<(&[u8], usize, usize)> = ext.levels.iter().map(|(v, w, h)| (v.as_slice(), *w, *h)).collect();
            let m = mos.pack(&lv).to_vec();
            let inp: Vec<f32> = m.iter().map(|&b| b as f32).collect();
            net.run(nnrun::f32_bytes(&inp), &mut outbuf);
            let s = net.out(&outbuf, "S");
            let mut all = Vec::new();
            for (l, &(lw, lh)) in sizes.iter().enumerate() {
                let mut v = Vec::new();
                mos.level_scores(s, l, lw, lh, &mut v);
                if noise {
                    for (k, x) in v.iter_mut().enumerate() {
                        let h = (k as u32 ^ (i as u32).wrapping_mul(2654435761)).wrapping_mul(2246822519);
                        *x = x.saturating_sub(((h >> 15) & 1) as u8);
                    }
                }
                all.push(v);
            }
            *hook::S.write().unwrap() = all;
            t_npu += t0.elapsed().as_secs_f64();
        }
        let t0 = std::time::Instant::now();
        let (k, d) = ext.extract(&[], 0, 0);
        t_cpu += t0.elapsed().as_secs_f64();
        o.write_all(&(k.len() as i32).to_le_bytes()).unwrap();
        for p in &k {
            for v in [p.x, p.y, p.angle, p.response, p.octave as f32] {
                o.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        for x in &d {
            o.write_all(x).unwrap();
        }
    }
    eprintln!("장면 {n}: 추출(CPU) {:.3} ms/장, ONNX+S 꺼내기 {:.3} ms/장", t_cpu * 1e3 / n as f64, t_npu * 1e3 / n as f64);
}
