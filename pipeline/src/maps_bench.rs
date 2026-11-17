//! 물체 기억 속도 재기(합성 장면): 물체 N 개가 있을 때 관측 1 회(update) · 병합 1 회(merge_pass) 시간.
//!   cargo test --release -p ondevice --bin ondevice bench_memory -- --ignored --nocapture
//! 공개 함수(Memory::new · update · merge_pass_with)만 써서 예전 판 maps.rs 에 붙여도 그대로 돈다(전후 비교용)
use super::{Crop, Memory};
use crate::geom::V3;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 상자 겉면의 점 n 개(실제 검출 하나가 넣는 점 수 상한 2,500 에 맞춤)
fn box_pts(r: &mut Rng, c: V3, s: V3, n: usize, col: [u8; 3]) -> Vec<(V3, [u8; 3])> {
    (0..n)
        .map(|_| {
            let ax = (r.f() * 3.0) as usize % 3;
            let mut p = [0.0; 3];
            for k in 0..3 {
                p[k] = c[k] + s[k] * if k == ax { if r.f() < 0.5 { -0.5 } else { 0.5 } } else { r.f() - 0.5 };
            }
            (p, col)
        })
        .collect()
}

fn crop() -> Crop {
    Crop { w: 1, h: 1, rgb: vec![0; 3].into(), mask: vec![255].into(), depth: vec![0].into() }
}

#[test]
#[ignore]
fn bench_memory() {
    for n in [50usize, 200, 1000] {
        let mut r = Rng(n as u64);
        let side = (n as f64).sqrt().ceil() as usize;
        let objs: Vec<(V3, V3, [u8; 3])> = (0..n)
            .map(|i| {
                let c = [(i % side) as f64 * 0.7, (i / side) as f64 * 0.7, 0.6];
                let s = [0.15 + r.f() * 0.3, 0.15 + r.f() * 0.3, 0.15 + r.f() * 0.3];
                (c, s, [(r.f() * 255.0) as u8, (r.f() * 255.0) as u8, (r.f() * 255.0) as u8])
            })
            .collect();
        // 사진 벡터 768 차원(물체마다 다른 방향)
        let emb = |id: u32| -> Option<Vec<f32>> {
            let mut v = vec![0.01f32; 768];
            v[id as usize % 768] = 1.0;
            let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            Some(v.into_iter().map(|x| x / l).collect())
        };
        let mut m = Memory::new(0.012, 0.04);
        let mut t = 0.0;
        let mut observe = |m: &mut Memory, r: &mut Rng, i: usize, jit: f64| {
            let (c, s, col) = objs[i];
            let c = [c[0] + (r.f() - 0.5) * jit, c[1] + (r.f() - 0.5) * jit, c[2] + (r.f() - 0.5) * jit];
            let pts = box_pts(r, c, s, 2500, col);
            t += 0.07;
            m.update("object", 0.9, &pts, 1.0, t, 1, crop, None)
        };
        for k in 0..3 * n {
            observe(&mut m, &mut r, k % n, 0.02);
        }
        m.merge_pass_with(&emb);
        let (mut up, mut nu, mut mg, mut nm) = (0.0f64, 0, 0.0f64, 0);
        let (mut up_max, mut mg_max) = (0.0f64, 0.0f64);
        for round in 0..20 {
            // 장면 15 장 · 장면마다 검출 6 개쯤: 병합 한 번 사이 관측 90 번(가끔 크게 어긋난 관측 = 같은 물체를 따로 잡음)
            for _ in 0..90 {
                let i = (r.f() * n as f64) as usize % n;
                let jit = if r.f() < 0.1 { 0.15 } else { 0.02 };
                let t0 = std::time::Instant::now();
                std::hint::black_box(observe(&mut m, &mut r, i, jit));
                let dt = t0.elapsed().as_secs_f64();
                up += dt;
                up_max = up_max.max(dt);
                nu += 1;
            }
            let t0 = std::time::Instant::now();
            std::hint::black_box(m.merge_pass_with(&emb));
            let dt = t0.elapsed().as_secs_f64();
            if round > 0 {
                mg += dt;
                mg_max = mg_max.max(dt);
                nm += 1;
            }
        }
        // 바뀐 것 없는 병합(시험용 입구가 사진 벡터를 물체마다 받아 견주는 몫이 대부분 — 실제 merge_pass 는 바뀐 벡터만 받음)
        let t0 = std::time::Instant::now();
        std::hint::black_box(m.merge_pass_with(&emb));
        let idle = t0.elapsed().as_secs_f64();
        println!(
            "bench N={n} objs={} update_us={:.0} update_max_us={:.0} merge_ms={:.3} merge_max_ms={:.3} merge_idle_ms={:.3} merged={}",
            m.objs.len(), up / nu as f64 * 1e6, up_max * 1e6, mg / nm as f64 * 1e3, mg_max * 1e3, idle * 1e3, m.merged
        );
    }
}
