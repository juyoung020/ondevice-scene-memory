//! 행렬곱 커널 단독 속도: gemmbench m k n [반복]
//! B 를 미리 편 판(Packed) · 줄 주소 바로 읽기(Direct) · 판마다 펴기(Gather) 로 번갈아 돌려 최솟값 비교
//! (Direct · Gather 의 B 는 1×1 합성곱처럼 k 줄이 n 칸씩 이어진 채널 평면)
use common::nn::gemm::{self, Panels, RowMajor};
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (m, k, n, reps) = (a[0], a[1], a[2], a.get(3).copied().unwrap_or(20));
    let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.01).collect();
    let pb = gemm::pack_b(&w, k, n, n);
    let rows: Vec<usize> = (0..k).map(|j| w[j * n..].as_ptr() as usize).collect();
    let mut o = vec![0f32; m * n];
    let ap = gemm::pack_a(&x, m, k, k);
    let sink = RowMajor { out: o.as_mut_ptr(), ldc: n, col_bias: None, row_bias: None, act: if std::env::var_os("ACT").is_some() { common::nn::Act::Silu } else { common::nn::Act::None } };
    let names = ["packed", "direct", "gather"];
    let mut best = [f64::MAX; 3];
    for r in 0..reps + 3 {
        for (i, b) in best.iter_mut().enumerate() {
            let p = match i {
                0 => Panels::Packed(&pb),
                1 => Panels::Direct { rows: &rows, avail: n },
                _ => Panels::Gather { rows: &rows, avail: n },
            };
            let t = std::time::Instant::now();
            gemm::gemm(&ap, m, k, n, p, &sink);
            if r >= 3 {
                *b = b.min(t.elapsed().as_secs_f64());
            }
        }
    }
    for (name, b) in names.iter().zip(best) {
        println!("{m}x{k}x{n} {name}: {:.3} ms, {:.0} GFLOPS ({} 일꾼, mr {})", b * 1e3, 2.0 * (m * k * n) as f64 / b / 1e9, common::nn::pool::threads(), gemm::mr_for(m));
    }
}
