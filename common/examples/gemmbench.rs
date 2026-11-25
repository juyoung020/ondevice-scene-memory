//! 행렬곱 커널 단독 속도: gemmbench m k n [반복]
use common::nn::gemm::{self, Panels, RowMajor};
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (m, k, n, reps) = (a[0], a[1], a[2], a.get(3).copied().unwrap_or(20));
    let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.01).collect();
    let pb = gemm::pack_b(&w, k, n, n);
    let mut o = vec![0f32; m * n];
    let ap = gemm::pack_a(&x, m, k, k);
    let sink = RowMajor { out: o.as_mut_ptr(), ldc: n, col_bias: None, row_bias: None, act: common::nn::Act::None };
    for _ in 0..3 {
        gemm::gemm(&ap, m, k, n, Panels::Packed(&pb), &sink);
    }
    let t = std::time::Instant::now();
    for _ in 0..reps {
        gemm::gemm(&ap, m, k, n, Panels::Packed(&pb), &sink);
    }
    let dt = t.elapsed().as_secs_f64() / reps as f64;
    println!("{m}x{k}x{n}: {:.3} ms, {:.0} GFLOPS ({} 일꾼, mr {})", dt * 1e3, 2.0 * (m * k * n) as f64 / dt / 1e9, common::nn::pool::threads(), gemm::mr_for(m));
}
