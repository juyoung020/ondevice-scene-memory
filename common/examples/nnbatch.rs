//! 여러 입력을 Rust 실행기로: nnbatch 모델.onnx 입력.f32 모양(쉼표) 출력폴더 → <폴더>/rust_<출력이름>.f32 (입력 차례대로 이어 붙임)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = common::nn::Model::load(&a[1]).unwrap();
    let shape: Vec<usize> = a[3].split(',').map(|d| d.parse().unwrap()).collect();
    let per: usize = shape.iter().product();
    let x: Vec<f32> = std::fs::read(&a[2]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let names: Vec<String> = m.output_names().map(|s| s.replace('/', "_")).collect();
    let mut outs: Vec<Vec<u8>> = vec![vec![]; names.len()];
    let mut ts = vec![];
    for c in x.chunks_exact(per) {
        let t = std::time::Instant::now();
        let o = m.run(vec![common::nn::Tensor::f(shape.clone(), c.to_vec())]).unwrap();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
        for (b, t) in outs.iter_mut().zip(o) {
            b.extend(t.into_f().iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    for (n, b) in names.iter().zip(outs) {
        std::fs::write(format!("{}/rust_{n}.f32", a[4]), b).unwrap();
    }
    ts.sort_by(f64::total_cmp);
    println!("{} 개, 중앙값 {:.2} ms ({} 일꾼)", ts.len(), ts[ts.len() / 2], common::nn::pool::threads());
}
