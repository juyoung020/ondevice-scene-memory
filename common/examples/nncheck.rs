//! Rust 실행기 대 ONNX Runtime 출력 대조: nncheck 모델.onnx 입력.bin 기준폴더 [반복]
//! 입력.bin: 모델 첫 입력 모양 그대로(u8 입력이면 바이트, 아니면 f32). 기준폴더/<출력이름>.bin: ort 출력(f32)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t0 = std::time::Instant::now();
    let m = common::nn::Model::load(&a[1]).unwrap();
    eprintln!("load {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    let (_, shape, u8_in) = m.inputs().next().unwrap();
    let shape: Vec<usize> = match std::env::var("NN_SHAPE") {
        Ok(s) => s.split(',').map(|d| d.parse().unwrap()).collect(),
        Err(_) => shape.iter().map(|&d| d.max(1) as usize).collect(),
    };
    let raw = std::fs::read(&a[2]).unwrap();
    let v: Vec<f32> = if u8_in { raw.iter().map(|&b| b as f32).collect() } else { raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let reps: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    let mut times = vec![];
    let mut outs = vec![];
    for _ in 0..reps {
        let t = std::time::Instant::now();
        outs = m.run(vec![common::nn::Tensor::f(shape.clone(), v.clone())]).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    println!("rust median {:.2} ms (min {:.2}, {} runs, {} threads)", times[times.len() / 2], times[0], reps, common::nn::pool::threads());
    for (name, o) in m.output_names().zip(outs) {
        let p = format!("{}/{}.bin", a[3], name.replace('/', "_"));
        let r: Vec<f32> = match std::fs::read(&p) {
            Ok(b) => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Err(_) => {
                println!("{name}: 기준 없음 {:?}", o.shape);
                continue;
            }
        };
        let o = o.into_f();
        if o.len() != r.len() {
            println!("{name}: 길이 {} ≠ 기준 {}", o.len(), r.len());
            continue;
        }
        let (mut d, mut na, mut nb, mut mx, mut ma) = (0f64, 0f64, 0f64, 0f32, 0f32);
        for (x, y) in o.iter().zip(&r) {
            d += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
            mx = mx.max((x - y).abs());
            ma = ma.max(y.abs());
        }
        println!("{name}: cos {:.7}, max|diff| {:.3e} (기준 최대 |값| {:.3e}), n {}", d / (na.sqrt() * nb.sqrt()).max(1e-30), mx, ma, o.len());
        std::fs::write(format!("{}/{}.rust.bin", a[3], name.replace('/', "_")), o.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
}
