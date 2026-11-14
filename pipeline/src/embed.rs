//! 물체 사진 → 검색용 벡터. SigLIP2 영상 인코더(PC: vision_model.onnx, 보드 NPU: .dxnn)가 사진 한 장을 768 차원 벡터로 바꾸고,
//! 길이를 1 로 맞춰 `objects/O<id>_emb.bin`(f32 리틀 엔디언 768 개)에 남긴다. 뷰어의 검색이 글 벡터와 내적해 닮은 물체를 찾는다.
//! 지도 쌓기와 따로 도는 스레드 하나가 맡는다: 사진이 바뀐 물체만 받아, 밀리면 물체마다 가장 새 사진만 계산한다.
use std::path::{Path, PathBuf};

/// 인코더 입력 한 변(화소)
pub const SIZE: usize = 256;
/// 벡터 차원
pub const DIM: usize = 768;

/// w×h RGB 를 256×256 으로 늘려(작아질 땐 넓게 평균) NCHW, 값 −1..1 (SigLIP 전처리: (x/255 − 0.5)/0.5)
#[cfg_attr(not(test), allow(dead_code))]
pub fn preprocess(rgb: &[u8], w: u32, h: u32, out: &mut Vec<f32>) {
    preprocess_into(rgb, w, h, false, out)
}

/// 사진(RGB w×h) → 256×256 쌍선형으로 줄여 (값 / 127.5 − 1) 을 out 에 바로 쓴다(중간 사진 없음).
/// patches = false: [3, 256, 256] (채널, 행, 열). true: 32×32 조각 64 개 [64, 3·32·32] (조각 = 행 우선, 안은 채널·행·열) —
/// 조각 나누기 합성곱을 행렬곱으로 바꿔 NPU 에 다 올린 모델(입력 `patches`)이 받는 배치
pub fn preprocess_into(rgb: &[u8], w: u32, h: u32, patches: bool, out: &mut Vec<f32>) {
    let (w, h) = (w as usize, h as usize);
    let n = SIZE * SIZE;
    out.resize(3 * n, 0.0);
    let axis = |dst: usize, src: usize| -> (usize, usize, f32) {
        let f = ((dst as f32 + 0.5) * src as f32 / SIZE as f32 - 0.5).max(0.0);
        let a = (f as usize).min(src - 1);
        (a, (a + 1).min(src - 1), f - a as f32)
    };
    // 가로 위치·가중(열마다 한 번). 한 행의 가로 섞기를 먼저 채널별 배열로 만든 뒤 세로 섞기를 32 칸씩 이어서 쓴다
    // (곱·더하기 순서는 화소마다 하던 식과 같아 값이 비트까지 같다. 세로 섞기 고리는 컴파일러가 NEON 으로 묶음)
    let (mut ix0, mut ix1, mut wx0, mut wx1) = ([0usize; SIZE], [0usize; SIZE], [0f32; SIZE], [0f32; SIZE]);
    for x in 0..SIZE {
        let (a, b, f) = axis(x, w);
        (ix0[x], ix1[x], wx0[x], wx1[x]) = (a * 3, b * 3, 1.0 - f, f);
    }
    const K: f32 = 1.0 / 127.5;
    let (mut top, mut bot) = ([0f32; SIZE], [0f32; SIZE]);
    for y in 0..SIZE {
        let (y0, y1, fy) = axis(y, h);
        let (r0, r1) = (&rgb[y0 * w * 3..(y0 + 1) * w * 3], &rgb[y1 * w * 3..(y1 + 1) * w * 3]);
        let (wy0, wy1) = (1.0 - fy, fy);
        for c in 0..3 {
            for x in 0..SIZE {
                top[x] = r0[ix0[x] + c] as f32 * wx0[x] + r0[ix1[x] + c] as f32 * wx1[x];
                bot[x] = r1[ix0[x] + c] as f32 * wx0[x] + r1[ix1[x] + c] as f32 * wx1[x];
            }
            for px in 0..SIZE / 32 {
                let o = if patches { ((y / 32) * 8 + px) * 3072 + c * 1024 + (y % 32) * 32 } else { c * n + y * SIZE + px * 32 };
                let (d, t, b) = (&mut out[o..o + 32], &top[px * 32..px * 32 + 32], &bot[px * 32..px * 32 + 32]);
                for k in 0..32 {
                    d[k] = (t[k] * wy0 + b[k] * wy1) * K - 1.0;
                }
            }
        }
    }
}

/// 길이 1 로 맞춤(0 벡터는 그대로)
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// 물체 번호 → 지금까지 계산한 사진 벡터의 합(이름 고르기용)
static SUMS: std::sync::Mutex<Option<std::collections::HashMap<u32, Vec<f32>>>> = std::sync::Mutex::new(None);

/// 물체 번호 → 가장 새 벡터. 계산 스레드가 채우고 물체 병합(maps::merge_pass)이 읽는다
static LATEST: std::sync::Mutex<Option<std::collections::HashMap<u32, Vec<f32>>>> = std::sync::Mutex::new(None);

/// 그 물체의 가장 새 사진 벡터(길이 1). 아직 계산 전이면 None
pub fn get(id: u32) -> Option<Vec<f32>> {
    LATEST.lock().unwrap().as_ref()?.get(&id).cloned()
}

/// SigLIP2 base patch32-256: 로짓 = SCALE · 코사인 (체크포인트의 logit_scale = e^4.717)
const SCALE: f32 = 111.83;
/// 사진으로 붙인 이름을 쓰는 가장 낮은 몫. 사무실 녹화의 물체 33 개에서 0.3 이상은 사진과 맞았고(오리 0.46, 지우개 0.50, 스프레이 통 0.35)
/// 그 아래는 틀린 이름이 섞였다(컵 → bedpan 0.27)
pub const NAME_MIN: f32 = 0.3;
/// 이름 없는 물체(object)에 사진 이름을 쓰는 가장 낮은 몫. ObjectSAM v1.0 으로 돌린 사무실 녹화에서 0.15 아래는 대부분 틀렸다
/// (breechcloth 0.06, ping-pong ball 0.08, vending machine 0.10), 0.2 대는 맞았다(chair 0.27, bottle 0.23, file cabinet 0.20)
pub const NAME_MIN_UNNAMED: f32 = 0.15;
/// 1 등 이름과 2 등 이름의 코사인 차가 이보다 작으면 이름을 붙이지 않는다. fr1_desk 물체 23 개(사람이 이름을 단 것)에서
/// 몫 0.15 만 쓸 때 맞은 이름 비율 0.52, 차 0.003 을 더하면 0.67(이름 붙은 물체 15 개)
pub const NAME_MARGIN: f32 = 0.003;

/// 물체가 아닌 면의 이름(이름 목록에 함께 넣어 둠)
pub const STRUCTURAL: [&str; 4] = ["floor", "wall", "ceiling", "baseboard"];

/// 이름 목록과 그 글 벡터(`--vocab`): PC 에서 `sgview --encode-vocab` 로 한 번 만들어 둔 파일이라 글 인코더 없이 쓴다
static VOCAB: std::sync::OnceLock<(Vec<String>, Vec<f32>)> = std::sync::OnceLock::new();
/// 물체 번호 → 사진이 가장 닮은 이름과 그 몫
static NAMES: std::sync::Mutex<Option<std::collections::HashMap<u32, (String, f32)>>> = std::sync::Mutex::new(None);

/// 글 벡터 파일(첫 줄 `SGVOC 1 <n> <차원>`, 이름 n 줄, 이어서 f32 n·차원 개)을 읽어 둔다. 이름 목록을 돌려준다
pub fn load_vocab(path: &str) -> Vec<String> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("--vocab {path}: {e}"));
    let mut pos = 0;
    let mut line = || {
        let end = pos + b[pos..].iter().position(|c| *c == b'\n').unwrap_or_else(|| panic!("--vocab {path}: 글 벡터 파일이 아님"));
        let s = std::str::from_utf8(&b[pos..end]).unwrap_or_else(|_| panic!("--vocab {path}: 글 벡터 파일이 아님")).to_string();
        pos = end + 1;
        s
    };
    let head = line();
    let h: Vec<&str> = head.split(' ').collect();
    assert!(h.len() == 4 && h[0] == "SGVOC" && h[1] == "1" && h[3].parse() == Ok(DIM), "--vocab {path}: 글 벡터 파일이 아님");
    let n: usize = h[2].parse().unwrap_or_else(|_| panic!("--vocab {path}: 글 벡터 파일이 아님"));
    let names: Vec<String> = (0..n).map(|_| line()).collect();
    assert_eq!(b.len(), pos + n * DIM * 4, "--vocab {path}: 벡터 수가 이름 수와 다름");
    let v: Vec<f32> = b[pos..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let _ = VOCAB.set((names.clone(), v));
    names
}

/// 사진 벡터(길이 1)가 가장 닮은 이름과 그 몫: 이름 전체에 대한 softmax(SCALE · 코사인). 1 등과 2 등의 코사인 차가 NAME_MARGIN 보다
/// 작으면(두 이름 사이에서 갈팡질팡) 몫을 0 으로 돌려 이름을 붙이지 않게 한다
fn look_name(names: &[String], vecs: &[f32], e: &[f32]) -> Option<(String, f32)> {
    let l: Vec<f32> = vecs.chunks(DIM).map(|t| SCALE * t.iter().zip(e).map(|(a, b)| a * b).sum::<f32>()).collect();
    let (i, m) = l.iter().cloned().enumerate().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let second = l.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| *x).fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = l.iter().map(|x| (x - m).exp()).sum();
    let share = if (m - second) / SCALE >= NAME_MARGIN { 1.0 / z } else { 0.0 };
    Some((names[i].clone(), share))
}

/// 그 물체의 사진으로 붙인 이름. 분할 모델이 이름을 준 물체(`sure` 참)는 몫 NAME_MIN 이상일 때만 바꾸고,
/// 이름 없는 물체(object)는 몫 NAME_MIN_UNNAMED 이상이면 가장 닮은 이름을 쓴다. `--vocab` 이 없거나 벡터 계산 전이면 None
pub fn name(id: u32, sure: bool) -> Option<(String, f32)> {
    NAMES.lock().unwrap().as_ref()?.get(&id).filter(|(_, p)| *p >= if sure { NAME_MIN } else { NAME_MIN_UNNAMED }).cloned()
}
#[cfg_attr(not(any(feature = "onnx", feature = "dxrt")), allow(dead_code))]
fn put(id: u32, v: Vec<f32>) {
    // 이름은 그 물체의 사진 벡터를 모두 더한 평균 방향으로(사진 한 장의 흔들림·어두움에 덜 휘둘림)
    let mean = {
        let mut sums = SUMS.lock().unwrap();
        let s = sums.get_or_insert_with(Default::default).entry(id).or_insert_with(|| vec![0.0; v.len()]);
        s.iter_mut().zip(&v).for_each(|(a, b)| *a += b);
        let mut m = s.clone();
        normalize(&mut m);
        m
    };
    if let Some((names, vecs)) = VOCAB.get() {
        if let Some(n) = look_name(names, vecs, &mean) {
            NAMES.lock().unwrap().get_or_insert_with(Default::default).insert(id, n);
        }
    }
    LATEST.lock().unwrap().get_or_insert_with(Default::default).insert(id, v);
}

/// 벡터 파일 경로(기억 폴더 기준 상대 경로)
pub fn rel_path(id: u32) -> String {
    format!("objects/O{id}_emb.bin")
}

#[cfg(any(feature = "onnx", feature = "dxrt"))]
mod worker {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{mpsc, Mutex};

    struct Job {
        id: u32,
        path: PathBuf,
        w: u32,
        h: u32,
        rgb: std::sync::Arc<Vec<u8>>, // 물체 사진(기억의 사진과 나눠 씀)
    }
    type Handle = (mpsc::Sender<Job>, std::thread::JoinHandle<()>);
    static W: Mutex<Option<Handle>> = Mutex::new(None);
    /// 맡겼지만 아직 끝나지 않은 사진 수
    static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// 인코더 실행기: 입력 NCHW f32(−1..1) 256×256 → 768 차원
    enum Net {
        #[cfg(feature = "onnx")]
        Onnx(ort::session::Session),
        /// DX-M1: 입력·출력 모두 f32 (패치 나누기와 끝 MLP 는 런타임이 CPU 로 돌림)
        #[cfg(feature = "dxrt")]
        Dx(crate::seg::dx::Engine),
    }
    impl Net {
        fn open(model: &str) -> Net {
            #[cfg(feature = "dxrt")]
            if model.ends_with(".dxnn") {
                let e = crate::seg::dx::Engine::with_cpu_tasks(model);
                assert_eq!(e.out.len(), DIM * 4, "embed model output");
                common::info!("embed: NPU 모델 {model}, 출력 {}", e.tensors[0].0);
                return Net::Dx(e);
            }
            #[cfg(feature = "onnx")]
            {
                // 지도 쌓기·분할과 코어를 다투지 않게 스레드 둘만
                let sess = ort::session::Session::builder(common::ort::env()).unwrap().with_intra_threads(2).unwrap()
                    .commit_from_file(model).unwrap_or_else(|e| panic!("embed model {model}: {e}"));
                Net::Onnx(sess)
            }
            #[cfg(not(feature = "onnx"))]
            panic!("--embed {model}: 이 빌드는 NPU 모델(.dxnn)을 받는다");
        }
        /// 입력이 조각 배치(`patches`)인 NPU 모델인가
        fn patches(&self) -> bool {
            match self {
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => e.input == "patches",
                #[allow(unreachable_patterns)]
                _ => false,
            }
        }
        fn submit(&mut self, buf: &[f32]) -> Ticket {
            match self {
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => {
                    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                    Ticket::Job(e.submit(bytes))
                }
                #[allow(unreachable_patterns)]
                _ => Ticket::Done(self.run(buf)),
            }
        }
        fn wait(&mut self, t: Ticket) -> Vec<f32> {
            match (self, t) {
                #[cfg(feature = "dxrt")]
                (Net::Dx(e), Ticket::Job(id)) => {
                    let mut v = Vec::new();
                    e.waiter().wait_into(id, &mut v);
                    v.truncate(DIM);
                    v
                }
                (_, Ticket::Done(v)) => v,
                #[allow(unreachable_patterns)]
                _ => unreachable!(),
            }
        }
        fn run(&mut self, buf: &[f32]) -> Vec<f32> {
            match self {
                #[cfg(feature = "onnx")]
                Net::Onnx(sess) => {
                    let inp = ort::value::TensorRef::from_array_view(([1usize, 3, SIZE, SIZE], buf)).unwrap();
                    let out = sess.run(ort::inputs![inp]).expect("embed run");
                    let (_, v) = out["pooler_output"].try_extract_tensor::<f32>().unwrap();
                    assert_eq!(v.len(), DIM, "embed model output");
                    v.to_vec()
                }
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => {
                    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                    e.run(bytes);
                    e.f32_at(0).to_vec()
                }
            }
        }
    }

    #[derive(Default)]
    struct Stat {
        n: u32,
        pre: f64,
        net: f64,
        write: f64,
    }
    /// 올린 추론 하나: NPU 는 작업 번호, CPU(ONNX)는 이미 끝난 벡터
    enum Ticket {
        Done(Vec<f32>),
        #[cfg(feature = "dxrt")]
        Job(i32),
    }
    fn finish_job(net: &mut Net, (j, tk, t0, pre): (Job, Ticket, std::time::Instant, f64), st: &mut Stat) {
        let mut v = net.wait(tk);
        let t1 = std::time::Instant::now();
        normalize(&mut v);
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        // 임시 파일에 쓰고 이름을 바꿔, 읽는 쪽이 반쯤 쓴 파일을 보지 않게
        let tmp = j.path.with_extension("bin.tmp");
        if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &j.path).is_ok() {
            crate::out::publish(&j.path, bytes); // --serve: 검색이 파일을 다시 읽지 않게
        }
        put(j.id, v);
        st.n += 1;
        st.pre += pre;
        st.net += (t1 - t0).as_secs_f64() * 1e3;
        st.write += t1.elapsed().as_secs_f64() * 1e3;
    }

    pub fn start(model: &str) {
        let mut net = Net::open(model);
        let patches = net.patches();
        let (tx, rx) = mpsc::channel::<Job>();
        let th = std::thread::Builder::new().name("embed".into()).spawn(move || {
            let mut bufs = [Vec::new(), Vec::new()];
            let mut st = Stat::default();
            while let Ok(first) = rx.recv() {
                // 밀린 일은 물체(파일)마다 가장 새 사진만
                let mut jobs: HashMap<PathBuf, Job> = HashMap::new();
                jobs.insert(first.path.clone(), first);
                let mut taken = 1;
                while let Ok(j) = rx.try_recv() {
                    jobs.insert(j.path.clone(), j);
                    taken += 1;
                }
                // NPU 에 한 장 올려 둔 동안 다음 사진을 전처리(입력 버퍼 둘을 번갈아)
                let mut prev: Option<(Job, Ticket, std::time::Instant, f64)> = None;
                for (i, j) in jobs.into_values().enumerate() {
                    let t = std::time::Instant::now();
                    preprocess_into(&j.rgb, j.w, j.h, patches, &mut bufs[i % 2]);
                    let pre = t.elapsed().as_secs_f64() * 1e3;
                    if let Some(p) = prev.take() {
                        finish_job(&mut net, p, &mut st);
                    }
                    let tk = net.submit(&bufs[i % 2]);
                    prev = Some((j, tk, std::time::Instant::now(), pre));
                }
                if let Some(p) = prev.take() {
                    finish_job(&mut net, p, &mut st);
                }
                PENDING.fetch_sub(taken, std::sync::atomic::Ordering::SeqCst);
            }
            let k = st.n.max(1) as f64;
            common::info!("embed: {} vectors, 전처리 {:.1} ms · 모델(올림 → 받음) {:.1} ms · 쓰기 {:.1} ms 씩", st.n, st.pre / k, st.net / k, st.write / k);
        }).unwrap();
        *W.lock().unwrap() = Some((tx, th));
    }

    pub fn on() -> bool {
        W.lock().unwrap().is_some()
    }

    /// 폴더의 사진(png·jpg, 이름 순)을 벡터로 바꿔 `vec_file` 에 쓴다. 파일이 이미 있으면 그 벡터와 코사인을 견준다
    /// (PC 의 ONNX 벡터를 기준으로 보드 NPU 벡터를 확인): `memmap --embed-check <모델> <사진 폴더> <벡터 파일>`
    pub fn check(model: &str, dir: &str, vec_file: &str) {
        let mut net = Net::open(model);
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir}: {e}")).filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg"))).collect();
        files.sort();
        let (mut buf, mut all, mut ms) = (Vec::new(), Vec::<f32>::new(), 0.0f64);
        for (i, f) in files.iter().enumerate() {
            let img = image::open(f).unwrap_or_else(|e| panic!("{}: {e}", f.display())).to_rgb8();
            preprocess_into(img.as_raw(), img.width(), img.height(), net.patches(), &mut buf);
            let t = std::time::Instant::now();
            let mut v = net.run(&buf);
            if i > 0 {
                ms += t.elapsed().as_secs_f64() * 1e3; // 첫 장은 모델 올리는 시간이 섞여 뺌
            }
            normalize(&mut v);
            all.extend(v);
        }
        println!("embed check: 사진 {} 장, 추론 {:.1} ms/장", files.len(), ms / (files.len().max(2) - 1) as f64);
        match std::fs::read(vec_file) {
            Ok(b) => {
                let r: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                assert_eq!(r.len(), all.len(), "{vec_file}: 사진 수가 다름");
                let cos: Vec<f32> = all.chunks(DIM).zip(r.chunks(DIM)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum()).collect();
                let mean = cos.iter().sum::<f32>() / cos.len().max(1) as f32;
                println!("기준 벡터와의 코사인: 평균 {mean:.4}, 최소 {:.4}", cos.iter().cloned().fold(f32::INFINITY, f32::min));
            }
            Err(_) => {
                let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(vec_file, bytes).unwrap_or_else(|e| panic!("{vec_file}: {e}"));
                println!("기준 벡터를 {vec_file} 에 씀");
            }
        }
    }

    pub fn submit(dir: &Path, id: u32, w: u32, h: u32, rgb: std::sync::Arc<Vec<u8>>) {
        if let Some((tx, _)) = &*W.lock().unwrap() {
            PENDING.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = tx.send(Job { id, path: dir.join(rel_path(id)), w, h, rgb });
        }
    }

    /// 맡긴 사진이 모두 벡터가 될 때까지 기다린다(끝낼 때 마지막 병합 앞에서)
    pub fn drain() {
        while PENDING.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub fn finish() {
        if let Some((tx, th)) = W.lock().unwrap().take() {
            drop(tx);
            let _ = th.join();
        }
    }
}

/// 인코더를 올리고 계산 스레드를 띄운다(`--embed <vision_model.onnx | .dxnn>`)
#[cfg(any(feature = "onnx", feature = "dxrt"))]
pub use worker::{check, drain, finish, on, start, submit};

#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn start(_model: &str) {
    panic!("--embed: 이 빌드에는 모델 실행기가 없다(onnx 또는 dxrt 기능으로 빌드)");
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn check(model: &str, _dir: &str, _vec_file: &str) {
    start(model);
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn on() -> bool {
    false
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn submit(_dir: &Path, _id: u32, _w: u32, _h: u32, _rgb: std::sync::Arc<Vec<u8>>) {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn finish() {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn drain() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 화소마다 하던 원래 식(기준)
    fn preprocess_ref(rgb: &[u8], w: usize, h: usize, patches: bool) -> Vec<f32> {
        let n = SIZE * SIZE;
        let mut out = vec![0.0; 3 * n];
        let axis = |dst: usize, src: usize| -> (usize, usize, f32) {
            let f = ((dst as f32 + 0.5) * src as f32 / SIZE as f32 - 0.5).max(0.0);
            let a = (f as usize).min(src - 1);
            (a, (a + 1).min(src - 1), f - a as f32)
        };
        for y in 0..SIZE {
            let (y0, y1, fy) = axis(y, h);
            for x in 0..SIZE {
                let (x0, x1, fx) = axis(x, w);
                for c in 0..3 {
                    let p = |yy: usize, xx: usize| rgb[(yy * w + xx) * 3 + c] as f32;
                    let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                    let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                    let i = if patches { ((y / 32) * 8 + x / 32) * 3072 + c * 1024 + (y % 32) * 32 + x % 32 } else { c * n + y * SIZE + x };
                    out[i] = (top * (1.0 - fy) + bot * fy) * (1.0 / 127.5) - 1.0;
                }
            }
        }
        out
    }

    #[test]
    fn preprocess_same_as_reference() {
        let mut s = 7u64;
        for &(w, h) in &[(37usize, 211usize), (640, 480), (256, 256), (3, 900)] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 56) as u8 }).collect();
            for patches in [false, true] {
                let mut out = vec![];
                preprocess_into(&rgb, w as u32, h as u32, patches, &mut out);
                let want = preprocess_ref(&rgb, w, h, patches);
                assert!(out.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{w}x{h} {patches}");
            }
        }
    }

    #[test]
    fn preprocess_range_and_layout() {
        // 왼쪽 절반 빨강, 오른쪽 절반 파랑인 4×2 사진
        let mut rgb = vec![];
        for _ in 0..2 {
            for x in 0..4 {
                rgb.extend_from_slice(if x < 2 { &[255, 0, 0] } else { &[0, 0, 255] });
            }
        }
        let mut out = vec![];
        preprocess(&rgb, 4, 2, &mut out);
        let n = SIZE * SIZE;
        assert_eq!(out.len(), 3 * n);
        let at = |c: usize, x: usize, y: usize| out[c * n + y * SIZE + x];
        assert!((at(0, 10, 100) - 1.0).abs() < 1e-4 && (at(2, 10, 100) + 1.0).abs() < 1e-4);
        assert!((at(0, 245, 100) + 1.0).abs() < 1e-4 && (at(2, 245, 100) - 1.0).abs() < 1e-4);
        assert!(out.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn look_name_picks_the_closest_word() {
        // 이름 셋이 축 0·1·2 를 가리키고, 사진은 축 1 쪽으로 조금 기움(글·사진 코사인은 실제 모델에서도 작다)
        let names: Vec<String> = ["chair", "monitor", "cup"].iter().map(|s| s.to_string()).collect();
        let mut vecs = vec![0.0; 3 * DIM];
        for i in 0..3 {
            vecs[i * DIM + i] = 1.0;
        }
        let mut e = vec![0.0; DIM];
        (e[1], e[9]) = (0.1, 0.995);
        let (n, p) = look_name(&names, &vecs, &e).unwrap();
        assert_eq!(n, "monitor");
        assert!(p > 0.99, "{p}");
        // 어느 이름도 닮지 않은 사진: 1 등과 2 등의 차가 NAME_MARGIN 아래라 몫 0(이름 안 붙음)
        e[1] = 0.0;
        assert_eq!(look_name(&names, &vecs, &e).unwrap().1, 0.0);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0; 4];
        normalize(&mut z);
        assert!(z.iter().all(|x| *x == 0.0));
    }
}
