//! YOLO26-seg, end-to-end 출력: [1,300,38] = x1 y1 x2 y2 점수 클래스 + 계수 32, 원형 [1,32,160,160].
//! 실행기 두 개가 같은 전처리(레터박스 RGB, 회색 114)·후처리(상자 안 화소만 계수·원형 내적)를 쓴다.
//!   - PC: ONNX Runtime (feature "onnx", models/yolo26s-seg.onnx, 입력 f32 NCHW /255)
//!   - Pi5 + DX-M1: DXRT C API (feature "dxrt", yolo26-s-seg_640x640.dxnn, 입력 u8 NHWC)

pub const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

pub struct Det {
    pub cls: usize,
    pub score: f32,
    pub box_: [i32; 4], // 영상 화소 x1 y1 x2 y2 (잘라 맞춤)
    pub mask: Vec<bool>, // 상자 크기 (x2-x1)*(y2-y1), 행 우선
}

/// 레터박스 결과: u8 NHWC s×s×3, 배율 r, 여백 (px, py)
pub struct Letterbox {
    pub buf: Vec<u8>,
    pub r: f32,
    pub px: usize,
    pub py: usize,
}

pub fn letterbox(rgb: &[u8], w: usize, h: usize, s: usize) -> Letterbox {
    let r = (s as f32 / w as f32).min(s as f32 / h as f32);
    let (nw, nh) = ((w as f32 * r).round() as usize, (h as f32 * r).round() as usize);
    let (px, py) = ((s - nw) / 2, (s - nh) / 2);
    let mut buf = vec![114u8; s * s * 3];
    // 가로 원본 위치는 줄마다 같으니 한 번만 계산
    let xs: Vec<usize> = (0..nw).map(|x| (((x as f32 + 0.5) / r) as usize).min(w - 1) * 3).collect();
    for y in 0..nh {
        let sy = (((y as f32 + 0.5) / r) as usize).min(h - 1);
        let src = &rgb[sy * w * 3..(sy + 1) * w * 3];
        let dst = &mut buf[((py + y) * s + px) * 3..((py + y) * s + px + nw) * 3];
        for (d, &o) in dst.chunks_exact_mut(3).zip(&xs) {
            d.copy_from_slice(&src[o..o + 3]);
        }
    }
    Letterbox { buf, r, px, py }
}

/// 상자·마스크 해독(입력 좌표 → 영상 좌표)
#[allow(clippy::too_many_arguments)]
pub fn decode(det: &[f32], proto: &[f32], pc: usize, ph: usize, pw: usize, s: usize, lb: &Letterbox, w: usize, h: usize, conf: f32) -> Vec<Det> {
    let (r, px, py) = (lb.r, lb.px as f32, lb.py as f32);
    let scale = ph as f32 / s as f32;
    let mut dets = vec![];
    for d in det.chunks(6 + pc) {
        if d[4] < conf {
            continue;
        }
        let x1 = (((d[0] - px) / r).floor() as i32).clamp(0, w as i32 - 1);
        let y1 = (((d[1] - py) / r).floor() as i32).clamp(0, h as i32 - 1);
        let x2 = (((d[2] - px) / r).ceil() as i32).clamp(x1 + 1, w as i32);
        let y2 = (((d[3] - py) / r).ceil() as i32).clamp(y1 + 1, h as i32);
        let coef = &d[6..6 + pc];
        let (bw, bh) = ((x2 - x1) as usize, (y2 - y1) as usize);
        // 원형 격자 위치(행·열마다 한 번): (아래 칸, 위 칸, 가중)
        let fy = |yy: usize| -> (usize, usize, f32) {
            let f = (((y1 as usize + yy) as f32 + 0.5) * r + py) * scale - 0.5;
            let y0 = (f.floor() as i32).clamp(0, ph as i32 - 1) as usize;
            (y0, (y0 + 1).min(ph - 1), (f - y0 as f32).clamp(0.0, 1.0))
        };
        let fx = |xx: usize| -> (usize, usize, f32) {
            let f = (((x1 as usize + xx) as f32 + 0.5) * r + px) * scale - 0.5;
            let x0 = (f.floor() as i32).clamp(0, pw as i32 - 1) as usize;
            (x0, (x0 + 1).min(pw - 1), (f - x0 as f32).clamp(0.0, 1.0))
        };
        let ys: Vec<_> = (0..bh).map(fy).collect();
        let xs: Vec<_> = (0..bw).map(fx).collect();
        // 계수·원형 내적은 선형이라 원형 격자(160²)에서 먼저 하고 화소마다 쌍선형으로 읽는다(화소마다 32 번 하던 것과 같은 값)
        let (gy0, gy1) = (ys[0].0, ys[bh - 1].1);
        let (gx0, gx1) = (xs[0].0, xs[bw - 1].1);
        let gw = gx1 - gx0 + 1;
        let mut g = vec![0f32; (gy1 - gy0 + 1) * gw];
        for (k, c) in coef.iter().enumerate() {
            let b = k * ph * pw;
            for yy in gy0..=gy1 {
                let src = &proto[b + yy * pw + gx0..b + yy * pw + gx1 + 1];
                let dst = &mut g[(yy - gy0) * gw..(yy - gy0) * gw + gw];
                for (o, s) in dst.iter_mut().zip(src) {
                    *o += c * s;
                }
            }
        }
        let mut mask = vec![false; bw * bh];
        for (yy, &(y0, y1p, ay)) in ys.iter().enumerate() {
            let (r0, r1) = (&g[(y0 - gy0) * gw..(y0 - gy0) * gw + gw], &g[(y1p - gy0) * gw..(y1p - gy0) * gw + gw]);
            for (xx, &(x0, x1p, ax)) in xs.iter().enumerate() {
                let (a0, a1) = (x0 - gx0, x1p - gx0);
                let top = r0[a0] * (1.0 - ax) + r0[a1] * ax;
                let bot = r1[a0] * (1.0 - ax) + r1[a1] * ax;
                mask[yy * bw + xx] = top * (1.0 - ay) + bot * ay > 0.0; // sigmoid(v) > 0.5
            }
        }        dets.push(Det { cls: d[5] as usize, score: d[4], box_: [x1, y1, x2, y2], mask });
    }
    dets
}

pub trait Segmenter: Send {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det>;
}

/// 깊이 모델: RGB → 화소마다 거리(미터). Pi 는 NPU(.dxnn), PC 는 ONNX Runtime(.onnx).
pub trait DepthNet: Send {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> Vec<f32>;
}
/// 특징점 모델(SuperPoint): RGB → 점·기술자(위치 추정용).
pub trait FeatNet: Send {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> crate::vo::Feat;
}

/// 정사각 입력(s²)으로 늘려 넣고 결과를 원래 크기로 되돌리기. 화소 위치·쌍선형 가중은 처음 한 번만 계산(위치표 캐시).
pub struct Square {
    s: usize,
    lut_in: Vec<u32>,
    lut_ax: Vec<(u32, u32, f32)>,
    lut_ay: Vec<(u32, u32, f32)>,
    wh: (usize, usize),
}
impl Square {
    pub fn new(s: usize) -> Square {
        Square { s, lut_in: vec![], lut_ax: vec![], lut_ay: vec![], wh: (0, 0) }
    }
    /// RGB(w×h) → s² RGB(가까운 화소)
    pub fn to_square(&mut self, rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
        let s = self.s;
        if self.wh != (w, h) {
            self.wh = (w, h);
            let xs: Vec<usize> = (0..s).map(|x| (((x as f32 + 0.5) * w as f32 / s as f32) as usize).min(w - 1)).collect();
            let ys: Vec<usize> = (0..s).map(|y| (((y as f32 + 0.5) * h as f32 / s as f32) as usize).min(h - 1)).collect();
            self.lut_in = (0..s * s).map(|i| (ys[i / s] * w + xs[i % s]) as u32 * 3).collect();
            let axis = |n: usize| -> Vec<(u32, u32, f32)> {
                (0..n)
                    .map(|u| {
                        let f = ((u as f32 + 0.5) * s as f32 / n as f32 - 0.5).clamp(0.0, (s - 1) as f32);
                        (f.floor() as u32, (f.floor() as u32 + 1).min(s as u32 - 1), f.fract())
                    })
                    .collect()
            };
            self.lut_ax = axis(w);
            self.lut_ay = axis(h);
        }
        let mut buf = vec![0u8; s * s * 3];
        for (d, &o) in buf.chunks_exact_mut(3).zip(&self.lut_in) {
            let o = o as usize;
            d.copy_from_slice(&rgb[o..o + 3]);
        }
        buf
    }
    /// s² 결과 → w×h(쌍선형)
    pub fn back(&self, d: &[f32]) -> Vec<f32> {
        let (s, (w, h)) = (self.s, self.wh);
        let mut out = vec![0f32; w * h];
        for (v, &(y0, y1, ay)) in self.lut_ay.iter().enumerate() {
            let (r0, r1) = (&d[y0 as usize * s..y0 as usize * s + s], &d[y1 as usize * s..y1 as usize * s + s]);
            let row = &mut out[v * w..v * w + w];
            for (o, &(x0, x1, ax)) in row.iter_mut().zip(&self.lut_ax) {
                let top = r0[x0 as usize] + (r0[x1 as usize] - r0[x0 as usize]) * ax;
                let bot = r1[x0 as usize] + (r1[x1 as usize] - r1[x0 as usize]) * ax;
                *o = top + (bot - top) * ay;
            }
        }
        out
    }
}

/// 흑백(정수 근사 BT.601)
pub fn gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks(3).map(|c| ((77 * c[0] as u32 + 150 * c[1] as u32 + 29 * c[2] as u32) >> 8) as u8).collect()
}

// ------------------------------------------------------------------ PC: ONNX Runtime
/// 한 프로세스에 ONNX Runtime 환경 하나(모델 셋이 같이 씀)
#[cfg(feature = "onnx")]
pub fn ort_env() -> &'static ort::environment::Environment {
    static ENV: std::sync::OnceLock<&'static ort::environment::Environment> = std::sync::OnceLock::new();
    ENV.get_or_init(|| Box::leak(Box::new(ort::init().build().expect("onnxruntime"))))
}
#[cfg(feature = "onnx")]
fn session(path: &str) -> ort::session::Session {
    #[allow(unused_mut)]
    let mut b = ort::session::Session::builder(ort_env()).unwrap().with_intra_threads(8).unwrap();
    // directml 기능이면 그래픽카드로(안 되면 ORT 가 CPU 로 물러남). 입력 크기를 고정해 두면 DirectML 이 빠르다.
    #[cfg(feature = "directml")]
    let mut b = b.with_execution_providers([ort::ep::DirectML::default().build()]).unwrap().with_dimension_override("batch", 1).unwrap();
    b.commit_from_file(path).unwrap_or_else(|e| panic!("onnx model {path}: {e}"))
}

/// 깊이(PC, yolo26s-depth.onnx — 보드 NPU 모델과 같은 구조): 보드와 같이 768² 로 늘려 넣고(RGB/255 NCHW)
/// 출력 [1,1,768,768] 미터를 원래 크기로 쌍선형. 레터박스보다 위치 오차가 작았다(TUM 7 경우 중 6, 책상 0.61 → 0.26 m).
#[cfg(feature = "onnx")]
pub struct OnnxDepth {
    sess: ort::session::Session,
    sq: Square,
}
#[cfg(feature = "onnx")]
impl OnnxDepth {
    pub fn new(path: &str) -> OnnxDepth {
        OnnxDepth { sess: session(path), sq: Square::new(768) }
    }
}
#[cfg(feature = "onnx")]
impl DepthNet for OnnxDepth {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let s = 768;
        let buf = self.sq.to_square(rgb, w, h);
        let inp = ndarray::Array4::<f32>::from_shape_fn((1, 3, s, s), |(_, c, y, x)| buf[(y * s + x) * 3 + c] as f32 / 255.0);
        let out = self.sess.run(ort::inputs![ort::value::TensorRef::from_array_view(&inp).unwrap()]).expect("onnx depth");
        let (_, d) = out[0].try_extract_tensor::<f32>().unwrap();
        self.sq.back(d)
    }
}/// 특징점(PC, SuperPoint ONNX 1024×768 흑백 0–1 입력 → 상위 512 점 (x, y)·점수·256 차원 기술자)
#[cfg(feature = "onnx")]
pub struct OnnxSuperPoint {
    sess: ort::session::Session,
}
#[cfg(feature = "onnx")]
impl OnnxSuperPoint {
    pub fn new(path: &str) -> OnnxSuperPoint {
        OnnxSuperPoint { sess: session(path) }
    }
}
#[cfg(feature = "onnx")]
impl FeatNet for OnnxSuperPoint {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> crate::vo::Feat {
        let (sw, sh) = (1024usize, 768usize);
        let g = gray(rgb);
        // 쌍선형으로 1024×768 에 맞춤
        let (fx, fy) = (w as f32 / sw as f32, h as f32 / sh as f32);
        let inp = ndarray::Array4::<f32>::from_shape_fn((1, 1, sh, sw), |(_, _, y, x)| {
            let u = ((x as f32 + 0.5) * fx - 0.5).clamp(0.0, (w - 1) as f32);
            let v = ((y as f32 + 0.5) * fy - 0.5).clamp(0.0, (h - 1) as f32);
            let (u0, v0) = (u as usize, v as usize);
            let (u1, v1) = ((u0 + 1).min(w - 1), (v0 + 1).min(h - 1));
            let (a, b) = (u - u0 as f32, v - v0 as f32);
            let p = |uu: usize, vv: usize| g[vv * w + uu] as f32;
            ((p(u0, v0) * (1.0 - a) + p(u1, v0) * a) * (1.0 - b) + (p(u0, v1) * (1.0 - a) + p(u1, v1) * a) * b) / 255.0
        });
        let out = self.sess.run(ort::inputs![ort::value::TensorRef::from_array_view(&inp).unwrap()]).expect("onnx superpoint");
        let (_, kp) = out[0].try_extract_tensor::<f32>().unwrap();
        let (_, sc) = out[1].try_extract_tensor::<f32>().unwrap();
        let (_, de) = out[2].try_extract_tensor::<f32>().unwrap();
        let mut f = crate::vo::Feat { kp: vec![], desc: vec![] };
        for i in 0..sc.len() {
            if sc[i] <= 0.005 {
                continue;
            }
            let (u, v) = (((kp[2 * i] + 0.5) * fx - 0.5).clamp(0.0, (w - 1) as f32), ((kp[2 * i + 1] + 0.5) * fy - 0.5).clamp(0.0, (h - 1) as f32));
            let d = &de[i * 256..i * 256 + 256];
            let n = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            f.kp.push((u.round(), v.round()));
            f.desc.extend(d.iter().map(|x| x / n));
        }
        f
    }
}
#[cfg(feature = "onnx")]
pub struct OnnxSeg {
    sess: ort::session::Session,
}
#[cfg(feature = "onnx")]
impl OnnxSeg {
    pub fn new(env: &ort::environment::Environment, path: &str) -> OnnxSeg {
        let _ = env;
        OnnxSeg { sess: session(path) }
    }
}
#[cfg(feature = "onnx")]
impl Segmenter for OnnxSeg {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det> {
        let s = 640;
        let lb = letterbox(rgb, w, h, s);
        let inp = ndarray::Array4::<f32>::from_shape_fn((1, 3, s, s), |(_, c, y, x)| lb.buf[(y * s + x) * 3 + c] as f32 / 255.0);
        let out = self.sess.run(ort::inputs![ort::value::TensorRef::from_array_view(&inp).unwrap()]).expect("onnx run");
        let (_, det) = out[0].try_extract_tensor::<f32>().unwrap();
        let (ps, proto) = out[1].try_extract_tensor::<f32>().unwrap();
        decode(det, proto, ps[1] as usize, ps[2] as usize, ps[3] as usize, s, &lb, w, h, conf)
    }
}

// ------------------------------------------------------------------ Pi5 + DX-M1: DXRT C API
#[cfg(feature = "dxrt")]
pub mod dx {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    #[repr(C)]
    pub struct Options {
        pub struct_size: u32,
        pub buffer_count: c_int,
        pub device_id: c_int,
        pub bound_option: u32,
        pub use_ort: c_int,
        pub reserved: [u32; 6],
    }
    #[repr(C)]
    pub struct TensorInfo {
        pub name: [c_char; 256],
        pub dtype: c_int,
        pub elem_size: u32,
        pub ndim: c_int,
        pub shape: [i64; 8],
        pub size_in_bytes: u64,
    }
    #[link(name = "dxrt")]
    unsafe extern "C" {
        fn dxrt_engine_create(path: *const c_char, opts: *const Options, out: *mut *mut c_void) -> c_int;
        fn dxrt_engine_destroy(e: *mut c_void);
        fn dxrt_engine_run(e: *mut c_void, input: *const c_void, output: *mut c_void) -> c_int;
        fn dxrt_engine_get_output_size(e: *mut c_void, size: *mut u64) -> c_int;
        fn dxrt_engine_get_output_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
        fn dxrt_last_error_message() -> *const c_char;
    }

    /// NPU 모델 하나: 입력 u8 NHWC, 출력은 모든 텐서를 이어 붙인 f32 버퍼
    pub struct Engine {
        e: *mut c_void,
        pub out: Vec<u8>,
        pub tensors: Vec<(String, Vec<i64>, usize)>, // 이름, 모양, 바이트 오프셋
    }
    impl Engine {
        pub fn new(path: &str) -> Engine {
            let o = Options { struct_size: std::mem::size_of::<Options>() as u32, buffer_count: 0, device_id: -1, bound_option: 0, use_ort: -1, reserved: [0; 6] };
            let mut e = std::ptr::null_mut();
            let p = CString::new(path).unwrap();
            unsafe {
                if dxrt_engine_create(p.as_ptr(), &o, &mut e) != 0 {
                    panic!("dxrt: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
                let mut sz = 0u64;
                dxrt_engine_get_output_size(e, &mut sz);
                let mut infos: Vec<TensorInfo> = (0..16).map(|_| std::mem::zeroed()).collect();
                let mut n: c_int = 16;
                dxrt_engine_get_output_tensor_info(e, infos.as_mut_ptr(), &mut n);
                let mut off = 0usize;
                let tensors = infos[..n as usize]
                    .iter()
                    .map(|t| {
                        let name = CStr::from_ptr(t.name.as_ptr()).to_string_lossy().into_owned();
                        let shape = t.shape[..t.ndim as usize].to_vec();
                        let r = (name, shape, off);
                        off += t.size_in_bytes as usize;
                        r
                    })
                    .collect();
                Engine { e, out: vec![0u8; sz as usize], tensors }
            }
        }
        pub fn run(&mut self, input: &[u8]) {
            unsafe {
                if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, self.out.as_mut_ptr() as *mut c_void) != 0 {
                    panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
            }
        }
        pub fn f32_at(&self, i: usize) -> &[f32] {
            let (_, shape, off) = &self.tensors[i];
            let n: i64 = shape.iter().product();
            unsafe { std::slice::from_raw_parts(self.out.as_ptr().add(*off) as *const f32, n as usize) }
        }
    }
    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe { dxrt_engine_destroy(self.e) }
        }
    }
    unsafe impl Send for Engine {}
}

#[cfg(feature = "dxrt")]
/// 깊이(DX-M1, 공식 yolo26-depth-s_768x768): 입력 u8 NHWC 768² RGB(늘려 넣기), 출력 [1,1,768,768] f32(미터).
pub struct DxDepth {
    eng: dx::Engine,
    sq: Square,
}
#[cfg(feature = "dxrt")]
impl DxDepth {
    pub fn new(path: &str) -> DxDepth {
        let eng = dx::Engine::new(path);
        let s = *eng.tensors[0].1.last().unwrap() as usize;
        DxDepth { eng, sq: Square::new(s) }
    }
}
#[cfg(feature = "dxrt")]
impl DepthNet for DxDepth {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let buf = self.sq.to_square(rgb, w, h);
        self.eng.run(&buf);
        self.sq.back(self.eng.f32_at(0))
    }
}#[cfg(feature = "dxrt")]
pub struct DxSuperPoint {
    eng: dx::Engine,
}
#[cfg(feature = "dxrt")]
impl DxSuperPoint {
    pub fn new(path: &str) -> DxSuperPoint {
        DxSuperPoint { eng: dx::Engine::new(path) }
    }
}
#[cfg(feature = "dxrt")]
impl FeatNet for DxSuperPoint {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> crate::vo::Feat {
        assert_eq!((w, h), (640, 480), "superpoint_480x640 expects 640x480");
        self.eng.run(&gray(rgb));
        let semi = self.eng.f32_at(0).to_vec();
        let desc = self.eng.f32_at(1);
        crate::vo::superpoint_decode(&semi, desc, 60, 80, 400, 0.005)
    }
}

#[cfg(feature = "dxrt")]
pub struct DxSeg {
    eng: dx::Engine,
}
#[cfg(feature = "dxrt")]
impl DxSeg {
    pub fn new(path: &str) -> DxSeg {
        DxSeg { eng: dx::Engine::new(path) }
    }
}
#[cfg(feature = "dxrt")]
impl Segmenter for DxSeg {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det> {
        let s = 640;
        let lb = letterbox(rgb, w, h, s);
        self.eng.run(&lb.buf);
        let ps = self.eng.tensors[1].1.clone();
        decode(self.eng.f32_at(0), self.eng.f32_at(1), ps[1] as usize, ps[2] as usize, ps[3] as usize, s, &lb, w, h, conf)
    }
}
