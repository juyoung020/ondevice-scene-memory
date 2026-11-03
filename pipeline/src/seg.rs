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
    for y in 0..nh {
        let sy = (((y as f32 + 0.5) / r) as usize).min(h - 1);
        for x in 0..nw {
            let sx = (((x as f32 + 0.5) / r) as usize).min(w - 1);
            let o = (sy * w + sx) * 3;
            let d = ((py + y) * s + px + x) * 3;
            buf[d..d + 3].copy_from_slice(&rgb[o..o + 3]);
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
        let mut mask = vec![false; bw * bh];
        for yy in 0..bh {
            let fy = (((y1 as usize + yy) as f32 + 0.5) * r + py) * scale - 0.5;
            let y0 = (fy.floor() as i32).clamp(0, ph as i32 - 1) as usize;
            let y1p = (y0 + 1).min(ph - 1);
            let ay = (fy - y0 as f32).clamp(0.0, 1.0);
            for xx in 0..bw {
                let fx = (((x1 as usize + xx) as f32 + 0.5) * r + px) * scale - 0.5;
                let x0 = (fx.floor() as i32).clamp(0, pw as i32 - 1) as usize;
                let x1p = (x0 + 1).min(pw - 1);
                let ax = (fx - x0 as f32).clamp(0.0, 1.0);
                let mut v = 0.0;
                for (k, c) in coef.iter().enumerate() {
                    let b = k * ph * pw;
                    let top = proto[b + y0 * pw + x0] * (1.0 - ax) + proto[b + y0 * pw + x1p] * ax;
                    let bot = proto[b + y1p * pw + x0] * (1.0 - ax) + proto[b + y1p * pw + x1p] * ax;
                    v += c * (top * (1.0 - ay) + bot * ay);
                }
                mask[yy * bw + xx] = v > 0.0; // sigmoid(v) > 0.5
            }
        }
        dets.push(Det { cls: d[5] as usize, score: d[4], box_: [x1, y1, x2, y2], mask });
    }
    dets
}

pub trait Segmenter: Send {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det>;
}

// ------------------------------------------------------------------ PC: ONNX Runtime
#[cfg(feature = "onnx")]
pub struct OnnxSeg {
    sess: ort::session::Session,
}
#[cfg(feature = "onnx")]
impl OnnxSeg {
    pub fn new(env: &ort::environment::Environment, path: &str) -> OnnxSeg {
        OnnxSeg { sess: ort::session::Session::builder(env).unwrap().with_intra_threads(8).unwrap().commit_from_file(path).expect("onnx model") }
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

/// 깊이(DX-M1, 공식 yolo26-depth-s_768x768): 입력 u8 NHWC 768² RGB 를 레터박스 없이 늘려 넣고,
/// 출력 [1,1,768,768] f32(미터)를 원래 크기로 쌍선형 축소.
#[cfg(feature = "dxrt")]
pub struct DxDepth {
    eng: dx::Engine,
    s: usize,
}
#[cfg(feature = "dxrt")]
impl DxDepth {
    pub fn new(path: &str) -> DxDepth {
        let eng = dx::Engine::new(path);
        let s = *eng.tensors[0].1.last().unwrap() as usize;
        DxDepth { eng, s }
    }
    pub fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let s = self.s;
        let mut buf = vec![0u8; s * s * 3];
        for y in 0..s {
            let sy = (((y as f32 + 0.5) * h as f32 / s as f32) as usize).min(h - 1);
            for x in 0..s {
                let sx = (((x as f32 + 0.5) * w as f32 / s as f32) as usize).min(w - 1);
                let o = (sy * w + sx) * 3;
                buf[(y * s + x) * 3..(y * s + x) * 3 + 3].copy_from_slice(&rgb[o..o + 3]);
            }
        }
        self.eng.run(&buf);
        let d = self.eng.f32_at(0);
        let mut out = vec![0f32; w * h];
        for v in 0..h {
            let fy = ((v as f32 + 0.5) * s as f32 / h as f32 - 0.5).clamp(0.0, (s - 1) as f32);
            let (y0, ay) = (fy.floor() as usize, fy.fract());
            let y1 = (y0 + 1).min(s - 1);
            for u in 0..w {
                let fx = ((u as f32 + 0.5) * s as f32 / w as f32 - 0.5).clamp(0.0, (s - 1) as f32);
                let (x0, ax) = (fx.floor() as usize, fx.fract());
                let x1 = (x0 + 1).min(s - 1);
                let top = d[y0 * s + x0] * (1.0 - ax) + d[y0 * s + x1] * ax;
                let bot = d[y1 * s + x0] * (1.0 - ax) + d[y1 * s + x1] * ax;
                out[v * w + u] = top * (1.0 - ay) + bot * ay;
            }
        }
        out
    }
}

/// SuperPoint(DX-M1, 공식 superpoint_480x640): 입력 흑백 u8 480×640, 출력 semi [1,65,60,80]·desc [1,256,60,80]
#[cfg(feature = "dxrt")]
pub struct DxSuperPoint {
    eng: dx::Engine,
}
#[cfg(feature = "dxrt")]
impl DxSuperPoint {
    pub fn new(path: &str) -> DxSuperPoint {
        DxSuperPoint { eng: dx::Engine::new(path) }
    }
    pub fn run(&mut self, rgb: &[u8], w: usize, h: usize) -> crate::vo::Feat {
        assert_eq!((w, h), (640, 480), "superpoint_480x640 expects 640x480");
        let g: Vec<u8> = rgb.chunks(3).map(|c| ((77 * c[0] as u32 + 150 * c[1] as u32 + 29 * c[2] as u32) >> 8) as u8).collect();
        self.eng.run(&g);
        let semi = self.eng.f32_at(0).to_vec();
        let desc = self.eng.f32_at(1);
        crate::vo::superpoint_decode(&semi, desc, 60, 80, 600, 0.005)
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
