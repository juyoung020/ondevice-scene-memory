//! Pi 5 + DX-M1: DXRT 3.4 C API(`dxrt_c_api.h`) — NPU 모델 하나 열기·돌리기와 런타임 설정.
//! 설정은 [`Config`] 하나로 모은다(명령줄 `--seg-npu` · `--embed-npu` · `--npu-buffers` · `--npu-nfh` ·
//! `--npu-profile` · `--npu-ort`). 무엇을 왜 골랐는지는 docs/npu_models.md "DXRT 설정".
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::Relaxed};

/// NPU 코어 묶기(DXRT bound_option): 0 = 셋 다, 1 · 2 · 3 = 코어 0 · 1 · 2, 4 = 0+1, 5 = 1+2, 6 = 0+2.
/// 분할(`--seg-npu`)과 사진 벡터(`--embed-npu`)를 다른 코어에 두면 서로 줄 서지 않고 함께 돈다
pub static SEG_BOUND: AtomicU32 = AtomicU32::new(0);
pub static EMBED_BOUND: AtomicU32 = AtomicU32::new(0);
/// 분할 · 사진 벡터 모델이 잡는 입출력 버퍼 수(DXRT buffer_count, 0 = 라이브러리 기본 6). 함께 올릴 수 있는 추론 수의 상한
pub static BUFFERS: AtomicI32 = AtomicI32::new(0);
/// 모델 안 CPU 작업을 DXRT 의 ONNX Runtime 으로 돌릴지(use_ort). 우리 모델은 모두 NPU 작업 하나라 기본은 끔
pub static USE_ORT: AtomicBool = AtomicBool::new(false);

/// "all" · "0" · "1" · "2" · "01" · "12" · "02" → bound_option
pub fn parse_bound(s: &str) -> u32 {
    match s {
        "all" | "" => 0,
        "0" => 1,
        "1" => 2,
        "2" => 3,
        "01" | "10" => 4,
        "12" | "21" => 5,
        "02" | "20" => 6,
        _ => panic!("NPU 코어 지정 {s}: all · 0 · 1 · 2 · 01 · 12 · 02"),
    }
}

/// 런타임 설정(첫 모델을 열기 전, 스레드를 만들기 전에 [`Config::apply`])
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub seg_bound: String,
    pub embed_bound: String,
    /// 0 = 라이브러리 기본(6)
    pub buffers: i32,
    /// NPU 형식 변환(NFH) 일꾼 스레드 수(입력, 출력). None = 라이브러리 기본(ARM 1, 2)
    pub nfh: Option<(u32, u32)>,
    /// DXRT 프로파일러: 끝날 때 작업 폴더에 profiler.json
    pub profile: bool,
    /// 모델 안 CPU 작업 허용(ONNX Runtime)
    pub ort: bool,
}

impl Config {
    /// 명령줄 값 읽기(`get("--키")` 값, `has("--키")` 표시)
    pub fn from_args(get: impl Fn(&str) -> Option<String>, has: impl Fn(&str) -> bool) -> Config {
        Config {
            seg_bound: get("--seg-npu").unwrap_or_default(),
            embed_bound: get("--embed-npu").unwrap_or_default(),
            buffers: get("--npu-buffers").and_then(|s| s.parse().ok()).unwrap_or(0),
            nfh: get("--npu-nfh").and_then(|s| {
                let (a, b) = s.split_once(',')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            }),
            profile: has("--npu-profile"),
            ort: has("--npu-ort"),
        }
    }
    pub fn apply(&self) {
        SEG_BOUND.store(parse_bound(&self.seg_bound), Relaxed);
        EMBED_BOUND.store(parse_bound(&self.embed_bound), Relaxed);
        BUFFERS.store(self.buffers.clamp(0, 100), Relaxed);
        USE_ORT.store(self.ort, Relaxed);
        if let Some((i, o)) = self.nfh {
            // 라이브러리가 처음 쓸 때 한 번 읽는 환경 변수(C API 로는 못 바꿈)
            // SAFETY: main 초입, 다른 스레드가 환경을 읽기 전
            unsafe {
                std::env::set_var("NFH_INPUT_WORKER_THREADS", i.to_string());
                std::env::set_var("NFH_OUTPUT_WORKER_THREADS", o.to_string());
            }
        }
        if self.profile {
            let v = CString::new("1").unwrap();
            unsafe {
                dxrt_config_set_enable(CONFIG_PROFILER, 1);
                dxrt_config_set_attribute(CONFIG_PROFILER, ATTR_PROFILER_SAVE_DATA, v.as_ptr());
            }
        }
    }
}

const CONFIG_PROFILER: c_int = 2;
const ATTR_PROFILER_SAVE_DATA: c_int = 1002;

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
    fn dxrt_engine_run_async(e: *mut c_void, input: *const c_void, user_arg: *mut c_void, output: *mut c_void, job_id: *mut c_int) -> c_int;
    fn dxrt_engine_wait(e: *mut c_void, job_id: c_int, output: *mut c_void) -> c_int;
    fn dxrt_engine_get_output_size(e: *mut c_void, size: *mut u64) -> c_int;
    fn dxrt_engine_get_output_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
    fn dxrt_engine_get_input_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
    fn dxrt_last_error_message() -> *const c_char;
    fn dxrt_config_set_enable(item: c_int, enabled: c_int) -> c_int;
    fn dxrt_config_set_attribute(item: c_int, attr: c_int, value: *const c_char) -> c_int;
}

/// NPU 모델 하나: 입력 바이트 그대로(u8 NHWC 또는 f32), 출력은 모든 텐서를 이어 붙인 버퍼
pub struct Engine {
    e: *mut c_void,
    pub out: Vec<u8>,
    pub tensors: Vec<(String, Vec<i64>, usize)>, // 이름, 모양, 바이트 오프셋
    pub input: String,                          // 첫 입력 텐서 이름
}
impl Engine {
    /// 분할 모델(`--seg-npu` 코어)
    pub fn new(path: &str) -> Engine {
        Engine::try_open(path, SEG_BOUND.load(Relaxed), BUFFERS.load(Relaxed)).unwrap_or_else(|e| panic!("{e}"))
    }
    /// 사진 벡터 모델(`--embed-npu` 코어)
    pub fn embed(path: &str) -> Engine {
        Engine::try_open(path, EMBED_BOUND.load(Relaxed), BUFFERS.load(Relaxed)).unwrap_or_else(|e| panic!("{e}"))
    }
    pub fn try_open(path: &str, bound_option: u32, buffer_count: i32) -> Result<Engine, String> {
        let use_ort = USE_ORT.load(Relaxed) as c_int;
        let o = Options { struct_size: std::mem::size_of::<Options>() as u32, buffer_count, device_id: -1, bound_option, use_ort, reserved: [0; 6] };
        let mut e = std::ptr::null_mut();
        let p = CString::new(path).map_err(|e| e.to_string())?;
        unsafe {
            if dxrt_engine_create(p.as_ptr(), &o, &mut e) != 0 {
                return Err(format!("dxrt {path}: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy()));
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
            let mut n: c_int = 16;
            dxrt_engine_get_input_tensor_info(e, infos.as_mut_ptr(), &mut n);
            let input = if n > 0 { CStr::from_ptr(infos[0].name.as_ptr()).to_string_lossy().into_owned() } else { String::new() };
            Ok(Engine { e, out: vec![0u8; sz as usize], tensors, input })
        }
    }
    pub fn run(&mut self, input: &[u8]) {
        unsafe {
            if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, self.out.as_mut_ptr() as *mut c_void) != 0 {
                panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
    }
    /// 결과를 받는 쪽 버퍼(f32, 모든 텐서 이어 붙임)에 바로 받기
    pub fn run_into(&mut self, input: &[u8], out: &mut Vec<f32>) {
        out.resize(self.out.len().div_ceil(4), 0.0);
        unsafe {
            if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void) != 0 {
                panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
    }
    /// 추론 하나를 NPU 에 올려 두고 바로 돌아옴(작업 번호). 입력 버퍼는 `Waiter::wait_into` 가 끝날 때까지 그대로 둔다.
    pub fn submit(&mut self, input: &[u8]) -> i32 {
        let mut id: c_int = 0;
        unsafe {
            if dxrt_engine_run_async(self.e, input.as_ptr() as *const c_void, std::ptr::null_mut(), std::ptr::null_mut(), &mut id) != 0 {
                panic!("dxrt run_async: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
        id
    }
    /// 올려 둔 작업의 결과를 다른 스레드에서 받는 손잡이. 엔진보다 먼저 없앤다.
    pub fn waiter(&self) -> Waiter {
        Waiter { e: self.e, len: self.out.len() }
    }
    /// 텐서 i 의 (f32 시작, 길이)
    pub fn f32_span(&self, i: usize) -> (usize, usize) {
        let (_, shape, off) = &self.tensors[i];
        let n: i64 = shape.iter().product();
        assert!(*off % 4 == 0, "dxrt 출력 텐서 {i} 의 시작이 4 바이트 경계가 아님");
        (*off / 4, n as usize)
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

pub struct Waiter {
    e: *mut c_void,
    len: usize,
}
impl Waiter {
    /// 작업이 끝날 때까지 기다려 결과(f32, 모든 텐서 이어 붙임)를 받음
    pub fn wait_into(&self, id: i32, out: &mut Vec<f32>) {
        out.resize(self.len.div_ceil(4), 0.0);
        unsafe {
            if dxrt_engine_wait(self.e, id, out.as_mut_ptr() as *mut c_void) != 0 {
                panic!("dxrt wait: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
    }
}
unsafe impl Send for Waiter {}
