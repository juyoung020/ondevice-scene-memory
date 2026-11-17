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
    /// 모델마다 잡는 추론 버퍼 벌 수(0 = 라이브러리 기본 6). 기본 2: 분할(--npu-jobs 1)·사진 벡터 모두 한 장씩 올리고 받는 동안
    /// 다음 장을 올릴 자리면 충분하다. 6 벌이면 분할 모델만 버퍼 130 MB 가 상주했다. --npu-jobs K 로 올리면 --npu-buffers K+1
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
            buffers: get("--npu-buffers").and_then(|s| s.parse().ok()).unwrap_or(2),
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
    fn dxrt_engine_register_callback(e: *mut c_void, cb: DoneCb, user_data: *mut c_void) -> c_int;
    fn dxrt_engine_get_output_size(e: *mut c_void, size: *mut u64) -> c_int;
    fn dxrt_engine_get_output_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
    fn dxrt_engine_get_input_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
    fn dxrt_last_error_message() -> *const c_char;
    fn dxrt_config_set_enable(item: c_int, enabled: c_int) -> c_int;
    fn dxrt_config_set_attribute(item: c_int, attr: c_int, value: *const c_char) -> c_int;
}

/// NPU 모델 하나: 입력 바이트 그대로(u8 NHWC 또는 f32), 출력은 모든 텐서를 이어 붙인 f32 버퍼.
/// 출력은 받을 쪽 버퍼를 올릴 때(`run_into` · `submit`) 함께 넘겨 DXRT 가 거기에 바로 쓴다(DXRT 안 버퍼 → 우리 버퍼 memcpy 없음:
/// `dxrt_engine_wait` 는 출력 포인터가 있으면 텐서마다 memcpy, NULL 이면 건너뜀 — libdxrt 3.4.0 디스어셈블로 확인, docs/threads_npu.md)
pub struct Engine {
    e: *mut c_void,
    /// 출력 전체 바이트(텐서를 이어 붙임)
    pub out_len: usize,
    pub tensors: Vec<(String, Vec<i64>, usize)>, // 이름, 모양, 바이트 오프셋
    pub input: String,                          // 첫 입력 텐서 이름
    /// 첫 입력 텐서 바이트
    pub in_len: usize,
    /// 끝 알림(`on_done`): 얇은 기다림 스레드에 넘길 줄, 또는 콜백이 부를 함수(엔진보다 오래 삶)
    done: Option<Done>,
}

/// NPU 작업 끝을 루프에 알리는 길(`--npu-wake`): 콜백 = DXRT 안 스레드가 부르는 함수가 알림(우리 스레드 0),
/// 스레드 = 모델마다 `wait` 만 하는 얇은 스레드가 올린 차례대로 기다렸다 알림
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wake {
    Thread,
    Callback,
}
static WAKE_CB: AtomicBool = AtomicBool::new(false);
/// `--npu-wake callback|thread`(기본 thread: 보드 시험 `ondevice npu-test` 를 통과한 뒤 callback 으로 바꿈)
pub fn set_wake(w: Wake) {
    WAKE_CB.store(w == Wake::Callback, Relaxed);
}
pub fn wake() -> Wake {
    if WAKE_CB.load(Relaxed) { Wake::Callback } else { Wake::Thread }
}

/// 작업 끝 알림: 올릴 때 준 표(token)를 그대로 받는다. 어느 스레드에서든 불린다
pub type Notify = std::sync::Arc<dyn Fn(u64) + Send + Sync>;

type DoneCb = unsafe extern "C" fn(outs: *mut c_void, n: c_int, user_arg: *mut c_void, user_data: *mut c_void) -> c_int;

enum Done {
    /// 얇은 스레드로 (작업 번호, 표)를 넘기는 줄
    Thread(std::sync::mpsc::Sender<(c_int, u64)>, Option<std::thread::JoinHandle<()>>),
    /// 콜백 user_data 로 넘긴 함수(엔진을 없앨 때 함께)
    Callback(*mut Notify),
}

/// DXRT 가 작업 끝에 부르는 함수: user_arg = 올릴 때 준 표, user_data = 알림 함수
unsafe extern "C" fn on_job_done(_outs: *mut c_void, _n: c_int, user_arg: *mut c_void, user_data: *mut c_void) -> c_int {
    if !user_data.is_null() {
        let f = unsafe { &*(user_data as *const Notify) };
        f(user_arg as u64);
    }
    0
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
            let in_len = if n > 0 { infos[0].size_in_bytes as usize } else { 0 };
            Ok(Engine { e, out_len: sz as usize, tensors, input, in_len, done: None })
        }
    }
    /// 동기 실행: 결과를 받는 쪽 버퍼(f32, 모든 텐서 이어 붙임)에 바로 받기
    pub fn run_into(&mut self, input: &[u8], out: &mut Vec<f32>) {
        out.resize(self.out_len.div_ceil(4), 0.0);
        unsafe {
            if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void) != 0 {
                panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
    }
    /// 추론 하나를 NPU 에 올려 두고 바로 돌아옴(작업 번호). 결과는 out 에 바로 쓰인다.
    /// 입력 버퍼와 out 은 `Waiter::wait` 가 끝날 때까지 그대로 둔다(옮기지도 크기를 바꾸지도 않음 — Vec 를 옮기는 것은 괜찮음, 힙 자리는 그대로).
    pub fn submit(&mut self, input: &[u8], out: &mut Vec<f32>) -> i32 {
        out.resize(self.out_len.div_ceil(4), 0.0);
        let mut id: c_int = 0;
        unsafe {
            if dxrt_engine_run_async(self.e, input.as_ptr() as *const c_void, std::ptr::null_mut(), out.as_mut_ptr() as *mut c_void, &mut id) != 0 {
                panic!("dxrt run_async: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
        id
    }
    /// 작업 끝 알림을 단다(`submit_tagged` 로 올린 작업마다 notify(표) 한 번). name = 기다림 스레드 이름 뒤에 붙일 모델 이름
    pub fn on_done(&mut self, mode: Wake, name: &str, notify: Notify) {
        match mode {
            Wake::Callback => {
                let p = Box::into_raw(Box::new(notify));
                unsafe {
                    if dxrt_engine_register_callback(self.e, on_job_done, p as *mut c_void) != 0 {
                        panic!("dxrt register_callback: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                    }
                }
                self.done = Some(Done::Callback(p));
            }
            Wake::Thread => {
                let (tx, rx) = std::sync::mpsc::channel::<(c_int, u64)>();
                let w = self.waiter();
                let th = std::thread::Builder::new()
                    .name(format!("npu-wait-{name}"))
                    .spawn(move || {
                        crate::prof::track_thread();
                        for (id, tag) in rx {
                            w.wait(id);
                            notify(tag);
                        }
                    })
                    .expect("npu-wait 스레드");
                self.done = Some(Done::Thread(tx, Some(th)));
            }
        }
    }
    /// 올리기(끝 알림 표 붙임): 결과는 out 에 바로. 끝나면 `on_done` 의 notify(tag).
    /// 콜백 길이면 알림을 받은 뒤 `Waiter::wait(id)` 로 작업 자리를 놓는다(이미 끝나 바로 돌아옴)
    pub fn submit_tagged(&mut self, input: &[u8], out: &mut Vec<f32>, tag: u64) -> i32 {
        out.resize(self.out_len.div_ceil(4), 0.0);
        let mut id: c_int = 0;
        unsafe {
            if dxrt_engine_run_async(self.e, input.as_ptr() as *const c_void, tag as *mut c_void, out.as_mut_ptr() as *mut c_void, &mut id) != 0 {
                panic!("dxrt run_async: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
        if let Some(Done::Thread(tx, _)) = &self.done {
            let _ = tx.send((id, tag));
        }
        id
    }
    /// 끝 알림이 콜백 길인지(알림 뒤 `wait` 로 자리를 놓아야 함)
    pub fn callback_mode(&self) -> bool {
        matches!(self.done, Some(Done::Callback(_)))
    }
    /// 올려 둔 작업의 끝을 다른 스레드에서 기다리는 손잡이. 엔진보다 먼저 없앤다.
    pub fn waiter(&self) -> Waiter {
        Waiter { e: self.e }
    }
    /// 텐서 i 의 (f32 시작, 길이)
    pub fn f32_span(&self, i: usize) -> (usize, usize) {
        let (_, shape, off) = &self.tensors[i];
        let n: i64 = shape.iter().product();
        assert!(*off % 4 == 0, "dxrt 출력 텐서 {i} 의 시작이 4 바이트 경계가 아님");
        (*off / 4, n as usize)
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        // 기다림 스레드를 먼저 끝냄(줄을 닫으면 남은 작업을 다 기다리고 끝남)
        if let Some(Done::Thread(tx, th)) = &mut self.done {
            let (dead, _) = std::sync::mpsc::channel();
            *tx = dead;
            if let Some(t) = th.take() {
                let _ = t.join();
            }
        }
        unsafe { dxrt_engine_destroy(self.e) }
        if let Some(Done::Callback(p)) = self.done.take() {
            drop(unsafe { Box::from_raw(p) });
        }
    }
}
unsafe impl Send for Engine {}

pub struct Waiter {
    e: *mut c_void,
}
impl Waiter {
    /// 작업이 끝날 때까지 기다림. 결과는 `Engine::submit` 에 준 버퍼에 이미 쓰여 있다(출력 포인터 NULL = DXRT 가 다시 복사하지 않음)
    pub fn wait(&self, id: i32) {
        unsafe {
            if dxrt_engine_wait(self.e, id, std::ptr::null_mut()) != 0 {
                panic!("dxrt wait: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
            }
        }
    }
}
unsafe impl Send for Waiter {}
