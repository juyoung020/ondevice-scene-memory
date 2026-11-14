//! librealsense2 C API(rs.h, rs_internal.h v2.59.1)를 직접 부름(librealsense2.so 링크, slam/src/io/realsense.rs 와 같은 방식)
//! 과 카메라 없이 rs2::align 을 돌리는 software_device 래퍼, SDK 가 고른 정렬 구현을 로그에서 잡는 것.
use crate::fast_align::{Extrinsics, Intrinsics};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

pub type P = *mut c_void;

#[repr(C)]
pub struct VideoStream {
    pub kind: i32,
    pub index: i32,
    pub uid: i32,
    pub width: i32,
    pub height: i32,
    pub fps: i32,
    pub bpp: i32,
    pub fmt: i32,
    pub intrinsics: Intrinsics,
}

#[repr(C)]
pub struct SoftwareVideoFrame {
    pub pixels: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut c_void)>,
    pub stride: i32,
    pub bpp: i32,
    pub timestamp: f64,
    pub domain: i32,
    pub frame_number: i32,
    pub profile: *const c_void,
    pub depth_units: f32,
}

pub type LogCb = unsafe extern "C" fn(severity: i32, msg: *const c_void, arg: *mut c_void);
pub type FrameCb = unsafe extern "C" fn(frame: P, user: *mut c_void);

#[link(name = "realsense2")]
unsafe extern "C" {
    pub fn rs2_get_api_version(e: *mut P) -> i32;
    pub fn rs2_create_context(api: i32, e: *mut P) -> P;
    pub fn rs2_query_devices(ctx: P, e: *mut P) -> P;
    pub fn rs2_get_device_count(list: P, e: *mut P) -> i32;
    pub fn rs2_create_device(list: P, i: i32, e: *mut P) -> P;
    pub fn rs2_supports_device_info(dev: P, info: i32, e: *mut P) -> i32;
    pub fn rs2_get_device_info(dev: P, info: i32, e: *mut P) -> *const c_char;
    pub fn rs2_query_sensors(dev: P, e: *mut P) -> P;
    pub fn rs2_get_sensors_count(list: P, e: *mut P) -> i32;
    pub fn rs2_create_sensor(list: P, i: i32, e: *mut P) -> P;
    pub fn rs2_supports_sensor_info(s: P, info: i32, e: *mut P) -> i32;
    pub fn rs2_get_sensor_info(s: P, info: i32, e: *mut P) -> *const c_char;
    pub fn rs2_is_sensor_extendable_to(s: P, ext: i32, e: *mut P) -> i32;
    pub fn rs2_supports_option(s: P, opt: i32, e: *mut P) -> i32;
    pub fn rs2_get_option(s: P, opt: i32, e: *mut P) -> f32;
    pub fn rs2_set_option(s: P, opt: i32, v: f32, e: *mut P);
    pub fn rs2_get_depth_scale(s: P, e: *mut P) -> f32;
    pub fn rs2_get_stream_profiles(s: P, e: *mut P) -> P;
    pub fn rs2_get_stream_profiles_count(list: P, e: *mut P) -> i32;
    pub fn rs2_get_stream_profile(list: P, i: i32, e: *mut P) -> P;
    pub fn rs2_get_stream_profile_data(p: P, stream: *mut i32, format: *mut i32, index: *mut i32, uid: *mut i32, fps: *mut i32, e: *mut P);
    pub fn rs2_stream_profile_is(p: P, ext: i32, e: *mut P) -> i32;
    pub fn rs2_get_video_stream_resolution(p: P, w: *mut i32, h: *mut i32, e: *mut P);
    pub fn rs2_get_video_stream_intrinsics(p: P, k: *mut Intrinsics, e: *mut P);
    pub fn rs2_get_extrinsics(from: P, to: P, x: *mut Extrinsics, e: *mut P);
    pub fn rs2_register_extrinsics(from: P, to: P, x: Extrinsics, e: *mut P);
    pub fn rs2_distortion_to_string(d: i32) -> *const c_char;
    pub fn rs2_option_to_string(o: i32) -> *const c_char;
    pub fn rs2_stream_to_string(s: i32) -> *const c_char;
    pub fn rs2_format_to_string(f: i32) -> *const c_char;
    pub fn rs2_rs400_visual_preset_to_string(p: i32) -> *const c_char;
    // 파이프라인
    pub fn rs2_create_pipeline(ctx: P, e: *mut P) -> P;
    pub fn rs2_create_config(e: *mut P) -> P;
    pub fn rs2_config_enable_device(cfg: P, serial: *const c_char, e: *mut P);
    pub fn rs2_config_enable_stream(cfg: P, stream: i32, index: i32, w: i32, h: i32, format: i32, fps: i32, e: *mut P);
    pub fn rs2_pipeline_start_with_config(pipe: P, cfg: P, e: *mut P) -> P;
    pub fn rs2_pipeline_stop(pipe: P, e: *mut P);
    pub fn rs2_pipeline_wait_for_frames(pipe: P, timeout: u32, e: *mut P) -> P;
    pub fn rs2_pipeline_profile_get_device(prof: P, e: *mut P) -> P;
    pub fn rs2_pipeline_profile_get_streams(prof: P, e: *mut P) -> P;
    // 장면·처리
    pub fn rs2_create_align(to: i32, e: *mut P) -> P;
    pub fn rs2_create_sync_processing_block(e: *mut P) -> P;
    pub fn rs2_create_frame_queue(cap: i32, e: *mut P) -> P;
    pub fn rs2_start_processing_queue(b: P, q: P, e: *mut P);
    pub fn rs2_process_frame(b: P, f: P, e: *mut P);
    pub fn rs2_poll_for_frame(q: P, out: *mut P, e: *mut P) -> i32;
    pub fn rs2_try_wait_for_frame(q: P, timeout: u32, out: *mut P, e: *mut P) -> i32;
    pub fn rs2_wait_for_frame(q: P, timeout: u32, e: *mut P) -> P;
    pub fn rs2_embedded_frames_count(f: P, e: *mut P) -> i32;
    pub fn rs2_extract_frame(f: P, i: i32, e: *mut P) -> P;
    pub fn rs2_get_frame_stream_profile(f: P, e: *mut P) -> P;
    pub fn rs2_is_frame_extendable_to(f: P, ext: i32, e: *mut P) -> i32;
    pub fn rs2_get_frame_data(f: P, e: *mut P) -> *const u8;
    pub fn rs2_get_frame_width(f: P, e: *mut P) -> i32;
    pub fn rs2_get_frame_height(f: P, e: *mut P) -> i32;
    pub fn rs2_get_frame_stride_in_bytes(f: P, e: *mut P) -> i32;
    pub fn rs2_supports_frame_metadata(f: P, m: i32, e: *mut P) -> i32;
    pub fn rs2_get_frame_metadata(f: P, m: i32, e: *mut P) -> i64;
    pub fn rs2_get_frame_timestamp(f: P, e: *mut P) -> f64;
    pub fn rs2_get_frame_number(f: P, e: *mut P) -> u64;
    pub fn rs2_frame_add_ref(f: P, e: *mut P);
    pub fn rs2_release_frame(f: P);
    // software_device
    pub fn rs2_create_software_device(e: *mut P) -> P;
    pub fn rs2_software_device_add_sensor(dev: P, name: *const c_char, e: *mut P) -> P;
    pub fn rs2_software_sensor_add_video_stream_ex(s: P, vs: VideoStream, is_default: i32, e: *mut P) -> P;
    pub fn rs2_software_sensor_add_read_only_option(s: P, opt: i32, v: f32, e: *mut P);
    pub fn rs2_software_sensor_on_video_frame(s: P, f: SoftwareVideoFrame, e: *mut P);
    pub fn rs2_software_device_create_matcher(dev: P, m: i32, e: *mut P);
    pub fn rs2_open(s: P, p: P, e: *mut P);
    pub fn rs2_close(s: P, e: *mut P);
    pub fn rs2_start(s: P, cb: FrameCb, user: *mut c_void, e: *mut P);
    pub fn rs2_stop(s: P, e: *mut P);
    // 로그
    pub fn rs2_log_to_callback(min: i32, cb: LogCb, arg: *mut c_void, e: *mut P);
    pub fn rs2_get_raw_log_message(msg: *const c_void, e: *mut P) -> *const c_char;
    // 오류
    pub fn rs2_get_error_message(e: P) -> *const c_char;
    pub fn rs2_get_failed_function(e: P) -> *const c_char;
    pub fn rs2_free_error(e: P);
}

unsafe extern "C" {
    fn malloc(n: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

// rs.h 2.59.1 의 값(헤더에서 확인)
pub const API_VERSION: i32 = 25901;
pub const STREAM_DEPTH: i32 = 1;
pub const STREAM_COLOR: i32 = 2;
pub const STREAM_GYRO: i32 = 5;
pub const STREAM_ACCEL: i32 = 6;
pub const STREAM_MOTION: i32 = 10;
pub const FORMAT_Z16: i32 = 1;
pub const FORMAT_RGB8: i32 = 5;
pub const OPT_EXPOSURE: i32 = 3;
pub const OPT_GAIN: i32 = 4;
pub const OPT_ENABLE_AUTO_EXPOSURE: i32 = 10;
pub const OPT_VISUAL_PRESET: i32 = 12;
pub const OPT_LASER_POWER: i32 = 13;
pub const OPT_EMITTER_ENABLED: i32 = 18;
pub const OPT_DEPTH_UNITS: i32 = 28;
pub const OPT_AUTO_EXPOSURE_PRIORITY: i32 = 30;
pub const OPT_GLOBAL_TIME_ENABLED: i32 = 53;
pub const INFO_NAME: i32 = 0;
pub const INFO_SERIAL: i32 = 1;
pub const INFO_FIRMWARE: i32 = 2;
pub const INFO_RECOMMENDED_FIRMWARE: i32 = 3;
pub const INFO_PRODUCT_ID: i32 = 7;
pub const INFO_USB_TYPE: i32 = 9;
pub const INFO_PRODUCT_LINE: i32 = 10;
pub const EXT_DEPTH_SENSOR: i32 = 7;
pub const EXT_DEPTH_FRAME: i32 = 12;
pub const EXT_VIDEO_PROFILE: i32 = 15;
pub const EXT_COLOR_SENSOR: i32 = 42;
pub const MATCHER_DLR_C: i32 = 2;
pub const TIMESTAMP_DOMAIN_HARDWARE_CLOCK: i32 = 0;
pub const LOG_SEVERITY_INFO: i32 = 1;
pub const META_ACTUAL_EXPOSURE: i32 = 3;
pub const META_GAIN_LEVEL: i32 = 4;
pub const RS400_VISUAL_PRESET_COUNT: i32 = 8;

/// librealsense 오류(rs2::error 의 what() 과 실패한 함수 이름) 또는 다른 오류
#[derive(Debug)]
pub enum Error {
    Rs { what: String, func: String },
    Other(String),
}

impl From<String> for Error {
    fn from(s: String) -> Error {
        Error::Other(s)
    }
}

/// rs2 함수를 부르고 오류가 있으면 Error::Rs
pub fn ck<T>(f: impl FnOnce(*mut P) -> T) -> Result<T, Error> {
    let mut e: P = std::ptr::null_mut();
    let v = f(&mut e);
    if e.is_null() {
        Ok(v)
    } else {
        let what = unsafe { cstr(rs2_get_error_message(e)) };
        let func = unsafe { cstr(rs2_get_failed_function(e)) };
        unsafe { rs2_free_error(e) };
        Err(Error::Rs { what, func })
    }
}

pub fn cstr(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() } }
}

/// RS2_API_FULL_VERSION_STR 과 같은 모양("2.59.1.0")
pub fn api_version() -> String {
    let v = ck(|e| unsafe { rs2_get_api_version(e) }).unwrap_or(0);
    format!("{}.{}.{}.0", v / 10000, v / 100 % 100, v % 100)
}

pub fn distortion_name(m: i32) -> String {
    cstr(unsafe { rs2_distortion_to_string(m) })
}

pub fn intrin_json(i: &Intrinsics) -> String {
    crate::common::intrin_json(i, &distortion_name(i.model))
}

pub fn device_info(dev: P, k: i32) -> Result<String, Error> {
    Ok(cstr(ck(|e| unsafe { rs2_get_device_info(dev, k, e) })?))
}

/// 묶음 장면에서 컬러(스트림 COLOR)·깊이(깊이 장면) 하나씩(없으면 null). 돌려준 장면은 rs2_release_frame 으로 놓아야 함
pub fn split(fs: P) -> Result<(P, P), Error> {
    let (mut color, mut depth): (P, P) = (std::ptr::null_mut(), std::ptr::null_mut());
    let n = ck(|e| unsafe { rs2_embedded_frames_count(fs, e) })?;
    for i in 0..n {
        let f = ck(|e| unsafe { rs2_extract_frame(fs, i, e) })?;
        let prof = ck(|e| unsafe { rs2_get_frame_stream_profile(f, e) })?;
        let (mut st, mut fm, mut ix, mut uid, mut fps) = (0, 0, 0, 0, 0);
        ck(|e| unsafe { rs2_get_stream_profile_data(prof, &mut st, &mut fm, &mut ix, &mut uid, &mut fps, e) })?;
        if color.is_null() && st == STREAM_COLOR {
            color = f;
        } else if depth.is_null() && ck(|e| unsafe { rs2_is_frame_extendable_to(f, EXT_DEPTH_FRAME, e) })? != 0 {
            depth = f;
        } else {
            unsafe { rs2_release_frame(f) };
        }
    }
    Ok((color, depth))
}

pub fn release(fs: &[P]) {
    for &f in fs {
        if !f.is_null() {
            unsafe { rs2_release_frame(f) };
        }
    }
}

/// rs2::align::process 와 같음: 묶음 장면을 넘기고(참조 하나 더함) 결과를 꺼냄. 없으면 rs2::filter 와 같은 오류
pub fn align_process(align: P, queue: P, fs: P) -> Result<P, Error> {
    ck(|e| unsafe { rs2_frame_add_ref(fs, e) })?;
    ck(|e| unsafe { rs2_process_frame(align, fs, e) })?;
    let mut out: P = std::ptr::null_mut();
    if ck(|e| unsafe { rs2_poll_for_frame(queue, &mut out, e) })? == 0 {
        return Err(Error::Other("Error occured during execution of the processing block! See the log for more info".into()));
    }
    Ok(out)
}

/// 정렬 블록(rs2::align(COLOR), 큐 1 칸)
pub fn create_align() -> Result<(P, P), Error> {
    let a = ck(|e| unsafe { rs2_create_align(STREAM_COLOR, e) })?;
    let q = ck(|e| unsafe { rs2_create_frame_queue(1, e) })?;
    ck(|e| unsafe { rs2_start_processing_queue(a, q, e) })?;
    Ok((a, q))
}

// ---- SDK 가 고른 정렬 구현(LOG_INFO "Using ... align implementation")을 잡음 ----
static ALIGN_IMPL: Mutex<String> = Mutex::new(String::new());

unsafe extern "C" fn on_log(_sev: i32, msg: *const c_void, _arg: *mut c_void) {
    let r = cstr(ck(|e| unsafe { rs2_get_raw_log_message(msg, e) }).unwrap_or(std::ptr::null()));
    if r.contains("align implementation") {
        *ALIGN_IMPL.lock().unwrap() = r;
    }
}

pub fn align_impl_log_install() {
    let _ = ck(|e| unsafe { rs2_log_to_callback(LOG_SEVERITY_INFO, on_log, std::ptr::null_mut(), e) });
}

pub fn align_impl_text() -> String {
    ALIGN_IMPL.lock().unwrap().clone()
}

// ---- 카메라 없이 rs2::align(SDK 실제 코드)을 돌림 ----

unsafe extern "C" fn to_syncer(frame: P, user: *mut c_void) {
    // rs2::syncer 에 넘김(장면 소유도 넘어감)
    let _ = ck(|e| unsafe { rs2_process_frame(user, frame, e) });
}

pub struct SdkAligner {
    di: Intrinsics,
    ci: Intrinsics,
    unit: f32,
    depth_s: P,
    color_s: P,
    dprof: P,
    cprof: P,
    sync_q: P,
    align: P,
    align_q: P,
    color: Vec<u8>,
    n: i32,
}

impl SdkAligner {
    pub fn new(di: &Intrinsics, ci: &Intrinsics, d2c: &Extrinsics, unit: f32) -> Result<SdkAligner, Error> {
        let (align, align_q) = create_align()?;
        let dev = ck(|e| unsafe { rs2_create_software_device(e) })?;
        let (dn, cn) = (CString::new("Depth").unwrap(), CString::new("Color").unwrap());
        let depth_s = ck(|e| unsafe { rs2_software_device_add_sensor(dev, dn.as_ptr(), e) })?;
        let color_s = ck(|e| unsafe { rs2_software_device_add_sensor(dev, cn.as_ptr(), e) })?;
        let ds = VideoStream { kind: STREAM_DEPTH, index: 0, uid: 0, width: di.width, height: di.height, fps: 30, bpp: 2, fmt: FORMAT_Z16, intrinsics: *di };
        let cs = VideoStream { kind: STREAM_COLOR, index: 0, uid: 1, width: ci.width, height: ci.height, fps: 30, bpp: 3, fmt: FORMAT_RGB8, intrinsics: *ci };
        let dprof = ck(|e| unsafe { rs2_software_sensor_add_video_stream_ex(depth_s, ds, 1, e) })?;
        let cprof = ck(|e| unsafe { rs2_software_sensor_add_video_stream_ex(color_s, cs, 1, e) })?;
        ck(|e| unsafe { rs2_software_sensor_add_read_only_option(depth_s, OPT_DEPTH_UNITS, unit, e) })?;
        ck(|e| unsafe { rs2_register_extrinsics(dprof, cprof, *d2c, e) })?;
        ck(|e| unsafe { rs2_software_device_create_matcher(dev, MATCHER_DLR_C, e) })?;
        ck(|e| unsafe { rs2_open(depth_s, dprof, e) })?;
        ck(|e| unsafe { rs2_open(color_s, cprof, e) })?;
        let sync = ck(|e| unsafe { rs2_create_sync_processing_block(e) })?;
        let sync_q = ck(|e| unsafe { rs2_create_frame_queue(1, e) })?;
        ck(|e| unsafe { rs2_start_processing_queue(sync, sync_q, e) })?;
        ck(|e| unsafe { rs2_start(depth_s, to_syncer, sync, e) })?;
        ck(|e| unsafe { rs2_start(color_s, to_syncer, sync, e) })?;
        let mut s = SdkAligner {
            di: *di,
            ci: *ci,
            unit,
            depth_s,
            color_s,
            dprof,
            cprof,
            sync_q,
            align,
            align_q,
            color: vec![0; (ci.width * ci.height * 3) as usize],
            n: 0,
        };
        // syncer 가 깊이·컬러 짝을 내기 시작할 때까지 빈 장면을 넣음(처음 수십 장은 짝 없이 나옴)
        let z0 = vec![0u16; (di.width * di.height) as usize];
        let mut o0 = vec![0u16; (ci.width * ci.height) as usize];
        let mut ok = 0;
        for _ in 0..200 {
            if ok >= 3 {
                break;
            }
            ok = if s.process(&z0, &mut o0)?.is_some() { ok + 1 } else { 0 };
        }
        if ok < 3 {
            return Err(Error::Other("software_device: 깊이·컬러 짝이 나오지 않음".into()));
        }
        Ok(s)
    }

    fn push(&self, sensor: P, prof: P, data: &[u8], stride: i32, bpp: i32, ts: f64, unit: f32) -> Result<(), Error> {
        let buf = unsafe { malloc(data.len()) };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf as *mut u8, data.len()) };
        let f = SoftwareVideoFrame {
            pixels: buf,
            deleter: Some(free),
            stride,
            bpp,
            timestamp: ts,
            domain: TIMESTAMP_DOMAIN_HARDWARE_CLOCK,
            frame_number: self.n,
            profile: prof,
            depth_units: unit,
        };
        ck(|e| unsafe { rs2_software_sensor_on_video_frame(sensor, f, e) })
    }

    /// depth → out(컬러 크기). 돌려주는 값 = align.process 시간 ms(프레임 할당·복사 포함한 SDK 처리부), 짝을 못 받으면 None
    pub fn process(&mut self, depth: &[u16], out: &mut [u16]) -> Result<Option<f64>, Error> {
        self.n += 1;
        let ts = 1000.0 + self.n as f64 * 33.333;
        let nd = (self.di.width * self.di.height) as usize;
        let db: Vec<u8> = depth[..nd].iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.push(self.depth_s, self.dprof, &db, self.di.width * 2, 2, ts, self.unit)?;
        self.push(self.color_s, self.cprof, &self.color, self.ci.width * 3, 3, ts, 0.0)?;
        let mut fs: P = std::ptr::null_mut();
        let mut have = false;
        for _ in 0..10 {
            release(&[fs]);
            fs = std::ptr::null_mut();
            if ck(|e| unsafe { rs2_try_wait_for_frame(self.sync_q, 200, &mut fs, e) })? == 0 {
                return Ok(None);
            }
            let (c, d) = split(fs)?;
            have = !c.is_null() && !d.is_null();
            release(&[c, d]);
            if have {
                break;
            }
        }
        if !have {
            release(&[fs]);
            return Ok(None);
        }
        let t0 = crate::common::now_ms();
        let al = align_process(self.align, self.align_q, fs);
        let ms = crate::common::now_ms() - t0;
        release(&[fs]);
        let al = al?;
        let (ac, ad) = split(al)?;
        let mut ok = false;
        if !ad.is_null() {
            let w = ck(|e| unsafe { rs2_get_frame_width(ad, e) })?;
            let h = ck(|e| unsafe { rs2_get_frame_height(ad, e) })?;
            if w == self.ci.width && h == self.ci.height {
                let p = ck(|e| unsafe { rs2_get_frame_data(ad, e) })? as *const u16;
                let n = (w * h) as usize;
                out[..n].copy_from_slice(unsafe { std::slice::from_raw_parts(p, n) });
                ok = true;
            }
        }
        release(&[ac, ad, al]);
        Ok(if ok { Some(ms) } else { None })
    }
}

impl Drop for SdkAligner {
    fn drop(&mut self) {
        let _ = ck(|e| unsafe { rs2_stop(self.depth_s, e) });
        let _ = ck(|e| unsafe { rs2_stop(self.color_s, e) });
        let _ = ck(|e| unsafe { rs2_close(self.depth_s, e) });
        let _ = ck(|e| unsafe { rs2_close(self.color_s, e) });
    }
}
