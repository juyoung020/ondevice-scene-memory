//! RealSense 카메라(D435·D455 등): 컬러 RGB8 + 깊이 Z16, 깊이를 컬러에 맞춤("rs2" = librealsense 의 align). IMU 는 켜지 않는다.
//! 원본: bridge/src/realsense_source.cpp · depth_align.cpp. librealsense2 C API(rs.h)를 직접 부른다(선언은 tools::rs2, librealsense2.so 링크).
use super::{optical_rotation, to_gray, FrameSource};
use crate::{Camera, Frame};
use std::ffi::{CStr, CString};
use std::time::Instant;

// librealsense2 C API 선언·상수는 tools::rs2 하나(녹화·정렬 확인 도구와 같은 것)
use tools::fast_align::Intrinsics;
use tools::rs2::*;

/// rs2 함수를 부르고 오류가 있으면 문장으로 돌려줌
fn ck<T>(f: impl FnOnce(*mut P) -> T) -> Result<T, String> {
    let mut e: P = std::ptr::null_mut();
    let v = f(&mut e);
    if e.is_null() {
        Ok(v)
    } else {
        let m = unsafe { CStr::from_ptr(rs2_get_error_message(e)).to_string_lossy().into_owned() };
        unsafe { rs2_free_error(e) };
        Err(format!("RealSense: {m}"))
    }
}

fn info(dev: P, i: i32) -> String {
    ck(|e| unsafe { rs2_get_device_info(dev, i, e) }).ok().filter(|p| !p.is_null()).map_or(String::new(), |p| unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
}

/// 묶음 장면에서 컬러(스트림 COLOR)·깊이(깊이 장면) 하나씩. 돌려준 장면은 rs2_release_frame 으로 놓아야 함
fn split(fs: P) -> Result<(P, P), String> {
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

pub struct RealSenseSource {
    pipe: P,
    align: P,
    queue: P,
    k: Intrinsics,
    depth_per_m: f32,
    read_ms: f64,
    wait_ms: f64,
    align_ms: f64,
    desc: String,
    id: i64,
}

// rs2 핸들은 한 스레드(읽기 스레드)에서만 쓴다
unsafe impl Send for RealSenseSource {}

impl RealSenseSource {
    /// serial: 빈 문자열이면 처음 찾은 카메라. exposure_ms > 0 이면 컬러 노출 고정(gain >= 0 이면 감도도)
    pub fn new(w: u32, h: u32, fps: u32, aligner: &str, serial: &str, exposure_ms: f64, gain: i32) -> Result<RealSenseSource, String> {
        if aligner != "rs2" {
            return Err(format!("모르는 정렬 방법: {aligner}"));
        }
        unsafe {
            let ctx = ck(|e| rs2_create_context(API_VERSION, e))?;
            let pipe = ck(|e| rs2_create_pipeline(ctx, e))?;
            let cfg = ck(|e| rs2_create_config(e))?;
            if !serial.is_empty() {
                let s = CString::new(serial).unwrap();
                ck(|e| rs2_config_enable_device(cfg, s.as_ptr(), e))?;
            }
            ck(|e| rs2_config_enable_stream(cfg, STREAM_COLOR, -1, w as i32, h as i32, FORMAT_RGB8, fps as i32, e))?;
            ck(|e| rs2_config_enable_stream(cfg, STREAM_DEPTH, -1, w as i32, h as i32, FORMAT_Z16, fps as i32, e))?;
            let prof = ck(|e| rs2_pipeline_start_with_config(pipe, cfg, e))?;
            let dev = ck(|e| rs2_pipeline_profile_get_device(prof, e))?;
            let sensors = ck(|e| rs2_query_sensors(dev, e))?;
            let mut depth_per_m = 1000.0f32;
            for i in 0..ck(|e| rs2_get_sensors_count(sensors, e))? {
                let s = ck(|e| rs2_create_sensor(sensors, i, e))?;
                let is_color = ck(|e| rs2_is_sensor_extendable_to(s, EXT_COLOR_SENSOR, e))? != 0;
                let sup = |o| ck(|e| rs2_supports_option(s, o, e)).unwrap_or(0) != 0;
                let set = |o, v: f32| ck(|e| rs2_set_option(s, o, v, e));
                // 컬러 자동 노출이 어두울 때 장면 수를 낮추지 않게(위치 추적은 고른 30 장/초가 낫다)
                if is_color && sup(OPT_AUTO_EXPOSURE_PRIORITY) {
                    set(OPT_AUTO_EXPOSURE_PRIORITY, 0.0)?;
                }
                // 컬러 노출을 짧게 고정(--exposure ms): 움직일 때 번지는 것을 막는다. 밝기는 감도(--gain)로 채움
                if is_color && exposure_ms > 0.0 {
                    set(OPT_ENABLE_AUTO_EXPOSURE, 0.0)?;
                    set(OPT_EXPOSURE, (exposure_ms * 10.0) as f32)?; // 단위 0.1 ms
                    if gain >= 0 {
                        set(OPT_GAIN, gain as f32)?;
                    }
                } else if is_color {
                    set(OPT_ENABLE_AUTO_EXPOSURE, 1.0)?;
                }
                if ck(|e| rs2_is_sensor_extendable_to(s, EXT_DEPTH_SENSOR, e))? != 0 {
                    depth_per_m = 1.0 / ck(|e| rs2_get_depth_scale(s, e))?;
                    if sup(OPT_EMITTER_ENABLED) {
                        set(OPT_EMITTER_ENABLED, 1.0)?;
                    }
                }
            }
            // 컬러 보정값
            let streams = ck(|e| rs2_pipeline_profile_get_streams(prof, e))?;
            let mut k = Intrinsics::default();
            for i in 0..ck(|e| rs2_get_stream_profiles_count(streams, e))? {
                let p = ck(|e| rs2_get_stream_profile(streams, i, e))?;
                let (mut st, mut fm, mut ix, mut uid, mut f) = (0, 0, 0, 0, 0);
                ck(|e| rs2_get_stream_profile_data(p, &mut st, &mut fm, &mut ix, &mut uid, &mut f, e))?;
                if st == STREAM_COLOR {
                    ck(|e| rs2_get_video_stream_intrinsics(p, &mut k, e))?;
                }
            }
            let align = ck(|e| rs2_create_align(STREAM_COLOR, e))?;
            let queue = ck(|e| rs2_create_frame_queue(1, e))?;
            ck(|e| rs2_start_processing_queue(align, queue, e))?;
            let dist = CStr::from_ptr(rs2_distortion_to_string(k.model)).to_string_lossy();
            let mut desc = format!(
                "RealSense {} {} 펌웨어 {}, {w}x{h}@{fps}, 깊이 {depth_per_m}/m, 컬러 보정 fx {} fy {} cx {} cy {} 왜곡 모델 {dist}, 정렬 rs2",
                info(dev, INFO_NAME),
                info(dev, INFO_SERIAL),
                info(dev, INFO_FIRMWARE),
                k.fx,
                k.fy,
                k.ppx,
                k.ppy
            );
            // 자동 노출이 자리 잡을 때까지 첫 장면들은 버린다(어두운 첫 장면이 위치 추적의 기준이 되면 바로 놓침)
            let mut last: P = std::ptr::null_mut();
            for _ in 0..30 {
                if !last.is_null() {
                    rs2_release_frame(last);
                }
                last = ck(|e| rs2_pipeline_wait_for_frames(pipe, 5000, e))?;
            }
            let (c, d) = split(last)?;
            if !c.is_null() {
                if ck(|e| rs2_supports_frame_metadata(c, META_ACTUAL_EXPOSURE, e))? != 0 {
                    desc += &format!(", 컬러 노출 {} ms", ck(|e| rs2_get_frame_metadata(c, META_ACTUAL_EXPOSURE, e))? as f64 / 10.0);
                }
                if ck(|e| rs2_supports_frame_metadata(c, META_GAIN_LEVEL, e))? != 0 {
                    desc += &format!(" 감도 {}", ck(|e| rs2_get_frame_metadata(c, META_GAIN_LEVEL, e))?);
                }
                desc += if exposure_ms > 0.0 { "(고정)" } else { "(자동)" };
                rs2_release_frame(c);
            }
            if !d.is_null() {
                rs2_release_frame(d);
            }
            rs2_release_frame(last);
            Ok(RealSenseSource { pipe, align, queue, k, depth_per_m, read_ms: 0.0, wait_ms: 0.0, align_ms: 0.0, desc, id: 0 })
        }
    }
    pub fn last_wait_ms(&self) -> f64 {
        self.wait_ms
    }
    pub fn last_align_ms(&self) -> f64 {
        self.align_ms
    }
}

impl Drop for RealSenseSource {
    fn drop(&mut self) {
        let _ = ck(|e| unsafe { rs2_pipeline_stop(self.pipe, e) });
    }
}

impl FrameSource for RealSenseSource {
    fn describe(&self) -> String {
        self.desc.clone()
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn next(&mut self) -> Result<Option<Frame>, String> {
        loop {
            let c0 = Instant::now();
            let fs = ck(|e| unsafe { rs2_pipeline_wait_for_frames(self.pipe, 5000, e) })?;
            let c1 = Instant::now();
            let (color, depth) = split(fs)?;
            if color.is_null() || depth.is_null() {
                unsafe {
                    for f in [color, depth, fs] {
                        if !f.is_null() {
                            rs2_release_frame(f);
                        }
                    }
                }
                continue;
            }
            unsafe { rs2_release_frame(depth) };
            // 깊이를 컬러 격자로(librealsense align). process_frame 이 묶음 하나를 가져가므로 참조를 하나 더함
            ck(|e| unsafe { rs2_frame_add_ref(fs, e) })?;
            ck(|e| unsafe { rs2_process_frame(self.align, fs, e) })?;
            let al = ck(|e| unsafe { rs2_wait_for_frame(self.queue, 5000, e) })?;
            let (ac, ad) = split(al)?;
            let read_plane = |f: P, bpp: usize| -> Result<(usize, usize, Vec<u8>), String> {
                let (w, h, st) = unsafe { (ck(|e| rs2_get_frame_width(f, e))? as usize, ck(|e| rs2_get_frame_height(f, e))? as usize, ck(|e| rs2_get_frame_stride_in_bytes(f, e))? as usize) };
                let d = ck(|e| unsafe { rs2_get_frame_data(f, e) })?;
                let mut out = Vec::with_capacity(w * h * bpp);
                for r in 0..h {
                    out.extend_from_slice(unsafe { std::slice::from_raw_parts(d.add(r * st), w * bpp) });
                }
                Ok((w, h, out))
            };
            // 깊이는 장치 버퍼에서 u16 벡터로 한 번만 복사(리틀 엔디언 기계라 바이트 그대로)
            let depth_raw: Vec<u16> = {
                let (w, h, st) = unsafe { (ck(|e| rs2_get_frame_width(ad, e))? as usize, ck(|e| rs2_get_frame_height(ad, e))? as usize, ck(|e| rs2_get_frame_stride_in_bytes(ad, e))? as usize) };
                let d = ck(|e| unsafe { rs2_get_frame_data(ad, e) })? as *const u8;
                let mut out = vec![0u16; w * h];
                for r in 0..h {
                    unsafe { std::ptr::copy_nonoverlapping(d.add(r * st), (out.as_mut_ptr() as *mut u8).add(r * w * 2), w * 2) };
                }
                out
            };
            let c2 = Instant::now();
            // 장면마다 새 버퍼에 담는다: 쓰기 큐·지도가 들고 있는 앞 장면과 메모리를 나누지 않게
            let (w, h, rgb) = read_plane(color, 3)?;
            let t = ck(|e| unsafe { rs2_get_frame_timestamp(color, e) })? * 1e-3;
            unsafe {
                for f in [ac, ad, al, color, fs] {
                    if !f.is_null() {
                        rs2_release_frame(f);
                    }
                }
            }
            let gray = to_gray(&rgb, true);
            let (depth_mm, depth_raw) = super::split_depth(depth_raw, self.depth_per_m); // 1000/m 이면 복사 없이 한 버퍼
            self.id += 1;
            let k = self.k;
            self.wait_ms = (c1 - c0).as_secs_f64() * 1e3;
            self.align_ms = (c2 - c1).as_secs_f64() * 1e3;
            self.read_ms = c1.elapsed().as_secs_f64() * 1e3;
            return Ok(Some(Frame {
                id: self.id,
                t,
                gray,
                color: rgb,
                color_is_rgb: true,
                depth_mm,
                depth_raw,
                depth_per_m: self.depth_per_m,
                cam: Camera { fx: k.fx as f64, fy: k.fy as f64, cx: k.ppx as f64, cy: k.ppy as f64, width: w as u32, height: h as u32, local: optical_rotation() },
            }));
        }
    }
}
