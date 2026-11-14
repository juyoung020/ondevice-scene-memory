//! UVC 메타데이터(V4L2_META_FMT_UVC 노드의 struct uvc_meta_buf) 해석.
//! 버퍼: u64 시스템 시각(ns), u16 SOF, 그 뒤 UVC 페이로드 머리(길이 u8, 정보 u8, PTS u32(µs), SCR 6) + D400 확장 메타데이터.
//! D400 확장 배치는 librealsense v2.59.1 `src/metadata.h`(md_capture_timing → md_capture_stats → md_rgb_control).

pub const MD_CAPTURE_TIMING: u32 = 0x8000_0001;
pub const MD_CAPTURE_STATS: u32 = 0x0000_0003;
pub const MD_RGB_CONTROL: u32 = 0x8000_0005;
pub const MD_DEPTH_CONTROL: u32 = 0x8000_0000;

/// 장면 하나의 메타데이터 중 우리가 쓰는 것
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameMeta {
    /// 메타 버퍼의 커널 시각(CLOCK_MONOTONIC ns)
    pub host_ns: u64,
    /// RS2_FRAME_METADATA_FRAME_TIMESTAMP: UVC 머리 PTS(µs, 32 비트)
    pub frame_ts_us: Option<u32>,
    pub frame_counter: Option<u32>,
    pub sensor_ts_us: Option<u32>,
    /// RS2_FRAME_METADATA_ACTUAL_EXPOSURE (컬러: md_rgb_control.manual_exp, 0.1 ms 단위)
    pub actual_exposure: Option<u32>,
    /// RS2_FRAME_METADATA_GAIN_LEVEL
    pub gain: Option<u32>,
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// uvc_meta_buf 전체(커널이 준 바이트)를 해석
pub fn parse_meta_buf(b: &[u8]) -> FrameMeta {
    let mut m = FrameMeta::default();
    if b.len() < 10 {
        return m;
    }
    m.host_ns = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let hdr = &b[10..];
    let len = hdr.first().copied().unwrap_or(0) as usize;
    let hdr = &hdr[..len.min(hdr.len())];
    parse_payload(hdr, &mut m);
    m
}

/// UVC 페이로드 머리(+ D400 확장)를 해석. metadata_raw 와 같은 배치
pub fn parse_payload(h: &[u8], m: &mut FrameMeta) {
    // 머리 12 바이트(정보 비트 PTS 있음 0x04)
    if h.len() >= 6 && h[1] & 0x04 != 0 {
        m.frame_ts_us = u32_at(h, 2);
    }
    if h.len() < 12 {
        return;
    }
    // 확장: librealsense 처럼 고정 위치(md_rgb_normal_mode / md_depth_y_normal_mode: 시간 12, 통계 52, 제어 132)에서
    // md_header 형식 번호가 맞는 것만 읽음
    for o in [12usize, 132] {
        let Some(ty) = u32_at(h, o) else { continue };
        let body = o + 8;
        match ty {
            MD_CAPTURE_TIMING => {
                // version, flags, frame_counter, sensor_timestamp ...
                if let (Some(fl), Some(fc), Some(st)) = (u32_at(h, body + 4), u32_at(h, body + 8), u32_at(h, body + 12)) {
                    if fl & 1 != 0 {
                        m.frame_counter = Some(fc);
                    }
                    if fl & 2 != 0 {
                        m.sensor_ts_us = Some(st);
                    }
                }
            }
            MD_RGB_CONTROL => {
                // version, flags, brightness, contrast, saturation, sharpness, ae_mode, awb_temp, gain, backlight, gamma, hue, manual_exp
                if let Some(fl) = u32_at(h, body + 4) {
                    if fl & (1 << 6) != 0 {
                        m.gain = u32_at(h, body + 8 + 6 * 4);
                    }
                    if fl & (1 << 10) != 0 {
                        m.actual_exposure = u32_at(h, body + 8 + 10 * 4);
                    }
                }
            }
            _ => {}
        }
    }
}

/// 32 비트 µs 시각을 감김 없이 이어 붙임
#[derive(Default, Clone, Copy, Debug)]
pub struct Unwrap {
    last: Option<u32>,
    high: u64,
}

impl Unwrap {
    pub fn push(&mut self, v: u32) -> u64 {
        if let Some(l) = self.last {
            if v < l && l - v > 0x8000_0000 {
                self.high += 1 << 32;
            }
        }
        self.last = Some(v);
        self.high + v as u64
    }
}
