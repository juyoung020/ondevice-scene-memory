//! D435 입력(`rsdriver` 기능): librealsense2 없이 V4L2·UVC 로 직접(rsdriver 크레이트). 장면 모양은 RealSenseSource 와 같다
//! (컬러 RGB8, 컬러 격자에 정렬한 깊이, 보정값 = 컬러 카메라).
//! 시각: rsdriver 는 컬러 장치 시각에 첫 장면에서 한 번 정한 시스템 시계 차이를 더한다(librealsense 의 global time 처럼
//! 장치·시스템 시계 차이를 계속 따라가지 않음). 장면끼리의 간격은 장치 시각 그대로다.
use super::{optical_rotation, split_depth, FrameSource};
use crate::{Camera, Frame};
use rsdriver::camera::{Camera as RsCamera, Config, Output};
use std::time::Instant;

pub struct RsDriverSource {
    cam: RsCamera,
    out: Output,
    id: i64,
    read_ms: f64,
    desc: String,
}

impl RsDriverSource {
    pub fn new(w: u32, h: u32, fps: u32, serial: &str, exposure_ms: f64, gain: i32) -> Result<RsDriverSource, String> {
        let cam = RsCamera::open(Config { width: w, height: h, fps, serial: serial.to_string(), exposure_ms, gain, ..Default::default() })?;
        let desc = cam.describe();
        crate::bufpool::enable();
        Ok(RsDriverSource { cam, out: Output { want_gray: true, ..Default::default() }, id: 0, read_ms: 0.0, desc })
    }
}

impl FrameSource for RsDriverSource {
    fn next(&mut self) -> Result<Option<Frame>, String> {
        let c0 = Instant::now();
        // 받는 쪽이 다 쓰고 돌려준 버퍼가 있으면 그것에 채움(길이가 맞아 0 으로 채우지 않음, 화소는 모두 덮어씀)
        let n = self.cam.cfg.width as usize * self.cam.cfg.height as usize;
        if self.out.rgb.capacity() == 0 {
            self.out.rgb = crate::bufpool::take_rgb(n * 3);
        }
        if self.out.aligned.capacity() == 0 {
            self.out.aligned = crate::bufpool::take_depth(n);
        }
        if self.out.gray.capacity() == 0 {
            self.out.gray = crate::bufpool::take_gray(n);
        }
        let info = self.cam.next_into(&mut self.out)?;
        // 장치 버퍼에서 한 번 채운 컬러·깊이를 그대로 장면으로 넘김(다음 장면은 돌려받은 버퍼나 새 버퍼에: 앞 장면은 memmap 이 들고 있음)
        let color = std::mem::take(&mut self.out.rgb);
        let depth_raw = std::mem::take(&mut self.out.aligned);
        let k = &self.cam.calib.color;
        let dpm = self.cam.depth_per_m();
        let gray = std::mem::take(&mut self.out.gray); // YUYV→RGB 와 같은 훑기에서 채운 회색조(to_gray(&color, true) 와 같은 값)
        let (depth_mm, depth_raw) = split_depth(depth_raw, dpm); // 1000/m 이면 복사 없이 한 버퍼
        self.id += 1;
        self.read_ms = c0.elapsed().as_secs_f64() * 1e3;
        Ok(Some(Frame {
            id: self.id,
            t: info.t_ms * 1e-3,
            gray,
            color,
            color_is_rgb: true,
            depth_mm,
            depth_raw,
            depth_per_m: dpm,
            cam: Camera { fx: k.fx as f64, fy: k.fy as f64, cx: k.ppx as f64, cy: k.ppy as f64, width: self.cam.cfg.width, height: self.cam.cfg.height, local: optical_rotation() },
        }))
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn describe(&self) -> String {
        self.desc.clone()
    }
}
