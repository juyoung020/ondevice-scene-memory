//! D435 입력(`rsdriver` 기능): librealsense2 없이 V4L2·UVC 로 직접(rsdriver 크레이트).
//! 장면 = 컬러 RGB8 · 컬러 격자에 정렬한 깊이(보정값 = 컬러 카메라) · 적외선 좌우(Y8I, 깊이와 같은 노출).
//! 시각: rsdriver 는 컬러 장치 시각에 첫 장면에서 한 번 정한 시스템 시계 차이를 더한다. 적외선 시각은 깊이 장치 시각에 같은 차이를 더한 값.
use super::{split_depth, FrameSource};
use crate::{Camera, Frame, Ir, StereoCalib};
use rsdriver::camera::{Camera as RsCamera, Config, Output};
use std::time::Instant;

pub struct RsDriverSource {
    cam: RsCamera,
    out: Output,
    id: i64,
    read_ms: f64,
    desc: String,
}

/// D435 열기 설정
pub struct RsOpen<'a> {
    pub w: u32,
    pub h: u32,
    pub color: (u32, u32),
    pub fps: u32,
    pub serial: &'a str,
    pub exposure_ms: f64,
    pub gain: i32,
    pub ae_cap_ms: f64,
    pub profile: &'a str,
    pub emitter: bool,
}

impl RsDriverSource {
    pub fn new(o: RsOpen) -> Result<RsDriverSource, String> {
        let profile = rsdriver::camera::Profile::parse(o.profile)?;
        let cfg = Config { width: o.w, height: o.h, color_width: o.color.0, color_height: o.color.1, fps: o.fps, serial: o.serial.to_string(), exposure_ms: o.exposure_ms, gain: o.gain, ae_cap_ms: o.ae_cap_ms, profile, emitter: o.emitter, ir: true, ..Default::default() };
        let cam = RsCamera::open(cfg)?;
        let desc = format!("{}, 적외선 좌우(레이저 {})", cam.describe(), if o.emitter { "켬" } else { "끔" });
        crate::bufpool::enable();
        Ok(RsDriverSource { cam, out: Output::default(), id: 0, read_ms: 0.0, desc })
    }

    /// 버퍼를 맞추고 한 묶음(block = 묶음이 올 때까지 기다림, 아니면 준비된 묶음만)
    fn read(&mut self, block: bool) -> Result<Option<Frame>, String> {
        let c0 = Instant::now();
        // 받는 쪽이 다 쓰고 돌려준 버퍼가 있으면 그것에 채움(길이가 맞아 0 으로 채우지 않음, 화소는 모두 덮어씀)
        let (cw, ch) = self.cam.cfg.color_size();
        let n = cw as usize * ch as usize;
        if self.out.rgb.capacity() == 0 {
            self.out.rgb = crate::bufpool::take_rgb(n * 3);
        }
        if self.out.aligned.capacity() == 0 {
            self.out.aligned = crate::bufpool::take_depth(n);
        }
        let info = if block {
            self.cam.next_into(&mut self.out)?
        } else {
            match self.cam.try_next_into(&mut self.out)? {
                Some(i) => i,
                None => return Ok(None),
            }
        };
        // 장치 버퍼에서 한 번 채운 컬러·깊이를 그대로 장면으로 넘김(다음 장면은 돌려받은 버퍼나 새 버퍼에)
        let color = std::mem::take(&mut self.out.rgb);
        let depth_raw = std::mem::take(&mut self.out.aligned);
        let k = &self.cam.calib.color;
        let dpm = self.cam.depth_per_m();
        let (depth_mm, depth_raw) = split_depth(depth_raw, dpm); // 1000/m 이면 복사 없이 한 버퍼
        let ir = Ir {
            left: std::mem::take(&mut self.out.ir_left),
            right: std::mem::take(&mut self.out.ir_right),
            w: self.cam.cfg.width,
            h: self.cam.cfg.height,
            t: (info.t_ms - info.color_hw_ms + info.depth_hw_ms) * 1e-3,
        };
        self.id += 1;
        self.read_ms = c0.elapsed().as_secs_f64() * 1e3;
        Ok(Some(Frame {
            id: self.id,
            t: info.t_ms * 1e-3,
            color,
            color_is_rgb: true,
            depth_mm,
            depth_raw,
            depth_per_m: dpm,
            cam: Camera { fx: k.fx as f64, fy: k.fy as f64, cx: k.ppx as f64, cy: k.ppy as f64, width: cw, height: ch },
            ir: Some(ir),
        }))
    }
}

impl FrameSource for RsDriverSource {
    fn next(&mut self) -> Result<Option<Frame>, String> {
        self.read(true)
    }
    fn try_next(&mut self) -> Result<Option<Option<Frame>>, String> {
        Ok(self.read(false)?.map(Some))
    }
    fn fds(&self) -> Vec<i32> {
        self.cam.fds()
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn describe(&self) -> String {
        self.desc.clone()
    }
    fn stereo(&self) -> Option<StereoCalib> {
        let c = &self.cam.calib;
        let (r, t) = (&c.depth_to_color.rotation, &c.depth_to_color.translation);
        // rs2_extrinsics: 회전은 열 우선 → 3×4 행 우선
        let m = |i: usize, j: usize| r[j * 3 + i] as f64;
        Some(StereoCalib {
            k: [c.depth.fx as f64, c.depth.fy as f64, c.depth.ppx as f64, c.depth.ppy as f64],
            w: self.cam.cfg.width,
            h: self.cam.cfg.height,
            baseline: c.baseline_m as f64,
            color_from_ir: [m(0, 0), m(0, 1), m(0, 2), t[0] as f64, m(1, 0), m(1, 1), m(1, 2), t[1] as f64, m(2, 0), m(2, 1), m(2, 2), t[2] as f64],
        })
    }
}
