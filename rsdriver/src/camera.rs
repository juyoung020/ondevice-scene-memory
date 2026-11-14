//! D435 장치 찾기·열기·흐름·짝·정렬(리눅스 전용).
use crate::calib::{self, Extrinsics, Gvd, Intrinsics};
use crate::format::{self, Yuy2Rule};
use crate::hwm;
use crate::meta::{self, FrameMeta, Unwrap};
use crate::sync::{self, Decision};
use crate::v4l2::{self, Buffer, Node, Stream};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tools::fast_align::{DepthAligner, Options};

pub const VID_INTEL: u16 = 0x8086;
/// D435
pub const PID_D435: u16 = 0x0b07;
/// 깊이 XU: unit 3
pub const DEPTH_XU_UNIT: u8 = 3;
pub const XU_HWMONITOR: u8 = 1;
pub const XU_EMITTER_ENABLED: u8 = 2;

/// sysfs 로 찾은 카메라 하나의 노드들
#[derive(Clone, Debug, Default)]
pub struct Found {
    pub usb_path: String,
    pub vid: u16,
    pub pid: u16,
    pub usb_serial: String,
    /// USB 속도(Mb/s, sysfs speed)
    pub speed: String,
    pub depth: String,
    pub depth_meta: Option<String>,
    pub color: String,
    pub color_meta: Option<String>,
}

fn rd(p: &Path) -> String {
    fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_default()
}

/// /sys/class/video4linux/video* 를 USB 장치별로 묶어 D435(8086:0b07)를 찾음. 인터페이스 0 = 깊이, 3 = 컬러, 같은 인터페이스의 META 노드 = 메타데이터
pub fn discover() -> Vec<Found> {
    let mut out: Vec<Found> = Vec::new();
    let Ok(rdir) = fs::read_dir("/sys/class/video4linux") else { return out };
    let mut names: Vec<String> = rdir.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("video")).collect();
    names.sort_by_key(|n| n[5..].parse::<u32>().unwrap_or(u32::MAX));
    for n in names {
        let Ok(intf) = fs::canonicalize(format!("/sys/class/video4linux/{n}/device")) else { continue };
        let Some(usb) = intf.parent() else { continue };
        let vid = u16::from_str_radix(&rd(&usb.join("idVendor")), 16).unwrap_or(0);
        let pid = u16::from_str_radix(&rd(&usb.join("idProduct")), 16).unwrap_or(0);
        if vid != VID_INTEL || pid != PID_D435 {
            continue;
        }
        let ifnum = u8::from_str_radix(&rd(&intf.join("bInterfaceNumber")), 16).unwrap_or(255);
        let dev = format!("/dev/{n}");
        let Ok(node) = Node::open(&dev) else { continue };
        let caps = node.device_caps().unwrap_or(0);
        let up = usb.to_string_lossy().into_owned();
        let i = match out.iter().position(|f| f.usb_path == up) {
            Some(i) => i,
            None => {
                out.push(Found { usb_path: up, vid, pid, usb_serial: rd(&usb.join("serial")), speed: rd(&usb.join("speed")), ..Default::default() });
                out.len() - 1
            }
        };
        let f = &mut out[i];
        let video = caps & v4l2::CAP_VIDEO_CAPTURE != 0;
        let metac = caps & v4l2::CAP_META_CAPTURE != 0;
        match (ifnum, video, metac) {
            (0, true, _) if f.depth.is_empty() => f.depth = dev,
            (0, _, true) if f.depth_meta.is_none() => f.depth_meta = Some(dev),
            (3, true, _) if f.color.is_empty() => f.color = dev,
            (3, _, true) if f.color_meta.is_none() => f.color_meta = Some(dev),
            _ => {}
        }
    }
    out.retain(|f| !f.depth.is_empty() && !f.color.is_empty());
    out
}

/// 깊이 노드의 XU 로 HW 모니터 명령
pub fn hwm_send(node: &Node, op: u32, p: [u32; 4]) -> Result<Vec<u8>, String> {
    let mut buf = hwm::build(op, p, &[]);
    let mut last = String::new();
    for _ in 0..5 {
        match node.xu(DEPTH_XU_UNIT, XU_HWMONITOR, v4l2::UVC_SET_CUR, &mut buf) {
            Ok(()) => {
                let mut resp = vec![0u8; hwm::BUFFER_SIZE];
                match node.xu(DEPTH_XU_UNIT, XU_HWMONITOR, v4l2::UVC_GET_CUR, &mut resp) {
                    Ok(()) => return hwm::parse(op, &resp),
                    Err(e) => last = format!("HWM GET_CUR: {e}"),
                }
            }
            Err(e) => last = format!("HWM SET_CUR: {e}"),
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Err(last)
}

/// HW 모니터로 읽은 원래 바이트(시험·비교용으로 저장 가능)
#[derive(Clone, Debug, Default)]
pub struct RawCalib {
    pub gvd: Vec<u8>,
    pub coeff: Vec<u8>,
    pub rgb: Vec<u8>,
    pub new_table: Vec<u8>,
    pub adv_mode: bool,
    pub depth_table: Vec<u8>,
}

pub fn read_raw_calib(node: &Node) -> Result<RawCalib, String> {
    let gvd = hwm_send(node, hwm::GVD, [0; 4])?;
    let g = calib::parse_gvd(&gvd)?;
    let coeff = hwm_send(node, hwm::GETINTCAL, [hwm::COEFFICIENTS_TABLE_ID, 0, 0, 0])?;
    let rgb = hwm_send(node, hwm::GETINTCAL, [hwm::RGB_CALIBRATION_ID, 0, 0, 0])?;
    let new_table = if calib::fw_at_least(g.fw, [5, 11, 9, 5]) { hwm_send(node, hwm::RECPARAMSGET, [0; 4])? } else { Vec::new() };
    let adv = hwm_send(node, hwm::UAMG, [0; 4])?;
    let adv_mode = adv.first().copied().unwrap_or(0) != 0;
    let depth_table = if adv_mode && calib::fw_at_least(g.fw, [5, 6, 3, 0]) { hwm_send(node, hwm::GET_ADV, [hwm::ET_DEPTH_TABLE_CONTROL, 0, 0, 0])? } else { Vec::new() };
    Ok(RawCalib { gvd, coeff, rgb, new_table, adv_mode, depth_table })
}

/// 원래 바이트 → 쓰는 값(librealsense 와 같은 계산)
#[derive(Clone, Debug)]
pub struct Calib {
    pub gvd: Gvd,
    pub depth: Intrinsics,
    pub color: Intrinsics,
    pub depth_to_color: Extrinsics,
    /// 깊이 단위(m)
    pub depth_scale: f32,
}

pub fn parse_calib(raw: &RawCalib, dw: u32, dh: u32, cw: u32, ch: u32) -> Result<Calib, String> {
    let depth_scale = if raw.depth_table.len() >= 4 { calib::depth_scale_from_units(u32::from_le_bytes(raw.depth_table[0..4].try_into().unwrap())) } else { 0.001f32 };
    Ok(Calib {
        gvd: calib::parse_gvd(&raw.gvd)?,
        depth: calib::depth_intrinsics(&raw.new_table, &raw.coeff, dw, dh)?,
        color: calib::color_intrinsics(&raw.rgb, cw, ch)?,
        depth_to_color: calib::depth_to_color(&raw.rgb)?,
        depth_scale,
    })
}

#[derive(Clone, Debug)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// 빈 문자열 = 처음 찾은 카메라
    pub serial: String,
    /// > 0 이면 컬러 노출 고정(ms), gain >= 0 이면 감도도
    pub exposure_ms: f64,
    pub gain: i32,
    pub emitter: bool,
    /// 처음 버리는 묶음 수
    pub warmup: u32,
    /// 정렬 스레드 수(fast_align)
    pub align_threads: i32,
    pub yuy2: Yuy2Rule,
}

impl Default for Config {
    fn default() -> Self {
        Config { width: 640, height: 480, fps: 30, serial: String::new(), exposure_ms: 0.0, gain: -1, emitter: true, warmup: 30, align_threads: 1, yuy2: Yuy2Rule::NATIVE }
    }
}

/// 한 묶음의 시각·메타데이터
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInfo {
    /// 컬러 장치 시각(ms, librealsense 의 하드웨어 시각과 같은 µs×0.001, 감김 풂)
    pub color_hw_ms: f64,
    pub depth_hw_ms: f64,
    /// 컬러 시각을 시스템 시계(ms, UNIX)로 옮긴 값: 첫 장면에서 정한 차이를 더함
    pub t_ms: f64,
    pub color_meta: FrameMeta,
    pub depth_meta: FrameMeta,
    pub color_seq: u32,
    pub depth_seq: u32,
    /// 이 묶음을 기다린 시간(ms)·정렬 시간(ms)
    pub wait_ms: f64,
    pub align_ms: f64,
}

/// 결과를 담을 버퍼(장면마다 다시 써도 됨). 크기는 처음에 맞춰짐
#[derive(Default)]
pub struct Output {
    pub rgb: Vec<u8>,
    /// 컬러 격자에 정렬한 깊이(카메라 단위)
    pub aligned: Vec<u16>,
    /// true 면 정렬 전 깊이도 채움
    pub want_raw: bool,
    pub depth_raw: Vec<u16>,
    /// true 면 컬러 원본 YUYV 도 채움(빽빽하게 w*2)
    pub want_yuyv: bool,
    pub yuyv: Vec<u8>,
}

struct Side {
    s: Stream,
    md: Option<Stream>,
    metas: VecDeque<(u32, FrameMeta)>,
    held: Option<Buffer>,
    stride: usize,
    unwrap: Unwrap,
    hw_ms: f64,
}

impl Side {
    fn new(node: Node, md: Option<String>, w: u32, h: u32, fourcc: u32, fps: u32) -> Result<Side, String> {
        let (stride, _) = node.set_video_format(w, h, fourcc).map_err(|e| format!("{}: 형식: {e}", node.path))?;
        node.set_fps(fps).map_err(|e| format!("{}: fps: {e}", node.path))?;
        let s = Stream::new(node, v4l2::BUF_TYPE_VIDEO_CAPTURE, 4).map_err(|e| format!("버퍼: {e}"))?;
        let md = match md {
            Some(p) => {
                let n = Node::open(&p).map_err(|e| format!("{p}: {e}"))?;
                n.set_meta_format().map_err(|e| format!("{p}: 메타 형식: {e}"))?;
                Some(Stream::new(n, v4l2::BUF_TYPE_META_CAPTURE, 8).map_err(|e| format!("{p}: 메타 버퍼: {e}"))?)
            }
            None => None,
        };
        Ok(Side { s, md, metas: VecDeque::new(), held: None, stride: stride as usize, unwrap: Unwrap::default(), hw_ms: 0.0 })
    }
    fn start(&mut self) -> Result<(), String> {
        if let Some(m) = &mut self.md {
            m.start().map_err(|e| format!("메타 STREAMON: {e}"))?;
        }
        self.s.start().map_err(|e| format!("STREAMON: {e}"))
    }
    fn drain_meta(&mut self) -> Result<(), String> {
        if let Some(m) = &self.md {
            while let Some(b) = m.dequeue().map_err(|e| format!("메타 DQBUF: {e}"))? {
                let fm = meta::parse_meta_buf(m.data(&b));
                self.metas.push_back((b.sequence, fm));
                if self.metas.len() > 16 {
                    self.metas.pop_front();
                }
                m.queue(b.index).map_err(|e| format!("메타 QBUF: {e}"))?;
            }
        }
        Ok(())
    }
    fn meta_of(&self, seq: u32) -> FrameMeta {
        self.metas.iter().rev().find(|(s, _)| *s == seq).map(|x| x.1).unwrap_or_default()
    }
    /// 버퍼 하나를 받아 들고 있게 함(이미 들고 있으면 그대로)
    fn take(&mut self) -> Result<bool, String> {
        if self.held.is_some() {
            return Ok(true);
        }
        self.drain_meta()?;
        if let Some(b) = self.s.dequeue().map_err(|e| format!("DQBUF: {e}"))? {
            if b.flags & 0x40 != 0 {
                // V4L2_BUF_FLAG_ERROR: 깨진 장면은 버림
                self.s.queue(b.index).map_err(|e| format!("QBUF: {e}"))?;
                return Ok(false);
            }
            self.drain_meta()?;
            let m = self.meta_of(b.sequence);
            self.hw_ms = match m.frame_ts_us {
                Some(us) => self.unwrap.push(us) as f64 * 0.001,
                None => b.timestamp.tv_sec as f64 * 1e3 + b.timestamp.tv_usec as f64 * 1e-3,
            };
            self.held = Some(b);
            return Ok(true);
        }
        Ok(false)
    }
    fn release(&mut self) -> Result<(), String> {
        if let Some(b) = self.held.take() {
            self.s.queue(b.index).map_err(|e| format!("QBUF: {e}"))?;
        }
        Ok(())
    }
}

pub struct Camera {
    pub found: Found,
    pub raw: RawCalib,
    pub calib: Calib,
    pub cfg: Config,
    ctl: Node,
    color: Side,
    depth: Side,
    aligner: DepthAligner,
    t_offset_ms: Option<f64>,
}

unsafe impl Send for Camera {}

fn sys_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64() * 1e3).unwrap_or(0.0)
}

impl Camera {
    /// 찾기 → 보정값 읽기 → 형식·fps → 흐름 시작 → 제어(컬러 노출 우선 끔, 노출·감도, 이미터) → 처음 warmup 묶음 버림
    pub fn open(cfg: Config) -> Result<Camera, String> {
        let list = discover();
        if list.is_empty() {
            return Err("RealSense D435(8086:0b07)를 찾지 못함(/sys/class/video4linux)".into());
        }
        let mut pick = None;
        for f in list {
            let n = Node::open(&f.depth).map_err(|e| format!("{}: {e}", f.depth))?;
            let raw = read_raw_calib(&n)?;
            let g = calib::parse_gvd(&raw.gvd)?;
            if cfg.serial.is_empty() || g.serial == cfg.serial || f.usb_serial == cfg.serial {
                pick = Some((f, raw, n));
                break;
            }
        }
        let Some((found, raw, ctl)) = pick else { return Err(format!("시리얼 {} 카메라가 없음", cfg.serial)) };
        let (w, h) = (cfg.width, cfg.height);
        let calib = parse_calib(&raw, w, h, w, h)?;
        // 흐름 노드는 따로 연다(제어용 fd 는 HWM·XU 에)
        let dnode = Node::open(&found.depth).map_err(|e| format!("{}: {e}", found.depth))?;
        let cnode = Node::open(&found.color).map_err(|e| format!("{}: {e}", found.color))?;
        let mut depth = Side::new(dnode, found.depth_meta.clone(), w, h, v4l2::FMT_Z16, cfg.fps)?;
        let mut color = Side::new(cnode, found.color_meta.clone(), w, h, v4l2::FMT_YUYV, cfg.fps)?;
        depth.start()?;
        color.start()?;
        // 제어: 컬러 PU 는 컬러 노드, 이미터는 깊이 XU
        let cc = &color.s.node;
        let _ = cc.set_ctrl(v4l2::CID_EXPOSURE_AUTO_PRIORITY, 0);
        if cfg.exposure_ms > 0.0 {
            cc.set_ctrl(v4l2::CID_EXPOSURE_AUTO, v4l2::EXPOSURE_MANUAL).map_err(|e| format!("자동 노출 끔: {e}"))?;
            cc.set_ctrl(v4l2::CID_EXPOSURE_ABSOLUTE, (cfg.exposure_ms * 10.0) as f32 as i32).map_err(|e| format!("노출: {e}"))?;
            if cfg.gain >= 0 {
                cc.set_ctrl(v4l2::CID_GAIN, cfg.gain).map_err(|e| format!("감도: {e}"))?;
            }
        } else {
            cc.set_ctrl(v4l2::CID_EXPOSURE_AUTO, v4l2::EXPOSURE_APERTURE_PRIORITY).map_err(|e| format!("자동 노출 켬: {e}"))?;
        }
        let mut em = [cfg.emitter as u8];
        let _ = ctl.xu(DEPTH_XU_UNIT, XU_EMITTER_ENABLED, v4l2::UVC_SET_CUR, &mut em);
        let opt = Options { threads: cfg.align_threads.max(1), ..Default::default() };
        let aligner = DepthAligner::new(&calib.depth, &calib.color, &calib.depth_to_color, calib.depth_scale, &opt)?;
        let mut cam = Camera { found, raw, calib, cfg, ctl, color, depth, aligner, t_offset_ms: None };
        let mut scratch = Output::default();
        for _ in 0..cam.cfg.warmup {
            cam.next_into(&mut scratch)?;
        }
        Ok(cam)
    }

    /// 깊이 단위 1 m 당 값(1 / depth_scale, f32)
    pub fn depth_per_m(&self) -> f32 {
        1.0f32 / self.calib.depth_scale
    }

    /// 색 XU·PU 조회(보드 시험용)
    pub fn control_node(&self) -> &Node {
        &self.ctl
    }
    pub fn color_node(&self) -> &Node {
        &self.color.s.node
    }

    /// 다음 묶음: 시각이 맞는 컬러·깊이 한 쌍을 mmap 버퍼에서 바로 out 으로(컬러 YUYV → RGB8, 깊이 → 컬러 격자 정렬)
    pub fn next_into(&mut self, out: &mut Output) -> Result<FrameInfo, String> {
        let t0 = Instant::now();
        let fps = self.cfg.fps as f64;
        loop {
            let c = self.color.take()?;
            let d = self.depth.take()?;
            if !(c && d) {
                if t0.elapsed() > Duration::from_secs(5) {
                    return Err("5 초 동안 장면이 오지 않음".into());
                }
                let mut fds = Vec::new();
                if !c {
                    fds.push(self.color.s.node.fd);
                    if let Some(m) = &self.color.md {
                        fds.push(m.node.fd);
                    }
                }
                if !d {
                    fds.push(self.depth.s.node.fd);
                    if let Some(m) = &self.depth.md {
                        fds.push(m.node.fd);
                    }
                }
                v4l2::poll(&fds, 100).map_err(|e| format!("poll: {e}"))?;
                continue;
            }
            match sync::decide(self.color.hw_ms, self.depth.hw_ms, fps) {
                Decision::DropColor => self.color.release()?,
                Decision::DropDepth => self.depth.release()?,
                Decision::Pair => break,
            }
        }
        let t1 = Instant::now();
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        let cb = self.color.held.unwrap();
        let db = self.depth.held.unwrap();
        let info_c = self.color.meta_of(cb.sequence);
        let info_d = self.depth.meta_of(db.sequence);
        {
            let cdata = self.color.s.data(&cb);
            let ddata = self.depth.s.data(&db);
            if cdata.len() < self.color.stride * (h - 1) + w * 2 || ddata.len() < self.depth.stride * (h - 1) + w * 2 {
                self.color.release()?;
                self.depth.release()?;
                return self.next_into(out);
            }
            out.rgb.resize(w * h * 3, 0);
            format::yuyv_to_rgb8(cdata, self.color.stride, w, &mut out.rgb, self.cfg.yuy2);
            if out.want_yuyv {
                out.yuyv.resize(w * h * 2, 0);
                for r in 0..h {
                    out.yuyv[r * w * 2..(r + 1) * w * 2].copy_from_slice(&cdata[r * self.color.stride..r * self.color.stride + w * 2]);
                }
            }
            out.aligned.resize(w * h, 0);
            let packed = self.depth.stride == w * 2 && (ddata.as_ptr() as usize) % 2 == 0;
            if packed && !out.want_raw {
                // mmap 깊이를 복사 없이 정렬 입력으로
                let d16 = unsafe { std::slice::from_raw_parts(ddata.as_ptr() as *const u16, w * h) };
                self.aligner.process(d16, &mut out.aligned);
            } else {
                out.depth_raw.resize(w * h, 0);
                format::z16_copy(ddata, self.depth.stride, w, &mut out.depth_raw);
                self.aligner.process(&out.depth_raw, &mut out.aligned);
            }
        }
        let t2 = Instant::now();
        let info = FrameInfo {
            color_hw_ms: self.color.hw_ms,
            depth_hw_ms: self.depth.hw_ms,
            t_ms: 0.0,
            color_meta: info_c,
            depth_meta: info_d,
            color_seq: cb.sequence,
            depth_seq: db.sequence,
            wait_ms: (t1 - t0).as_secs_f64() * 1e3,
            align_ms: (t2 - t1).as_secs_f64() * 1e3,
        };
        self.color.release()?;
        self.depth.release()?;
        let off = *self.t_offset_ms.get_or_insert_with(|| sys_ms() - info.color_hw_ms);
        Ok(FrameInfo { t_ms: info.color_hw_ms + off, ..info })
    }

    /// slam 의 desc 와 같은 꼴의 한 줄
    pub fn describe(&self) -> String {
        let k = &self.calib.color;
        format!(
            "RealSense Intel RealSense D435 {} 펌웨어 {}, {}x{}@{}, 깊이 {}/m, 컬러 보정 fx {} fy {} cx {} cy {} 왜곡 모델 Inverse Brown Conrady, 정렬 rsdriver(fast_align)",
            self.calib.gvd.serial,
            self.calib.gvd.firmware,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            self.depth_per_m(),
            k.fx,
            k.fy,
            k.ppx,
            k.ppy
        )
    }
}
