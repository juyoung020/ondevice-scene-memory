//! 장면 입력: 녹화 재생(TUM 꼴 폴더, rsd_rec 녹화), RealSense D435(`rsdriver` 기능).
//! 깊이 mm 바꾸기는 OpenCV 4.10 Mat::convertTo 와 비트까지 같게 한다.
use crate::{Camera, Frame, Ir, StereoCalib};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "rsdriver")]
pub mod rsdriver;
#[cfg(feature = "rsdriver")]
pub use self::rsdriver::RsDriverSource;

/// 장면 입력의 공통 모양
pub trait FrameSource: Send {
    /// 다음 장면. 끝이면 Ok(None)
    fn next(&mut self) -> Result<Option<Frame>, String>;
    /// 장면 읽기에 쓴 시간(ms, 마지막 next 의 카메라 대기·파일 읽기·정렬)
    fn last_read_ms(&self) -> f64;
    fn describe(&self) -> String;
    /// 장치 파일 기술자(카메라): 처리 루프 epoll 에 건다. 비었으면 파일 입력(풀에서 미리 풂)
    fn fds(&self) -> Vec<i32> {
        vec![]
    }
    /// 기다리지 않는 다음 장면(카메라): Ok(None) = 아직(장치 알림을 기다림). 기본은 next 그대로
    fn try_next(&mut self) -> Result<Option<Option<Frame>>, String> {
        self.next().map(Some)
    }
    /// 찍은 속도 재생(--rate R > 0): 잠들지 않게 하고 (배속, 재생 시계)를 돌려줌 — 부르는 쪽(처리 루프)이 장면을 제때까지 둠
    fn caller_paces(&mut self) -> Option<(f64, Arc<std::sync::OnceLock<(f64, Instant)>>)> {
        None
    }
    /// 적외선 좌우 보정값(장면에 적외선이 실려 올 때)
    fn stereo(&self) -> Option<StereoCalib> {
        None
    }
}

/// 깊이 단위가 mm(1000/m)라 원본 값이 곧 mm 인가
pub fn is_mm(depth_per_m: f32) -> bool {
    (depth_per_m - 1000.0).abs() <= 1e-3
}

/// 원본 깊이 → (mm 깊이, 스트림용 원본). 1000/m 이면 두 값이 같아 버퍼 하나만 mm 칸에 두고 원본 칸은 비운다
/// (복사 614,400 B 없음, 보낼 때 `Frame::take_stream_depth` 가 mm 칸을 넘김)
pub fn split_depth(raw: Vec<u16>, depth_per_m: f32) -> (crate::DepthBuf, crate::DepthBuf) {
    if is_mm(depth_per_m) {
        (raw.into(), Default::default())
    } else {
        (depth_to_mm(&raw, depth_per_m).into(), raw.into())
    }
}

/// 깊이 값 → mm(u16). 1000/m 이 아니면 (값/m ÷ 1000) 으로 나눔.
/// OpenCV convertTo(16U→16U, 배율) 는 float 로 값×배율을 구해 짝수 쪽 반올림 후 0..65535 로 자름
pub fn depth_to_mm(raw: &[u16], depth_per_m: f32) -> Vec<u16> {
    if is_mm(depth_per_m) {
        return raw.to_vec();
    }
    let div = depth_per_m as f64 / 1000.0;
    let alpha = (1.0 / div) as f32;
    raw.iter().map(|&v| (v as f32 * alpha).round_ties_even().clamp(0.0, 65535.0) as u16).collect()
}

/// PNG 풀기. OpenCV imread 와 같은 화소(무손실)
fn read_png(path: &str) -> Result<image::DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("장면 읽기 실패: {path}: {e}"))?;
    image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).map_err(|e| format!("장면 읽기 실패: {path}: {e}"))
}

fn read_list(path: &str) -> Result<Vec<(f64, String)>, String> {
    common::tum::read_list(path).map_err(|_| format!("읽을 수 없음: {path}"))
}

/// calib.json 의 "key": 다음 숫자(from 바이트 뒤 첫 번째)
fn json_num(j: &str, key: &str, from: usize) -> Option<f64> {
    let i = j[from..].find(&format!("\"{key}\":"))? + from + key.len() + 3;
    j[i..].trim_start().split(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace()).next()?.parse().ok()
}

/// calib.json 의 "key": [ ... ] 숫자 배열
fn json_arr(j: &str, key: &str) -> Option<Vec<f64>> {
    let i = j.find(&format!("\"{key}\":"))?;
    let a = j[i..].find('[')? + i + 1;
    let b = j[a..].find(']')? + a;
    j[a..b].split(',').map(|x| x.trim().parse().ok()).collect()
}

/// rsd_rec 녹화의 적외선 보정값(calib.json: depth_intrinsics · depth_to_color · stereo_baseline_m)
fn stereo_from_json(j: &str) -> Option<StereoCalib> {
    let d0 = j.find("\"depth_intrinsics\"")?;
    let k = [json_num(j, "fx", d0)?, json_num(j, "fy", d0)?, json_num(j, "ppx", d0)?, json_num(j, "ppy", d0)?];
    let (w, h) = (json_num(j, "width", d0)? as u32, json_num(j, "height", d0)? as u32);
    let r = json_arr(j, "rotation_col_major").filter(|v| v.len() == 9)?;
    let t = json_arr(j, "translation_m").filter(|v| v.len() == 3)?;
    // 열 우선 → 3×4 행 우선
    let color_from_ir = [r[0], r[3], r[6], t[0], r[1], r[4], r[7], t[1], r[2], r[5], r[8], t[2]];
    Some(StereoCalib { k, w, h, baseline: json_num(j, "stereo_baseline_m", 0)?, color_from_ir })
}

/// 녹화 폴더 재생(rgb.txt · depth.txt, 깊이 TUM 5000/m · rsd_rec 1000/m).
/// rsd_rec --ir 녹화(ir_times.txt + mav0/cam0 · cam1)면 장면마다 가장 가까운 시각의 적외선 좌우도 싣는다
pub struct TumSource {
    dir: String,
    rate: f64,
    rgb: Vec<(f64, String)>,
    depth: Vec<(f64, String)>,
    /// 적외선 시각(ns 글자, 초)
    ir: Vec<(String, f64)>,
    stereo: Option<StereoCalib>,
    i: usize,
    k: [f64; 4],
    /// 깊이 PNG 값/m: TUM 5000, rsd_rec 녹화(calib.json 의 depth_png_per_m) 1000
    depth_per_m: f32,
    read_ms: f64,
    start: Option<(f64, Instant)>,
    /// 재생 시계(첫 장면 시각, 그때 벽시계): IMU 재생 스레드가 같은 시계로 표본을 넣음
    clock: Arc<std::sync::OnceLock<(f64, Instant)>>,
    /// 찍은 속도 재생을 부르는 쪽이 맞춤(밀린 장면 버리기만 여기서, 잠들지 않음)
    no_sleep: bool,
    /// 밀려 건너뛴 장면 수
    pub skipped: usize,
    id: i64,
}

impl TumSource {
    /// rate: 0 = 최대 속도(모든 장면), 1 = 찍은 시각 간격대로(밀리면 건너뜀), 2 = 두 배 빠르게 ...
    pub fn new(dir: &str, rate: f64) -> Result<TumSource, String> {
        let rgb = read_list(&format!("{dir}/rgb.txt"))?;
        let depth = read_list(&format!("{dir}/depth.txt"))?;
        if rgb.is_empty() || depth.is_empty() {
            return Err(format!("빈 녹화 폴더: {dir}"));
        }
        // 시퀀스별 보정값(TUM 문서): 폴더 이름으로 고름(물체 기억의 TUM 입력과 같은 표)
        let mut k = common::tum::camera_k(dir);
        // 폴더에 calibration.txt(fx fy cx cy, ETH3D 형식)가 있으면 그 값을 씀
        if let Ok(s) = std::fs::read_to_string(format!("{dir}/calibration.txt")) {
            let c: Vec<f64> = s.split_whitespace().take(4).map_while(|w| w.parse().ok()).collect();
            if c.len() == 4 {
                k.copy_from_slice(&c);
            }
        }
        // rsd_rec 녹화 폴더: calib.json 의 컬러 보정 · 깊이 배율 · 적외선 보정
        let mut depth_per_m = 5000.0f32;
        let mut stereo = None;
        if let Ok(j) = std::fs::read_to_string(format!("{dir}/calib.json")) {
            if let Some(v) = json_num(&j, "depth_png_per_m", 0) {
                depth_per_m = v as f32;
            }
            if let Some(c0) = j.find("\"color_intrinsics\"") {
                if let (Some(fx), Some(fy), Some(cx), Some(cy)) = (json_num(&j, "fx", c0), json_num(&j, "fy", c0), json_num(&j, "ppx", c0), json_num(&j, "ppy", c0)) {
                    k = [fx, fy, cx, cy];
                }
            }
            stereo = stereo_from_json(&j);
        }
        let ir: Vec<(String, f64)> = std::fs::read_to_string(format!("{dir}/ir_times.txt"))
            .map(|s| s.lines().filter_map(|l| l.trim().parse::<i64>().ok().map(|ns| (l.trim().to_string(), ns as f64 * 1e-9))).collect())
            .unwrap_or_default();
        if !ir.is_empty() && stereo.is_none() {
            return Err(format!("{dir}: 적외선 녹화인데 calib.json 에 적외선 보정값(depth_intrinsics · depth_to_color · stereo_baseline_m)이 없음"));
        }
        Ok(TumSource { dir: dir.to_string(), rate, rgb, depth, stereo: if ir.is_empty() { None } else { stereo }, ir, i: 0, k, depth_per_m, read_ms: 0.0, start: None, clock: Default::default(), no_sleep: false, skipped: 0, id: 0 })
    }
    /// 재생 시계(첫 장면을 낼 때 정해짐)
    pub fn clock(&self) -> Arc<std::sync::OnceLock<(f64, Instant)>> {
        self.clock.clone()
    }
    pub fn len(&self) -> usize {
        self.rgb.len()
    }

    /// 시각 t 에 가장 가까운 깊이(memmap tum.rs kinect_depth 와 같은 짝)
    fn pair(&self, t: f64) -> usize {
        let d = &self.depth;
        let mut j = d.partition_point(|x| x.0 < t);
        if j >= d.len() {
            j = d.len() - 1;
        } else if j > 0 && (d[j].0 - t).abs() >= (t - d[j - 1].0).abs() {
            j -= 1;
        }
        j
    }

    /// 시각 t 의 적외선 좌우(20 ms 안에 없으면 None)
    fn ir_at(&self, t: f64) -> Result<Option<Ir>, String> {
        let s = &self.ir;
        if s.is_empty() {
            return Ok(None);
        }
        let j = s.partition_point(|x| x.1 < t);
        let j = [j.wrapping_sub(1), j].into_iter().filter(|&i| i < s.len()).min_by(|&a, &b| (s[a].1 - t).abs().total_cmp(&(s[b].1 - t).abs())).unwrap();
        if (s[j].1 - t).abs() > 0.02 {
            return Ok(None);
        }
        let (l, r) = (format!("{}/mav0/cam0/data/{}.png", self.dir, s[j].0), format!("{}/mav0/cam1/data/{}.png", self.dir, s[j].0));
        let (li, ri) = rayon::join(|| read_png(&l), || read_png(&r));
        let (li, ri) = (li?.into_luma8(), ri?.into_luma8());
        Ok(Some(Ir { w: li.width(), h: li.height(), left: li.into_raw(), right: ri.into_raw(), t: s[j].1 }))
    }
}

impl FrameSource for TumSource {
    fn caller_paces(&mut self) -> Option<(f64, Arc<std::sync::OnceLock<(f64, Instant)>>)> {
        (self.rate > 0.0).then(|| {
            self.no_sleep = true;
            (self.rate, self.clock.clone())
        })
    }
    fn describe(&self) -> String {
        let r = if self.rate > 0.0 { format!("{:.6}배", self.rate) } else { "최대".into() };
        format!("녹화 {} ({} 장, 속도 {r}{})", self.dir, self.rgb.len(), if self.ir.is_empty() { "" } else { ", 적외선 좌우" })
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn stereo(&self) -> Option<StereoCalib> {
        self.stereo.clone()
    }
    fn next(&mut self) -> Result<Option<Frame>, String> {
        while self.i < self.rgb.len() {
            let (t, rel) = self.rgb[self.i].clone();
            self.i += 1;
            if self.rate > 0.0 {
                match self.start {
                    None => {
                        self.start = Some((t, Instant::now()));
                        let _ = self.clock.set((t, Instant::now()));
                    }
                    Some((t0, w0)) => {
                        let ahead = (t - t0) / self.rate - w0.elapsed().as_secs_f64();
                        if ahead < -0.1 {
                            // 처리가 밀렸으면 카메라처럼 지난 장면은 버림
                            self.skipped += 1;
                            continue;
                        }
                        if ahead > 0.0 && !self.no_sleep {
                            std::thread::sleep(Duration::from_secs_f64(ahead));
                        }
                    }
                }
            }
            let c0 = Instant::now();
            let j = self.pair(t);
            let (cp, dp) = (format!("{}/{}", self.dir, rel), format!("{}/{}", self.dir, self.depth[j].1));
            // 컬러와 깊이 PNG 를 나란히 푼다(작업 풀)
            let (ci, di) = rayon::join(|| read_png(&cp), || read_png(&dp));
            let (ci, di) = (ci?, di?);
            let (w, h) = (ci.width(), ci.height());
            let color = match ci {
                image::DynamicImage::ImageRgb8(b) => b.into_raw(),
                o => o.to_rgb8().into_raw(),
            };
            let depth_raw = match di {
                image::DynamicImage::ImageLuma16(b) if b.width() == w && b.height() == h => b.into_raw(),
                _ => return Err(format!("장면 읽기 실패: {rel}")),
            };
            let ir = self.ir_at(t)?;
            let depth_mm = depth_to_mm(&depth_raw, self.depth_per_m).into();
            self.id += 1;
            let k = self.k;
            let f = Frame {
                id: self.id,
                t,
                color,
                color_is_rgb: true,
                depth_mm,
                depth_raw: depth_raw.into(),
                depth_per_m: self.depth_per_m,
                cam: Camera { fx: k[0], fy: k[1], cx: k[2], cy: k[3], width: w, height: h },
                ir,
            };
            self.read_ms = c0.elapsed().as_secs_f64() * 1e3;
            return Ok(Some(f));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_rounding() {
        assert_eq!(depth_to_mm(&[0, 2, 3, 7, 8, 65535], 5000.0), vec![0, 0, 1, 1, 2, 13107]);
    }

    /// rsd_rec calib.json(walk1 의 실제 값 꼴)에서 적외선 보정: 열 우선 회전 → 행 우선
    #[test]
    fn stereo_calib_json() {
        let j = r#"{
  "depth_intrinsics": {"width": 640, "height": 480, "ppx": 321.706024, "ppy": 238.147339, "fx": 382.494873, "fy": 382.494873, "model": 4, "coeffs": [0, 0, 0, 0, 0]},
  "color_intrinsics": {"width": 640, "height": 480, "ppx": 324.950439, "ppy": 241.556976, "fx": 614.201904, "fy": 614.689209, "model": 2, "coeffs": [0, 0, 0, 0, 0]},
  "depth_to_color": {"rotation_col_major": [0.999734461, -0.0217354111, -0.00765662361, 0.0217765607, 0.999748647, 0.00533265807, 0.00753879128, -0.00549797667, 0.999956489], "translation_m": [0.014739234, -0.000185367098, 0.000469290564]},
  "depth_png_per_m": 999.999939,
  "stereo_baseline_m": 0.0499
}"#;
        let s = stereo_from_json(j).unwrap();
        assert_eq!((s.w, s.h), (640, 480));
        assert!((s.k[0] - 382.494873).abs() < 1e-6 && (s.k[3] - 238.147339).abs() < 1e-6);
        assert!((s.baseline - 0.0499).abs() < 1e-9);
        // 행 0 = 열 우선 배열의 0, 3, 6 번
        assert!((s.color_from_ir[1] - 0.0217765607).abs() < 1e-9 && (s.color_from_ir[4] + 0.0217354111).abs() < 1e-9);
        assert!((s.color_from_ir[3] - 0.014739234).abs() < 1e-9);
    }
}
