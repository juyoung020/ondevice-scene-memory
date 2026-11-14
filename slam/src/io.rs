//! 장면 입력: TUM 폴더 재생, RealSense 카메라(`realsense` 기능), 읽기 스레드(PrefetchSource).
//! 원본: bridge/src/frame_source.h · tum_source.cpp · prefetch_source.h · realsense_source.cpp · depth_align.cpp.
//! 회색조·깊이 mm 바꾸기는 OpenCV 4.10 cvtColor(COLOR_BGR2GRAY) · Mat::convertTo 와 비트까지 같게 한다.
use crate::{Camera, Features, Frame, Pose};
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "realsense")]
pub mod realsense;
#[cfg(feature = "realsense")]
pub use realsense::RealSenseSource;
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
}

/// 카메라 광학 축(x 오른쪽, y 아래, z 앞) → 몸체 축(x 앞, y 왼쪽, z 위). RTAB-Map CameraModel::opticalRotation
pub fn optical_rotation() -> Pose {
    let m = Matrix3::new(0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0);
    Pose::from_parts(Translation3::identity(), UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(m)))
}

/// BGR(또는 RGB) → 회색조. OpenCV cvtColor COLOR_BGR2GRAY 의 8 비트 고정소수점 계산 그대로:
/// (B·3735 + G·19235 + R·9798 + 2^14) >> 15 (4.10 의 cvtColor 가 쓰는 15 비트 계수)
pub fn to_gray(color: &[u8], is_rgb: bool) -> Vec<u8> {
    let (cb, cr) = if is_rgb { (2, 0) } else { (0, 2) };
    color
        .chunks_exact(3)
        .map(|p| ((p[cb] as u32 * 3735 + p[1] as u32 * 19235 + p[cr] as u32 * 9798 + (1 << 14)) >> 15) as u8)
        .collect()
}

/// 깊이 단위가 mm(1000/m)라 원본 값이 곧 mm 인가
pub fn is_mm(depth_per_m: f32) -> bool {
    (depth_per_m - 1000.0).abs() <= 1e-3
}

/// 원본 깊이 → (mm 깊이, 스트림용 원본). 1000/m 이면 두 값이 같아 버퍼 하나만 mm 칸에 두고 원본 칸은 비운다
/// (복사 614,400 B 없음, 보낼 때 `Frame::take_stream_depth` 가 mm 칸을 넘김)
pub fn split_depth(raw: Vec<u16>, depth_per_m: f32) -> (Vec<u16>, Vec<u16>) {
    if is_mm(depth_per_m) {
        (raw, Vec::new())
    } else {
        (depth_to_mm(&raw, depth_per_m), raw)
    }
}

/// 깊이 값 → mm(u16). 원본 bridge main.cpp: 1000/m 이 아니면 Mat 를 (값/m ÷ 1000) 으로 나눔.
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
    let text = std::fs::read_to_string(path).map_err(|_| format!("읽을 수 없음: {path}"))?;
    let mut out = vec![];
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        if let (Some(t), Some(rel)) = (it.next().and_then(|s| s.parse::<f64>().ok()), it.next()) {
            out.push((t, rel.to_string()));
        }
    }
    Ok(out)
}

/// TUM RGB-D 폴더 재생(rgb.txt · depth.txt, 깊이 5000/m): 카메라 없이 시험할 때
pub struct TumSource {
    dir: String,
    rate: f64,
    rgb: Vec<(f64, String)>,
    depth: Vec<(f64, String)>,
    i: usize,
    skipped: Arc<AtomicUsize>,
    k: [f64; 4],
    read_ms: f64,
    start: Option<(f64, Instant)>,
    id: i64,
}

impl TumSource {
    /// rate: 0 = 최대 속도(모든 장면), 1 = 찍은 시각 간격대로(밀리면 건너뜀), 2 = 두 배 빠르게 ...
    pub fn new(dir: &str, rate: f64) -> Result<TumSource, String> {
        let rgb = read_list(&format!("{dir}/rgb.txt"))?;
        let depth = read_list(&format!("{dir}/depth.txt"))?;
        if rgb.is_empty() || depth.is_empty() {
            return Err(format!("빈 TUM 폴더: {dir}"));
        }
        // 시퀀스별 보정값(TUM 문서). memmap tum.rs 와 같은 규칙: 폴더 이름으로 고름
        let mut k = if dir.contains("freiburg1") {
            [517.3, 516.5, 318.6, 255.3]
        } else if dir.contains("freiburg2") {
            [520.9, 521.0, 325.1, 249.7]
        } else {
            [535.4, 539.2, 320.1, 247.6]
        };
        // 폴더에 calibration.txt(fx fy cx cy, ETH3D 형식)가 있으면 그 값을 씀
        if let Ok(s) = std::fs::read_to_string(format!("{dir}/calibration.txt")) {
            let c: Vec<f64> = s.split_whitespace().take(4).map_while(|w| w.parse().ok()).collect();
            if c.len() == 4 {
                k.copy_from_slice(&c);
            }
        }
        Ok(TumSource { dir: dir.to_string(), rate, rgb, depth, i: 0, skipped: Arc::new(AtomicUsize::new(0)), k, read_ms: 0.0, start: None, id: 0 })
    }
    pub fn len(&self) -> usize {
        self.rgb.len()
    }
    /// 재생에서 밀려 건너뛴 장면 수(읽기 스레드로 옮긴 뒤에도 읽을 수 있게 나눠 가짐)
    pub fn skipped_counter(&self) -> Arc<AtomicUsize> {
        self.skipped.clone()
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
}

impl FrameSource for TumSource {
    fn describe(&self) -> String {
        let r = if self.rate > 0.0 { format!("{:.6}배", self.rate) } else { "최대".into() };
        format!("TUM {} ({} 장, 속도 {r})", self.dir, self.rgb.len())
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn next(&mut self) -> Result<Option<Frame>, String> {
        while self.i < self.rgb.len() {
            let (t, rel) = self.rgb[self.i].clone();
            self.i += 1;
            if self.rate > 0.0 {
                match self.start {
                    None => self.start = Some((t, Instant::now())),
                    Some((t0, w0)) => {
                        let ahead = (t - t0) / self.rate - w0.elapsed().as_secs_f64();
                        if ahead < -0.1 {
                            // 처리가 밀렸으면 카메라처럼 지난 장면은 버림
                            self.skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        if ahead > 0.0 {
                            std::thread::sleep(Duration::from_secs_f64(ahead));
                        }
                    }
                }
            }
            let c0 = Instant::now();
            let j = self.pair(t);
            let (cp, dp) = (format!("{}/{}", self.dir, rel), format!("{}/{}", self.dir, self.depth[j].1));
            // 컬러와 깊이 PNG 를 나란히 푼다
            let (ci, di) = std::thread::scope(|s| {
                let h = s.spawn(|| read_png(&dp));
                let c = read_png(&cp);
                (c, h.join().unwrap())
            });
            let (ci, di) = (ci?, di?);
            let (w, h) = (ci.width(), ci.height());
            // 푼 RGB 그대로 둔다(스트림·memmap 이 RGB 를 씀). 회색조는 채널 차례만 맞춰 imread(BGR) + cvtColor 와 같은 값
            let color = match ci {
                image::DynamicImage::ImageRgb8(b) => b.into_raw(),
                o => o.to_rgb8().into_raw(),
            };
            let depth_raw = match di {
                image::DynamicImage::ImageLuma16(b) if b.width() == w && b.height() == h => b.into_raw(),
                _ => return Err(format!("장면 읽기 실패: {rel}")),
            };
            let gray = to_gray(&color, true);
            let depth_mm = depth_to_mm(&depth_raw, 5000.0);
            self.id += 1;
            let k = self.k;
            let f = Frame {
                id: self.id,
                t,
                gray,
                color,
                color_is_rgb: true,
                depth_mm,
                depth_raw,
                depth_per_m: 5000.0,
                cam: Camera { fx: k[0], fy: k[1], cx: k[2], cy: k[3], width: w, height: h, local: optical_rotation() },
            };
            self.read_ms = c0.elapsed().as_secs_f64() * 1e3;
            return Ok(Some(f));
        }
        Ok(None)
    }
}

/// 읽기 스레드가 장면마다 미리 해 둘 일(특징 추출 등). 결과는 장면과 함께 넘겨짐
pub type Prep = Box<dyn FnMut(&Frame) -> Features + Send>;

struct Slot {
    frame: Option<(Frame, f64, Option<Features>)>,
    done: bool,
    stop: bool,
    err: Option<String>,
}

/// 장면 읽기(카메라 대기·깊이 정렬·PNG 풀기)를 따로 도는 스레드에 맡긴다: 위치 추적이 읽기를 기다리지 않게.
/// drop = true(카메라·찍은 속도 재생): 위치 추적이 밀리면 아직 안 가져간 장면을 새 장면으로 바꿈
/// drop = false(최대 속도 재생): 한 장 앞서 읽고 가져갈 때까지 기다림(장면을 하나도 버리지 않음)
pub struct PrefetchSource {
    sh: Arc<(Mutex<Slot>, Condvar)>,
    th: Option<std::thread::JoinHandle<()>>,
    desc: String,
    read_ms: f64,
    dropped: Arc<AtomicUsize>,
    prepped: Option<Features>,
}

impl PrefetchSource {
    pub fn new(inner: Box<dyn FrameSource>, drop: bool) -> PrefetchSource {
        Self::with_prep(inner, drop, None)
    }

    /// prep 이 있으면 읽기 스레드가 장면마다 그것을 돌려 결과를 장면과 함께 넘긴다(take_prepped 로 꺼냄).
    /// 읽기 시간(last_read_ms)에는 prep 시간이 들어가지 않는다
    pub fn with_prep(mut inner: Box<dyn FrameSource>, drop: bool, mut prep: Option<Prep>) -> PrefetchSource {
        let desc = inner.describe() + if drop { ", 읽기 스레드(밀리면 지난 장면 버림)" } else { ", 읽기 스레드" };
        let sh = Arc::new((Mutex::new(Slot { frame: None, done: false, stop: false, err: None }), Condvar::new()));
        let dropped = Arc::new(AtomicUsize::new(0));
        let (sh2, dr2) = (sh.clone(), dropped.clone());
        let th = std::thread::spawn(move || {
            let (m, cv) = &*sh2;
            loop {
                match inner.next() {
                    Ok(Some(f)) => {
                        let ms = inner.last_read_ms();
                        let pf = prep.as_mut().map(|p| p(&f));
                        let mut s = m.lock().unwrap();
                        if !drop {
                            s = cv.wait_while(s, |s| s.frame.is_some() && !s.stop).unwrap();
                        }
                        if s.stop {
                            return;
                        }
                        if s.frame.is_some() {
                            dr2.fetch_add(1, Ordering::Relaxed);
                        }
                        s.frame = Some((f, ms, pf));
                        cv.notify_all();
                    }
                    Ok(None) => break,
                    Err(e) => {
                        m.lock().unwrap().err = Some(e);
                        break;
                    }
                }
            }
            m.lock().unwrap().done = true;
            cv.notify_all();
        });
        PrefetchSource { sh, th: Some(th), desc, read_ms: 0.0, dropped, prepped: None }
    }
    /// 위치 추적이 밀려 가져가기 전에 새 장면으로 바뀐 장면 수
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
    /// 마지막 next 장면에 읽기 스레드가 해 둔 prep 결과(한 번만 꺼낼 수 있음)
    pub fn take_prepped(&mut self) -> Option<Features> {
        self.prepped.take()
    }
}

impl FrameSource for PrefetchSource {
    fn next(&mut self) -> Result<Option<Frame>, String> {
        let (m, cv) = &*self.sh;
        let mut s = cv.wait_while(m.lock().unwrap(), |s| s.frame.is_none() && !s.done).unwrap();
        match s.frame.take() {
            Some((f, ms, pf)) => {
                self.read_ms = ms;
                self.prepped = pf;
                cv.notify_all();
                Ok(Some(f))
            }
            None => match s.err.take() {
                Some(e) => Err(e),
                None => Ok(None),
            },
        }
    }
    fn last_read_ms(&self) -> f64 {
        self.read_ms
    }
    fn describe(&self) -> String {
        self.desc.clone()
    }
}

impl Drop for PrefetchSource {
    fn drop(&mut self) {
        let (m, cv) = &*self.sh;
        m.lock().unwrap().stop = true;
        cv.notify_all();
        // 읽기 스레드는 지금 읽던 장면을 마치면 스스로 끝난다. 카메라 대기 중일 수 있어 끝난 것만 거둔다
        if let Some(th) = self.th.take() {
            if th.is_finished() {
                let _ = th.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_and_depth_rounding() {
        // OpenCV 값: (255,0,0)BGR → 29, (0,255,0) → 150, (0,0,255) → 76
        assert_eq!(to_gray(&[255, 0, 0, 0, 255, 0, 0, 0, 255], false), vec![29, 150, 76]);
        assert_eq!(depth_to_mm(&[0, 2, 3, 7, 8, 65535], 5000.0), vec![0, 0, 1, 1, 2, 13107]);
    }

    /// 전체 시퀀스를 OpenCV 결과와 비교: SLAM_IO_REF=<C++ 하네스 출력> SLAM_IO_TUM=<폴더>
    /// 하네스 출력 = 장면마다 [회색 w·h][깊이 mm u16 w·h]
    #[test]
    #[ignore]
    fn match_opencv_whole_sequence() {
        let (Ok(rf), Ok(dir)) = (std::env::var("SLAM_IO_REF"), std::env::var("SLAM_IO_TUM")) else { return };
        let r = std::fs::read(rf).unwrap();
        let mut src = TumSource::new(&dir, 0.0).unwrap();
        let (mut off, mut n, mut ms) = (0usize, 0usize, 0.0);
        while let Some(f) = src.next().unwrap() {
            ms += src.last_read_ms();
            let np = f.gray.len();
            assert_eq!(&r[off..off + np], &f.gray[..], "회색 장면 {n}");
            off += np;
            let d: Vec<u8> = f.depth_mm.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(&r[off..off + 2 * np], &d[..], "깊이 장면 {n}");
            off += 2 * np;
            n += 1;
        }
        assert_eq!(off, r.len());
        eprintln!("같은 장면 {n} 개, 읽기 평균 {:.2} ms", ms / n as f64);
    }
}
