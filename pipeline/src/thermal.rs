//! 비상 열 보호: 몇 초마다(기본 5 초) 온도를 읽어 문턱(CPU 80 °C = 펌웨어 스로틀 직전, NPU 는 `--thermal-npu`)을 넘으면
//! 선택 일만 줄인다 — 1 단 신경망 루프(MegaLoc)를 노드 3 개에 하나로, 2 단 SigLIP2 사진 벡터 묶음 늦추기. 분할 · 위치 추적은 줄이지 않는다.
//! 문턱보다 5 °C 내려가면 되돌린다. `--no-thermal` 로 끔. 읽기는 루프 마감(타이머)으로만 해 따로 깨우지 않는다.
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub struct Thermal {
    on: bool,
    cpu_max: f32,
    npu_max: Option<f32>,
    cpu_file: PathBuf,
    npu_file: Option<PathBuf>,
    every: Duration,
    last: Instant,
    /// 0 = 보통, 1 = 신경망 루프 노드 3 개에 하나, 2 = + 사진 벡터 늦춤
    pub level: u8,
    /// 비상 보호가 걸린 횟수 · 가장 높았던 CPU · NPU 온도
    pub hits: u32,
    pub cpu_peak: f32,
    pub npu_peak: f32,
    /// 걸려 있던 시간(초)
    pub hot_s: f64,
    hot_since: Option<Instant>,
}

/// 밀리 °C 정수 파일(sysfs thermal_zone · hwmon) → °C
fn read_c(p: &std::path::Path) -> Option<f32> {
    let s = std::fs::read_to_string(p).ok()?;
    let v: f32 = s.trim().parse().ok()?;
    Some(if v > 1000.0 { v / 1000.0 } else { v })
}

impl Thermal {
    /// `--no-thermal` · `--thermal-cpu 80` · `--thermal-npu <°C>` · `--thermal-npu-file <밀리 °C 파일>` · `--thermal-file <CPU 밀리 °C 파일>`(PC 흉내) · `--thermal-every 5`
    pub fn from_args(v: &common::cli::Argv) -> Thermal {
        let cpu_file = v.get("--thermal-file").map(PathBuf::from).unwrap_or_else(|| "/sys/class/thermal/thermal_zone0/temp".into());
        Thermal {
            on: !v.flag("--no-thermal") && (cpu_file.exists() || v.get("--thermal-npu-file").is_some()),
            cpu_max: v.get("--thermal-cpu").and_then(|s| s.parse().ok()).unwrap_or(80.0),
            npu_max: v.get("--thermal-npu").and_then(|s| s.parse().ok()),
            cpu_file,
            npu_file: v.get("--thermal-npu-file").map(PathBuf::from),
            every: Duration::from_secs_f64(v.get("--thermal-every").and_then(|s| s.parse().ok()).unwrap_or(5.0)),
            last: Instant::now(),
            level: 0,
            hits: 0,
            cpu_peak: 0.0,
            npu_peak: 0.0,
            hot_s: 0.0,
            hot_since: None,
        }
    }

    /// 다음 읽기까지 남은 시간(루프 잠 마감)
    pub fn timeout(&self) -> Option<Duration> {
        self.on.then(|| self.every.saturating_sub(self.last.elapsed()))
    }

    /// 때가 됐으면 읽고 단을 정한다. 돌려줌: 단이 바뀌었으면 Some(새 단)
    pub fn check(&mut self) -> Option<u8> {
        if !self.on || self.last.elapsed() < self.every {
            return None;
        }
        self.last = Instant::now();
        let cpu = read_c(&self.cpu_file);
        let npu = self.npu_file.as_deref().and_then(read_c);
        self.step(cpu, npu)
    }

    /// 읽은 온도로 단 정하기(시험에서 바로 부름)
    pub fn step(&mut self, cpu: Option<f32>, npu: Option<f32>) -> Option<u8> {
        if let Some(c) = cpu {
            self.cpu_peak = self.cpu_peak.max(c);
        }
        if let Some(n) = npu {
            self.npu_peak = self.npu_peak.max(n);
        }
        // 넘은 정도: 문턱 위 0 – 3 °C = 1 단, 3 °C 넘게 = 2 단. 문턱 − 5 °C 아래로 내려가면 0
        let over = |t: Option<f32>, m: Option<f32>| match (t, m) {
            (Some(t), Some(m)) => Some(t - m),
            _ => None,
        };
        let d = [over(cpu, Some(self.cpu_max)), over(npu, self.npu_max)].into_iter().flatten().fold(f32::NEG_INFINITY, f32::max);
        let want = if d >= 3.0 {
            2
        } else if d >= 0.0 {
            self.level.max(1)
        } else if d < -5.0 {
            0
        } else {
            self.level
        };
        if want == self.level {
            return None;
        }
        if self.level == 0 && want > 0 {
            self.hits += 1;
            self.hot_since = Some(Instant::now());
        }
        if want == 0 {
            if let Some(c) = self.hot_since.take() {
                self.hot_s += c.elapsed().as_secs_f64();
            }
        }
        self.level = want;
        Some(want)
    }

    pub fn json(&self) -> serde_json::Value {
        let hot = self.hot_s + self.hot_since.map_or(0.0, |c| c.elapsed().as_secs_f64());
        serde_json::json!({"on": self.on, "cpu_max": self.cpu_max, "npu_max": self.npu_max, "hits": self.hits, "level": self.level, "hot_s": hot, "cpu_peak": self.cpu_peak, "npu_peak": self.npu_peak})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Thermal {
        let v = common::cli::Argv(vec!["ondevice".into(), "--thermal-file".into(), "/nonexistent".into()]);
        let mut t = Thermal::from_args(&v);
        t.on = true;
        t
    }

    /// 문턱 80 °C: 넘으면 1 단, 3 °C 넘게면 2 단, 75 °C 아래로 내려가야 되돌림(사이에서는 그대로)
    #[test]
    fn levels_with_hysteresis() {
        let mut t = t();
        assert_eq!(t.step(Some(70.0), None), None);
        assert_eq!(t.step(Some(80.5), None), Some(1));
        assert_eq!(t.step(Some(84.0), None), Some(2));
        assert_eq!(t.step(Some(78.0), None), None); // 75 – 80: 그대로
        assert_eq!(t.step(Some(74.0), None), Some(0));
        assert_eq!(t.hits, 1);
        assert_eq!(t.cpu_peak, 84.0);
    }

    /// PC 흉내: 온도 파일(밀리 °C)을 바꿔 가며 읽기
    #[test]
    fn reads_mock_file() {
        let p = std::env::temp_dir().join(format!("thermal_mock_{}", std::process::id()));
        std::fs::write(&p, "82000\n").unwrap();
        let v = common::cli::Argv(vec!["ondevice".into(), "--thermal-file".into(), p.to_string_lossy().into_owned(), "--thermal-every".into(), "0".into()]);
        let mut t = Thermal::from_args(&v);
        assert_eq!(t.check(), Some(1));
        std::fs::write(&p, "60000\n").unwrap();
        assert_eq!(t.check(), Some(0));
        let _ = std::fs::remove_file(&p);
    }

    /// NPU 문턱만 넘어도 걸림
    #[test]
    fn npu_threshold() {
        let mut t = t();
        t.npu_max = Some(70.0);
        assert_eq!(t.step(Some(60.0), Some(74.0)), Some(2));
    }
}
