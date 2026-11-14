//! RTAB-Map 파라미터: 기본값 ← ini([Core] 의 `이름=값`) ← 명령줄(`--Group/Name 값`). 원본 Parameters.h 의 이름을 그대로 쓴다.
//! 각 모듈이 자기 파라미터를 `get` 으로 읽고, 기본값은 그 모듈이 RTAB-Map 0.24.2 Parameters.h 의 값으로 적는다.
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct Params(pub BTreeMap<String, String>);

impl Params {
    /// ini 파일의 `이름=값` 줄(`;`·`#` 주석, `[절]` 줄은 건너뜀)
    pub fn read_ini(path: &str) -> Result<Params, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut p = Params::default();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with(';') || l.starts_with('#') || l.starts_with('[') {
                continue;
            }
            if let Some((k, v)) = l.split_once('=') {
                p.0.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        Ok(p)
    }

    pub fn set(&mut self, k: &str, v: &str) {
        self.0.insert(k.to_string(), v.to_string());
    }

    /// 값이 있으면 그것, 없거나 못 읽으면 기본값
    pub fn get<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.0.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// 참·거짓("true"/"false"/"1"/"0")
    pub fn get_bool(&self, k: &str, default: bool) -> bool {
        match self.0.get(k).map(|s| s.to_ascii_lowercase()) {
            Some(s) if s == "true" || s == "1" => true,
            Some(s) if s == "false" || s == "0" => false,
            _ => default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ini_and_overrides() {
        let p = std::env::temp_dir().join(format!("slam_params_{}.ini", std::process::id()));
        std::fs::write(&p, "; 주석\n[Core]\nVis/MaxFeatures=400\nRtabmap/DetectionRate=1\n").unwrap();
        let mut a = Params::read_ini(p.to_str().unwrap()).unwrap();
        assert_eq!(a.get("Vis/MaxFeatures", 1000), 400);
        a.set("Vis/MaxFeatures", "300");
        assert_eq!(a.get("Vis/MaxFeatures", 1000), 300);
        assert_eq!(a.get("Vis/MinInliers", 20), 20);
        assert!(!a.get_bool("Reg/Force3DoF", false));
        let _ = std::fs::remove_file(p);
    }
}
