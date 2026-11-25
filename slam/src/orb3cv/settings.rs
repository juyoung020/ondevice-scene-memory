//! ORB-SLAM3 설정 읽기(Settings, File.version "1.0", 핀홀 RGB-D)와 Tracking::newParameterLoader · Frame 이 만드는 값.
//! cv::FileStorage YAML 의 쓰는 부분만: `이름: 값` 한 줄(따옴표 문자열 · 정수 · 실수), `#` 주석, `%YAML` 머리.
//! 실수는 FileStorage 처럼 f64 로 읽고(strtod: 정확한 반올림) Settings 가 f32 로 바꾼다 → `parse::<f64>() as f32` 와 같다.
//! 이 파일은 crate 의 다른 모듈을 쓰지 않는다(저장소 밖 하네스가 `#[path]` 로 그대로 끌어와 비교).

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum Val {
    Int(i64),
    Real(f64),
    Str(String),
}

/// Settings 가 읽어 둔 값(RGB-D 에 쓰는 것)
#[derive(Clone, Debug, PartialEq)]
pub struct OrbSettings {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// k1 k2 p1 p2 [k3] (Camera1.k1 이 없으면 비어 있음)
    pub dist: Vec<f32>,
    pub width: i32,
    pub height: i32,
    /// Camera.newWidth · newHeight 가 있으면 그 크기(이때 fx · cx · fy · cy 는 비율만큼 바뀐 값)
    pub new_width: i32,
    pub new_height: i32,
    pub need_resize: bool,
    pub fps: f32,
    pub rgb: bool,
    pub depth_map_factor: f32,
    pub th_depth: f32,
    pub b: f32,
    pub bf: f32,
    pub nfeatures: i32,
    pub scale_factor: f32,
    pub nlevels: i32,
    pub ini_th_fast: f32,
    pub min_th_fast: f32,
    /// System.thFarPoints(없으면 0)
    pub th_far_points: f32,
    pub load_atlas: String,
    pub save_atlas: String,
}

/// Tracking::newParameterLoader(RGB-D) 와 Frame 생성자가 만드는 값
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingParams {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub invfx: f32,
    pub invfy: f32,
    /// mbf
    pub bf: f32,
    /// Frame::mb = mbf / fx
    pub b: f32,
    /// mThDepth = Settings b × ThDepth (Frame 에 그대로 넘어감)
    pub th_depth: f32,
    /// mDepthMapFactor = 1 / DepthMapFactor (|값| < 1e-5 면 1)
    pub depth_scale: f32,
    pub max_frames: i32,
    pub rgb: bool,
}

fn parse_val(s: &str) -> Result<Val, String> {
    let s = s.trim();
    if let Some(r) = s.strip_prefix('"') {
        let e = r.find('"').ok_or_else(|| format!("닫는 따옴표 없음: {s}"))?;
        return Ok(Val::Str(r[..e].to_string()));
    }
    // 끝 주석
    let s = match s.find('#') {
        Some(i) => s[..i].trim(),
        None => s,
    };
    // FileStorage: 숫자에 '.' · 'e' · 'E' 가 있거나 inf/nan 이면 실수, 아니면 정수
    let first = s.chars().next().unwrap_or(' ');
    if first.is_ascii_digit() || first == '-' || first == '+' || first == '.' {
        if s.contains(['.', 'e', 'E']) {
            return s.parse::<f64>().map(Val::Real).map_err(|e| format!("{s}: {e}"));
        }
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Val::Int(i));
        }
    }
    Ok(Val::Str(s.to_string()))
}

fn parse_map(text: &str) -> Result<HashMap<String, Val>, String> {
    let mut m = HashMap::new();
    for line in text.lines() {
        let t = line.trim_end();
        if t.is_empty() || t.starts_with('#') || t.starts_with('%') || t.starts_with("---") {
            continue;
        }
        if t.starts_with(' ') || t.starts_with('\t') {
            // 중첩 값(행렬 등): RGB-D 핀홀에서는 안 씀
            continue;
        }
        let Some(i) = t.find(':') else { continue };
        let (k, v) = (t[..i].trim(), &t[i + 1..]);
        if v.trim().is_empty() || v.trim_start().starts_with("!!") {
            continue;
        }
        m.insert(k.to_string(), parse_val(v)?);
    }
    Ok(m)
}

struct Reader(HashMap<String, Val>);

impl Reader {
    // readParameter<float>: 실수만(정수면 원본은 중단)
    fn f(&self, k: &str) -> Result<Option<f32>, String> {
        match self.0.get(k) {
            None => Ok(None),
            Some(Val::Real(x)) => Ok(Some(*x as f32)),
            Some(_) => Err(format!("{k} 는 실수여야 함")),
        }
    }
    fn freq(&self, k: &str) -> Result<f32, String> {
        self.f(k)?.ok_or_else(|| format!("{k} 없음"))
    }
    // readParameter<int>
    fn i(&self, k: &str) -> Result<Option<i32>, String> {
        match self.0.get(k) {
            None => Ok(None),
            Some(Val::Int(x)) => Ok(Some(*x as i32)),
            Some(_) => Err(format!("{k} 는 정수여야 함")),
        }
    }
    fn ireq(&self, k: &str) -> Result<i32, String> {
        self.i(k)?.ok_or_else(|| format!("{k} 없음"))
    }
    fn s(&self, k: &str) -> Result<Option<String>, String> {
        match self.0.get(k) {
            None => Ok(None),
            Some(Val::Str(x)) => Ok(Some(x.clone())),
            Some(_) => Err(format!("{k} 는 문자열이어야 함")),
        }
    }
}

impl OrbSettings {
    /// Settings(설정 파일, System::RGBD)
    pub fn parse(text: &str) -> Result<OrbSettings, String> {
        let r = Reader(parse_map(text)?);
        let ty = r.s("Camera.type")?.ok_or("Camera.type 없음")?;
        if ty != "PinHole" {
            return Err(format!("핀홀만 옮김: {ty}"));
        }
        let (mut fx, mut fy, mut cx, mut cy) = (r.freq("Camera1.fx")?, r.freq("Camera1.fy")?, r.freq("Camera1.cx")?, r.freq("Camera1.cy")?);
        let mut dist = Vec::new();
        if r.f("Camera1.k1")?.is_some() {
            dist = vec![r.freq("Camera1.k1")?, r.freq("Camera1.k2")?, r.freq("Camera1.p1")?, r.freq("Camera1.p2")?];
            if let Some(k3) = r.f("Camera1.k3")? {
                dist.push(k3);
            }
        }
        let height = r.ireq("Camera.height")?;
        let width = r.ireq("Camera.width")?;
        let (mut new_width, mut new_height, mut need_resize) = (width, height, false);
        if let Some(nh) = r.i("Camera.newHeight")? {
            need_resize = true;
            new_height = nh;
            let s = nh as f32 / height as f32;
            fy *= s;
            cy *= s;
        }
        if let Some(nw) = r.i("Camera.newWidth")? {
            need_resize = true;
            new_width = nw;
            let s = nw as f32 / width as f32;
            fx *= s;
            cx *= s;
        }
        let fps = r.ireq("Camera.fps")? as f32;
        let rgb = r.ireq("Camera.RGB")? != 0;
        let depth_map_factor = r.freq("RGBD.DepthMapFactor")?;
        let th_depth = r.freq("Stereo.ThDepth")?;
        let b = r.freq("Stereo.b")?;
        // bf_ = b_ · calibration1_->getParameter(0) (newWidth 로 바뀐 fx 를 읽음: readImageInfo 가 readRGBD 보다 먼저)
        let bf = b * fx;
        Ok(OrbSettings {
            fx,
            fy,
            cx,
            cy,
            dist,
            width,
            height,
            new_width,
            new_height,
            need_resize,
            fps,
            rgb,
            depth_map_factor,
            th_depth,
            b,
            bf,
            nfeatures: r.ireq("ORBextractor.nFeatures")?,
            scale_factor: r.freq("ORBextractor.scaleFactor")?,
            nlevels: r.ireq("ORBextractor.nLevels")?,
            ini_th_fast: r.ireq("ORBextractor.iniThFAST")? as f32,
            min_th_fast: r.ireq("ORBextractor.minThFAST")? as f32,
            th_far_points: r.f("System.thFarPoints")?.unwrap_or(0.0),
            load_atlas: r.s("System.LoadAtlasFromFile")?.unwrap_or_default(),
            save_atlas: r.s("System.SaveAtlasToFile")?.unwrap_or_default(),
        })
    }

    /// Tracking::newParameterLoader(RGB-D) + Frame 생성자
    pub fn tracking(&self) -> TrackingParams {
        let f = self.depth_map_factor;
        TrackingParams {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            invfx: 1.0f32 / self.fx,
            invfy: 1.0f32 / self.fy,
            bf: self.bf,
            b: self.bf / self.fx,
            th_depth: self.b * self.th_depth,
            depth_scale: if f.abs() < 1e-5 { 1.0 } else { 1.0f32 / f },
            // mMaxFrames = settings->fps() (float → int 잘라냄)
            max_frames: self.fps as i32,
            rgb: self.rgb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_basic() {
        let y = "%YAML:1.0\nFile.version: \"1.0\"\nCamera.type: \"PinHole\"\nCamera1.fx: 517.306408\nCamera1.fy: 516.469215\nCamera1.cx: 318.643040\nCamera1.cy: 255.313989\n\
Camera.width: 640\nCamera.height: 480\nCamera.fps: 30\nCamera.RGB: 1\nStereo.ThDepth: 40.0\nStereo.b: 0.07732\nRGBD.DepthMapFactor: 5000.0 # 주석\n\
ORBextractor.nFeatures: 1000\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n";
        let s = OrbSettings::parse(y).unwrap();
        assert_eq!(s.fx, 517.306408f32);
        assert!(s.dist.is_empty());
        assert_eq!(s.tracking().depth_scale, 1.0f32 / 5000.0f32);
        assert!(OrbSettings::parse("Camera.type: \"KannalaBrandt8\"\n").is_err());
    }
}
