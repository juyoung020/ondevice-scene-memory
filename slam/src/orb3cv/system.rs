//! System 진입(RGB-D): 설정 읽기(Settings) → Tracking::newParameterLoader 값 · 카메라 · ORB 층 표 · 첫 장면 경계(FrameCtx),
//! 설정의 System.LoadAtlasFromFile 이 있으면 Atlas 불러오기, 없으면 빈 Atlas(지도 0 하나).
//! 장면 입력은 Tracking::GrabImageRGBD 와 같게: 3 · 4 채널 컬러 → 회색(mbRGB 로 RGB/BGR), 16 비트 깊이 → f32(× mDepthMapFactor).
//! 스레드(추적 · 지도 · 루프)와 단어장 읽기는 각 모듈(TRACK · LOOP) 몫이라 여기서는 묶음 값만 만든다.

use super::atlas_io;
use super::img::{self, ColorOrder};
use super::settings::{OrbSettings, TrackingParams};
use crate::orb3rs::types::{Atlas, FrameCtx, Map, Pyramid};
use std::sync::Arc;

pub struct SystemInit {
    pub settings: OrbSettings,
    pub tracking: TrackingParams,
    pub ctx: Arc<FrameCtx>,
    pub atlas: Atlas,
    /// Frame::nNextId(불러온 Atlas 에서, 아니면 0)
    pub next_frame: u64,
    /// 설정의 System.SaveAtlasToFile(끝낼 때 atlas_io::save 로)
    pub save_to: Option<String>,
}

impl SystemInit {
    /// System(단어장, 설정 파일, RGBD). w · h = 들어오는 영상 크기(원본은 첫 장면에서 경계를 잰다)
    pub fn open(settings_path: &str, w: i32, h: i32) -> Result<SystemInit, String> {
        let text = std::fs::read_to_string(settings_path).map_err(|e| format!("설정 {settings_path}: {e}"))?;
        Self::from_settings(OrbSettings::parse(&text)?, w, h)
    }

    pub fn from_settings(settings: OrbSettings, w: i32, h: i32) -> Result<SystemInit, String> {
        if settings.need_resize {
            return Err("Camera.newWidth · newHeight(영상 줄이기)는 안 옮김".into());
        }
        let tracking = settings.tracking();
        // ORBextractor(nFeatures, scaleFactor, nLevels, (int)iniThFAST, (int)minThFAST)
        let pyr = Pyramid::new(settings.nfeatures, settings.scale_factor, settings.nlevels, settings.ini_th_fast as i32, settings.min_th_fast as i32);
        let ctx = Arc::new(FrameCtx::new(tracking.camera(w, h), pyr, w, h));
        let (atlas, next_frame) = if settings.load_atlas.is_empty() {
            // Atlas(0): 지도 0 을 만들어 현재로
            (Atlas { maps: vec![Map::default()], ..Default::default() }, 0)
        } else {
            // 원본은 파일 이름 뒤에 ".osa" 를 붙임
            let p = format!("{}.osa", settings.load_atlas);
            let mut f = std::io::BufReader::new(std::fs::File::open(&p).map_err(|e| format!("Atlas {p}: {e}"))?);
            let (mut a, nf) = atlas_io::load(&mut f).map_err(|e| format!("Atlas {p}: {e}"))?;
            // 불러온 키프레임도 지금 설정의 카메라 · 층 표를 같이 씀(원본은 키프레임마다 저장값)
            for k in a.kfs.values_mut() {
                if k.ctx.cam != ctx.cam || k.ctx.pyr != ctx.pyr {
                    return Err("불러온 Atlas 의 카메라 · 층 표가 설정과 다름".into());
                }
                k.ctx = ctx.clone();
            }
            (a, nf)
        };
        let save_to = (!settings.save_atlas.is_empty()).then(|| format!("{}.osa", settings.save_atlas));
        Ok(SystemInit { settings, tracking, ctx, atlas, next_frame, save_to })
    }

    /// GrabImageRGBD 의 회색 변환: 채널 3 · 4 를 mbRGB 에 따라, 1 채널은 그대로
    pub fn gray(&self, im: &[u8], w: usize, h: usize, channels: usize, out: &mut Vec<u8>) {
        let order = match (channels, self.tracking.rgb) {
            (3, true) => ColorOrder::Rgb,
            (3, false) => ColorOrder::Bgr,
            (4, true) => ColorOrder::Rgba,
            (4, false) => ColorOrder::Bgra,
            _ => {
                out.clear();
                out.extend_from_slice(&im[..w * h]);
                return;
            }
        };
        img::cvt_gray(im, w, h, order, out);
    }

    /// GrabImageRGBD 의 깊이: 16 비트는 늘 convertTo(CV_32F, mDepthMapFactor)
    pub fn depth(&self, raw: &[u16], out: &mut Vec<f32>) {
        img::depth_to_f32(raw, self.tracking.depth_scale, out);
    }

    /// System::SaveAtlas(설정에 저장 파일이 있을 때)
    pub fn save_atlas(&self, atlas: &mut Atlas, next_frame: u64) -> Result<(), String> {
        let Some(p) = &self.save_to else { return Ok(()) };
        atlas_io::pre_save(atlas);
        let next_map = atlas.maps.iter().map(|m| m.id + 1).max().unwrap_or(0);
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).map_err(|e| format!("{p}: {e}"))?);
        atlas_io::save(atlas, next_frame, next_map, &mut f).map_err(|e| format!("{p}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_and_reload() {
        let dir = std::env::temp_dir().join(format!("orb3cv_sys_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = "%YAML:1.0\nFile.version: \"1.0\"\nCamera.type: \"PinHole\"\nCamera1.fx: 517.306408\nCamera1.fy: 516.469215\nCamera1.cx: 318.643040\nCamera1.cy: 255.313989\n\
Camera.width: 640\nCamera.height: 480\nCamera.fps: 30\nCamera.RGB: 1\nStereo.ThDepth: 40.0\nStereo.b: 0.07732\nRGBD.DepthMapFactor: 5000.0\n\
ORBextractor.nFeatures: 1000\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n";
        let atlas = dir.join("a");
        let y1 = dir.join("s1.yaml");
        std::fs::write(&y1, format!("{base}System.SaveAtlasToFile: \"{}\"\n", atlas.display())).unwrap();
        let s = SystemInit::open(y1.to_str().unwrap(), 640, 480).unwrap();
        assert_eq!(s.atlas.maps.len(), 1);
        assert_eq!(s.ctx.pyr.nlevels, 8);
        let mut d = Vec::new();
        s.depth(&[5000, 0, 1], &mut d);
        assert_eq!(d[0], 5000.0f32 * (1.0f32 / 5000.0f32));
        // 키프레임 하나 넣고 저장 → 다른 설정으로 불러오기
        let mut a = s.atlas.clone();
        let k = crate::orb3rs::types::KeyFrame { id: 0, ctx: s.ctx.clone(), ..Default::default() };
        a.kfs.insert(0, k);
        a.maps[0].kfs.push(0);
        a.next_kf = 1;
        s.save_atlas(&mut a, 7).unwrap();
        let y2 = dir.join("s2.yaml");
        std::fs::write(&y2, format!("{base}System.LoadAtlasFromFile: \"{}\"\n", atlas.display())).unwrap();
        let s2 = SystemInit::open(y2.to_str().unwrap(), 640, 480).unwrap();
        assert_eq!(s2.next_frame, 7);
        assert_eq!(s2.atlas.maps.len(), 2);
        assert!(Arc::ptr_eq(&s2.atlas.kfs[&0].ctx, &s2.ctx));
        std::fs::remove_dir_all(&dir).ok();
    }
}
