//! RealSense D435 드라이버: librealsense2 v2.59.1 중 우리가 쓰는 경로만 Rust 로(리눅스 V4L2·uvcvideo + UVC 확장 단위).
//! - [`calib`]: 보정표(계수표·RGB 표·새 깊이 표) 해석, 깊이→컬러, 깊이 단위 — librealsense 와 같은 f32 연산
//! - [`hwm`]: HW 모니터 명령(깊이 XU 제어 1)
//! - [`format`]: YUYV → RGB8(librealsense SSE·NEON·일반 규칙), Z16
//! - [`meta`]: UVC 메타데이터(장치 시각, 컬러 노출·감도)
//! - [`sync`]: 컬러·깊이 짝(librealsense syncer 규칙)
//! - `camera`(리눅스): sysfs 로 찾기, mmap 버퍼 흐름, 제어, 정렬(`tools::fast_align`)
//! 원본 출처: librealsense (Apache-2.0) `src/ds/d400/*`, `src/hw-monitor.cpp`, `src/linux/backend-v4l2.cpp`, `src/metadata.h`, `src/sync.cpp`, `src/proc/*`.
pub mod calib;
pub mod format;
pub mod hwm;
pub mod meta;
pub mod sync;

#[cfg(target_os = "linux")]
pub mod camera;
#[cfg(target_os = "linux")]
pub mod v4l2;

pub use calib::{Extrinsics, Intrinsics};
pub use format::Yuy2Rule;
