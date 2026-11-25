//! ORB-SLAM3(UZ-SLAMLab, 1.0, + cpp/ORB_SLAM3.patch) 를 Rust 로 옮기는 곳. 원본과 비트 동일이 원칙(docs/orb3_port.md).
//! 공통 자료 구조(이 파일)는 모든 포터가 같이 쓴다: Frame · KeyFrame · MapPoint · Map · Atlas.
//! 원본의 포인터(KeyFrame* · MapPoint*)는 id(KfId · MpId)로, 소유는 Atlas 의 표로 바꾼다.
//! 실수 자료형은 원본 그대로(Sophus::SE3f · Eigen float = f32, 크기 · 시그마 = f32, 시간 = f64).
pub mod types;
pub mod extract;
pub mod matcher;
pub mod frame;
pub mod stdsort;
// 포터 OPT: 포즈 최적화 · PnP
pub mod g2o_pose;
pub mod pinhole;
pub mod pose_opt;

pub use types::*;
