//! ORB-SLAM3 장소 인식 · 재위치 · 루프 · Atlas 합치기 몫을 Rust 로 비트 동일하게 옮긴 것(포터 LOOP).
//! 원본: UZ-SLAMLab/ORB_SLAM3 v1.0 + slam/cpp/ORB_SLAM3.patch. 대조 기록은 docs/orb3_port.md "포터 LOOP".
pub mod dbow2;
pub mod kfdb;
pub mod crand;
pub mod eig4;
pub mod sim3;
pub mod reloc;
pub mod detect;
