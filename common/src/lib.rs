//! ondevice 크레이트(slam · pipeline · sgview · tools · rsdriver)가 함께 쓰는 작은 조각.
//! - [`cli`]: 명령줄 읽기(`--키 값`, `--표시`), 묶음 인자 나누기
//! - [`control`]: 한 프로그램 안의 조종 신호(뷰어 리셋 · 수평 · 저장 버튼)
//! - [`threads`]: 쓸 수 있는 코어, 스레드 우선순위·코어 묶기, 이름 붙은 rayon 풀
//! - [`tum`]: TUM RGB-D 목록·위치 파일 읽기, 시퀀스별 카메라 보정값
//! - [`quat`]: 사원수 → 회전 행렬(f64)
//! - [`img`]: PNG · PLY 바이트 짜기(기억 폴더와 녹화 도구가 같은 인코더)
//! - `ort`(`onnx` 기능): 프로세스에 하나인 ONNX Runtime 환경
//! - `dxrt`(`dxrt` 기능): DX-M1 NPU 모델 실행과 DXRT 런타임 설정
pub mod cli;
pub mod control;
pub mod img;
pub mod log;
pub mod quat;
pub mod threads;
pub mod tum;

#[cfg(feature = "dxrt")]
pub mod dxrt;

#[cfg(feature = "onnx")]
pub mod ort;
