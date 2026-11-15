//! 깊이 → 컬러 정렬 라이브러리 [`fast_align`]과 도구 공통 코드.
//! - [`fast_align`]: librealsense rs2::align 과 같은 결과(비트 단위)를 더 적은 CPU 로
//! - [`rsref`]: 비교 기준으로 옮겨 온 librealsense v2.59.1 정렬 코드(Apache-2.0)
//! - [`scenes`]: 시험·벤치용 합성 장면과 D455 가정 보정값
//! - [`pb`]: ONNX 모델 손질용 작은 protobuf 읽기·쓰기(`npu_prep`)
//! - [`common`]: JSON·PNG·TUM 목록·시간·코어 묶기·결과 비교
pub mod common;
pub mod fast_align;
pub mod pb;
pub mod rsref;
pub mod scenes;

