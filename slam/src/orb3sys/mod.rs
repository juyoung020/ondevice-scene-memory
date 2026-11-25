//! ORB-SLAM3 System 흐름을 Rust 로 엮는 자리. `orb3_backend` 는 [`Orb3System`] 만 보고, 그 뒤가 원본(libORB_SLAM3, `crate::orb3::System`)이든
//! Rust 판이든 같게 부른다.
//!
//! 바꿔 끼우는 단위: 원본 C++ 와 Rust 판은 지도 자료(KeyFrame* · MapPoint* 대 `orb3rs::types::Atlas`)가 따로라, 한 실행 안에서
//! 함수 하나씩 원본으로 대신 부르려면 그 호출마다 자료를 양쪽으로 옮겨야 한다(TRACK 재생 도구의 `TrackOps` 가 그 방식: 덤프 값을 넘김).
//! 실제 실행에서는 System 통째(`crate::orb3::System`)가 원본 대신 판이고, Rust 판은 하위 단계가 모두 옮겨진 뒤 [`Orb3System`] 로 들어온다.
//! 스레드 · 멈춤 신호는 [`threads`](원본 교착을 구조로 없앤 판).
pub mod loopflow;
pub mod mapping;
pub mod rsys;
pub mod threads;

/// `crate::orb3::Track` 과 같은 뜻(추적 상태 · 카메라 → 세계 3×4 · 지도 id · 기준 키프레임)
pub use crate::orb3::Track;

/// 위치 백엔드가 쓰는 ORB-SLAM3 System 기능
pub trait Orb3System: Send {
    /// 컬러(3 채널) + 깊이(16 비트) + IMU 위쪽으로 한 장면 추적
    fn track(&mut self, rgb: &[u8], depth: &[u16], w: u32, h: u32, t: f64, up: Option<[f64; 3]>) -> Track;
    /// 앞에 물은 뒤로 큰 지도 바뀜(루프 · 합치기 · 옮겨 붙이기 · 전체 최적화)이 있었나
    fn map_changed(&mut self) -> bool;
    /// 키프레임(id, 지도 id, 카메라 → 세계). all = 모든 지도
    fn keyframes(&mut self, all: bool) -> Vec<(u64, u64, [f64; 12])>;
    /// 지도 from 을 지도 into 좌표로 옮겨 붙이기 부탁(s = into ← from)
    fn graft(&mut self, from: u64, into: u64, s: &[f64; 12]) -> bool;
    /// 지도 스레드가 쌓인 키프레임을 다 처리할 때까지(또는 멈춤 · 끝날 때까지) 기다림
    fn wait_idle(&mut self);
    /// 키프레임 id 의 지금 위치 · 지도 id(지워졌으면 부모를 따라감)
    fn kf_now(&mut self, id: u64) -> Option<(u64, [f64; 12])>;
}

impl Orb3System for crate::orb3::System {
    fn track(&mut self, rgb: &[u8], depth: &[u16], w: u32, h: u32, t: f64, up: Option<[f64; 3]>) -> Track {
        crate::orb3::System::track(self, rgb, depth, w, h, t, up)
    }
    fn map_changed(&mut self) -> bool {
        crate::orb3::System::map_changed(self)
    }
    fn keyframes(&mut self, all: bool) -> Vec<(u64, u64, [f64; 12])> {
        crate::orb3::System::keyframes(self, all)
    }
    fn graft(&mut self, from: u64, into: u64, s: &[f64; 12]) -> bool {
        crate::orb3::System::graft(self, from, into, s)
    }
    fn wait_idle(&mut self) {
        crate::orb3::System::wait_idle(self)
    }
    fn kf_now(&mut self, id: u64) -> Option<(u64, [f64; 12])> {
        crate::orb3::System::kf_now(self, id)
    }
}

/// 단어장 · 설정 파일로 System 을 연다(지금은 원본 판)
pub fn open(voc: &str, settings: &str) -> Result<Box<dyn Orb3System>, String> {
    Ok(Box::new(crate::orb3::System::new(voc, settings)?))
}
