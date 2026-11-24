//! 위치 프로그램 갈아 끼우기: 같은 프로세스 처리 루프(pipeline `stream.rs`)는 이 trait 만 본다.
//! 백엔드는 장면을 받아 장면 스트림 항목(`OutFrame`: 위치 · 플래그 · 컬러 · 깊이, 그래프 블록: 루프 보정 · 지도 합치기로 움직인 노드)을
//! 넣은 차례대로 내놓는다. 형식과 좌표 약속은 docs/bridge.md "스트림 형식". 물체 기억 · 점 지도 · 뷰어는 백엔드와 상관없이 같다.
use crate::io::{FrameSource, Prep};
use crate::stream::OutFrame;
use crate::{Features, Frame};

pub trait SlamBackend: Send {
    /// 장면 하나를 위치 추적에 넣는다(출력은 `pop` 으로). false = 끝(받는 쪽이 닫힘 등)
    fn step(&mut self, f: Frame, read_ms: f64, prepped: Option<Features>) -> bool;
    /// 다음 출력(장면 · 그래프 블록 · 리셋 블록), 넣은 차례대로
    fn pop(&mut self) -> Option<OutFrame>;
    /// 지금까지 넣은 장면 수
    fn frames(&self) -> i64;
    /// 끝 요약(끝 그래프 · 통계). 부르는 쪽은 그 전에 `pop` 으로 남은 출력을 꺼내 둔다
    fn finish(self: Box<Self>) -> Result<i32, String>;
    /// 장면 사이 IMU 위치(IMU 융합 백엔드만)
    fn imu_ring(&self) -> Option<&std::sync::Arc<crate::imu::Ring>> {
        None
    }
}

/// RTAB-Map 포팅판(`app::Runner`)
impl SlamBackend for crate::app::Runner {
    fn step(&mut self, f: Frame, read_ms: f64, prepped: Option<Features>) -> bool {
        crate::app::Runner::step(self, f, read_ms, prepped)
    }
    fn pop(&mut self) -> Option<OutFrame> {
        self.writer.local().and_then(|l| l.q.pop_front())
    }
    fn frames(&self) -> i64 {
        self.n
    }
    fn finish(self: Box<Self>) -> Result<i32, String> {
        crate::app::Runner::finish(*self, 0)
    }
    fn imu_ring(&self) -> Option<&std::sync::Arc<crate::imu::Ring>> {
        self.imu_ring.as_ref()
    }
}

/// 같은 프로세스 처리 루프용 백엔드와 장면 입력: ORB-SLAM3 RGB-D(`orb3_backend`)
pub fn open(a: crate::app::Args) -> Result<(Box<dyn SlamBackend>, Box<dyn FrameSource>, Option<Prep>), String> {
    crate::orb3_backend::open(a)
}
