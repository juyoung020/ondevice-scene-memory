//! 한 프로그램 안의 조종 신호: 뷰어 버튼(리셋 · 수평 · 저장)을 위치(slam) · 물체 기억(pipeline)이 장면마다 원자 값 하나로 읽는다.
//! `wire()` 를 부른 프로세스(`ondevice --serve`)에서만 뷰어가 신호를 켜고, 따로 띄운 뷰어(`ondevice view`)는 기억 폴더에 요청 파일을 남긴다.
use std::sync::atomic::{AtomicBool, Ordering};

static WIRED: AtomicBool = AtomicBool::new(false);
static RESET: AtomicBool = AtomicBool::new(false);
static LEVEL: AtomicBool = AtomicBool::new(false);
static SAVE: AtomicBool = AtomicBool::new(false);

/// 신호를 받는 쪽(위치 · 물체 기억)이 이 프로세스에 있음
pub fn wire() {
    WIRED.store(true, Ordering::Release);
}

pub fn wired() -> bool {
    WIRED.load(Ordering::Acquire)
}

/// 리셋: 위치 추적·지도를 처음부터(위치 프로그램이 받아 스트림에 리셋 블록을 보냄)
pub fn request_reset() {
    RESET.store(true, Ordering::Release);
    if let Some(f) = WAKE.get() {
        f();
    }
}
pub fn take_reset() -> bool {
    RESET.swap(false, Ordering::AcqRel)
}

/// 수평: 지금 카메라 자세를 수평으로 삼아 지도를 다시 쌓음(물체 기억이 받음)
pub fn request_level() {
    LEVEL.store(true, Ordering::Release);
    if let Some(f) = WAKE.get() {
        f();
    }
}
pub fn take_level() -> bool {
    LEVEL.swap(false, Ordering::AcqRel)
}

/// 저장: 메모리에 든 기억 폴더를 지금 디스크에 씀
pub fn request_save() {
    SAVE.store(true, Ordering::Release);
    if let Some(f) = WAKE.get() {
        f();
    }
}
pub fn take_save() -> bool {
    SAVE.swap(false, Ordering::AcqRel)
}

/// 신호가 오면 받는 쪽(처리 루프)을 깨울 함수: 한 번만 단다
static WAKE: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>> = std::sync::OnceLock::new();
pub fn on_signal(f: impl Fn() + Send + Sync + 'static) {
    let _ = WAKE.set(Box::new(f));
}

#[cfg(test)]
mod tests {
    #[test]
    fn take_once() {
        super::request_level();
        assert!(super::take_level() && !super::take_level());
        super::request_reset();
        assert!(super::take_reset() && !super::take_reset());
    }
}
