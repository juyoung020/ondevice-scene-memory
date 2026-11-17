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
}
pub fn take_reset() -> bool {
    RESET.swap(false, Ordering::AcqRel)
}

/// 수평: 지금 카메라 자세를 수평으로 삼아 지도를 다시 쌓음(물체 기억이 받음)
pub fn request_level() {
    LEVEL.store(true, Ordering::Release);
}
pub fn take_level() -> bool {
    LEVEL.swap(false, Ordering::AcqRel)
}

/// 저장: 메모리에 든 기억 폴더를 지금 디스크에 씀
pub fn request_save() {
    let _g = SAVE_WAKE.0.lock().unwrap();
    SAVE.store(true, Ordering::Release);
    SAVE_WAKE.1.notify_all();
}
pub fn take_save() -> bool {
    SAVE.swap(false, Ordering::AcqRel)
}
/// 저장 요청이 오거나 timeout 이 지날 때까지 잠듦(None = 요청까지). 돌려줌: 요청이 있었는지(받아 감)
pub fn wait_save(timeout: Option<std::time::Duration>) -> bool {
    let mut g = SAVE_WAKE.0.lock().unwrap();
    let end = timeout.map(|t| std::time::Instant::now() + t);
    while !SAVE.load(Ordering::Acquire) {
        match end {
            Some(e) => {
                let left = e.saturating_duration_since(std::time::Instant::now());
                if left.is_zero() {
                    return false;
                }
                g = SAVE_WAKE.1.wait_timeout(g, left).unwrap().0;
            }
            None => g = SAVE_WAKE.1.wait(g).unwrap(),
        }
    }
    take_save()
}
static SAVE_WAKE: (std::sync::Mutex<()>, std::sync::Condvar) = (std::sync::Mutex::new(()), std::sync::Condvar::new());

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
