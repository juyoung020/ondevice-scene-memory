//! 작은 기록기: 진행 글은 `common::info!`(eprintln! 과 같은 글), `--quiet` 이면 끔. 줄 하나를 한 번에 써서 스레드끼리 섞이지 않는다.
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

static QUIET: AtomicBool = AtomicBool::new(false);

/// 진행 글 끄기·켜기(`--quiet`)
pub fn set_quiet(q: bool) {
    QUIET.store(q, Relaxed);
}

pub fn quiet() -> bool {
    QUIET.load(Relaxed)
}

/// 줄 하나(끝에 줄바꿈 붙임)를 표준 오류로 한 번에
pub fn line(args: std::fmt::Arguments) {
    use std::io::Write;
    thread_local! { static BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) }; }
    BUF.with(|b| {
        let mut b = b.borrow_mut();
        b.clear();
        let _ = b.write_fmt(args);
        b.push(b'\n');
        let _ = std::io::stderr().lock().write_all(&b);
    });
}

/// 진행 글(eprintln! 자리): `--quiet` 이면 찍지 않음
#[macro_export]
macro_rules! info {
    ($($t:tt)*) => {
        if !$crate::log::quiet() {
            $crate::log::line(format_args!($($t)*));
        }
    };
}
