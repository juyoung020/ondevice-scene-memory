//! slam_bridge: RGB-D 입력 → 위치 추적 + 지도 → 장면 스트림(memmap --stream). 본체는 `slam::app`(memmap --slam 도 같은 것을 부름).
//!
//!   slam_bridge --tum <폴더> [--rate 1] | memmap --stream - ...
//!   slam_bridge --realsense [--align rs2] | memmap --stream - ...
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let code = match slam::app::run(&argv, None) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("slam_bridge 오류: {e}");
            1
        }
    };
    std::process::exit(code);
}
