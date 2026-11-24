//! 위치 백엔드(slam: ORB-SLAM3)의 공유 라이브러리 폴더를 실행 파일 rpath 로(설치 없이 실행)
fn main() {
    if let Ok(p) = std::env::var("DEP_ORB3_RPATH") {
        // DT_RPATH(옛 방식): 라이브러리가 끌어오는 라이브러리(Pangolin 끼리)도 같은 폴더에서 찾게
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{p}");
    }
}
