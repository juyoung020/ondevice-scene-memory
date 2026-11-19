//! `orb3` 기능: cpp/orb3.cc(ORB-SLAM3 C 함수 래퍼)를 빌드하고 ORB-SLAM3 원본 라이브러리에 링크한다.
//! ORB_SLAM3_DIR(기본 ~/ext/ORB_SLAM3) = 원본을 빌드한 폴더(lib/libORB_SLAM3.so, Thirdparty/DBoW2 · g2o, include).
//! 실행할 때 세 라이브러리 폴더는 /etc/ld.so.conf.d/orb_slam3.conf 로 찾는다(docs/board_bridge.md).
fn main() {
    println!("cargo:rerun-if-changed=cpp/orb3.cc");
    println!("cargo:rerun-if-env-changed=ORB_SLAM3_DIR");
    if std::env::var_os("CARGO_FEATURE_ORB3").is_none() {
        return;
    }
    let orb = std::env::var("ORB_SLAM3_DIR").unwrap_or_else(|_| format!("{}/ext/ORB_SLAM3", std::env::var("HOME").unwrap_or_default()));
    let lib = format!("{orb}/lib/libORB_SLAM3.so");
    if !std::path::Path::new(&lib).exists() {
        panic!("ORB-SLAM3 를 찾지 못함: {lib} (ORB_SLAM3_DIR 로 빌드한 폴더를 알려 주기)");
    }
    // 원본 라이브러리를 빌드할 때와 같은 정의(ORB_SLAM3 · Pangolin 대상의 컴파일 정의): 헤더 속 클래스 배치가 라이브러리와 같아야 함
    cc::Build::new()
        .cpp(true)
        .file("cpp/orb3.cc")
        .flag("-std=c++17")
        .define("COMPILEDWITHC11", None)
        .define("HAVE_EIGEN", None)
        .define("HAVE_GLEW", None)
        .define("PANGO_DEFAULT_WIN_URI", "\"x11\"")
        .define("_LINUX_", None)
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter")
        .include(&orb)
        .include(format!("{orb}/include"))
        .include(format!("{orb}/include/CameraModels"))
        .include(format!("{orb}/Thirdparty/Sophus"))
        .include("/usr/include/eigen3")
        .include("/usr/include/opencv4")
        .compile("orb3shim");
    let dirs = [format!("{orb}/lib"), format!("{orb}/Thirdparty/DBoW2/lib"), format!("{orb}/Thirdparty/g2o/lib")];
    for d in &dirs {
        println!("cargo:rustc-link-search=native={d}");
    }
    // System.h 가 끌어오는 뷰어 헤더의 인라인 코드가 Pangolin 심볼을 직접 씀
    for l in ["ORB_SLAM3", "DBoW2", "g2o", "opencv_core", "pango_display", "pango_opengl", "pango_windowing", "pango_core", "GLEW", "OpenGL", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
}
