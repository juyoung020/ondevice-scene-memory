//! cpp/orb3.cc(ORB-SLAM3 C 함수 래퍼)를 빌드하고 ORB-SLAM3 라이브러리에 링크한다.
//! ORB_SLAM3_DIR(기본 ~/ext/ORB_SLAM3) = cpp/ORB_SLAM3.patch 를 적용해 빌드한 원본 폴더(lib/libORB_SLAM3.so, Thirdparty/DBoW2 · g2o, include, Vocabulary).
//! PANGOLIN_DIR(기본 /usr/local) = ORB-SLAM3 헤더가 끌어오는 Pangolin 설치 폴더. 실행 파일에 라이브러리 폴더들을 rpath 로 넣는다.
fn main() {
    println!("cargo:rerun-if-changed=cpp/orb3.cc");
    println!("cargo:rerun-if-env-changed=ORB_SLAM3_DIR");
    println!("cargo:rerun-if-env-changed=PANGOLIN_DIR");
    // ORB-SLAM3 헤더가 끌어오는 Pangolin(설치 폴더: include · lib, 기본 /usr/local)
    let pango = std::env::var("PANGOLIN_DIR").unwrap_or_else(|_| "/usr/local".into());
    let orb = std::env::var("ORB_SLAM3_DIR").unwrap_or_else(|_| format!("{}/ext/ORB_SLAM3", std::env::var("HOME").unwrap_or_default()));
    println!("cargo:rustc-env=ORB_SLAM3_DIR={orb}");
    let lib = format!("{orb}/lib/libORB_SLAM3.so");
    if !std::path::Path::new(&lib).exists() {
        panic!("ORB-SLAM3 를 찾지 못함: {lib} (ORB_SLAM3_DIR 로 cpp/ORB_SLAM3.patch 를 적용해 빌드한 폴더를 알려 주기)");
    }
    // 원본 라이브러리를 빌드할 때와 같은 정의: 헤더 속 클래스 배치가 라이브러리와 같아야 함
    cc::Build::new()
        .cpp(true)
        .file("cpp/orb3.cc")
        .flag("-std=c++17")
        .define("COMPILEDWITHC11", None)
        .define("HAVE_EIGEN", None)
        .define("HAVE_GLEW", None)
        .define("PANGO_DEFAULT_WIN_URI", "\"x11\"")
        .define("_LINUX_", None)
        .flag_if_supported("-w")
        // 원본 라이브러리와 같은 -march=native: Eigen 정렬(헤더 속 클래스 배치)이 같아야 함
        .flag("-march=native")
        .include(&orb)
        .include(format!("{orb}/include"))
        .include(format!("{orb}/include/CameraModels"))
        .include(format!("{orb}/Thirdparty/Sophus"))
        .include(format!("{pango}/include"))
        .include("/usr/include/eigen3")
        .include("/usr/include/opencv4")
        .compile("orb3shim");
    let dirs = [format!("{orb}/lib"), format!("{orb}/Thirdparty/DBoW2/lib"), format!("{orb}/Thirdparty/g2o/lib"), format!("{pango}/lib")];
    for d in &dirs {
        println!("cargo:rustc-link-search=native={d}");
    }
    // 실행 파일(pipeline build.rs)이 rpath 로 넣을 폴더
    println!("cargo:rpath={}", dirs.join(":"));
    // System.h 가 끌어오는 뷰어 헤더의 인라인 코드가 Pangolin 심볼을 직접 씀
    for l in ["ORB_SLAM3", "DBoW2", "g2o", "opencv_core", "pango_display", "pango_opengl", "pango_windowing", "pango_core", "GLEW", "OpenGL", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
}
