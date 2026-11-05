// 깊이 → 컬러 정렬(depth-to-color registration). librealsense rs2::align 과 같은 결과를 더 적은 CPU 로 낸다.
//
// 입력: 깊이 u16 영상(깊이 내부 보정값), 컬러 내부 보정값, 깊이→컬러 외부 변환, 깊이 단위(m/값).
// 출력: 컬러 화면 크기의 u16 깊이. 값은 원래 깊이 값(깊이 카메라 기준 z, 단위 그대로)이고, 여러 깊이 화소가
//       한 컬러 화소에 겹치면 가장 가까운(작은) 값, 아무것도 안 오면 0.
//
// 방식(Method):
//   Sdk     — rs2::align 의 SIMD 경로(x86 SSE, ARM NEON: v2.59.1 src/proc/sse/sse-align.cpp,
//             src/proc/neon/neon-align.cpp)와 같은 규칙. Windows SDK 와 Pi 소스 빌드(BUILD_WITH_NEON 기본 켬)가 쓰는 경로.
//             깊이 화소의 왼쪽 위 모서리(x-0.5, y-0.5)를 컬러로 옮겨 반올림한 화소에 쓴다. 깊이의 각도당 화소가 컬러보다
//             적을 때(또는 640×480→640×360 같은 특수 해상도)만 오른쪽 아래 모서리까지 옮겨 그 사각형을 채운다.
//             컬러 왜곡은 MODIFIED_BROWN_CONRADY 일 때만 적용한다(SDK 와 같음, Options::color_distortion 으로 D455 의 IBC 도).
//   Generic — rs2::align 의 일반(비 SIMD) 경로(src/proc/align.cpp align_images + rsutil). 두 모서리를 늘 옮겨
//             사각형을 채우고, 사각형이 화면을 벗어나면 그 화소는 통째로 버리며, 컬러 왜곡을 rs2_project_point_to_pixel 대로 적용.
//
// 수치: 각 방식은 같은 순서의 float 연산을 쓰므로 그 플랫폼의 SDK 와 비트 단위로 같은 결과를 목표로 한다
//       (x86 은 SSE 순서 곱·더하기, aarch64 는 NEON 의 fma 순서). tests/ 의 이식본과 0 화소 차이로 확인한다.
#pragma once
#include <cstdint>
#include <memory>

namespace fast_align {

// rs2_distortion 과 같은 번호
enum class Distortion : int {
    None = 0,
    ModifiedBrownConrady = 1,
    InverseBrownConrady = 2,
    FTheta = 3,
    BrownConrady = 4,
    KannalaBrandt4 = 5,
};

// rs2_intrinsics 와 같은 배치(복사해 넘길 수 있게)
struct Intrinsics {
    int width = 0, height = 0;
    float ppx = 0, ppy = 0, fx = 0, fy = 0;
    Distortion model = Distortion::None;
    float coeffs[5] = {0, 0, 0, 0, 0};
};

// rs2_extrinsics 와 같은 배치: rotation 은 열 우선 3×3, translation 은 m
struct Extrinsics {
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float translation[3] = {0, 0, 0};
};

enum class Method { Sdk, Generic };

struct Options {
    Method method = Method::Sdk;
    int threads = 1;          // 줄 단위 나눔. 1 = 부른 스레드만
    bool force_rect = false;  // Sdk 방식에서 늘 두 모서리 사각형을 채움(구멍이 줄어듦, SDK 와 결과가 달라짐)
    // Sdk 방식에서 컬러 Inverse Brown-Conrady 왜곡도 rs2_project_point_to_pixel 식으로 적용(D455 컬러는 계수가 있음).
    // SDK SIMD 경로는 이 왜곡을 무시하므로 켜면 SDK 와 결과가 달라지고, 정렬된 깊이가 실제 컬러 화소 위치에 맞는다.
    bool color_distortion = false;
    // true: SDK 와 같은 순서의 float 연산(비트 단위 같은 결과). false: Sdk 방식에서 나눗셈 하나(1/Z)로 계산해
    // 더 빠르지만, 반올림 경계에 걸린 드문 화소가 옆 화소로 옮겨진다(tests 에서 비율을 잰다).
    bool exact = true;
};

// 보정값이 같은 동안 다시 쓰는 정렬기. 생성 때 모서리 방향 표를 만들고 스레드를 띄운다.
// process 는 한 번에 하나씩 부른다(같은 객체를 여러 스레드에서 동시에 부르지 않음).
class DepthAligner {
public:
    DepthAligner(const Intrinsics& depth, const Intrinsics& color, const Extrinsics& depth_to_color,
                 float depth_unit, const Options& opt = {});
    ~DepthAligner();
    DepthAligner(const DepthAligner&) = delete;
    DepthAligner& operator=(const DepthAligner&) = delete;

    // depth: depth.width × depth.height, out: color.width × color.height(먼저 0 으로 채움)
    void process(const uint16_t* depth, uint16_t* out);

    // Sdk 방식에서 사각형(두 모서리)을 채우는지. Generic 은 늘 true.
    bool uses_rect() const;
    // 마지막 process 가 겹침 때문에 한 스레드로 다시 돌았는지(여러 스레드일 때)
    bool last_fell_back_serial() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// 한 번만 쓸 때
void align_depth_to_color(const uint16_t* depth, uint16_t* out, const Intrinsics& depth_intrin,
                          const Intrinsics& color_intrin, const Extrinsics& depth_to_color, float depth_unit,
                          const Options& opt = {});

// rs2_fov 와 같은 계산(도)
void fov_deg(const Intrinsics& in, float out[2]);

}  // namespace fast_align
