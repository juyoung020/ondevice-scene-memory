// 장면 입력의 공통 모양: 컬러(BGR8) + 컬러에 맞춘 깊이(u16) + 보정값 + 시각.
#pragma once
#include <opencv2/core.hpp>
#include <string>

struct RgbdFrame {
    double t = 0.0;          // 초
    cv::Mat bgr;             // CV_8UC3, BGR
    cv::Mat depth;           // CV_16UC1, 컬러 화소 격자에 맞춘 깊이, 0 = 없음
    float depth_per_m = 1000.f;  // 깊이 값 / 미터 (RealSense 1000, TUM 5000)
    double fx = 0, fy = 0, cx = 0, cy = 0;  // 컬러 카메라 보정값(왜곡 없음)
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    // 다음 장면. 끝이면 false.
    virtual bool next(RgbdFrame& f) = 0;
    // 장면 읽기에 쓴 시간(ms, 마지막 next 의 카메라 대기·파일 읽기·정렬)
    virtual double last_read_ms() const = 0;
    virtual std::string describe() const = 0;
};
