// RealSense 카메라(D455 등): 컬러 RGB8 + 깊이 Z16, 깊이를 컬러에 맞춤(depth_align). IMU 는 켜지 않는다.
#pragma once
#include "depth_align.h"
#include "frame_source.h"
#include <librealsense2/rs.hpp>
#include <memory>
#include <string>

class RealSenseSource : public FrameSource {
public:
    // serial: 빈 문자열이면 처음 찾은 카메라
    RealSenseSource(int w, int h, int fps, const std::string& aligner, const std::string& serial);
    ~RealSenseSource() override;
    bool next(RgbdFrame& f) override;
    double last_read_ms() const override { return read_ms_; }
    double last_wait_ms() const { return wait_ms_; }
    double last_align_ms() const { return align_ms_; }
    std::string describe() const override { return desc_; }

private:
    rs2::pipeline pipe_;
    std::unique_ptr<DepthAligner> aligner_;
    rs2_intrinsics color_k_{};
    float depth_per_m_ = 1000.f;
    double read_ms_ = 0, wait_ms_ = 0, align_ms_ = 0;
    std::string desc_;
};
