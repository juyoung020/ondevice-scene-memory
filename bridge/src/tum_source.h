// TUM RGB-D 폴더 재생(rgb.txt · depth.txt, 깊이 5000/m): 카메라 없이 시험할 때.
#pragma once
#include "frame_source.h"
#include <chrono>
#include <string>
#include <utility>
#include <vector>

class TumSource : public FrameSource {
public:
    // rate: 0 = 최대 속도(모든 장면), 1 = 찍은 시각 간격대로(밀리면 건너뜀), 2 = 두 배 빠르게 ...
    TumSource(const std::string& dir, double rate);
    bool next(RgbdFrame& f) override;
    double last_read_ms() const override { return read_ms_; }
    std::string describe() const override;
    size_t size() const { return rgb_.size(); }
    size_t skipped() const { return skipped_; }

private:
    std::string dir_;
    double rate_;
    std::vector<std::pair<double, std::string>> rgb_, depth_;
    size_t i_ = 0, skipped_ = 0;
    double k_[4];
    double read_ms_ = 0;
    bool started_ = false;
    double t_first_ = 0;
    std::chrono::steady_clock::time_point wall0_;
};
