#include "tum_source.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

static std::vector<std::pair<double, std::string>> read_list(const std::string& path) {
    std::vector<std::pair<double, std::string>> out;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("읽을 수 없음: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t;
        std::string rel;
        if (ss >> t >> rel) out.emplace_back(t, rel);
    }
    return out;
}

TumSource::TumSource(const std::string& dir, double rate) : dir_(dir), rate_(rate) {
    rgb_ = read_list(dir + "/rgb.txt");
    depth_ = read_list(dir + "/depth.txt");
    if (rgb_.empty() || depth_.empty()) throw std::runtime_error("빈 TUM 폴더: " + dir);
    // 시퀀스별 보정값(TUM 문서). memmap tum.rs 와 같은 규칙: 폴더 이름으로 고름
    if (dir.find("freiburg1") != std::string::npos) {
        k_[0] = 517.3; k_[1] = 516.5; k_[2] = 318.6; k_[3] = 255.3;
    } else if (dir.find("freiburg2") != std::string::npos) {
        k_[0] = 520.9; k_[1] = 521.0; k_[2] = 325.1; k_[3] = 249.7;
    } else {
        k_[0] = 535.4; k_[1] = 539.2; k_[2] = 320.1; k_[3] = 247.6;
    }
    // 폴더에 calibration.txt(fx fy cx cy, ETH3D 형식)가 있으면 그 값을 씀
    std::ifstream cal(dir + "/calibration.txt");
    double c[4];
    if (cal && (cal >> c[0] >> c[1] >> c[2] >> c[3])) std::copy(c, c + 4, k_);
}

std::string TumSource::describe() const {
    std::ostringstream s;
    s << "TUM " << dir_ << " (" << rgb_.size() << " 장, 속도 " << (rate_ > 0 ? std::to_string(rate_) + "배" : std::string("최대")) << ")";
    return s.str();
}

bool TumSource::next(RgbdFrame& f) {
    using clk = std::chrono::steady_clock;
    while (i_ < rgb_.size()) {
        const auto& [t, rel] = rgb_[i_++];
        if (rate_ > 0) {
            if (!started_) {
                started_ = true;
                t_first_ = t;
                wall0_ = clk::now();
            } else {
                double ahead = (t - t_first_) / rate_ - std::chrono::duration<double>(clk::now() - wall0_).count();
                if (ahead < -0.1) {  // 처리가 밀렸으면 카메라처럼 지난 장면은 버림
                    ++skipped_;
                    continue;
                }
                if (ahead > 0) std::this_thread::sleep_for(std::chrono::duration<double>(ahead));
            }
        }
        auto c0 = clk::now();
        // 가장 가까운 시각의 깊이(memmap tum.rs kinect_depth 와 같은 짝)
        size_t j = std::lower_bound(depth_.begin(), depth_.end(), t, [](const auto& d, double v) { return d.first < v; }) - depth_.begin();
        if (j >= depth_.size()) j = depth_.size() - 1;
        else if (j > 0 && std::abs(depth_[j].first - t) >= std::abs(t - depth_[j - 1].first)) j = j - 1;
        f.t = t;
        f.bgr = cv::imread(dir_ + "/" + rel, cv::IMREAD_COLOR);
        f.depth = cv::imread(dir_ + "/" + depth_[j].second, cv::IMREAD_UNCHANGED);
        if (f.bgr.empty() || f.depth.empty() || f.depth.type() != CV_16UC1) throw std::runtime_error("장면 읽기 실패: " + rel);
        f.depth_per_m = 5000.f;
        f.fx = k_[0]; f.fy = k_[1]; f.cx = k_[2]; f.cy = k_[3];
        read_ms_ = std::chrono::duration<double, std::milli>(clk::now() - c0).count();
        return true;
    }
    return false;
}
