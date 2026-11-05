#include "realsense_source.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <sstream>
#include <stdexcept>

RealSenseSource::RealSenseSource(int w, int h, int fps, const std::string& aligner, const std::string& serial) {
    rs2::config cfg;
    if (!serial.empty()) cfg.enable_device(serial);
    cfg.enable_stream(RS2_STREAM_COLOR, w, h, RS2_FORMAT_RGB8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, fps);
    rs2::pipeline_profile prof = pipe_.start(cfg);
    rs2::device dev = prof.get_device();
    for (rs2::sensor& s : dev.query_sensors()) {
        if (rs2::depth_sensor ds = s.as<rs2::depth_sensor>()) {
            depth_per_m_ = 1.0f / ds.get_depth_scale();
            if (ds.supports(RS2_OPTION_EMITTER_ENABLED)) ds.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f);
        }
    }
    auto cp = prof.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    color_k_ = cp.get_intrinsics();
    aligner_ = make_aligner(aligner);
    std::ostringstream s;
    s << "RealSense " << dev.get_info(RS2_CAMERA_INFO_NAME) << " " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
      << " 펌웨어 " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << ", " << w << "x" << h << "@" << fps
      << ", 깊이 " << depth_per_m_ << "/m, 컬러 보정 fx " << color_k_.fx << " fy " << color_k_.fy << " cx " << color_k_.ppx
      << " cy " << color_k_.ppy << " 왜곡 모델 " << rs2_distortion_to_string(color_k_.model) << ", 정렬 " << aligner_->name();
    desc_ = s.str();
}

RealSenseSource::~RealSenseSource() {
    try {
        pipe_.stop();
    } catch (...) {
    }
}

bool RealSenseSource::next(RgbdFrame& f) {
    using clk = std::chrono::steady_clock;
    auto c0 = clk::now();
    rs2::frameset fs = pipe_.wait_for_frames(5000);
    auto c1 = clk::now();
    rs2::video_frame color = fs.get_color_frame();
    if (!color || !fs.get_depth_frame()) return next(f);
    f.depth = aligner_->align(fs);
    auto c2 = clk::now();
    cv::Mat rgb(color.get_height(), color.get_width(), CV_8UC3, const_cast<void*>(color.get_data()), (size_t)color.get_stride_in_bytes());
    cv::cvtColor(rgb, f.bgr, cv::COLOR_RGB2BGR);
    f.t = color.get_timestamp() * 1e-3;
    f.depth_per_m = depth_per_m_;
    f.fx = color_k_.fx;
    f.fy = color_k_.fy;
    f.cx = color_k_.ppx;
    f.cy = color_k_.ppy;
    wait_ms_ = std::chrono::duration<double, std::milli>(c1 - c0).count();
    align_ms_ = std::chrono::duration<double, std::milli>(c2 - c1).count();
    read_ms_ = std::chrono::duration<double, std::milli>(clk::now() - c1).count();
    return true;
}
