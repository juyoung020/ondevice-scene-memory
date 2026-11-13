#include "realsense_source.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <sstream>
#include <stdexcept>

RealSenseSource::RealSenseSource(int w, int h, int fps, const std::string& aligner, const std::string& serial, double exposure_ms, int gain) {
    rs2::config cfg;
    if (!serial.empty()) cfg.enable_device(serial);
    cfg.enable_stream(RS2_STREAM_COLOR, w, h, RS2_FORMAT_RGB8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, fps);
    rs2::pipeline_profile prof = pipe_.start(cfg);
    rs2::device dev = prof.get_device();
    for (rs2::sensor& s : dev.query_sensors()) {
        // 컬러 자동 노출이 어두울 때 장면 수를 낮추지 않게(위치 추적은 고른 30 장/초가 낫다)
        if (s.is<rs2::color_sensor>() && s.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)) s.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.f);
        // 컬러 노출을 짧게 고정(--exposure ms): 어두운 곳에서 자동 노출이 길어져 움직일 때 번지는 것을 막는다. 밝기는 감도(--gain)로 채움
        if (s.is<rs2::color_sensor>() && exposure_ms > 0) {
            s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.f);
            s.set_option(RS2_OPTION_EXPOSURE, static_cast<float>(exposure_ms * 10.0)); // 단위 0.1 ms
            if (gain >= 0) s.set_option(RS2_OPTION_GAIN, static_cast<float>(gain));
        } else if (s.is<rs2::color_sensor>()) {
            s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);
        }
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
    // 자동 노출이 자리 잡을 때까지 첫 장면들은 버린다(어두운 첫 장면이 위치 추적의 기준이 되면 바로 놓침)
    rs2::frameset last;
    for (int i = 0; i < 30; ++i) last = pipe_.wait_for_frames(5000);
    if (rs2::video_frame c = last.get_color_frame()) {
        if (c.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE))
            s << ", 컬러 노출 " << c.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE) / 10.0 << " ms";
        if (c.supports_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL)) s << " 감도 " << c.get_frame_metadata(RS2_FRAME_METADATA_GAIN_LEVEL);
        s << (exposure_ms > 0 ? "(고정)" : "(자동)");
    }
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
    // 장면마다 새 버퍼에 담는다: 쓰기 큐·RTAB-Map 이 들고 있는 앞 장면과 메모리를 나누지 않게
    f.rgb = rgb.clone();
    cv::Mat bgr;
    cv::cvtColor(f.rgb, bgr, cv::COLOR_RGB2BGR);
    f.bgr = bgr;
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
