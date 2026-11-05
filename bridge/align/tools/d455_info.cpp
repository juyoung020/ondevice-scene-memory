// D455 정보를 JSON 으로: 장치·펌웨어·USB, 센서별 스트림 목록, IMU 유무, 깊이 단위, 고른 해상도의 깊이·컬러 내부 보정값과
// 깊이→컬러 외부 변환(스트리밍 없이 프로파일에서 읽음).
//   d455_info [--depth 640x480] [--color 640x480] [--fps 30] [> d455_info.json]
// 장치가 없으면 {"devices": [], "error": ...} 를 내고 끝 코드 2.
#include <cstdio>
#include <set>
#include <sstream>
#include <string>

#include "common.hpp"

using namespace tools;

static bool parse_wh(const std::string& s, int& w, int& h) { return std::sscanf(s.c_str(), "%dx%d", &w, &h) == 2; }

static std::string info(const rs2::device& d, rs2_camera_info k) {
    return d.supports(k) ? d.get_info(k) : "";
}

int main(int argc, char** argv) {
    console_utf8();
    int dw = 640, dh = 480, cw = 640, ch = 480, fps = 30;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string k = argv[i];
        if (k == "--depth") parse_wh(argv[++i], dw, dh);
        else if (k == "--color") parse_wh(argv[++i], cw, ch);
        else if (k == "--fps") fps = std::atoi(argv[++i]);
    }
    try {
        rs2::context ctx;
        auto list = ctx.query_devices();
        std::ostringstream js;
        js << "{\n  \"librealsense\": " << jstr(RS2_API_FULL_VERSION_STR) << ",\n";
        if (list.size() == 0) {
            js << "  \"devices\": [],\n  \"error\": \"장치 없음: D455 를 USB3 포트에 꽂고 다시\"\n}\n";
            std::fputs(js.str().c_str(), stdout);
            std::fprintf(stderr, "d455_info: 장치 없음\n");
            return 2;
        }
        js << "  \"devices\": [\n";
        for (uint32_t di = 0; di < list.size(); ++di) {
            rs2::device dev = list[di];
            js << "    {\n";
            js << "      \"name\": " << jstr(info(dev, RS2_CAMERA_INFO_NAME)) << ",\n";
            js << "      \"serial\": " << jstr(info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER)) << ",\n";
            js << "      \"firmware\": " << jstr(info(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION)) << ",\n";
            js << "      \"recommended_firmware\": " << jstr(info(dev, RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION)) << ",\n";
            js << "      \"usb\": " << jstr(info(dev, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)) << ",\n";
            js << "      \"product_line\": " << jstr(info(dev, RS2_CAMERA_INFO_PRODUCT_LINE)) << ",\n";
            js << "      \"product_id\": " << jstr(info(dev, RS2_CAMERA_INFO_PRODUCT_ID)) << ",\n";
            bool imu = false;
            float depth_unit = 0;
            rs2::video_stream_profile dprof, cprof;
            js << "      \"sensors\": [\n";
            auto sensors = dev.query_sensors();
            for (size_t si = 0; si < sensors.size(); ++si) {
                auto& s = sensors[si];
                js << "        {\"name\": " << jstr(s.supports(RS2_CAMERA_INFO_NAME) ? s.get_info(RS2_CAMERA_INFO_NAME) : "") << ", ";
                if (auto ds = s.as<rs2::depth_sensor>()) {
                    depth_unit = ds.get_depth_scale();
                    js << "\"depth_unit_m\": " << jnum(depth_unit) << ", ";
                }
                // 지금 옵션 값 몇 개
                js << "\"options\": {";
                bool first = true;
                for (rs2_option o : {RS2_OPTION_EMITTER_ENABLED, RS2_OPTION_LASER_POWER, RS2_OPTION_VISUAL_PRESET, RS2_OPTION_ENABLE_AUTO_EXPOSURE,
                                     RS2_OPTION_EXPOSURE, RS2_OPTION_GAIN, RS2_OPTION_GLOBAL_TIME_ENABLED}) {
                    if (!s.supports(o)) continue;
                    try {
                        js << (first ? "" : ", ") << jstr(rs2_option_to_string(o)) << ": " << jnum(s.get_option(o));
                        first = false;
                    } catch (...) {
                    }
                }
                js << "}, \"streams\": [";
                std::set<std::string> seen;
                bool firsts = true;
                for (auto& p : s.get_stream_profiles()) {
                    if (p.stream_type() == RS2_STREAM_GYRO || p.stream_type() == RS2_STREAM_ACCEL || p.stream_type() == RS2_STREAM_MOTION)
                        imu = true;
                    std::ostringstream one;
                    one << "\"" << rs2_stream_to_string(p.stream_type()) << " " << p.stream_index() << " " << rs2_format_to_string(p.format());
                    if (auto vp = p.as<rs2::video_stream_profile>()) {
                        one << " " << vp.width() << "x" << vp.height();
                        if (p.stream_type() == RS2_STREAM_DEPTH && p.format() == RS2_FORMAT_Z16 && vp.width() == dw && vp.height() == dh &&
                            p.fps() == fps)
                            dprof = vp;
                        if (p.stream_type() == RS2_STREAM_COLOR && p.format() == RS2_FORMAT_RGB8 && vp.width() == cw && vp.height() == ch &&
                            p.fps() == fps)
                            cprof = vp;
                    }
                    one << " @" << p.fps() << "\"";
                    if (seen.insert(one.str()).second) {
                        js << (firsts ? "" : ", ") << one.str();
                        firsts = false;
                    }
                }
                js << "]}" << (si + 1 < sensors.size() ? ",\n" : "\n");
            }
            js << "      ],\n";
            js << "      \"imu\": " << (imu ? "true" : "false") << ",\n";
            js << "      \"depth_unit_m\": " << jnum(depth_unit) << ",\n";
            js << "      \"chosen\": {\"depth\": \"" << dw << "x" << dh << "\", \"color\": \"" << cw << "x" << ch << "\", \"fps\": " << fps << "},\n";
            if (dprof && cprof) {
                js << "      \"depth_intrinsics\": " << intrin_json(dprof.get_intrinsics()) << ",\n";
                js << "      \"color_intrinsics\": " << intrin_json(cprof.get_intrinsics()) << ",\n";
                js << "      \"depth_to_color\": " << extrin_json(dprof.get_extrinsics_to(cprof)) << ",\n";
                fast_align::DepthAligner a(to_fa(dprof.get_intrinsics()), to_fa(cprof.get_intrinsics()),
                                           to_fa(dprof.get_extrinsics_to(cprof)), depth_unit);
                float fd[2], fc[2];
                fast_align::fov_deg(to_fa(dprof.get_intrinsics()), fd);
                fast_align::fov_deg(to_fa(cprof.get_intrinsics()), fc);
                js << "      \"fov_deg\": {\"depth\": [" << jnum(fd[0]) << ", " << jnum(fd[1]) << "], \"color\": [" << jnum(fc[0]) << ", "
                   << jnum(fc[1]) << "]},\n";
                js << "      \"sdk_align_fills_rect\": " << (a.uses_rect() ? "true" : "false") << "\n";
            } else {
                js << "      \"calibration_error\": \"고른 해상도·프레임의 Z16 깊이 또는 RGB8 컬러 스트림이 없음\"\n";
            }
            js << "    }" << (di + 1 < list.size() ? ",\n" : "\n");
        }
        js << "  ]\n}\n";
        std::fputs(js.str().c_str(), stdout);
    } catch (const rs2::error& e) {
        std::printf("{\"error\": %s}\n", jstr(std::string("librealsense: ") + e.what()).c_str());
        return 3;
    }
    return 0;
}
