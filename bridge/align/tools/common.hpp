// D455 도구 공통: rs2 ↔ fast_align 변환, 보정값 JSON 쓰기·읽기, 16 비트/RGB PNG, 시간 재기,
// 카메라 없이 rs2::align 을 돌리는 software_device 래퍼.
#pragma once
#include <librealsense2/hpp/rs_internal.hpp>
#include <librealsense2/rs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_align/fast_align.hpp"
#include "lodepng.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sched.h>
#endif

namespace tools {

static_assert(sizeof(rs2_intrinsics) == sizeof(fast_align::Intrinsics), "rs2_intrinsics 배치");
static_assert(sizeof(rs2_extrinsics) == sizeof(fast_align::Extrinsics), "rs2_extrinsics 배치");

inline void console_utf8() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
}

inline bool pin_cores(int first, int n) {
#ifdef _WIN32
    DWORD_PTR mask = 0;
    for (int i = 0; i < n; ++i) mask |= (DWORD_PTR)1 << (first + i);
    return SetProcessAffinityMask(GetCurrentProcess(), mask) && SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int i = 0; i < n; ++i) CPU_SET(first + i, &s);
    return sched_setaffinity(0, sizeof(s), &s) == 0;
#endif
}

inline fast_align::Intrinsics to_fa(const rs2_intrinsics& in) {
    fast_align::Intrinsics o;
    std::memcpy((void*)&o, &in, sizeof o);
    return o;
}
inline fast_align::Extrinsics to_fa(const rs2_extrinsics& in) {
    fast_align::Extrinsics o;
    std::memcpy((void*)&o, &in, sizeof o);
    return o;
}
inline rs2_intrinsics to_rs(const fast_align::Intrinsics& in) {
    rs2_intrinsics o;
    std::memcpy((void*)&o, &in, sizeof o);
    return o;
}
inline rs2_extrinsics to_rs(const fast_align::Extrinsics& in) {
    rs2_extrinsics o;
    std::memcpy((void*)&o, &in, sizeof o);
    return o;
}

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5));
    return v[i];
}

// ---- JSON(직접 씀: 우리 파일만 읽으므로 작은 읽기기로 충분) ----
inline std::string jnum(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}
inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '\n') {
            o += "\\n";
            continue;
        }
        o += c;
    }
    return o + "\"";
}
inline std::string intrin_json(const rs2_intrinsics& in) {
    std::ostringstream s;
    s << "{\"width\": " << in.width << ", \"height\": " << in.height << ", \"ppx\": " << jnum(in.ppx) << ", \"ppy\": " << jnum(in.ppy)
      << ", \"fx\": " << jnum(in.fx) << ", \"fy\": " << jnum(in.fy) << ", \"model\": " << (int)in.model << ", \"model_name\": "
      << jstr(rs2_distortion_to_string(in.model)) << ", \"coeffs\": [";
    for (int i = 0; i < 5; ++i) s << (i ? ", " : "") << jnum(in.coeffs[i]);
    s << "]}";
    return s.str();
}
inline std::string extrin_json(const rs2_extrinsics& e) {
    std::ostringstream s;
    s << "{\"rotation_col_major\": [";
    for (int i = 0; i < 9; ++i) s << (i ? ", " : "") << jnum(e.rotation[i]);
    s << "], \"translation_m\": [";
    for (int i = 0; i < 3; ++i) s << (i ? ", " : "") << jnum(e.translation[i]);
    s << "]}";
    return s.str();
}

// "key": 다음의 객체/값 문자열을 찾음(첫 번째)
inline std::string jfind(const std::string& js, const std::string& key, size_t from = 0) {
    size_t p = js.find("\"" + key + "\"", from);
    if (p == std::string::npos) throw std::runtime_error("JSON 에 " + key + " 없음");
    p = js.find(':', p) + 1;
    while (js[p] == ' ') ++p;
    if (js[p] == '{' || js[p] == '[') {
        char open = js[p], close = open == '{' ? '}' : ']';
        int depth = 0;
        size_t q = p;
        for (; q < js.size(); ++q) {
            if (js[q] == open) ++depth;
            if (js[q] == close && --depth == 0) break;
        }
        return js.substr(p, q - p + 1);
    }
    size_t q = js.find_first_of(",}\n", p);
    return js.substr(p, q - p);
}
inline std::vector<double> jarray(const std::string& a) {
    std::vector<double> v;
    std::string s = a;
    for (char& c : s)
        if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream is(s);
    double x;
    while (is >> x) v.push_back(x);
    return v;
}
inline rs2_intrinsics intrin_from_json(const std::string& o) {
    rs2_intrinsics in{};
    in.width = std::atoi(jfind(o, "width").c_str());
    in.height = std::atoi(jfind(o, "height").c_str());
    in.ppx = (float)std::atof(jfind(o, "ppx").c_str());
    in.ppy = (float)std::atof(jfind(o, "ppy").c_str());
    in.fx = (float)std::atof(jfind(o, "fx").c_str());
    in.fy = (float)std::atof(jfind(o, "fy").c_str());
    in.model = (rs2_distortion)std::atoi(jfind(o, "model").c_str());
    auto c = jarray(jfind(o, "coeffs"));
    for (int i = 0; i < 5 && i < (int)c.size(); ++i) in.coeffs[i] = (float)c[i];
    return in;
}
inline rs2_extrinsics extrin_from_json(const std::string& o) {
    rs2_extrinsics e{};
    auto r = jarray(jfind(o, "rotation_col_major"));
    auto t = jarray(jfind(o, "translation_m"));
    for (int i = 0; i < 9; ++i) e.rotation[i] = (float)r.at(i);
    for (int i = 0; i < 3; ++i) e.translation[i] = (float)t.at(i);
    return e;
}
inline std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("파일 없음: " + p);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// ---- PNG ----
inline bool write_png16(const std::string& path, const uint16_t* d, int w, int h) {
    std::vector<unsigned char> be((size_t)w * h * 2);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        be[2 * i] = (unsigned char)(d[i] >> 8);
        be[2 * i + 1] = (unsigned char)(d[i] & 0xff);
    }
    lodepng::State st;
    st.info_raw.colortype = LCT_GREY;
    st.info_raw.bitdepth = 16;
    st.info_png.color.colortype = LCT_GREY;
    st.info_png.color.bitdepth = 16;
    st.encoder.auto_convert = 0;
    st.encoder.zlibsettings.windowsize = 2048;
    std::vector<unsigned char> out;
    if (lodepng::encode(out, be, w, h, st)) return false;
    return lodepng::save_file(out, path) == 0;
}
inline bool write_png_rgb(const std::string& path, const uint8_t* rgb, int w, int h) {
    lodepng::State st;
    st.info_raw.colortype = LCT_RGB;
    st.info_raw.bitdepth = 8;
    st.info_png.color.colortype = LCT_RGB;
    st.info_png.color.bitdepth = 8;
    st.encoder.auto_convert = 0;
    st.encoder.zlibsettings.windowsize = 2048;
    std::vector<unsigned char> out;
    if (lodepng::encode(out, rgb, w, h, st)) return false;
    return lodepng::save_file(out, path) == 0;
}
inline bool read_png16(const std::string& path, std::vector<uint16_t>& d, int& w, int& h) {
    std::vector<unsigned char> raw;
    unsigned uw, uh;
    if (lodepng::decode(raw, uw, uh, path, LCT_GREY, 16)) return false;
    w = (int)uw;
    h = (int)uh;
    d.resize((size_t)w * h);
    for (size_t i = 0; i < d.size(); ++i) d[i] = (uint16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);
    return true;
}

// TUM 목록(시각 경로)
inline std::vector<std::pair<std::string, std::string>> read_list(const std::string& p) {
    std::vector<std::pair<std::string, std::string>> v;
    std::istringstream is(read_file(p));
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string t, f;
        if (ls >> t >> f) v.push_back({t, f});
    }
    return v;
}

// ---- 카메라 없이 rs2::align(SDK 실제 코드)을 돌림 ----
class SdkAligner {
public:
    SdkAligner(const rs2_intrinsics& di, const rs2_intrinsics& ci, const rs2_extrinsics& d2c, float depth_unit)
        : di_(di), ci_(ci), unit_(depth_unit), align_(RS2_STREAM_COLOR) {
        depth_s_ = std::make_unique<rs2::software_sensor>(dev_.add_sensor("Depth"));
        color_s_ = std::make_unique<rs2::software_sensor>(dev_.add_sensor("Color"));
        rs2_video_stream ds{RS2_STREAM_DEPTH, 0, 0, di.width, di.height, 30, 2, RS2_FORMAT_Z16, di};
        rs2_video_stream cs{RS2_STREAM_COLOR, 0, 1, ci.width, ci.height, 30, 3, RS2_FORMAT_RGB8, ci};
        dprof_ = depth_s_->add_video_stream(ds, true);
        cprof_ = color_s_->add_video_stream(cs, true);
        depth_s_->add_read_only_option(RS2_OPTION_DEPTH_UNITS, depth_unit);
        dprof_.register_extrinsics_to(cprof_, d2c);
        dev_.create_matcher(RS2_MATCHER_DLR_C);
        depth_s_->open(dprof_);
        color_s_->open(cprof_);
        depth_s_->start(sync_);
        color_s_->start(sync_);
        color_.assign((size_t)ci.width * ci.height * 3, 0);
        // syncer 가 깊이·컬러 짝을 내기 시작할 때까지 빈 장면을 넣음(처음 수십 장은 짝 없이 나옴)
        std::vector<uint16_t> z0((size_t)di.width * di.height, 0), o0((size_t)ci.width * ci.height);
        int ok = 0;
        for (int i = 0; i < 200 && ok < 3; ++i) {
            double ms;
            ok = process(z0.data(), o0.data(), ms) ? ok + 1 : 0;
        }
        if (ok < 3) throw std::runtime_error("software_device: 깊이·컬러 짝이 나오지 않음");
    }
    ~SdkAligner() {
        try {
            depth_s_->stop();
            color_s_->stop();
            depth_s_->close();
            color_s_->close();
        } catch (...) {
        }
    }
    // depth → out(컬러 크기). align_ms 에 align.process 시간(프레임 할당·복사 포함한 SDK 처리부)
    bool process(const uint16_t* depth, uint16_t* out, double& align_ms) {
        ++n_;
        const double ts = 1000.0 + n_ * 33.333;
        auto* dbuf = new uint16_t[(size_t)di_.width * di_.height];
        std::memcpy(dbuf, depth, sizeof(uint16_t) * di_.width * di_.height);
        auto* cbuf = new uint8_t[color_.size()];
        std::memcpy(cbuf, color_.data(), color_.size());
        depth_s_->on_video_frame({dbuf, [](void* p) { delete[] (uint16_t*)p; }, di_.width * 2, 2, ts,
                                  RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK, n_, dprof_, unit_});
        color_s_->on_video_frame({cbuf, [](void* p) { delete[] (uint8_t*)p; }, ci_.width * 3, 3, ts,
                                  RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK, n_, cprof_, 0.f});
        rs2::frameset fs;
        for (int tries = 0; tries < 10; ++tries) {
            if (!sync_.try_wait_for_frames(&fs, 200)) return false;
            if (fs.get_depth_frame() && fs.get_color_frame()) break;
        }
        if (!fs.get_depth_frame() || !fs.get_color_frame()) return false;
        const double t0 = now_ms();
        rs2::frameset al = align_.process(fs);
        align_ms = now_ms() - t0;
        rs2::depth_frame ad = al.get_depth_frame();
        if (!ad || ad.get_width() != ci_.width || ad.get_height() != ci_.height) return false;
        std::memcpy(out, ad.get_data(), sizeof(uint16_t) * ci_.width * ci_.height);
        return true;
    }

private:
    rs2_intrinsics di_, ci_;
    float unit_;
    rs2::software_device dev_;
    std::unique_ptr<rs2::software_sensor> depth_s_, color_s_;
    rs2::stream_profile dprof_, cprof_;
    rs2::syncer sync_;
    rs2::align align_;
    std::vector<uint8_t> color_;
    int n_ = 0;
};

// SDK 가 고른 정렬 구현(LOG_INFO "Using ... align implementation")을 잡음
struct AlignImplLog {
    static std::string& text() {
        static std::string s;
        return s;
    }
    static void install() {
        rs2::log_to_callback(RS2_LOG_SEVERITY_INFO, [](rs2_log_severity, rs2::log_message const& m) {
            std::string r = m.raw();
            if (r.find("align implementation") != std::string::npos) text() = r;
        });
    }
};

// 정렬 결과 두 장 비교
struct Diff {
    size_t both = 0, equal = 0, only_a = 0, only_b = 0, total = 0, edge = 0, edge_diff = 0;
    double sum_abs = 0;  // 둘 다 있고 다른 곳의 |차|(값 단위)
};
// edge: b(기준) 에서 3×3 안 최대-최소 > edge_jump 값이거나 0 이웃이 있는 화소
inline Diff compare(const uint16_t* a, const uint16_t* b, int w, int h, int edge_jump) {
    Diff d;
    d.total = (size_t)w * h;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            size_t i = (size_t)y * w + x;
            const uint16_t va = a[i], vb = b[i];
            bool edge = false;
            if (vb) {
                int lo = 65535, hi = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        int v = b[(size_t)yy * w + xx];
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                    }
                edge = lo == 0 || hi - lo > edge_jump;
            }
            if (va && vb) {
                ++d.both;
                if (va == vb) ++d.equal;
                else d.sum_abs += std::abs((int)va - (int)vb);
            } else if (va) {
                ++d.only_a;
            } else if (vb) {
                ++d.only_b;
            }
            if (edge) {
                ++d.edge;
                if (va != vb) ++d.edge_diff;
            }
        }
    return d;
}

}  // namespace tools
