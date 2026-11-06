// RealSense 녹화: 컬러(RGB8)·원래 깊이(정렬 전)·rs2::align 결과·fast_align 결과를 TUM 형식 폴더로.
//   rs_rec <출력 폴더> [초=10] [--emitter on|off] [--preset default|hand|high_accuracy|high_density|medium_density]
//            [--depth 640x480] [--color 640x480] [--fps 30] [--threads 1] [--skip 30]
// 폴더:
//   rgb/<시각>.png        컬러 RGB8                      rgb.txt
//   depth/<시각>.png      fast_align 정렬 깊이(컬러 크기)  depth.txt   ← TUM 짝(association.txt)
//   depth_raw/<시각>.png  정렬 전 원래 깊이               depth_raw.txt
//   depth_rs/<시각>.png   rs2::align 결과                depth_rs.txt
//   calib.json(보정값·설정·rgbd2tum 명령), rec_stats.json(장면 수·시간)
// 깊이 PNG 값은 카메라 깊이 단위 그대로(D400 기본 1 mm = 1000/m). memmap 은 rgbd2tum 으로 TUM fr3(5000/m)로 바꿔 읽는다.
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"

using namespace tools;
namespace fs = std::filesystem;

struct Job {
    std::string tc, td;
    int cw, ch, dw, dh;
    std::vector<uint8_t> rgb;
    std::vector<uint16_t> raw, rs, ours;
};

class Writer {
public:
    Writer(const fs::path& out, int n, size_t cap) : out_(out), cap_(cap) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Writer() { finish(); }
    bool push(Job&& j) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.size() >= cap_) return false;
        q_.push_back(std::move(j));
        cv_.notify_one();
        return true;
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_)
            if (t.joinable()) t.join();
    }
    std::atomic<int> failed{0};

private:
    void loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if (q_.empty()) return;
                j = std::move(q_.front());
                q_.pop_front();
            }
            bool ok = write_png_rgb((out_ / "rgb" / (j.tc + ".png")).string(), j.rgb.data(), j.cw, j.ch);
            ok &= write_png16((out_ / "depth" / (j.td + ".png")).string(), j.ours.data(), j.cw, j.ch);
            ok &= write_png16((out_ / "depth_rs" / (j.td + ".png")).string(), j.rs.data(), j.cw, j.ch);
            ok &= write_png16((out_ / "depth_raw" / (j.td + ".png")).string(), j.raw.data(), j.dw, j.dh);
            if (!ok) ++failed;
        }
    }
    fs::path out_;
    size_t cap_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<Job> q_;
    bool stop_ = false;
};

static bool parse_wh(const std::string& s, int& w, int& h) { return std::sscanf(s.c_str(), "%dx%d", &w, &h) == 2; }

static std::string tstr(double ms) {
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", ms / 1000.0);
    return b;
}

int main(int argc, char** argv) {
    console_utf8();
    if (argc < 2) {
        std::fprintf(stderr,
                     "rs_rec <출력 폴더> [초=10] [--emitter on|off] [--preset 이름] [--depth WxH] [--color WxH] [--fps N] [--threads N] [--skip N]\n");
        return 1;
    }
    fs::path out = argv[1];
    double secs = 10;
    std::string emitter = "on", preset;
    int dw = 640, dh = 480, cw = 640, ch = 480, fps = 30, threads = 1, skip = 30;
    int ai = 2;
    if (argc > 2 && argv[2][0] != '-') secs = std::atof(argv[ai++]);
    for (int i = ai; i + 1 < argc; ++i) {
        std::string k = argv[i];
        if (k == "--emitter") emitter = argv[++i];
        else if (k == "--preset") preset = argv[++i];
        else if (k == "--depth") parse_wh(argv[++i], dw, dh);
        else if (k == "--color") parse_wh(argv[++i], cw, ch);
        else if (k == "--fps") fps = std::atoi(argv[++i]);
        else if (k == "--threads") threads = std::atoi(argv[++i]);
        else if (k == "--skip") skip = std::atoi(argv[++i]);
    }
    AlignImplLog::install();
    try {
        rs2::context ctx;
        if (ctx.query_devices().size() == 0) {
            std::fprintf(stderr, "rs_rec: 장치 없음(RealSense 를 USB3 포트에 꽂고 다시)\n");
            return 2;
        }
        for (const char* sub : {"rgb", "depth", "depth_raw", "depth_rs"}) fs::create_directories(out / sub);

        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH, dw, dh, RS2_FORMAT_Z16, fps);
        cfg.enable_stream(RS2_STREAM_COLOR, cw, ch, RS2_FORMAT_RGB8, fps);
        rs2::pipeline pipe(ctx);
        rs2::pipeline_profile prof = pipe.start(cfg);
        rs2::device dev = prof.get_device();
        rs2::depth_sensor ds = dev.first<rs2::depth_sensor>();
        if (!preset.empty()) {
            int pv = -1;
            for (int p = 0; p < RS2_RS400_VISUAL_PRESET_COUNT; ++p) {
                std::string n = rs2_rs400_visual_preset_to_string((rs2_rs400_visual_preset)p);
                for (auto& c : n) c = (char)std::tolower((unsigned char)c);
                for (auto& c : n)
                    if (c == ' ') c = '_';
                if (n == preset) pv = p;
            }
            if (pv < 0) throw std::runtime_error("모르는 프리셋: " + preset);
            ds.set_option(RS2_OPTION_VISUAL_PRESET, (float)pv);
        }
        if (ds.supports(RS2_OPTION_EMITTER_ENABLED)) ds.set_option(RS2_OPTION_EMITTER_ENABLED, emitter == "off" ? 0.f : 1.f);
        for (auto& s : dev.query_sensors())
            if (s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) s.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.f);
        // 컬러 자동 노출이 어두울 때 프레임 수를 낮추지 않게(위치 추적은 고른 30 장/초가 낫다)
        for (auto& s : dev.query_sensors())
            if (s.is<rs2::color_sensor>() && s.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)) s.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.f);

        auto dprof = prof.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        auto cprof = prof.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        const rs2_intrinsics di = dprof.get_intrinsics(), ci = cprof.get_intrinsics();
        const rs2_extrinsics e = dprof.get_extrinsics_to(cprof);
        const float unit = ds.get_depth_scale();

        rs2::align align(RS2_STREAM_COLOR);
        fast_align::Options fo;
        fo.threads = threads;
        fast_align::DepthAligner ours(to_fa(di), to_fa(ci), to_fa(e), unit, fo);
        Writer writer(out, std::max(2u, std::thread::hardware_concurrency() / 4), 240);

        std::printf("rs_rec: %s %s, 깊이 %dx%d · 컬러 %dx%d @%d, 레이저 %s, 프리셋 %s, 깊이 단위 %g m, %g 초 → %s\n",
                    dev.get_info(RS2_CAMERA_INFO_NAME), dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER), dw, dh, cw, ch, fps, emitter.c_str(),
                    preset.empty() ? "(그대로)" : preset.c_str(), unit, secs, out.string().c_str());
        std::ofstream frgb(out / "rgb.txt"), fdep(out / "depth.txt"), fraw(out / "depth_raw.txt"), frs(out / "depth_rs.txt"),
            fas(out / "association.txt");
        frgb << "# color RGB8\n# timestamp filename\n";
        fdep << "# depth aligned to color by fast_align (units " << unit << " m)\n# timestamp filename\n";
        fraw << "# raw depth (units " << unit << " m)\n# timestamp filename\n";
        frs << "# depth aligned by rs2::align (units " << unit << " m)\n# timestamp filename\n";

        std::vector<double> t_rs, t_ours;
        int n = 0, dropped = 0, gaps = 0, skipped = 0;
        long long last_fn = -1;
        size_t diff_px = 0, all_px = 0;
        double center_m = 0, center_valid = 0, valid_ratio = 0;
        const double t_start = now_ms();
        while (now_ms() - t_start < (secs + skip / (double)fps) * 1000.0) {
            rs2::frameset fsr = pipe.wait_for_frames(5000);
            rs2::depth_frame df = fsr.get_depth_frame();
            rs2::video_frame cf = fsr.get_color_frame();
            if (!df || !cf) continue;
            if (skipped < skip) {  // 자동 노출이 자리 잡을 때까지
                ++skipped;
                continue;
            }
            long long fn = (long long)df.get_frame_number();
            if (last_fn >= 0 && fn > last_fn + 1) gaps += (int)(fn - last_fn - 1);
            last_fn = fn;
            Job j;
            j.cw = cw;
            j.ch = ch;
            j.dw = dw;
            j.dh = dh;
            j.tc = tstr(cf.get_timestamp());
            j.td = tstr(df.get_timestamp());
            const uint16_t* zp = (const uint16_t*)df.get_data();
            j.raw.assign(zp, zp + (size_t)dw * dh);
            // 화면 가운데 40×40 깊이 중앙값과 전체 유효 비율(최소 거리 확인용: 벽에 다가가며 값이 사라지는 거리)
            {
                std::vector<double> c;
                for (int y = dh / 2 - 20; y < dh / 2 + 20; ++y)
                    for (int x = dw / 2 - 20; x < dw / 2 + 20; ++x)
                        if (zp[(size_t)y * dw + x]) c.push_back(zp[(size_t)y * dw + x] * unit);
                size_t valid = 0;
                for (size_t i = 0; i < (size_t)dw * dh; ++i) valid += zp[i] != 0;
                center_m = c.size() > 200 ? median(c) : 0.0;
                center_valid = c.size() / 1600.0;
                valid_ratio = (double)valid / ((size_t)dw * dh);
            }
            const uint8_t* cp = (const uint8_t*)cf.get_data();
            j.rgb.assign(cp, cp + (size_t)cw * ch * 3);
            double t0 = now_ms();
            rs2::frameset al = align.process(fsr);
            double t1 = now_ms();
            rs2::depth_frame ad = al.get_depth_frame();
            const uint16_t* ap = (const uint16_t*)ad.get_data();
            j.rs.assign(ap, ap + (size_t)cw * ch);
            j.ours.resize((size_t)cw * ch);
            double t2 = now_ms();
            ours.process(j.raw.data(), j.ours.data());
            double t3 = now_ms();
            if (n > 0) {
                t_rs.push_back(t1 - t0);
                t_ours.push_back(t3 - t2);
            }
            for (size_t i = 0; i < j.ours.size(); ++i) diff_px += j.ours[i] != j.rs[i];
            all_px += j.ours.size();
            const std::string tc = j.tc, td = j.td;
            ++n;
            if (!writer.push(std::move(j))) {  // 쓰기가 밀리면 이 장면은 목록에도 넣지 않음
                ++dropped;
                continue;
            }
            frgb << tc << " rgb/" << tc << ".png\n";
            fdep << td << " depth/" << td << ".png\n";
            fraw << td << " depth_raw/" << td << ".png\n";
            frs << td << " depth_rs/" << td << ".png\n";
            fas << tc << " rgb/" << tc << ".png " << td << " depth/" << td << ".png\n";
            if (n % fps == 0)
                std::printf("  %d 장, rs2::align %.2f ms · fast_align %.2f ms(중앙), 쓰기 밀림 %d | 가운데 깊이 %.3f m(유효 %.0f %%), 전체 유효 %.0f %%\n",
                            n, median(t_rs), median(t_ours), dropped, center_m, 100 * center_valid, 100 * valid_ratio);
        }
        pipe.stop();
        writer.finish();

        const std::string impl = AlignImplLog::text();
        char cmd[512];
        std::snprintf(cmd, sizeof cmd, "rgbd2tum %s %s_freiburg3 %.4f %.4f %.4f %.4f %.1f", out.string().c_str(), out.string().c_str(), ci.fx,
                      ci.fy, ci.ppx, ci.ppy, 1.0 / unit);
        std::ofstream cj(out / "calib.json");
        cj << "{\n  \"librealsense\": " << jstr(RS2_API_FULL_VERSION_STR) << ",\n  \"device\": " << jstr(dev.get_info(RS2_CAMERA_INFO_NAME))
           << ",\n  \"serial\": " << jstr(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
           << ",\n  \"firmware\": " << jstr(dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION))
           << ",\n  \"usb\": " << jstr(dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) ? dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "")
           << ",\n  \"depth_intrinsics\": " << intrin_json(di) << ",\n  \"color_intrinsics\": " << intrin_json(ci)
           << ",\n  \"depth_to_color\": " << extrin_json(e) << ",\n  \"depth_unit_m\": " << jnum(unit)
           << ",\n  \"depth_png_per_m\": " << jnum(1.0 / unit) << ",\n  \"settings\": {\"fps\": " << fps << ", \"emitter\": " << jstr(emitter)
           << ", \"preset\": " << jstr(preset.empty() ? "unchanged" : preset) << ", \"skip\": " << skip << "},\n  \"sdk_align_impl\": " << jstr(impl)
           << ",\n  \"rgbd2tum\": " << jstr(cmd) << "\n}\n";
        std::ofstream sj(out / "rec_stats.json");
        sj << "{\n  \"frames\": " << n << ",\n  \"write_dropped\": " << dropped << ",\n  \"write_failed\": " << writer.failed.load()
           << ",\n  \"depth_frame_gaps\": " << gaps << ",\n  \"rs_align_ms_median\": " << jnum(median(t_rs))
           << ",\n  \"rs_align_ms_p90\": " << jnum(percentile(t_rs, 0.9)) << ",\n  \"fast_align_ms_median\": " << jnum(median(t_ours))
           << ",\n  \"fast_align_ms_p90\": " << jnum(percentile(t_ours, 0.9)) << ",\n  \"fast_align_threads\": " << threads
           << ",\n  \"pixels_differ_from_rs\": " << diff_px << ",\n  \"pixels_total\": " << all_px << "\n}\n";
        std::printf("끝: %d 장(쓰기 밀려 빠진 %d, 깊이 프레임 번호 건너뜀 %d), rs2::align %.2f ms · fast_align %.2f ms(중앙)\n", n, dropped, gaps,
                    median(t_rs), median(t_ours));
        std::printf("SDK 정렬 구현: %s\n", impl.empty() ? "(로그 없음)" : impl.c_str());
        std::printf("rs2::align 과 다른 화소: %zu / %zu\n", diff_px, all_px);
        std::printf("memmap 용 TUM fr3 변환: %s\n", cmd);
        if (dropped) std::printf("주의: 쓰기가 밀려 %d 장을 버림(폴더·목록 모두에 없음)\n", dropped);
    } catch (const rs2::error& e) {
        std::fprintf(stderr, "librealsense 오류: %s (%s)\n", e.what(), e.get_failed_function().c_str());
        return 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "오류: %s\n", e.what());
        return 3;
    }
    return 0;
}
