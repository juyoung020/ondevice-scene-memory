// 카메라 없이 SDK 의 실제 rs2::align(이 기계의 realsense2 라이브러리)과 fast_align 을 같은 합성 장면에서 비교한다.
// software_device 에 깊이·컬러 프레임과 보정값을 넣어 rs2::align 을 돌리고, 결과 화소 차이와 시간을 잰다.
//   align_sdk_check [--calib <녹화 폴더/calib.json>] [--frames N] [--core C] [--threads T] [--write-rec <폴더>]
// --write-rec: 640x480 합성 장면을 rs_rec 과 같은 폴더 모양(원래 깊이·rs2::align 결과·calib.json)으로 써서
//              카메라 없이 rs_align_check 를 시험할 수 있게 한다.
// --calib 이 없으면 D455 비슷한 가정 보정값(tests/scenes.hpp)으로 640x480·848x480 깊이 두 가지.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "scenes.hpp"

using namespace tools;

static int run(const char* name, const rs2_intrinsics& di, const rs2_intrinsics& ci, const rs2_extrinsics& e, float unit, int frames,
               int threads, int core, const std::string& rec = "") {
    std::ofstream lraw, lrs, ldep, lrgb, las;
    if (!rec.empty()) {
        for (const char* sub : {"rgb", "depth", "depth_raw", "depth_rs"}) std::filesystem::create_directories(std::filesystem::path(rec) / sub);
        lraw.open(rec + "/depth_raw.txt");
        lrs.open(rec + "/depth_rs.txt");
        ldep.open(rec + "/depth.txt");
        lrgb.open(rec + "/rgb.txt");
        las.open(rec + "/association.txt");
        std::ofstream cj(rec + "/calib.json");
        cj << "{\n  \"device\": \"synthetic (align_sdk_check)\",\n  \"depth_intrinsics\": " << intrin_json(di)
           << ",\n  \"color_intrinsics\": " << intrin_json(ci) << ",\n  \"depth_to_color\": " << extrin_json(e)
           << ",\n  \"depth_unit_m\": " << jnum(unit) << "\n}\n";
    }

    std::printf("\n== %s: 깊이 %dx%d → 컬러 %dx%d, 컬러 왜곡 %s ==\n", name, di.width, di.height, ci.width, ci.height,
                rs2_distortion_to_string(ci.model));
    pin_cores(core, 1);
    SdkAligner sdk(di, ci, e, unit);
    fast_align::DepthAligner ours(to_fa(di), to_fa(ci), to_fa(e), unit);
    fast_align::Options fo;
    fo.exact = false;
    fast_align::DepthAligner ours_fast(to_fa(di), to_fa(ci), to_fa(e), unit, fo);
    const size_t N = (size_t)ci.width * ci.height;
    std::vector<uint16_t> a(N), b(N), c(N);
    std::vector<double> t_sdk, t_ours, t_fast;
    size_t diff_exact = 0, diff_fast = 0, px = 0;
    int bad = 0;
    for (int f = 0; f < frames; ++f) {
        // 장면마다 잡음 씨앗을 바꿔 다른 깊이
        auto z = scenes::render_depth(scenes::room(), to_fa(di), unit, 0.004, 0.03, 100 + f);
        double ms = 0;
        if (!sdk.process(z.data(), a.data(), ms)) {
            std::printf("  장면 %d: SDK 정렬 결과를 못 받음\n", f);
            ++bad;
            continue;
        }
        double t0 = now_ms();
        ours.process(z.data(), b.data());
        double t1 = now_ms();
        ours_fast.process(z.data(), c.data());
        double t2 = now_ms();
        if (f > 0) {  // 첫 장면은 SDK 가 표를 만듦
            t_sdk.push_back(ms);
            t_ours.push_back(t1 - t0);
            t_fast.push_back(t2 - t1);
        }
        for (size_t i = 0; i < N; ++i) {
            diff_exact += a[i] != b[i];
            diff_fast += a[i] != c[i];
        }
        px += N;
        if (!rec.empty()) {
            char ts[32];
            std::snprintf(ts, sizeof ts, "%.6f", 1.0 + f / 30.0);
            std::vector<uint8_t> rgb((size_t)ci.width * ci.height * 3, 128);
            write_png16(rec + "/depth_raw/" + ts + ".png", z.data(), di.width, di.height);
            write_png16(rec + "/depth_rs/" + ts + ".png", a.data(), ci.width, ci.height);
            write_png16(rec + "/depth/" + ts + ".png", b.data(), ci.width, ci.height);
            write_png_rgb(rec + "/rgb/" + ts + ".png", rgb.data(), ci.width, ci.height);
            lraw << ts << " depth_raw/" << ts << ".png\n";
            lrs << ts << " depth_rs/" << ts << ".png\n";
            ldep << ts << " depth/" << ts << ".png\n";
            lrgb << ts << " rgb/" << ts << ".png\n";
            las << ts << " rgb/" << ts << ".png " << ts << " depth/" << ts << ".png\n";
        }
    }
    std::printf("  SDK 가 고른 정렬 구현: %s\n", AlignImplLog::text().empty() ? "(로그 없음)" : AlignImplLog::text().c_str());
    std::printf("  rs2::align 과 다른 화소: fast_align 정확한 식 %zu / %zu, 빠른 식 %zu (%.4f %%)\n", diff_exact, px, diff_fast,
                px ? 100.0 * diff_fast / px : 0.0);
    std::printf("  시간(코어 %d 하나, 장면 %zu 개 중앙값 / 90 %%):\n", core, t_sdk.size());
    std::printf("    rs2::align.process         %7.3f / %7.3f ms\n", median(t_sdk), percentile(t_sdk, 0.9));
    std::printf("    fast_align Sdk 정확한 식   %7.3f / %7.3f ms\n", median(t_ours), percentile(t_ours, 0.9));
    std::printf("    fast_align Sdk 빠른 식     %7.3f / %7.3f ms\n", median(t_fast), percentile(t_fast, 0.9));
    if (threads > 1) {
        pin_cores(core - threads + 1, threads);
        fast_align::Options o;
        o.threads = threads;
        fast_align::DepthAligner mt(to_fa(di), to_fa(ci), to_fa(e), unit, o);
        auto z = scenes::render_depth(scenes::room(), to_fa(di), unit, 0.004, 0.03, 7);
        std::vector<double> t;
        for (int i = 0; i < frames; ++i) {
            double t0 = now_ms();
            mt.process(z.data(), b.data());
            t.push_back(now_ms() - t0);
        }
        std::printf("    fast_align Sdk 스레드 %d    %7.3f / %7.3f ms (코어 %d 개)\n", threads, median(t), percentile(t, 0.9), threads);
    }
    return (diff_exact == 0 && bad == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
    console_utf8();
    std::string calib, rec;
    int frames = 60, core = 29, threads = 4;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string k = argv[i];
        if (k == "--calib") calib = argv[++i];
        else if (k == "--frames") frames = std::atoi(argv[++i]);
        else if (k == "--core") core = std::atoi(argv[++i]);
        else if (k == "--threads") threads = std::atoi(argv[++i]);
        else if (k == "--write-rec") rec = argv[++i];
    }
    std::printf("align_sdk_check: librealsense %s\n", RS2_API_FULL_VERSION_STR);
    AlignImplLog::install();
    int rc = 0;
    try {
        if (!calib.empty()) {
            std::string js = read_file(calib);
            rs2_intrinsics di = intrin_from_json(jfind(js, "depth_intrinsics"));
            rs2_intrinsics ci = intrin_from_json(jfind(js, "color_intrinsics"));
            rs2_extrinsics e = extrin_from_json(jfind(js, "depth_to_color"));
            float unit = (float)std::atof(jfind(js, "depth_unit_m").c_str());
            rc |= run(calib.c_str(), di, ci, e, unit, frames, threads, core);
        } else {
            const rs2_extrinsics e = to_rs(scenes::d455_extrin());
            rc |= run("D455 가정 640x480", to_rs(scenes::d455_depth_640()), to_rs(scenes::d455_color_640()), e, 0.001f, frames, threads, core, rec);
            rc |= run("D455 가정 848x480", to_rs(scenes::d455_depth_848()), to_rs(scenes::d455_color_640()), e, 0.001f, frames, threads, core);
        }
    } catch (const rs2::error& er) {
        std::printf("librealsense 오류: %s (%s)\n", er.what(), er.get_failed_function().c_str());
        return 3;
    } catch (const std::exception& ex) {
        std::printf("오류: %s\n", ex.what());
        return 3;
    }
    std::printf("\n%s\n", rc ? "rs2::align 과 다른 결과 있음" : "rs2::align 과 화소 단위로 같음(정확한 식)");
    return rc;
}
