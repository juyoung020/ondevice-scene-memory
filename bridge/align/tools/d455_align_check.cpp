// 녹화 폴더(d455_rec)의 원래 깊이에 fast_align 을 다시 돌려 rs2::align 결과(depth_rs/)와 화소 차이·시간을 비교한다.
// 같은 원래 깊이로 rs2::align 도 카메라 없이(software_device) 다시 돌려 시간을 같은 조건에서 잰다.
//   d455_align_check <녹화 폴더> [--core 29] [--threads 4] [--max N]
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common.hpp"

using namespace tools;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    console_utf8();
    if (argc < 2) {
        std::fprintf(stderr, "d455_align_check <녹화 폴더> [--core C] [--threads T] [--max N]\n");
        return 1;
    }
    fs::path dir = argv[1];
    int core = 29, threads = 4, maxn = 1 << 30;
    for (int i = 2; i + 1 < argc; ++i) {
        std::string k = argv[i];
        if (k == "--core") core = std::atoi(argv[++i]);
        else if (k == "--threads") threads = std::atoi(argv[++i]);
        else if (k == "--max") maxn = std::atoi(argv[++i]);
    }
    AlignImplLog::install();
    try {
        const std::string js = read_file((dir / "calib.json").string());
        const rs2_intrinsics di = intrin_from_json(jfind(js, "depth_intrinsics"));
        const rs2_intrinsics ci = intrin_from_json(jfind(js, "color_intrinsics"));
        const rs2_extrinsics e = extrin_from_json(jfind(js, "depth_to_color"));
        const float unit = (float)std::atof(jfind(js, "depth_unit_m").c_str());
        auto raw = read_list((dir / "depth_raw.txt").string());
        auto rsl = read_list((dir / "depth_rs.txt").string());
        if (raw.size() != rsl.size()) throw std::runtime_error("depth_raw.txt 와 depth_rs.txt 장면 수가 다름");
        std::printf("d455_align_check: %s, %zu 장, 깊이 %dx%d → 컬러 %dx%d, 컬러 왜곡 %s, 깊이 단위 %g m\n", dir.string().c_str(), raw.size(),
                    di.width, di.height, ci.width, ci.height, rs2_distortion_to_string(ci.model), unit);

        pin_cores(core, 1);
        fast_align::DepthAligner ours(to_fa(di), to_fa(ci), to_fa(e), unit);
        fast_align::Options fo;
        fo.exact = false;
        fast_align::DepthAligner fast(to_fa(di), to_fa(ci), to_fa(e), unit, fo);
        fast_align::Options co;
        co.color_distortion = true;
        fast_align::DepthAligner cdist(to_fa(di), to_fa(ci), to_fa(e), unit, co);
        SdkAligner sdk(di, ci, e, unit);
        const size_t N = (size_t)ci.width * ci.height;
        std::vector<uint16_t> z, rs, a(N), b(N), s(N), cd(N);
        Diff tot, totf, tots, totcd;
        std::vector<double> t_ours, t_fast, t_sdk;
        int frames = 0, w, h;
        const int jump = (int)(0.05f / unit);  // 경계: 3×3 안 5 cm 넘는 차
        for (size_t k = 0; k < raw.size() && frames < maxn; ++k) {
            if (!read_png16((dir / raw[k].second).string(), z, w, h) || w != di.width || h != di.height) continue;
            if (!read_png16((dir / rsl[k].second).string(), rs, w, h) || w != ci.width || h != ci.height) continue;
            double t0 = now_ms();
            ours.process(z.data(), a.data());
            double t1 = now_ms();
            fast.process(z.data(), b.data());
            double t2 = now_ms();
            double ms = 0;
            bool sdk_ok = sdk.process(z.data(), s.data(), ms);
            cdist.process(z.data(), cd.data());
            if (frames > 0) {
                t_ours.push_back(t1 - t0);
                t_fast.push_back(t2 - t1);
                if (sdk_ok) t_sdk.push_back(ms);
            }
            auto add = [](Diff& T, const Diff& d) {
                T.both += d.both;
                T.equal += d.equal;
                T.only_a += d.only_a;
                T.only_b += d.only_b;
                T.total += d.total;
                T.edge += d.edge;
                T.edge_diff += d.edge_diff;
                T.sum_abs += d.sum_abs;
            };
            add(tot, compare(a.data(), rs.data(), ci.width, ci.height, jump));
            add(totf, compare(b.data(), rs.data(), ci.width, ci.height, jump));
            if (sdk_ok) add(tots, compare(s.data(), rs.data(), ci.width, ci.height, jump));
            add(totcd, compare(cd.data(), rs.data(), ci.width, ci.height, jump));
            ++frames;
        }
        if (!frames) throw std::runtime_error("읽은 장면 없음");
        auto show = [&](const char* name, const Diff& d) {
            const size_t diff = d.both - d.equal + d.only_a + d.only_b;
            std::printf("  %-34s 일치 %.5f (다른 화소 %zu / %zu), 둘 다 값 있는데 다름 %zu(평균 |차| %.1f 단위), 우리만 %zu, rs 만 %zu, "
                        "경계 화소 중 다름 %.4f\n",
                        name, 1.0 - (double)diff / d.total, diff, d.total, d.both - d.equal,
                        d.both - d.equal ? d.sum_abs / (d.both - d.equal) : 0.0, d.only_a, d.only_b,
                        d.edge ? (double)d.edge_diff / d.edge : 0.0);
        };
        std::printf("\n녹화된 rs2::align 결과(depth_rs/)와 비교, %d 장:\n", frames);
        show("fast_align Sdk 정확한 식", tot);
        show("fast_align Sdk 빠른 식", totf);
        show("rs2::align 다시(software_device)", tots);
        show("fast_align 컬러 왜곡 적용(SDK 와 다름)", totcd);
        std::printf("\n시간(코어 %d 하나, 중앙 / 90 %%):\n", core);
        std::printf("  rs2::align.process(다시)  %7.3f / %7.3f ms\n", median(t_sdk), percentile(t_sdk, 0.9));
        std::printf("  fast_align 정확한 식      %7.3f / %7.3f ms\n", median(t_ours), percentile(t_ours, 0.9));
        std::printf("  fast_align 빠른 식        %7.3f / %7.3f ms\n", median(t_fast), percentile(t_fast, 0.9));
        try {
            const std::string st = read_file((dir / "rec_stats.json").string());
            std::printf("  녹화 중 rs2::align        %7.3f ms, fast_align %7.3f ms(rec_stats.json, 코어 묶지 않음)\n",
                        std::atof(jfind(st, "rs_align_ms_median").c_str()), std::atof(jfind(st, "fast_align_ms_median").c_str()));
        } catch (...) {
        }
        if (threads > 1) {
            pin_cores(core - threads + 1, threads);
            fast_align::Options o;
            o.threads = threads;
            fast_align::DepthAligner mt(to_fa(di), to_fa(ci), to_fa(e), unit, o);
            std::vector<double> t;
            for (int i = 0; i < 50; ++i) {
                double t0 = now_ms();
                mt.process(z.data(), a.data());
                t.push_back(now_ms() - t0);
            }
            std::printf("  fast_align 스레드 %d       %7.3f / %7.3f ms(마지막 장면 50 번)\n", threads, median(t), percentile(t, 0.9));
        }
        std::printf("SDK 정렬 구현(이 기계): %s\n", AlignImplLog::text().empty() ? "(로그 없음)" : AlignImplLog::text().c_str());
        const size_t d0 = tot.both - tot.equal + tot.only_a + tot.only_b;
        return d0 == 0 ? 0 : 1;
    } catch (const rs2::error& er) {
        std::fprintf(stderr, "librealsense 오류: %s (%s)\n", er.what(), er.get_failed_function().c_str());
        return 3;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "오류: %s\n", ex.what());
        return 3;
    }
}
