// 정렬 시간 재기(ms/장면). 프로세스를 코어 하나(또는 여러 개)에 묶고, 여러 번 돌려 중앙값.
//   bench_align [--core N] [--reps R] [--rounds K]
// 재는 것: fast_align Sdk(스레드 1·2·4), fast_align Generic, 옮겨 온 librealsense SIMD 경로(x86 SSE / ARM NEON:
// rs2::align 의 계산부, 프레임 할당 빼고), 옮겨 온 librealsense 일반 경로(align_images + rsutil, API 호출 비용 빼고).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "fast_align/fast_align.hpp"
#include "rs_ref.hpp"
#include "scenes.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sched.h>
#endif

using namespace fast_align;

static bool pin_cores(int first, int n) {
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

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// fn 을 reps 번 돌린 평균(ms)을 rounds 번 → 중앙값과 최소
static void timeit(const char* name, int reps, int rounds, const std::function<void()>& fn, double& med, double& mn) {
    fn();  // 데우기
    std::vector<double> r;
    for (int k = 0; k < rounds; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        r.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
    }
    med = median(r);
    mn = *std::min_element(r.begin(), r.end());
    std::printf("    %-40s 중앙 %7.3f ms  최소 %7.3f ms\n", name, med, mn);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    int core = 30, reps = 20, rounds = 15;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--core")) core = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--rounds")) rounds = std::atoi(argv[++i]);
    }
#if defined(__aarch64__)
    const char* arch = "aarch64(NEON)";
#else
    const char* arch = "x86-64(SSE)";
#endif
    std::printf("bench_align: %s, 코어 %d 부터, %d 번 평균 × %d 회 중앙값\n", arch, core, reps, rounds);
    struct Cfg {
        const char* name;
        Intrinsics d, c;
    };
    const Cfg cfgs[] = {
        {"깊이 640x480 → 컬러 640x480", scenes::d455_depth_640(), scenes::d455_color_640()},
        {"깊이 848x480 → 컬러 640x480", scenes::d455_depth_848(), scenes::d455_color_640()},
        {"깊이 640x480 → 컬러 1280x720", scenes::d455_depth_640(), scenes::d455_color_1280x720()},
    };
    const Extrinsics e = scenes::d455_extrin();
    const float unit = 0.001f;
    std::printf("| 구성 | 방식 | 중앙 ms | 최소 ms |\n");
    for (const auto& cf : cfgs) {
        auto z = scenes::render_depth(scenes::room(), cf.d, unit, 0.004, 0.03, 3);
        std::vector<uint16_t> out((size_t)cf.c.width * cf.c.height);
        std::printf("  %s\n", cf.name);
        double med, mn;
        std::vector<std::string> rows;
        auto row = [&](const char* m) {
            char b[256];
            std::snprintf(b, sizeof b, "| %s | %s | %.3f | %.3f |", cf.name, m, med, mn);
            rows.push_back(b);
        };
        pin_cores(core, 1);
        {
            rsref::SimdAlign s(cf.d, unit);
            timeit("librealsense SIMD 경로 이식(1 코어)", reps, rounds, [&] { s.align_depth_to_other(z.data(), out.data(), cf.c, e); }, med, mn);
            row("librealsense SIMD 경로 이식");
        }
        timeit("librealsense 일반 경로 이식(1 코어)", std::max(1, reps / 4), rounds,
               [&] { rsref::align_z_to_other_generic(out.data(), z.data(), unit, cf.d, e, cf.c); }, med, mn);
        row("librealsense 일반 경로 이식");
        {
            Options o;
            o.method = Method::Generic;
            DepthAligner a(cf.d, cf.c, e, unit, o);
            timeit("fast_align Generic(1 코어)", reps, rounds, [&] { a.process(z.data(), out.data()); }, med, mn);
            row("fast_align Generic");
        }
        {
            DepthAligner a(cf.d, cf.c, e, unit, Options{});
            timeit("fast_align Sdk(1 코어)", reps, rounds, [&] { a.process(z.data(), out.data()); }, med, mn);
            row("fast_align Sdk, 스레드 1");
        }
        {
            Options o;
            o.exact = false;
            DepthAligner a(cf.d, cf.c, e, unit, o);
            timeit("fast_align Sdk 빠른 식(1 코어)", reps, rounds, [&] { a.process(z.data(), out.data()); }, med, mn);
            row("fast_align Sdk 빠른 식(exact=false), 스레드 1");
        }
        for (int T : {2, 4}) {
            pin_cores(core - T + 1, T);
            for (bool exact : {true, false}) {
                Options o;
                o.threads = T;
                o.exact = exact;
                DepthAligner a(cf.d, cf.c, e, unit, o);
                char nm[96];
                std::snprintf(nm, sizeof nm, "fast_align Sdk%s(스레드 %d, 코어 %d)", exact ? "" : " 빠른 식", T, T);
                timeit(nm, reps, rounds, [&] { a.process(z.data(), out.data()); }, med, mn);
                if (a.last_fell_back_serial()) std::printf("      (한 스레드로 다시 돈 장면 있음)\n");
                std::snprintf(nm, sizeof nm, "fast_align Sdk%s, 스레드 %d(코어 %d)", exact ? "" : " 빠른 식", T, T);
                row(nm);
            }
        }
        for (auto& r : rows) std::printf("%s\n", r.c_str());
    }
    return 0;
}
