// 정확성 시험:
//  A) 같은 입력에서 fast_align 이 옮겨 온 librealsense 코드(rs_ref.hpp)와 화소 하나까지 같은지(Sdk ↔ SIMD 경로, Generic ↔ 일반 경로),
//     스레드 1·3·4 개 모두.
//  B) 깊이로 확인 가능한 합성 장면(정면 평면·계단 경계·방)에서 정답(컬러 화소가 보는 면의 깊이)과 비교.
// 실패하면 0 아닌 값으로 끝난다.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fast_align/fast_align.hpp"
#include "rs_ref.hpp"
#include "scenes.hpp"

using namespace fast_align;

static int g_fail = 0;
#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            std::printf("  실패: " __VA_ARGS__); \
            std::printf("\n");                   \
            ++g_fail;                            \
        }                                        \
    } while (0)

static size_t count_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
    return n;
}

struct Case {
    const char* name;
    Intrinsics d, c;
    Extrinsics e;
};

static void exact_case(const Case& cs, const std::vector<uint16_t>& z, const char* zname) {
    const float unit = 0.001f;
    const size_t N = (size_t)cs.c.width * cs.c.height;
    // SIMD 경로 기준
    std::vector<uint16_t> ref(N), out(N), gref(N);
    rsref::SimdAlign simd(cs.d, unit);
    simd.align_depth_to_other(z.data(), ref.data(), cs.c, cs.e);
    rsref::align_z_to_other_generic(gref.data(), z.data(), unit, cs.d, cs.e, cs.c);
    for (int T : {1, 3, 4}) {
        Options o;
        o.threads = T;
        o.method = Method::Sdk;
        DepthAligner a(cs.d, cs.c, cs.e, unit, o);
        a.process(z.data(), out.data());
        size_t nd = count_diff(out, ref);
        std::printf("  %-34s %-6s Sdk     스레드 %d: 다른 화소 %zu / %zu (사각형 %s, SDK 사각형 %s%s)\n", cs.name, zname, T, nd, N,
                    a.uses_rect() ? "예" : "아니오", simd.last_rect ? "예" : "아니오", a.last_fell_back_serial() ? ", 한 스레드로 다시" : "");
        CHECK(nd == 0, "%s %s Sdk T=%d 차이 %zu", cs.name, zname, T, nd);
        CHECK(a.uses_rect() == simd.last_rect, "%s 사각형 판단 다름", cs.name);
        o.method = Method::Generic;
        DepthAligner g(cs.d, cs.c, cs.e, unit, o);
        g.process(z.data(), out.data());
        nd = count_diff(out, gref);
        std::printf("  %-34s %-6s Generic 스레드 %d: 다른 화소 %zu / %zu\n", cs.name, zname, T, nd, N);
        CHECK(nd == 0, "%s %s Generic T=%d 차이 %zu", cs.name, zname, T, nd);
    }
    // 빠른 식(exact=false): SDK 와 다른 화소 비율과, 다른 곳의 값 차이
    Options o;
    o.exact = false;
    DepthAligner a(cs.d, cs.c, cs.e, unit, o);
    a.process(z.data(), out.data());
    size_t nd = count_diff(out, ref), nz = 0;
    for (size_t i = 0; i < N; ++i) nz += (out[i] == 0) != (ref[i] == 0);
    std::printf("  %-34s %-6s Sdk 빠른 식      : 다른 화소 %zu (%.4f %%), 그중 한쪽만 빈 화소 %zu\n", cs.name, zname, nd,
                100.0 * nd / N, nz);
    CHECK(nd * 200 <= N, "%s %s 빠른 식 차이 %zu 가 0.5 %% 넘음", cs.name, zname, nd);
}

// 정답 비교 통계
struct GeoStat {
    size_t truth = 0, filled = 0, ok = 0, hole = 0, bleed = 0, edge_n = 0, edge_ok = 0;
};
static GeoStat geo_compare(const std::vector<uint16_t>& out, const std::vector<float>& truth, const Intrinsics& ci, float unit) {
    GeoStat s;
    const int W = ci.width, H = ci.height;
    for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) {
            size_t i = (size_t)y * W + x;
            float t = truth[i];
            if (t <= 0) continue;
            // 경계: 3×3 안 정답이 5 cm 넘게 다르거나 없음
            bool edge = false;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    float tn = truth[(size_t)(y + dy) * W + x + dx];
                    if (tn <= 0 || std::fabs(tn - t) > 0.05f) edge = true;
                }
            float o = out[i] * unit;
            bool good = out[i] && std::fabs(o - t) <= std::max(0.003f, 0.01f * t);
            if (edge) {
                ++s.edge_n;
                s.edge_ok += good;
                continue;
            }
            ++s.truth;
            if (!out[i]) {
                ++s.hole;
                continue;
            }
            ++s.filled;
            if (good) ++s.ok;
            else if (o > t + 0.05f) ++s.bleed;  // 뒤쪽 값이 앞을 덮음
        }
    return s;
}

static void geo_case(const char* name, const scenes::Scene& sc, const Intrinsics& di, const Intrinsics& ci, const Extrinsics& e,
                     double min_ok, double max_hole_sdk, bool sdk_strict = true) {
    const float unit = 0.001f;
    auto z = scenes::render_depth(sc, di, unit);
    auto truth = scenes::truth_in_color(sc, di, ci, e);
    std::vector<uint16_t> out((size_t)ci.width * ci.height);
    for (int variant = 0; variant < 3; ++variant) {
        if (variant == 2 && sdk_strict) break;  // 컬러 왜곡 켠 Sdk 는 C 에서만
        const Method m = variant == 1 ? Method::Generic : Method::Sdk;
        Options o;
        o.method = m;
        o.color_distortion = variant == 2;
        DepthAligner a(di, ci, e, unit, o);
        a.process(z.data(), out.data());
        GeoStat s = geo_compare(out, truth, ci, unit);
        double okr = s.filled ? (double)s.ok / s.filled : 0, holer = s.truth ? (double)s.hole / s.truth : 0;
        double edger = s.edge_n ? (double)s.edge_ok / s.edge_n : 1;
        std::printf("  %-26s %-7s 안쪽 %zu 화소: 값 맞음 %.4f, 구멍 %.4f, 뒤값 덮음 %zu | 경계 %zu 화소 중 맞음 %.3f\n", name,
                    variant == 0 ? "Sdk" : variant == 1 ? "Generic" : "Sdk+왜곡", s.truth, okr, holer, s.bleed, s.edge_n, edger);
        if (m == Method::Generic || sdk_strict || variant == 2) CHECK(okr >= min_ok, "%s 값 맞음 %.4f < %.4f", name, okr, min_ok);
        if (m == Method::Sdk) CHECK(holer <= max_hole_sdk, "%s Sdk 구멍 %.4f > %.4f", name, holer, max_hole_sdk);
        if (m == Method::Generic) CHECK(holer <= 0.001, "%s Generic 구멍 %.4f", name, holer);
    }
}

int main() {
    using namespace scenes;
    std::printf("== A. librealsense 이식본과 화소 단위 비교 ==\n");
    Intrinsics c640_nodist = d455_color_640();
    c640_nodist.model = Distortion::None;
    Intrinsics c640_mbc = d455_color_640();
    c640_mbc.model = Distortion::ModifiedBrownConrady;
    Intrinsics d640_ibc = d455_depth_640();  // 깊이에 왜곡이 있는 경우(표가 화소마다)
    d640_ibc.model = Distortion::InverseBrownConrady;
    d640_ibc.coeffs[0] = 0.02f;
    d640_ibc.coeffs[1] = -0.01f;
    d640_ibc.coeffs[2] = 0.001f;
    Intrinsics c640x360 = make_intrin(640, 360, 320.f, 320.f, 320.f, 180.f);
    Intrinsics d640_hi = make_intrin(640, 480, 300.f, 300.f, 320.f, 240.f);  // 깊이가 더 넓게 봄 → 사각형 채움
    const Case cases[] = {
        {"D455 640x480 → 640x480(IBC)", d455_depth_640(), d455_color_640(), d455_extrin()},
        {"D455 848x480 → 640x480(IBC)", d455_depth_848(), d455_color_640(), d455_extrin()},
        {"D455 640x480 → 1280x720(IBC)", d455_depth_640(), d455_color_1280x720(), d455_extrin()},
        {"640x480 → 640x480 왜곡 없음", d455_depth_640(), c640_nodist, d455_extrin()},
        {"640x480 → 640x480 MBC", d455_depth_640(), c640_mbc, d455_extrin()},
        {"깊이 IBC → 640x480(화소별 표)", d640_ibc, d455_color_640(), d455_extrin()},
        {"640x480 → 640x360(특수 해상도)", d455_depth_640(), c640x360, d455_extrin()},
        {"넓은 깊이 → 640x480(사각형)", d640_hi, c640_mbc, d455_extrin()},
    };
    for (const auto& cs : cases) {
        auto zr = room();
        auto z1 = render_depth(zr, cs.d, 0.001f, 0.004, 0.03, 7);
        exact_case(cs, z1, "방");
        auto z2 = random_depth(cs.d, 11);
        exact_case(cs, z2, "무작위");
    }

    std::printf("\n== B. 합성 장면 정답 비교(컬러 왜곡 없음, 정답 = 컬러 화소 중심이 보는 면의 깊이 카메라 z) ==\n");
    // 정면 평면 2 m, 옆으로 59 mm: 모든 값이 2000
    {
        Intrinsics di = d455_depth_640(), ci = c640_nodist;
        auto z = render_depth(front_plane(2.0), di, 0.001f);
        std::vector<uint16_t> out((size_t)ci.width * ci.height);
        for (Method m : {Method::Sdk, Method::Generic}) {
            Options o;
            o.method = m;
            DepthAligner a(di, ci, shift_x(-0.059f), 0.001f, o);
            a.process(z.data(), out.data());
            size_t filled = 0, wrong = 0;
            for (auto v : out) {
                filled += v != 0;
                wrong += v != 0 && v != 2000;
            }
            std::printf("  정면 평면 2 m %-7s: 채움 %.4f, 2000 아닌 값 %zu\n", m == Method::Sdk ? "Sdk" : "Generic",
                        (double)filled / out.size(), wrong);
            CHECK(wrong == 0, "정면 평면 값 다름 %zu", wrong);
        }
    }
    geo_case("계단(1 m 판 / 2 m 벽)", step(), d455_depth_640(), c640_nodist, d455_extrin(), 0.995, 0.08);
    geo_case("방(벽·바닥·상자)", room(), d455_depth_640(), c640_nodist, d455_extrin(), 0.99, 0.08);
    geo_case("방 848x480 깊이", room(), d455_depth_848(), c640_nodist, d455_extrin(), 0.99, 0.02);

    std::printf("\n== C. D455 컬러 왜곡(Inverse Brown-Conrady) 그대로: Sdk 는 컬러 왜곡을 무시(SDK SIMD 경로와 같음) ==\n");
    geo_case("방, 컬러 IBC", room(), d455_depth_640(), d455_color_640(), d455_extrin(), 0.99, 0.08, false);
    // 컬러 왜곡을 무시할 때 화소가 얼마나 옮겨지는지(같은 광선을 왜곡 넣고/빼고 투영)
    {
        Intrinsics ci = d455_color_640(), cn = ci;
        cn.model = Distortion::None;
        double mx = 0, sum = 0;
        size_t n = 0, over1 = 0;
        for (int y = 0; y < ci.height; y += 4)
            for (int x = 0; x < ci.width; x += 4) {
                float px[2] = {(float)x, (float)y}, ray[3], a[2], b[2];
                rsref::rs2_deproject_pixel_to_point(ray, &ci, px, 1.0f);
                rsref::rs2_project_point_to_pixel(a, &ci, ray);
                rsref::rs2_project_point_to_pixel(b, &cn, ray);
                double d = std::hypot(a[0] - b[0], a[1] - b[1]);
                mx = std::max(mx, d);
                sum += d;
                ++n;
                over1 += d > 1.0;
            }
        std::printf("  컬러 왜곡 무시로 옮겨지는 거리(640x480 가정 계수): 평균 %.2f 화소, 최대 %.2f 화소, 1 화소 넘는 곳 %.1f %%\n", sum / n, mx,
                    100.0 * over1 / n);
    }

    std::printf("\n%s (실패 %d)\n", g_fail ? "실패" : "모두 통과", g_fail);
    return g_fail ? 1 : 0;
}
