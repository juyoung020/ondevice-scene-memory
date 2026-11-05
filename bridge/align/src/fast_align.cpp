// 깊이 → 컬러 정렬 구현. 규칙과 수치 순서는 librealsense v2.59.1(Apache-2.0)의
//   src/proc/sse/sse-align.cpp · src/proc/neon/neon-align.cpp(Sdk 방식), src/proc/align.cpp · src/rs.cpp 의 rsutil(Generic 방식)
// 을 따른다. 빠르게 하는 방법:
//   1) 모서리 역투영 방향 표: 깊이 왜곡이 없으면 열마다 x, 줄마다 y 만 둔다(SDK 는 화소마다 float 2 개 × 모서리 2 개).
//   2) 한 줄씩 계산 → 바로 흩뿌리기: 줄 하나 분량의 정수 좌표만 L1 에 두고 쓴다(SDK 는 영상 전체 int2 배열을 쓰고 다시 읽음).
//   3) 계산 루프는 분기 없는 float 식이라 자동 벡터화된다(x86 SSE/AVX, aarch64 NEON).
//   4) 여러 스레드: 깊이 줄 덩어리를 짝수·홀수 두 번에 나눠 돌리고, 동시에 돈 덩어리가 쓴 컬러 줄 범위가 겹치면 한 스레드로 다시 한다.
#include "fast_align/fast_align.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#define FA_NEON_ORDER 1  // SDK NEON 경로와 같은 fma 순서
#else
#define FA_NEON_ORDER 0  // SDK SSE 경로와 같은 곱·더하기 순서
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define FA_RESTRICT __restrict
#else
#define FA_RESTRICT __restrict__
#endif

namespace fast_align {

void fov_deg(const Intrinsics& in, float to_fov[2]) {
    // rs2_fov(src/rs.cpp) 그대로
    to_fov[0] = (atan2f(in.ppx + 0.5f, in.fx) + atan2f(in.width - (in.ppx + 0.5f), in.fx)) * 57.2957795f;
    to_fov[1] = (atan2f(in.ppy + 0.5f, in.fy) + atan2f(in.height - (in.ppy + 0.5f), in.fy)) * 57.2957795f;
}

namespace {

bool is_special_resolution(const Intrinsics& depth, const Intrinsics& to) {
    return (depth.width == 640 && depth.height == 240 && to.width == 320 && to.height == 180) ||
           (depth.width == 640 && depth.height == 480 && to.width == 640 && to.height == 360);
}

bool distortion_zero(const Intrinsics& in) {
    for (float c : in.coeffs)
        if (std::fabs(c) >= 1.1920929e-07f) return false;  // FLT_EPSILON
    return true;
}

// 정수 변환: SDK 는 (u + 0.5) 를 0 쪽으로 자른다(cvttps / vcvtq). 범위 밖 값은 화면 밖이 되게 묶는다.
inline int32_t to_int(float v) {
#if FA_NEON_ORDER
    // aarch64: fmaxnm/fminnm 한 개씩으로 벡터화됨
    return static_cast<int32_t>(std::fmin(std::fmax(v, -1.0e9f), 1.0e9f));
#else
    // 삼항식으로 써야 MSVC 도 벡터화함(std::min/max 는 참조를 돌려줘서 못 함)
    v = v < -1.0e9f ? -1.0e9f : v;
    v = v > 1.0e9f ? 1.0e9f : v;
    return static_cast<int32_t>(v);
#endif
}

// ---------------- 모서리 역투영 표 ----------------

// Sdk: pre_compute_x_y_map(sse-align.cpp / neon-align.cpp)
void sdk_corner(const Intrinsics& d, int w, int h, float offset, float& ox, float& oy) {
    const float pixel[] = {(float)w + offset, (float)h + offset};
    float x = (pixel[0] - d.ppx) / d.fx;
    float y = (pixel[1] - d.ppy) / d.fy;
    if (d.model == Distortion::InverseBrownConrady) {
        const float* c = d.coeffs;
        const float r2 = x * x + y * y;
#if FA_NEON_ORDER
        const float f = 1.0f + (c[0] + (c[1] + c[4] * r2) * r2) * r2;
        const float ux = x * f + 2.0f * c[2] * x * y + c[3] * (r2 + 2.0f * x * x);
        const float uy = y * f + 2.0f * c[3] * x * y + c[2] * (r2 + 2.0f * y * y);
#else
        float f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2;
        float ux = x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x);
        float uy = y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y);
#endif
        x = ux;
        y = uy;
    }
    ox = x;
    oy = y;
}

// Generic: rs2_deproject_pixel_to_point(src/rs.cpp)의 깊이 1 일 때 x, y
void generic_corner(const Intrinsics& in, int w, int h, float offset, float& ox, float& oy) {
    const float pixel[2] = {w + offset, h + offset};
    float x = (pixel[0] - in.ppx) / in.fx;
    float y = (pixel[1] - in.ppy) / in.fy;
    const float xo = x, yo = y;
    const float* c = in.coeffs;
    if (!distortion_zero(in)) {
        if (in.model == Distortion::InverseBrownConrady) {
            for (int i = 0; i < 10; i++) {
                float r2 = x * x + y * y;
                float icdist = (float)1 / (float)(1 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                float xq = x / icdist;
                float yq = y / icdist;
                float delta_x = 2 * c[2] * xq * yq + c[3] * (r2 + 2 * xq * xq);
                float delta_y = 2 * c[3] * xq * yq + c[2] * (r2 + 2 * yq * yq);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
        if (in.model == Distortion::BrownConrady) {
            for (int i = 0; i < 10; i++) {
                float r2 = x * x + y * y;
                float icdist = (float)1 / (float)(1 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                float delta_x = 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x);
                float delta_y = 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
    }
    if (in.model == Distortion::FTheta || in.model == Distortion::KannalaBrandt4)
        throw std::invalid_argument("fast_align: 깊이 왜곡 모델 FTheta/KannalaBrandt4 는 지원하지 않음");
    ox = x;
    oy = y;
}

// ---------------- 스레드 풀 ----------------

class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    // fn(0..count-1) 을 일꾼 + 부른 스레드가 나눠 돌리고 다 끝날 때까지 기다림
    void run(int count, const std::function<void(int)>& fn) {
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            count_ = count;
            next_.store(0);
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return done_ == count_; });
        fn_ = nullptr;
    }

private:
    void work() {
        for (;;) {
            int i = next_.fetch_add(1);
            if (i >= count_) return;
            (*fn_)(i);
            std::lock_guard<std::mutex> lk(m_);
            if (++done_ == count_) done_cv_.notify_all();
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            work();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)>* fn_ = nullptr;
    int count_ = 0, done_ = 0;
    std::atomic<int> next_{0};
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace

// ---------------- 본체 ----------------

struct DepthAligner::Impl {
    Intrinsics d, c;
    Extrinsics e;
    float unit;
    Options opt;
    bool rect = false;
    bool sep = true;      // 표가 열·줄로 나뉨(깊이 왜곡 없음)
    bool color_dist = false;
    bool fell_back = false;
    // 표[모서리]: sep 이면 x 는 열 수, y 는 줄 수 길이, 아니면 화소 수 길이
    std::vector<float> xt[2], yt[2];
    std::unique_ptr<Pool> pool;
    struct Scratch {
        std::vector<int32_t> u0, v0, u1, v1;
        std::vector<float> zf;
        int vmin, vmax;
    };
    std::vector<Scratch> scratch;  // 덩어리마다(여러 스레드), 또는 하나

    void build_tables() {
        const int W = d.width, H = d.height;
        std::vector<float> fx[2], fy[2];
        for (int k = 0; k < 2; ++k) {
            fx[k].resize((size_t)W * H);
            fy[k].resize((size_t)W * H);
            const float off = k == 0 ? -0.5f : 0.5f;
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    float ox, oy;
                    if (opt.method == Method::Sdk)
                        sdk_corner(d, w, h, off, ox, oy);
                    else
                        generic_corner(d, w, h, off, ox, oy);
                    fx[k][(size_t)h * W + w] = ox;
                    fy[k][(size_t)h * W + w] = oy;
                }
        }
        // 나눌 수 있는지: x 가 줄마다 같고 y 가 열마다 같은지(비트 단위)
        sep = true;
        for (int k = 0; k < 2 && sep; ++k)
            for (int h = 0; h < H && sep; ++h)
                for (int w = 0; w < W; ++w)
                    if (std::memcmp(&fx[k][(size_t)h * W + w], &fx[k][w], 4) != 0 ||
                        std::memcmp(&fy[k][(size_t)h * W + w], &fy[k][(size_t)h * W], 4) != 0) {
                        sep = false;
                        break;
                    }
        for (int k = 0; k < 2; ++k) {
            if (sep) {
                xt[k].assign(fx[k].begin(), fx[k].begin() + W);
                yt[k].resize(H);
                for (int h = 0; h < H; ++h) yt[k][h] = fy[k][(size_t)h * W];
            } else {
                xt[k] = std::move(fx[k]);
                yt[k] = std::move(fy[k]);
            }
        }
    }

    // 한 줄, 한 모서리: 깊이 화소 → 컬러 정수 좌표
    // IDX: 화면 안이고 깊이가 있으면 컬러 화소 번호(v*W+u), 아니면 -1 을 uo 에(왼쪽 위 모서리만 쓸 때)
    // FAST: 나눗셈 하나(1/Z)와 줄마다 미리 곱한 회전 방향으로(SDK 와 반올림이 드물게 달라짐)
    template <bool SEP, bool DIST, bool IDX, bool FAST>
    void corner_row_sdk(const float* FA_RESTRICT z, int y, int k, int32_t* FA_RESTRICT uo,
                        int32_t* FA_RESTRICT vo) const {
        const int W = d.width;
        const int CW = c.width, CH = c.height;
        const float* FA_RESTRICT xs = SEP ? xt[k].data() : xt[k].data() + (size_t)y * W;
        const float* FA_RESTRICT ys = SEP ? nullptr : yt[k].data() + (size_t)y * W;
        const float yrow = SEP ? yt[k][y] : 0.f;
        const float* r = e.rotation;
        const float r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4], r5 = r[5], r6 = r[6], r7 = r[7], r8 = r[8];
        const float t0 = e.translation[0], t1 = e.translation[1], t2 = e.translation[2];
        const float fx = c.fx, fy = c.fy, ppx = c.ppx, ppy = c.ppy;
        const float c0 = c.coeffs[0], c1 = c.coeffs[1], c2 = c.coeffs[2], c3 = c.coeffs[3], c4 = c.coeffs[4];
        const float scale = unit;
        // FAST 용: 회전한 방향 a = R (x, y, 1) 의 줄 상수 부분
        const float kx = r3 * yrow + r6, ky = r4 * yrow + r7, kz = r5 * yrow + r8;
        uint32_t rmin = 0xFFFFFFFFu;  // IDX: 이 줄에서 쓴 화소 번호 최소·최대(-1 은 부호 없이 가장 커서 최소에서 빠짐)
        int32_t rmax = -1;
        for (int x = 0; x < W; ++x) {
            const float depth = z[x] * scale;  // z = (float) 깊이 값(정확히 표현됨)
            float X, Y;
            if constexpr (FAST) {
                const float yy = SEP ? 0.f : ys[x];
                const float ax = SEP ? r0 * xs[x] + kx : r0 * xs[x] + (r3 * yy + r6);
                const float ay = SEP ? r1 * xs[x] + ky : r1 * xs[x] + (r4 * yy + r7);
                const float az = SEP ? r2 * xs[x] + kz : r2 * xs[x] + (r5 * yy + r8);
                const float w = 1.0f / (depth * az + t2);
                X = (depth * ax + t0) * w;
                Y = (depth * ay + t1) * w;
            } else {
                const float px = depth * xs[x];
                const float py = depth * (SEP ? yrow : ys[x]);
#if FA_NEON_ORDER
                X = std::fma(r0, px, std::fma(r3, py, std::fma(r6, depth, t0)));
                Y = std::fma(r1, px, std::fma(r4, py, std::fma(r7, depth, t1)));
                const float Z = std::fma(r2, px, std::fma(r5, py, std::fma(r8, depth, t2)));
#else
                X = r0 * px + (r3 * py + (r6 * depth + t0));
                Y = r1 * px + (r4 * py + (r7 * depth + t1));
                const float Z = r2 * px + (r5 * py + (r8 * depth + t2));
#endif
                X = X / Z;
                Y = Y / Z;
            }
            if constexpr (DIST) {  // MODIFIED_BROWN_CONRADY
#if FA_NEON_ORDER
                const float rr = X * X + Y * Y;
                const float f = std::fma(rr, std::fma(rr, std::fma(rr, c4, c1), c0), 1.0f);
                const float xf = X * f, yf = Y * f;
                X = std::fma(c3, rr, xf * std::fma(2.0f, std::fma(c3, xf, c2 * yf), 1.0f));
                Y = std::fma(c2, rr, yf * std::fma(2.0f, std::fma(c2, yf, c3 * xf), 1.0f));
#else
                const float rr = X * X + Y * Y;
                const float r3_ = c1 * (rr * rr) + c4 * (rr * (rr * rr));
                const float f = 1.0f + (c0 * rr + r3_);
                const float xf = X * f, yf = Y * f;
                const float r4_ = c3 * (rr + 2.0f * (xf * xf));
                X = xf + (2.0f * (c2 * (xf * yf)) + r4_);
                const float r5_ = c2 * (rr + 2.0f * (yf * yf));
                Y = yf + (2.0f * (c3 * (xf * yf)) + r5_);
#endif
            }
#if FA_NEON_ORDER
            const float u = std::fma(X, fx, ppx);
            const float v = std::fma(Y, fy, ppy);
#else
            const float u = X * fx + ppx;
            const float v = Y * fy + ppy;
#endif
            const int32_t ui = to_int(u + 0.5f), vi = to_int(v + 0.5f);
            if constexpr (IDX) {
                // 분기·비교 없이(부호 비트): 화면 밖이거나 깊이 0 이면 -1, 아니면 vi*CW+ui. 묶은 좌표는 ±1e9 라 넘침 없음
                const int32_t zi = (int32_t)z[x];
                const int32_t bad = (ui | (CW - 1 - ui) | vi | (CH - 1 - vi) | (zi - 1)) >> 31;
                const int32_t id = (int32_t)((uint32_t)vi * (uint32_t)CW + (uint32_t)ui) | bad;
                uo[x] = id;
                rmin = (uint32_t)id < rmin ? (uint32_t)id : rmin;
                rmax = id > rmax ? id : rmax;
            } else {
                uo[x] = ui;
                vo[x] = vi;
            }
        }
        if constexpr (IDX) {
            vo[0] = (int32_t)rmin;
            vo[1] = rmax;
        }
    }

    // Generic: rs2_deproject → rs2_transform_point_to_point → rs2_project_point_to_pixel 순서 그대로
    template <bool SEP, int CMODEL>
    void corner_row_generic(const float* FA_RESTRICT z, int y, int k, int32_t* FA_RESTRICT uo,
                            int32_t* FA_RESTRICT vo) const {
        const int W = d.width;
        const float* FA_RESTRICT xs = SEP ? xt[k].data() : xt[k].data() + (size_t)y * W;
        const float* FA_RESTRICT ys = SEP ? nullptr : yt[k].data() + (size_t)y * W;
        const float yrow = SEP ? yt[k][y] : 0.f;
        const float* r = e.rotation;
        const float r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4], r5 = r[5], r6 = r[6], r7 = r[7], r8 = r[8];
        const float t0 = e.translation[0], t1 = e.translation[1], t2 = e.translation[2];
        const float fx = c.fx, fy = c.fy, ppx = c.ppx, ppy = c.ppy;
        const float c0 = c.coeffs[0], c1 = c.coeffs[1], c2 = c.coeffs[2], c3 = c.coeffs[3], c4 = c.coeffs[4];
        const float scale = unit;
        for (int x = 0; x < W; ++x) {
            const float depth = scale * z[x];
            const float p0 = depth * xs[x];
            const float p1 = depth * (SEP ? yrow : ys[x]);
            const float p2 = depth;
            const float X = r0 * p0 + r3 * p1 + r6 * p2 + t0;
            const float Y = r1 * p0 + r4 * p1 + r7 * p2 + t1;
            const float Z = r2 * p0 + r5 * p1 + r8 * p2 + t2;
            float xx = X / Z, yy = Y / Z;
            if (CMODEL == 1) {  // MODIFIED / INVERSE BROWN_CONRADY
                float rr = xx * xx + yy * yy;
                float f = 1 + c0 * rr + c1 * rr * rr + c4 * rr * rr * rr;
                xx *= f;
                yy *= f;
                float dx = xx + 2 * c2 * xx * yy + c3 * (rr + 2 * xx * xx);
                float dy = yy + 2 * c3 * xx * yy + c2 * (rr + 2 * yy * yy);
                xx = dx;
                yy = dy;
            } else if (CMODEL == 2) {  // BROWN_CONRADY
                float rr = xx * xx + yy * yy;
                float f = 1 + c0 * rr + c1 * rr * rr + c4 * rr * rr * rr;
                float xf = xx * f;
                float yf = yy * f;
                float dx = xf + 2 * c2 * xx * yy + c3 * (rr + 2 * xx * xx);
                float dy = yf + 2 * c3 * xx * yy + c2 * (rr + 2 * yy * yy);
                xx = dx;
                yy = dy;
            }
            const float u = xx * fx + ppx;
            const float v = yy * fy + ppy;
            uo[x] = to_int(u + 0.5f);
            vo[x] = to_int(v + 0.5f);
        }
    }

    // Sdk 핵심 루프 고르기: 표 모양(SEP) × 컬러 왜곡(DIST) × 빠른 식(FAST)
    template <bool IDX>
    void sdk_row(const float* z, int y, int k, int32_t* uo, int32_t* vo) const {
        const bool fast = !opt.exact;
#define FA_SDK(S, D, F) corner_row_sdk<S, D, IDX, F>(z, y, k, uo, vo)
        if (sep) {
            if (color_dist) fast ? FA_SDK(true, true, true) : FA_SDK(true, true, false);
            else fast ? FA_SDK(true, false, true) : FA_SDK(true, false, false);
        } else {
            if (color_dist) fast ? FA_SDK(false, true, true) : FA_SDK(false, true, false);
            else fast ? FA_SDK(false, false, true) : FA_SDK(false, false, false);
        }
#undef FA_SDK
    }

    void corner_row(const float* z, int y, int k, int32_t* uo, int32_t* vo) const {
        if (opt.method == Method::Sdk) {
            sdk_row<false>(z, y, k, uo, vo);
        } else {
            int cm = 0;
            if (c.model == Distortion::ModifiedBrownConrady || c.model == Distortion::InverseBrownConrady) cm = 1;
            if (c.model == Distortion::BrownConrady) cm = 2;
            if (sep) {
                if (cm == 1) corner_row_generic<true, 1>(z, y, k, uo, vo);
                else if (cm == 2) corner_row_generic<true, 2>(z, y, k, uo, vo);
                else corner_row_generic<true, 0>(z, y, k, uo, vo);
            } else {
                if (cm == 1) corner_row_generic<false, 1>(z, y, k, uo, vo);
                else if (cm == 2) corner_row_generic<false, 2>(z, y, k, uo, vo);
                else corner_row_generic<false, 0>(z, y, k, uo, vo);
            }
        }
    }

    // mm[0], mm[1] = 쓴 화소 번호 최소·최대(없으면 -1)
    void index_row(const float* z, int y, int32_t* idx, int32_t* mm) const { sdk_row<true>(z, y, 0, idx, mm); }

    static inline void put(uint16_t* o, uint16_t zz) {
        const uint16_t cur = *o;
        *o = (cur && cur < zz) ? cur : zz;  // 비었으면 새 값, 아니면 가까운 값
    }

    // 깊이 줄 [y0, y1) 를 정렬해 out 에 씀. 쓴 컬러 줄 범위를 sc.vmin/vmax 에 남김
    void rows(const uint16_t* depth, uint16_t* out, int y0, int y1, Scratch& sc) const {
        const int W = d.width, CW = c.width, CH = c.height;
        int vmin = CH, vmax = -1;
        for (int y = y0; y < y1; ++y) {
            const uint16_t* z = depth + (size_t)y * W;
            float* zf = sc.zf.data();
            for (int x = 0; x < W; ++x) zf[x] = (float)z[x];
            if (!rect) {  // Sdk, 왼쪽 위 모서리만: 화소 번호까지 벡터 루프에서 내고 여기서는 흩뿌리기만
                int32_t* idx = sc.u0.data();
                int32_t mm[2];
                index_row(zf, y, idx, mm);
                for (int x = 0; x < W; ++x) {
                    const int32_t i = idx[x];
                    if (i < 0) continue;
                    put(out + i, z[x]);
                }
                if (mm[1] >= 0) {
                    vmin = std::min(vmin, mm[0] / CW);
                    vmax = std::max(vmax, mm[1] / CW);
                }
                continue;
            }
            corner_row(zf, y, 0, sc.u0.data(), sc.v0.data());
            corner_row(zf, y, 1, sc.u1.data(), sc.v1.data());
            const int32_t *u0 = sc.u0.data(), *v0 = sc.v0.data(), *u1 = sc.u1.data(), *v1 = sc.v1.data();
            if (opt.method == Method::Sdk) {
                // move_depth_to_other: 화소마다 화면 안만
                for (int x = 0; x < W; ++x) {
                    const uint16_t zz = z[x];
                    if (!zz) continue;
                    const int ya = std::max(v0[x], 0), yb = std::min(v1[x], CH - 1);
                    const int xa = std::max(u0[x], 0), xb = std::min(u1[x], CW - 1);
                    if (ya > yb || xa > xb) continue;
                    for (int oy = ya; oy <= yb; ++oy) {
                        uint16_t* row = out + (size_t)oy * CW;
                        for (int ox = xa; ox <= xb; ++ox) put(row + ox, zz);
                    }
                    vmin = std::min(vmin, ya);
                    vmax = std::max(vmax, yb);
                }
            } else {
                // align_images: 사각형이 화면을 벗어나면 통째로 버림
                for (int x = 0; x < W; ++x) {
                    const uint16_t zz = z[x];
                    if (!zz) continue;
                    if (u0[x] < 0 || v0[x] < 0 || u1[x] >= CW || v1[x] >= CH) continue;
                    if (v0[x] > v1[x] || u0[x] > u1[x]) continue;
                    for (int oy = v0[x]; oy <= v1[x]; ++oy) {
                        uint16_t* row = out + (size_t)oy * CW;
                        for (int ox = u0[x]; ox <= u1[x]; ++ox) put(row + ox, zz);
                    }
                    vmin = std::min(vmin, v0[x]);
                    vmax = std::max(vmax, v1[x]);
                }
            }
        }
        sc.vmin = vmin;
        sc.vmax = vmax;
    }

    void process(const uint16_t* depth, uint16_t* out) {
        const int H = d.height;
        std::memset(out, 0, sizeof(uint16_t) * (size_t)c.width * c.height);
        fell_back = false;
        const int T = std::max(1, opt.threads);
        if (T == 1 || H < 4 * T) {
            rows(depth, out, 0, H, scratch[0]);
            return;
        }
        const int nch = 2 * T;
        auto chunk_y = [&](int i) { return (int)((long long)H * i / nch); };
        for (int pass = 0; pass < 2; ++pass) {
            std::function<void(int)> fn = [&, pass](int j) {
                const int i = 2 * j + pass;
                rows(depth, out, chunk_y(i), chunk_y(i + 1), scratch[i]);
            };
            pool->run(T, fn);
        }
        // 같은 번에 동시에 돈 덩어리끼리 컬러 줄 범위가 겹쳤으면 경쟁이 있었을 수 있음 → 한 스레드로 다시
        for (int pass = 0; pass < 2 && !fell_back; ++pass)
            for (int a = pass; a < nch && !fell_back; a += 2)
                for (int b = a + 2; b < nch; b += 2) {
                    const Scratch &A = scratch[a], &B = scratch[b];
                    if (A.vmax < A.vmin || B.vmax < B.vmin) continue;
                    if (A.vmin <= B.vmax && B.vmin <= A.vmax) {
                        fell_back = true;
                        break;
                    }
                }
        if (fell_back) {
            std::memset(out, 0, sizeof(uint16_t) * (size_t)c.width * c.height);
            rows(depth, out, 0, H, scratch[0]);
        }
    }
};

DepthAligner::DepthAligner(const Intrinsics& depth, const Intrinsics& color, const Extrinsics& d2c,
                           float depth_unit, const Options& opt)
    : impl_(new Impl) {
    Impl& m = *impl_;
    m.d = depth;
    m.c = color;
    m.e = d2c;
    m.unit = depth_unit;
    m.opt = opt;
    if (depth.width <= 0 || depth.height <= 0 || color.width <= 0 || color.height <= 0)
        throw std::invalid_argument("fast_align: 영상 크기가 0");
    if (opt.method == Method::Sdk) {
        m.color_dist = color.model == Distortion::ModifiedBrownConrady ||
                       (opt.color_distortion && color.model == Distortion::InverseBrownConrady);
        float fov[2];
        fov_deg(depth, fov);
        const float dpx = (float)depth.width / fov[0], dpy = (float)depth.height / fov[1];
        fov_deg(color, fov);
        const float cpx = (float)color.width / fov[0], cpy = (float)color.height / fov[1];
        m.rect = opt.force_rect || dpx < cpx || dpy < cpy || is_special_resolution(depth, color);
    } else {
        if (color.model == Distortion::FTheta || color.model == Distortion::KannalaBrandt4)
            throw std::invalid_argument("fast_align: 컬러 왜곡 모델 FTheta/KannalaBrandt4 는 지원하지 않음");
        m.rect = true;
    }
    m.build_tables();
    const int T = std::max(1, opt.threads);
    m.scratch.resize(T == 1 ? 1 : 2 * T);
    for (auto& s : m.scratch) {
        s.u0.resize(depth.width);
        s.v0.resize(depth.width);
        s.u1.resize(depth.width);
        s.v1.resize(depth.width);
        s.zf.resize(depth.width);
    }
    if (T > 1) m.pool.reset(new Pool(T - 1));
}

DepthAligner::~DepthAligner() = default;

void DepthAligner::process(const uint16_t* depth, uint16_t* out) { impl_->process(depth, out); }
bool DepthAligner::uses_rect() const { return impl_->rect; }
bool DepthAligner::last_fell_back_serial() const { return impl_->fell_back; }

void align_depth_to_color(const uint16_t* depth, uint16_t* out, const Intrinsics& di, const Intrinsics& ci,
                          const Extrinsics& e, float unit, const Options& opt) {
    DepthAligner a(di, ci, e, unit, opt);
    a.process(depth, out);
}

}  // namespace fast_align
