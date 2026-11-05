// 시험·벤치용 합성 장면과 D455 비슷한 보정값(가정 값: 실제 값은 d455_info 로 받는다).
// 장면은 깊이 카메라 좌표(OpenCV: x 오른쪽, y 아래, z 앞, m)의 평면·상자로 만들고 광선을 쏴서 깊이를 그린다.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fast_align/fast_align.hpp"
#include "rs_ref.hpp"

namespace scenes {

using fast_align::Distortion;
using fast_align::Extrinsics;
using fast_align::Intrinsics;

inline Intrinsics make_intrin(int w, int h, float fx, float fy, float ppx, float ppy, Distortion m = Distortion::None,
                              std::initializer_list<float> k = {}) {
    Intrinsics in;
    in.width = w;
    in.height = h;
    in.fx = fx;
    in.fy = fy;
    in.ppx = ppx;
    in.ppy = ppy;
    in.model = m;
    int i = 0;
    for (float v : k) in.coeffs[i++] = v;
    return in;
}

// D455 비슷한 값(공개 덤프 수준의 가정): 깊이 = 왼쪽 IR, Brown-Conrady 계수 0. 컬러 = Inverse Brown-Conrady 계수 있음.
inline Intrinsics d455_depth_640() { return make_intrin(640, 480, 390.6f, 390.6f, 319.4f, 238.7f, Distortion::BrownConrady); }
inline Intrinsics d455_depth_848() { return make_intrin(848, 480, 425.9f, 425.9f, 423.6f, 238.9f, Distortion::BrownConrady); }
inline Intrinsics d455_color_640() {
    return make_intrin(640, 480, 383.1f, 382.4f, 321.6f, 244.9f, Distortion::InverseBrownConrady,
                       {-0.0566f, 0.0679f, 0.00015f, 0.00043f, -0.0218f});
}
inline Intrinsics d455_color_1280x720() {
    return make_intrin(1280, 720, 642.0f, 641.6f, 652.6f, 360.3f, Distortion::InverseBrownConrady,
                       {-0.0566f, 0.0679f, 0.00015f, 0.00043f, -0.0218f});
}
// 깊이→컬러: 약 59 mm 옆, 아주 작은 회전(열 우선)
inline Extrinsics d455_extrin() {
    Extrinsics e;
    const float ax = 0.0011f, ay = -0.0024f, az = 0.0007f;  // 라디안(작은 각 근사로 회전 행렬)
    // R = I + [w]x, 열 우선
    float R[3][3] = {{1, -az, ay}, {az, 1, -ax}, {-ay, ax, 1}};
    // 정규 직교로 다듬기(그람-슈미트)
    auto norm = [](float* v) {
        float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int i = 0; i < 3; ++i) v[i] /= n;
    };
    float c0[3] = {R[0][0], R[1][0], R[2][0]}, c1[3] = {R[0][1], R[1][1], R[2][1]}, c2[3];
    norm(c0);
    float d = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
    for (int i = 0; i < 3; ++i) c1[i] -= d * c0[i];
    norm(c1);
    c2[0] = c0[1] * c1[2] - c0[2] * c1[1];
    c2[1] = c0[2] * c1[0] - c0[0] * c1[2];
    c2[2] = c0[0] * c1[1] - c0[1] * c1[0];
    for (int i = 0; i < 3; ++i) {
        e.rotation[0 + i] = c0[i];
        e.rotation[3 + i] = c1[i];
        e.rotation[6 + i] = c2[i];
    }
    e.translation[0] = -0.0591f;
    e.translation[1] = 0.00012f;
    e.translation[2] = 0.00041f;
    return e;
}
inline Extrinsics shift_x(float tx) {
    Extrinsics e;
    e.translation[0] = tx;
    return e;
}

// ---- 기하 ----
struct Plane {  // n·p = d, 범위 제한 없음
    double n[3], d;
};
struct Box {  // 축 정렬 상자(깊이 카메라 좌표)
    double lo[3], hi[3];
};
struct Scene {
    std::vector<Plane> planes;
    std::vector<Box> boxes;
    // 원점 o, 방향 v 의 광선이 처음 닿는 거리 t(없으면 -1)
    double hit(const double o[3], const double v[3]) const {
        double best = 1e30;
        for (const auto& p : planes) {
            double nv = p.n[0] * v[0] + p.n[1] * v[1] + p.n[2] * v[2];
            if (std::fabs(nv) < 1e-12) continue;
            double t = (p.d - (p.n[0] * o[0] + p.n[1] * o[1] + p.n[2] * o[2])) / nv;
            if (t > 1e-6 && t < best) best = t;
        }
        for (const auto& b : boxes) {
            double t0 = -1e30, t1 = 1e30;
            bool ok = true;
            for (int i = 0; i < 3 && ok; ++i) {
                if (std::fabs(v[i]) < 1e-15) {
                    if (o[i] < b.lo[i] || o[i] > b.hi[i]) ok = false;
                    continue;
                }
                double a = (b.lo[i] - o[i]) / v[i], c = (b.hi[i] - o[i]) / v[i];
                if (a > c) std::swap(a, c);
                t0 = std::max(t0, a);
                t1 = std::min(t1, c);
            }
            if (ok && t1 >= t0 && t0 > 1e-6 && t0 < best) best = t0;
        }
        return best < 1e29 ? best : -1;
    }
};

// 방: 뒷벽(살짝 기울어짐) + 바닥 + 상자 두 개(가까운 것은 0.6 m)
inline Scene room() {
    Scene s;
    s.planes.push_back({{0.05, 0.0, 0.9987}, 3.2});   // 뒷벽
    s.planes.push_back({{0.0, -1.0, 0.0}, -0.75});    // 바닥(y = 0.75 m 아래)
    s.boxes.push_back({{-0.45, -0.25, 1.4}, {0.05, 0.75, 1.9}});  // 책상 크기 상자
    s.boxes.push_back({{0.15, -0.05, 0.62}, {0.32, 0.20, 0.75}}); // 가까운 물건
    return s;
}
// 정면 평면 z = d
inline Scene front_plane(double d) {
    Scene s;
    s.planes.push_back({{0, 0, 1}, d});
    return s;
}
// 계단: 왼쪽 절반 z=1.0, 오른쪽 z=2.0(앞 판이 화면 왼쪽 절반을 가림) + 뒷벽 2.0
inline Scene step() {
    Scene s;
    s.planes.push_back({{0, 0, 1}, 2.0});
    s.boxes.push_back({{-10.0, -10.0, 1.0}, {0.0, 10.0, 1.01}});
    return s;
}

// 깊이 카메라 화소 중심 광선으로 깊이(z, m) 그리기. 왜곡 없는 깊이 카메라로 봄
inline std::vector<uint16_t> render_depth(const Scene& s, const Intrinsics& di, float unit, double noise_rel = 0.0,
                                          double hole_frac = 0.0, uint32_t seed = 1) {
    std::vector<uint16_t> z((size_t)di.width * di.height, 0);
    uint32_t st = seed * 2654435761u + 12345u;
    auto rnd = [&]() {  // [0,1)
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        return (st & 0xFFFFFF) / 16777216.0;
    };
    const double o[3] = {0, 0, 0};
    for (int y = 0; y < di.height; ++y)
        for (int x = 0; x < di.width; ++x) {
            double v[3] = {(x - di.ppx) / di.fx, (y - di.ppy) / di.fy, 1.0};
            double t = s.hit(o, v);
            double r1 = rnd(), r2 = rnd(), r3 = rnd();
            if (t <= 0 || t > 10.0) continue;
            if (r1 < hole_frac) continue;
            double zz = t;  // v[2]=1 이라 t = z
            if (noise_rel > 0) {
                double g = std::sqrt(-2.0 * std::log(std::max(r2, 1e-12))) * std::cos(6.283185307 * r3);
                zz *= 1.0 + noise_rel * g;
            }
            long q = std::lround(zz / unit);
            z[(size_t)y * di.width + x] = (uint16_t)std::min(std::max(q, 0L), 65535L);
        }
    return z;
}

// 정답: 컬러 화소 중심이 보는 면의 점을 깊이 카메라 좌표 z 로(정렬 결과가 담는 값과 같은 뜻). 0 = 깊이 카메라에서 안 보임
// (컬러에서 보이는 점이 깊이 카메라에서 가려졌는지는 깊이 카메라 쪽 광선으로 다시 확인).
inline std::vector<float> truth_in_color(const Scene& s, const Intrinsics& di, const Intrinsics& ci, const Extrinsics& d2c) {
    std::vector<float> out((size_t)ci.width * ci.height, 0.f);
    // 컬러 → 깊이: p_d = R^T (p_c - t)
    const float* R = d2c.rotation;  // 열 우선: p_c = R p_d + t, R(i,j) = rotation[j*3+i]
    double oc[3];
    for (int i = 0; i < 3; ++i) oc[i] = -(R[i * 3 + 0] * d2c.translation[0] + R[i * 3 + 1] * d2c.translation[1] + R[i * 3 + 2] * d2c.translation[2]);
    for (int y = 0; y < ci.height; ++y)
        for (int x = 0; x < ci.width; ++x) {
            float px[2] = {(float)x, (float)y}, ray[3];
            rsref::rs2_deproject_pixel_to_point(ray, &ci, px, 1.0f);  // 컬러 왜곡 풀기(rsutil 규칙)
            double vd[3];
            for (int i = 0; i < 3; ++i) vd[i] = R[i * 3 + 0] * ray[0] + R[i * 3 + 1] * ray[1] + R[i * 3 + 2] * ray[2];  // R^T v
            double t = s.hit(oc, vd);
            if (t <= 0) continue;
            double p[3] = {oc[0] + t * vd[0], oc[1] + t * vd[1], oc[2] + t * vd[2]};
            if (p[2] <= 0) continue;
            // 깊이 카메라에서 보이는지(같은 점까지 막힘 없음)
            const double o0[3] = {0, 0, 0};
            double v2[3] = {p[0] / p[2], p[1] / p[2], 1.0};
            double t2 = s.hit(o0, v2);
            if (t2 <= 0 || t2 < p[2] - 1e-4) continue;
            // 깊이 영상 안에 들어오는지(화소 중심 기준 반 화소까지)
            const double ud = di.fx * v2[0] + di.ppx, vdd = di.fy * v2[1] + di.ppy;
            if (ud < -0.5 || vdd < -0.5 || ud >= di.width - 0.5 || vdd >= di.height - 0.5) continue;
            out[(size_t)y * ci.width + x] = (float)p[2];
        }
    return out;
}

// 무작위 깊이(겹침·경계 최악 조건): 덩어리 + 잡음 + 구멍
inline std::vector<uint16_t> random_depth(const Intrinsics& di, uint32_t seed) {
    std::vector<uint16_t> z((size_t)di.width * di.height);
    uint32_t st = seed * 747796405u + 2891336453u;
    auto rnd = [&]() {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        return st;
    };
    for (size_t i = 0; i < z.size(); ++i) {
        uint32_t r = rnd();
        z[i] = (r % 17 == 0) ? 0 : (uint16_t)(250 + (r >> 8) % 12000);
    }
    return z;
}

}  // namespace scenes
