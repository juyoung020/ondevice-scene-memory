// librealsense v2.59.1 의 정렬 코드를 시험·비교용으로 옮겨 온 것(느리지만 SDK 와 같은 결과의 기준).
//
// 출처(Apache License 2.0, Copyright(c) 2015-2024 RealSense, Inc.):
//   https://github.com/realsenseai/librealsense/blob/v2.59.1/src/rs.cpp          rs2_project_point_to_pixel,
//                                                                                 rs2_deproject_pixel_to_point,
//                                                                                 rs2_transform_point_to_point
//   https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/align.cpp   align_images, align_z_to_other
//   https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/sse/sse-align.cpp   (x86 SIMD 경로)
//   https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/neon/neon-align.cpp (ARM NEON 경로)
// 바꾼 점: 프레임·프로파일 객체 대신 보정값과 버퍼를 직접 받음, OpenMP 줄 없음, rs2 타입 이름을 fast_align 타입에 연결.
// 계산식과 연산 순서는 원본 그대로 둔다(비트 단위 비교용).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fast_align/fast_align.hpp"

#if defined(__SSSE3__) || defined(_M_X64) || defined(__x86_64__)
#define RSREF_SSE 1
#include <tmmintrin.h>
#endif
#if defined(__aarch64__)
#define RSREF_NEON 1
#include <arm_neon.h>
#endif

namespace rsref {

using rs2_intrinsics = fast_align::Intrinsics;
using rs2_extrinsics = fast_align::Extrinsics;
using rs2_distortion = fast_align::Distortion;
constexpr rs2_distortion RS2_DISTORTION_NONE = rs2_distortion::None;
constexpr rs2_distortion RS2_DISTORTION_MODIFIED_BROWN_CONRADY = rs2_distortion::ModifiedBrownConrady;
constexpr rs2_distortion RS2_DISTORTION_INVERSE_BROWN_CONRADY = rs2_distortion::InverseBrownConrady;
constexpr rs2_distortion RS2_DISTORTION_FTHETA = rs2_distortion::FTheta;
constexpr rs2_distortion RS2_DISTORTION_BROWN_CONRADY = rs2_distortion::BrownConrady;
constexpr rs2_distortion RS2_DISTORTION_KANNALA_BRANDT4 = rs2_distortion::KannalaBrandt4;
constexpr float RS_FLT_EPSILON = 1.1920929e-07f;

// ---- src/rs.cpp ----
inline void rs2_project_point_to_pixel(float pixel[2], const rs2_intrinsics* intrin, const float point[3]) {
    float x = point[0] / point[2], y = point[1] / point[2];

    if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
        (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)) {
        float r2 = x * x + y * y;
        float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
        x *= f;
        y *= f;
        float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
        float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
        x = dx;
        y = dy;
    }

    if (intrin->model == RS2_DISTORTION_BROWN_CONRADY) {
        float r2 = x * x + y * y;
        float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;

        float xf = x * f;
        float yf = y * f;

        float dx = xf + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
        float dy = yf + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);

        x = dx;
        y = dy;
    }

    pixel[0] = x * intrin->fx + intrin->ppx;
    pixel[1] = y * intrin->fy + intrin->ppy;
}

inline bool is_intrinsics_distortion_zero(const rs2_intrinsics* intrin) {
    return (std::fabs(intrin->coeffs[0]) < RS_FLT_EPSILON && std::fabs(intrin->coeffs[1]) < RS_FLT_EPSILON &&
            std::fabs(intrin->coeffs[2]) < RS_FLT_EPSILON && std::fabs(intrin->coeffs[3]) < RS_FLT_EPSILON &&
            std::fabs(intrin->coeffs[4]) < RS_FLT_EPSILON);
}

inline void rs2_deproject_pixel_to_point(float point[3], const rs2_intrinsics* intrin, const float pixel[2], float depth) {
    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;

    float xo = x;
    float yo = y;

    if (!is_intrinsics_distortion_zero(intrin)) {
        if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) {
            for (int i = 0; i < 10; i++) {
                float r2 = x * x + y * y;
                float icdist = (float)1 / (float)(1 + ((intrin->coeffs[4] * r2 + intrin->coeffs[1]) * r2 + intrin->coeffs[0]) * r2);
                float xq = x / icdist;
                float yq = y / icdist;
                float delta_x = 2 * intrin->coeffs[2] * xq * yq + intrin->coeffs[3] * (r2 + 2 * xq * xq);
                float delta_y = 2 * intrin->coeffs[3] * xq * yq + intrin->coeffs[2] * (r2 + 2 * yq * yq);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
        if (intrin->model == RS2_DISTORTION_BROWN_CONRADY) {
            for (int i = 0; i < 10; i++) {
                float r2 = x * x + y * y;
                float icdist = (float)1 / (float)(1 + ((intrin->coeffs[4] * r2 + intrin->coeffs[1]) * r2 + intrin->coeffs[0]) * r2);
                float delta_x = 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
                float delta_y = 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
    }

    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

inline void rs2_transform_point_to_point(float to_point[3], const rs2_extrinsics* extrin, const float from_point[3]) {
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

inline void rs2_fov(const rs2_intrinsics* intrin, float to_fov[2]) {
    to_fov[0] = (atan2f(intrin->ppx + 0.5f, intrin->fx) + atan2f(intrin->width - (intrin->ppx + 0.5f), intrin->fx)) * 57.2957795f;
    to_fov[1] = (atan2f(intrin->ppy + 0.5f, intrin->fy) + atan2f(intrin->height - (intrin->ppy + 0.5f), intrin->fy)) * 57.2957795f;
}

// ---- src/proc/align.cpp (일반 경로) ----
template <class GET_DEPTH, class TRANSFER_PIXEL>
void align_images(const rs2_intrinsics& depth_intrin, const rs2_extrinsics& depth_to_other,
                  const rs2_intrinsics& other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel) {
    for (int depth_y = 0; depth_y < depth_intrin.height; ++depth_y) {
        int depth_pixel_index = depth_y * depth_intrin.width;
        for (int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index) {
            if (float depth = get_depth(depth_pixel_index)) {
                float depth_pixel[2] = {depth_x - 0.5f, depth_y - 0.5f}, depth_point[3], other_point[3], other_pixel[2];
                rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
                rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

                depth_pixel[0] = depth_x + 0.5f;
                depth_pixel[1] = depth_y + 0.5f;
                rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
                rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

                if (other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height)
                    continue;

                for (int y = other_y0; y <= other_y1; ++y)
                    for (int x = other_x0; x <= other_x1; ++x)
                        transfer_pixel(depth_pixel_index, y * other_intrin.width + x);
            }
        }
    }
}

inline void align_z_to_other_generic(uint16_t* out_z, const uint16_t* z_pixels, float z_scale, const rs2_intrinsics& z_intrin,
                                     const rs2_extrinsics& z_to_other, const rs2_intrinsics& other_intrin) {
    std::memset(out_z, 0, sizeof(uint16_t) * other_intrin.width * other_intrin.height);
    align_images(
        z_intrin, z_to_other, other_intrin, [z_pixels, z_scale](int z_pixel_index) { return z_scale * z_pixels[z_pixel_index]; },
        [out_z, z_pixels](int z_pixel_index, int other_pixel_index) {
            out_z[other_pixel_index] = out_z[other_pixel_index]
                                           ? std::min((int)out_z[other_pixel_index], (int)z_pixels[z_pixel_index])
                                           : z_pixels[z_pixel_index];
        });
}

// ---- src/proc/sse/sse-align.cpp · src/proc/neon/neon-align.cpp (SIMD 경로) ----
struct int2 {
    int x, y;
};

inline bool is_special_resolution(const rs2_intrinsics& depth, const rs2_intrinsics& to) {
    if ((depth.width == 640 && depth.height == 240 && to.width == 320 && to.height == 180) ||
        (depth.width == 640 && depth.height == 480 && to.width == 640 && to.height == 360))
        return true;
    return false;
}

#if RSREF_SSE
template <rs2_distortion dist>
inline void distorte_x_y(const __m128& x, const __m128& y, __m128* distorted_x, __m128* distorted_y, const rs2_intrinsics& to) {
    *distorted_x = x;
    *distorted_y = y;
}
template <>
inline void distorte_x_y<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(const __m128& x, const __m128& y, __m128* distorted_x,
                                                                 __m128* distorted_y, const rs2_intrinsics& to) {
    __m128 c[5];
    auto two = _mm_set_ps1(2);
    auto one = _mm_set_ps1(1);
    for (int i = 0; i < 5; ++i) c[i] = _mm_set_ps1(to.coeffs[i]);
    auto r2_0 = _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y));
    auto r3_0 = _mm_add_ps(_mm_mul_ps(c[1], _mm_mul_ps(r2_0, r2_0)), _mm_mul_ps(c[4], _mm_mul_ps(r2_0, _mm_mul_ps(r2_0, r2_0))));
    auto f_0 = _mm_add_ps(one, _mm_add_ps(_mm_mul_ps(c[0], r2_0), r3_0));
    auto x_f0 = _mm_mul_ps(x, f_0);
    auto y_f0 = _mm_mul_ps(y, f_0);
    auto r4_0 = _mm_mul_ps(c[3], _mm_add_ps(r2_0, _mm_mul_ps(two, _mm_mul_ps(x_f0, x_f0))));
    auto d_x0 = _mm_add_ps(x_f0, _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(c[2], _mm_mul_ps(x_f0, y_f0))), r4_0));
    auto r5_0 = _mm_mul_ps(c[2], _mm_add_ps(r2_0, _mm_mul_ps(two, _mm_mul_ps(y_f0, y_f0))));
    auto d_y0 = _mm_add_ps(y_f0, _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(c[3], _mm_mul_ps(x_f0, y_f0))), r5_0));
    *distorted_x = d_x0;
    *distorted_y = d_y0;
}

template <rs2_distortion dist>
inline void get_texture_map_simd(const uint16_t* depth, float depth_scale, const unsigned int size, const float* pre_compute_x,
                                 const float* pre_compute_y, uint8_t* pixels_ptr_int, const rs2_intrinsics& to,
                                 const rs2_extrinsics& from_to_other) {
    const __m128i mask0 = _mm_set_epi8((char)0xff, (char)0xff, (char)7, (char)6, (char)0xff, (char)0xff, (char)5, (char)4,
                                       (char)0xff, (char)0xff, (char)3, (char)2, (char)0xff, (char)0xff, (char)1, (char)0);
    const __m128i mask1 = _mm_set_epi8((char)0xff, (char)0xff, (char)15, (char)14, (char)0xff, (char)0xff, (char)13, (char)12,
                                       (char)0xff, (char)0xff, (char)11, (char)10, (char)0xff, (char)0xff, (char)9, (char)8);
    auto scale = _mm_set_ps1(depth_scale);
    auto mapx = pre_compute_x;
    auto mapy = pre_compute_y;
    auto res = reinterpret_cast<__m128i*>(pixels_ptr_int);
    __m128 r[9];
    __m128 t[3];
    for (int i = 0; i < 9; ++i) r[i] = _mm_set_ps1(from_to_other.rotation[i]);
    for (int i = 0; i < 3; ++i) t[i] = _mm_set_ps1(from_to_other.translation[i]);
    auto zero = _mm_set_ps1(0);
    auto fx = _mm_set_ps1(to.fx);
    auto fy = _mm_set_ps1(to.fy);
    auto ppx = _mm_set_ps1(to.ppx);
    auto ppy = _mm_set_ps1(to.ppy);

    for (unsigned int i = 0; i < size; i += 8) {
        auto x0 = _mm_load_ps(mapx + i);
        auto x1 = _mm_load_ps(mapx + i + 4);
        auto y0 = _mm_load_ps(mapy + i);
        auto y1 = _mm_load_ps(mapy + i + 4);
        __m128i d = _mm_load_si128((__m128i const*)(depth + i));
        __m128i d0 = _mm_shuffle_epi8(d, mask0);
        __m128i d1 = _mm_shuffle_epi8(d, mask1);
        __m128 depth0 = _mm_cvtepi32_ps(d0);
        __m128 depth1 = _mm_cvtepi32_ps(d1);
        depth0 = _mm_mul_ps(depth0, scale);
        depth1 = _mm_mul_ps(depth1, scale);
        auto p0x = _mm_mul_ps(depth0, x0);
        auto p0y = _mm_mul_ps(depth0, y0);
        auto p1x = _mm_mul_ps(depth1, x1);
        auto p1y = _mm_mul_ps(depth1, y1);
        auto p_x0 = _mm_add_ps(_mm_mul_ps(r[0], p0x), _mm_add_ps(_mm_mul_ps(r[3], p0y), _mm_add_ps(_mm_mul_ps(r[6], depth0), t[0])));
        auto p_y0 = _mm_add_ps(_mm_mul_ps(r[1], p0x), _mm_add_ps(_mm_mul_ps(r[4], p0y), _mm_add_ps(_mm_mul_ps(r[7], depth0), t[1])));
        auto p_z0 = _mm_add_ps(_mm_mul_ps(r[2], p0x), _mm_add_ps(_mm_mul_ps(r[5], p0y), _mm_add_ps(_mm_mul_ps(r[8], depth0), t[2])));
        auto p_x1 = _mm_add_ps(_mm_mul_ps(r[0], p1x), _mm_add_ps(_mm_mul_ps(r[3], p1y), _mm_add_ps(_mm_mul_ps(r[6], depth1), t[0])));
        auto p_y1 = _mm_add_ps(_mm_mul_ps(r[1], p1x), _mm_add_ps(_mm_mul_ps(r[4], p1y), _mm_add_ps(_mm_mul_ps(r[7], depth1), t[1])));
        auto p_z1 = _mm_add_ps(_mm_mul_ps(r[2], p1x), _mm_add_ps(_mm_mul_ps(r[5], p1y), _mm_add_ps(_mm_mul_ps(r[8], depth1), t[2])));
        p_x0 = _mm_div_ps(p_x0, p_z0);
        p_y0 = _mm_div_ps(p_y0, p_z0);
        p_x1 = _mm_div_ps(p_x1, p_z1);
        p_y1 = _mm_div_ps(p_y1, p_z1);
        distorte_x_y<dist>(p_x0, p_y0, &p_x0, &p_y0, to);
        distorte_x_y<dist>(p_x1, p_y1, &p_x1, &p_y1, to);
        auto cmp = _mm_cmpneq_ps(depth0, zero);
        p_x0 = _mm_and_ps(_mm_add_ps(_mm_mul_ps(p_x0, fx), ppx), cmp);
        p_y0 = _mm_and_ps(_mm_add_ps(_mm_mul_ps(p_y0, fy), ppy), cmp);
        p_x1 = _mm_add_ps(_mm_mul_ps(p_x1, fx), ppx);
        p_y1 = _mm_add_ps(_mm_mul_ps(p_y1, fy), ppy);
        cmp = _mm_cmpneq_ps(depth0, zero);
        auto half = _mm_set_ps1(0.5);
        auto u_round0 = _mm_and_ps(_mm_add_ps(p_x0, half), cmp);
        auto v_round0 = _mm_and_ps(_mm_add_ps(p_y0, half), cmp);
        auto uuvv1_0 = _mm_shuffle_ps(u_round0, v_round0, _MM_SHUFFLE(1, 0, 1, 0));
        auto uuvv2_0 = _mm_shuffle_ps(u_round0, v_round0, _MM_SHUFFLE(3, 2, 3, 2));
        auto res1_0 = _mm_shuffle_ps(uuvv1_0, uuvv1_0, _MM_SHUFFLE(3, 1, 2, 0));
        auto res2_0 = _mm_shuffle_ps(uuvv2_0, uuvv2_0, _MM_SHUFFLE(3, 1, 2, 0));
        auto res1_int0 = _mm_cvttps_epi32(res1_0);
        auto res2_int0 = _mm_cvttps_epi32(res2_0);
        _mm_stream_si128(&res[0], res1_int0);
        _mm_stream_si128(&res[1], res2_int0);
        res += 2;
        cmp = _mm_cmpneq_ps(depth1, zero);
        auto u_round1 = _mm_and_ps(_mm_add_ps(p_x1, half), cmp);
        auto v_round1 = _mm_and_ps(_mm_add_ps(p_y1, half), cmp);
        auto uuvv1_1 = _mm_shuffle_ps(u_round1, v_round1, _MM_SHUFFLE(1, 0, 1, 0));
        auto uuvv2_1 = _mm_shuffle_ps(u_round1, v_round1, _MM_SHUFFLE(3, 2, 3, 2));
        auto res1 = _mm_shuffle_ps(uuvv1_1, uuvv1_1, _MM_SHUFFLE(3, 1, 2, 0));
        auto res2 = _mm_shuffle_ps(uuvv2_1, uuvv2_1, _MM_SHUFFLE(3, 1, 2, 0));
        auto res1_int1 = _mm_cvttps_epi32(res1);
        auto res2_int1 = _mm_cvttps_epi32(res2);
        _mm_stream_si128(&res[0], res1_int1);
        _mm_stream_si128(&res[1], res2_int1);
        res += 2;
    }
}
#endif

#if RSREF_NEON
template <rs2_distortion dist>
static inline void distorte_x_y(const float32x4_t& x, const float32x4_t& y, float32x4_t* distorted_x, float32x4_t* distorted_y,
                                const float32x4_t (&c)[5]) {
    *distorted_x = x;
    *distorted_y = y;
}
template <>
inline void distorte_x_y<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(const float32x4_t& x, const float32x4_t& y, float32x4_t* distorted_x,
                                                                 float32x4_t* distorted_y, const float32x4_t (&c)[5]) {
    const auto one = vdupq_n_f32(1);
    const auto two = vdupq_n_f32(2);
    const auto r2 = vaddq_f32(vmulq_f32(x, x), vmulq_f32(y, y));
    const auto f = vfmaq_f32(one, r2, vfmaq_f32(c[0], r2, vfmaq_f32(c[1], r2, c[4])));
    const auto x_f = vmulq_f32(x, f);
    const auto y_f = vmulq_f32(y, f);
    *distorted_x = vfmaq_f32(vmulq_f32(x_f, vfmaq_f32(one, two, vfmaq_f32(vmulq_f32(c[2], y_f), c[3], x_f))), c[3], r2);
    *distorted_y = vfmaq_f32(vmulq_f32(y_f, vfmaq_f32(one, two, vfmaq_f32(vmulq_f32(c[3], x_f), c[2], y_f))), c[2], r2);
}

template <rs2_distortion dist>
static inline void get_texture_map_simd(const uint16_t* depth, float depth_scale, const unsigned int size, const float* pre_compute_x,
                                        const float* pre_compute_y, uint8_t* pixels_ptr_int, const rs2_intrinsics& to,
                                        const rs2_extrinsics& from_to_other) {
    auto res = reinterpret_cast<int32_t*>(pixels_ptr_int);
    float32x4_t r[9];
    float32x4_t t[3];
    float32x4_t c[5];
    for (int i = 0; i < 9; ++i) r[i] = vdupq_n_f32(from_to_other.rotation[i]);
    for (int i = 0; i < 3; ++i) t[i] = vdupq_n_f32(from_to_other.translation[i]);
    for (int i = 0; i < 5; ++i) c[i] = vdupq_n_f32(to.coeffs[i]);
    const auto zero = vdupq_n_f32(0.0f);
    const auto half = vdupq_n_f32(0.5f);
    const auto fx = vdupq_n_f32(to.fx);
    const auto fy = vdupq_n_f32(to.fy);
    const auto ppx = vdupq_n_f32(to.ppx);
    const auto ppy = vdupq_n_f32(to.ppy);
    const auto scale = vdupq_n_f32(depth_scale);
    for (unsigned int i = 0; i < size; i += 8) {
        const auto x0 = vld1q_f32(pre_compute_x + i);
        const auto x1 = vld1q_f32(pre_compute_x + i + 4);
        const auto y0 = vld1q_f32(pre_compute_y + i);
        const auto y1 = vld1q_f32(pre_compute_y + i + 4);
        const auto d = vld1q_u16(depth + i);
        const auto depth0 = vmulq_f32(vcvtq_f32_s32((int32x4_t)vmovl_u16(vget_low_u16(d))), scale);
        const auto depth1 = vmulq_f32(vcvtq_f32_s32((int32x4_t)vmovl_u16(vget_high_u16(d))), scale);
        const auto p0x = vmulq_f32(depth0, x0);
        const auto p0y = vmulq_f32(depth0, y0);
        const auto p1x = vmulq_f32(depth1, x1);
        const auto p1y = vmulq_f32(depth1, y1);
        auto p_x0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[0], r[6], depth0), r[3], p0y), r[0], p0x);
        auto p_y0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[1], r[7], depth0), r[4], p0y), r[1], p0x);
        auto p_z0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[2], r[8], depth0), r[5], p0y), r[2], p0x);
        auto p_x1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[0], r[6], depth1), r[3], p1y), r[0], p1x);
        auto p_y1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[1], r[7], depth1), r[4], p1y), r[1], p1x);
        auto p_z1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(t[2], r[8], depth1), r[5], p1y), r[2], p1x);
        p_x0 = vdivq_f32(p_x0, p_z0);
        p_y0 = vdivq_f32(p_y0, p_z0);
        p_x1 = vdivq_f32(p_x1, p_z1);
        p_y1 = vdivq_f32(p_y1, p_z1);
        distorte_x_y<dist>(p_x0, p_y0, &p_x0, &p_y0, c);
        distorte_x_y<dist>(p_x1, p_y1, &p_x1, &p_y1, c);
        p_x0 = vfmaq_f32(ppx, p_x0, fx);
        p_y0 = vfmaq_f32(ppy, p_y0, fy);
        p_x1 = vfmaq_f32(ppx, p_x1, fx);
        p_y1 = vfmaq_f32(ppy, p_y1, fy);
        {
            const uint32x4_t gt_zero = vcgtq_f32(depth0, zero);
            const auto u = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vaddq_f32(p_x0, half)), gt_zero));
            const auto v = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vaddq_f32(p_y0, half)), gt_zero));
            int32x4x2_t res_int;
            res_int.val[0] = vcvtq_s32_f32(u);
            res_int.val[1] = vcvtq_s32_f32(v);
            vst2q_s32(res, res_int);
            res += 8;
        }
        {
            const uint32x4_t gt_zero = vcgtq_f32(depth1, zero);
            const auto u = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vaddq_f32(p_x1, half)), gt_zero));
            const auto v = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vaddq_f32(p_y1, half)), gt_zero));
            int32x4x2_t res_int;
            res_int.val[0] = vcvtq_s32_f32(u);
            res_int.val[1] = vcvtq_s32_f32(v);
            vst2q_s32(res, res_int);
            res += 8;
        }
    }
}
#endif

// image_transform / align_neon_helper: 표를 한 번 만들고 장면마다 align_depth_to_other
class SimdAlign {
public:
    SimdAlign(const rs2_intrinsics& from, float depth_scale)
        : _depth(from), _depth_scale(depth_scale), _pixel_top_left_int(from.width * from.height),
          _pixel_bottom_right_int(from.width * from.height) {
        pre_compute_x_y_map(_x_tl, _y_tl, -0.5f);
        pre_compute_x_y_map(_x_br, _y_br, 0.5f);
    }

    void align_depth_to_other(const uint16_t* z_pixels, uint16_t* dest, const rs2_intrinsics& to, const rs2_extrinsics& from_to_other) {
        std::memset(dest, 0, sizeof(uint16_t) * to.width * to.height);
        if (to.model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY)
            run<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(z_pixels, dest, to, from_to_other);
        else
            run<RS2_DISTORTION_NONE>(z_pixels, dest, to, from_to_other);
    }

    bool last_rect = false;

private:
    void pre_compute_x_y_map(std::vector<float>& mx, std::vector<float>& my, float offset) {
        mx.resize(_depth.width * _depth.height);
        my.resize(_depth.width * _depth.height);
        for (int h = 0; h < _depth.height; ++h)
            for (int w = 0; w < _depth.width; ++w) {
                const float pixel[] = {(float)w + offset, (float)h + offset};
                float x = (pixel[0] - _depth.ppx) / _depth.fx;
                float y = (pixel[1] - _depth.ppy) / _depth.fy;
                if (_depth.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) {
#if RSREF_NEON
                    const float r2 = x * x + y * y;
                    const float f = 1.0f + (_depth.coeffs[0] + (_depth.coeffs[1] + _depth.coeffs[4] * r2) * r2) * r2;
                    const float ux = x * f + 2.0f * _depth.coeffs[2] * x * y + _depth.coeffs[3] * (r2 + 2.0f * x * x);
                    const float uy = y * f + 2.0f * _depth.coeffs[3] * x * y + _depth.coeffs[2] * (r2 + 2.0f * y * y);
#else
                    float r2 = x * x + y * y;
                    float f = 1 + _depth.coeffs[0] * r2 + _depth.coeffs[1] * r2 * r2 + _depth.coeffs[4] * r2 * r2 * r2;
                    float ux = x * f + 2 * _depth.coeffs[2] * x * y + _depth.coeffs[3] * (r2 + 2 * x * x);
                    float uy = y * f + 2 * _depth.coeffs[3] * x * y + _depth.coeffs[2] * (r2 + 2 * y * y);
#endif
                    x = ux;
                    y = uy;
                }
                mx[h * _depth.width + w] = x;
                my[h * _depth.width + w] = y;
            }
    }

    void move_depth_to_other(const uint16_t* z_pixels, uint16_t* dest, const rs2_intrinsics& to, const std::vector<int2>& tl,
                             const std::vector<int2>& br) {
        for (int y = 0; y < _depth.height; ++y)
            for (int x = 0; x < _depth.width; ++x) {
                auto depth_pixel_index = y * _depth.width + x;
                if (z_pixels[depth_pixel_index]) {
                    for (int other_y = tl[depth_pixel_index].y; other_y <= br[depth_pixel_index].y; ++other_y) {
                        if (other_y < 0 || other_y >= to.height) continue;
                        const auto y_width = other_y * to.width;
                        for (int other_x = tl[depth_pixel_index].x; other_x <= br[depth_pixel_index].x; ++other_x) {
                            if (other_x < 0 || other_x >= to.width) continue;
                            const auto other_ind = y_width + other_x;
                            dest[other_ind] = dest[other_ind] ? std::min(dest[other_ind], z_pixels[depth_pixel_index])
                                                              : z_pixels[depth_pixel_index];
                        }
                    }
                }
            }
    }

    template <rs2_distortion dist>
    void run(const uint16_t* z_pixels, uint16_t* dest, const rs2_intrinsics& to, const rs2_extrinsics& from_to_other) {
#if RSREF_SSE || RSREF_NEON
        get_texture_map_simd<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _x_tl.data(), _y_tl.data(),
                                   (uint8_t*)_pixel_top_left_int.data(), to, from_to_other);
        float fov[2];
        rs2_fov(&_depth, fov);
        float ppa_d[2] = {(float)_depth.width / fov[0], (float)_depth.height / fov[1]};
        rs2_fov(&to, fov);
        float ppa_t[2] = {(float)to.width / fov[0], (float)to.height / fov[1]};
        if (ppa_d[0] < ppa_t[0] || ppa_d[1] < ppa_t[1] || is_special_resolution(_depth, to)) {
            last_rect = true;
            get_texture_map_simd<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _x_br.data(), _y_br.data(),
                                       (uint8_t*)_pixel_bottom_right_int.data(), to, from_to_other);
            move_depth_to_other(z_pixels, dest, to, _pixel_top_left_int, _pixel_bottom_right_int);
        } else {
            last_rect = false;
            move_depth_to_other(z_pixels, dest, to, _pixel_top_left_int, _pixel_top_left_int);
        }
#endif
    }

    rs2_intrinsics _depth;
    float _depth_scale;
    std::vector<float> _x_tl, _y_tl, _x_br, _y_br;  // 원본은 std::vector(16 바이트 정렬 할당자에 기댐)
    std::vector<int2> _pixel_top_left_int, _pixel_bottom_right_int;
};

}  // namespace rsref
