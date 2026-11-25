// LD_PRELOAD 로 ORB-SLAM3 의 ORBextractor::operator() 를 바꿔, 미리 만든 점 · 기술자 덤프(orbnpu_bench dump/maps 형식)를
// 장면 순서대로 내준다. NPU 지도로 만든 특징이 최종 궤적을 얼마나 바꾸는지 원본 rgbd_tum 그대로 재려는 시험 도구.
// 환경 변수 ORBNPU_REPLAY=<덤프>. RGB-D(왼쪽 추출기 한 번 / 장면)만.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <list>
#include <cstdio>
#include <cstdlib>
#define protected public
#include "ORBextractor.h"
#undef protected

namespace ORB_SLAM3 {
int ORBextractor::operator()(cv::InputArray _image, cv::InputArray, std::vector<cv::KeyPoint>& _keypoints,
                             cv::OutputArray _descriptors, std::vector<int>&) {
    static FILE* f = nullptr;
    static int frame = 0;
    if (!f) {
        const char* p = getenv("ORBNPU_REPLAY");
        f = p ? fopen(p, "rb") : nullptr;
        if (!f) { fprintf(stderr, "ORBNPU_REPLAY 덤프를 열 수 없음\n"); abort(); }
    }
    ComputePyramid(_image.getMat());  // 피라미드는 원본 그대로(다른 곳에서 쓸 때를 위해)
    int n = 0;
    if (fread(&n, 4, 1, f) != 1) { fprintf(stderr, "덤프가 장면 %d 에서 끝남\n", frame); abort(); }
    _keypoints.assign(n, cv::KeyPoint());
    std::vector<float> r(5);
    for (int i = 0; i < n; ++i) {
        if (fread(r.data(), 4, 5, f) != 5) abort();
        _keypoints[i] = cv::KeyPoint(r[0], r[1], (float)(int)(31 * mvScaleFactor[(int)r[4]]), r[2], r[3], (int)r[4]);
    }
    if (n) {
        _descriptors.create(n, 32, CV_8U);
        cv::Mat d = _descriptors.getMat();
        if (fread(d.data, 1, (size_t)n * 32, f) != (size_t)n * 32) abort();
    } else _descriptors.release();
    ++frame;
    return n;
}
}  // namespace ORB_SLAM3
