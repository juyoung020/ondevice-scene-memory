// 깊이를 컬러 화소 격자로 옮기는 곳(한 곳). 다른 정렬 구현으로 바꿀 때는 make_aligner 에 이름 하나를 더한다.
#pragma once
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <string>

class DepthAligner {
public:
    virtual ~DepthAligner() = default;
    // fs: 같은 시각의 컬러·깊이 장면 묶음. 돌려줌: 컬러와 같은 크기의 CV_16UC1 깊이(카메라 깊이 단위, 0 = 없음)
    virtual cv::Mat align(const rs2::frameset& fs) = 0;
    virtual const char* name() const = 0;
};

// "rs2": librealsense 의 rs2::align(CPU)
std::unique_ptr<DepthAligner> make_aligner(const std::string& name);
