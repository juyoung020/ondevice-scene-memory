#include "depth_align.h"
#include <stdexcept>

namespace {

class Rs2Aligner : public DepthAligner {
public:
    Rs2Aligner() : align_(RS2_STREAM_COLOR) {}
    cv::Mat align(const rs2::frameset& fs) override {
        rs2::frameset a = align_.process(fs);
        rs2::depth_frame d = a.get_depth_frame();
        // rs2 프레임 메모리를 감싼 뒤 복사(프레임이 풀로 돌아가도 안전하게)
        cv::Mat m(d.get_height(), d.get_width(), CV_16UC1, const_cast<void*>(d.get_data()), (size_t)d.get_stride_in_bytes());
        return m.clone();
    }
    const char* name() const override { return "rs2"; }

private:
    rs2::align align_;
};

}  // namespace

std::unique_ptr<DepthAligner> make_aligner(const std::string& name) {
    if (name == "rs2") return std::make_unique<Rs2Aligner>();
    throw std::runtime_error("모르는 정렬 방법: " + name);
}
