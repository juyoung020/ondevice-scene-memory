// 장면 스트림 쓰기(형식: docs/bridge.md "스트림 형식"). 쓰기는 따로 스레드가 맡아 위치 추적이 memmap 을 기다리지 않는다.
#pragma once
#include <opencv2/core.hpp>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace stream {

constexpr uint32_t kHeaderBytes = 176;
constexpr uint32_t kLost = 1;         // 위치 놓침: 위치 칸은 의미 없음
constexpr uint32_t kNewMap = 2;       // 이 장면에서 RTAB-Map 이 새 지도를 시작함
constexpr uint32_t kMapCorrected = 4; // 이 장면에서 지도 보정(루프 클로저·최적화)이 바뀌어 위치가 뛸 수 있음
constexpr uint32_t kGraphHeaderBytes = 16;  // 그래프 블록 MMG1 머리: 표시 4 + 머리 크기 4 + 노드 수 4 + 예약 4

// 지도 노드 위치(OpenCV 카메라 → 세계, 장면 위치와 같은 좌표)
struct GraphNode {
    uint32_t id = 0;
    double T[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
};

struct OutFrame {
    uint32_t flags = 0;
    uint32_t map_id = 0;
    uint64_t seq = 0;
    double t = 0;
    double T[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // 3×4 행 우선: OpenCV 카메라 → 세계
    double fx = 0, fy = 0, cx = 0, cy = 0;
    float depth_per_m = 1000.f;
    cv::Mat bgr;    // CV_8UC3 BGR (쓸 때 RGB 로 바꿈)
    cv::Mat depth;  // CV_16UC1
    uint32_t anchor = 0;  // 기준 지도 노드 id(0 = 없음): 장면 위치 = 노드 위치 × 상대 위치
    // 비어 있지 않으면 장면이 아니라 그래프 블록(MMG1): 새로 생기거나 움직인 노드. 큐가 차도 버리지 않음
    std::vector<GraphNode> graph;
    bool is_graph() const { return !graph.empty(); }
};

// 그래프 블록(MMG1) 바이트: 머리 16 + 노드마다 id u32 + 예약 u32 + 3×4 f64
std::vector<uint8_t> encode_graph(const std::vector<GraphNode>& nodes);

// 머리 176 바이트를 버퍼에 채움(리틀 엔디언 기계 기준)
void encode_header(const OutFrame& f, uint8_t* out);

class Writer {
public:
    // path: 파일·이름 있는 파이프(FIFO). fd >= 0 이면 path 대신 그 파일 기술자(표준출력을 떼어 둔 것)에 씀
    // block = true: 큐가 차면 기다림(재생 검증용, 장면을 빠짐없이). false: 가장 오래된 장면을 버림(카메라용)
    Writer(const std::string& path, int fd, size_t queue, bool block);
    ~Writer();
    // 받는 쪽이 닫혔으면 false
    bool push(OutFrame&& f);
    void close();
    size_t written() const { return written_; }
    size_t dropped() const { return dropped_; }
    // 장면마다 쓰기 시간(ms): RGB 바꾸기 + 파이프 쓰기(받는 쪽이 늦으면 기다린 시간 포함)
    const std::vector<double>& write_ms() const { return write_ms_; }

private:
    void run();
    bool write_all(const void* p, size_t n);
    int fd_ = -1;
    bool own_fd_ = false;
    size_t cap_;
    bool block_;
    std::deque<OutFrame> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false, broken_ = false;
    size_t written_ = 0, dropped_ = 0;
    std::vector<double> write_ms_;
    std::thread th_;
};

}  // namespace stream
