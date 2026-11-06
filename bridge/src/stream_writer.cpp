#include "stream_writer.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace stream {

template <class T>
static void put(uint8_t* out, size_t off, T v) {
    std::memcpy(out + off, &v, sizeof(T));
}

void encode_header(const OutFrame& f, uint8_t* o) {
    static_assert(sizeof(double) == 8 && sizeof(float) == 4, "형식은 IEEE 754");
    std::memset(o, 0, kHeaderBytes);
    std::memcpy(o, "MMF1", 4);
    put<uint32_t>(o, 4, kHeaderBytes);
    put<uint32_t>(o, 8, f.flags);
    put<uint32_t>(o, 12, f.map_id);
    put<uint64_t>(o, 16, f.seq);
    put<double>(o, 24, f.t);
    for (int i = 0; i < 12; ++i) put<double>(o, 32 + 8 * i, f.T[i]);
    put<double>(o, 128, f.fx);
    put<double>(o, 136, f.fy);
    put<double>(o, 144, f.cx);
    put<double>(o, 152, f.cy);
    put<uint32_t>(o, 160, (uint32_t)f.bgr.cols);
    put<uint32_t>(o, 164, (uint32_t)f.bgr.rows);
    put<float>(o, 168, f.depth_per_m);
    put<uint32_t>(o, 172, f.anchor);
}

std::vector<uint8_t> encode_graph(const std::vector<GraphNode>& nodes) {
    std::vector<uint8_t> b(kGraphHeaderBytes + nodes.size() * (8 + 96), 0);
    std::memcpy(b.data(), "MMG1", 4);
    put<uint32_t>(b.data(), 4, kGraphHeaderBytes);
    put<uint32_t>(b.data(), 8, (uint32_t)nodes.size());
    size_t o = kGraphHeaderBytes;
    for (const GraphNode& n : nodes) {
        put<uint32_t>(b.data(), o, n.id);
        for (int i = 0; i < 12; ++i) put<double>(b.data(), o + 8 + 8 * i, n.T[i]);
        o += 8 + 96;
    }
    return b;
}

Writer::Writer(const std::string& path, int fd, size_t queue, bool block) : cap_(queue < 1 ? 1 : queue), block_(block) {
    std::signal(SIGPIPE, SIG_IGN);  // 받는 쪽이 끝나면 쓰기가 EPIPE 로 돌아오게
    if (fd >= 0) {
        fd_ = fd;
        own_fd_ = true;
    } else {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        own_fd_ = true;
        if (fd_ < 0) throw std::runtime_error("스트림 열기 실패: " + path + ": " + std::strerror(errno));
    }
    th_ = std::thread(&Writer::run, this);
}

Writer::~Writer() { close(); }

bool Writer::push(OutFrame&& f) {
    std::unique_lock<std::mutex> lk(m_);
    if (broken_) return false;
    // 그래프 블록은 큐 길이와 상관없이 넣는다(노드 위치를 잃으면 memmap 이 장면을 고칠 수 없음)
    if (!f.is_graph() && q_.size() >= cap_) {
        if (block_) {
            cv_.wait(lk, [&] { return q_.size() < cap_ || broken_; });
            if (broken_) return false;
        } else {
            // 가장 오래된 장면을 버림(그래프 블록은 남김)
            auto it = std::find_if(q_.begin(), q_.end(), [](const OutFrame& x) { return !x.is_graph(); });
            if (it != q_.end()) {
                q_.erase(it);
                ++dropped_;
            }
        }
    }
    q_.push_back(std::move(f));
    cv_.notify_all();
    return true;
}

void Writer::close() {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (done_) return;
        done_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    if (own_fd_ && fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool Writer::write_all(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n > 0) {
        ssize_t k = ::write(fd_, b, n);
        if (k < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        b += k;
        n -= (size_t)k;
    }
    return true;
}

void Writer::run() {
    uint8_t head[kHeaderBytes];
    cv::Mat rgb;
    for (;;) {
        OutFrame f;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return !q_.empty() || done_; });
            if (q_.empty()) return;  // done_ 이고 다 썼음
            f = std::move(q_.front());
            q_.pop_front();
            cv_.notify_all();
        }
        if (f.is_graph()) {
            std::vector<uint8_t> b = encode_graph(f.graph);
            if (!write_all(b.data(), b.size())) {
                std::lock_guard<std::mutex> lk(m_);
                broken_ = true;
                cv_.notify_all();
                return;
            }
            continue;
        }
        auto c0 = std::chrono::steady_clock::now();
        cv::cvtColor(f.bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat depth = f.depth.isContinuous() ? f.depth : f.depth.clone();
        encode_header(f, head);
        bool ok = write_all(head, sizeof(head)) && write_all(rgb.data, rgb.total() * 3) && write_all(depth.data, depth.total() * 2);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        std::lock_guard<std::mutex> lk(m_);
        if (!ok) {
            broken_ = true;
            cv_.notify_all();
            return;
        }
        ++written_;
        write_ms_.push_back(ms);
    }
}

}  // namespace stream
