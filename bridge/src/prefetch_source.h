// 장면 읽기(카메라 대기·깊이 정렬·PNG 풀기)를 따로 도는 스레드에 맡긴다: 위치 추적이 읽기를 기다리지 않게.
// drop = true(카메라·찍은 속도 재생): 위치 추적이 밀리면 아직 안 가져간 장면을 새 장면으로 바꿈(카메라처럼 지난 장면은 버림)
// drop = false(최대 속도 재생): 한 장 앞서 읽고 가져갈 때까지 기다림(장면을 하나도 버리지 않음)
#pragma once
#include "frame_source.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

class PrefetchSource : public FrameSource {
public:
    PrefetchSource(std::unique_ptr<FrameSource> inner, bool drop) : inner_(std::move(inner)), drop_(drop) {
        th_ = std::thread([this] { run(); });
    }
    ~PrefetchSource() override {
        {
            std::lock_guard<std::mutex> l(m_);
            stop_ = true;
        }
        cv_.notify_all();
        th_.join();
    }
    bool next(RgbdFrame& f) override {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return full_ || done_; });
        if (!full_) {
            if (err_) std::rethrow_exception(err_);
            return false;
        }
        f = std::move(slot_);
        read_ms_ = slot_ms_;
        full_ = false;
        cv_.notify_all();
        return true;
    }
    double last_read_ms() const override { return read_ms_; }
    std::string describe() const override { return inner_->describe() + (drop_ ? ", 읽기 스레드(밀리면 지난 장면 버림)" : ", 읽기 스레드"); }
    // 위치 추적이 밀려 가져가기 전에 새 장면으로 바뀐 장면 수
    size_t dropped() const { return dropped_; }

private:
    void run() {
        try {
            for (;;) {
                RgbdFrame f;
                if (!inner_->next(f)) break;
                std::unique_lock<std::mutex> l(m_);
                if (!drop_) cv_.wait(l, [this] { return !full_ || stop_; });
                if (stop_) return;
                if (full_) ++dropped_;
                slot_ = std::move(f);
                slot_ms_ = inner_->last_read_ms();
                full_ = true;
                cv_.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> l(m_);
            err_ = std::current_exception();
        }
        std::lock_guard<std::mutex> l(m_);
        done_ = true;
        cv_.notify_all();
    }

    std::unique_ptr<FrameSource> inner_;
    bool drop_;
    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    RgbdFrame slot_;
    double slot_ms_ = 0, read_ms_ = 0;
    bool full_ = false, done_ = false, stop_ = false;
    std::exception_ptr err_;
    std::atomic<size_t> dropped_{0};
};
