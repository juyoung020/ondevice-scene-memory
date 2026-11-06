// rtab_bridge: RGB-D 입력 → RTAB-Map(위치 추적 매 장 + 지도 갱신 DetectionRate) → 장면 스트림(memmap --stream).
// 처리 순서는 RTAB-Map 의 rtabmap-rgbd_dataset 도구와 같다(오프라인 위치 파일을 만든 도구): 그래서 같은 입력이면 같은 위치가 나온다.
//
//   rtab_bridge --tum <폴더> [--rate 1] | memmap --stream - ...
//   rtab_bridge --realsense [--align rs2] | memmap --stream - ...
//   옵션: --config docs/rtabmap_pi.ini  --out <파일|->  --send-every N  --queue N  --block|--drop
//         --poses <파일>(보낸 위치, TUM)  --final-poses <파일>(끝에 최적화된 지도 노드 위치, TUM)  --stats <파일>(장면별 시간)
//         --db <파일>(RTAB-Map 데이터베이스로 남김)  --max-frames N  --Param/Name 값(RTAB-Map 파라미터 덮어쓰기)
#include "frame_source.h"
#include "stream_writer.h"
#include "tum_source.h"
#ifdef BRIDGE_HAVE_REALSENSE
#include "realsense_source.h"
#endif

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryEvent.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/utilite/ULogger.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <string>
#include <vector>

using namespace rtabmap;
using clk = std::chrono::steady_clock;

static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

struct Summary {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    std::string str() const {
        if (v.empty()) return "-";
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        double mean = 0;
        for (double x : s) mean += x;
        mean /= s.size();
        char b[160];
        std::snprintf(b, sizeof b, "평균 %.2f 중앙 %.2f 95%% %.2f 최대 %.2f (%zu)", mean, s[s.size() / 2], s[(size_t)((s.size() - 1) * 0.95)], s.back(), s.size());
        return b;
    }
};

// 위치를 TUM 한 줄로: 시각 x y z qx qy qz qw
static void write_tum(std::ostream& o, double t, const Transform& T) {
    Eigen::Quaternionf q = T.getQuaternionf();
    char b[200];
    std::snprintf(b, sizeof b, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", t, T.x(), T.y(), T.z(), q.x(), q.y(), q.z(), q.w());
    o << b;
}

static void usage() {
    std::fprintf(stderr,
                 "rtab_bridge (--tum <폴더> [--rate R] | --realsense [--align rs2] [--serial S] [--size 640x480] [--fps 30])\n"
                 "            [--config <ini>] [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop]\n"
                 "            [--poses <파일>] [--final-poses <파일>] [--stats <파일>] [--db <파일>] [--max-frames N] [--Group/Name 값 ...]\n");
}

static int run(int argc, char** argv) {
    std::string tum, config, out = "-", poses_path, final_path, stats_path, db, align = "rs2", serial;
    bool realsense = false;
    int block = -1;  // -1: 입력에 따라(재생 = 기다림, 카메라 = 버림)
    double rate = 0;
    int send_every = 1, queue = 4, w = 640, h = 480, fps = 30;
    long max_frames = -1;
    ParametersMap overrides;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--tum") tum = val();
        else if (a == "--realsense") realsense = true;
        else if (a == "--rate") rate = std::stod(val());
        else if (a == "--align") align = val();
        else if (a == "--serial") serial = val();
        else if (a == "--size") {
            std::string s = val();
            std::sscanf(s.c_str(), "%dx%d", &w, &h);
        } else if (a == "--fps") fps = std::stoi(val());
        else if (a == "--config") config = val();
        else if (a == "--out") out = val();
        else if (a == "--send-every") send_every = std::max(1, std::stoi(val()));
        else if (a == "--queue") queue = std::max(1, std::stoi(val()));
        else if (a == "--block") block = 1;
        else if (a == "--drop") block = 0;
        else if (a == "--poses") poses_path = val();
        else if (a == "--final-poses") final_path = val();
        else if (a == "--stats") stats_path = val();
        else if (a == "--db") db = val();
        else if (a == "--max-frames") max_frames = std::stol(val());
        else if (a.rfind("--", 0) == 0 && a.find('/') != std::string::npos) {
            std::string name = a.substr(2);
            if (Parameters::getDefaultParameters().count(name) == 0) {
                std::fprintf(stderr, "모르는 RTAB-Map 파라미터: %s\n", name.c_str());
                return 2;
            }
            overrides[name] = val();
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "모르는 인자: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (tum.empty() == !realsense) {
        usage();
        return 2;
    }
    // 스트림이 표준출력이면 그 파일 기술자를 따로 떼어 두고 표준출력은 표준오류로 돌린다:
    // RTAB-Map 로그(ULogger 콘솔)·라이브러리 printf 가 장면 바이트 사이에 섞이지 않게
    int stream_fd = -1;
    if (out == "-") {
        stream_fd = ::dup(STDOUT_FILENO);
        std::fflush(stdout);
        ::dup2(STDERR_FILENO, STDOUT_FILENO);
    }
    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);

    std::unique_ptr<FrameSource> src;
    if (!tum.empty()) {
        src = std::make_unique<TumSource>(tum, rate);
    } else {
#ifdef BRIDGE_HAVE_REALSENSE
        src = std::make_unique<RealSenseSource>(w, h, fps, align, serial);
#else
        std::fprintf(stderr, "이 빌드에는 RealSense 입력이 없다(librealsense2 를 찾지 못함)\n");
        return 2;
#endif
    }
    if (block < 0) block = (!tum.empty() && rate == 0) ? 1 : 0;

    // RTAB-Map 파라미터: 기본값 ← ini([Core]) ← 명령줄
    ParametersMap params;
    if (!config.empty()) Parameters::readINI(config, params);
    for (auto& kv : overrides) params[kv.first] = kv.second;
    bool intermediate = Parameters::defaultRtabmapCreateIntermediateNodes();
    float detection_rate = Parameters::defaultRtabmapDetectionRate();
    Parameters::parse(params, Parameters::kRtabmapCreateIntermediateNodes(), intermediate);
    Parameters::parse(params, Parameters::kRtabmapDetectionRate(), detection_rate);
    params.erase(Parameters::kRtabmapPublishRAMUsage());

    std::fprintf(stderr, "입력: %s\n", src->describe().c_str());
    std::fprintf(stderr, "RTAB-Map %s, 설정 %s, 바꾼 파라미터 %zu 개:", RTABMAP_VERSION, config.empty() ? "(기본값)" : config.c_str(), params.size());
    for (auto& kv : params) std::fprintf(stderr, " %s=%s", kv.first.c_str(), kv.second.c_str());
    std::fprintf(stderr, "\n출력: %s (%s, 큐 %d, %d 장마다)\n", out.c_str(), block ? "받는 쪽을 기다림" : "밀리면 오래된 장면 버림", queue, send_every);

    std::unique_ptr<Odometry> odom(Odometry::create(params));
    Rtabmap rtabmap;
    rtabmap.init(params, db);

    stream::Writer writer(out, stream_fd, (size_t)queue, block == 1);
    std::ofstream poses_f, stats_f;
    if (!poses_path.empty()) {
        poses_f.open(poses_path);
        poses_f << "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 보낸 그대로)\n";
    }
    if (!stats_path.empty()) {
        stats_f.open(stats_path);
        stats_f << "seq,t,read_ms,odom_ms,slam_ms,send_ms,lost,new_map,map_corrected,loop,features,inliers\n";
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // 카메라 광학 축(x 오른쪽, y 아래, z 앞) → RTAB-Map 몸체 축(x 앞, y 왼쪽, z 위)
    const Transform optical = CameraModel::opticalRotation();
    Summary s_read, s_odom, s_slam, s_send, s_total;
    long n = 0, n_lost = 0, n_newmap = 0, n_loop = 0, n_corr = 0, n_graph = 0, n_graph_nodes = 0;
    std::map<int, Transform> sent_nodes;  // memmap 에 보낸 노드 위치(지도 몸체 축)
    uint32_t map_id = 0;
    double prev_stamp = 0;
    Transform prev_corr = Transform::getIdentity();
    auto t_run = clk::now();
    RgbdFrame fr;
    while (!g_stop && (max_frames < 0 || n < max_frames)) {
        auto c0 = clk::now();
        if (!src->next(fr)) break;
        double read_ms = src->last_read_ms();
        // RTAB-Map 용 깊이: 밀리미터 u16(TUM 5000/m 은 rtabmap-rgbd_dataset 처럼 5 로 나눔)
        cv::Mat depth_mm = fr.depth;
        if (std::abs(fr.depth_per_m - 1000.f) > 1e-3f) {
            double div = fr.depth_per_m / 1000.0;
            depth_mm = fr.depth.clone();
            if (div > 1.0) depth_mm /= div;
            else fr.depth.convertTo(depth_mm, CV_16UC1, 1.0 / div);
        }
        CameraModel model(fr.fx, fr.fy, fr.cx, fr.cy, optical, 0, fr.bgr.size());
        SensorData data(fr.bgr, depth_mm, model, (int)n + 1, fr.t);

        auto c1 = clk::now();
        OdometryInfo info;
        Transform pose = odom->process(data, &info);
        double odom_ms = ms_since(c1);

        bool new_map = false;
        if (n != 0 && !info.reg.covariance.empty() && info.reg.covariance.at<double>(0, 0) >= 9999) {
            std::fprintf(stderr, "장면 %ld: 위치 추적 다시 시작 → 새 지도\n", n);
            int id = rtabmap.triggerNewMap();
            map_id = id >= 0 ? (uint32_t)id : map_id + 1;
            new_map = true;
            ++n_newmap;
        }
        bool process = true;
        if (detection_rate > 0 && prev_stamp > 0 && data.stamp() > prev_stamp && data.stamp() - prev_stamp < 1.0 / detection_rate) process = false;
        if (process) prev_stamp = data.stamp();
        if (!process) {
            data.setId(-1);  // 중간 장면(지도 노드 아님)
            data.setFeatures(std::vector<cv::KeyPoint>(), std::vector<cv::Point3f>(), cv::Mat());
            process = intermediate;
        }
        auto c2 = clk::now();
        double slam_ms = -1;
        bool loop = false;
        if (process) {
            OdometryEvent e(SensorData(), Transform(), info);
            rtabmap.process(data, pose, info.reg.covariance, e.velocity());
            slam_ms = ms_since(c2);
            loop = rtabmap.getLoopClosureId() > 0;
            n_loop += loop;
        }
        // 이번 장면의 지도 위치 = 지도 보정 × 오도메트리 위치 × 광학 축
        Transform corr = rtabmap.getMapCorrection();
        bool corrected = false;
        if (!corr.isNull())
            for (int i = 0; i < 12; ++i) corrected |= std::abs(corr.data()[i] - prev_corr.data()[i]) > 1e-6f;
        if (corrected) {
            prev_corr = corr;
            ++n_corr;
        }
        // 놓친 장면: 위치를 못 구함. Odom/ResetCountdown 으로 다시 시작한 장면도 위치가 '마지막 위치'로 채워질 뿐이라 놓침으로 표시
        bool lost = pose.isNull() || info.lost || new_map;
        n_lost += lost;

        // 그래프 블록: 지도 갱신 뒤 새로 생기거나 움직인(행렬 원소 1e-3 = 1 mm·약 0.06° 넘게) 노드를 이 장면보다 먼저 보낸다.
        // memmap 은 장면을 기준 노드와 함께 보관했다가 노드가 움직이면 고친 위치로 지도·물체를 다시 쌓는다
        if (process) {
            std::vector<stream::GraphNode> upd;
            for (const auto& [id, P] : rtabmap.getLocalOptimizedPoses()) {
                if (id <= 0 || P.isNull()) continue;
                auto it = sent_nodes.find(id);
                if (it != sent_nodes.end()) {
                    float dmax = 0;
                    for (int i = 0; i < 12; ++i) dmax = std::max(dmax, std::abs(it->second.data()[i] - P.data()[i]));
                    if (dmax < 1e-3f) continue;
                }
                sent_nodes[id] = P;
                stream::GraphNode gn;
                gn.id = (uint32_t)id;
                const Transform Tn = P * optical;
                for (int i = 0; i < 12; ++i) gn.T[i] = Tn.data()[i];
                upd.push_back(gn);
            }
            if (!upd.empty()) {
                n_graph_nodes += (long)upd.size();
                ++n_graph;
                stream::OutFrame g;
                g.graph = std::move(upd);
                if (!writer.push(std::move(g))) {
                    std::fprintf(stderr, "받는 쪽이 닫힘: 끝냄\n");
                    break;
                }
            }
        }

        stream::OutFrame of;
        of.flags = (lost ? stream::kLost : 0) | (new_map ? stream::kNewMap : 0) | (corrected ? stream::kMapCorrected : 0);
        of.map_id = map_id;
        of.seq = (uint64_t)n;
        of.t = fr.t;
        Transform Twc;
        if (!lost) {
            Twc = (corr.isNull() ? Transform::getIdentity() : corr) * pose * optical;
            const float* d = Twc.data();  // 3×4 행 우선
            for (int i = 0; i < 12; ++i) of.T[i] = d[i];
            if (poses_f) write_tum(poses_f, fr.t, Twc);
            // 기준 노드: 마지막 지도 노드(이 장면 위치 = 그 노드 위치 × 오도메트리 상대 위치)
            int last_id = rtabmap.getLastLocationId();
            of.anchor = sent_nodes.count(last_id) ? (uint32_t)last_id : 0;
        }
        of.fx = fr.fx;
        of.fy = fr.fy;
        of.cx = fr.cx;
        of.cy = fr.cy;
        of.depth_per_m = fr.depth_per_m;
        double send_ms = 0;
        if (n % send_every == 0) {
            auto c3 = clk::now();
            of.bgr = fr.bgr;
            of.depth = fr.depth;
            if (!writer.push(std::move(of))) {
                std::fprintf(stderr, "받는 쪽이 닫힘: 끝냄\n");
                break;
            }
            send_ms = ms_since(c3);  // 큐에 넣기(기다림 포함). 실제 파이프 쓰기는 쓰기 스레드 시간으로 따로 셈
        }
        double total_ms = ms_since(c0);
        s_read.add(read_ms);
        if (n > 0) s_odom.add(odom_ms);  // 첫 장면(초기화)은 뺌
        if (slam_ms >= 0 && n > 0) s_slam.add(slam_ms);
        s_send.add(send_ms);
        s_total.add(total_ms);
        if (stats_f) {
            char b[256];
            std::snprintf(b, sizeof b, "%ld,%.6f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d\n", n, fr.t, read_ms, odom_ms, slam_ms, send_ms, lost, new_map, corrected, loop,
                          info.features, info.reg.inliers);
            stats_f << b;
        }
        ++n;
    }
    double wall = std::chrono::duration<double>(clk::now() - t_run).count();
    writer.close();
    Summary s_write;
    for (double x : writer.write_ms()) s_write.add(x);

    if (!final_path.empty()) {
        std::map<int, Transform> poses;
        std::multimap<int, Link> links;
        std::map<int, Signature> sigs;
        rtabmap.getGraph(poses, links, true, true, &sigs);
        std::ofstream f(final_path);
        f << "# timestamp x y z qx qy qz qw (OpenCV 카메라 → RTAB-Map 지도, 끝에 최적화된 지도 노드)\n";
        for (auto& [id, T] : poses) {
            auto it = sigs.find(id);
            if (it == sigs.end()) continue;
            write_tum(f, it->second.getStamp(), T * optical);
        }
        std::fprintf(stderr, "최적화된 지도 노드 %zu 개 → %s\n", poses.size(), final_path.c_str());
    }
    rtabmap.close(!db.empty());

    std::fprintf(stderr, "장면 %ld 개, %.1f 초(%.1f 장/초), 놓침 %ld, 새 지도 %ld, 루프 클로저 %ld, 지도 보정 바뀜 %ld\n", n, wall, n / std::max(wall, 1e-9), n_lost, n_newmap,
                 n_loop, n_corr);
    if (auto* ts = dynamic_cast<TumSource*>(src.get())) std::fprintf(stderr, "재생에서 밀려 건너뛴 장면 %zu\n", ts->skipped());
    std::fprintf(stderr, "보냄 %zu, 밀려 버림 %zu, 그래프 블록 %ld 개(노드 %ld 개)\n", writer.written(), writer.dropped(), n_graph, n_graph_nodes);
    std::fprintf(stderr, "ms/장  읽기    %s\n", s_read.str().c_str());
    std::fprintf(stderr, "ms/장  위치추적 %s\n", s_odom.str().c_str());
    std::fprintf(stderr, "ms/회  지도갱신 %s\n", s_slam.str().c_str());
    std::fprintf(stderr, "ms/장  큐넣기  %s\n", s_send.str().c_str());
    std::fprintf(stderr, "ms/장  파이프쓰기(쓰기 스레드) %s\n", s_write.str().c_str());
    std::fprintf(stderr, "ms/장  합(본 스레드) %s\n", s_total.str().c_str());
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rtab_bridge 오류: %s\n", e.what());
        return 1;
    }
}
