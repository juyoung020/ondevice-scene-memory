// 원본 MLPnPsolver(기준 빌드)를 표준 입출력으로 부르는 FFI 도우미. Rust 재위치 흐름 대조 시험이 자식 프로세스로 띄운다.
// 난수(DUtils::Random::RandomInt)는 시험이 넘긴 원본 실행 값을 그대로 돌려준다(범위 a, b 가 다르면 중단).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <thread>
#include <any>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#define private public
#define protected public
#include <Frame.h>
#include <MapPoint.h>
#include <MLPnPsolver.h>
#include <KeyFrame.h>
#include <Optimizer.h>
#include <Pinhole.h>
#include <DBoW2/DUtils/Random.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static std::deque<int32_t> g_feed;
int DUtils::Random::RandomInt(int a, int b) {
    if (g_feed.size() < 3) { fprintf(stderr, "mlp_srv: 난수 모자람\n"); abort(); }
    int ea = g_feed[0], eb = g_feed[1], r = g_feed[2];
    g_feed.erase(g_feed.begin(), g_feed.begin() + 3);
    if (ea != a || eb != b) { fprintf(stderr, "mlp_srv: 난수 범위 다름 (%d,%d) != (%d,%d)\n", a, b, ea, eb); abort(); }
    return r;
}

template <class T> static T R() { T v; if (fread(&v, sizeof(T), 1, stdin) != 1) exit(0); return v; }
template <class T> static void W(const T &v) { fwrite(&v, sizeof(T), 1, stdout); }

int main() {
    std::vector<MLPnPsolver *> sol;
    std::vector<MapPoint *> pool;
    std::vector<GeometricCamera *> cams;
    for (;;) {
        uint8_t c = R<uint8_t>();
        if (c == 1) {
            std::vector<float> p(4); for (auto &x : p) x = R<float>();
            GeometricCamera *cam = new Pinhole(p); cams.push_back(cam);
            Frame F; F.mpCamera = cam;
            uint32_t nl = R<uint32_t>(); F.mvLevelSigma2.resize(nl); for (auto &x : F.mvLevelSigma2) x = R<float>();
            uint32_t n = R<uint32_t>();
            F.mvKeysUn.resize(n); F.mvpMapPoints.assign(n, nullptr); F.N = n;
            std::vector<MapPoint *> m(n, nullptr);
            for (uint32_t i = 0; i < n; i++) {
                F.mvKeysUn[i].pt.x = R<float>(); F.mvKeysUn[i].pt.y = R<float>(); F.mvKeysUn[i].octave = R<int32_t>();
                if (R<uint8_t>()) {
                    MapPoint *mp = new MapPoint(); mp->mbBad = false; mp->mpReplaced = nullptr;
                    float x = R<float>(), y = R<float>(), z = R<float>();
                    mp->mWorldPos = Eigen::Vector3f(x, y, z);
                    m[i] = mp; pool.push_back(mp);
                }
            }
            MLPnPsolver *s = new MLPnPsolver(F, m);
            s->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);
            F.mpCamera = nullptr;
            sol.push_back(s);
            W<uint32_t>(sol.size() - 1);
        } else if (c == 2) {
            uint32_t h = R<uint32_t>(); int32_t it = R<int32_t>(); uint32_t nr = R<uint32_t>();
            g_feed.clear(); for (uint32_t i = 0; i < nr; i++) g_feed.push_back(R<int32_t>());
            bool nomore = false; std::vector<bool> inl; int ninl = 0; Eigen::Matrix4f T = Eigen::Matrix4f::Zero();
            bool r = sol.at(h)->iterate(it, nomore, inl, ninl, T);
            W<uint8_t>(r); W<uint8_t>(nomore); W<int32_t>(ninl);
            W<uint32_t>(inl.size()); for (bool b : inl) W<uint8_t>(b);
            fwrite(T.data(), 4, 16, stdout);
            W<uint32_t>(nr - g_feed.size());
            // Relocalization 의 Sophus::SE3f Tcw(eigTcw) 그대로(행렬 → 사원수 비트를 원본이 정함)
            Sophus::SE3f S; if (r) S = Sophus::SE3f(T);
            fwrite(S.unit_quaternion().coeffs().data(), 4, 4, stdout); fwrite(S.translation().data(), 4, 3, stdout);
        } else if (c == 4) {
            // Optimizer::OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, H, bAllPoints) 를 가짜 키프레임 · 지도 점으로
            KeyFrame *K[2];
            std::vector<MapPoint *> own;
            for (int k = 0; k < 2; k++) {
                KeyFrame *kf = new KeyFrame();
                Sophus::SE3f T; float *d = T.data(); for (int i = 0; i < 7; i++) d[i] = R<float>();
                kf->SetPose(T);
                std::vector<float> p(4); for (auto &x : p) x = R<float>();
                kf->mpCamera = new Pinhole(p);
                uint32_t nl = R<uint32_t>(); auto &is = const_cast<std::vector<float> &>(kf->mvInvLevelSigma2); is.resize(nl); for (auto &x : is) x = R<float>();
                uint32_t nk = R<uint32_t>(); auto &ku = const_cast<std::vector<cv::KeyPoint> &>(kf->mvKeysUn); ku.resize(nk);
                for (auto &q : ku) { q.pt.x = R<float>(); q.pt.y = R<float>(); q.octave = R<int32_t>(); }
                kf->mvpMapPoints.assign(nk, nullptr);
                K[k] = kf;
            }
            auto mkmp = [&](uint8_t st) -> MapPoint * {
                MapPoint *mp = new MapPoint(); mp->mbBad = st == 2; mp->mpReplaced = nullptr;
                float x = R<float>(), y = R<float>(), z = R<float>(); mp->mWorldPos = Eigen::Vector3f(x, y, z);
                own.push_back(mp); return mp;
            };
            uint32_t n = R<uint32_t>();
            std::vector<MapPoint *> m(n, nullptr);
            for (uint32_t i = 0; i < n; i++) {
                uint8_t s1 = R<uint8_t>(); if (s1) K[0]->mvpMapPoints[i] = mkmp(s1);
                uint8_t s2 = R<uint8_t>();
                if (s2) { MapPoint *mp = mkmp(s2); int32_t i2 = R<int32_t>(); mp->mnTrackScaleLevel = R<int32_t>(); if (i2 >= 0) mp->mObservations[K[1]] = std::make_tuple(i2, -1); m[i] = mp; }
            }
            double q[4], t[3]; for (auto &x : q) x = R<double>(); for (auto &x : t) x = R<double>(); double sc = R<double>();
            g2o::Sim3 S(Eigen::Quaterniond(q[3], q[0], q[1], q[2]), Eigen::Vector3d(t[0], t[1], t[2]), sc);
            float th2 = R<float>(); bool fix = R<uint8_t>(), all = R<uint8_t>();
            std::vector<MapPoint *> m0 = m;
            Eigen::Matrix<double, 7, 7> H;
            int r = Optimizer::OptimizeSim3(K[0], K[1], m, S, th2, fix, H, all);
            W<int32_t>(r);
            for (uint32_t i = 0; i < n; i++) W<uint8_t>(m[i] != nullptr);
            fwrite(S.rotation().coeffs().data(), 8, 4, stdout); fwrite(S.translation().data(), 8, 3, stdout); W(S.scale());
        } else if (c == 3) {
            sol.clear();  // 원본 Relocalization 도 지우지 않는다(새는 것 그대로)
        } else {
            fprintf(stderr, "mlp_srv: 모르는 명령 %d\n", c); return 1;
        }
        fflush(stdout);
    }
}
