// 원본 LoopClosing 의 DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF 대조 덤프(기준 빌드).
// 호출마다: 들어갈 때 상태(현재 키프레임 · 후보 · 둘레 키프레임(공시성 3 단계) · 지도 점 · 출력 칸의 이전 값),
// 안에서 쓴 난수 · OptimizeSim3 입출력, 나올 때 출력.
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <any>
#include <mutex>
#include <thread>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#define private public
#define protected public
#include <System.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Map.h>
#include <LoopClosing.h>
#include <LocalMapping.h>
#include <Optimizer.h>
#include <DBoW2/DUtils/Random.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static FILE *g = nullptr, *g_file = nullptr;
static char *g_buf = nullptr; static size_t g_bufn = 0;
static bool g_osim = false; static long g_boring = 0, g_every = 1;
static std::mutex g_mx;
static thread_local bool g_in = false;
static thread_local bool g_nested = false;  // BoW 덤프 안에서 부르는 LastKF 시험 호출
static thread_local std::vector<int32_t> g_rand;
static long g_n = 0, g_max = 1000000, g_skip = 0, g_seen = 0;
template <class T> static void W(const T &v) { fwrite(&v, sizeof(T), 1, g); }
static void Wb(const void *p, size_t n) { fwrite(p, 1, n, g); }
static void *L() { static void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD); return h; }

int DUtils::Random::RandomInt(int a, int b) {
    typedef int (*Fn)(int, int);
    static Fn orig = (Fn)dlsym(RTLD_NEXT, "_ZN6DUtils6Random9RandomIntEii");
    int r = orig(a, b);
    if (g_in) { g_rand.push_back(a); g_rand.push_back(b); g_rand.push_back(r); }
    return r;
}

static void wkeys(const std::vector<cv::KeyPoint> &k) {
    W<uint32_t>(k.size());
    for (auto &p : k) { W(p.pt.x); W(p.pt.y); W(p.size); W(p.angle); W(p.response); W<int32_t>(p.octave); W<int32_t>(p.class_id); }
}
static void wdesc(const cv::Mat &d) { for (int i = 0; i < d.rows; i++) Wb(d.ptr(i), 32); }
static void wfv(const std::vector<float> &v) { W<uint32_t>(v.size()); Wb(v.data(), 4 * v.size()); }
static void wfeat(const DBoW2::FeatureVector &f) {
    W<uint32_t>(f.size());
    for (auto &e : f) { W<uint32_t>(e.first); W<uint32_t>(e.second.size()); for (auto i : e.second) W<uint32_t>(i); }
}
static void wse3(const Sophus::SE3f &T) { Wb(T.unit_quaternion().coeffs().data(), 16); Wb(T.translation().data(), 12); }
static void wsim(const g2o::Sim3 &S) { Wb(S.rotation().coeffs().data(), 32); Wb(S.translation().data(), 24); W(S.scale()); }
static int64_t mpid(MapPoint *p) { return p ? (int64_t)p->mnId : -1; }
static int64_t kfid(KeyFrame *p) { return p ? (int64_t)p->mnId : -1; }
static void wmps(const std::vector<MapPoint *> &v) { W<uint32_t>(v.size()); for (auto *p : v) W<int64_t>(mpid(p)); }

// 둘레 키프레임: 시작점들에서 GetBestCovisibilityKeyFrames(10) 를 depth 번 따라감
static void grow(std::map<long unsigned, KeyFrame *> &S, KeyFrame *k, int depth) {
    if (!k) return;
    S[k->mnId] = k;
    if (depth == 0) return;
    for (KeyFrame *c : k->GetBestCovisibilityKeyFrames(10)) grow(S, c, depth - 1);
}

static void dumpWorld(std::map<long unsigned, KeyFrame *> &S) {
    KeyFrame *k0 = S.begin()->second;
    W(k0->fx); W(k0->fy); W(k0->cx); W(k0->cy); W(k0->invfx); W(k0->invfy); W(k0->mbf); W(k0->mb); W(k0->mThDepth);
    W((float)k0->mnMinX); W((float)k0->mnMaxX); W((float)k0->mnMinY); W((float)k0->mnMaxY); W(k0->mfGridElementWidthInv); W(k0->mfGridElementHeightInv);
    W<int32_t>(k0->mnScaleLevels); W(k0->mfScaleFactor); W(k0->mfLogScaleFactor);
    wfv(k0->mvScaleFactors); { std::vector<float> iv; for (float s : k0->mvScaleFactors) iv.push_back(1.0f / s); wfv(iv); } wfv(k0->mvLevelSigma2); wfv(k0->mvInvLevelSigma2);
    W<uint32_t>(S.size());
    std::map<long unsigned, MapPoint *> mps;
    for (auto &e : S) {
        KeyFrame *k = e.second;
        W<uint64_t>(k->mnId); W<uint64_t>(k->mnFrameId); W<uint8_t>(k->isBad());
        wse3(k->GetPose()); wse3(k->GetPoseInverse());
        wkeys(k->mvKeysUn); wdesc(k->mDescriptors); wfv(k->mvuRight); wfv(k->mvDepth);
        auto m = k->GetMapPointMatches(); W<uint32_t>(m.size());
        for (auto *p : m) { W<int64_t>(mpid(p)); if (p) mps[p->mnId] = p; }
        wfeat(k->mFeatVec);
        for (int c = 0; c < k->mnGridCols; c++) for (int r = 0; r < k->mnGridRows; r++) {
            auto &cell = k->mGrid[c][r]; W<uint32_t>(cell.size()); for (auto i : cell) W<uint32_t>(i);
        }
        {
            std::unique_lock<std::mutex> lk(k->mMutexConnections);
            W<uint32_t>(k->mConnectedKeyFrameWeights.size());
            std::map<long unsigned, int> cw;
            for (auto &c : k->mConnectedKeyFrameWeights) cw[c.first->mnId] = c.second;
            for (auto &c : cw) { W<uint64_t>(c.first); W<int32_t>(c.second); }
            W<uint32_t>(k->mvpOrderedConnectedKeyFrames.size());
            for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) { W<uint64_t>(k->mvpOrderedConnectedKeyFrames[i]->mnId); W<int32_t>(k->mvOrderedWeights[i]); }
        }
    }
    W<uint32_t>(mps.size());
    for (auto &e : mps) {
        MapPoint *p = e.second;
        W<uint64_t>(p->mnId); W<uint8_t>(p->isBad());
        Eigen::Vector3f x = p->GetWorldPos(), n = p->GetNormal(); Wb(x.data(), 12); Wb(n.data(), 12);
        cv::Mat d = p->GetDescriptor(); if (d.empty()) { uint8_t z[32] = {0}; Wb(z, 32); } else Wb(d.ptr(0), 32);
        W(p->mfMinDistance); W(p->mfMaxDistance); W<int32_t>(p->mnTrackScaleLevel);
        auto obs = p->GetObservations();
        std::map<long unsigned, std::tuple<int, int>> o;
        for (auto &q : obs) if (S.count(q.first->mnId) && S[q.first->mnId] == q.first) o[q.first->mnId] = q.second;
        W<uint32_t>(o.size());
        for (auto &q : o) { W<uint64_t>(q.first); W<int32_t>(std::get<0>(q.second)); W<int32_t>(std::get<1>(q.second)); }
    }
}

typedef int (*OsFn)(KeyFrame *, KeyFrame *, std::vector<MapPoint *> &, g2o::Sim3 &, const float, const bool, Eigen::Matrix<double, 7, 7> &, const bool);
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double, 7, 7> &mAcumHessian, const bool bAllPoints) {
    static OsFn orig = (OsFn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer12OptimizeSim3EPNS_8KeyFrameES2_RSt6vectorIPNS_8MapPointESaIS5_EERN3g2o4Sim3EfbRN5Eigen6MatrixIdLi7ELi7ELi0ELi7ELi7EEEb");
    if (!g_in) return orig(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, mAcumHessian, bAllPoints);
    g_osim = true;
    W<uint32_t>(0x4d495350u);  // OSIM
    W<uint64_t>(pKF1->mnId); W<uint64_t>(pKF2->mnId); wmps(vpMatches1); wsim(g2oS12); W(th2); W<uint8_t>(bFixScale); W<uint8_t>(bAllPoints);
    int r = orig(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, mAcumHessian, bAllPoints);
    W<int32_t>(r); wmps(vpMatches1); wsim(g2oS12);
    return r;
}

// 지도 스레드를 멈춰 덤프한 상태 그대로 계산하게 함(원본 루프 스레드는 지도 스레드와 같이 돌아 덤프 뒤 상태가 바뀔 수 있음)
static bool stopMapper(LocalMapping *m) {
    m->RequestStop();
    for (int i = 0; i < 3000; i++) { if (m->isStopped()) return true; usleep(1000); }
    m->Release();
    return false;
}

static void beginRec() { g = open_memstream(&g_buf, &g_bufn); g_osim = false; }
// OptimizeSim3 까지 간 호출 · LastKF 호출은 모두, 나머지는 g_every 개마다 하나만 남김
static void endRec(bool force) {
    fclose(g); g = g_file;
    bool keep = force || g_osim || (g_boring++ % g_every == 0);
    if (keep) { fwrite(g_buf, 1, g_bufn, g_file); fflush(g_file); g_n++; }
    free(g_buf); g_buf = nullptr;
}

static void endRand() { W<uint32_t>(0x444e4152u); W<uint32_t>(g_rand.size()); Wb(g_rand.data(), 4 * g_rand.size()); }

typedef bool (*BowFn)(LoopClosing *, std::vector<KeyFrame *> &, KeyFrame *&, KeyFrame *&, g2o::Sim3 &, int &, std::vector<MapPoint *> &, std::vector<MapPoint *> &);
bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame *> &vpBowCand, KeyFrame *&pMatchedKF2, KeyFrame *&pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint *> &vpMPs, std::vector<MapPoint *> &vpMatchedMPs) {
    static BowFn orig = (BowFn)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing26DetectCommonRegionsFromBoWERSt6vectorIPNS_8KeyFrameESaIS3_EERS3_S7_RN3g2o4Sim3ERiRS1_IPNS_8MapPointESaISD_EESG_");
    if (!g || g_n >= g_max || g_seen++ < g_skip) return orig(this, vpBowCand, pMatchedKF2, pLastCurrentKF, g2oScw, nNumCoincidences, vpMPs, vpMatchedMPs);
    if (!stopMapper(mpLocalMapper)) { fprintf(stderr, "지도 스레드 안 멈춤: skip\n"); return orig(this, vpBowCand, pMatchedKF2, pLastCurrentKF, g2oScw, nNumCoincidences, vpMPs, vpMatchedMPs); }
    std::unique_lock<std::mutex> lk(g_mx);
    beginRec();
    W<uint32_t>(0x42524344u);  // DCRB
    W<uint64_t>(mpCurrentKF->mnId); W<uint8_t>(mbFixScale);
    W<uint32_t>(vpBowCand.size()); for (auto *k : vpBowCand) W<int64_t>(kfid(k));
    W<int64_t>(kfid(pMatchedKF2)); W<int64_t>(kfid(pLastCurrentKF)); wsim(g2oScw); W<int32_t>(nNumCoincidences); wmps(vpMPs); wmps(vpMatchedMPs);
    std::map<long unsigned, KeyFrame *> S;
    grow(S, mpCurrentKF, 2);
    for (auto *k : vpBowCand) grow(S, k, 3);
    dumpWorld(S);
    g_rand.clear(); g_in = true;
    bool r = orig(this, vpBowCand, pMatchedKF2, pLastCurrentKF, g2oScw, nNumCoincidences, vpMPs, vpMatchedMPs);
    g_in = false;
    endRand();
    W<uint32_t>(0x45524344u);  // DCRE
    W<uint8_t>(r); W<int64_t>(kfid(pMatchedKF2)); W<int64_t>(kfid(pLastCurrentKF)); wsim(g2oScw); W<int32_t>(nNumCoincidences); wmps(vpMPs); wmps(vpMatchedMPs);
    bool found = g_osim && pMatchedKF2 && pLastCurrentKF == mpCurrentKF;
    endRec(false);
    // 원본 실행에서 LastKF 단계(앞 검출의 일치 수가 1~2 일 때 다음 키프레임들에서)는 드물어서, 찾은 직후 같은 상태로
    // DetectAndReffineSim3FromLastKF 를 현재 키프레임 · 공시 키프레임들에 대해 불러 본다(지도를 바꾸지 않는 함수, 출력은 복사본)
    if (found) {
        g_nested = true;
        // 원본 OptimizeSim3 는 KF1 = mpCurrentKF 로 vpMatchedMPs(pCurrentKF 크기)를 읽으므로 pCurrentKF 는 현재 키프레임만(실제 흐름과 같음)
        std::vector<KeyFrame *> ks{mpCurrentKF};
        for (KeyFrame *kj : ks) {
            Sophus::SE3d Tjc = (kj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
            g2o::Sim3 gS = g2o::Sim3(Tjc.unit_quaternion(), Tjc.translation(), 1.0) * g2oScw;
            int np = 0; std::vector<MapPoint *> a = vpMPs, b;
            DetectAndReffineSim3FromLastKF(kj, pMatchedKF2, gS, np, a, b);
        }
        g_nested = false;
    }
    mpLocalMapper->Release();
    return r;
}

typedef bool (*LastFn)(LoopClosing *, KeyFrame *, KeyFrame *, g2o::Sim3 &, int &, std::vector<MapPoint *> &, std::vector<MapPoint *> &);
bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame *pCurrentKF, KeyFrame *pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint *> &vpMPs, std::vector<MapPoint *> &vpMatchedMPs) {
    static LastFn orig = (LastFn)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing30DetectAndReffineSim3FromLastKFEPNS_8KeyFrameES2_RN3g2o4Sim3ERiRSt6vectorIPNS_8MapPointESaIS9_EESC_");
    if (!g || g_n >= g_max || (!g_nested && g_seen++ < g_skip)) return orig(this, pCurrentKF, pMatchedKF, gScw, nNumProjMatches, vpMPs, vpMatchedMPs);
    if (!g_nested && !stopMapper(mpLocalMapper)) { fprintf(stderr, "지도 스레드 안 멈춤: skip\n"); return orig(this, pCurrentKF, pMatchedKF, gScw, nNumProjMatches, vpMPs, vpMatchedMPs); }
    std::unique_lock<std::mutex> lk(g_mx, std::defer_lock);
    if (!g_nested) lk.lock();
    beginRec();
    W<uint32_t>(0x4c524344u);  // DCRL
    W<uint64_t>(pCurrentKF->mnId); W<uint64_t>(mpCurrentKF->mnId); W<uint64_t>(pMatchedKF->mnId); W<uint8_t>(mbFixScale);
    wsim(gScw); W<int32_t>(nNumProjMatches); wmps(vpMPs); wmps(vpMatchedMPs);
    std::map<long unsigned, KeyFrame *> S;
    grow(S, pCurrentKF, 1); grow(S, mpCurrentKF, 0);
    grow(S, pMatchedKF, 2);
    dumpWorld(S);
    g_rand.clear(); g_in = true;
    bool r = orig(this, pCurrentKF, pMatchedKF, gScw, nNumProjMatches, vpMPs, vpMatchedMPs);
    g_in = false;
    endRand();
    W<uint32_t>(0x454c5244u);  // DRLE
    W<uint8_t>(r); wsim(gScw); W<int32_t>(nNumProjMatches); wmps(vpMPs); wmps(vpMatchedMPs);
    endRec(true);
    if (!g_nested) mpLocalMapper->Release();
    return r;
}

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}

// 순서: "a-b,c-d:s" = a..b-1 을 차례로, c..d-1 을 s 칸씩
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "loop_dump voc yaml seqdir out.bin order [max_calls]\n"); return 1; }
    std::string dir = argv[3];
    g = g_file = fopen(argv[4], "wb");
    if (argc > 6) g_max = atol(argv[6]);
    if (argc > 7) g_skip = atol(argv[7]);
    if (argc > 8) g_every = atol(argv[8]);
    std::vector<double> tr, td; std::vector<std::string> pr, pd;
    load(dir + "/rgb.txt", tr, pr); load(dir + "/depth.txt", td, pd);
    std::vector<std::pair<int, int>> fr;
    size_t j = 0;
    for (size_t i = 0; i < tr.size(); i++) {
        while (j + 1 < td.size() && std::abs(td[j + 1] - tr[i]) <= std::abs(td[j] - tr[i])) j++;
        if (std::abs(td[j] - tr[i]) <= 0.02) fr.push_back({(int)i, (int)j});
    }
    std::vector<int> order;
    std::stringstream os(argv[5]); std::string seg;
    while (std::getline(os, seg, ',')) {
        int a = 0, b = 0, s = 1; char c;
        std::stringstream ss(seg); ss >> a >> c >> b; if (ss >> c) ss >> s;
        for (int k = a; k < b && k < (int)fr.size(); k += s) order.push_back(k);
    }
    System S(argv[1], argv[2], System::RGBD, false);
    double t = 0; int n = 0;
    for (int k : order) {
        cv::Mat C = cv::imread(dir + "/" + pr[fr[k].first], cv::IMREAD_UNCHANGED);
        cv::Mat D = cv::imread(dir + "/" + pd[fr[k].second], cv::IMREAD_UNCHANGED);
        if (C.empty() || D.empty()) continue;
        t += 1.0 / 30;
        S.TrackRGBD(C, D, t);
        usleep(20000);
        n++;
    }
    S.Shutdown();
    { std::unique_lock<std::mutex> lk(g_mx); fclose(g_file); g = g_file = nullptr; }
    fprintf(stderr, "frames %d calls %ld\n", n, g_n);
    return 0;
}
