// 원본 Tracking::Relocalization 대조 덤프. 추적을 잃게 하는 입력(되감기 · 건너뛰기)으로 원본(기준 빌드)을 돌리며
// 재위치 시도마다: 후보 질의 직후의 장면 · 후보 키프레임 · 지도 점 상태, MLPnP iterate 호출(난수 · 출력), 끝 결과를 덤프.
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
#include <KeyFrameDatabase.h>
#include <Tracking.h>
#include <MLPnPsolver.h>
#include <DBoW2/DUtils/Random.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static FILE *g = nullptr;
static thread_local bool g_in = false;  // Relocalization 안
static thread_local bool g_it = false;  // iterate 안
static thread_local std::vector<int32_t> g_rand;
static thread_local std::map<MLPnPsolver *, int> g_sol;
static thread_local int g_ncand = 0;
static long g_nrel = 0, g_max = 1000000;
template <class T> static void W(const T &v) { fwrite(&v, sizeof(T), 1, g); }
static void Wb(const void *p, size_t n) { fwrite(p, 1, n, g); }
static void *L() { static void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD); return h; }

int DUtils::Random::RandomInt(int a, int b) {
    typedef int (*Fn)(int, int);
    static Fn orig = (Fn)dlsym(RTLD_NEXT, "_ZN6DUtils6Random9RandomIntEii");
    int r = orig(a, b);
    if (g_it) { g_rand.push_back(a); g_rand.push_back(b); g_rand.push_back(r); }
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
static int64_t mpid(MapPoint *p) { return p ? (int64_t)p->mnId : -1; }

typedef std::vector<KeyFrame *> (*DetFn)(KeyFrameDatabase *, Frame *, Map *);
std::vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map *pMap) {
    static DetFn orig = (DetFn)dlsym(L(), "_ZN9ORB_SLAM316KeyFrameDatabase30DetectRelocalizationCandidatesEPNS_5FrameEPNS_3MapE");
    std::vector<KeyFrame *> v = orig(this, F, pMap);
    if (!g_in || !g) return v;
    W<uint32_t>(0x434c4552u);  // RELC
    W<uint64_t>(F->mnId); W(F->mTimeStamp); W<uint64_t>(pMap ? pMap->GetId() : 0);
    W(Frame::fx); W(Frame::fy); W(Frame::cx); W(Frame::cy); W(Frame::invfx); W(Frame::invfy); W(F->mbf); W(F->mb); W(F->mThDepth);
    W(Frame::mnMinX); W(Frame::mnMaxX); W(Frame::mnMinY); W(Frame::mnMaxY); W(Frame::mfGridElementWidthInv); W(Frame::mfGridElementHeightInv);
    W<int32_t>(F->mnScaleLevels); W(F->mfScaleFactor); W(F->mfLogScaleFactor);
    wfv(F->mvScaleFactors); wfv(F->mvInvScaleFactors); wfv(F->mvLevelSigma2); wfv(F->mvInvLevelSigma2);
    wkeys(F->mvKeysUn); wdesc(F->mDescriptors); wfv(F->mvuRight); wfv(F->mvDepth);
    for (int c = 0; c < FRAME_GRID_COLS; c++) for (int r = 0; r < FRAME_GRID_ROWS; r++) {
        auto &cell = F->mGrid[c][r]; W<uint32_t>(cell.size()); for (auto i : cell) W<uint32_t>(i);
    }
    W<uint32_t>(F->mBowVec.size()); for (auto &e : F->mBowVec) { W<uint32_t>(e.first); W<double>(e.second); }
    wfeat(F->mFeatVec);
    // 들어올 때 장면 상태(재위치가 실패하면 그대로 남음)
    W<uint8_t>(F->HasPose()); wse3(F->HasPose() ? F->GetPose() : Sophus::SE3f());
    for (int i = 0; i < F->N; i++) W<int64_t>(mpid(F->mvpMapPoints[i]));
    for (int i = 0; i < F->N; i++) W<uint8_t>(F->mvbOutlier[i]);
    W<uint32_t>(v.size());
    std::map<long unsigned, MapPoint *> mps;
    for (KeyFrame *k : v) {
        W<uint64_t>(k->mnId); W<uint64_t>(k->mnFrameId); W<uint8_t>(k->isBad());
        wse3(k->GetPose()); wse3(k->GetPoseInverse());
        wkeys(k->mvKeysUn); wdesc(k->mDescriptors); wfv(k->mvuRight); wfv(k->mvDepth);
        auto m = k->GetMapPointMatches(); W<uint32_t>(m.size());
        for (auto *p : m) { W<int64_t>(mpid(p)); if (p) mps[p->mnId] = p; }
        wfeat(k->mFeatVec);
    }
    W<uint32_t>(mps.size());
    for (auto &e : mps) {
        MapPoint *p = e.second;
        W<uint64_t>(p->mnId); W<uint8_t>(p->isBad());
        Eigen::Vector3f x = p->GetWorldPos(), n = p->GetNormal(); Wb(x.data(), 12); Wb(n.data(), 12);
        cv::Mat d = p->GetDescriptor(); if (d.empty()) { uint8_t z[32] = {0}; Wb(z, 32); } else Wb(d.ptr(0), 32);
        W(p->mfMinDistance); W(p->mfMaxDistance);
    }
    g_ncand = v.size();
    return v;
}

typedef bool (*ItFn)(MLPnPsolver *, int, bool &, std::vector<bool> &, int &, Eigen::Matrix4f &);
bool MLPnPsolver::iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f &Tout) {
    static ItFn orig = (ItFn)dlsym(L(), "_ZN9ORB_SLAM311MLPnPsolver7iterateEiRbRSt6vectorIbSaIbEERiRN5Eigen6MatrixIfLi4ELi4ELi0ELi4ELi4EEE");
    g_rand.clear(); g_it = true;
    bool r = orig(this, nIterations, bNoMore, vbInliers, nInliers, Tout);
    g_it = false;
    if (g_in && g) {
        W<uint32_t>(0x49504e50u);  // PNPI
        auto it = g_sol.find(this);
        if (it == g_sol.end()) it = g_sol.emplace(this, (int)g_sol.size()).first;
        W<int32_t>(it->second);
        W<uint32_t>(mvpMapPointMatches.size()); for (auto *p : mvpMapPointMatches) W<int64_t>(mpid(p));
        W<int32_t>(nIterations);
        W<uint32_t>(g_rand.size()); Wb(g_rand.data(), 4 * g_rand.size());
        W<uint8_t>(r); W<uint8_t>(bNoMore); W<int32_t>(nInliers);
        W<uint32_t>(vbInliers.size()); for (bool b : vbInliers) W<uint8_t>(b);
        Wb(Tout.data(), 64);
    }
    return r;
}

typedef bool (*RelFn)(Tracking *, Map *);
bool Tracking::Relocalization(Map *pMap) {
    static RelFn orig = (RelFn)dlsym(L(), "_ZN9ORB_SLAM38Tracking14RelocalizationEPNS_3MapE");
    g_in = g && g_nrel < g_max; g_sol.clear(); g_ncand = -1;
    bool r = orig(this, pMap);
    if (g_in) {
        if (g_ncand < 0) { fprintf(stderr, "후보 질의 없이 끝남\n"); abort(); }
        W<uint32_t>(0x454c4552u);  // RELE
        W<uint8_t>(r); W<uint8_t>(mCurrentFrame.HasPose());
        wse3(mCurrentFrame.HasPose() ? mCurrentFrame.GetPose() : Sophus::SE3f());
        W<uint32_t>(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++) W<int64_t>(mpid(mCurrentFrame.mvpMapPoints[i]));
        for (int i = 0; i < mCurrentFrame.N; i++) W<uint8_t>(mCurrentFrame.mvbOutlier[i]);
        g_nrel++;
        fflush(g);
    }
    g_in = false;
    return r;
}

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}

// 순서: "a-b,c-d:s" = a..b-1 을 차례로, c..d-1 을 s 칸씩(되감기 · 건너뛰기)
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "reloc_dump voc yaml seqdir out.bin order [max_reloc]\n"); return 1; }
    std::string dir = argv[3];
    g = fopen(argv[4], "wb");
    if (argc > 6) g_max = atol(argv[6]);
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
    fclose(g);
    fprintf(stderr, "frames %d reloc %ld\n", n, g_nrel);
    return 0;
}
