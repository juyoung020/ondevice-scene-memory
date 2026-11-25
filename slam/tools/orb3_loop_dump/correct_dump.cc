// 원본 LoopClosing::CorrectLoop 대조 덤프(번호 순 비교자 기준 빌드 ~/opt/orb3_refid).
// 루프 닫기마다: 지도 스레드 멈춘 뒤 지도 통째 상태 · 입력(LOOP), OptimizeEssentialGraph 들어갈 때 입력 · 지도(ESSB),
// 나올 때 지도(ESSA), CorrectLoop 끝 지도(LOOE).
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
#include <Atlas.h>
#include <LoopClosing.h>
#include <LocalMapping.h>
#include <Optimizer.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static FILE *g = nullptr;
static thread_local bool g_in = false;
static thread_local bool g_endDumped = false;
static long g_n = 0;
template <class T> static void W(const T &v) { fwrite(&v, sizeof(T), 1, g); }
static void Wb(const void *p, size_t n) { fwrite(p, 1, n, g); }
static void *L() { static void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD); return h; }

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
template <class C> static void wkfset(const C &s) { W<uint32_t>(s.size()); for (auto *k : s) W<uint64_t>(k->mnId); }

// 지도 통째: 이 지도의 키프레임 전부(나쁜 것 포함, mspKeyFrames) · 그 키프레임이 가리키는 지도 점 전부
static void dumpMap(Map *M) {
    std::map<long unsigned, KeyFrame *> S;
    for (KeyFrame *k : M->mspKeyFrames) S[k->mnId] = k;
    std::map<long unsigned, MapPoint *> mps;
    for (MapPoint *p : M->mspMapPoints) mps[p->mnId] = p;
    for (auto &e : S) for (MapPoint *p : e.second->mvpMapPoints) if (p) mps[p->mnId] = p;
    // 지도 점이 가리키는 키프레임(관측 · 기준)도 모두(다른 지도 · 지도에서 빠진 것 포함)
    for (auto &e : mps) { for (auto &q : e.second->mObservations) S[q.first->mnId] = q.first; if (e.second->mpRefKF) S[e.second->mpRefKF->mnId] = e.second->mpRefKF; }
    KeyFrame *k0 = S.begin()->second;
    W<uint64_t>(M->GetId());
    W(k0->fx); W(k0->fy); W(k0->cx); W(k0->cy); W(k0->invfx); W(k0->invfy); W(k0->mbf); W(k0->mb); W(k0->mThDepth);
    W((float)k0->mnMinX); W((float)k0->mnMaxX); W((float)k0->mnMinY); W((float)k0->mnMaxY); W(k0->mfGridElementWidthInv); W(k0->mfGridElementHeightInv);
    W<int32_t>(k0->mnScaleLevels); W(k0->mfScaleFactor); W(k0->mfLogScaleFactor);
    wfv(k0->mvScaleFactors); { std::vector<float> iv; for (float s : k0->mvScaleFactors) iv.push_back(1.0f / s); wfv(iv); } wfv(k0->mvLevelSigma2); wfv(k0->mvInvLevelSigma2);
    W<uint32_t>(S.size());
    for (auto &e : S) {
        KeyFrame *k = e.second;
        W<uint64_t>(k->mnId); W<uint64_t>(k->mnFrameId); W<uint8_t>(k->mbBad); W<uint8_t>(k->mbNotErase);
        W<int64_t>(k->mpMap ? (int64_t)k->mpMap->GetId() : -1);
        wse3(k->mTcw); wse3(k->mTwc);
        wkeys(k->mvKeysUn); wdesc(k->mDescriptors); wfv(k->mvuRight); wfv(k->mvDepth);
        wmps(k->mvpMapPoints);
        wfeat(k->mFeatVec);
        for (int c = 0; c < k->mnGridCols; c++) for (int r = 0; r < k->mnGridRows; r++) {
            auto &cell = k->mGrid[c][r]; W<uint32_t>(cell.size()); for (auto i : cell) W<uint32_t>(i);
        }
        W<uint32_t>(k->mConnectedKeyFrameWeights.size());
        for (auto &c : k->mConnectedKeyFrameWeights) { W<uint64_t>(c.first->mnId); W<int32_t>(c.second); }
        W<uint32_t>(k->mvpOrderedConnectedKeyFrames.size());
        for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) { W<uint64_t>(k->mvpOrderedConnectedKeyFrames[i]->mnId); W<int32_t>(k->mvOrderedWeights[i]); }
        W<int64_t>(kfid(k->mpParent)); wkfset(k->mspChildrens); wkfset(k->mspLoopEdges); wkfset(k->mspMergeEdges);
    }
    W<uint32_t>(mps.size());
    for (auto &e : mps) {
        MapPoint *p = e.second;
        W<uint64_t>(p->mnId); W<uint8_t>(p->mbBad); W<int64_t>(p->mpMap ? (int64_t)p->mpMap->GetId() : -1);
        Wb(p->mWorldPos.data(), 12); Wb(p->mNormalVector.data(), 12);
        if (p->mDescriptor.empty()) { uint8_t z[32] = {0}; Wb(z, 32); } else Wb(p->mDescriptor.ptr(0), 32);
        W(p->mfMinDistance); W(p->mfMaxDistance); W<int32_t>(p->mnTrackScaleLevel);
        W<int64_t>(kfid(p->mpRefKF)); W<int64_t>(mpid(p->mpReplaced)); W<int32_t>(p->nObs); W<int32_t>(p->mnVisible); W<int32_t>(p->mnFound);
        W<uint64_t>(p->mnCorrectedByKF); W<uint64_t>(p->mnCorrectedReference);
        W<uint32_t>(p->mObservations.size());
        for (auto &q : p->mObservations) { W<uint64_t>(q.first->mnId); W<int32_t>(std::get<0>(q.second)); W<int32_t>(std::get<1>(q.second)); }
    }
}

typedef void (*EssFn)(Map *, KeyFrame *, KeyFrame *, const LoopClosing::KeyFrameAndPose &, const LoopClosing::KeyFrameAndPose &,
                      const std::map<KeyFrame *, std::set<KeyFrame *>> &, const bool &);
void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF, const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3, const std::map<KeyFrame *, std::set<KeyFrame *>> &LoopConnections,
                                       const bool &bFixScale) {
    static EssFn orig = (EssFn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer22OptimizeEssentialGraphEPNS_3MapEPNS_8KeyFrameES4_RKSt3mapIS4_N3g2o4Sim3ESt4lessIS4_EN5Eigen17aligned_allocatorISt4pairIKS4_S7_EEEESI_RKS5_IS4_St3setIS4_S9_SaIS4_EES9_SaISC_ISD_SL_EEERKb");
    if (!g_in) { orig(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale); return; }
    W<uint32_t>(0x42535345u);  // ESSB
    W<uint64_t>(pLoopKF->mnId); W<uint64_t>(pCurKF->mnId); W<uint8_t>(bFixScale);
    for (auto *m : {&NonCorrectedSim3, &CorrectedSim3}) { W<uint32_t>(m->size()); for (auto &e : *m) { W<uint64_t>(e.first->mnId); wsim(e.second); } }
    W<uint32_t>(LoopConnections.size());
    for (auto &e : LoopConnections) { W<uint64_t>(e.first->mnId); wkfset(e.second); }
    dumpMap(pMap);
    orig(pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale);
    W<uint32_t>(0x41535345u);  // ESSA
    dumpMap(pMap);
}

static thread_local LoopClosing *g_lc = nullptr;
// CorrectLoop 끝의 mpLocalMapper->Release() 직전(지도 스레드가 다시 돌기 전) 지도를 덤프
typedef void (*RelFn)(LocalMapping *);
void LocalMapping::Release() {
    static RelFn orig = (RelFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping7ReleaseEv");
    if (g_in && g && !g_endDumped && g_lc) {
        W<uint32_t>(0x454f4f4cu);  // LOOE
        W<uint8_t>(g_lc->mbRunningGBA);
        dumpMap(g_lc->mpCurrentKF->GetMap());
        g_endDumped = true;
    }
    orig(this);
}

typedef void (*CorFn)(LoopClosing *);
void LoopClosing::CorrectLoop() {
    static CorFn orig = (CorFn)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing11CorrectLoopEv");
    if (!g) { orig(this); return; }
    // 원본 앞부분(멈춤 요청 · 대기열 비우기 · GBA 멈춤 · 멈출 때까지)을 먼저 해서 그 뒤 상태를 덤프. 원본이 다시 해도 바뀌는 것 없음
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue();
    if (isRunningGBA()) {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbStopGBA = true;
        mnFullBAIdx++;
        if (mpThreadGBA) { mpThreadGBA->detach(); delete mpThreadGBA; mpThreadGBA = nullptr; }
    }
    while (!mpLocalMapper->isStopped()) usleep(1000);
    W<uint32_t>(0x504f4f4cu);  // LOOP
    W<uint64_t>(mpCurrentKF->mnId); W<uint64_t>(mpLoopMatchedKF->mnId); W<uint8_t>(mbFixScale);
    wsim(mg2oLoopScw); wmps(mvpLoopMapPoints); wmps(mvpLoopMatchedMPs);
    W<uint32_t>(mpAtlas->CountMaps());
    dumpMap(mpCurrentKF->GetMap());
    g_in = true; g_endDumped = false; g_lc = this;
    orig(this);
    g_in = false;
    if (!g_endDumped) { fprintf(stderr, "CorrectLoop 끝 덤프 없음"); fputc(10, stderr); abort(); }
    fflush(g); g_n++;
}

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "correct_dump voc yaml seqdir out.bin order\n"); return 1; }
    std::string dir = argv[3];
    g = fopen(argv[4], "wb");
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
    fclose(g); g = nullptr;
    fprintf(stderr, "frames %d loops %ld\n", n, g_n);
    return 0;
}
