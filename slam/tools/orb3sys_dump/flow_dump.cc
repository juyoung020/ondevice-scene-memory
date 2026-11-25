// (correct_dump.cc 바탕) 원본 RunGlobalBundleAdjustment 대조 덤프(번호 순 비교자 기준 빌드 ~/opt/orb3_refid).
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
#include <algorithm>
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


// 원본 LoopClosing::RunGlobalBundleAdjustment 결과 넣기 대조(통합 orb3sys::gbaops::apply_gba).
// GlobalBundleAdjustemnt 가 끝난 직후: 지도 + 키프레임 mnBAGlobalForKF · mTcwGBA + 점 mnBAGlobalForKF · mPosGBA + mvpKeyFrameOrigins (GBAB)
// 같은 스레드의 다음 LocalMapping::Release(결과 넣기 끝, 지도 스레드 다시 돌기 전): 지도 (GBAA)
static thread_local Map *g_gmap = nullptr;
static thread_local long g_gloop = -1;
static thread_local bool g_snap = false;
static void dumpBefore() {
    Map *pMap = g_gmap; unsigned long nLoopKF = (unsigned long)g_gloop;
    W<uint32_t>(0x42414247u);  // GBAB
    W<uint64_t>(nLoopKF); W<uint8_t>(0);
    dumpMap(pMap);
    std::vector<KeyFrame *> kfs = pMap->GetAllKeyFrames();
    W<uint32_t>(kfs.size());
    for (KeyFrame *k : kfs) { W<uint64_t>(k->mnId); W<uint64_t>(k->mnBAGlobalForKF); wse3(k->mTcwGBA); }
    std::vector<MapPoint *> mps = pMap->GetAllMapPoints();
    W<uint32_t>(mps.size());
    for (MapPoint *p : mps) { W<uint64_t>(p->mnId); W<uint64_t>(p->mnBAGlobalForKF); Wb(p->mPosGBA.data(), 12); }
    W<uint32_t>(pMap->mvpKeyFrameOrigins.size());
    for (KeyFrame *k : pMap->mvpKeyFrameOrigins) W<uint64_t>(k->mnId);
    fflush(g);
}
// 결과 넣기 직전 상태: GBA 스레드에서 지도 스레드가 멈췄다고 처음 답할 때(원본은 곧바로 지도 잠금을 잡고 넣음)
typedef bool (*StFn)(LocalMapping *);
bool LocalMapping::isStopped() {
    static StFn orig = (StFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping9isStoppedEv");
    bool r = orig(this);
    if (r && g && g_gmap && !g_snap) { std::unique_lock<std::mutex> lk(g_gmap->mMutexMapUpdate); dumpBefore(); g_snap = true; }
    return r;
}
typedef void (*GbaFn)(Map *, int, bool *, unsigned long, bool);
void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust) {
    static GbaFn orig = (GbaFn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer22GlobalBundleAdjustemntEPNS_3MapEiPbmb");
    orig(pMap, nIterations, pbStopFlag, nLoopKF, bRobust);
    if (!g || nLoopKF == 0) return;
    g_snap = false;
    g_gmap = pMap; g_gloop = (long)nLoopKF;
}
typedef void (*RelFn)(LocalMapping *);
void LocalMapping::Release() {
    static RelFn orig = (RelFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping7ReleaseEv");
    if (g && g_gmap && g_snap) {
        W<uint32_t>(0x41414247u);  // GBAA
        W<uint64_t>((uint64_t)g_gloop);
        dumpMap(g_gmap);
        fflush(g); g_n++;
        g_gmap = nullptr;
    }
    orig(this);
}


// 새 점 넣기 대조(통합 orb3sys::rsys::create_new_map_points_into): ORBSYS_CNMP_EVERY=n 이면 n 번째마다
// CNMB(지금 KF · MapPoint::nNextId · 지도 통째) → 원본 CreateNewMapPoints(지도 갱신 잠금 쥠, CheckNewKeyFrames 답 기록) → CNMA(답들 · 새 점 전부 · 최근 점 목록 끝)
static thread_local std::vector<uint8_t> *t_chk = nullptr;
typedef bool (*ChkFn)(LocalMapping *);
bool LocalMapping::CheckNewKeyFrames() {
    static ChkFn orig = (ChkFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping17CheckNewKeyFramesEv");
    bool r = orig(this);
    if (t_chk) t_chk->push_back(r);
    return r;
}
static long g_cn = 0;
typedef void (*CnFn)(LocalMapping *);
void LocalMapping::CreateNewMapPoints() {
    static CnFn orig = (CnFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping18CreateNewMapPointsEv");
    static long every = getenv("ORBSYS_CNMP_EVERY") ? atol(getenv("ORBSYS_CNMP_EVERY")) : 0;
    if (!g || every <= 0 || (g_cn++ % every) != 0) { orig(this); return; }
    KeyFrame *K = mpCurrentKeyFrame;
    Map *M = K->GetMap();
    std::unique_lock<std::mutex> lk(M->mMutexMapUpdate);
    const long unsigned n0 = MapPoint::nNextId;
    const size_t r0 = mlpRecentAddedMapPoints.size();
    W<uint32_t>(0x424d4e43u);  // CNMB
    W<uint64_t>(K->mnId); W<uint64_t>(n0); W<uint8_t>(mbFarPoints); W(mThFarPoints);
    dumpMap(M);
    // 보정 전 좌표 mvKeys(UnprojectStereo 가 씀): 지도의 키프레임 전부
    { std::vector<KeyFrame *> ks = M->GetAllKeyFrames(); W<uint32_t>(ks.size());
      for (KeyFrame *k : ks) { W<uint64_t>(k->mnId); W<uint32_t>(k->mvKeys.size()); for (auto &p : k->mvKeys) { W(p.pt.x); W(p.pt.y); } } }
    std::vector<uint8_t> chk; t_chk = &chk;
    orig(this);
    t_chk = nullptr;
    W<uint32_t>(0x414d4e43u);  // CNMA
    W<uint32_t>(chk.size()); for (uint8_t c : chk) W(c);
    std::vector<MapPoint *> nw;
    for (MapPoint *p : M->GetAllMapPoints()) if (p->mnId >= n0) nw.push_back(p);
    std::sort(nw.begin(), nw.end(), [](MapPoint *a, MapPoint *b) { return a->mnId < b->mnId; });
    W<uint32_t>(nw.size());
    for (MapPoint *p : nw) {
        W<uint64_t>(p->mnId); W<uint8_t>(p->mbBad); Wb(p->mWorldPos.data(), 12); Wb(p->mNormalVector.data(), 12);
        Wb(p->mDescriptor.ptr(0), 32); W(p->mfMinDistance); W(p->mfMaxDistance);
        W<int64_t>(kfid(p->mpRefKF)); W<int32_t>(p->nObs); W<int32_t>(p->mnVisible); W<int32_t>(p->mnFound);
        W<int64_t>(p->mnFirstKFid); W<int64_t>(p->mnFirstFrame);
        W<uint32_t>(p->mObservations.size());
        for (auto &q : p->mObservations) { W<uint64_t>(q.first->mnId); W<int32_t>(std::get<0>(q.second)); W<int32_t>(std::get<1>(q.second)); }
    }
    { std::vector<int64_t> rt; auto it = mlpRecentAddedMapPoints.begin(); std::advance(it, std::min(r0, mlpRecentAddedMapPoints.size())); for (; it != mlpRecentAddedMapPoints.end(); ++it) rt.push_back((*it)->mnId);
      W<uint32_t>(rt.size()); for (auto v : rt) W(v); }
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
    if (argc < 6) { fprintf(stderr, "gba_dump voc yaml seqdir out.bin order\n"); return 1; }
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
    fprintf(stderr, "frames %d gba %ld\n", n, g_n);
    return 0;
}
