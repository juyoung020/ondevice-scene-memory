// 원본 LoopClosing::MergeLocal 대조 덤프(MergeLocal 판).(번호 순 비교자 기준 빌드 ~/opt/orb3_refid).
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

static FILE *g = nullptr, *g_file = nullptr;
static char *g_buf = nullptr; static size_t g_bufn = 0;
static thread_local bool g_in = false;
static thread_local bool g_merge = false;
static thread_local int g_rel = 0;
static thread_local LoopClosing *g_lc = nullptr;
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

// 지도 신원: 합치기 들어올 때(MERG) 지도 포인터마다 그때 번호. 원본 MergeLocal 의 ChangeId 뒤에는 합칠 지도와 옛 지도가
// 같은 번호가 되므로, 덤프는 번호 대신 포인터 신원을 쓴다(원본 비교도 포인터)
static std::map<Map *, uint64_t> g_ident;
static uint64_t mid(Map *m) { auto it = g_ident.find(m); return it != g_ident.end() ? it->second : m->mnId; }
// 지도 통째: 이 지도의 키프레임 전부(나쁜 것 포함, mspKeyFrames) · 그 키프레임이 가리키는 지도 점 전부
static void dumpMap(Atlas *A) {
    std::map<long unsigned, KeyFrame *> S;
    std::map<long unsigned, MapPoint *> mps;
    std::vector<Map *> ms(A->mspMaps.begin(), A->mspMaps.end());
    for (Map *b : A->mspBadMaps) ms.push_back(b);
    W<uint32_t>(ms.size());
    for (Map *M : ms) {
        W<uint64_t>(mid(M)); W<uint8_t>(M->mbBad); W<uint8_t>(M == A->mpCurrentMap);
        W<int64_t>(kfid(M->mpKFinitial)); W<int64_t>(kfid(M->mpKFlowerID)); W<uint64_t>(M->mnInitKFid); W<uint64_t>(M->mnMaxKFid);
        wkfset(M->mspKeyFrames); W<uint32_t>(M->mspMapPoints.size()); for (MapPoint *p : M->mspMapPoints) W<uint64_t>(p->mnId);
        for (KeyFrame *k : M->mspKeyFrames) S[k->mnId] = k;
        for (MapPoint *p : M->mspMapPoints) mps[p->mnId] = p;
    }
    for (auto &e : S) for (MapPoint *p : e.second->mvpMapPoints) if (p) mps[p->mnId] = p;
    // 지도 점이 가리키는 키프레임(관측 · 기준)도 모두(다른 지도 · 지도에서 빠진 것 포함)
    for (auto &e : mps) { for (auto &q : e.second->mObservations) S[q.first->mnId] = q.first; if (e.second->mpRefKF) S[e.second->mpRefKF->mnId] = e.second->mpRefKF; }
    KeyFrame *k0 = S.begin()->second;
    W(k0->fx); W(k0->fy); W(k0->cx); W(k0->cy); W(k0->invfx); W(k0->invfy); W(k0->mbf); W(k0->mb); W(k0->mThDepth);
    W((float)k0->mnMinX); W((float)k0->mnMaxX); W((float)k0->mnMinY); W((float)k0->mnMaxY); W(k0->mfGridElementWidthInv); W(k0->mfGridElementHeightInv);
    W<int32_t>(k0->mnScaleLevels); W(k0->mfScaleFactor); W(k0->mfLogScaleFactor);
    wfv(k0->mvScaleFactors); { std::vector<float> iv; for (float s : k0->mvScaleFactors) iv.push_back(1.0f / s); wfv(iv); } wfv(k0->mvLevelSigma2); wfv(k0->mvInvLevelSigma2);
    W<uint32_t>(S.size());
    for (auto &e : S) {
        KeyFrame *k = e.second;
        W<uint64_t>(k->mnId); W<uint64_t>(k->mnFrameId); W<uint8_t>(k->mbBad); W<uint8_t>(k->mbNotErase); W<uint8_t>(k->mbFirstConnection);
        W<int64_t>(k->mpMap ? (int64_t)mid(k->mpMap) : -1);
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
        W<uint64_t>(p->mnId); W<uint8_t>(p->mbBad); W<int64_t>(p->mpMap ? (int64_t)mid(p->mpMap) : -1);
        Wb(p->mWorldPos.data(), 12); Wb(p->mNormalVector.data(), 12);
        if (p->mDescriptor.empty()) { uint8_t z[32] = {0}; Wb(z, 32); } else Wb(p->mDescriptor.ptr(0), 32);
        W(p->mfMinDistance); W(p->mfMaxDistance); W<int32_t>(p->mnTrackScaleLevel);
        W<int64_t>(kfid(p->mpRefKF)); W<int64_t>(mpid(p->mpReplaced)); W<int32_t>(p->nObs); W<int32_t>(p->mnVisible); W<int32_t>(p->mnFound);
        W<uint64_t>(p->mnCorrectedByKF); W<uint64_t>(p->mnCorrectedReference);
        W<uint32_t>(p->mObservations.size());
        for (auto &q : p->mObservations) { W<uint64_t>(q.first->mnId); W<int32_t>(std::get<0>(q.second)); W<int32_t>(std::get<1>(q.second)); }
    }
}


static void wkfv(const std::vector<KeyFrame *> &v) { W<uint32_t>(v.size()); for (auto *k : v) W<int64_t>(kfid(k)); }

typedef void (*LbaFn)(KeyFrame *, std::vector<KeyFrame *>, std::vector<KeyFrame *>, bool *);
void Optimizer::LocalBundleAdjustment(KeyFrame *pMainKF, std::vector<KeyFrame *> vpAdjustKF, std::vector<KeyFrame *> vpFixedKF, bool *pbStopFlag) {
    static LbaFn orig = (LbaFn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer21LocalBundleAdjustmentEPNS_8KeyFrameESt6vectorIS2_SaIS2_EES5_Pb");
    if (!g_merge) { orig(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag); return; }
    W<uint32_t>(0x4241424cu);  // LBAB
    W<uint64_t>(pMainKF->mnId); wkfv(vpAdjustKF); wkfv(vpFixedKF);
    dumpMap(g_lc->mpAtlas);
    orig(pMainKF, vpAdjustKF, vpFixedKF, pbStopFlag);
    W<uint32_t>(0x4141424cu);  // LBAA
    dumpMap(g_lc->mpAtlas);
}

typedef void (*Ess2Fn)(KeyFrame *, std::vector<KeyFrame *> &, std::vector<KeyFrame *> &, std::vector<KeyFrame *> &, std::vector<MapPoint *> &);
void Optimizer::OptimizeEssentialGraph(KeyFrame *pCurKF, std::vector<KeyFrame *> &vpFixedKFs, std::vector<KeyFrame *> &vpFixedCorrectedKFs,
                                       std::vector<KeyFrame *> &vpNonFixedKFs, std::vector<MapPoint *> &vpNonCorrectedMPs) {
    static Ess2Fn orig = (Ess2Fn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer22OptimizeEssentialGraphEPNS_8KeyFrameERSt6vectorIS2_SaIS2_EES6_S6_RS3_IPNS_8MapPointESaIS8_EE");
    if (!g_merge) { orig(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs); return; }
    W<uint32_t>(0x42325345u);  // ES2B
    W<uint64_t>(pCurKF->mnId); wkfv(vpFixedKFs); wkfv(vpFixedCorrectedKFs); wkfv(vpNonFixedKFs); wmps(vpNonCorrectedMPs);
    dumpMap(g_lc->mpAtlas);
    orig(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs);
    W<uint32_t>(0x41325345u);  // ES2A
    dumpMap(g_lc->mpAtlas);
}

// MergeLocal 안의 Release: 첫째(전역 맞춤 BA 뒤)는 풀지 않음(원본은 풀었다 곧 다시 멈춤 — 그 사이 지도 스레드가 돌면 덤프가 어긋남),
// 둘째(지도 옮긴 뒤) 직전에 지도 덤프
typedef void (*RelFn)(LocalMapping *);
void LocalMapping::Release() {
    static RelFn orig = (RelFn)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping7ReleaseEv");
    if (g_merge && g) {
        g_rel++;
        if (g_rel == 1) return;
        if (g_rel == 2) { W<uint32_t>(0x4c45524du); dumpMap(g_lc->mpAtlas); }  // MREL
    }
    orig(this);
}

// 패치의 옮겨 붙이기(GRAFT)가 먼저 지도를 합쳐 MergeLocal 이 거의 안 불림 → NOGRAFT=1 이면 TryGraft 를 끔(MergeLocal 대조 입력용)
typedef void (*TgFn)(Tracking *);
void Tracking::TryGraft() {
    static TgFn orig = (TgFn)dlsym(L(), "_ZN9ORB_SLAM38Tracking8TryGraftEv");
    static bool off = getenv("NOGRAFT") != nullptr;
    if (!off) orig(this);
}

typedef void (*MrgFn)(LoopClosing *);
void LoopClosing::MergeLocal() {
    static MrgFn orig = (MrgFn)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing10MergeLocalEv");
    if (!g) { orig(this); return; }
    bool gba = isRunningGBA();
    mpLocalMapper->RequestStop();
    while (!mpLocalMapper->isStopped()) usleep(1000);
    mpLocalMapper->EmptyQueue();
    g = open_memstream(&g_buf, &g_bufn);  // 합치기 하나를 다 쓴 뒤에만 파일로(끝날 때 잘린 기록 없게)
    W<uint32_t>(0x4752454du);  // MERG
    W<uint64_t>(mpCurrentKF->mnId); W<uint64_t>(mpMergeMatchedKF->mnId); W<uint8_t>(mbFixScale); W<uint8_t>(gba);
    wsim(mg2oMergeScw);
    g_ident.clear();
    for (Map *m : mpAtlas->mspMaps) g_ident[m] = m->mnId;
    for (Map *m : mpAtlas->mspBadMaps) g_ident[m] = m->mnId;
    dumpMap(mpAtlas);
    g_merge = true; g_rel = 0; g_lc = this;
    orig(this);
    g_merge = false;
    g_ident.clear();
    if (g_rel < 2) { fprintf(stderr, "MergeLocal Release %d", g_rel); fputc(10, stderr); abort(); }
    W<uint32_t>(0x45475245u);  // ERGE
    W<uint8_t>(mbRunningGBA);
    W<uint32_t>(mpAtlas->mspMaps.size()); for (Map *m : mpAtlas->mspMaps) W<uint64_t>(m->mnId);
    W<uint64_t>(mpAtlas->mpCurrentMap->mnId);
    wkfset(mpCurrentKF->mspMergeEdges); wkfset(mpMergeMatchedKF->mspMergeEdges);
    W<uint8_t>(mpCurrentKF->mbNotErase); W<uint8_t>(mpMergeMatchedKF->mbNotErase);
    fclose(g); fwrite(g_buf, 1, g_bufn, g_file); fflush(g_file); free(g_buf); g_buf = nullptr; g = g_file; g_n++;
}

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}

// 입력: (seqdir order) 짝을 여럿 — 순서 "a-b,c-d:s". 여러 녹화를 이어 붙여 다른 지도가 생겼다 같은 곳을 다시 보게 함
int main(int argc, char **argv) {
    if (argc < 6 || (argc - 4) % 2) { fprintf(stderr, "merge_dump voc yaml out.bin seqdir order [seqdir order ...]\n"); return 1; }
    g = g_file = fopen(argv[3], "wb");
    std::vector<std::pair<std::string, std::string>> frames;  // rgb, depth 경로
    for (int a = 4; a + 1 < argc; a += 2) {
        std::string dir = argv[a];
        std::vector<double> tr, td; std::vector<std::string> pr, pd;
        load(dir + "/rgb.txt", tr, pr); load(dir + "/depth.txt", td, pd);
        std::vector<std::pair<int, int>> fr;
        size_t j = 0;
        for (size_t i = 0; i < tr.size(); i++) {
            while (j + 1 < td.size() && std::abs(td[j + 1] - tr[i]) <= std::abs(td[j] - tr[i])) j++;
            if (std::abs(td[j] - tr[i]) <= 0.02) fr.push_back({(int)i, (int)j});
        }
        std::stringstream os(argv[a + 1]); std::string seg;
        while (std::getline(os, seg, ',')) {
            int x = 0, y = 0, st = 1; char c;
            std::stringstream ss(seg); ss >> x >> c >> y; if (ss >> c) ss >> st;
            for (int k = x; k < y && k < (int)fr.size(); k += st) frames.push_back({dir + "/" + pr[fr[k].first], dir + "/" + pd[fr[k].second]});
        }
    }
    System S(argv[1], argv[2], System::RGBD, false);
    double t = 0; int n = 0;
    for (auto &f : frames) {
        cv::Mat C = cv::imread(f.first, cv::IMREAD_UNCHANGED), D = cv::imread(f.second, cv::IMREAD_UNCHANGED);
        if (C.empty() || D.empty()) continue;
        t += 1.0 / 30;
        S.TrackRGBD(C, D, t);
        usleep(20000);
        n++;
    }
    S.Shutdown();
    fclose(g_file); g = g_file = nullptr;
    fprintf(stderr, "frames %d merges %ld", n, g_n); fputc(10, stderr);
    return 0;
}
