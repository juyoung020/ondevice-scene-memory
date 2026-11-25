// 지도 스레드 단계별 지도 요약 대조(원본 = 기준 빌드 ~/opt/orb3_refid). 키프레임마다 ProcessNewKeyFrame · MapPointCulling ·
// CreateNewMapPoints · SearchInNeighbors · LocalBundleAdjustment · KeyFrameCulling 뒤에 지금 지도 요약 한 줄(steps.txt).
// 요약 식은 Rust `orb3sys::mapping::step_digest` 와 같음. 재생은 examples/orb3_sys_run 과 같은 장면 고르기 · 시각 · 장면마다 WaitMappingIdle.
// argv: voc yaml dir max_frames out.txt
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <any>
#include <mutex>
#include <thread>
#include <list>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#define private public
#define protected public
#include <System.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Atlas.h>
#include <LocalMapping.h>
#include <Tracking.h>
#include <Optimizer.h>
#include <LoopClosing.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static FILE *g = nullptr;
static void *L() { static void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD); return h; }
struct H { uint64_t h = 1469598103934665603ull; void b(const void *p, size_t n) { auto *c = (const uint8_t *)p; for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; } }
  void u(uint64_t v) { b(&v, 8); } void i(int32_t v) { b(&v, 4); } void f(float v) { b(&v, 4); } };
static void digest(LocalMapping *lm, const char *step) {
    if (!g) return;
    KeyFrame *K = lm->mpCurrentKeyFrame; Map *M = K->GetMap();
    std::map<uint64_t, KeyFrame *> ks; for (KeyFrame *k : M->mspKeyFrames) if (!k->mbBad) ks[k->mnId] = k;
    std::map<uint64_t, MapPoint *> ps; for (MapPoint *p : M->mspMapPoints) if (!p->mbBad) ps[p->mnId] = p;
    H hk, hc, hm, hp, ho;
    for (auto &e : ks) { KeyFrame *k = e.second;
        hk.u(k->mnId); auto q = k->mTcw.unit_quaternion(); hk.f(q.x()); hk.f(q.y()); hk.f(q.z()); hk.f(q.w()); auto t = k->mTcw.translation(); hk.f(t.x()); hk.f(t.y()); hk.f(t.z());
        hc.u(k->mnId); for (auto &c : k->mConnectedKeyFrameWeights) { hc.u(c.first->mnId); hc.i(c.second); } hc.u(k->mpParent ? k->mpParent->mnId : ~0ull);
        for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) { hc.u(k->mvpOrderedConnectedKeyFrames[i]->mnId); hc.i(k->mvOrderedWeights[i]); }
        hm.u(k->mnId); for (MapPoint *p : k->mvpMapPoints) hm.u(p ? p->mnId : ~0ull); }
    for (auto &e : ps) { MapPoint *p = e.second;
        hp.u(p->mnId); for (int j = 0; j < 3; j++) hp.f(p->mWorldPos[j]); for (int j = 0; j < 3; j++) hp.f(p->mNormalVector[j]); hp.f(p->mfMinDistance); hp.f(p->mfMaxDistance);
        ho.u(p->mnId); ho.i(p->nObs); ho.u(p->mpRefKF ? p->mpRefKF->mnId : ~0ull); for (auto &o : p->mObservations) { ho.u(o.first->mnId); ho.i(std::get<0>(o.second)); ho.i(std::get<1>(o.second)); } }
    if (getenv("ORBSYS_STEPS_FULL")) for (auto &e : ks) { KeyFrame *k = e.second; fprintf(g, "  K%lu p%ld w", k->mnId, k->mpParent ? (long)k->mpParent->mnId : -1L);
        for (auto &c : k->mConnectedKeyFrameWeights) fprintf(g, " %lu:%d", c.first->mnId, c.second); fprintf(g, " o");
        for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) fprintf(g, " %lu:%d", k->mvpOrderedConnectedKeyFrames[i]->mnId, k->mvOrderedWeights[i]); fprintf(g, "\n"); }
    fprintf(g, "%lu %s %zu %zu %016lx %016lx %016lx %016lx %016lx\n", K->mnId, step, ks.size(), ps.size(), hk.h, hc.h, hm.h, hp.h, ho.h);
}
#define HOOK(NAME, SYM, STEP) \
  void LocalMapping::NAME() { typedef void (*F)(LocalMapping *); static F o = (F)dlsym(L(), SYM); o(this); digest(this, STEP); }
HOOK(ProcessNewKeyFrame, "_ZN9ORB_SLAM312LocalMapping18ProcessNewKeyFrameEv", "pnk")
HOOK(MapPointCulling, "_ZN9ORB_SLAM312LocalMapping15MapPointCullingEv", "mpc")
HOOK(CreateNewMapPoints, "_ZN9ORB_SLAM312LocalMapping18CreateNewMapPointsEv", "cnmp")
HOOK(SearchInNeighbors, "_ZN9ORB_SLAM312LocalMapping17SearchInNeighborsEv", "sin")
HOOK(KeyFrameCulling, "_ZN9ORB_SLAM312LocalMapping15KeyFrameCullingEv", "kfc")
static LocalMapping *g_lm = nullptr;
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int &a, int &b, int &c, int &d) {
    typedef void (*F)(KeyFrame *, bool *, Map *, int &, int &, int &, int &);
    static F o = (F)dlsym(L(), "_ZN9ORB_SLAM39Optimizer21LocalBundleAdjustmentEPNS_8KeyFrameEPbPNS_3MapERiS6_S6_S6_");
    o(pKF, pbStopFlag, pMap, a, b, c, d);
    if (g_lm) digest(g_lm, "lba");
}
// 루프 스레드까지 기다리기(Rust 쪽 ORBSYS_WAIT_LOOP 와 짝): 넣은 수 · 공통 영역 찾기 끝난 수
#include <atomic>
static std::atomic<long> g_last{-1};
void LoopClosing::InsertKeyFrame(KeyFrame *pKF) {
    typedef void (*F)(LoopClosing *, KeyFrame *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing14InsertKeyFrameEPNS_8KeyFrameE");
    if (pKF->mnId != 0) g_last = (long)pKF->mnId; o(this, pKF);  // 원본도 0 번은 대기열에 안 넣음
}
// Run 한 바퀴 끝의 mpLastCurrentKF = mpCurrentKF 가 마지막으로 넣은 키프레임이 될 때까지(고치기 · 합치기 · 전역 BA 포함)
static void waitLoop(System &S) {
    LoopClosing *lc = S.mpLoopCloser;
    while (g_last.load() >= 0 && !(lc->mpLastCurrentKF && (long)lc->mpLastCurrentKF->mnId == g_last.load() && !lc->CheckNewKeyFrames())) usleep(200);
    while (lc->isRunningGBA()) usleep(1000);
}
static void load(const std::string &f, std::vector<std::pair<double, std::string>> &v) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) { if (l.empty() || l[0] == '#') continue; std::istringstream s(l); double t; std::string q; s >> t >> q; v.push_back({t, q}); }
}
int main(int argc, char **argv) {
    std::string dir = argv[3]; int maxf = atoi(argv[4]);
    std::vector<std::pair<double, std::string>> rgb, dep; load(dir + "/rgb.txt", rgb); load(dir + "/depth.txt", dep);
    System S(argv[1], argv[2], System::RGBD, false);
    g_lm = S.mpLocalMapper;
    g = fopen(argv[5], "w");
    size_t j = 0; int n = 0;
    for (size_t i = 0; i < rgb.size() && n < maxf; i++) {
        double t = rgb[i].first;
        while (j + 1 < dep.size() && std::fabs(dep[j + 1].first - t) < std::fabs(dep[j].first - t)) j++;
        if (std::fabs(dep[j].first - t) > 0.02) continue;
        cv::Mat C = cv::imread(dir + "/" + rgb[i].second, cv::IMREAD_UNCHANGED), D = cv::imread(dir + "/" + dep[j].second, cv::IMREAD_UNCHANGED);
        S.TrackRGBD(C, D, t); S.GetTracker()->WaitMappingIdle(); waitLoop(S); S.GetTracker()->WaitMappingIdle(); n++;
        fflush(g);
    }
    S.Shutdown(); fclose(g); g = nullptr; fprintf(stderr, "frames %d\n", n); return 0;
}
