// 지도 스레드 단계별 지도 요약 대조(원본 = 기준 빌드 ~/opt/orb3_refid). 키프레임마다 ProcessNewKeyFrame · MapPointCulling ·
// CreateNewMapPoints · SearchInNeighbors · LocalBundleAdjustment · KeyFrameCulling 뒤에 지금 지도 요약 한 줄(steps.txt).
// 요약 식은 Rust `orb3sys::mapping::step_digest` 와 같음. 재생은 examples/orb3_sys_run 과 같은 장면 고르기 · 시각 · 장면마다 WaitMappingIdle.
// argv: voc yaml dir max_frames out.txt
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
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
#include <ORBmatcher.h>
#include <LoopClosing.h>
#undef private
#undef protected
using namespace ORB_SLAM3;

static FILE *g = nullptr;
static void *L() { static void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD); return h; }
struct H { uint64_t h = 1469598103934665603ull; void b(const void *p, size_t n) { auto *c = (const uint8_t *)p; for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; } }
  void u(uint64_t v) { b(&v, 8); } void i(int32_t v) { b(&v, 4); } void f(float v) { b(&v, 4); } };
static void digestK(KeyFrame *K, const char *step);
static void digest(LocalMapping *lm, const char *step) { digestK(lm->mpCurrentKeyFrame, step); }
static void digestK(KeyFrame *K, const char *step) {
    if (!g) return;
    Map *M = K->GetMap();
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
// 옮겨 붙이기: 들어온 S(DoPendingGraft 가 YawOnly 한 번 한 값) 비트 · 끝난 뒤 합친 지도 요약(첫 키프레임 이름으로)
static void pose(const char *w, unsigned long id, const Sophus::SE3f &p, long n);
static unsigned hx(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
void Tracking::GraftMap(Map *pFrom, Map *pInto, const Sophus::SE3f &S0) {
    typedef void (*F)(Tracking *, Map *, Map *, const Sophus::SE3f &);
    static F o = (F)dlsym(L(), "_ZN9ORB_SLAM38Tracking8GraftMapEPNS_3MapES2_RKN6Sophus3SE3IfLi0EEE");
    if (g) { auto q = S0.unit_quaternion(); auto t = S0.translation();
      fprintf(g, "G %lu %lu %08x %08x %08x %08x %08x %08x %08x\n", pFrom->GetId(), pInto->GetId(), hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z())); }
    KeyFrame *kf1 = nullptr; for (KeyFrame *k : pFrom->mspKeyFrames) if (!kf1 || k->mnId < kf1->mnId) kf1 = k;
    MapPoint *mp1 = nullptr; for (MapPoint *p : pFrom->mspMapPoints) if (!mp1 || p->mnId < mp1->mnId) mp1 = p;
    auto pk = [&](const char *w) { if (!g) return; Sophus::SE3f p = kf1->GetPose(); auto q = p.unit_quaternion(); auto t = p.translation(); Eigen::Vector3f x = mp1->GetWorldPos(), nv = mp1->GetNormal();
      fprintf(g, "%s K%lu %08x %08x %08x %08x %08x %08x %08x P%lu %08x %08x %08x N %08x %08x %08x\n", w, kf1->mnId, hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z()), mp1->mnId, hx(x[0]), hx(x[1]), hx(x[2]), hx(nv[0]), hx(nv[1]), hx(nv[2])); };
    if (kf1 && mp1) pk("GB");
    o(this, pFrom, pInto, S0);
    if (kf1 && mp1) pk("GA");
    KeyFrame *k0 = nullptr; for (KeyFrame *k : pInto->mspKeyFrames) if (!k->mbBad && (!k0 || k->mnId < k0->mnId)) k0 = k;
    if (k0) digestK(k0, "graft");
    { KeyFrame *km = nullptr; for (KeyFrame *k : pInto->mspKeyFrames) if (!km || k->mnId > km->mnId) km = k;
      for (KeyFrame *k : pFrom->mspKeyFrames) if (!km || k->mnId > km->mnId) km = k;
      if (g && km) pose("GK", km->mnId, km->GetPose(), (long)km->GetMap()->GetId()); }
    if (g) { Sophus::SE3f p = mLastFrame.GetPose(); auto q = p.unit_quaternion(); auto t = p.translation();
      fprintf(g, "GL %08x %08x %08x %08x %08x %08x %08x\n", hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z())); }
}
// 추적 단계 자세(ORBSYS_TRK=시작-끝 장면 번호): UpdateLastFrame 뒤 마지막 장면 · TrackLocalMap 앞뒤 지금 장면 · 재위치 결과
static bool trk(unsigned long id) { static long a = -1, b = -1; if (a < 0) { const char *e = getenv("ORBSYS_TRK"); if (!e || sscanf(e, "%ld-%ld", &a, &b) != 2) { a = 1; b = 0; } } return g && (long)id >= a && (long)id <= b; }
static void pose(const char *w, unsigned long id, const Sophus::SE3f &p, long n) { auto q = p.unit_quaternion(); auto t = p.translation();
    fprintf(g, "T %lu %s %08x %08x %08x %08x %08x %08x %08x %ld\n", id, w, hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z()), n); }
static long nmp(Frame &F) { long n = 0; for (int i = 0; i < F.N; i++) if (F.mvpMapPoints[i] && !F.mvbOutlier[i]) n++; return n; }
void Tracking::UpdateLastFrame() {
    typedef void (*F)(Tracking *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM38Tracking15UpdateLastFrameEv");
    if (trk(mCurrentFrame.mnId) && mLastFrame.mpReferenceKF) { pose("tlr", mCurrentFrame.mnId, mlRelativeFramePoses.back(), (long)mLastFrame.mpReferenceKF->mnId); pose("ref", mCurrentFrame.mnId, mLastFrame.mpReferenceKF->GetPose(), (long)mLastFrame.mpReferenceKF->GetMap()->GetId()); }
    if (trk(mCurrentFrame.mnId)) { uint64_t h = 1469598103934665603ull; for (int i = 0; i < mLastFrame.N; i++) { MapPoint *p = mLastFrame.mvpMapPoints[i]; uint64_t v = (p ? p->mnId : ~0ull) * 2 + (mLastFrame.mvbOutlier[i] ? 1 : 0); h ^= v; h *= 1099511628211ull; }
      pose("vel", mCurrentFrame.mnId, mVelocity, (long)(h & 0x7fffffff)); }
    o(this);
    if (trk(mCurrentFrame.mnId)) pose("ulf", mCurrentFrame.mnId, mLastFrame.GetPose(), nmp(mLastFrame));
}
bool Tracking::TrackLocalMap() {
    typedef bool (*F)(Tracking *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM38Tracking13TrackLocalMapEv");
    if (trk(mCurrentFrame.mnId)) pose("tlm0", mCurrentFrame.mnId, mCurrentFrame.GetPose(), nmp(mCurrentFrame));
    bool r = o(this);
    if (trk(mCurrentFrame.mnId)) pose("tlm1", mCurrentFrame.mnId, mCurrentFrame.GetPose(), r ? mnMatchesInliers : -1);
    return r;
}
bool Tracking::Relocalization(Map *pMap) {
    typedef bool (*F)(Tracking *, Map *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM38Tracking14RelocalizationEPNS_3MapE");
    bool r = o(this, pMap);
    if (trk(mCurrentFrame.mnId)) pose(r ? "rel1" : "rel0", mCurrentFrame.mnId, mCurrentFrame.GetPose(), pMap ? (long)pMap->GetId() : -1);
    return r;
}
// 자세 최적화 입력 · 출력(ORBSYS_PO=파일, ORBSYS_PO_RANGE=시작-끝 장면): Rust pose_opt 를 같은 입력으로 돌려 비트 비교
static FILE *gpo() { static FILE *f = nullptr; static bool t = false; if (!t) { t = true; const char *e = getenv("ORBSYS_PO"); if (e) f = fopen(e, "w"); } return f; }
static bool porange(unsigned long id) { static long a = -2, b = -1; if (a == -2) { const char *e = getenv("ORBSYS_PO_RANGE"); if (!e || sscanf(e, "%ld-%ld", &a, &b) != 2) { a = 0; b = 1L << 40; } } return (long)id >= a && (long)id <= b; }
int Optimizer::PoseOptimization(Frame *F) {
    typedef int (*Fn)(Frame *); static Fn o = (Fn)dlsym(L(), "_ZN9ORB_SLAM39Optimizer16PoseOptimizationEPNS_5FrameE");
    FILE *f = gpo();
    if (!f || !porange(F->mnId)) return o(F);
    Sophus::SE3f p = F->GetPose(); auto q = p.unit_quaternion(); auto t = p.translation();
    const std::vector<float> &cp = F->mpCamera->mvParameters;
    int n = 0; for (int i = 0; i < F->N; i++) if (F->mvpMapPoints[i]) n++;
    fprintf(f, "PO %lu %d %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", F->mnId, n, hx(cp[0]), hx(cp[1]), hx(cp[2]), hx(cp[3]),
            hx(F->fx), hx(F->fy), hx(F->cx), hx(F->cy), hx(F->mbf), hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z()));
    for (int i = 0; i < F->N; i++) { MapPoint *m = F->mvpMapPoints[i]; if (!m) continue; const cv::KeyPoint &k = F->mvKeysUn[i]; Eigen::Vector3f x = m->GetWorldPos();
        fprintf(f, "O %08x %08x %08x %08x %08x %08x %08x %lu %d\n", hx(k.pt.x), hx(k.pt.y), hx(F->mvuRight[i]), hx(F->mvInvLevelSigma2[k.octave]), hx(x[0]), hx(x[1]), hx(x[2]), m->mnId, i); }
    int r = o(F);
    p = F->GetPose(); q = p.unit_quaternion(); t = p.translation();
    fprintf(f, "R %d %08x %08x %08x %08x %08x %08x %08x ", r, hx(q.x()), hx(q.y()), hx(q.z()), hx(q.w()), hx(t.x()), hx(t.y()), hx(t.z()));
    for (int i = 0; i < F->N; i++) if (F->mvpMapPoints[i]) fputc(F->mvbOutlier[i] ? '1' : '0', f);
    fputc('\n', f);
    return r;
}
// 움직임 모형 짝짓기에서 점 하나 따라가기(ORBSYS_SBP=장면:점번호): 투영 · 반경 · 후보 거리, 끝난 뒤 그 점이 붙은 칸
int ORBmatcher::SearchByProjection(Frame &C, const Frame &Lf, const float th, const bool bMono) {
    typedef int (*Fn)(ORBmatcher *, Frame &, const Frame &, float, bool); static Fn o = (Fn)dlsym(L(), "_ZN9ORB_SLAM310ORBmatcher18SearchByProjectionERNS_5FrameERKS1_fb");
    unsigned long fa = 0, mpw = 0; const char *e = getenv("ORBSYS_SBP");
    bool on = e && sscanf(e, "%lu:%lu", &fa, &mpw) == 2 && C.mnId == fa;
    if (on) for (int i = 0; i < Lf.N; i++) { MapPoint *m = Lf.mvpMapPoints[i]; if (!m || m->mnId != mpw) continue;
        Eigen::Vector3f xc = C.GetPose() * m->GetWorldPos(); float invzc = 1.0 / xc(2); Eigen::Vector2f uv = C.mpCamera->project(xc);
        int oc = Lf.mvKeys[i].octave; float r = th * C.mvScaleFactors[oc];
        { Eigen::Vector3f w = m->GetWorldPos(); fprintf(stderr, "SBP i%d out%d bad%d nobs%d w %08x %08x %08x xc %08x %08x %08x\n", i, (int)Lf.mvbOutlier[i], (int)m->isBad(), m->Observations(), hx(w[0]), hx(w[1]), hx(w[2]), hx(xc[0]), hx(xc[1]), hx(xc[2])); }
        for (size_t i2 : C.GetFeaturesInArea(uv(0), uv(1), r, oc - 1, oc + 1)) { float ur = uv(0) - C.mbf * invzc;
            fprintf(stderr, "  c %zu d %d ur %08x er %08x cur %ld\n", i2, DescriptorDistance(m->GetDescriptor(), C.mDescriptors.row(i2)), hx(C.mvuRight[i2]), hx(fabs(ur - C.mvuRight[i2])), C.mvpMapPoints[i2] ? (long)C.mvpMapPoints[i2]->mnId : -1L); } }
    int n = o(this, C, Lf, th, bMono);
    if (on) { fprintf(stderr, "SBP n %d th %g:", n, th); for (int i = 0; i < C.N; i++) if (C.mvpMapPoints[i] && C.mvpMapPoints[i]->mnId == mpw) fprintf(stderr, " at %d", i); fprintf(stderr, "\n"); }
    return n;
}
// 루프 스레드: 공통 영역 찾기 결과(L 키프레임 루프 합치기 루프짝 합치기짝) · 고치기 · 합치기 뒤 요약
bool LoopClosing::NewDetectCommonRegions() {
    typedef bool (*F)(LoopClosing *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing22NewDetectCommonRegionsEv");
    bool r = o(this);
    if (g) fprintf(g, "L %lu %d %d %ld %ld\n", mpCurrentKF->mnId, (int)mbLoopDetected, (int)mbMergeDetected, mpLoopMatchedKF ? (long)mpLoopMatchedKF->mnId : -1L, mpMergeMatchedKF ? (long)mpMergeMatchedKF->mnId : -1L);
    return r;
}
void LoopClosing::CorrectLoop() {
    typedef void (*F)(LoopClosing *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing11CorrectLoopEv");
    digestK(mpCurrentKF, "corr0");
    if (g) { auto h = [](std::vector<MapPoint *> &v) { uint64_t x = 1469598103934665603ull; for (MapPoint *p : v) { x ^= p ? p->mnId : ~0ull; x *= 1099511628211ull; } return x; };
      fprintf(g, "C %lu %lu %zu %zu %016lx %016lx\n", mpCurrentKF->mnId, mpLoopMatchedKF->mnId, mvpLoopMapPoints.size(), mvpLoopMatchedMPs.size(), h(mvpLoopMapPoints), h(mvpLoopMatchedMPs));
      auto d = [](double v) { uint64_t u; memcpy(&u, &v, 8); return (unsigned long)u; }; auto &S = mg2oLoopScw; auto q = S.rotation().coeffs(); auto t = S.translation();
      fprintf(g, "CS %016lx %016lx %016lx %016lx %016lx %016lx %016lx %016lx\n", d(q[0]), d(q[1]), d(q[2]), d(q[3]), d(t[0]), d(t[1]), d(t[2]), d(S.scale()));
      fprintf(g, "CM"); for (MapPoint *p : mvpLoopMatchedMPs) fprintf(g, " %ld", p ? (long)p->mnId : -1L); fprintf(g, "\n"); }
    o(this); digestK(mpCurrentKF, "corr");
}
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, std::vector<MapPoint *> &vpMapPoints) {
    typedef void (*F)(LoopClosing *, const KeyFrameAndPose &, std::vector<MapPoint *> &);
    static F o = (F)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing13SearchAndFuseERKSt3mapIPNS_8KeyFrameEN3g2o4Sim3ESt4lessIS3_EN5Eigen17aligned_allocatorISt4pairIKS3_S5_EEEERSt6vectorIPNS_8MapPointESaISJ_EE");
    o(this, CorrectedPosesMap, vpMapPoints); digestK(mpCurrentKF, "fuse");
}
void LoopClosing::MergeLocal() {
    typedef void (*F)(LoopClosing *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM311LoopClosing10MergeLocalEv");
    o(this); digestK(mpCurrentKF, "merge");
}
// 새 키프레임 넘기기를 장면 끝으로(Rust RustSystem 과 같은 순서): 원본은 CreateNewKeyFrame 에서 바로 넘겨, 같은 장면 끝의 TryGraft
// 재위치(rand())와 지도 · 루프 스레드(Sim3Solver rand())가 겹쳐 옮겨 붙이기 S 가 실행마다 달라짐
static bool g_inTrack = false;
static std::vector<std::pair<LocalMapping *, KeyFrame *>> g_defer;
void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
    typedef void (*F)(LocalMapping *, KeyFrame *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping14InsertKeyFrameEPNS_8KeyFrameE");
    if (g_inTrack && pKF->mnId != pKF->GetMap()->GetInitKFid()) { g_defer.push_back({this, pKF}); return; }
    o(this, pKF);
}
static void flushDefer() {
    typedef void (*F)(LocalMapping *, KeyFrame *); static F o = (F)dlsym(L(), "_ZN9ORB_SLAM312LocalMapping14InsertKeyFrameEPNS_8KeyFrameE");
    for (auto &e : g_defer) o(e.first, e.second);
    g_defer.clear();
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
// 패치 WaitMappingIdle 은 2 초 넘으면 · 지도 스레드가 멈춰 있으면 그냥 돌아가 다음 장면 추적이 지도 · 루프 스레드와 겹칠 수 있음
// (재위치 · Sim3 RANSAC 이 rand() 하나를 나눠 써 결과가 실행마다 달라짐) → 시간 제한 없이 둘 다 한가할 때까지
static void waitMap(System &S) {
    LocalMapping *lm = S.mpLocalMapper;
    while (!lm->isFinished() && !(lm->KeyframesInQueue() == 0 && lm->AcceptKeyFrames() && !lm->isStopped() && !lm->stopRequested())) usleep(200);
}
static void waitAll(System &S) {
    for (int k = 0; k < 3; k++) { waitMap(S); waitLoop(S); }
    waitMap(S);
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
    FILE *gt = fopen((std::string(argv[5]) + ".tsv").c_str(), "w");
    size_t j = 0; int n = 0;
    for (size_t i = 0; i < rgb.size() && n < maxf; i++) {
        double t = rgb[i].first;
        while (j + 1 < dep.size() && std::fabs(dep[j + 1].first - t) < std::fabs(dep[j].first - t)) j++;
        if (std::fabs(dep[j].first - t) > 0.02) continue;
        cv::Mat C = cv::imread(dir + "/" + rgb[i].second, cv::IMREAD_UNCHANGED), D = cv::imread(dir + "/" + dep[j].second, cv::IMREAD_UNCHANGED);
        g_inTrack = true;
        Sophus::SE3f tcw = S.TrackRGBD(C, D, t);
        g_inTrack = false; flushDefer();
        if (gt) { Eigen::Matrix<float, 3, 4> m = tcw.inverse().matrix3x4(); fprintf(gt, "%.6f\t%d\t%lu", t, S.GetTrackingState(), S.mpAtlas->GetCurrentMap()->GetId());
          for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) fprintf(gt, "\t%.17g", (double)m(r, c)); fprintf(gt, "\n"); }
        waitAll(S); n++;
        fflush(g);
    }
    S.Shutdown(); fclose(gt); fclose(g); g = nullptr; fprintf(stderr, "frames %d\n", n); return 0;
}
