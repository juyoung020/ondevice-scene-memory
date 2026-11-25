// ORB-SLAM3 원본을 RGB-D 로 돌리며 Tracking · LocalMapping 의 판단 함수들 입력 · 출력을 덤프한다(포터 TRACK 비트 비교).
// libORB_SLAM3.so 안의 호출은 PLT 를 거치므로 이 실행 파일의 같은 이름 정의가 가로채고, 원본은 dlsym 으로 부른다.
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <any>
#include <variant>
#include <optional>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <deque>
#include <condition_variable>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#define private public
#define protected public
#include <System.h>
#include <Tracking.h>
#include <LocalMapping.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Atlas.h>
#include <ORBmatcher.h>
#include <GeometricTools.h>
#undef private
#undef protected

using namespace ORB_SLAM3;
static FILE *g_dump = nullptr;
static std::mutex g_mx;
static long g_frame = 0;
static int g_ulmp_every = 4;
static int g_sync = 0;
static std::map<std::string, long> g_cnt;

struct Buf {
    std::vector<uint8_t> b;
    template <class T> void w(const T &v) { const uint8_t *p = (const uint8_t *)&v; b.insert(b.end(), p, p + sizeof(T)); }
    void flush(const char *tag) {
        std::lock_guard<std::mutex> l(g_mx);
        if (!g_dump) return;
        uint32_t t; memcpy(&t, tag, 4);
        uint32_t n = (uint32_t)b.size();
        fwrite(&t, 4, 1, g_dump); fwrite(&n, 4, 1, g_dump); fwrite(b.data(), 1, b.size(), g_dump);
        g_cnt[tag]++;
    }
};

static void *orig_sym(const char *name) {
    void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD);
    void *f = dlsym(h, name);
    if (!f) { fprintf(stderr, "dlsym fail %s\n", name); abort(); }
    return f;
}
static const uint64_t NONE = ~0ull;
struct Ev { int32_t code; int64_t a; int32_t r; };
static thread_local std::vector<Ev> *t_ev = nullptr;
static thread_local int t_quiet = 0;
static void ev(int code, int64_t a, int r) { if (t_ev && !t_quiet) t_ev->push_back({code, a, r}); }
static uint64_t kid(KeyFrame *k) { return k ? (uint64_t)k->mnId : NONE; }

// 키프레임 표 한 줄: id, 주소(원본 std::set/map 순서 열쇠), 나쁨, 표시, 이웃 10, 자식(집합 순), 부모
static void put_kf(Buf &B, KeyFrame *k, int nbest) {
    B.w<uint64_t>(k->mnId); B.w<uint64_t>((uint64_t)(uintptr_t)k); B.w<uint8_t>(k->isBad());
    B.w<uint64_t>(k->mnTrackReferenceForFrame);
    std::vector<KeyFrame *> best = k->GetBestCovisibilityKeyFrames(nbest);
    B.w<uint32_t>(best.size());
    for (KeyFrame *b : best) { B.w<uint64_t>(b->mnId); B.w<uint8_t>(b->isBad()); B.w<uint64_t>(b->mnTrackReferenceForFrame); }
    std::set<KeyFrame *> ch = k->GetChilds();
    B.w<uint32_t>(ch.size());
    for (KeyFrame *c : ch) { B.w<uint64_t>(c->mnId); B.w<uint8_t>(c->isBad()); B.w<uint64_t>(c->mnTrackReferenceForFrame); B.w<uint64_t>((uint64_t)(uintptr_t)c); }
    KeyFrame *p = k->GetParent();
    B.w<uint64_t>(kid(p));
    if (p) { B.w<uint8_t>(p->isBad()); B.w<uint64_t>(p->mnTrackReferenceForFrame); }
}

// ---------------- Tracking::NeedNewKeyFrame ----------------
static uint64_t nnkf_state(Tracking *t) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    LocalMapping *L = t->mpLocalMapper;
    mix(L->isStopped()); mix(L->stopRequested()); mix(L->AcceptKeyFrames()); mix(L->IsInitializing()); mix(L->KeyframesInQueue());
    mix(t->mpAtlas->KeyFramesInMap());
    for (auto *m : t->mpReferenceKF->GetMapPointMatches()) if (m) { mix(m->isBad()); mix(m->Observations()); }
    return h;
}
bool Tracking::NeedNewKeyFrame() {
    typedef bool (*Fn)(Tracking *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking15NeedNewKeyFrameEv");
    const uint64_t h0 = nnkf_state(this);  // 덤프 읽기 앞: 읽기 · 원본 호출 사이 어디서 바뀌어도 잡힘
    Buf B;
    B.w<uint64_t>(mCurrentFrame.mnId);
    B.w<int32_t>(mSensor); B.w<int32_t>(mState);
    B.w<uint8_t>(mbOnlyTracking);
    B.w<uint8_t>(mpLocalMapper->isStopped()); B.w<uint8_t>(mpLocalMapper->stopRequested());
    B.w<int32_t>((int32_t)mpAtlas->KeyFramesInMap());
    B.w<uint64_t>(mnLastRelocFrameId); B.w<int32_t>(mMaxFrames); B.w<int32_t>(mMinFrames); B.w<uint64_t>(mnLastKeyFrameId);
    {
        KeyFrame *r = mpReferenceKF;
        std::vector<MapPoint *> v = r->GetMapPointMatches();
        uint32_t n = 0; for (auto *m : v) if (m) n++;
        B.w<uint32_t>(n);
        for (auto *m : v) if (m) { B.w<uint8_t>(m->isBad()); B.w<int32_t>(m->Observations()); }
    }
    B.w<uint8_t>(mpLocalMapper->AcceptKeyFrames()); B.w<uint8_t>(mpLocalMapper->IsInitializing());
    B.w<int32_t>(mpLocalMapper->KeyframesInQueue());
    const int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
    B.w<uint32_t>(N);
    for (int i = 0; i < N; i++) { B.w<float>(mCurrentFrame.mvDepth[i]); B.w<uint8_t>((mCurrentFrame.mvpMapPoints[i] ? 1 : 0) | (mCurrentFrame.mvbOutlier[i] ? 2 : 0)); }
    B.w<float>(mThDepth); B.w<int32_t>(mnMatchesInliers);
    B.w<uint8_t>(mpCamera2 != nullptr);
    B.w<double>(mCurrentFrame.mTimeStamp); B.w<double>(mpLastKeyFrame ? mpLastKeyFrame->mTimeStamp : -1.0); B.w<uint8_t>(mpLastKeyFrame != nullptr);
    bool r = orig(this);
    // 원본이 부르는 동안 다른 스레드가 입력(지역 지도 스레드 상태 · 기준 키프레임 관측 수)을 바꿨나
    B.w<uint8_t>(r); B.w<uint8_t>(nnkf_state(this) != h0);
    B.flush("NNKF");
    ev(8, 0, r);
    return r;
}

// ---------------- Tracking::UpdateLocalKeyFrames ----------------
void Tracking::UpdateLocalKeyFrames() {
    typedef void (*Fn)(Tracking *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking20UpdateLocalKeyFramesEv");
    Buf B;
    const bool imu = mpAtlas->isImuInitialized();
    B.w<uint64_t>(mCurrentFrame.mnId); B.w<uint64_t>(mnLastRelocFrameId); B.w<uint8_t>(imu); B.w<int32_t>(mSensor);
    const bool useCur = !imu || (mCurrentFrame.mnId < mnLastRelocFrameId + 2);
    Frame &F = useCur ? mCurrentFrame : mLastFrame;
    std::set<KeyFrame *> kfs;
    B.w<uint32_t>(F.N);
    for (int i = 0; i < F.N; i++) {
        MapPoint *m = F.mvpMapPoints[i];
        if (!m) { B.w<uint8_t>(0); continue; }
        if (m->isBad()) { B.w<uint8_t>(1); continue; }
        B.w<uint8_t>(2);
        auto obs = m->GetObservations();
        B.w<uint32_t>(obs.size());
        for (auto &o : obs) { B.w<uint64_t>(o.first->mnId); kfs.insert(o.first); }
    }
    B.w<uint32_t>(kfs.size());
    for (KeyFrame *k : kfs) put_kf(B, k, 10);
    B.w<uint64_t>(kid(mpReferenceKF));
    orig(this);
    B.w<uint32_t>(mvpLocalKeyFrames.size());
    for (KeyFrame *k : mvpLocalKeyFrames) B.w<uint64_t>(k->mnId);
    B.w<uint64_t>(kid(mpReferenceKF));
    for (int i = 0; i < F.N; i++) B.w<uint8_t>(F.mvpMapPoints[i] ? 1 : 0);
    B.flush("ULKF");
}

// ---------------- Tracking::UpdateLocalPoints ----------------
void Tracking::UpdateLocalPoints() {
    typedef void (*Fn)(Tracking *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking17UpdateLocalPointsEv");
    if (mCurrentFrame.mnId % g_ulmp_every != 0) { orig(this); return; }
    Buf B;
    B.w<uint64_t>(mCurrentFrame.mnId);
    std::unordered_map<MapPoint *, int> seen;
    std::vector<MapPoint *> uniq;
    B.w<uint32_t>(mvpLocalKeyFrames.size());
    for (KeyFrame *k : mvpLocalKeyFrames) {
        std::vector<MapPoint *> v = k->GetMapPointMatches();
        B.w<uint64_t>(k->mnId); B.w<uint32_t>(v.size());
        uint32_t n = 0; for (auto *m : v) if (m) n++;
        B.w<uint32_t>(n);
        for (uint32_t i = 0; i < v.size(); i++) if (v[i]) {
            B.w<uint32_t>(i); B.w<uint64_t>(v[i]->mnId);
            if (!seen.count(v[i])) { seen[v[i]] = 1; uniq.push_back(v[i]); }
        }
    }
    B.w<uint32_t>(uniq.size());
    for (MapPoint *m : uniq) { B.w<uint64_t>(m->mnId); B.w<uint8_t>(m->isBad()); B.w<uint64_t>(m->mnTrackReferenceForFrame); }
    orig(this);
    B.w<uint32_t>(mvpLocalMapPoints.size());
    for (MapPoint *m : mvpLocalMapPoints) B.w<uint64_t>(m->mnId);
    B.flush("ULMP");
}

// ---------------- Tracking::CreateNewKeyFrame ----------------
void Tracking::CreateNewKeyFrame() {
    typedef void (*Fn)(Tracking *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking17CreateNewKeyFrameEv");
    Buf B;
    Frame &F = mCurrentFrame;
    B.w<uint64_t>(F.mnId); B.w<int32_t>(mSensor);
    const Sophus::SE3f T = F.GetPose();
    const Eigen::Quaternionf q = T.unit_quaternion();
    B.w(q.x()); B.w(q.y()); B.w(q.z()); B.w(q.w());
    B.w(T.translation()[0]); B.w(T.translation()[1]); B.w(T.translation()[2]);
    B.w(F.cx); B.w(F.cy); B.w(F.invfx); B.w(F.invfy); B.w(mThDepth);
    B.w<int32_t>(F.Nleft); B.w<uint32_t>(F.N);
    std::vector<MapPoint *> before = F.mvpMapPoints;
    for (int i = 0; i < F.N; i++) {
        B.w(F.mvKeysUn[i].pt.x); B.w(F.mvKeysUn[i].pt.y); B.w(F.mvDepth[i]);
        MapPoint *m = F.mvpMapPoints[i];
        B.w<int32_t>(m ? m->Observations() : -1);
    }
    const unsigned long lastKF = mnLastKeyFrameId; KeyFrame *lastKFp = mpLastKeyFrame;
    orig(this);
    const bool made = mpLastKeyFrame != lastKFp || mnLastKeyFrameId != lastKF;
    B.w<uint8_t>(made);
    // 원본 뒤 자세 행렬(UpdatePoseMatrices)
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) B.w(F.mRwc(r, c));
    B.w(F.mOw[0]); B.w(F.mOw[1]); B.w(F.mOw[2]);
    std::vector<std::pair<int, Eigen::Vector3f>> created; std::vector<int> cleared;
    for (int i = 0; i < F.N; i++) {
        if (F.mvpMapPoints[i] != before[i]) {
            if (F.mvpMapPoints[i]) created.push_back({i, F.mvpMapPoints[i]->GetWorldPos()});
            else cleared.push_back(i);
        }
    }
    B.w<uint32_t>(created.size());
    for (auto &c : created) { B.w<int32_t>(c.first); B.w(c.second[0]); B.w(c.second[1]); B.w(c.second[2]); }
    B.w<uint32_t>(cleared.size());
    for (int c : cleared) B.w<int32_t>(c);
    B.flush("CNKF");
    ev(9, 0, made);
}

static thread_local std::vector<uint64_t> *t_mpbad = nullptr;
void MapPoint::SetBadFlag() {
    typedef void (*Fn)(MapPoint *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38MapPoint10SetBadFlagEv");
    if (t_mpbad) t_mpbad->push_back(mnId);
    orig(this);
}

// ---------------- LocalMapping::CreateNewMapPoints (+ 짝 찾기 · 새 키프레임 확인 결과를 같이) ----------------
static thread_local std::vector<std::pair<uint64_t, std::vector<std::pair<size_t, size_t>>>> *t_tri = nullptr;
static thread_local std::vector<uint8_t> *t_chk = nullptr;
static long g_tria = 0;
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<std::pair<size_t, size_t>> &v, const bool bOnlyStereo, const bool bCoarse) {
    typedef int (*Fn)(ORBmatcher *, KeyFrame *, KeyFrame *, std::vector<std::pair<size_t, size_t>> &, bool, bool);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM310ORBmatcher22SearchForTriangulationEPNS_8KeyFrameES2_RSt6vectorISt4pairImmESaIS5_EEbb");
    int r = orig(this, pKF1, pKF2, v, bOnlyStereo, bCoarse);
    if (t_tri) t_tri->push_back({pKF2->mnId, v});
    return r;
}
bool LocalMapping::CheckNewKeyFrames() {
    typedef bool (*Fn)(LocalMapping *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM312LocalMapping17CheckNewKeyFramesEv");
    bool r = orig(this);
    if (t_chk) t_chk->push_back(r);
    return r;
}
bool GeometricTools::Triangulate(Eigen::Vector3f &x1, Eigen::Vector3f &x2, Eigen::Matrix<float, 3, 4> &T1, Eigen::Matrix<float, 3, 4> &T2, Eigen::Vector3f &x3D) {
    typedef bool (*Fn)(Eigen::Vector3f &, Eigen::Vector3f &, Eigen::Matrix<float, 3, 4> &, Eigen::Matrix<float, 3, 4> &, Eigen::Vector3f &);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM314GeometricTools11TriangulateERN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEES4_RNS2_IfLi3ELi4ELi0ELi3ELi4EEES6_S4_");
    bool r = orig(x1, x2, T1, T2, x3D);
    if (g_tria < 200000) {
        g_tria++;
        Buf B;
        for (int k = 0; k < 3; k++) B.w(x1[k]);
        for (int k = 0; k < 3; k++) B.w(x2[k]);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) B.w(T1(i, j));
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) B.w(T2(i, j));
        B.w<uint8_t>(r);
        for (int k = 0; k < 3; k++) B.w(x3D[k]);
        B.flush("TRIA");
    }
    return r;
}
static void put_kfdata(Buf &B, KeyFrame *k) {
    B.w<uint64_t>(k->mnId);
    const Sophus::SE3f T = k->GetPose();
    const Eigen::Quaternionf q = T.unit_quaternion();
    B.w(q.x()); B.w(q.y()); B.w(q.z()); B.w(q.w());
    B.w(T.translation()[0]); B.w(T.translation()[1]); B.w(T.translation()[2]);
    { std::unique_lock<std::mutex> lp(k->mMutexPose); for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) B.w(k->mRwc(i, j)); for (int i = 0; i < 3; i++) B.w(k->mTwc.translation()[i]); }
    B.w<int32_t>(k->NLeft); B.w<uint8_t>(k->mpCamera2 != nullptr);
    B.w<uint32_t>(k->N);
    for (int i = 0; i < k->N; i++) {
        B.w(k->mvKeysUn[i].pt.x); B.w(k->mvKeysUn[i].pt.y); B.w<int32_t>(k->mvKeysUn[i].octave);
        B.w(k->mvKeys[i].pt.x); B.w(k->mvKeys[i].pt.y);
        B.w(k->mvuRight[i]); B.w(k->mvDepth[i]);
    }
}
void LocalMapping::CreateNewMapPoints() {
    typedef void (*Fn)(LocalMapping *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM312LocalMapping18CreateNewMapPointsEv");
    Buf B;
    KeyFrame *K = mpCurrentKeyFrame;
    // 고리 닫기 · 전체 BA 가 도중에 자세 · 점을 고치지 못하게(덤프 값과 원본 계산이 같은 순간이 되게)
    std::unique_lock<std::mutex> lkMap(K->GetMap()->mMutexMapUpdate);
    B.w<uint8_t>(mbMonocular); B.w<uint8_t>(mbInertial); B.w<uint8_t>(mbFarPoints); B.w(mThFarPoints);
    B.w(K->fx); B.w(K->fy); B.w(K->cx); B.w(K->cy); B.w(K->invfx); B.w(K->invfy); B.w(K->mb); B.w(K->mbf); B.w(K->mfScaleFactor);
    for (int k = 0; k < 4; k++) B.w(K->mpCamera->getParameter(k));
    B.w<uint32_t>(K->mvScaleFactors.size());
    for (size_t l = 0; l < K->mvScaleFactors.size(); l++) { B.w(K->mvScaleFactors[l]); B.w(K->mvLevelSigma2[l]); }
    put_kfdata(B, K);
    std::vector<KeyFrame *> nb = K->GetBestCovisibilityKeyFrames(mbMonocular ? 30 : 10);
    B.w<uint32_t>(nb.size());
    for (KeyFrame *k : nb) { put_kfdata(B, k); B.w(k->mb); B.w(k->fx); B.w(k->fy); B.w(k->cx); B.w(k->cy); B.w(k->invfx); B.w(k->invfy); }
    const size_t n0 = mlpRecentAddedMapPoints.size();
    std::vector<std::pair<uint64_t, std::vector<std::pair<size_t, size_t>>>> tri; std::vector<uint8_t> chk;
    t_tri = &tri; t_chk = &chk;
    orig(this);
    t_tri = nullptr; t_chk = nullptr;
    B.w<uint32_t>(tri.size());
    for (auto &t : tri) { B.w<uint64_t>(t.first); B.w<uint32_t>(t.second.size()); for (auto &pr : t.second) { B.w<uint32_t>(pr.first); B.w<uint32_t>(pr.second); } }
    B.w<uint32_t>(chk.size());
    for (uint8_t c : chk) B.w(c);
    std::vector<MapPoint *> added;
    { auto it = mlpRecentAddedMapPoints.begin(); std::advance(it, std::min(n0, mlpRecentAddedMapPoints.size())); for (; it != mlpRecentAddedMapPoints.end(); ++it) added.push_back(*it); }
    B.w<uint32_t>(added.size());
    for (MapPoint *m : added) {
        Eigen::Vector3f X = m->GetWorldPos();
        B.w(X[0]); B.w(X[1]); B.w(X[2]);
        auto obs = m->GetObservations();
        int i1 = -1, i2 = -1; uint64_t k2 = NONE;
        for (auto &o : obs) { if (o.first == K) i1 = std::get<0>(o.second); else { k2 = o.first->mnId; i2 = std::get<0>(o.second); } }
        B.w<int32_t>(i1); B.w<uint64_t>(k2); B.w<int32_t>(i2);
    }
    B.flush("CNMP");
}


// ---------------- MapPoint::UpdateNormalAndDepth (UNDP) ----------------
// 입력: 위치, 기준 키프레임, 관측(주소 순: id · 주소 · 왼쪽 번호 · 카메라 중심 · 그 번호 octave), 크기 표 / 결과: 법선 · 최소 · 최대 거리
void MapPoint::UpdateNormalAndDepth() {
    typedef void (*Fn)(MapPoint *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38MapPoint20UpdateNormalAndDepthEv");
    Buf B;
    bool bad; std::map<KeyFrame *, std::tuple<int, int>> obs; KeyFrame *ref; Eigen::Vector3f pos;
    { std::unique_lock<std::mutex> l1(mMutexFeatures); std::unique_lock<std::mutex> l2(mMutexPos); bad = mbBad; obs = mObservations; ref = mpRefKF; pos = mWorldPos; }
    B.w<uint8_t>(bad); B.w(pos[0]); B.w(pos[1]); B.w(pos[2]);
    B.w<uint64_t>(kid(ref)); B.w<uint64_t>((uint64_t)(uintptr_t)ref);
    B.w<uint32_t>(obs.size());
    for (auto &o : obs) {
        KeyFrame *k = o.first; int li = std::get<0>(o.second);
        B.w<uint64_t>(k->mnId); B.w<uint64_t>((uint64_t)(uintptr_t)k); B.w<int32_t>(li);
        Eigen::Vector3f c = k->GetCameraCenter(); B.w(c[0]); B.w(c[1]); B.w(c[2]);
        B.w<int32_t>(li >= 0 ? k->mvKeysUn[li].octave : -1);
    }
    if (ref) {
        Eigen::Vector3f c = ref->GetCameraCenter(); B.w(c[0]); B.w(c[1]); B.w(c[2]);
        B.w<uint32_t>(ref->mvScaleFactors.size()); for (float f : ref->mvScaleFactors) B.w(f);
    }
    orig(this);
    { std::unique_lock<std::mutex> l2(mMutexPos); B.w(mNormalVector[0]); B.w(mNormalVector[1]); B.w(mNormalVector[2]); B.w(mfMinDistance); B.w(mfMaxDistance); }
    B.flush("UNDP");
}

// ---------------- KeyFrame::UpdateConnections (UCON) ----------------
static void put_conn(Buf &B, KeyFrame *k) {
    B.w<uint32_t>(k->mConnectedKeyFrameWeights.size());
    for (auto &c : k->mConnectedKeyFrameWeights) { B.w<uint64_t>(c.first->mnId); B.w<uint64_t>((uint64_t)(uintptr_t)c.first); B.w<int32_t>(c.second); B.w<uint8_t>(c.first->isBad()); }
    B.w<uint32_t>(k->mvpOrderedConnectedKeyFrames.size());
    for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) { B.w<uint64_t>(k->mvpOrderedConnectedKeyFrames[i]->mnId); B.w<int32_t>(k->mvOrderedWeights[i]); }
}
void KeyFrame::UpdateConnections(bool up) {
    typedef void (*Fn)(KeyFrame *, bool);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM38KeyFrame17UpdateConnectionsEb");
    Buf B;
    B.w<uint64_t>(mnId); B.w<uint8_t>(up); B.w<uint8_t>(mbFirstConnection); B.w<uint64_t>(mpMap->GetInitKFid());
    B.w<uint64_t>(kid(mpParent));
    std::set<KeyFrame *> others;
    std::vector<MapPoint *> v = GetMapPointMatches();
    uint32_t n = 0; for (auto *m : v) if (m) n++;
    B.w<uint32_t>(n);
    for (auto *m : v) if (m) {
        B.w<uint8_t>(m->isBad());
        auto ob = m->GetObservations();
        B.w<uint32_t>(ob.size());
        for (auto &o : ob) { B.w<uint64_t>(o.first->mnId); B.w<uint64_t>((uint64_t)(uintptr_t)o.first); B.w<uint8_t>(o.first->isBad()); B.w<uint8_t>(o.first->GetMap() == mpMap); others.insert(o.first); }
    }
    others.erase(this);
    B.w<uint32_t>(others.size());
    for (KeyFrame *k : others) { B.w<uint64_t>(k->mnId); B.w<uint8_t>(k->isBad()); put_conn(B, k); }
    orig(this, up);
    put_conn(B, this); B.w<uint64_t>(kid(mpParent));
    for (KeyFrame *k : others) put_conn(B, k);
    B.flush("UCON");
}

// ---------------- LocalMapping::MapPointCulling ----------------
void LocalMapping::MapPointCulling() {
    typedef void (*Fn)(LocalMapping *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM312LocalMapping15MapPointCullingEv");
    Buf B;
    B.w<uint64_t>(mpCurrentKeyFrame->mnId); B.w<uint8_t>(mbMonocular);
    std::vector<MapPoint *> v(mlpRecentAddedMapPoints.begin(), mlpRecentAddedMapPoints.end());
    B.w<uint32_t>(v.size());
    std::vector<uint8_t> bad0;
    for (MapPoint *m : v) {
        // 원본은 이 칸들을 잠금 아래 읽음; 추적 스레드가 바꾸는 중일 수 있어 나중에 "흔들림" 으로 셈
        B.w<uint64_t>(m->mnId); B.w<uint8_t>(m->isBad()); B.w<int32_t>(m->mnFound); B.w<int32_t>(m->mnVisible);
        B.w<int64_t>(m->mnFirstKFid); B.w<int32_t>(m->Observations());
        bad0.push_back(m->isBad());
    }
    std::vector<uint64_t> called; t_mpbad = &called;
    orig(this);
    t_mpbad = nullptr;
    B.w<uint32_t>(mlpRecentAddedMapPoints.size());
    for (MapPoint *m : mlpRecentAddedMapPoints) B.w<uint64_t>(m->mnId);
    for (size_t k = 0; k < v.size(); k++) B.w<uint8_t>(v[k]->isBad());
    B.w<uint32_t>(called.size());
    for (uint64_t c : called) B.w<uint64_t>(c);
    B.flush("MPCL");
}

// ---------------- LocalMapping::KeyFrameCulling ----------------
static thread_local std::vector<uint64_t> *t_kfbad = nullptr;

typedef void (*KfBadFn)(KeyFrame *);
void KeyFrame::SetBadFlag() {
    static KfBadFn orig = (KfBadFn)orig_sym("_ZN9ORB_SLAM38KeyFrame10SetBadFlagEv");
    if (t_kfbad) t_kfbad->push_back(mnId);
    orig(this);
}

void LocalMapping::KeyFrameCulling() {
    typedef void (*Fn)(LocalMapping *);
    static Fn orig = (Fn)orig_sym("_ZN9ORB_SLAM312LocalMapping15KeyFrameCullingEv");
    Buf B;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    std::vector<KeyFrame *> loc = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    B.w<uint64_t>(mpCurrentKeyFrame->mnId); B.w<uint8_t>(mbMonocular); B.w<uint8_t>(mbInertial); B.w<uint8_t>(mbAbortBA);
    std::unordered_map<MapPoint *, int> seen; std::vector<MapPoint *> uniq;
    B.w<uint32_t>(loc.size());
    for (KeyFrame *k : loc) {
        B.w<uint64_t>(k->mnId); B.w<uint8_t>(k->mnId == k->GetMap()->GetInitKFid()); B.w<uint8_t>(k->isBad());
        B.w<uint8_t>(k->mbNotErase); B.w<float>(k->mThDepth); B.w<int32_t>(k->NLeft);
        std::vector<MapPoint *> v = k->GetMapPointMatches();
        uint32_t n = 0; for (auto *m : v) if (m) n++;
        B.w<uint32_t>(v.size()); B.w<uint32_t>(n);
        for (uint32_t i = 0; i < v.size(); i++) if (v[i]) {
            B.w<uint32_t>(i); B.w<uint64_t>(v[i]->mnId); B.w<float>(k->mvDepth[i]); B.w<int32_t>(k->mvKeysUn[i].octave); B.w<float>(k->mvuRight[i]);
            if (!seen.count(v[i])) { seen[v[i]] = 1; uniq.push_back(v[i]); }
        }
    }
    B.w<uint32_t>(uniq.size());
    for (MapPoint *m : uniq) {
        B.w<uint64_t>(m->mnId); B.w<uint8_t>(m->isBad()); B.w<int32_t>(m->Observations());
        auto obs = m->GetObservations();
        B.w<uint32_t>(obs.size());
        for (auto &o : obs) {
            KeyFrame *k = o.first; int li = std::get<0>(o.second);
            B.w<uint64_t>(k->mnId); B.w<int32_t>(li); B.w<int32_t>(std::get<1>(o.second));
            B.w<int32_t>(li >= 0 ? k->mvKeysUn[li].octave : -1);
            B.w<float>(li >= 0 ? k->mvuRight[li] : -1.f);
        }
    }
    std::vector<uint64_t> called;
    t_kfbad = &called;
    orig(this);
    t_kfbad = nullptr;
    B.w<uint32_t>(called.size());
    for (uint64_t c : called) B.w<uint64_t>(c);
    B.w<uint8_t>(mbAbortBA);
    B.flush("KFCL");
}


// ---------------- Track() 흐름: 하위 호출 결과를 사건으로 ----------------
#define HOOK_BOOL(cls, name, mangled, code) \
bool cls::name() { typedef bool (*Fn)(cls *); static Fn o = (Fn)orig_sym(mangled); bool r = o(this); ev(code, 0, r); return r; }
HOOK_BOOL(Tracking, TrackReferenceKeyFrame, "_ZN9ORB_SLAM38Tracking22TrackReferenceKeyFrameEv", 1)
HOOK_BOOL(Tracking, TrackWithMotionModel, "_ZN9ORB_SLAM38Tracking20TrackWithMotionModelEv", 2)
bool Tracking::Relocalization(Map *pMap) {
    typedef bool (*Fn)(Tracking *, Map *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking14RelocalizationEPNS_3MapE");
    int64_t id = pMap ? (int64_t)pMap->GetId() : -1;
    bool r = o(this, pMap); ev(3, id, r); return r;
}
bool Tracking::TrackLocalMap() {
    typedef bool (*Fn)(Tracking *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking13TrackLocalMapEv");
    bool r = o(this); ev(4, mCurrentFrame.GetPose().matrix().allFinite() ? 1 : 0, r); return r;
}
void Tracking::StereoInitialization() {
    typedef void (*Fn)(Tracking *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking20StereoInitializationEv");
    o(this); ev(5, 0, mState);
}
void Tracking::CreateMapInAtlas() {
    typedef void (*Fn)(Tracking *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking16CreateMapInAtlasEv");
    o(this); ev(6, 0, 0);
}
void System::ResetActiveMap() {
    typedef void (*Fn)(System *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM36System14ResetActiveMapEv");
    o(this); ev(7, 0, 0);
}
void Atlas::ChangeMap(Map *pMap) {
    typedef void (*Fn)(Atlas *, Map *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM35Atlas9ChangeMapEPNS_3MapE");
    o(this, pMap); ev(13, pMap ? (int64_t)pMap->GetId() : -1, 0);
}
void Tracking::DoPendingGraft() {
    typedef void (*Fn)(Tracking *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking14DoPendingGraftEv");
    t_quiet++; o(this); t_quiet--;
}
void Tracking::Track() {
    typedef void (*Fn)(Tracking *); static Fn o = (Fn)orig_sym("_ZN9ORB_SLAM38Tracking5TrackEv");
    // 머리: 흐름이 읽는 값(장면 시작 때)
    DoPendingGraft();
    Buf B;
    B.w<uint64_t>(mCurrentFrame.mnId); B.w<double>(mCurrentFrame.mTimeStamp); B.w<double>(mLastFrame.mTimeStamp);
    B.w<int32_t>(mState); B.w<uint8_t>(mbOnlyTracking); B.w<uint8_t>(mbVelocity); B.w<uint64_t>(mnLastRelocFrameId);
    B.w<int32_t>(mSensor); B.w<double>(mTimeStampLost);
    Map *cur = mpAtlas->GetCurrentMap();
    B.w<int64_t>(cur ? (int64_t)cur->GetId() : -1); B.w<uint8_t>(cur && cur->isImuInitialized());
    std::vector<Map *> ms = mpAtlas->GetAllMaps();
    B.w<uint32_t>(ms.size());
    for (Map *m : ms) { B.w<int64_t>(m->GetId()); B.w<int32_t>(m->KeyFramesInMap()); }
    B.w<uint8_t>(mpLastKeyFrame != nullptr); B.w<uint8_t>(mLastFrame.isSet());
    std::vector<Ev> evs; t_ev = &evs;
    o(this);
    t_ev = nullptr;
    B.w<uint32_t>(evs.size());
    for (auto &e : evs) { B.w(e.code); B.w(e.a); B.w(e.r); }
    B.w<int32_t>(mState); B.w<double>(mTimeStampLost); B.w<uint8_t>(mbVelocity); B.w<uint8_t>(mpLastKeyFrame != nullptr);
    B.w<uint8_t>(mCurrentFrame.isSet());
    B.flush("TRAK");
}

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "track_dump voc yaml seqdir out.bin maxframes [ulmp_every]\n"); return 1; }
    std::string dir = argv[3];
    g_dump = fopen(argv[4], "wb");
    int maxf = atoi(argv[5]);
    if (argc > 6) g_ulmp_every = atoi(argv[6]);
    if (argc > 7) g_sync = atoi(argv[7]);
    std::vector<double> tr, td; std::vector<std::string> pr, pd;
    load(dir + "/rgb.txt", tr, pr); load(dir + "/depth.txt", td, pd);
    System S(argv[1], argv[2], System::RGBD, false);
    size_t j = 0; int n = 0;
    for (size_t i = 0; i < tr.size() && n < maxf; i++) {
        while (j + 1 < td.size() && std::abs(td[j + 1] - tr[i]) <= std::abs(td[j] - tr[i])) j++;
        if (std::abs(td[j] - tr[i]) > 0.02) continue;
        cv::Mat C = cv::imread(dir + "/" + pr[i], cv::IMREAD_UNCHANGED);
        cv::Mat D = cv::imread(dir + "/" + pd[j], cv::IMREAD_UNCHANGED);
        if (C.empty() || D.empty()) continue;
        Sophus::SE3f T = S.TrackRGBD(C, D, tr[i]);
        if (g_sync) S.mpTracker->WaitMappingIdle();
        Buf B; B.w<double>(tr[i]); B.w<int32_t>(S.GetTrackingState());
        B.w<uint64_t>(S.mpAtlas->GetCurrentMap()->GetId()); B.w<int32_t>((int32_t)S.mpAtlas->KeyFramesInMap());
        B.flush("FRAM");
        n++; g_frame++;
    }
    S.Shutdown();
    { std::lock_guard<std::mutex> l(g_mx); fclose(g_dump); g_dump = nullptr; }
    fprintf(stderr, "frames %d", n);
    for (auto &c : g_cnt) fprintf(stderr, " %s=%ld", c.first.c_str(), c.second);
    fprintf(stderr, "\n");
    _exit(0);
}
