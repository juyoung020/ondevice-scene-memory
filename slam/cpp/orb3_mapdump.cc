// 시험 도구: ORB-SLAM3 원본(기준 빌드)을 RGB-D 로 돌리며 KeyFrame · MapPoint · Map 메서드 호출 앞뒤 상태를 덤프한다
// (orb3rs::mapdata 비트 대조 입력, docs/orb3_port.md "지도 자료 메서드").
// libORB_SLAM3.so 안의 호출은 PLT 를 거치므로 이 실행 파일의 같은 이름 정의가 가로챔. 바꾸는 메서드는 모두 전역 재귀 잠금 G 아래서
// 돌아(덤프 중 다른 스레드가 그래프를 못 바꿈), 위치 · 수를 바꾸는 SetPose · SetWorldPos · IncreaseVisible/Found 도 G 를 거친다.
// 빌드: g++ -O2 -std=c++17 -I(원본)/include ... orb3_mapdump.cc -lORB_SLAM3 ...  (원본 디렉터리 기준, docs 참고)
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Core>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <any>
#include <pangolin/pangolin.h>
#define private public
#define protected public
#include <System.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Frame.h>
#undef private
#undef protected

using namespace ORB_SLAM3;
static FILE *g_out = nullptr;
static std::recursive_mutex G;
static thread_local int g_depth = 0;
static std::unordered_set<unsigned long> g_static;
static bool g_hdr = false;
enum Op : uint8_t { UC = 1, KSB, KSE, AC, EC, ACH, ECH, CHP, ALE, AME, SNE, KAM, KEMI, KEMP, KRM,
                    MAO, MEO, MSB, MRP, MCD, MUN, MPSK, MPSF, PAK, PEK, PAM, PEM, PBC, PIC, PSR, UBC };
static long g_cnt[64], g_rec[64], g_cap[64];

static void *sym(const char *n) {
    void *h = dlopen("libORB_SLAM3.so", RTLD_NOW | RTLD_NOLOAD);
    void *p = h ? dlsym(h, n) : nullptr;
    if (!p) { fprintf(stderr, "dlsym %s\n", n); abort(); }
    return p;
}
#define ORIGN(n, ...) using T_ = __VA_ARGS__; static T_ o_ = nullptr; if (!o_) o_ = (T_)sym(n)

struct Scope {
    std::unique_lock<std::recursive_mutex> l;
    bool top;
    Scope() : l(G) { top = g_depth++ == 0; }
    ~Scope() { g_depth--; }
};
// 앞쪽 절반은 모두, 그 뒤는 다섯 번에 한 번(실행 후반도 담기게)
static bool want(Op op) {
    if (!g_out) return false;
    long k = g_cnt[op]++;
    long cap = g_cap[op] ? g_cap[op] : 3000;
    if (g_rec[op] >= cap) return false;
    if (k < cap / 2 || k % 5 == 0) { g_rec[op]++; return true; }
    return false;
}

static std::vector<uint8_t> B;
template <class T> static void W(const T &v) { const uint8_t *p = (const uint8_t *)&v; B.insert(B.end(), p, p + sizeof(T)); }
static void flush() { fwrite(B.data(), 1, B.size(), g_out); fflush(g_out); B.clear(); }
static int64_t kid(KeyFrame *k) { return k ? (int64_t)k->mnId : -1; }
static int64_t mid(MapPoint *p) { return p ? (int64_t)p->mnId : -1; }
static void Wse3(const Sophus::SE3f &T) {
    auto q = T.unit_quaternion(); W(q.x()); W(q.y()); W(q.z()); W(q.w());
    W(T.translation()[0]); W(T.translation()[1]); W(T.translation()[2]);
}
// 키프레임마다 한 번: 점 층 · 오른쪽 좌표 · 기술자(안 바뀌는 칸)
static void wstatic(KeyFrame *k) {
    if (!k || g_static.count(k->mnId)) return;
    g_static.insert(k->mnId);
    std::vector<uint8_t> save; save.swap(B);
    if (!g_hdr) {
        g_hdr = true;
        W<uint32_t>(0x4448444d); W<int32_t>(k->mnScaleLevels); W(k->mfLogScaleFactor);
        for (int i = 0; i < k->mnScaleLevels; i++) W(k->mvScaleFactors[i]);
    }
    W<uint32_t>(0x5453444d); W<uint64_t>(k->mnId); W<uint64_t>((uint64_t)k); W<uint32_t>(k->N);
    for (int i = 0; i < k->N; i++) {
        W<int32_t>(k->mvKeysUn[i].octave); W(k->mvuRight[i]);
        B.insert(B.end(), k->mDescriptors.ptr<uint8_t>(i), k->mDescriptors.ptr<uint8_t>(i) + 32);
    }
    flush(); B.swap(save);
}
static void wkf(KeyFrame *k) {
    wstatic(k);
    // 가리키는 키프레임도 주소를 알게(시험이 원본 집합 순서 = 주소 순을 되살림)
    for (auto &e : k->mConnectedKeyFrameWeights) wstatic(e.first);
    for (auto *x : k->mvpOrderedConnectedKeyFrames) wstatic(x);
    wstatic(k->mpParent);
    for (auto *s : {&k->mspChildrens, &k->mspLoopEdges, &k->mspMergeEdges}) for (auto *x : *s) wstatic(x);
    W<uint64_t>(k->mnId); W<uint8_t>(k->mbBad); W<int64_t>(k->mpMap ? (int64_t)k->mpMap->mnId : -1);
    W<uint8_t>(k->mbFirstConnection); W<uint8_t>(k->mbNotErase); W<uint8_t>(k->mbToBeErased);
    Wse3(k->mTcw); Wse3(k->mTwc); Wse3(k->mTcp);
    W<uint32_t>(k->mConnectedKeyFrameWeights.size());
    for (auto &e : k->mConnectedKeyFrameWeights) { W<uint64_t>(e.first->mnId); W<int32_t>(e.second); }
    W<uint32_t>(k->mvpOrderedConnectedKeyFrames.size());
    for (size_t i = 0; i < k->mvpOrderedConnectedKeyFrames.size(); i++) { W<uint64_t>(k->mvpOrderedConnectedKeyFrames[i]->mnId); W<int32_t>(k->mvOrderedWeights[i]); }
    W<int64_t>(kid(k->mpParent));
    for (auto *s : {&k->mspChildrens, &k->mspLoopEdges, &k->mspMergeEdges}) { W<uint32_t>(s->size()); for (auto *x : *s) W<uint64_t>(x->mnId); }
}
static void wlist(KeyFrame *k) {
    wstatic(k);
    W<uint64_t>(k->mnId); uint32_t n = 0;
    for (auto *p : k->mvpMapPoints) n += p != nullptr;
    W(n);
    for (size_t i = 0; i < k->mvpMapPoints.size(); i++) if (k->mvpMapPoints[i]) { W<uint32_t>(i); W<uint64_t>(k->mvpMapPoints[i]->mnId); }
}
static void wmp(MapPoint *p) {
    for (auto &o : p->mObservations) wstatic(o.first);
    wstatic(p->mpRefKF);
    W<uint64_t>(p->mnId); W<uint8_t>(p->mbBad); W<int64_t>(p->mpMap ? (int64_t)p->mpMap->mnId : -1);
    W<uint8_t>(p->mpMap ? p->mpMap->mspMapPoints.count(p) : 0);
    for (int i = 0; i < 3; i++) W(p->mWorldPos[i]);
    for (int i = 0; i < 3; i++) W(p->mNormalVector[i]);
    W(p->mfMinDistance); W(p->mfMaxDistance);
    uint8_t d[32] = {0};
    if (!p->mDescriptor.empty()) memcpy(d, p->mDescriptor.ptr<uint8_t>(0), 32);
    for (int i = 0; i < 32; i++) W(d[i]);
    W<int32_t>(p->nObs); W<int64_t>(kid(p->mpRefKF)); W<int32_t>(p->mnVisible); W<int32_t>(p->mnFound);
    W<int64_t>(mid(p->mpReplaced)); W<int64_t>((int64_t)p->mnFirstKFid);
    W<uint32_t>(p->mObservations.size());
    for (auto &o : p->mObservations) { W<uint64_t>(o.first->mnId); W<int32_t>(std::get<0>(o.second)); W<int32_t>(std::get<1>(o.second)); }
}
static void wmap(Map *m) {
    W<uint64_t>(m->mnId); W<uint64_t>(m->mnInitKFid); W<uint64_t>(m->mnMaxKFid); bool ne = !m->mspKeyFrames.empty();  // 처음 키프레임 전에는 두 포인터가 초기화 안 됨
    W<int64_t>(ne ? kid(m->mpKFlowerID) : -1);
    W<int64_t>(ne ? kid(m->mpKFinitial) : -1); W<int32_t>(m->mnBigChangeIdx); W<int32_t>(m->mnMapChange);
    W<uint64_t>(m->mspMapPoints.size()); W<uint32_t>(m->mvpReferenceMapPoints.size());
    W<uint32_t>(m->mspKeyFrames.size()); for (auto *k : m->mspKeyFrames) { wstatic(k); W<uint64_t>(k->mnId); }
}
// 덤프할 묶음: g = 그래프 칸, l = 지도 점 칸, m = 지도 점, maps = 지도
struct Sets { std::set<KeyFrame *> g, l; std::set<MapPoint *> m; std::set<Map *> maps; };
static void wstate(const Sets &s) {
    W<uint32_t>(s.maps.size()); for (auto *m : s.maps) wmap(m);
    W<uint32_t>(s.g.size()); for (auto *k : s.g) wkf(k);
    W<uint32_t>(s.l.size()); for (auto *k : s.l) wlist(k);
    W<uint32_t>(s.m.size()); for (auto *p : s.m) wmp(p);
}
static void addconn(Sets &s, KeyFrame *k) {
    for (auto &e : k->mConnectedKeyFrameWeights) s.g.insert(e.first);
    for (auto *x : k->mvpOrderedConnectedKeyFrames) s.g.insert(x);
}
static void begin(Op op, int64_t a, int64_t b, int64_t c, const Sets &s) {
    W<uint32_t>(0x4352444d); W<uint8_t>(op); W(a); W(b); W(c); wstate(s);
}
// 지도 키프레임에서 그래프로 닿는 것 전부
static void bfs_all(Sets &s, Map *m) {
    std::vector<KeyFrame *> q(m->mspKeyFrames.begin(), m->mspKeyFrames.end());
    for (auto *k : s.g) q.push_back(k);
    std::set<KeyFrame *> seen;
    while (!q.empty()) {
        KeyFrame *k = q.back(); q.pop_back();
        if (!k || !seen.insert(k).second) continue;
        s.g.insert(k);
        std::vector<KeyFrame *> n;
        for (auto &e : k->mConnectedKeyFrameWeights) n.push_back(e.first);
        for (auto *x : k->mvpOrderedConnectedKeyFrames) n.push_back(x);
        n.push_back(k->mpParent);
        for (auto *x : k->mspChildrens) n.push_back(x);
        for (auto *x : k->mspLoopEdges) n.push_back(x);
        for (auto *x : k->mspMergeEdges) n.push_back(x);
        for (auto *x : n) if (x && !seen.count(x)) q.push_back(x);
    }
}

// ---------- KeyFrame ----------
void KeyFrame::UpdateConnections(bool up) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame17UpdateConnectionsEb", void (*)(KeyFrame *, bool));
    Scope sc; bool d = sc.top && want(UC);
    Sets s;
    if (d) {
        s.g.insert(this); s.l.insert(this); if (mpParent) s.g.insert(mpParent); s.maps.insert(mpMap);
        std::vector<KeyFrame *> g0{this};
        for (auto *p : mvpMapPoints) if (p) { s.m.insert(p); for (auto &o : p->mObservations) { s.g.insert(o.first); g0.push_back(o.first); } }
        for (auto *k : g0) addconn(s, k);
        begin(UC, mnId, up, 0, s);
    }
    o_(this, up);
    if (d) {
        wstate(s);
        for (int n : {5, 10, 20, 30}) { auto v = GetBestCovisibilityKeyFrames(n); W<uint32_t>(v.size()); for (auto *x : v) W<uint64_t>(x->mnId); }
        for (int w : {0, 15, 30, 60, 100}) { auto v = GetCovisiblesByWeight(w); W<uint32_t>(v.size()); for (auto *x : v) W<uint64_t>(x->mnId); }
        flush();
    }
}
static void kf_sb_sets(KeyFrame *k, Sets &s) {
    s.g.insert(k); s.l.insert(k); s.maps.insert(k->mpMap);
    for (auto *p : k->mvpMapPoints) if (p) { s.m.insert(p); for (auto &o : p->mObservations) s.l.insert(o.first); }
    bfs_all(s, k->mpMap);
}
void KeyFrame::SetBadFlag() {
    ORIGN("_ZN9ORB_SLAM38KeyFrame10SetBadFlagEv", void (*)(KeyFrame *));
    Scope sc; bool d = sc.top && want(KSB);
    Sets s;
    if (d) { kf_sb_sets(this, s); begin(KSB, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::SetErase() {
    ORIGN("_ZN9ORB_SLAM38KeyFrame8SetEraseEv", void (*)(KeyFrame *));
    Scope sc; bool d = sc.top && want(KSE);
    Sets s;
    if (d) { kf_sb_sets(this, s); begin(KSE, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
#define KF2(NAME, SYM, OP, EXTRA)                                                      \
    void KeyFrame::NAME(KeyFrame *o) {                                                 \
        ORIGN(SYM, void (*)(KeyFrame *, KeyFrame *));                                    \
        Scope sc; bool d = sc.top && want(OP);                                          \
        Sets s;                                                                         \
        if (d) { s.g.insert(this); if (o) s.g.insert(o); if (mpParent) s.g.insert(mpParent); EXTRA; begin(OP, mnId, kid(o), 0, s); } \
        o_(this, o);                                                                    \
        if (d) { wstate(s); flush(); }                                                  \
    }
KF2(EraseConnection, "_ZN9ORB_SLAM38KeyFrame15EraseConnectionEPS0_", EC, addconn(s, this))
KF2(AddChild, "_ZN9ORB_SLAM38KeyFrame8AddChildEPS0_", ACH, )
KF2(EraseChild, "_ZN9ORB_SLAM38KeyFrame10EraseChildEPS0_", ECH, )
KF2(ChangeParent, "_ZN9ORB_SLAM38KeyFrame12ChangeParentEPS0_", CHP, )
KF2(AddLoopEdge, "_ZN9ORB_SLAM38KeyFrame11AddLoopEdgeEPS0_", ALE, )
KF2(AddMergeEdge, "_ZN9ORB_SLAM38KeyFrame12AddMergeEdgeEPS0_", AME, )
void KeyFrame::AddConnection(KeyFrame *o, const int &w) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame13AddConnectionEPS0_RKi", void (*)(KeyFrame *, KeyFrame *, const int &));
    Scope sc; bool d = sc.top && want(AC);
    Sets s;
    if (d) { s.g.insert(this); s.g.insert(o); addconn(s, this); begin(AC, mnId, kid(o), w, s); }
    o_(this, o, w);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::UpdateBestCovisibles() {
    ORIGN("_ZN9ORB_SLAM38KeyFrame20UpdateBestCovisiblesEv", void (*)(KeyFrame *));
    Scope sc; bool d = sc.top && want(UBC);
    Sets s;
    if (d) { s.g.insert(this); addconn(s, this); begin(UBC, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::SetNotErase() {
    ORIGN("_ZN9ORB_SLAM38KeyFrame11SetNotEraseEv", void (*)(KeyFrame *));
    Scope sc; bool d = sc.top && want(SNE);
    Sets s;
    if (d) { s.g.insert(this); begin(SNE, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::AddMapPoint(MapPoint *p, const size_t &idx) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame11AddMapPointEPNS_8MapPointERKm", void (*)(KeyFrame *, MapPoint *, const size_t &));
    Scope sc; bool d = sc.top && want(KAM);
    Sets s;
    if (d) { s.l.insert(this); begin(KAM, mnId, mid(p), idx, s); }
    o_(this, p, idx);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint *p) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame20ReplaceMapPointMatchERKiPNS_8MapPointE", void (*)(KeyFrame *, const int &, MapPoint *));
    Scope sc; bool d = sc.top && want(KRM);
    Sets s;
    if (d) { s.l.insert(this); begin(KRM, mnId, mid(p), idx, s); }
    o_(this, idx, p);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::EraseMapPointMatch(const int &idx) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame18EraseMapPointMatchERKi", void (*)(KeyFrame *, const int &));
    Scope sc; bool d = sc.top && want(KEMI);
    Sets s;
    if (d) { s.l.insert(this); begin(KEMI, mnId, 0, idx, s); }
    o_(this, idx);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::EraseMapPointMatch(MapPoint *p) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame18EraseMapPointMatchEPNS_8MapPointE", void (*)(KeyFrame *, MapPoint *));
    Scope sc; bool d = sc.top && want(KEMP);
    Sets s;
    if (d) { s.l.insert(this); s.m.insert(p); begin(KEMP, mnId, mid(p), 0, s); }
    o_(this, p);
    if (d) { wstate(s); flush(); }
}
void KeyFrame::SetPose(const Sophus::SE3f &T) {
    ORIGN("_ZN9ORB_SLAM38KeyFrame7SetPoseERKN6Sophus3SE3IfLi0EEE", void (*)(KeyFrame *, const Sophus::SE3f &));
    Scope sc; o_(this, T);
}

// ---------- MapPoint ----------
static void mp_sets(MapPoint *p, Sets &s, bool g) {
    s.m.insert(p); if (p->mpMap) s.maps.insert(p->mpMap);
    for (auto &o : p->mObservations) { s.l.insert(o.first); if (g) s.g.insert(o.first); }
    if (g && p->mpRefKF) s.g.insert(p->mpRefKF);
}
void MapPoint::AddObservation(KeyFrame *k, int idx) {
    ORIGN("_ZN9ORB_SLAM38MapPoint14AddObservationEPNS_8KeyFrameEi", void (*)(MapPoint *, KeyFrame *, int));
    Scope sc; bool d = sc.top && want(MAO);
    Sets s;
    if (d) { s.m.insert(this); wstatic(k); begin(MAO, mnId, kid(k), idx, s); }
    o_(this, k, idx);
    if (d) { wstate(s); flush(); }
}
void MapPoint::EraseObservation(KeyFrame *k) {
    ORIGN("_ZN9ORB_SLAM38MapPoint16EraseObservationEPNS_8KeyFrameE", void (*)(MapPoint *, KeyFrame *));
    Scope sc; bool d = sc.top && want(MEO);
    Sets s;
    if (d) { mp_sets(this, s, false); s.l.insert(k); begin(MEO, mnId, kid(k), 0, s); }
    o_(this, k);
    if (d) { wstate(s); flush(); }
}
void MapPoint::SetBadFlag() {
    ORIGN("_ZN9ORB_SLAM38MapPoint10SetBadFlagEv", void (*)(MapPoint *));
    Scope sc; bool d = sc.top && want(MSB);
    Sets s;
    if (d) { mp_sets(this, s, false); begin(MSB, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
void MapPoint::Replace(MapPoint *n) {
    ORIGN("_ZN9ORB_SLAM38MapPoint7ReplaceEPS0_", void (*)(MapPoint *, MapPoint *));
    Scope sc; bool d = sc.top && want(MRP);
    Sets s;
    if (d) { mp_sets(this, s, true); mp_sets(n, s, true); begin(MRP, mnId, mid(n), 0, s); }
    o_(this, n);
    if (d) { wstate(s); flush(); }
}
void MapPoint::ComputeDistinctiveDescriptors() {
    ORIGN("_ZN9ORB_SLAM38MapPoint29ComputeDistinctiveDescriptorsEv", void (*)(MapPoint *));
    Scope sc; bool d = sc.top && want(MCD);
    Sets s;
    if (d) { mp_sets(this, s, true); s.l.clear(); begin(MCD, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
void MapPoint::UpdateNormalAndDepth() {
    ORIGN("_ZN9ORB_SLAM38MapPoint20UpdateNormalAndDepthEv", void (*)(MapPoint *));
    Scope sc; bool d = sc.top && want(MUN);
    Sets s;
    if (d) { mp_sets(this, s, true); s.l.clear(); begin(MUN, mnId, 0, 0, s); }
    o_(this);
    if (d) { wstate(s); flush(); }
}
int MapPoint::PredictScale(const float &dist, KeyFrame *k) {
    ORIGN("_ZN9ORB_SLAM38MapPoint12PredictScaleERKfPNS_8KeyFrameE", int (*)(MapPoint *, const float &, KeyFrame *));
    float mx = mfMaxDistance;
    int r = o_(this, dist, k);
    std::lock_guard<std::recursive_mutex> l(G);
    if (g_depth == 0 && want(MPSK)) { W<uint32_t>(0x5350444d); W(mx); W(dist); W(k->mfLogScaleFactor); W<int32_t>(k->mnScaleLevels); W<int32_t>(r); flush(); }
    return r;
}
int MapPoint::PredictScale(const float &dist, Frame *f) {
    ORIGN("_ZN9ORB_SLAM38MapPoint12PredictScaleERKfPNS_5FrameE", int (*)(MapPoint *, const float &, Frame *));
    float mx = mfMaxDistance;
    int r = o_(this, dist, f);
    std::lock_guard<std::recursive_mutex> l(G);
    if (g_depth == 0 && want(MPSF)) { W<uint32_t>(0x5350444d); W(mx); W(dist); W(f->mfLogScaleFactor); W<int32_t>(f->mnScaleLevels); W<int32_t>(r); flush(); }
    return r;
}
void MapPoint::SetWorldPos(const Eigen::Vector3f &P) {
    ORIGN("_ZN9ORB_SLAM38MapPoint11SetWorldPosERKN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEE", void (*)(MapPoint *, const Eigen::Vector3f &));
    Scope sc; o_(this, P);
}
void MapPoint::IncreaseVisible(int n) {
    ORIGN("_ZN9ORB_SLAM38MapPoint15IncreaseVisibleEi", void (*)(MapPoint *, int));
    Scope sc; o_(this, n);
}
void MapPoint::IncreaseFound(int n) {
    ORIGN("_ZN9ORB_SLAM38MapPoint13IncreaseFoundEi", void (*)(MapPoint *, int));
    Scope sc; o_(this, n);
}

// ---------- Map ----------
typedef void (*FK)(Map *, KeyFrame *);
typedef void (*FP)(Map *, MapPoint *);
typedef void (*F0)(Map *);
typedef void (*FV)(Map *, const std::vector<MapPoint *> &);
#define MAPOP(SIG, ARGS, SYM, OP, FT, SETUP)                         \
    void Map::SIG {                                                 \
        ORIGN(SYM, FT);                               \
        Scope sc; bool d = sc.top && want(OP);                      \
        Sets s;                                                     \
        if (d) { s.maps.insert(this); SETUP; }                      \
        o_ ARGS;                                                    \
        if (d) { wstate(s); flush(); }                              \
    }
MAPOP(AddKeyFrame(KeyFrame *k), (this, k), "_ZN9ORB_SLAM33Map11AddKeyFrameEPNS_8KeyFrameE", PAK, FK, begin(PAK, mnId, kid(k), 0, s))
MAPOP(EraseKeyFrame(KeyFrame *k), (this, k), "_ZN9ORB_SLAM33Map13EraseKeyFrameEPNS_8KeyFrameE", PEK, FK, begin(PEK, mnId, kid(k), 0, s))
MAPOP(AddMapPoint(MapPoint *p), (this, p), "_ZN9ORB_SLAM33Map11AddMapPointEPNS_8MapPointE", PAM, FP, s.m.insert(p); begin(PAM, mnId, mid(p), (int64_t)mspMapPoints.count(p), s))
MAPOP(EraseMapPoint(MapPoint *p), (this, p), "_ZN9ORB_SLAM33Map13EraseMapPointEPNS_8MapPointE", PEM, FP, s.m.insert(p); begin(PEM, mnId, mid(p), (int64_t)mspMapPoints.count(p), s))
MAPOP(InformNewBigChange(), (this), "_ZN9ORB_SLAM33Map18InformNewBigChangeEv", PBC, F0, begin(PBC, mnId, 0, 0, s))
MAPOP(IncreaseChangeIndex(), (this), "_ZN9ORB_SLAM33Map19IncreaseChangeIndexEv", PIC, F0, begin(PIC, mnId, 0, 0, s))
MAPOP(SetReferenceMapPoints(const std::vector<MapPoint *> &v), (this, v), "_ZN9ORB_SLAM33Map21SetReferenceMapPointsERKSt6vectorIPNS_8MapPointESaIS3_EE", PSR, FV, begin(PSR, mnId, v.size(), 0, s))

static void load(const std::string &f, std::vector<double> &ts, std::vector<std::string> &p) {
    std::ifstream in(f); std::string l;
    while (std::getline(in, l)) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream s(l); double t; std::string q; s >> t >> q; ts.push_back(t); p.push_back(q);
    }
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "orb3_mapdump voc yaml seqdir out.bin maxframes\n"); return 1; }
    std::string dir = argv[3];
    g_cap[UC] = 800; g_cap[KSB] = 200; g_cap[KSE] = 200; g_cap[MRP] = 2000; g_cap[MPSK] = 20000; g_cap[MPSF] = 20000;
    int maxf = atoi(argv[5]);
    std::vector<double> tr, td; std::vector<std::string> pr, pd;
    load(dir + "/rgb.txt", tr, pr); load(dir + "/depth.txt", td, pd);
    System S(argv[1], argv[2], System::RGBD, false);
    g_out = fopen(argv[4], "wb");
    size_t j = 0; int n = 0;
    for (size_t i = 0; i < tr.size() && n < maxf; i++) {
        while (j + 1 < td.size() && std::abs(td[j + 1] - tr[i]) <= std::abs(td[j] - tr[i])) j++;
        if (std::abs(td[j] - tr[i]) > 0.02) continue;
        cv::Mat C = cv::imread(dir + "/" + pr[i], cv::IMREAD_UNCHANGED);
        cv::Mat D = cv::imread(dir + "/" + pd[j], cv::IMREAD_UNCHANGED);
        if (C.empty() || D.empty()) continue;
        S.TrackRGBD(C, D, tr[i]);
        n++;
    }
    S.Shutdown();
    { std::lock_guard<std::recursive_mutex> l(G); fclose(g_out); g_out = nullptr; }
    fprintf(stderr, "frames %d\n", n);
    for (int k = 1; k < 32; k++) fprintf(stderr, "op%d calls %ld rec %ld\n", k, g_cnt[k], g_rec[k]);
    return 0;
}
