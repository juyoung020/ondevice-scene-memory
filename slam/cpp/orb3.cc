// ORB-SLAM3(UZ-SLAMLab, GPL-3.0) RGB-D 위치 추적을 C 함수로 감싼 것. Rust(slam::orb3)가 부른다.
// 원본은 ORB_SLAM3_DIR 를 cpp/ORB_SLAM3.patch 로 고쳐 빌드한 것(System.h 의 GetAtlas · GetTracker 접근자,
// 지도별 재위치 · 겹치는 지도 옮겨 붙이기 · 작은 지도 버리지 않기).
// 위치는 모두 3×4 행 우선(카메라 → 세계, 컬러 카메라 광학 좌표).
#include <System.h>
#include <Atlas.h>
#include <KeyFrame.h>
#include <Map.h>
#include <Tracking.h>
#include <opencv2/core.hpp>
#include <cstdint>
#include <exception>
#include <iostream>
#include <unordered_map>

using namespace ORB_SLAM3;

static void put34(const Sophus::SE3f &t, double *o) {
    const Eigen::Matrix<float, 3, 4> m = t.matrix3x4();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) o[i * 4 + j] = m(i, j);
}

// 장면의 기준 키프레임(나중에 지워져도 부모를 따라 끝 위치를 구하려고 남김. ORB-SLAM3 는 키프레임을 지우지 않고 나쁨 표시만 함)
static std::unordered_map<uint64_t, KeyFrame *> &kfs() {
    static std::unordered_map<uint64_t, KeyFrame *> m;
    return m;
}

extern "C" {

// 단어장 · 설정 파일로 RGB-D 위치 추적을 만든다(창 없음). 실패하면 NULL
void *orb3_new(const char *voc, const char *settings) {
    try {
        return new System(voc, settings, System::RGBD, false);
    } catch (const std::exception &e) {
        std::cerr << "orb3_new: " << e.what() << std::endl;
        return nullptr;
    }
}

// 컬러(8 비트 3 채널 w×h, 설정의 Camera.RGB 순서) + 같은 격자의 깊이(16 비트, 설정의 DepthMapFactor 단위)를 넣는다.
// 돌려줌: 추적 상태(Tracking::eTrackingState). twc · 지도 id · 기준 키프레임 id(없으면 -1)를 채움
int orb3_track(void *h, const uint8_t *rgb, const uint16_t *depth, int w, int hh, double t, double *twc, uint64_t *map_id,
               int64_t *ref_kf) {
    System *s = static_cast<System *>(h);
    cv::Mat C(hh, w, CV_8UC3, const_cast<uint8_t *>(rgb)), D(hh, w, CV_16UC1, const_cast<uint16_t *>(depth));
    const Sophus::SE3f tcw = s->TrackRGBD(C, D, t);
    put34(tcw.inverse(), twc);
    Map *m = s->GetAtlas()->GetCurrentMap();
    *map_id = m ? m->GetId() : 0;
    KeyFrame *kf = s->GetTracker()->mCurrentFrame.mpReferenceKF;
    *ref_kf = kf ? (int64_t)kf->mnId : -1;
    if (kf) kfs()[kf->mnId] = kf;
    return s->GetTrackingState();
}

// 앞에 물은 뒤로 큰 지도 바뀜(루프 클로저 · 지도 합치기 · 옮겨 붙이기 · 전체 최적화)이 있었나
int orb3_map_changed(void *h) { return static_cast<System *>(h)->MapChanged() ? 1 : 0; }

// 키프레임(나쁜 것 빼고): id · 지도 id · 위치(키프레임 카메라 → 세계). all = 모든 지도, 아니면 현재 지도만.
// cap 개까지 채우고 전체 개수를 돌려줌
int orb3_keyframes(void *h, int all, uint64_t *ids, uint64_t *maps, double *poses, int cap) {
    Atlas *a = static_cast<System *>(h)->GetAtlas();
    std::vector<Map *> ms;
    if (all) ms = a->GetAllMaps();
    else if (Map *m = a->GetCurrentMap()) ms.push_back(m);
    int k = 0;
    for (Map *m : ms)
        for (KeyFrame *kf : m->GetAllKeyFrames()) {
            if (!kf || kf->isBad()) continue;
            if (k < cap) {
                ids[k] = kf->mnId;
                maps[k] = m->GetId();
                put34(kf->GetPoseInverse(), poses + 12 * k);
            }
            k++;
        }
    return k;
}

// 지도 from 을 지도 into 좌표로 옮겨 붙이기를 부탁(다음 장면 시작에 함). s = into 좌표 ← from 좌표(3×4 행 우선). 받으면 1
int orb3_graft(void *h, uint64_t from, uint64_t into, const double *s) {
    Eigen::Matrix3f r;
    Eigen::Vector3f t;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) r(i, j) = (float)s[i * 4 + j];
        t(i) = (float)s[i * 4 + 3];
    }
    const Sophus::SE3f S(Eigen::Quaternionf(r).normalized().toRotationMatrix(), t);
    return static_cast<System *>(h)->GetTracker()->RequestGraft(from, into, S) ? 1 : 0;
}

// 지도 스레드가 쌓인 키프레임을 다 처리할 때까지 기다림(최대 속도 재생)
void orb3_wait_idle(void *h) { static_cast<System *>(h)->GetTracker()->WaitMappingIdle(); }

// 기준 키프레임 id 의 지금 위치(키프레임 카메라 → 세계)와 지도 id. 지워진 키프레임은 부모를 따라감(원본 SaveTrajectory 와 같음). 없으면 0
int orb3_kf_now(void *h, uint64_t id, double *twc, uint64_t *map) {
    (void)h;
    auto it = kfs().find(id);
    if (it == kfs().end()) return 0;
    KeyFrame *k = it->second;
    Sophus::SE3f trw;
    while (k && k->isBad()) {
        trw = trw * k->mTcp;
        k = k->GetParent();
    }
    if (!k || !k->GetMap()) return 0;
    put34((trw * k->GetPose()).inverse(), twc);
    *map = k->GetMap()->GetId();
    return 1;
}

void orb3_free(void *h) {
    System *s = static_cast<System *>(h);
    s->Shutdown();
    delete s;
}
}
