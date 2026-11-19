// ORB-SLAM3(UZ-SLAMLab, GPL-3.0) 스테레오-관성 위치 추적을 C 함수로 감싼 것. Rust(slam::orb3)가 부른다.
// 원본은 보드 ~/ext/ORB_SLAM3(ORB_SLAM3_DIR) 를 빌드한 그대로 쓰고, System.h 에 GetAtlas · GetTracker(지도 · 현재 장면을 읽는 접근자)만 더했다.
// 위치는 모두 3×4 행 우선(카메라 → 세계, 적외선 왼쪽 카메라 좌표).
#include <System.h>
#include <Atlas.h>
#include <KeyFrame.h>
#include <Map.h>
#include <Tracking.h>
#include <opencv2/core.hpp>
#include <cstdint>
#include <exception>
#include <iostream>

using namespace ORB_SLAM3;

static void put34(const Sophus::SE3f &t, double *o) {
    const Eigen::Matrix<float, 3, 4> m = t.matrix3x4();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) o[i * 4 + j] = m(i, j);
}

extern "C" {

// 단어장 · 설정 파일로 위치 추적을 만든다(창 없음). inertial = 스테레오-관성, 아니면 스테레오. 실패하면 NULL
void *orb3_new(const char *voc, const char *settings, int inertial) {
    try {
        return new System(voc, settings, inertial ? System::IMU_STEREO : System::STEREO, false);
    } catch (const std::exception &e) {
        std::cerr << "orb3_new: " << e.what() << std::endl;
        return nullptr;
    }
}

// 적외선 좌우(회색 8 비트 w×h) 한 쌍 + 앞 장면 뒤 IMU 표본 n 개(표본마다 t · 가속도 xyz · 자이로 xyz, 카메라 시계)를 넣는다.
// 돌려줌: 추적 상태(Tracking::eTrackingState). twc · 지도 id · 기준 키프레임 id(없으면 -1) · 지도가 중력에 맞춰졌는지를 채움.
// 중력 맞춤 = IMU 초기화 뒤 첫 관성 정밀화(VIBA 1, 초기화 5 초 뒤)까지 들어섬: 초기화 순간에는 지도 스레드가 세계를 돌리는 동안
// 추적 스레드가 돌리기 전 좌표의 위치를 내기도 한다
int orb3_track(void *h, const uint8_t *l, const uint8_t *r, int w, int hh, double t, const double *imu, int n,
               double *twc, uint64_t *map_id, int64_t *ref_kf, int *imu_init) {
    System *s = static_cast<System *>(h);
    cv::Mat L(hh, w, CV_8UC1, const_cast<uint8_t *>(l)), R(hh, w, CV_8UC1, const_cast<uint8_t *>(r));
    std::vector<IMU::Point> v;
    v.reserve(n);
    for (int i = 0; i < n; i++) {
        const double *p = imu + 7 * i;
        v.emplace_back((float)p[1], (float)p[2], (float)p[3], (float)p[4], (float)p[5], (float)p[6], p[0]);
    }
    const Sophus::SE3f tcw = s->TrackStereo(L, R, t, v);
    put34(tcw.inverse(), twc);
    Map *m = s->GetAtlas()->GetCurrentMap();
    *map_id = m ? m->GetId() : 0;
    *imu_init = m && m->isImuInitialized() && m->GetIniertialBA1();
    KeyFrame *kf = s->GetTracker()->mCurrentFrame.mpReferenceKF;
    *ref_kf = kf ? (int64_t)kf->mnId : -1;
    return s->GetTrackingState();
}

// 앞에 물은 뒤로 큰 지도 바뀜(루프 클로저 · 지도 합치기 · 전체 최적화 · IMU 초기화 단계)이 있었나
int orb3_map_changed(void *h) { return static_cast<System *>(h)->MapChanged() ? 1 : 0; }

// 현재 지도의 키프레임(나쁜 것 빼고): id · 위치(키프레임 카메라 → 세계). cap 개까지 채우고 전체 개수를 돌려줌
int orb3_keyframes(void *h, uint64_t *ids, double *poses, int cap) {
    Map *m = static_cast<System *>(h)->GetAtlas()->GetCurrentMap();
    if (!m) return 0;
    int k = 0;
    for (KeyFrame *kf : m->GetAllKeyFrames()) {
        if (!kf || kf->isBad()) continue;
        if (k < cap) {
            ids[k] = kf->mnId;
            put34(kf->GetPoseInverse(), poses + 12 * k);
        }
        k++;
    }
    return k;
}

void orb3_free(void *h) {
    System *s = static_cast<System *>(h);
    s->Shutdown();
    delete s;
}
}
