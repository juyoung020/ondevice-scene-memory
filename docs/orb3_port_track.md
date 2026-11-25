# ORB-SLAM3 포팅 — 포터 TRACK (Tracking 판단 · LocalMapping)

원본 덤프 재생 비교: `slam/tools/orb3_track_dump/`(track_dump.cc = 원본 함수를 가로채 입력 · 결과를 기록, WSL 에서 build.sh 로 빌드)
→ `slam/examples/orb3rs_track_replay.rs` 가 같은 입력으로 Rust 를 돌려 비트 비교.
실행: `DUMP=track_dump_new bash runall.sh 태그 [장면...]` (ds · tum_desk2 · tum_room · tum_360 · tum_floor · tum_rpy), 결과 `~/orbtrack_out/result_태그.txt`.

## 비트 동일 확인된 것
- Tracking: NeedNewKeyFrame · UpdateLocalKeyFrames · UpdateLocalPoints · CreateNewKeyFrame 새 점 · Track 흐름
- LocalMapping: MapPointCulling · KeyFrameCulling · CreateNewMapPoints(RGB-D) · Triangulate(JacobiSVD)
- v5(6 장면: ds 3807 · desk2 640 · room 1362 · 360 756 · floor 1236 · rpy 723 장면) 모든 표 불일치 0.
  ds: CreateNewMapPoints 1187/1187, 삼각측량 새 점 79300/79300, 새 키프레임 점 139775/139775.
  NeedNewKeyFrame 중 원본 호출 도중 입력이 바뀐 경우(장면당 0–7)는 따로 세고 그것도 전부 같음.
- 통합판(7829f22) 병합 뒤 tum_360 · tum_floor 다시 돌려 불일치 0.

## UpdateNormalAndDepth · UpdateConnections(`localmap::update_normal_and_depth` · `update_connections` · `add_connection` · `update_best_covisibles`)
- tum_floor: 원본 순회(KeyFrame* 주소 순)가 id 순과 같은 호출은 전부 비트 동일 — UpdateNormalAndDepth 217,957/217,957, UpdateConnections 18/18.
- 주소 순 ≠ id 순인 호출(UNDP 83,043 중 30,735 · UCON 154 중 75 다름)은 원본 자체가 실행마다 다른 부분: 법선 합 순서 끝자리,
  같은 무게 키프레임의 순서(예: (2,41) · (3,41)). Rust 는 통합판 약속대로 id 순.
- 덤프 UNDP · UCON 은 주소도 기록해 두 경우를 나눠 셈.

## 통합판 연결(`slam/src/orb3sys/steps.rs` TrackSteps: KfSteps)
- 채움: MapPointCulling · KeyFrameCulling
- 남음(hole): ProcessNewKeyFrame(ComputeBoW · UpdateNormalAndDepth · UpdateConnections 필요), CreateNewMapPoints 의 새 점 지도 넣기
  (삼각측량 계산은 끝, MapPoint 생성 · UpdateNormalAndDepth 잇기 남음), SearchInNeighbors(Fuse)
- Tracking 빈자리(StereoInitialization · CheckReplacedInLastFrame · TrackReferenceKeyFrame · TrackWithMotionModel · TrackLocalMap · CreateNewKeyFrame)는 다음 차례

## 고친 것
- KeyFrame::UnprojectStereo 는 mvKeys(보정 전) 좌표를 씀 → `KeyFrame.keys_raw` 칸 추가
- KeyFrame::mRwc = mTwc.rotationMatrix() 는 켤레 사원수를 다시 정규화해 만들어 Rcw 전치와 끝자리가 다름
  → `KfPose.rwc_stored`(UnprojectStereo 용), 광선은 원본처럼 전치(`rwc`)
- Track 흐름: CreateNewKeyFrame 뒤 mpLastKeyFrame · mpReferenceKF · mnLastKeyFrameId 갱신(floor 1143 불일치 원인)
- 덤프: CreateNewMapPoints 기호 이름 오타(17→18), CNMP 기록 중 지도 갱신 잠금, NeedNewKeyFrame 은 읽기~원본 호출 사이 입력이 바뀌면 표시(따로 셈)

## 원본 쪽 문제
- ds 장면에서 원본 LoopClosing::RunGlobalBundleAdjustment 가 KeyFrame::GetChilds 를 끝없이 도는 경우 한 번 봄(신장 트리 순환). runall 에 timeout 2400 s
