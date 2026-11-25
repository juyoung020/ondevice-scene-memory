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

## 기준 원본 = 결정성 패치판(opt/pose-parb 73c48c5, WSL ~/slam-bench/orb3-parb)
- 덤프 도구 빌드: `ORB=~/slam-bench/orb3-parb ORB_INC=~/slam-bench/orb3-parb/include/OrderById.h OUT=track_dump_parb bash build.sh`,
  실행 `ORB=~/slam-bench/orb3-parb DUMP=track_dump_parb bash runall.sh 태그 [장면...]`. 재생 도구의 키프레임 순서도 id 순.
- p2(6 장면): UpdateNormalAndDepth 9,913,752/9,913,752 · UpdateConnections 4,891/4,891 (주소 순 ≠ id 순 0), 나머지 표 전부 같음.
  예외 NeedNewKeyFrame 4 건(ds 1 · room 3, 원본 false · rust true, 도중 입력 바뀜 표시 없음) — room 을 다시 돌리면 0 이라 지역 지도 스레드 경합으로 봄, 미해결.

## 통합판 연결(`slam/src/orb3sys/rsys.rs` RsSteps)
- MapPointCulling · CreateNewMapPoints · KeyFrameCulling: 통합 담당이 연결(삼각측량 · 컬링은 위 덤프로 비트 동일)
- ProcessNewKeyFrame(ComputeBoW · 관측 추가 · UpdateNormalAndDepth · ComputeDistinctiveDescriptors · UpdateConnections · AddKeyFrame),
  SearchInNeighbors(1 · 2 차 이웃, Fuse 양방향, 점 갱신, UpdateConnections): 포터 TRACK 이 채움. 부품은 비트 확인, 두 단계를 통째로 원본과 비교하는 덤프는 아직
- Fuse 는 MapPoint::n_obs 만 바꾸므로 단계 끝에 TrackExt::mp_nobs 를 맞춤
- 남음: Tracking 빈자리(StereoInitialization · CheckReplacedInLastFrame · TrackReferenceKeyFrame · TrackWithMotionModel · TrackLocalMap · CreateNewKeyFrame),
  ComputeDistinctiveDescriptors 관측 순서(id 순 vs 원본 주소 순) 덤프 확인

## 통합 담당 알림(11-25 저녁)
- 지도 자료 메서드를 `orb3rs::mapdata` 하나로 모음: localmap 의 update_normal_and_depth · update_connections · add_connection · update_best_covisibles · mp_set_bad · mp_erase_observation · kf_set_bad 는 지웠고, `TrackExt` 의 mp_nobs · kf_not_erase · map_init_kf 칸도 지움(MapPoint.n_obs · KeyFrame.not_erase · Map.init_kf_id 를 씀). 재생 도구도 맞춤.
- 덤프 도구가 `~/slam-bench/src/ORB_SLAM3`(FMA 합치기 빌드)에 링크돼 있음 — 기준은 `~/opt/orb3_ref`(-ffp-contract=off). 모은 mapdata 는 기준 빌드와 비트 동일이라 지금 재생에서 UpdateNormalAndDepth 끝자리가 다르게 나옴 → 덤프 도구를 기준 빌드에 다시 링크해 주세요.
