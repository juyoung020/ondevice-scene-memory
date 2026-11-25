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

## 기준 원본 = ~/opt/orb3_refid(-ffp-contract=off + 번호 순 비교자, 공통 기준)
- 한 번에: `bash slam/tools/orb3_track_dump/go.sh 태그 [장면...]` — 재생 도구 · 덤프 도구를 저장소 판으로 다시 빌드(기준 빌드에 링크)하고
  6 장면(또는 고른 장면)을 돌려 `${OUTD:-~/orbtrack_out}/result_태그.txt`. `TAGS=` 로 덤프 태그를 고름(UNDP · UCON 은 덤프가 수 GB 라 빼고 돌림),
  `SLP_EVERY`(TrackWithMotionModel · SearchLocalPoints · TrackLocalMap 표본 장면 간격, 기본 8) · `NB_EVERY`(PNKF · SINB 표본 키프레임 간격, 기본 16).
- Rust 산술은 곱합 합치기 없이 원본 연산 순서(track.rs · localmap.rs 의 mul_add 를 모두 뺌).

## Tracking 단계(`orb3rs::track`, RustSystem 에 엮음 — `rsys::HOLES` 비움)
- check_replaced_in_last_frame · track_reference_keyframe · track_with_motion_model(UpdateLastFrame 포함) · search_local_points ·
  track_local_map(UpdateLocalMap → SearchLocalPoints → PoseOptimization(OPT 판 `reloc::pose_optimization_frame`) → track_local_map_finish) ·
  stereo_initialization · create_new_keyframe(new_kf_map_point) · update_motion_and_clean(속도 · VO 짝 정리).
- RustSystem: 흐름(track_flow)이 Tracker 를 쥐고 있어 하위 단계는 흐름 시작 판 + Edits(기준 키프레임 · 맞은 수 · 지역 키프레임/점 · 첫 키프레임)를 보고 끝에 넣음.
  Track 끝(바깥점 칸 비움 · mpReferenceKF · mlRelativeFramePoses.back())은 track_tail. mVelocity 는 RustSystem.velocity.
- t4(6 장면 합): CheckReplacedInLastFrame 8,407/8,407 · TrackReferenceKeyFrame 52/52(점 칸 52/52) · TrackWithMotionModel 1,048/1,048(점 칸 같음) ·
  SearchLocalPoints 1,048/1,048(점 시야 칸 같음) · TrackLocalMap(자세 최적화 · 맞은 수 · 판단) 1,048/1,048 · StereoInitialization 43/43(새 점 14,248/14,248 위치 · 법선 · 거리 · 기술자) ·
  CreateNewKeyFrame 새 점 2,333/2,333 · Track 흐름 8,524/8,524 · UpdateLocalKeyFrames · UpdateLocalPoints 전부 같음.
- 덤프 한계: TrackReferenceKeyFrame 의 ComputeBoW 는 원본 FeatureVector 를 받아 씀(BoW 변환 자체는 LOOP 비교 몫). TrackLocalMap 은 SearchLocalPoints 뒤 장면에서 시작(앞 단계는 ULKF · ULMP · SLPT 로 따로).

## LocalMapping 단계 통째(PNKF · SINB)
- `localmap::process_new_keyframe` · `localmap::search_in_neighbors` 로 빼서 RsSteps 가 부름. 덤프는 단계 앞뒤 지도 조각(중심 키프레임 · 점 · 점을 보는 키프레임 통째, 참조만 되는 키프레임은 짧게)을 기록.
- t4(6 장면, 16 번째 키프레임마다): ProcessNewKeyFrame 148/148(키프레임 연결 · 순서 · 부모 · 자식 · 점, 점 법선 · 거리 · 기술자 · 관측 · nObs, 최근 점 목록, 지도 넣기) ·
  SearchInNeighbors 148/148(같은 칸 + 교체 · 나쁨). 도중 mbAbortBA 가 선 호출 0.
- 삼각측량 · 컬링(곱합 합치기 뺀 뒤, t5 6 장면): CreateNewMapPoints · Triangulate · MapPointCulling · KeyFrameCulling 전부 같음, 삼각측량 새 점 94,376/94,376(ds 83,054). 통합 시험 orb3sys_cnmp(desk2) 통과.

## NeedNewKeyFrame 불일치(원본 false · Rust true, ds · room · desk2 · rpy 에서 장면당 0–1 건)
- 원인: 원본 LocalMapping::Run 이 일이 없어도 3 ms 마다 `SetAcceptKeyFrames(false)` … `SetAcceptKeyFrames(true)` 를 되풀이함(LocalMapping.cc Run 8 · 210 줄).
  덤프한 입력으로 계산하면 c1b(idle) · c2 가 참이라 true 가 나와야 하고(예: rpy 장면 330, 기준 점 296 × 0.75 = 222 > 맞은 수 218, 마지막 KF 323),
  false 가 되는 길은 원본 호출 순간 AcceptKeyFrames 가 false 인 경우뿐 — 짧은 창이라 입력 해시(앞뒤)에도 안 잡힘. Rust 지도 스레드는 이 깜빡임이 없음.
  원본 쪽 경합이라 Rust 는 고치지 않음.

## 통합판 연결(`slam/src/orb3sys/rsys.rs` RsSteps)
- MapPointCulling · CreateNewMapPoints · KeyFrameCulling: 통합 담당이 연결
- ProcessNewKeyFrame(ComputeBoW 는 RsSteps, 나머지 localmap::process_new_keyframe) · SearchInNeighbors(localmap::search_in_neighbors): 포터 TRACK
