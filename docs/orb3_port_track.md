# ORB-SLAM3 포팅 — 포터 TRACK (Tracking 판단 · LocalMapping)

원본 덤프 재생 비교: `slam/tools/orb3_track_dump/`(track_dump.cc = 원본 함수를 가로채 입력 · 결과를 기록, WSL 에서 build.sh 로 빌드)
→ `slam/examples/orb3rs_track_replay.rs` 가 같은 입력으로 Rust 를 돌려 비트 비교.
실행: `DUMP=track_dump_new bash runall.sh 태그 [장면...]` (ds · tum_desk2 · tum_room · tum_360 · tum_floor · tum_rpy), 결과 `~/orbtrack_out/result_태그.txt`.

## 비트 동일 확인된 것
- Tracking: NeedNewKeyFrame · UpdateLocalKeyFrames · UpdateLocalPoints · CreateNewKeyFrame 새 점 · Track 흐름
- LocalMapping: MapPointCulling · KeyFrameCulling · CreateNewMapPoints(RGB-D) · Triangulate(JacobiSVD)
- TUM 360(756 장면): CreateNewMapPoints 201/201, 새 점 3299/3299, NeedNewKeyFrame 755/755

## 고친 것
- KeyFrame::UnprojectStereo 는 mvKeys(보정 전) 좌표를 씀 → `KeyFrame.keys_raw` 칸 추가
- KeyFrame::mRwc = mTwc.rotationMatrix() 는 켤레 사원수를 다시 정규화해 만들어 Rcw 전치와 끝자리가 다름
  → `KfPose.rwc_stored`(UnprojectStereo 용), 광선은 원본처럼 전치(`rwc`)
- Track 흐름: CreateNewKeyFrame 뒤 mpLastKeyFrame · mpReferenceKF · mnLastKeyFrameId 갱신(floor 1143 불일치 원인)
- 덤프: CreateNewMapPoints 기호 이름 오타(17→18), CNMP 기록 중 지도 갱신 잠금, NeedNewKeyFrame 은 읽기~원본 호출 사이 입력이 바뀌면 표시(따로 셈)

## 원본 쪽 문제
- ds 장면에서 원본 LoopClosing::RunGlobalBundleAdjustment 가 KeyFrame::GetChilds 를 끝없이 도는 경우 한 번 봄(신장 트리 순환). runall 에 timeout 2400 s
