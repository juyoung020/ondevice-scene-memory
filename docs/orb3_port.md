# ORB-SLAM3 Rust 포팅

## 포터 LOOP: 장소 인식 · 재위치 · 루프 · Atlas 합치기

코드: `slam/src/orb3loop/`(`dbow2.rs` · `kfdb.rs` · `crand.rs` · `eig4.rs` · `sim3.rs`), 대조 시험: `slam/tests/orb3loop_bit.rs`.
원본: UZ-SLAMLab/ORB_SLAM3 + `slam/cpp/ORB_SLAM3.patch`, Eigen 3.4.0, GCC 13.3 `-O3 -march=native`(Ryzen 9 9950X).

### 비트 대조 결과

| 모듈 | 원본 | 입력 | 같은 것 |
|---|---|---|---|
| 어휘 로딩(텍스트) | `loadFromTextFile` | ORBvoc.txt | 노드 1,082,074 / 1,082,074 (끝 빈 줄 노드 포함) |
| 어휘 이진 캐시 | — | 같은 표 | 텍스트와 노드 · 무게 비트 같음, 로딩 원본 텍스트 약 5 s → 이진 0.055 s |
| BowVector · FeatureVector(levelsup 4) | `transform` | 강의실 녹화 96 장 · TUM fr1_room 137 장의 원본 ORBextractor 설명자 | 233 / 233, 233 / 233 |
| L1 점수 | `L1Scoring::score` | 위 장면 모든 쌍 | 27,985 / 27,985 |
| 루프 · 합치기 후보 | `DetectNBestCandidates` | 위 BoW + 공시성 · 지도 3 개 · 지움 · 나쁜 지도 · clearMap 시나리오 | 114 / 114 질의 |
| 재위치 후보(패치판) | `DetectRelocalizationCandidates` | 같은 시나리오, 지도별 | 174 / 174 질의 (후보 390 개) |
| 질의 뒤 키프레임 상태 | mnPlaceRecognition* · mnReloc* | 같은 시나리오 | 116 / 116 |
| glibc rand | `rand` · `RandomInt` | 원본 실행이 Sim3 에서 쓴 값 | 753 / 753 호출의 수열 |
| Hessenberg · Francis QR · 두 행 떼기 · 고유벡터 역대입 | Eigen 내부 함수 | TUM fr1_room 실제 실행(LD_PRELOAD 로 가로챔) | 8,539 · 30,053 · 13,891 · 8,539 모두 같음 |
| ComputeSim3 | `Sim3Solver::ComputeSim3` | 같은 실행 | 8,539 / 8,539 (R · t · s · T12 · T21) |
| Sim3 RANSAC | `Sim3Solver::iterate`(5 인자) | 같은 실행(루프 검출 포함) | 753 / 753 (인라이어 · 수렴 · 반복 수 · 행렬 · 투영) |

### 맞추며 알게 된 원본 동작
- ORBvoc.txt 끝 줄바꿈 뒤 빈 줄이 노드 하나(단어 971814)가 된다: 부모는 앞 줄 부모, 무게 0, 설명자 0. 여기로 떨어진 설명자는 BoW 에서 빠진다.
- KeyFrameDatabase 는 질의 표지 · 단어 수 · 점수를 키프레임 멤버에 남겨 다음 질의에서도 읽는다(문턱 못 넘은 이웃의 지난 점수가 누적에 더해짐). Rust 판은 같은 상태를 키프레임 번호별로 들고 있다. 원본 mRelocScore 는 초기화되지 않으며 Rust 판은 0 으로 시작한다.
- DetectNBestCandidates 의 `isBad` 분기는 원본에서 무한 반복이 된다(역색인에 나쁜 키프레임이 없어 닿지 않음). Rust 판은 assert.
- GCC 가 곱셈 · 덧셈을 FMA 로 합친 자리가 자리마다 다르다(예: 3×3 곱의 원소 0..7 과 원소 8 의 순서가 다름). 같은 플래그로 다시 컴파일한 GIMPLE(`-fdump-tree-optimized-lineno`)에서 읽어 `mul_add` 로 그대로 옮겼다.
- Eigen `eigenvectors()` 정규화는 복소 나눗셈이라 실수부가 `(x*s)/(s*s)` 다(`x/s` 와 비트가 다름).
- RGB-D 는 srand 를 부르지 않아 rand 씨앗 1. 모든 rand 호출자가 한 수열을 같이 쓰므로 통합 때 `GlibcRand` 하나를 나눠 써야 한다.

### 하네스(저장소 밖, WSL `~/opt/orb3_loop_bg`)
- `bow_dump`: 원본 어휘 · ORBextractor · transform · score 덤프(`voc.bin` · `bow.bin`).
- `kfdb_scn`: 원본 `KeyFrameDatabase.cc` 를 가짜 KeyFrame/Map/Frame 헤더로 그대로 컴파일해 시나리오 실행(`kfdb.bin`).
- `run_tum` + `sim3_hook.so` · `eig_hook.so`(LD_PRELOAD): 뷰어 없는 rgbd_tum 으로 fr1_room 을 돌리며 `Sim3Solver::iterate` · `ComputeSim3` · Eigen 내부 단계 입출력과 rand 값을 가로채 덤프.
- 시험: `ORBLOOP_DUMPS=~/orbloop_out1:~/orbloop_out2 ORBLOOP_VOC=.../ORBvoc.txt cargo test --release -p slam --test orb3loop_bit`.

### 확인 못 한 것 · 남은 일
- 크기 자유 Sim3(단안 · IMU 초기화 전)의 scale 경로: 식은 GIMPLE 대로 옮겼으나 RGB-D 실행에선 안 지나가 대조 안 됨.
- Sim3Solver 생성자(지도 점 → 카메라 좌표 · 오차 문턱), `Tracking::Relocalization` 흐름, `LoopClosing`(DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF · CorrectLoop · MergeLocal), `Atlas`/`Map` 관리, 패치의 지도별 재위치 · GRAFT · 작은 지도 유지: 아직 안 옮김(짝짓기 · BA 몫 인터페이스가 정해지면 이어서).
- KeyFrameDatabase 대조는 실제 BoW + 만든 공시성 시나리오. 실제 실행 중 질의를 가로챈 대조는 안 함.
- 강의실 녹화(ds_1118)는 BoW 단계만; Sim3 · 고유 풀이 대조는 TUM fr1_room 실행 하나.
