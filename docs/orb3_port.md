# ORB-SLAM3 → Rust · NPU 옮기기

지금 위치 백엔드는 C++ ORB-SLAM3(원본 + `slam/cpp/ORB_SLAM3.patch`)를 링크해 쓴다(`slam/src/orb3.rs` · `slam/cpp/orb3.cc`). 이를 모듈별로 Rust(`slam/src/orb3rs/`)로 옮기고, 다 옮기면 C++ 링크를 뺀다.

## 원칙

- 원본과 비트 동일: 자료형 · 연산 순서 · 반올림 · 난수(DUtils::Random 씨앗)까지 맞춘다. 원본 C++ 를 저장소 밖 하네스(WSL `~/opt/orb3_<모듈>`)에 링크해 같은 입력으로 비교, 결과 숫자를 아래 표에.
- 한 모듈씩: 그 모듈만 Rust 로 바꿔 끼워 비교 → 통합 담당(`opt/orb3-rust`)이 받아 C++ 링크를 줄임.
- 비트가 바뀌는 속도 개선 · NPU 결과는 따로 보고(짝 수 · 위치 오차).

## 공통 규칙(모든 포터)

- **비교 기준 빌드 = `-ffp-contract=off` 로 다시 빌드한 원본**(WSL `~/opt/orb3_ref/ORB_SLAM3`, 빌드 `~/opt/orb3_ref/build.sh`, 하네스 빌드 `~/opt/orb3_ref/cxx.sh 소스 출력`, 실행은 `LD_LIBRARY_PATH=~/slam-bench/install/lib`). 원본 · g2o · DBoW2 모두 같은 플래그.
  - 까닭: GCC 13 기본값(-ffp-contract=fast)은 곱 · 덧셈을 FMA 로 합치는데 **같은 식도 인라인된 자리마다 합치는 곱이 다르다**(실측: Sophus `R*p+t` 가 따로 컴파일한 함수에서는 행마다 `fma(r0,p0,fma(r2,p2,r1·p1))+t`, 다른 자리에서는 0 · 1 행 `fma(r0,p0,fma(r1,p1,r2·p2))+t` · 2 행이 또 다름). 자리마다 GIMPLE 을 읽어 따라 하면 맞출 수는 있으나 원본 코드를 고치거나 컴파일러 판이 바뀌면 깨지고, 보드(aarch64 GCC)의 원본도 다른 자리를 합쳐 PC 원본과 이미 비트가 다르다. 소스 순서 그대로 계산하는 빌드를 기준으로 하면 Rust(FMA 로 합치지 않음)와 PC · 보드 모두에서 같은 값이 나온다.
  - LOOP 포터(`opt/orb3-rs-loop` 4e193cc)는 FMA 빌드를 자리마다 따라 해 맞췄다(운영자 확인). 통합 때 그 `mul_add` 자리를 소스 순서로 되돌리고 이 기준 빌드로 다시 비교한다(또는 그 브랜치에서 바꿔 줄 것). FMA 를 쓴 게 속도에 필요하면 "비트가 바뀌는 개선"으로 따로 보고.
  - 기준을 FMA 빌드에서 이 빌드로 바꾼 것 자체가 원본 동작을 바꾸는지(궤적 오차 · 짝 수)는 통합 단계에서 TUM 으로 재서 따로 보고한다.
- 난수: RGB-D 는 srand 를 부르지 않아 glibc rand 씨앗 1, Sim3Solver · MLPnP RANSAC 이 rand 수열 **하나를 함께** 쓴다 → 통합 쪽에서 `GlibcRand` 하나를 모든 모듈이 나눠 쓴다(LOOP 의 `slam/src/orb3loop/crand.rs` 를 공통으로 올림).
- ORBvoc.txt 끝 빈 줄 때문에 무게 0 인 가짜 단어(971814)가 생김 — 단어장 읽기는 이것까지 같게.
- Eigen 고유벡터 정규화는 복소 나눗셈 `(x·s)/(s·s)` 꼴.
- KeyFrameDatabase 는 지난 질의 점수를 키프레임 칸에 남겨 다음 질의가 다시 씀 — 상태를 같게 옮길 것.
- `std::sort`(불안정)는 `orb3rs::stdsort`(libstdc++ 알고리즘 그대로)로.
- **비교 시험은 입력이 없으면 실패**(환경 변수 · 덤프 파일이 없을 때 조용히 통과 금지). 하네스 출력 파일 열기 실패도 실패로.

## 나눠 맡기

| 몫 | 브랜치 | 원본 파일 |
|---|---|---|
| ORB 추출 · 짝짓기 · 공통 자료 구조 · 통합 | `opt/orb3-rust` | ORBextractor · ORBmatcher · Frame · KeyFrame · MapPoint · Map · Atlas |
| 포즈 최적화 · PnP | `opt/orb3-rs-opt` | Optimizer::PoseOptimization · MLPnPsolver |
| 지역 BA · 전역 BA · 그래프 최적화 | `opt/orb3-rs-ba` | Optimizer(LocalBA · GlobalBA · EssentialGraph · Sim3) · g2o |
| BoW · 키프레임 DB · 재위치 · 루프 · Atlas 합치기 | `opt/orb3-rs-loop` | DBoW2 · KeyFrameDatabase · LoopClosing · Sim3Solver |
| NPU 준비 | `opt/orb3-rs-npu` | ORB 추출 · 기술자 · 장소 인식 후보 |

## 최종 목표

최종 판은 ORB-SLAM3 · g2o · DBoW2 · OpenCV · Pangolin · Boost · Eigen · Sophus 를 하나도 링크하지 않는 순수 Rust. FFI 로 원본 일부를 바꿔 끼우는 것은 중간 단계(비트 비교용)로만. ONNX Runtime 걷어내기는 `opt/ort-rust`(ORT 포터) 몫.

## C++ 의존 목록(우리 쓰임: RGB-D, IMU 는 위쪽 벡터만, 뷰어 없음)

| 의존 | 쓰는 곳 | 처리 | 맡은 몫 |
|---|---|---|---|
| ORBextractor | Frame | Rust 옮김(비트 동일) | 통합(이 브랜치) |
| ORBmatcher | Tracking · LocalMapping · LoopClosing | 옮김 | 통합 |
| Frame · KeyFrame · MapPoint · Map · Atlas | 전체 | 자료 구조 정의함, 동작 함수는 옮김 | 통합 |
| Tracking(추적 상태 기계 · 지역 지도 추적 · 키프레임 결정 · 우리 패치: 위쪽 · 옮겨 붙이기) | System | 옮김 | TRACK |
| LocalMapping(점 만들기 · 지우기 · 키프레임 걸러내기) | 지도 스레드 | 옮김 | TRACK(없으면 통합) |
| Optimizer::PoseOptimization · MLPnPsolver | 추적 · 재위치 | 옮김 | OPT |
| Optimizer LocalBA(+ 우리 중력 항 EdgeGravityUp) · GlobalBA · EssentialGraph · OptimizeSim3 · g2o(Levenberg · 희소 Cholesky · SE3Quat · Sim3 · 정점/가장자리) | 지도 · 루프 스레드 | 옮김 | BA |
| OptimizableTypes(EdgeSE3ProjectXYZ 등) | Optimizer | 옮김 | OPT · BA |
| DBoW2 · ORBvoc.txt 읽기 · KeyFrameDatabase · LoopClosing · Sim3Solver · Atlas 합치기 | 루프 스레드 · 재위치 | 옮김 | LOOP |
| OpenCV imgproc(resize · GaussianBlur · cvtColor) · features2d(FAST) | ORBextractor · Tracking | resize · GaussianBlur · FAST 옮김(비트 동일), cvtColor 남음 | CV(없으면 통합) |
| OpenCV core(cv::Mat · 난수 cv::RNG 등) | 전역 | Rust 자료형으로 바꿈, 쓰는 곳마다 같은 계산으로 | CV |
| Sophus SE3f · Sim3f, Eigen float 수학 | 전역 | `types::Se3` 에 원본 순서 그대로의 곱 · 역 · 지수/로그 추가 | 통합(OPT · BA 가 쓰는 함수는 그쪽이 먼저 넣으면 받음) |
| Settings · 설정 yaml 읽기 · Converter | System | 지움(Rust 는 설정 구조체를 바로 받음) | 통합 |
| GeometricCamera · Pinhole | Frame · 최적화 | `types::Camera` 로(왜곡 0) | 통합 |
| KannalaBrandt8 · TwoViewReconstruction(단안 초기화) · ImuTypes · G2oTypes(IMU 가장자리) · 관성 초기화 | 단안 · 어안 · IMU 모드 전용 | 지움(안 씀) | 통합 |
| Viewer · FrameDrawer · MapDrawer · Pangolin · GLEW · OpenGL | 뷰어(꺼 둠) | 지움 | 통합 |
| Boost serialization(Atlas 저장 · 읽기) | 지도 저장(지금 안 씀) | 지움, 필요해지면 Rust 직렬화 | 통합 |
| System(스레드 3 개: 추적 · 지도 · 루프, 잠금) | 백엔드 | Rust 스레드 · 잠금으로 | 통합 |

## 공통 인터페이스(`slam/src/orb3rs/types.rs`)

원본 포인터는 id 로, 소유는 `Atlas` 의 표(`BTreeMap`)로. 실수는 원본과 같은 f32(Sophus SE3f · Eigen float), 시간 f64.

- `type KfId = u64; type MpId = u64; type MapId = u64;`
- `struct KeyPoint { x, y, size, angle, response: f32, octave, class_id: i32 }` — cv::KeyPoint 와 같은 칸
- `type Desc = [u8; 32]`, `fn desc_dist(&Desc, &Desc) -> i32`(ORBmatcher::DescriptorDistance)
- `struct Se3 { q: [f32; 4] /* x y z w, Eigen 저장 순서 */, t: [f32; 3] }` — Sophus::SE3f 와 같은 저장. 곱 · 역은 Sophus/Eigen 순서 그대로 옮길 것(포즈 최적화 몫이 함수 추가)
- `struct Camera { fx, fy, cx, cy, invfx, invfy, bf, b, th_depth: f32, width, height: i32 }`
- `struct Pyramid { nfeatures, scale_factor, nlevels, ini_th_fast, min_th_fast, scale, inv_scale, sigma2, inv_sigma2, log_scale_factor }`
- `struct Frame { id, timestamp, keys, desc, u_right, depth, map_points: Vec<Option<MpId>>, outlier, grid(col*48+row), grid_inv_w/h, min/max_x/y, tcw: Option<Se3>, ref_kf, bow: BTreeMap<u32,f64>, feat: BTreeMap<u32,Vec<u32>> }`
- `struct KeyFrame { id, frame_id, map, timestamp, keys, desc, u_right, depth, map_points, grid, tcw, bad, bow, feat, conn_weights, ordered_conn, parent, children, loop_edges, merge_edges, tcp }`
- `struct MapPoint { id, map, pos, normal, desc, obs: BTreeMap<KfId,(i32,i32)>, ref_kf, first_kf, visible, found, min_dist, max_dist, bad, replaced, track_* , last_frame_seen }`
- `struct Map { id, kfs, mps, origin_kf, bad, change_id }`, `struct Atlas { maps, current, kfs, mps, next_kf, next_mp }`
- 원본의 `std::set<KeyFrame*>` · `std::set<MapPoint*>`(주소 순, 실행마다 다를 수 있음)는 id 순. 하네스에서는 원본 쪽 순회를 id 순으로 정렬해 비교한다.

추출(`orb3rs::extract`): `Extractor::new(Pyramid 설정) ; fn extract(&mut self, gray: &[u8], w, h) -> (Vec<KeyPoint>, Vec<Desc>)`(RGB-D 는 vLappingArea = [0,0] 이라 원본 출력 순서 그대로).

## 모듈 상태

| 모듈 | Rust | 비트 비교 | 속도 전 → 후 | NPU 후보 |
|---|---|---|---|---|
| 공통 자료 구조 | 정의함 | — | — | — |
| ORB 추출(피라미드 · FAST · 사분 트리 · 방향 · rBRIEF) | `orb3rs::extract` 옮김 | **비트 동일**(FMA 빌드 · 기준 빌드 둘 다): 65 장(강의실 40 · TUM office 15 · desk2 10) 점 77,753 개 좌표 · 크기 · 각도 · 응답 · 층 · 기술자 전부 같음 | 원본 5.9 ms(부하 낮을 때) · 같은 부하 8.2–10.4 ms → Rust 1 스레드 9.3–9.6 ms, 층 병렬 5.4–7.5 ms | 점 검출 · 기술자(NPU 몫) |
| Sophus SE3f · Eigen 3×3 수학(`types`: 곱 · 역 · 점 변환 · 회전 행렬 · dot · norm · 3×3 곱 · 역 · hat) | 옮김 | 비트 동일: 무작위 20 만 + 10 만 건 전부 같음(기준 빌드) | — | — |
| Frame(RGB-D 생성 · 격자 · 위치 행렬 · 영역 검색 · 시야 판정 · 깊이 3D) · MapPoint(장면에서 만들기 · PredictScale) · KeyFrame(장면에서) | `orb3rs::frame` 옮김 | 짝짓기 시험 안에서 같이 확인 | — | — |
| 짝짓기: SearchByProjection(지금, 앞 장면) | 옮김 | **비트 동일**: 강의실 30 쌍 · TUM office 20 쌍 × 위치 3 가지 = 150 번, 짝 122,590 개 전부 같음 | 원본 1.3–1.9 ms → Rust 0.85–0.98 ms(한 번) | — |
| 짝짓기: SearchByProjection(지역 지도, isInFrustum 포함) · SearchByBoW(KF, F) · SearchByProjection(F, KF, 재위치) · SearchForTriangulation(+ Pinhole 극선 조건) | 옮김 | **비트 동일**: 50 쌍 × 4 가지 전부 같음(짝 46,424 · 21,504 · 41,490 · 2,916 개) | Rust 0.54–0.63 · 0.17 · 0.60–0.71 · 0.16–0.20 ms(원본 시간은 덤프에 안 잼) | — |
| 짝짓기: SearchByBoW(KF, KF) | 옮김 | **비트 동일**: 50 쌍(두 키프레임 각자 지도 점, 0.75) 전부 같음(짝 21,501 개) | Rust 0.54–0.59 ms | — |
| 짝짓기: Sim3 투영 2 가지 · Fuse(KF, Sim3) · Fuse(KF, 점들, + MapPoint::Replace · AddObservation · ComputeDistinctiveDescriptors) · SearchBySim3, `types::Sim3`(RxSO3 곱 · 역 · SE3 로) | 옮김 | **비트 동일**: 50 쌍 × 5 가지 전부 같음(짝 12,485 · 12,485 · 879 · 27,408 · 10,815 개, Fuse 는 뒤 지도 점 상태 · 기술자 · 바뀐 점까지) | Rust 0.45–0.65 ms | — |

ORBmatcher 의 RGB-D 경로는 전부 옮김(SearchForInitialization 은 단안 전용이라 뺌).

## 정확도 브랜치 따라가기

포터 기준은 `opt/pose-class` e989ca3. 정확도 수정은 `opt/pose-para`(V 자 틀어짐 · 겹침 · 끝 지도 기울기 3 초 평균) · `opt/pose-parb`(실행마다 흔들림)에서 진행 중. 통합 담당이 그쪽의 `slam/cpp/ORB_SLAM3.patch` · `mapmerge.rs` · `refine.rs` · `stitch.rs` · `orb3_backend.rs` 변경을 주기적으로 보고, 확정된 것만 아래 목록에 넣어 해당 모듈 포팅 때 같이 옮긴다(병합하지 않음).

| 확인 시각 | 브랜치 · 커밋 | 패치 변경 | 옮길 모듈 |
|---|---|---|---|
| 11-25 | `opt/pose-para` 33c92fb | 없음(Rust 쪽 refine · orb3_backend: 끝 지도 기울기 가우스 3 초, 판정 보류) | 백엔드 통합 때 |
| 11-25 | `opt/pose-parb` 73c48c5 | 결정성: 루프 스레드까지 기다림 · 스레드별 RANSAC 씨앗 · 포인터 대신 번호 순 비교자(-include) — 미확정 | 확정되면 모든 모듈(하네스 순회 순서도 이것에 맞춤) |

## 실험 기록

### ORB 추출 (11-25)

- 하네스: WSL `~/opt/orb3_extract/ref.cc`(원본 `ORBextractor(1250, 1.2, 8, 20, 7)` 를 libORB_SLAM3 에 링크, `cvtColor(BGR2GRAY)` 회색 + 점 · 기술자 덤프), `~/opt/orb3_extract/rs`(같은 회색으로 Rust 추출, 비트 비교), `rz.cc`(cv::resize · GaussianBlur 단독 비교). 입력 목록 `list.txt` 65 장.
- 시스템 OpenCV 는 4.6(`libopencv_*.so.406`). 원본이 부르는 OpenCV 함수를 같은 계산으로 옮김:
  - resize INTER_LINEAR 8U: 계수 11 비트, 세로는 `((S>>4)·β)>>16` 두 개 합 `+2 >> 2`. **4.6 은 줄 끝까지 이 SIMD 식**(4.10 소스처럼 끝 몇 칸을 스칼라 `+2^21 >> 22` 로 하면 마지막 5 칸이 1 씩 달라 점 185 개 각도가 어긋남 → 실측으로 확인).
  - GaussianBlur 7×7 σ2 8U: 비트 정확 고정소수 커널 [18,34,48,56,48,34,18]/256(오차 확산 반올림), `(ΣΣ + 2^15) >> 16`, 반사 101.
  - FAST 9/16 + 비최대 억제: 점수 = 연속 9 칸 차이의 min · −max 중 큰 것 − 1(SIMD 판과 같은 값).
  - fastAtan2 7 차 다항식, cvRound = cvtss2si(x86) · round_ties_even(aarch64).
  - 사분 트리 노드 정렬은 `std::sort`(불안정)라 libstdc++ 알고리즘을 그대로 옮김: `orb3rs::stdsort`(다른 모듈도 `std::sort` 를 옮길 때 이것을 씀).
- 원본 테두리(EDGE_THRESHOLD 19 칸 반사)는 어떤 계산도 읽지 않아(FAST 칸 · 방향 원 · 기술자 무늬 모두 층 영상 안) 뺌.
- 속도: 처음 15 ms → cvRound 를 libm `rintf` 대신 cvtss2si 로(기술자 3 배), 흐림을 대칭 계수 · u16 가로 합으로, 경계 검사를 구역 assert 한 번으로 → 1 스레드 원본과 비슷. 층마다 rayon 병렬(층끼리 주고받는 값 없음, 비트 같음) → 원본보다 빠름. 다른 에이전트와 CPU 를 나눠 써(부하 12) 절대값은 흔들림.
- 보드 확인 필요: 보드(aarch64)의 OpenCV 판 · SIMD 폭이 다르면 resize 끝 칸 식이 다를 수 있음(이 Rust 판은 PC OpenCV 4.6 과 같음).
- 남은 것: 회색 변환(cvtColor RGB2GRAY 고정소수)은 TRACK · CV 몫과 겹침 — 맡은 이 없으면 통합 때 옮김.

### 기하 · 짝짓기 (11-25)

- `~/opt/orb3_geom`: g.cc(SE3 정규화 · 점 · 역 · 곱 · 회전 행렬 · R·p+t · norm · dot, 20 만 건), g3.cc(3×3 곱 · 역 · hat · 극선 F 행렬, 10 만 건). FMA 빌드와는 수만 건씩 달랐고(위 "공통 규칙"), 기준 빌드와는 전부 같음.
- `~/opt/orb3_match`: ref.cc 가 원본 Frame(RGB-D) 을 만들고 앞 장면(위치 I)의 깊이 있는 점마다 MapPoint(Frame 생성자) → 지금 장면 위치 3 가지(그대로 · 뒤로 0.1 m · 돌림+앞으로 0.1 m: bForward · bBackward · 보통 가지 모두)로 SearchByProjection(th 15) → 짝 덤프. rs 가 같은 회색 · 깊이(float)로 Rust 추출 · Frame · MapPoint · 짝짓기 후 비교. BoW(FeatureVector)는 원본 ORBvoc 로 C++ 가 계산해 덤프(`bow.bin`), Rust 는 그것을 받아 씀(BoW 자체는 LOOP 몫). 같은 쌍에서 지역 지도 투영(isInFrustum 0.5, th 3) · SearchByBoW(0.7) · 재위치 투영(th 10, ORBdist 100) · 삼각측량 짝(0.6, 방향 검사 없음, 위치 돌림+0.1 m)도 비교.

## 포터 LOOP: 장소 인식 · 재위치 · 루프 · Atlas 합치기

코드: `slam/src/orb3loop/`(`dbow2.rs` · `kfdb.rs` · `crand.rs` · `eig4.rs` · `sim3.rs`), 대조 시험: `slam/tests/orb3loop_bit.rs`.
원본: UZ-SLAMLab/ORB_SLAM3 + `slam/cpp/ORB_SLAM3.patch`, Eigen 3.4.0, GCC 13.3 `-O3 -march=native -ffp-contract=off`(공통 기준 빌드 `~/opt/orb3_ref`, Ryzen 9 9950X).

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
| glibc rand | `rand` · `RandomInt` | 원본 실행이 Sim3 에서 쓴 값 | 827 / 827 호출의 수열 |
| Hessenberg · Francis QR · 두 행 떼기 · 고유벡터 역대입 | Eigen 내부 함수 | TUM fr1_room 기준 빌드 실행(LD_PRELOAD 로 가로챔) | 10,576 · 37,365 · 17,200 · 10,576 모두 같음 |
| ComputeSim3 | `Sim3Solver::ComputeSim3` | 같은 실행 | 10,576 / 10,576 (R · t · s · T12 · T21) |
| Sim3Solver 생성자 | `Sim3Solver(pKF1, pKF2, vpMatched12, …)` | 같은 실행(ctor_hook.so) | 499 / 499 (카메라 좌표 · 투영 · 오차 문턱 · 색인 · RANSAC 설정), 호출 중 다른 스레드가 포즈를 바꾼 2 번은 뺌 |
| Sim3 RANSAC | `Sim3Solver::iterate`(5 인자) | 같은 실행(루프 검출 포함) | 827 / 827 (인라이어 · 수렴 · 반복 수 · 행렬 · 투영) |

### 맞추며 알게 된 원본 동작
- ORBvoc.txt 끝 줄바꿈 뒤 빈 줄이 노드 하나(단어 971814)가 된다: 부모는 앞 줄 부모, 무게 0, 설명자 0. 여기로 떨어진 설명자는 BoW 에서 빠진다.
- KeyFrameDatabase 는 질의 표지 · 단어 수 · 점수를 키프레임 멤버에 남겨 다음 질의에서도 읽는다(문턱 못 넘은 이웃의 지난 점수가 누적에 더해짐). Rust 판은 같은 상태를 키프레임 번호별로 들고 있다. 원본 mRelocScore 는 초기화되지 않으며 Rust 판은 0 으로 시작한다.
- DetectNBestCandidates 의 `isBad` 분기는 원본에서 무한 반복이 된다(역색인에 나쁜 키프레임이 없어 닿지 않음). Rust 판은 assert.
- 기준 빌드(-ffp-contract=off)에서도 **Eigen 의 명시적 pmadd(`_mm*_fmadd_ps` 내장 함수)는 FMA 로 남는다**: gemv 커널을 타는 자리(Hessenberg 의 `applyHouseholderOnTheRight` 안 `tmp = right * essential`, 고유벡터 되돌리기 `eivec.leftCols(j+1) * T.col(j)`)는 Rust 에서도 `mul_add`, 나머지(컴파일러가 합치던 스칼라 · 벡터 식, 작은 내적 redux)는 곱한 뒤 더함. 어느 쪽인지는 기준 빌드 덤프로 자리마다 확인했다. 다른 포터도 Eigen 동적 크기 행렬 곱을 옮길 때 같다.
- 덧셈 묶음 순서는 Eigen 전개 순서(예: 3×3·벡터 = `r0·p0 + (r1·p1 + r2·p2)`), GIMPLE(`-fdump-tree-optimized-lineno`)에서 읽음.
- Eigen `eigenvectors()` 정규화는 복소 나눗셈이라 실수부가 `(x*s)/(s*s)` 다(`x/s` 와 비트가 다름).
- RGB-D 는 srand 를 부르지 않아 rand 씨앗 1. 모든 rand 호출자가 한 수열을 같이 쓰므로 통합 때 `GlibcRand` 하나를 나눠 써야 한다.

### 하네스(저장소 밖, WSL `~/opt/orb3_loop_bg`)
- `bow_dump`: 원본 어휘 · ORBextractor · transform · score 덤프(`voc.bin` · `bow.bin`).
- `kfdb_scn`: 원본 `KeyFrameDatabase.cc` 를 가짜 KeyFrame/Map/Frame 헤더로 그대로 컴파일해 시나리오 실행(`kfdb.bin`).
- `ref/run_tum` + `ref/sim3_hook.so` · `ref/eig_hook.so` · `ref/ctor_hook.so`(`build_ref.sh` 로 기준 빌드에 링크, LD_PRELOAD 셋 함께, 덤프 경로 `SIM3_DUMP` · `SIM3C_DUMP` · `EIG_DUMP` · `CTOR_DUMP`, 연관 목록 `/tmp/assoc_room.txt`): 뷰어 없는 rgbd_tum 으로 fr1_room 을 돌리며 Sim3Solver 생성자 · `iterate` · `ComputeSim3` · Eigen 내부 단계 입출력과 rand 값을 가로채 덤프(`~/orbloop_ref`, 약 43 MB).
- 시험: `ORBLOOP_DUMPS=~/orbloop_ref:~/orbloop_out2 ORBLOOP_VOC=.../ORBvoc.txt cargo test --release -p slam --test orb3loop_bit`.

### 루프 닫기 ↔ 지도 스레드 멈춤 조사(진행 중)
- 재현 시도: 화면 없는 원본 rgbd_tum, fr1_desk 12 번(각 150 초 제한, 멈추면 gdb 로 모든 스레드 스택 덤프 — `~/opt/orb3_loop_bg/hang/run.sh`). 12 번 모두 끝까지 감(루프 검출은 1 번에서만 1 회), **재현 안 됨**. CV 포터 기록은 6 번 중 2 번 멈춤 → 드물고 타이밍에 달림.
- 원본 코드에서 본 위험 자리(아직 스택으로 확인 못 함):
  - `CorrectLoop` 가 `mpLocalMapper->EmptyQueue()` 를 **루프 스레드에서** 부름 → `ProcessNewKeyFrame` 이 지도 스레드의 처리와 겹쳐 돎(같은 mpCurrentKeyFrame · mlpRecentAddedMapPoints 를 두 스레드가 씀).
  - 지도 스레드는 큐가 빌 때만 `Stop()` 을 봄. 멈춤 요청 쪽이 여럿(CorrectLoop · MergeLocal · GBA 끝 · 우리 GraftMap)인데 `Release()` 하나가 `mbStopRequested` 를 통째로 지움 → 다른 요청자가 `isStopped()` 를 영영 못 볼 수 있음(GBA 끝 · GraftMap 은 `isFinished` 로만 빠져나옴, CorrectLoop 의 기다림은 빠질 길 없음).
- 포팅판 방침: 지도 스레드 멈춤을 요청자별 토큰(세는 방식)으로, 큐 비우기는 지도 스레드 자신만, 루프 · 합치기 쪽 기다림에 종료 확인을 넣음. 원본 멈춤 스택을 얻으면 위 둘 중 어느 쪽인지 기록.

### 확인 못 한 것 · 남은 일
- 크기 자유 Sim3(단안 · IMU 초기화 전)의 scale 경로: 식은 GIMPLE 대로 옮겼으나 RGB-D 실행에선 안 지나가 대조 안 됨.
- `Tracking::Relocalization` 흐름, `LoopClosing`(DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF · CorrectLoop · MergeLocal), `Atlas`/`Map` 관리, 패치의 지도별 재위치 · GRAFT · 작은 지도 유지: 아직 안 옮김(짝짓기 · BA 몫 인터페이스가 정해지면 이어서).
- KeyFrameDatabase 대조는 실제 BoW + 만든 공시성 시나리오. 실제 실행 중 질의를 가로챈 대조는 안 함.
- 강의실 녹화(ds_1118)는 BoW 단계만; Sim3 · 고유 풀이 대조는 TUM fr1_room 실행 하나.

- `~/opt/orb3_match/ref3.cc` · `rs3`(11-25): 같은 쌍으로 Sim3 투영(S = 사원수 (0.01, −0.02, 0.005, 1.01), t (0.02, −0.01, 0.1), th 10, 1.5, 키프레임 점 10 개마다 미리 짝), 점 키프레임 판, Fuse(KF, Sim3, th 4), Fuse(KF, 점들, th 3: 앞 점은 짝수만 앞 키프레임 관측 · 뒤 점은 모두 관측 → Replace 두 방향 모두 지남), SearchBySim3(BoW(KF,KF) 0.75 로 미리 짝, S12 따로, th 7.5). Sim3 수학(Tcw = SE3(회전 행렬, t/s) · 역 · 점 곱)도 따로 같음. 사원수←행렬 대각 합은 (m00 + m11) + m22(기준 빌드).
- 지도에서 지우기(Map::EraseMapPoint)는 Fuse 안에서 하지 않음 — 나쁨 표시 · replaced 만 두고 지도 표 정리는 통합(Atlas) 때.

### 포터 OPT: 포즈 최적화 · PnP (`opt/orb3-rs-opt`, 11-25)

모듈(`slam/src/orb3rs/`):

- `pinhole.rs` — `GeometricCamera`/`Pinhole` 의 project(f64 · f32) · unproject · projectJac · uncertainty2. 매개변수 f32 그대로.
- `g2o_pose.rs` — g2o 의 단일 자세 최적화 부분: `SE3Quat`(사원수 곱 · 점 사상 · exp · normalizeRotation · 행렬→사원수), Huber 강건 커널, `BaseUnaryEdge::constructQuadraticForm`, `LinearSolverDense`(Eigen `LDLT<MatrixXd>` 분해 + 풀이).
- `pose_opt.rs` — `Optimizer::PoseOptimization`: 단안 간선(EdgeSE3ProjectXYZOnlyPose) · RGB-D 간선(EdgeStereoSE3ProjectXYZOnlyPose), Levenberg(λ 초기값 · ρ · 실패 10 번 · Raul 멈춤), 4 라운드 이상치 재분류 · 3 라운드째 강건 커널 끔. 입력 `Se3f`(Sophus 저장 그대로) · `PoseCam` · `PoseObs`(맵 점 붙은 키포인트만), 출력 (내점 수, 새 자세, 이상치 표시). 어안 두 번째 카메라(mpCamera2) 경로는 RGB-D 에 없어 뺌.

비트 맞추기에서 알아낸 것(다른 몫도 g2o · Eigen 을 옮길 때 같은 함정):

- 원본은 `-O3 -march=native`(GCC 13, AVX-512 · FMA)라 곱 다음 덧셈이 FMA 로 합쳐지는데 자리가 식마다 다르다. 벡터로 묶인 `a·b + c·d` 는 **두 번째** 곱을, 스칼라로 남은 같은 식은 **첫** 곱을 합치는 경우가 있음(예: 사원수 점 회전의 외적 x · y 칸 vs z 칸). 무작위 입력 비교 + 역어셈블로 하나씩 정함.
- 링크 시 약한 기호(`SE3Quat::exp` · `_transformVector` · LDLT 템플릿)는 PLT 로 불려 실행 파일 쪽 사본이 이길 수 있음 → 비교 하네스는 원본 라이브러리 속 코드를 직접 불러(dlsym · 주소) 확인.
- `RobustKernelHuber::dsqr` 는 **float** 칸(δ² 를 float 로 내림).
- LDLT: 대각 갱신 내적은 합침 없음, `A21 -= A20·temp` GEMV 는 행 묶음(4 · 2 행 Packet, 나머지 스칼라)마다 FMA 여부가 다름, Lᵀ 풀이 내적은 항 1 개면 FMA · 5 개면 앞 4 항 곱 반올림.
- 행렬→사원수의 대각 합은 (R22 + R11) + R00 순. Ω² 대각 두 항의 합침 자리도 칸마다 다름.
- 덤프 입력 자세는 **정규화 없이** 비트 그대로 넣어야 함(SE3f 생성자가 다시 정규화하면 3.6 % 호출이 1 ulp 어긋남).

하네스(WSL `~/opt/orb3_opt_bg`, 저장소 밖): `run_dump`(원본 RGB-D 실행, PLT 가로채기로 PoseOptimization 입력 · 출력 덤프, 호출 중 맵 점이 바뀐 호출은 표시), `libreplay`(덤프로 가짜 Frame 을 만들어 라이브러리 원본을 다시 부름 — 덤프 완전성 확인 · 원본 시간), `probe`/`probe2`(작은 연산 무작위 비교), `rs/`(Rust 비교: `dump_check` · `probe_check`).

| 모듈 | 데이터 | 호출 수 | 비트 같음(반환 · 자세 · 이상치) | 원본 µs/호출 | Rust µs/호출 |
|---|---|---|---|---|---|
| PoseOptimization | TUM fr1_desk2 전체(640 프레임) | 1,278(관측 499,213, RGB-D 446,654) | 1,278 / 1,278 | 910 | 195 |
| PoseOptimization | 강의실 ds_1118_121652 앞 1,500 프레임 | 2,938(관측 1,130,633) | 2,938 / 2,938 | 1,196 | 262 |
| 작은 연산(probe) | 무작위 각 20,000 | 사원수 곱 · 점 사상 · SE3 곱 · exp · χ² · 이차 형식 누적 · 단안 야코비안 · Huber · LDLT(라이브러리 속 코드) · float 변환 | 전부 20,000 / 20,000 | — | — |

원본 시간은 `libreplay`(한 스레드, 덤프 입력으로 라이브러리 원본 호출), Rust 는 같은 입력 `dump_check`. 둘 다 이 PC(다른 에이전트와 CPU 공유)라 절대값은 흔들림. 원본 대비 빠른 까닭: g2o 그래프 · 블록 행렬 · 동적 할당 · 가상 호출이 없고 6x6 를 고정 크기로 풂 — 연산 순서는 그대로라 비트 같음.

확인 안 한 것: 관측 3 개 미만 · 활성 간선 0 · Hessian 이 양이 아닌 경우(LDLT 실패 뒤 이전 x 재사용) 경로는 덤프에 나온 만큼만 확인(활성 간선 0 은 강의실 덤프에 있음). 보드(aarch64)에서는 `mul_add` 가 fmadd 라 같은 비트가 나와야 하나 보드에서 돌려 보지는 않음(보드 금지).
