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
| 짝짓기: SearchByProjection(지역 지도) · SearchByBoW(KF, F) · SearchByProjection(F, KF, 재위치) · SearchForTriangulation(+ Pinhole 극선 조건) | 옮김 | 비교 전 | | |
| 짝짓기: Fuse 2 가지 · Sim3 투영 2 가지 · SearchBySim3 · SearchByBoW(KF, KF) | 남음 | | | |

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
- `~/opt/orb3_match`: ref.cc 가 원본 Frame(RGB-D) 을 만들고 앞 장면(위치 I)의 깊이 있는 점마다 MapPoint(Frame 생성자) → 지금 장면 위치 3 가지(그대로 · 뒤로 0.1 m · 돌림+앞으로 0.1 m: bForward · bBackward · 보통 가지 모두)로 SearchByProjection(th 15) → 짝 덤프. rs 가 같은 회색 · 깊이(float)로 Rust 추출 · Frame · MapPoint · 짝짓기 후 비교.
