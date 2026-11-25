# ORB-SLAM3 → Rust · NPU 옮기기

지금 위치 백엔드는 C++ ORB-SLAM3(원본 + `slam/cpp/ORB_SLAM3.patch`)를 링크해 쓴다(`slam/src/orb3.rs` · `slam/cpp/orb3.cc`). 이를 모듈별로 Rust(`slam/src/orb3rs/`)로 옮기고, 다 옮기면 C++ 링크를 뺀다.

## 원칙

- 원본과 비트 동일: 자료형 · 연산 순서 · 반올림 · 난수(DUtils::Random 씨앗)까지 맞춘다. 원본 C++ 를 저장소 밖 하네스(WSL `~/opt/orb3_<모듈>`)에 링크해 같은 입력으로 비교, 결과 숫자를 아래 표에.
- 한 모듈씩: 그 모듈만 Rust 로 바꿔 끼워 비교 → 통합 담당(`opt/orb3-rust`)이 받아 C++ 링크를 줄임.
- 비트가 바뀌는 속도 개선 · NPU 결과는 따로 보고(짝 수 · 위치 오차).

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
| ORB 추출(피라미드 · FAST · 사분 트리 · 방향 · rBRIEF) | `orb3rs::extract` 옮김 | **비트 동일**: 65 장(강의실 40 · TUM office 15 · desk2 10) 점 77,753 개 좌표 · 크기 · 각도 · 응답 · 층 · 기술자 전부 같음 | 원본 5.9 ms(부하 낮을 때) · 같은 부하 8.2–10.4 ms → Rust 1 스레드 9.3–9.6 ms, 층 병렬 5.4–7.5 ms | 점 검출 · 기술자(NPU 몫) |
| 짝짓기(ORBmatcher) | | | | |

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

#### MLPnPsolver(재위치 PnP + RANSAC)

모듈:

- `mlpnp.rs` — `MlpnpSolver`(생성자 + `SetRansacParameters` · `iterate` · `CheckInliers`), `compute_pose6`(MLPnP 6 점: 비평면 12 열 · 평면 9 열 두 경로), 가우스-뉴턴(최대 5 회), Eigen `JacobiSVD`(정사각 · 1x3 Householder 사전 QR), `FullPivHouseholderQR<Matrix3d>` 랭크, `SelfAdjointEigenSolver<Matrix3d>`(3x3 닫힌 꼴 삼중 대각화 + QR 단계), 행렬식 · 로드리게스.
- `mlpnp_jacs.rs` — `mlpnpJacs`(2x6 야코비안, 식 1,000 줄) 생성 코드. `mlpnp_inv4.inc` — Eigen `compute_inverse_size4`(SSE 판) 생성 코드. 원본 오브젝트 GIMPLE(`-fdump-tree-optimized`, 원본 빌드 플래그 그대로)을 하네스 `gen/gimple_to_rust.py` · `gimple_vec2_to_rust.py` 로 그대로 옮김(FMA 자리 · 연산 순서 · `sincos` 합치기까지).
- 난수는 `iterate(n, &mut |a, b| RandomInt(a, b))` 로 받는다 — 원본은 glibc `rand()` 하나를 Sim3Solver 등과 나눠 쓰므로 통합 쪽이 `opt/orb3-rs-loop` 의 `crand.rs`(glibc rand, 씨앗 1) 생성기 하나를 넘겨야 같은 수열.
- 원본 `Refine()` 의 `computePose`(내점 전부)는 결과를 버리는 코드(mRi 를 갱신하지 않음)라 부르지 않고, `Refine` 의 `CheckInliers` 는 같은 mRi 로 다시 세는 것이라 직전 결과를 씀 — 출력 · 상태 비트는 같고 시간만 줄어듦.

비트 맞추기에서 알아낸 것(Eigen 커널마다 GCC 의 FMA 합치기가 다름 — GIMPLE 을 읽어 정하고 하네스로 확인):

- `JacobiSVD` 회전: 행(왼쪽) `x' = fma(c, x, s·y)`, `y' = fma(−s, x, c·y)`, 열(오른쪽)은 전치 회전으로 같은 꼴. U 는 `applyOnTheRight(j_leftᵀ)` 가 다시 전치해 j_left 그대로. `real_2x2_jacobi_svd` 의 2x2 왼쪽 회전은 2 칸 벡터 FMA.
- 1x3 영공간(Householder): `tailSqNorm = fma(x2, x2, x1²)`, `beta = sqrt(fma(c0, c0, tail))`, Q 대각 `fma(−τe, e, 1)`.
- AᵀA(GEMM gebp, 깊이 12): 4 열 묶음은 k < 8 을 짝 · 홀 두 누산기로 나눠 FMA 뒤 합치고 남은 k 를 FMA, 묶음 밖 열은 한 사슬. 9x9(평면)의 마지막 행은 "한 행씩" 경로 — k 0..8 을 ((0+2)+(4+6)) · ((1+3)+(5+7)) 로 따로 더한 뒤 FMA, 끝에 짝 + 홀.
- 3x3 `U·Vᵀ`: 비평면 경로는 칸마다 곱 반올림 후 차례로 더함(합침 없음), 평면 경로는 지연 곱(목적지 16 바이트 정렬: 열마다 두 행 Packet FMA 사슬, 한 행 스칼라) — 같은 식인데 인라인 자리마다 다름.
- 고정 3x3 곱 · 행렬×벡터: 0 · 1 행은 k 오름차순 FMA 사슬, 2 행은 `fma(x0, y0, fma(x2, y2, x1·y1))`(Eigen 펼친 합 p0 + (p1 + p2)). 노름은 `sqrt(fma(z, z, x² + y²))`, 내적은 `fma(a2, b2, a0b0 + a1b1)`.
- `CheckInliers`: double 로 `fma(z, R02, fma(x, R00, y·R01)) + t` 를 float 로 내린 뒤 float 투영, `e² = fma(dx, dx, dy²)`.
- 가우스-뉴턴의 JᵀJ 는 GEMM(6 행 = 반 · 4 분의 1 Packet, 열 0..4 묶음 · 4 · 5 단독), g = Jᵀr · dl = J dx 는 GEMV(0 에서 시작하는 FMA 사슬), LDLT 는 포즈 최적화의 `ldlt_factor6` · `ldlt_back6` 그대로 맞음.

하네스(WSL `~/opt/orb3_opt_bg`): `run_mlpnp`(원본 RGB-D 실행 중 `MLPnPsolver::iterate` 를 가로채 입력 상태 · RandomInt(스레드별) · 출력 · 상태를 덤프, 원본 iterate 시간 측정. `HOP=n` 이면 지도 구간 안에서 n 번 순간 이동해 재위치를 일으킴), `libmlpref.so`(`mlpref.cc`: 라이브러리 원본 `computePose` · `mlpnp_gn` · `mlpnpJacs` · 로드리게스 · Eigen 약한 기호(JacobiSVD 생성자 · 4x4 역행렬 · 행렬식 · FullPivQR)를 dlsym 으로 부름 + computePose 비평면 · 평면 본문을 그대로 베낀 추적판), `rs/src/mlpnp_check.rs`(`unit` 함수별 무작위 대조, `trace`/`planar` 중간값 대조, `dump` 덤프 재실행 + 표본마다 원본 computePose 대조), `gen/`(GIMPLE → Rust 생성기).

| 대상 | 데이터 | 호출 수 | 비트 같음 | 원본 µs | Rust µs |
|---|---|---|---|---|---|
| iterate(출력 · 상태 전부) | 강의실 ds_1118_121652 1,000 프레임 + 순간 이동 600 번 | 25,396(표본 32,385) | 25,396 / 25,396 | 108.5 / 호출 | 30.7 / 호출 |
| iterate | TUM fr1_desk2 + 순간 이동 800 번 | 779(표본 2,909) | 779 / 779 | — | 96.5 / 호출 |
| iterate | TUM desk2 → room 되감기(실패 시도 포함) | 21(표본 150) | 21 / 21 | — | — |
| computePose(6 점) | 위 덤프 표본 | 4,682 | 4,682 / 4,682 | 56–75 | 22–25 |
| computePose 비평면 · 평면(랭크 2) | 무작위 장면 각 5,000 | 10,000 | 10,000 / 10,000 | — | — |
| 단위(무작위 각 3,000) | 영공간 · 야코비안 · 로드리게스 양방향 · 4x4 역행렬 · 행렬식 · 랭크(일반 · 평면) · SVD 3x3(U · V) · 12x12(V) · 가우스-뉴턴 | — | 전부 같음 | — | — |

원본 iterate 시간은 `run_mlpnp` 안에서 원본 함수 호출을 잰 것(실행 중 다른 스레드와 같이 돎), Rust 는 같은 입력 재실행. 원본 대비 빠른 까닭: 결과를 버리는 Refine 의 내점 전부 computePose 를 건너뜀 · 고정 크기 행렬(동적 할당 없음)· 희소 행렬 P 없음. 원본 실행은 스레드 때문에 매번 재위치 시도 수가 다름(같은 설정에서 370 · 25,396 호출) — 비교는 덤프 단위라 영향 없음.

확인 안 한 것: 평면 경로는 실제 덤프에 한 번도 나오지 않음(세계 원점을 지나는 평면 위 6 점에서만 생김) — 무작위 평면 장면으로만 확인. 최소 집합 6 이 아닌 설정(`SetRansacParameters` 의 minSet)은 옮기지 않음(원본 Tracking 은 6 만 씀). 고윳값 반복이 30·n 회를 넘는(수렴 실패) 경로 · NaN 입력 경로는 확인 못 함.
