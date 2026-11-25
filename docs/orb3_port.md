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
