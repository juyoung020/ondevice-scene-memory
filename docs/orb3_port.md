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
| 지역 BA · 전역 BA | `opt/orb3-rs-ba` | Optimizer(LocalBA · GlobalBA) · g2o |
| Sim3 최적화 · 본질 그래프 | `opt/orb3-rs-cv` | Optimizer(OptimizeSim3 · OptimizeEssentialGraph) |
| BoW · 키프레임 DB · 재위치 · 루프 · Atlas 합치기 | `opt/orb3-rs-loop` | DBoW2 · KeyFrameDatabase · LoopClosing · Sim3Solver |
| NPU 준비 | `opt/orb3-rs-npu` | ORB 추출 · 기술자 · 장소 인식 후보 |

## 최종 목표

최종 판은 ORB-SLAM3 · g2o · DBoW2 · OpenCV · Pangolin · Boost · Eigen · Sophus 를 하나도 링크하지 않는 순수 Rust. FFI 로 원본 일부를 바꿔 끼우는 것은 중간 단계(비트 비교용)로만. ONNX Runtime 걷어내기는 `opt/ort-rust`(ORT 포터) 몫.

## C++ 의존 목록(우리 쓰임: RGB-D, IMU 는 위쪽 벡터만, 뷰어 없음)

| 의존 | 쓰는 곳 | 처리 | 맡은 몫 |
|---|---|---|---|
| ORBextractor | Frame | Rust 옮김(비트 동일) | 통합(이 브랜치) |
| ORBmatcher | Tracking · LocalMapping · LoopClosing | 옮김 | 통합 |
| Frame · KeyFrame · MapPoint · Map · Atlas | 전체 | 자료 구조 정의함, 동작 함수는 옮김. KeyFrame.cc · MapPoint.cc · Map.cc 메서드는 `orb3rs::mapdata`(비트 동일) | 통합 · OPT |
| Tracking(추적 상태 기계 · 지역 지도 추적 · 키프레임 결정 · 우리 패치: 위쪽 · 옮겨 붙이기) | System | 옮김 | TRACK |
| LocalMapping(점 만들기 · 지우기 · 키프레임 걸러내기) | 지도 스레드 | 옮김 | TRACK(없으면 통합) |
| Optimizer::PoseOptimization · MLPnPsolver | 추적 · 재위치 | 옮김 | OPT |
| Optimizer LocalBA(+ 우리 중력 항 EdgeGravityUp) · GlobalBA · EssentialGraph · OptimizeSim3 · g2o(Levenberg · 희소 Cholesky · SE3Quat · Sim3 · 정점/가장자리) | 지도 · 루프 스레드 | 옮김 | BA |
| OptimizableTypes(EdgeSE3ProjectXYZ 등) | Optimizer | 옮김 | OPT · BA |
| DBoW2 · ORBvoc.txt 읽기 · KeyFrameDatabase · LoopClosing · Sim3Solver · Atlas 합치기 | 루프 스레드 · 재위치 | 옮김 | LOOP |
| OpenCV imgproc(resize · GaussianBlur · cvtColor) · features2d(FAST) · calib3d(undistortPoints) · core(convertTo · copyMakeBorder) | ORBextractor · Tracking · Frame | resize · GaussianBlur · FAST 는 `orb3rs::extract`, cvtColor · convertTo · copyMakeBorder · undistortPoints 는 `orb3cv::img`(비트 동일) | CV |
| OpenCV core(cv::Mat · 난수 cv::RNG 등) | 전역 | Rust 자료형으로 바꿈, 쓰는 곳마다 같은 계산으로 | CV |
| g2o::Sim3(double) | 루프 · 합치기 · 본질 그래프 | `orb3cv::g2osim3`(비트 동일) | CV |
| Sophus SE3f · Sim3f, Eigen float 수학 | 전역 | `types::Se3` 에 원본 순서 그대로의 곱 · 역 · 지수/로그 추가 | 통합(OPT · BA 가 쓰는 함수는 그쪽이 먼저 넣으면 받음) |
| Settings · 설정 yaml 읽기 · Converter | System | `orb3cv::settings`(RGB-D 핀홀 설정 읽기 + Tracking · Frame 파생 값, 비트 동일), Converter 는 지움 | CV |
| Eigen JacobiSVD · GeometricTools::Triangulate | LocalMapping(새 점 삼각측량) | `orb3cv::svd`(비트 동일) | CV |
| GeometricCamera · Pinhole | Frame · 최적화 | `types::Camera` 로(왜곡 0) | 통합 |
| KannalaBrandt8 · TwoViewReconstruction(단안 초기화) · ImuTypes · G2oTypes(IMU 가장자리) · 관성 초기화 | 단안 · 어안 · IMU 모드 전용 | 지움(안 씀) | 통합 |
| Viewer · FrameDrawer · MapDrawer · Pangolin · GLEW · OpenGL | 뷰어(꺼 둠) | **패치에서 지움**(아래 'Pangolin 없이 빌드') | CV |
| Boost serialization(Atlas 저장 · 읽기) | 지도 저장(지금 안 씀) | `orb3cv::atlas_io`(우리 이진 형식, 원본 PreSave · PostLoad 와 같은 칸 남기기 · 되돌리기) | CV |
| System(스레드 3 개: 추적 · 지도 · 루프, 잠금) | 백엔드 | 진입(설정 · 카메라 · 층 표 · Atlas 불러오기 · 회색 · 깊이 변환)은 `orb3cv::system::SystemInit`, 스레드 · 잠금은 남음 | 통합 |

## Pangolin 없이 빌드

`slam/cpp/ORB_SLAM3.patch` 가 Viewer · FrameDrawer · MapDrawer 와 Pangolin 참조를 걷어내고, `slam/build.rs` 는 Pangolin · GLEW · OpenGL 을 링크하지 않는다(PANGOLIN_DIR 없어짐).

```sh
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3 && cd ORB_SLAM3 && git checkout 4452a3c
git apply /path/to/slam/cpp/ORB_SLAM3.patch     # 손대지 않은 원본에 그대로 적용됨(11-25 확인)
./build.sh                                     # Pangolin 설치 불필요
ORB_SLAM3_DIR=$PWD cargo build -p slam
```

WSL 사본: `~/slam-bench/orb3-nopango`(빌드됨). 비트 비교 기준 빌드(`-ffp-contract=off`)도 같은 패치로 만든다.

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
- `struct Map { id, kfs, mps /* 둘 다 id 순 정렬 유지 */, origin_kf, bad, change_id, init_kf_id, max_kf_id, lower_kf, ref_mps, big_change_idx, change_notified }`, KeyFrame 에 `first_conn_done`(= !mbFirstConnection) · `not_erase` · `to_be_erased`, `struct Atlas { maps, current, kfs, mps, next_kf, next_mp }`
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

## 통합 상태(opt/orb3-rust, 11-25)

- 받은 브랜치: LOOP(4b1e3ff까지) · OPT(a4fbc75) · CV(e805fef) · TRACK(dac48f1) · ORT(a79d7f1). 공통 `types.rs` 는 통합판을 유지(TRACK 판은 그 부분집합이라 그대로 빌드됨).
- 빌드: `ORB_SLAM3_DIR=~/slam-bench/orb3-nopango`(화면 제거 패치판, PANGOLIN_DIR 불필요).
- 병합 뒤 확인: 짝짓기 하네스(`~/opt/orb3_match` rs · rs3) 50 쌍 전부 비트 동일, LOOP 시험 9 개 통과(`ORBLOOP_DUMPS=~/orbloop_ref:~/orbloop_out2`), OPT 덤프 4,216 호출 비트 동일. TRACK 재생(`examples/orb3rs_track_replay`)은 지금 남은 덤프 `~/orbtrack_out/ds.bin` 이 TRACK 포터의 새 덤프 형식이라 대조 못 함(그쪽 결과 기록 `result_v2.txt`: tum_desk2 전 항목 비트 동일).
- 공통 함정(LOOP): 기준 빌드(`-ffp-contract=off`)에서도 Eigen 이 명시적으로 부르는 pmadd(`_mm*_fmadd_ps`)는 FMA 로 남음 — Hessenberg applyHouseholderOnTheRight 행렬 곱 · 고유벡터 되돌리기 행렬 곱 등 Eigen 동적 행렬 곱을 옮길 때는 `mul_add`.
- 원본 버그(포팅판에서 고칠 것): `*Loop detected` → `Local Mapping STOP` 뒤 멈추는 교착 — 원본 · 화면 제거판 둘 다 재현. 포팅판의 루프 닫기 · 지도 스레드 정지 신호에서 고친다.
- 받은 브랜치(2 차): CV cd4e91f(Map::PreSave EraseObservation), LOOP cd2fb9c(통합판을 받은 병합뿐). 병합 뒤 LOOP 시험 9 개 통과, 작업 공간 빌드 통과.
- System 자리 `slam/src/orb3sys/`: `orb3_backend` 는 trait `Orb3System`(track · map_changed · keyframes · graft · wait_idle · kf_now)만 보고 `orb3sys::open` 으로 연다 — 지금은 원본 `orb3::System` 이 그 구현. 원본 C++ 와 Rust 판은 지도 자료(KeyFrame* 대 `orb3rs::types::Atlas`)가 따로라, 한 실행 안에서 함수 하나씩 원본으로 대신 부르려면 호출마다 자료를 양쪽으로 옮겨야 함(TRACK 재생의 `TrackOps` 가 덤프 값으로 하는 방식). 그래서 실제 실행에서 바꿔 끼우는 단위는 System 통째: Rust 판은 BA · LoopClosing 흐름 · Relocalization 이 들어온 뒤 `Orb3System` 구현으로 붙인다.
- 스레드 · 잠금 `orb3sys::threads`(교착을 구조로 없앤 판, CV 포터의 스레드 설계와 맞출 기준안): 원본 멈춤 신호의 세 문제 — ① `RequestStop` 깃발 하나에 `Release` 를 누가 부르든 함께 내림(루프 닫기와 옮겨 붙이기가 같이 세우면 먼저 끝난 쪽이 다른 쪽 작업 중에 지도 스레드를 다시 돌림) ② `usleep` 바쁜 대기 ③ 지도 갱신 잠금을 쥔 채 멈춤을 기다리는 자리(패치 DoPendingGraft 는 Track() 안, 지역 BA 는 같은 잠금을 잡으러 감). 이 판: 멈춤은 `StopToken` 세는 표(마지막 토큰이 놓일 때만 다시 돎), 기다림은 모두 Condvar(끝도 깨움), 지도 갱신 잠금은 `MapUpdate` 로만 잡고 쥔 채 멈춤 · 한가함 · 대기열을 기다리면 바로 패닉. 결과에 닿는 순서는 원본대로(대기열을 먼저 비우고 멈춤, 멈춘 뒤 SetNotStop(true) 거절, Release 때 대기열 비움, 새 키프레임이 오면 BA 중단 깃발). 시험 4 개(`cargo test -p slam --lib orb3sys`): 토큰 둘 · 대기열 먼저 · 잠금 쥔 기다림 패닉 · 세 스레드 3,000 장면 섞어 돌리기 시간 안에 끝남.
- TRACK 재생 길이 문제: 새 덤프(`~/orbtrack_out/ds.bin`, 1.5 GB, CNMP · TRIA 등 새 칸)를 읽는 재생 도구는 TRACK 작업 트리(`C:\ondevice-orb-track`)에 아직 커밋 전. 커밋된 옛 도구로 읽으면 기록 길이가 안 맞아 패닉 → 릴리스 `panic = "abort"` 라 WSL 이 1.5 GB 코어 덤프를 `wsl-crashes` 에 남김(지움). 큰 덤프를 읽는 시험은 `ulimit -c 0` 을 붙여 돌린다. TRACK 이 새 도구를 커밋하면 받아서 다시 대조.
- 원본 대조 전체 재생(강의실 녹화)은 Rust System 이 들어온 뒤: 지금 통합판의 System 은 원본 라이브러리 그대로라 같은 것끼리 비교가 됨.
- 받은 브랜치(3 차): BA 5f6ea69(g2o Levenberg · 슈어 · LDLT · LBA 간선 3 종) · LOOP 3a68a19(Relocalization 흐름 `orb3loop::reloc`, 멈춤 조사 기록). 병합 뒤 BA 시험(`full_calls` 강의실 · desk2 · room, 단계 0 · 4, se3) · LOOP 9 개 · orb3sys 4 개 통과. BA 의 `stage_first_iteration` 이 입력 없을 때 건너뛰던 것을 실패로 바꿈. `loopclose::refdump::*_digiteo2` 2 개 실패는 옛 gtsam 루프 모듈(이번 병합과 무관, 전부터).
- **LocalBundleAdjustment 전체 Rust(`slam::orb3ba::lba`)**: 지역 · 고정 키프레임 고르기 → 그래프(`build_graph`) → `optimize_stop(10)`(g2o 강제 멈춤 깃발 = 반복 시작 · Levenberg 시도 사이) → 이상치 관측 지우기(단안 먼저 · 스테레오) → 자세 · 점 되써 넣기 · `mp_update_normal_and_depth` · 지도 변경 번호. 원본 `KeyFrame` 패치 칸 mbUp · mUpCam 은 `KeyFrame::up_cam`. 원본 표시 칸(mnBALocalForKF · mnBAFixedForKF)은 함수 안 집합으로.
  - 비교: 저장소 밖 하네스 WSL `~/opt/orb3_lba`(BA 하네스의 덤프 판 Optimizer 에 LBA 앞 · 뒤 지도 상태 덤프를 더함, `run.sh 이름 녹화 장수 호출수 [imu] [설정]`) → `~/orblba_ref/{desk2,room,ds}_{w,ba}.bin`. 시험 `ORB3LBA=~/orblba_ref/desk2:~/orblba_ref/room:~/orblba_ref/ds cargo test --release -p slam --lib orb3ba::lba_tests`: 그래프(정점 · 간선 비트) · 반복 수 · 지운 관측 · 지역 키프레임 자세 · 점 칸 · 점 위치 · 법선 · 거리 · 관측 수 · 기준 키프레임 **37 호출 전부 비트 동일**(TUM desk2 12 · room 12 · 강의실 중력 항 포함 13).
  - 원본의 관측 순회(`map<KeyFrame*,…>`)는 포인터 순서라 번호 순과 다르다(실측: 점 관측 순서 [2, 0, 1]) — 간선 추가 순서 · 법선 합 순서 · 기준 키프레임 넘김(begin())이 이 순서를 따른다. Rust 기본은 번호 순(`MapPoint::obs`), 시험만 `ObsOrder` 로 덤프의 포인터 순서를 넣어 나머지를 비트 비교. 실제 실행 결과는 원본 판에서도 할당 주소에 따라 달라지는 자리라 궤적 대조 때 이 차이가 남는다(pose-parb 의 번호 순 비교자가 확정되면 원본도 같아짐).
  - 공통 함수 고침: `localmap::mp_erase_observation` 에 원본의 `if(mpRefKF==pKF) mpRefKF=mObservations.begin()->first` 를 넣음(빠져 있었음).
  - 남은 엮기: `LocalMapperOps::local_ba` 를 이것으로 — ProcessNewKeyFrame · CreateNewMapPoints · SearchInNeighbors(TRACK 작업 트리에 미커밋) 가 들어오면 `Mapper` 위에 Rust LocalMapping 구현을 만든다.
- 받은 브랜치(4 차, 11-25 오후): OPT 65522dd(MLPnPsolver Rust) · LOOP 6579546(재위치 흐름 원본 대조 236/236, Rust MLPnP 를 기본 PnP 로) · CV 5980d71(g2o::Sim3 · Orb3System 용 keyframes · kf_now) · BA 85fe50f(커널 없는 간선 경로 · 실제 GBA 비트 동일 · g2o::Sim3). 문서 충돌만 손으로 풂. g2o::Sim3 는 한 벌로 줄임(아래 5 차).
- 병합 뒤 시험(전부 통과, 입력 없으면 실패하는 시험은 아래 환경변수가 필요. 공통 `CARGO_TARGET_DIR=~/tgt-orbrs ORB_SLAM3_DIR=~/slam-bench/orb3-nopango`, `bash -lc`, 큰 덤프는 `ulimit -c 0`):
  - LOOP 비트 9 개: `ORBLOOP_DUMPS=~/orbloop_ref:~/orbloop_out2 ORBLOOP_VOC=~/slam-bench/orb3-nopango/Vocabulary/ORBvoc.txt cargo test --release -p slam --test orb3loop_bit`
  - 재위치 흐름 2 개(원본 MLPnP FFI · Rust MLPnP): `ORBLOOP_RELOC=~/orbloop_reloc/room.bin:~/orbloop_reloc/ds.bin ORBLOOP_MLPSRV=~/opt/orb3_loop_bg/reloc/mlp_srv cargo test --release -p slam --test orb3loop_reloc`
  - BA 5 개: 입력을 먼저 만듦 — `cd ~/opt/orb3_ba_bg && ./replay ~/orbba_ref/ds_lba.bin /tmp/stage_0.bin 0; cd se3 && ./se3ref /tmp/se3.bin 30000 && ./sim3ref /tmp/sim3.bin 40000` → `ORB3BA_DUMP=~/orbba_ref/ds_lba.bin ORB3BA_STAGE=/tmp/stage_0.bin ORB3BA_CALL=0 ORB3BA_SE3=/tmp/se3.bin ORB3BA_SIM3=/tmp/sim3.bin cargo test --release -p slam --lib orb3ba::tests -- --test-threads=1`. 실제 전역 BA: `ORB3BA_DUMP=~/orbba_ref/room_gba.bin cargo test --release -p slam --lib orb3ba::tests::full_calls`
  - LBA 37 호출: `ORB3LBA=~/orblba_ref/desk2:~/orblba_ref/room:~/orblba_ref/ds cargo test --release -p slam --lib orb3ba::lba_tests`
  - 스레드 9 개: `cargo test --release -p slam --lib orb3sys`
  - MLPnP 비트 대조는 저장소 밖 하네스(`~/opt/orb3_opt_bg/rs`, 포터 OPT 절)라 여기서 다시 돌리지 않음 — 저장소 안에서는 재위치 흐름 시험이 Rust MLPnP 를 지나며 원본과 같음을 확인.
  - `loopclose::refdump::{real,replay}_digiteo2` 2 개 실패: `slam/src/loopclose/` 는 78ba500 뒤로 orb3 작업 커밋이 건드린 적 없음(옛 gtsam 루프 모듈 · 데이터 쪽 문제) → 이번 병합들과 무관.
- **LoopClosing 흐름 `orb3sys::loopflow`**: `LoopQueue` · `StopToken` · `MapUpdate` 위에 Run(합치기 먼저 · 같이 잡힌 루프 버림 · 척도 나쁨이면 continue · 후보 표지 지우기) · CorrectLoop(멈춤 요청 → 돌던 전역 BA 중단(idx++) → 멈춤 기다림 → 잠금 안 고치기 → 전역 BA 시작 → 멈춤 놓기) · MergeLocal(전역 BA 중단이 멈춤 요청보다 먼저) · 전역 BA 스레드(끝에 번호 · 중단 확인 → 지도 스레드 멈춤 → 잠금 안 넣기 → 다시 번호 확인). 계산은 `LoopOps`(detect · merge_accept · merge_local · loop_accept · correct_loop · clear_*) · `GbaOps`(run · apply) 로 받음 — 포터 LOOP · CV · BA 판이 들어오는 자리. 원본과 다른 점: `EmptyQueue`(루프 스레드가 지도 스레드 대기열을 대신 처리) 안 함 — 지도 스레드가 대기열을 다 비운 뒤에만 멈추므로 멈춘 시점 지도는 같고 두 스레드 동시 처리 경합이 없어짐 · 전역 BA 는 잠금 안에서 떠 간 값으로 풀고 mMutexGBA 를 쥔 채 멈춤을 기다리지 않음. 시험 4 개(루프 → BA 넣기 순서, 두 번째 루프가 돌던 BA 를 버림, 합치기 · 척도 거절 · IMU 전 · 루프 거절 갈래, 200 키프레임 루프 28 번).
  - 시험에서 드러난 원본 동작: 멈춤이 풀릴 때(Release) 지도 대기열을 비우므로 추적이 멈춤 중 키프레임을 넣으면 사라진다 — 원본은 NeedNewKeyFrame 이 멈춤 중 키프레임을 안 만들어 막음. Rust 추적도 같은 검사를 지켜야 한다(`Mapper::stop_requested` · `is_stopped`).
- **LocalMapping 스레드 `orb3sys::mapping`**: `Mapper::next` → 지도 잠금 안에서 TRACK 의 `localmap::run_once`(원본 순서) → 루프 대기열. `LocalMapperOps::local_ba` = `orb3ba::lba::local_bundle_adjustment`(중단 깃발 = `Mapper::abort_ba`). 키프레임 단계(ProcessNewKeyFrame · MapPointCulling · CreateNewMapPoints · SearchInNeighbors · KeyFrameCulling)는 `KfSteps` 로 받음 — TRACK 이 커밋하면 이어 붙임(TRACK 작업 트리의 `localmap.rs` 미커밋 수정 351 줄과 겹치지 않게 새 파일로). 원본은 단계 대부분을 잠금 없이 하지만 Rust 판은 키프레임 하나 처리 전체를 잠금 안에서(지역 BA 최적화 동안 추적이 기다림 — 속도는 전체 재생 때 확인).
- 관측 순서(⑤): Rust 기본은 번호 순(`MapPoint::obs` · `ObsOrder(None)`), 비트 시험만 덤프의 포인터 순서를 넣음. 원본 판은 `map<KeyFrame*>` 할당 주소 순이라 실행마다 간선 순서 · 법선 합 · 기준 키프레임이 흔들리고, 실제 실행 궤적 대조에서 이 차이는 남는다. `opt/pose-parb` 의 번호 순 비교자(-include, 미확정)가 들어오면 원본도 번호 순이 되어 맞춰짐.
- 받은 브랜치(5 차): LOOP 334c205(DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF · FindMatchesByProjection `orb3loop::detect`). 시험: `ORBLOOP_LOOPDUMP=~/orbloop_reloc/loop_room.bin:~/orbloop_reloc/loop_ds.bin ORBLOOP_MLPSRV=~/opt/orb3_loop_bg/reloc/mlp_srv cargo test --release -p slam --test orb3loop_detect` 통과. TRACK 은 아직 커밋 없음(작업 트리에 localmap · track · types 수정).
- **g2o::Sim3 한 벌**: 타입은 `orb3cv::g2osim3::G2oSim3` 하나. BA 판(`orb3ba::sim3`, 쓰는 곳은 시험뿐)은 지움 — CV 판이 BA 하네스 덤프(`/tmp/sim3.bin`, sim3ref 4 만 건 × 7 연산)도 비트 동일로 통과(`ORB3BA_SIM3=/tmp/sim3.bin cargo test --release -p slam --lib orb3ba::tests::sim3_ops`). LOOP 의 `detect::G2oSim3` 도 같은 타입(곱 · 회전 식이 같음)이고, LoopClosing.cc 안에 인라인된 자리의 행렬→사원수 대각 합 순서 (m00+m11)+m22 만 `G2oSim3::from_rows` 로 따로 둠(probe 판 `from_mat` 은 (R22+R11)+R00 — 같은 식도 인라인 자리마다 순서가 다름).
- **Rust System 뼈대 `orb3sys::rsys::RustSystem`**(`Orb3System` 구현, 컴파일됨): 설정 · 입력 변환(`orb3cv::system`) → ORB 추출 → `Frame::new_rgbd` → `track::track_flow` → 지도 스레드(`mapping::run`, Rust LBA) → 루프 대기열. 빈자리는 원본 FFI 로 대신 부르지 않고 `hole(이름)` 으로 멈춤 — 남은 목록 `rsys::HOLES`: StereoInitialization · CheckReplacedInLastFrame · TrackReferenceKeyFrame · TrackWithMotionModel · TrackLocalMap · CreateNewKeyFrame(이상 TRACK 몫) . 메워지면 `orb3_backend` 가 `RustSystem::open(설정, w, h)` + `set_vocabulary` 로 연다.
- 받은 브랜치(6 차): TRACK fcb3ec0(CreateNewMapPoints RGB-D · 삼각측량 비트 동일, Track 흐름 마지막 키프레임 갱신, 덤프 도구 `slam/tools/orb3_track_dump`). 통합 쪽 변경: `TrackOps::need_new_keyframe(&mut self, t: &Tracker)` — 흐름이 `&mut Tracker` 를 쥔 채 부르므로 지금 추적 칸을 인자로 받음(재생 도구 구현도 맞춤).
- 지도 스레드 단계 `rsys::RsSteps`(KfSteps): MapPointCulling · CreateNewMapPoints(ORBmatcher(0.6, false) SearchForTriangulation, 공시성 10, 새 점 = AddObservation 둘 · KF 점 칸 · ComputeDistinctiveDescriptors · UpdateNormalAndDepth · mlpRecentAddedMapPoints) · KeyFrameCulling(공시성 전체 순서, mbAbortBA = `Mapper::abort_ba`)은 TRACK 비트 동일 함수를 엮음. 새 점을 지도에 넣는 부분은 원본 대조 전(전체 재생 때 확인).
- 지도 관리(원본 그대로, 실수 계산 없는 상태 조작): Atlas::CreateNewMap(mnLastInitKFidMap = 지금 지도 최대 KF + 1, 새 지도 id = Map::nNextId) · ChangeMap · ResetActiveMap(지도 스레드 한가 대기 → KFDB clearMap → 지금 지도의 KF · 점 지우기(지도 id 유지) → mnLastInitFrameId · mnInitialFrameId · mnLastRelocFrameId = 지금 장면, 상태 NO_IMAGES_YET · 기준/마지막 KF 없음 · 속도 없음, 장면 비움). 흐름 중 바뀌는 추적 칸은 `Edits` 로 모아 흐름 뒤에 넣고 NeedNewKeyFrame 은 넣은 판으로 판단(RGB-D 에서 Reset 뒤 흐름은 바로 끝나 순서 같음).
- Relocalization–KeyFrameDatabase: ComputeBoW(transform 4 층) → `KeyFrameDatabase::detect_reloc`(Atlas 위 `AtlasGraph`, DB 번호 u32) → `orb3loop::reloc::relocalize`(Rust MLPnP, `GlibcRand` 씨앗 1 하나) → 성공 시 mnLastRelocFrameId. KFDB 에 키프레임 넣기는 루프 스레드(LoopOps) 몫이라 아직 비어 있음.
- 6 차 병합 뒤 전체 재실행(11-25 저녁, 전부 통과): LOOP 비트 9 · 재위치 흐름 2 · 공통 영역 찾기 1 · BA 5(입력 다시 만듦) · 실제 전역 BA 1 · LBA 37 호출 1 · orb3sys 9 · orb3cv 10. TRACK 재생(`runall.sh`, 원본 덤프를 새로 만들어야 함, 1.5 GB)은 포터 TRACK 이 자기 브랜치에서 돌린 결과(`~/orbtrack_out/result_v5.txt`)를 따름.
- 받은 브랜치(7 차, 11-25 저녁): OPT 753042f(mapdata) · 64533c4(Sim3 자세 그래프 sim3pg), CV 096e56b(OptimizeSim3 · 본질 그래프 루프 판), LOOP 91816e5(공통 영역 33/33 · CorrectLoop 초안), BA 6d189fc(전역 BA orb3ba::gba), TRACK 648748b(ProcessNewKeyFrame · SearchInNeighbors). OPT 가 옛 orb3ba 사본(sim3.rs 포함)을 함께 올려 add/add 충돌 → 통합판 orb3ba 유지, sim3pg 는 `orb3cv::g2osim3::G2oSim3` 를 씀.
- **지도 자료 메서드 한 벌 `orb3rs::mapdata`**: 지운 옛 판 — TRACK localmap 의 mp_set_bad · mp_erase_observation · kf_set_bad · update_normal_and_depth · update_best_covisibles · add_connection · update_connections, CV atlas_io 의 mp_erase_observation · mp_set_bad, matcher::replace_map_point(→ `mapdata::mp_replace_parts`, 지도 집합 없는 자리용), LOOP correct.rs 의 G2oSim3 inverse · map · Replace. LBA · GBA 의 UpdateNormalAndDepth 는 관측 순서만 넣는 `mapdata::mp_update_normal_and_depth_obs`. `TrackExt` 의 mp_nobs · kf_not_erase · map_init_kf 칸은 지우고 공통 자료(MapPoint.n_obs · KeyFrame.not_erase · Map.init_kf_id)로. 기준 관측이 없을 때 원본 observations[pRefKF] 의 (0, 0) 를 mapdata 에도 넣음. 남은 겹침: LOOP `orb3loop::graph`(update_connections · add_connection 등, Atlas 없이 kfs · mps 판) — LOOP 가 mapdata 로 옮길 차례(운영자가 LOOP 에 알림).
- **기준 빌드 어긋남(TRACK)**: TRACK 덤프 도구(`~/opt/orb3_track_bg`, runall.sh)는 `~/slam-bench/src/ORB_SLAM3`(GCC 기본 FMA 합치기 빌드)에 링크돼 있어, 기준 빌드(-ffp-contract=off)와 비트 같은 mapdata 로 모은 뒤 TRACK 재생에서 UpdateNormalAndDepth 법선 · 거리가 끝자리 다름(TRACK 옛 판은 dot3 = fma 사슬로 FMA 빌드를 따랐음). OPT 대조(mapdata, 기준 빌드)는 강의실 686 · TUM room 206 호출 통과. TRACK 은 덤프 도구를 `~/opt/orb3_ref` 에 다시 링크해야 함(TRACK 에 알림).
- **옮겨 붙이기 `orb3sys::graft`**(패치 TryGraft · DoPendingGraft · GraftMap · RequestGraft): 원본 계산 대조 probe(`~/opt/orb3_sys/graft/probe.cc`, 기준 빌드 플래그) 5 만 건 — YawOnly · 키프레임 자세 × S⁻¹ · S × 점 · 법선 회전 · TryGraft dt · da **비트 동일**(`ORB3GRAFT=/tmp/graft.bin cargo test --release -p slam --lib orb3sys::graft`). 알게 된 것: 패치의 `atan2(R(1,0), R(0,0))` 는 이름공간 없는 C atan2(double), `da * 180.f / M_PI` 는 double 나눗셈, 그 자리 Quaternionf(Matrix3f) 대각 합은 (m22+m11)+m00. 루프 · 전체 BA 중이면 미루는 검사(mMutexBusy · isRunningGBA)는 루프 스레드를 붙일 때(`RustSystem::loop_busy`, 지금 false).
- **ResetActiveMap 원본대로**: Track() 안 mpSystem->ResetActiveMap() 은 표지만, 실제 Tracking::ResetActiveMap 은 다음 장면 시작(장면 만들기 전). 지도 스레드는 `Mapper::reset_active_map`(지금 키프레임을 마친 뒤 대기열 버림 · 최근 점 목록 비움 = ResetIfRequested) — 이전 판의 "한가할 때까지 기다림"은 지움.
- **OptimizeSim3 Rust(`orb3loop::detect::RustSim3`, CV sim3opt 를 detect 트레잇에)**: 공통 영역 찾기 시험에 `ORBLOOP_RUST_S3=1` 을 더하면 FFI 대신 Rust — 원본 기록과 13/13 · 호출 33/33 같음.
- **전역 BA 결과 넣기 `orb3sys::gbaops::RsGba`**(GbaOps): run = 잠금 안에서 뜬 지도로 `gba::global_bundle_adjustment`(10 번, 강건 없음), apply = RunGlobalBundleAdjustment 뒷부분(첫 키프레임부터 신장 트리 BFS · 자식 mTcwGBA 전파 · mTcwBefGBA · 점은 BA 값 또는 기준 키프레임 보정 · 큰 변화 표시). 원본 대조 덤프 없음(전역 BA 하네스는 BA 호출만) — 루프 전체 재생 때 확인.
- KFDB 넣기(mpKeyFrameDB->add)는 원본 LoopClosing::NewDetectCommonRegions 안(작은 지도 · 검출 · 끝) — LoopOps::detect(통합 담당)를 엮을 때 같이 넣음. LOOP 문서는 KeyFrameDatabase 자체만 맡음(겹침 없음).
- 7 차 병합 뒤 시험(전부 통과, 아래 둘 빼고): mapdata(덤프 orb3_mapdump.sh run 으로 다시 만듦) · sim3pg 300 그래프 · graft 5 만 · LOOP 비트 9 · 재위치 2 · 공통 영역 33/13(FFI · Rust 둘 다) · BA 5 · 실제 GBA · gba_tests(room_gba2) · LBA 37 · orb3sys 9 · orb3cv 12. 실패 ① `orb3loop_correct`(LOOP CorrectLoop 초안 — LOOP 브랜치 91816e5 그대로도 0/2, 병합 탓 아님) ② TRACK desk2 재생 UpdateNormalAndDepth · UpdateConnections(주소 순) 차이 — 위 기준 빌드 어긋남.
- **LoopOps Rust 판 `orb3sys::detectops::RsLoop`**: detect = LoopClosing::NewDetectCommonRegions 순서 그대로(`Detector`: 루프 · 합치기 갈래 칸 mpXMatchedKF · mpXLastCurrentKF · mg2oXSlw · 일치 · 못 찾음 수 · 점 목록, 작은 지도(<12 KF) 바로 넣기, 앞 키프레임 이어 찾기 → DetectNBestCandidates(3) → DetectCommonRegionsFromBoW → **KeyFrameDatabase::add**(원본 자리 그대로) → 못 찾으면 SetErase). OptimizeSim3 는 `RustSim3`. merge_accept · loop_accept 는 관성 없음 갈래(늘 받아들임), clear_* 는 SetErase 둘 · 칸 비움(`LoopOps::clear_*` 가 `&mut` 지도를 받게 바꿈). CorrectLoop · MergeLocal 은 빈자리(LOOP · CV 판 대기). KeyFrameDatabase 그래프 어댑터는 `orb3sys::kfgraph::AtlasGraph` 로 뺌. 흐름 단위 원본 대조는 아직 — LOOP 덤프는 Detect* 호출 단위라 흐름 칸(일치 수 등)을 기록하지 않음.
- **흐름 상태 덤프 하네스 `slam/tools/orb3sys_dump`**(`run.sh build | run | graft`, 원본 = `~/opt/orb3_refid`: 번호 순 비교자 + -ffp-contract=off. PLT 로 원본 함수를 가로챔):
  - 전역 BA 결과 넣기: GlobalBundleAdjustemnt 뒤 GBA 스레드가 지도 스레드 멈춤을 처음 확인한 순간(LocalMapping::isStopped 참, 지도 갱신 잠금) 지도 · mTcwGBA · mPosGBA · mnBAGlobalForKF · 첫 키프레임(GBAB) → Release 직전 지도(GBAA). 처음엔 GBA 직후에 찍었다가 그 사이 지역 BA 가 새 키프레임 자세를 바꿔 어긋남 → 넣기 직전으로 옮김. 시험 `ORBSYS_GBA=~/orbsys_gba_room.bin:~/orbsys_gba_ds.bin cargo test --release -p slam --test orb3sys_gba_apply`: **4/4 비트 동일**(TUM room 1 · 강의실 3, 키프레임 자세 · 역 · 점 위치).
  - 새 점 넣기: `ORBSYS_CNMP_EVERY=10` — CreateNewMapPoints 들어갈 때 지도 · mvKeys · MapPoint::nNextId, CheckNewKeyFrames 답, 새 점 전부 · 최근 점 목록. 시험 `ORBSYS_CNMP=~/orbsys_cnmp_desk2.bin cargo test --release -p slam --test orb3sys_cnmp`: TUM desk2 16 호출 · 새 점 157 개 — **구조(새 점 id · 수 · 관측 · nObs · 보임 · 찾음 · 첫 KF · 기준 KF · 최근 목록) 16/16 같음**, 비트는 1/16: 위치가 끝자리 다름 → 포터 TRACK 삼각측량이 FMA 빌드 기준(위 기준 빌드 어긋남). TRACK 이 기준 빌드로 다시 맞추면 통과해야 함(시험은 엄격하게 남겨 둠, 지금 실패).
  - 옮겨 붙이기 계산 probe(graft_probe.cc)도 여기로.
  - 아직: KFDB 그래프 · 공통 영역 찾기 흐름 · 지도 관리(CreateNewMap · ChangeMap · ResetActiveMap · GraftMap) 덤프.
- `create_new_map_points_into`(rsys): 새 점 넣기를 시험할 수 있게 함수로 뺌, 지도 집합 넣기는 `Map::add_map_point`(원본 set).
- 남은 C++ 링크: `orb3.rs` · `cpp/orb3.cc` 가 아직 libORB_SLAM3 를 씀. 다음 단계는 System 흐름(추적 · 지도 · 루프 스레드)을 Rust 모듈로 엮어 `orb3_backend` 를 바꿔 끼우는 것 — 아직 안 옮긴 것: LocalMapping 의 BA 호출부(BA 포터 대기), LoopClosing 흐름 · CorrectLoop · MergeLocal, MLPnPsolver · Relocalization.

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

- `~/opt/orb3_match/ref3.cc` · `rs3`(11-25): 같은 쌍으로 Sim3 투영(S = 사원수 (0.01, −0.02, 0.005, 1.01), t (0.02, −0.01, 0.1), th 10, 1.5, 키프레임 점 10 개마다 미리 짝), 점 키프레임 판, Fuse(KF, Sim3, th 4), Fuse(KF, 점들, th 3: 앞 점은 짝수만 앞 키프레임 관측 · 뒤 점은 모두 관측 → Replace 두 방향 모두 지남), SearchBySim3(BoW(KF,KF) 0.75 로 미리 짝, S12 따로, th 7.5). Sim3 수학(Tcw = SE3(회전 행렬, t/s) · 역 · 점 곱)도 따로 같음. 사원수←행렬 대각 합은 (m00 + m11) + m22(기준 빌드).
- 지도에서 지우기(Map::EraseMapPoint)는 Fuse 안에서 하지 않음 — 나쁨 표시 · replaced 만 두고 지도 표 정리는 통합(Atlas) 때.

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
- 구현: 통합 담당의 `orb3sys/threads.rs`(`Mapper::request_stop` → `StopToken`, 세는 표 · Condvar · 큐 먼저 비우고 멈춤 · `MapUpdate` 잠금 쥔 채 기다리면 패닉) 하나로 씀. LOOP 판 `stopctl.rs` 는 같은 일이라 지움.
- `Tracking::Relocalization` 흐름: `orb3loop/reloc.rs` `relocalize`(후보 → SearchByBoW 0.75 → PnP 5 번씩 → PoseOptimization → 투영 보강 10/100 · 3/64, 원본 순서 그대로), PnP 는 `PnpFactory` 트레잇(원본 MLPnP FFI 또는 포터 OPT 의 Rust 판을 끼움). 부품(짝짓기 · PoseOptimization)은 비트 대조 끝난 것. 기본 PnP 는 포터 OPT 의 Rust MLPnP(`RustPnp`, 난수는 부르는 쪽 `GlibcRand` 를 빌림 — Sim3Solver 와 한 수열).
- 재위치 흐름 대조(`slam/tests/orb3loop_reloc.rs`): 원본 기준 빌드를 되감기 · 건너뛰기 순서로 돌린 덤프(`~/opt/orb3_loop_bg/reloc/reloc_dump`, 후보 질의 직후의 장면 · 후보 키프레임 · 지도 점 · 들어올 때 장면 상태, iterate 마다 짝 · 난수 · 출력, 끝 결과)를 Rust 자료로 옮겨 `relocalize` 를 돌림. PnP 를 ① 원본 MLPnPsolver FFI(`mlp_srv`, 기준 빌드에 링크한 자식 프로세스, 원본 실행 난수를 그대로 받음) ② Rust MLPnP 두 가지로 끼워 둘 다 대조.
  - TUM fr1_room(`0-500,120-500:3,600-1300:6`) 86 번 + 강의실 ds_1118(`0-900,200-900:4,1500-3800:10`) 150 번 = 재위치 236 번(성공 7) **236 / 236 같음**(성공 · 자세 비트 · 지도 점 · 이상치), iterate 26 / 26(짝 목록 · 결과 · 난수 수), FFI · Rust 둘 다.
  - 실패한 재위치는 장면이 들어올 때 상태(앞 추적의 자세 · 짝)를 그대로 남기므로 덤프에 그 상태도 넣음.
  - 실행: `ORBLOOP_RELOC=~/orbloop_reloc/room.bin:~/orbloop_reloc/ds.bin ORBLOOP_MLPSRV=~/opt/orb3_loop_bg/reloc/mlp_srv cargo test --release -p slam --test orb3loop_reloc`(입력 없으면 실패).

### 공통 영역 찾기(DetectCommonRegionsFromBoW · DetectAndReffineSim3FromLastKF) — `orb3loop/detect.rs`
- 옮긴 것: FindMatchesByProjection · DetectCommonRegionsFromLastKF · DetectAndReffineSim3FromLastKF · DetectCommonRegionsFromBoW, g2o::Sim3(double: 사원수 곱 · 점 회전 · 행렬↔사원수 · 곱 · toSophus). OptimizeSim3 는 트레잇(BA 몫이 오면 바꿔 낌).
- 대조(`slam/tests/orb3loop_detect.rs`): 원본 기준 빌드에 `~/opt/orb3_loop_bg/reloc/loop_dump`(두 함수 · OptimizeSim3 · RandomInt 가로챔)를 링크해 돌림. 호출마다 들어갈 때 상태(현재 키프레임 둘레 2 단계 · 후보 둘레 3 단계 공시 키프레임, 그 지도 점과 관측, 출력 칸의 이전 값), 안에서 쓴 난수, OptimizeSim3 입출력, 나올 때 출력. OptimizeSim3 은 원본 FFI(`mlp_srv` 명령 4: 기준 빌드의 Optimizer::OptimizeSim3 를 가짜 키프레임 · 지도 점으로 부름).
  - TUM fr1_room(`0-1362,0-700` 되감기) + 강의실 ds_1118(`0-2400`): 호출 33(참 2) **33 / 33 같음**(결과 · 찾은 키프레임 · gScw 비트 · 일치 수 · 점 목록 · 난수 수), OptimizeSim3 13 / 13(입력 짝 · S · 결과 · 남은 짝 · S 비트). 앞선 실행 덤프(room · ds 3 개 루프)로도 52 / 52 · 13 / 13.
  - OptimizeSim3 까지 간 호출은 모두, 나머지는 25 개 중 하나만 남김(덤프 크기). LastKF 단계는 실제 실행에 드물어 BoW 로 찾은 직후 같은 상태로 원본 DetectAndReffineSim3FromLastKF 를 한 번 더 불러 덤프(지도를 바꾸지 않는 함수).
  - 덤프하는 동안 지도 스레드를 멈춤(RequestStop · isStopped 대기 · Release). 안 멈추면 원본 루프 스레드가 지도 스레드와 같이 돌아 덤프 뒤 상태가 바뀜 — 처음 실행에서 254 호출 중 32 개가 그래서 난수 범위(Sim3 짝 수)가 달랐음.
- 맞추며 알게 된 원본 동작: `SE3f.cast<double>()` 은 SO3d 생성자를 거쳐 double 로 다시 정규화(`unit_quaternion().cast<double>()` 은 안 함 — CorrectLoop 의 NonCorrectedSim3 가 이쪽). Quaterniond 곱은 SSE2 Packet2d 판 식(`quatd_mul`). 원본 버릇: BoW 단계 `pMostBoWMatchesKF` 는 늘 후보 자신, `DetectCommonRegionsFromLastKF` 가 `vpMapPoints` 를 덮어씀, FindMatchesByProjection 의 2 단계 이웃은 걸러지지 않고 통째로 붙음, DetectAndReffine 은 OptimizeSim3 결과를 버림(모두 `detect.rs` 머리말).
- 강의실 실행에서 원본이 두 번 멈춤(`Loop detected` 뒤 `Local Mapping STOP` 다음 진행 없음, 25 분)과 한 번 `std::system_error: Invalid argument` 로 죽음. 덤프 쪽 멈춤 요청 · 풀기(Release 하나가 다른 요청자의 멈춤까지 지움)가 원본의 그 위험을 더 드러냈을 수 있음 — 포팅판은 `orb3sys::threads` 의 StopToken.

### 확인 못 한 것 · 남은 일
- 크기 자유 Sim3(단안 · IMU 초기화 전)의 scale 경로: 식은 GIMPLE 대로 옮겼으나 RGB-D 실행에선 안 지나가 대조 안 됨.
- `LoopClosing`(CorrectLoop · MergeLocal), `Atlas`/`Map` 관리, 패치의 지도별 재위치 · GRAFT · 작은 지도 유지: 아직 안 옮김(짝짓기 · BA 몫 인터페이스가 정해지면 이어서).
- KeyFrameDatabase 대조는 실제 BoW + 만든 공시성 시나리오. 실제 실행 중 질의를 가로챈 대조는 안 함.
- 강의실 녹화(ds_1118)는 BoW 단계만; Sim3 · 고유 풀이 대조는 TUM fr1_room 실행 하나.

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

#### 지도 자료 메서드(KeyFrame.cc · MapPoint.cc · Map.cc)

모듈 `slam/src/orb3rs/mapdata.rs`(통합 타입 위, 이미 옮겨진 `MapPoint::add_observation` · `compute_distinctive_descriptors` · `predict_scale` 은 그대로 씀):

- KeyFrame: `kf_update_connections`(공시성 세기 · 15 문턱 · 최대 하나 · 처음 연결 부모) · `kf_update_best_covisibles` · `kf_add_connection` · `kf_erase_connection` · `kf_set_bad_flag`(연결 끊기 · 점 관측 지우기 · 신장 트리 다시 잇기 · mTcp · 지도에서 빼기, KeyFrameDatabase 빼기는 true 를 받은 쪽) · `kf_set_erase` · `kf_change_parent` · `kf_erase_map_point_match` · `kf_tracked_map_points`, 자기 칸만 보는 `weight` · `best_covisibility` · `covisibles_by_weight` · `vector_covisible` · `add/erase_child` · `add_loop_edge` · `add_merge_edge` · `add_map_point` 등.
- MapPoint: `mp_erase_observation`(관측 수 · 새 기준 kf · 2 이하면 나쁨) · `mp_set_bad_flag` · `mp_replace`(+ 지도에서 빼기) · `mp_update_normal_and_depth`, `found_ratio` · `index_in_keyframe` 등.
- Map: `add_keyframe`(첫 kf · 최대 · 최저 id) · `erase_keyframe` · `add/erase_map_point` · `set_reference_map_points` · `inform_new_big_change` · `increase_change_index` 등. kfs · mps 는 id 순 정렬(std::set 대신).
- 겹치는 옛 판: `orb3cv::atlas_io::mp_erase_observation/mp_set_bad`(CV) · `orb3rs::localmap::mp_set_bad/mp_erase_observation/kf_set_bad`(TRACK, nObs 를 TrackExt 에 둠) · `matcher::replace_map_point`(지도에서 빼기 없음)는 이 모듈과 같은 일을 함 — 통합 때 이것으로 모으면 됨.

순서: 원본 `std::map/set<KeyFrame*>` 은 **주소 순**이고 실제 실행에서 주소 순 ≠ id 순이 흔함(강의실 · TUM 모두, 기록의 대부분). 결과가 순서에 닿는 자리 — `UpdateBestCovisibles` · `UpdateConnections` 의 무게 동률(pair 비교가 포인터로 갈림) · 최대 개수 동률, `SetBadFlag` 자식 동률, `EraseObservation` 의 새 기준 kf, `ComputeDistinctiveDescriptors` 의 중앙값 동률, **`UpdateNormalAndDepth` 의 법선 합(실수 더하기 순서라 1 ulp 갈림)** — 에서 Rust(id 순)는 주소 순 원본과 다를 수 있다. 원본 자체가 실행마다 다르므로 포팅판은 id 순(결정적)으로 두고, 시험은 기록마다 주소 순위로 id 를 바꿔(같음 비교만 쓰는 id 라 의미가 같음) 원본 순서를 되살려 대조한다.

하네스: `slam/cpp/orb3_mapdump.cc` + `orb3_mapdump.sh`(build · run). 기준 빌드 원본(`~/opt/orb3_ref`)을 RGB-D 로 돌리며 바꾸는 메서드 30 개를 PLT 로 가로채 호출 앞뒤 상태(지도 칸 · 관련 키프레임 그래프 · 점 칸 · 지도 점 · 키프레임별 층 · 오른쪽 좌표 · 기술자 · 주소)를 덤프. 바꾸는 메서드와 SetPose · SetWorldPos · IncreaseVisible/Found 는 전역 재귀 잠금 하나 아래서 돌아 덤프 중 다른 스레드가 상태를 못 바꿈. 같은 오브젝트 안 호출(AddConnection · EraseConnection · ChangeParent 등을 KeyFrame.cc 안에서 부르는 것)은 PLT 를 안 거쳐 바깥 호출 안에 포함돼 확인됨. 시험 `ORBMAP_DUMPS=a.bin:b.bin cargo test --release -p slam --lib orb3rs::mapdata`(입력 없으면 실패).

| 데이터 | 기록(호출) | 비트 같음 | 주소 순위로 바꿔 대조 |
|---|---|---|---|
| 강의실 ds_1118_121652 전체 3,807 프레임(루프 · 지도 합치기 포함) | UpdateConnections 738 · KF SetBadFlag 147(실제 지움 147) · SetErase 200 · UpdateBestCovisibles 741 · ChangeParent 10 · EraseChild 6 · Loop/MergeEdge 6 · SetNotErase 1,205 · EraseObservation 3,000 · MP SetBadFlag 3,000 · Replace 2,000 · ComputeDistinctiveDescriptors 3,000 · UpdateNormalAndDepth 3,000 · EraseMapPointMatch 3,000 · AddObservation 3,000 · KF AddMapPoint 3,000 · PredictScale 40,000 · Map Add/EraseKeyFrame 1,346/131 · Add/EraseMapPoint 3,000/2,609 · 큰 변화 7 · 변화 번호 542 · 기준 점 1,940 | 전부 | 대부분(예: UpdateConnections 726, UpdateNormalAndDepth 1,450) |
| TUM fr1_room 1,362 프레임(루프 2) | 같은 종류 28,000 여(SetBadFlag 는 미룸만) | 전부 | 대부분 |
| TUM fr1_desk2 640 프레임 | 같은 종류 | 전부 | 대부분 |

비트 맞추기에서 알게 된 것: `UpdateNormalAndDepth` 는 노름 (y²+z²)+x²(= a0+(a1+a2) 와 같은 값) · 칸마다 나눗셈 · `normal/n` 도 나눗셈(역수 곱 아님, 역어셈블 확인). `PredictScale` 은 float `logf`. `UpdateConnections` 는 공시성 표에 15 미만도 다 넣고(정렬 목록에는 문턱 이상만), 자기 정렬 목록은 나쁜 kf 를 거르지 않음(남의 것은 UpdateBestCovisibles 가 거름). 원본 `SetBadFlag` 는 부모 후보에 못 붙은 자식을 원래 부모로 옮긴 뒤에도 자기 자식 목록에서 지우지 않음 — 그대로 옮김.

확인 안 한 것: 오른쪽 카메라 관측(RGB-D 에 없음, 만나면 assert). 덤프 실행은 원본 교착(루프 뒤 지도 스레드 멈춤)이 가끔 걸려 한 번 다시 돌림. `ComputeSceneMedianDepth`(단안 전용)는 옮기지 않음.

### 포터 CV (11-25)

브랜치 `opt/orb3-rs-cv`(`opt/orb3-rust` 위). 하네스 WSL `~/opt/orb3_cv/`: `ref.cc`(OpenCV 4.6 시스템 판, 입력 · 출력 덤프) · `svd_ref.cc` · `set_ref.cc`(기준 빌드 `cxx.sh` 로 원본 라이브러리 링크), Rust 쪽 `rs/`(`#[path]` 로 `slam/src/orb3cv/*.rs` 를 그대로 끌어와 비교: `cargo run --release -- ../dump` · `--bin svd -- ../svd.bin` · `--bin settings -- ../yaml`). 덤프 · 설정이 없으면 실패.

| 모듈 | 원본 | 비교 | 결과 |
|---|---|---|---|
| `orb3cv::img::cvt_gray` | cvtColor RGB/BGR/RGBA/BGRA2GRAY(15 비트 고정소수) | TUM fr1_desk · desk2 18 장 × 4 배치, 2,211 만 칸 | **비트 동일** |
| `orb3cv::img::depth_to_f32` | `convertTo(CV_32F, 1/DepthMapFactor)` | 같은 18 장 553 만 칸 | **비트 동일** |
| `orb3cv::img::copy_make_border_reflect101` | copyMakeBorder BORDER_REFLECT_101(19 칸 · 영상보다 넓은 테두리) | 632 만 + 2.5 만 칸 | **비트 동일** |
| `orb3cv::img::undistort_points` | undistortPoints(K, 왜곡 5 · 8 계수, P = K) — Frame::UndistortKeyPoints | 점 14.3 만 × 2 | **비트 동일** |
| `orb3cv::svd::jacobi_svd` | Eigen 3.4.0 JacobiSVD(정사각 f32, V · U) | 4×4 20 만(무작위 · 크기 10^±4 섞임 · 계수 부족 · 0 · 같은 행) · 3×3 U·V 10 만 | **비트 동일**(특이값 · V · U · 0 아닌 수) |
| `orb3cv::svd::triangulate` | GeometricTools::Triangulate | 실제 같은 두 시점 배치 20 만 | **비트 동일** |
| `orb3cv::settings` | Settings(RGB-D 핀홀) + Tracking::newParameterLoader · Frame 파생 값 | 예제 설정 4 개 + 무작위 40 개(newWidth · 왜곡 · thFarPoints 섞음) | **비트 동일** |

- RGB-D 는 `needToUndistort` 가 단안에서만 참이라 mDistCoef = 0 → undistortPoints 는 실제로 불리지 않음(옮겨는 둠). copyMakeBorder 도 ORB 추출이 테두리를 읽지 않아 쓰는 곳 없음. 남은 OpenCV 쓰임(RGB-D 경로)은 이것으로 끝: cv::resize(System, newWidth 가 있을 때만 — `orb3rs::extract` 의 resize 와 같은 함수) · cv::Mat 그릇.
- SVD · Rodrigues: 원본 RGB-D 경로에 OpenCV SVD · Rodrigues 는 없음(전부 Eigen). Eigen JacobiSVD 는 LocalMapping 삼각측량(4×4)에만 — 옮김. 단안 초기화 · IMU 쪽 JacobiSVD(TwoViewReconstruction · ImuTypes)는 안 씀.
- **고친 것**: `orb3rs::frame::Camera::pinhole` 의 `th_depth = bf·ThDepth/fx`(ORB-SLAM2 식)는 원본 3 판(Settings 경로) `mThDepth = b·ThDepth` 와 마지막 비트가 다름(무작위 설정 36 개 중 12 개). 인자를 설정 `b` 로 받게 바꿈(쓰는 곳 없었음). 설정에서 바로: `OrbSettings::parse(..)?.tracking().camera(w, h)`.
- 화면 제거판 · 원본 TUM 실행(각 3 번, `~/opt/orb3_cv_bg/tum.txt`, ATE m): desk 원본 0.0192 · **멈춤** · 0.0168, 제거판 **멈춤** · 0.0223 · 0.0166 / room 원본 0.0660 · 0.0756 · 0.0739, 제거판 0.0751 · 0.0744 · 0.0632. 900 초 멈춤은 원본(Pangolin 있음)에도 같은 꼴로 나옴(로그 끝 `*Loop detected` → `Local Mapping STOP` 뒤 진행 없음) → 화면 제거 탓 아님, 원본의 루프 닫기 · 지도 스레드 멈춤. 실행마다 흔들림은 `opt/pose-parb` 의 결정성 작업 몫.
- Atlas 저장 · 불러오기 `orb3cv::atlas_io`: Boost 형식은 읽을 쪽이 없어 우리 이진 형식으로. 남기는 칸 · 버리는 칸은 원본과 같게 — 저장은 id 순 지도마다 나쁘지 않은 키프레임 · 점, 다른 지도 · 나쁜 것을 가리키는 연결 · 자식 · 가장자리 · 점 번호는 빼거나 −1, 키프레임 없는 지도는 버림 / 불러오기는 MapPoint mnVisible · mnFound = 1 · 추적 임시 칸 0, ordered_conn = UpdateBestCovisibles, 그 뒤 새 빈 지도를 현재로(System). 원본 Map::PreSave 의 EraseObservation(살아 있는 지도를 바꿈)도 Rust 로 옮겨 System 저장이 먼저 부름(cd4e91f). 시험: 무작위 Atlas 저장 → 불러오기 → 다시 저장이 같은 바이트, 저장 칸 비트 같음.
- System 진입 `orb3cv::system::SystemInit::open(설정, w, h)`: OrbSettings → TrackingParams → Camera · Pyramid(ORBextractor 층 표) · FrameCtx(ComputeImageBounds), System.LoadAtlasFromFile / SaveAtlasToFile(".osa" 붙임), GrabImageRGBD 의 회색 · 깊이 변환. newWidth(영상 줄이기)는 안 씀이라 Err.
- `orb3rs::frame::depth_to_float` 를 원본 식(1.0f / 값, |값| < 1e-5 면 1)으로(나눗셈 결과는 전과 같음: f64 거쳐 f32 로 바꿔도 이중 반올림 없음).

- **PreSave EraseObservation**(`atlas_io::pre_save` · `mp_erase_observation` · `mp_set_bad`): 원본 Map::PreSave 첫 고리 — id 순 지도마다 나쁘지 않은 점의 다른 지도 · 나쁜 키프레임 관측을 지우며 nObs(오른쪽 좌표 있으면 2) 줄이기, 기준 키프레임을 남은 관측 첫째로(원본은 주소 순 첫째 — 실행마다 다를 수 있음, 관측이 다 없어지면 원본은 미정의 동작이라 그대로 둠), nObs ≤ 2 면 SetBadFlag(관측 키프레임의 그 번호 비우기 · 지도에서 빼기). 키프레임 없는 지도는 나쁨 표시. `SystemInit::save_atlas` 가 먼저 부름. 시험 `pre_save_erases_foreign_obs`.
- **g2o::Sim3**(`orb3cv::g2osim3`, double): 회전 행렬에서 · exp(7 벡터) · log(Eigen PartialPivLU 3×3 풀이 포함) · 역 · 곱 · 사상. 기준 빌드와 같은 플래그로 헤더를 컴파일한 probe(`~/opt/orb3_cv/sim3/probe.cc` → sim3.bin, `rs --bin sim3 -- ../sim3/sim3.bin`) 20 만 건 × 10 연산 **비트 동일**(작은 각 · 작은 σ 갈래 섞음). 알게 된 순서: Quaterniond(Matrix3d) 대각 합 (R22+R11)+R00, Matrix3d×Vector3d 행 0 · 1 은 pmadd FMA 사슬 · 행 2 는 스칼라 (a21v1+a22v2)+a20v0, log 의 `B*Omega*Omega` 는 (B·Ω)·Ω, 삼각 풀이는 완전 펼침(행마다 내적). 한계: probe 는 따로 컴파일한 함수라 LoopClosing.cc 안에 인라인될 때 순서가 다르면 그 자리에서 다시 대조해야 함.
- **Orb3System 연결**: `orb3cv::system::keyframes` · `kf_now`(orb3.cc 의 orb3_keyframes · orb3_kf_now 를 Rust Atlas 로 — 지도 · 키프레임 id 순, 나쁜 키프레임은 mTcp 곱하며 부모로, 부모 고리면 None). 추적 · 지도 · 루프 본체가 다 옮겨지면 Rust 판 `Orb3System` 이 이것과 `SystemInit` 을 씀.
- 루프 ↔ 지도 멈춤 원인(코드 읽기): 원본 LocalMapping 멈춤은 불 하나를 CorrectLoop · 전체 최적화 지도 갱신 · MergeLocal · GraftMap 이 같이 쓰고 Release 가 남의 요청까지 지움 + 전체 최적화 스레드가 갱신 내내 mMutexGBA 를 쥔 채 지역 지도 멈춤 · 지도 잠금을 기다림. 로그 끝 `*Loop detected → STOP → RELEASE → STOP`(전체 최적화의 STOP 뒤 RELEASE 없음). desk 단독 실행 18 번은 재현 안 됨(부하 겹칠 때 나옴) → gdb 역추적은 못 남김. 스레드 · 잠금 Rust 판은 통합 담당 `orb3sys::threads`(멈춤 토큰 개수) 기준.

#### Sim3 최적화 · 본질 그래프(`orb3cv::sim3opt`, BA 와 나눔: 이 두 함수는 CV 몫)

| 원본 | Rust | 비교(기준 빌드 덤프, `~/orbs3_ref`) | 결과 |
|---|---|---|---|
| Optimizer::OptimizeSim3 — 그래프 만들기 · Levenberg 5 번 · χ² > th2 이상치 지우기 · 강건 커널 떼고 5/10 번 · nIn | `build_sim3_problem` · `optimize_sim3` | TUM room ×3 · 360 ×3 · ETH3D large_loop_1 ×2 의 실제 호출 37 개(그래프 만들기 입력까지 덤프한 것 8 개) — 첫 최적화 S12 · 간선 χ² · 반복 수 · nIn · 끝 S12 · 지운 짝 | **비트 동일** |
| 첫 반복 단계(H 7×7 · b · λ 넣어 푼 x) | `sim3_first_stage` | room 4 호출(`s3stage stage`) | **비트 동일** |
| g2o LinearSolverDense = Eigen 3.4 LDLT<MatrixXd>(대각 피벗) | `dense_ldlt_solve` | 무작위 7×7 2 만 건(0 열 · 음 · 0 행렬 섞음) | **비트 동일** |
| OptimizeEssentialGraph(루프 판) — 그래프 만들기(LoopConnections · 신장 트리 · 루프 · 공시성 ≥100 간선) · Levenberg 20 번(λ₀ 1e-16, SimplicialLDLT = `orb3ba::ldlt`) · Tiw(SE3f) · 점 고치기 | `build_eg_loop` · `optimize_essential_graph` · `sim3_to_se3f` · `eg_loop_points` | 실제 루프 9 번(정점 118–293 · 간선 599–2184) 끝 정점 · χ² · 반복 수, 키프레임 자세 1,870, 점 13.3 만 + 첫 반복 단계(오차 · 간선마다 야코비안 · Hpp 블록 2,455 · b · x) | **비트 동일** |
| OptimizeEssentialGraph(합치기 판) | 최적화 핵심은 같은 `optimize_essential_graph`(그래프 만들기 · 마무리는 남음) | 합치기가 나는 입력을 못 만듦(TUM · ETH3D · slamint 끊김 시퀀스에서 합치기 0 번) | 미검증 |

- 쓰는 법: OptimizeSim3 는 `Sim3Kf`(Tcw 회전 행 우선 · 이동, mvKeysUn, mvInvLevelSigma2, 카메라) 둘 + `Sim3Match`(vpMatches1 있는 자리) → `build_sim3_problem` → `optimize_sim3`(결과 nIn · S12 · 짝마다 지움). 본질 그래프는 `EgLoopInput`(키프레임을 원본 vpKFs 순서로 · 루프 · 자식 · 공시성 목록 순서 그대로) → `build_eg_loop` → `optimize_essential_graph(&mut g.vertices, &g.edges, 20)` → `sim3_to_se3f` · `eg_loop_points`. 원본 순서가 std::set 주소 순인 자리(vpKFs · GetLoopEdges · LoopConnections)는 호출자가 준 순서를 따름 — Rust Atlas 는 id 순으로 준다.
- 맞추며 알게 된 것: 고정 7×7 에 작은 곱을 더하는 대입은 모든 열에서 0..6 행이 Packet2d 묶음(FMA 사슬) · 6 행 낱개(반 나눔 합). 7 길이 연속 내적은 ((m0+(m2+m4))+(m1+(m3+m5)))+m6. 전치 왼쪽(행 우선) 곱은 전부 낱개. LDLT GEMV 는 앞 4 · 다음 2 행 묶음(AVX-512 반 · 4 분의 1 묶음). VertexSim3Expmap::oplus 가 고정 축척일 때 풀이 벡터 x[6] 을 0 으로 바꿔 computeScale 에도 들어감. OptimizeSim3 의 KF2 밖 점은 `cv::KeyPoint(Point2f, mnTrackScaleLevel)` 둘째 인자가 size 라 octave 0 의 정보 행렬을 씀(원본 그대로). 실패한 선형 풀이 · 미정의 갈래(풀이 실패 시 이전 x)는 덤프에 없음.
- BA 와의 경계: `orb3ba::ldlt::Ldlt`(analyze · factorize · solve) 만 가져다 씀. BA 쪽 "남음" 표의 OptimizeEssentialGraph · OptimizeSim3 는 여기서 끝남(합치기 판 그래프 만들기만 남음). GlobalBA · 합치기용 BA 는 BA 몫.
- 하네스(저장소 밖 WSL `~/opt/orb3_cv/s3opt`): `mkpatch.py`+`mkpatch_add.py` 가 원본 Optimizer.cc 에 덤프를 끼워 `libs3dump.so`(build.sh, 기준 빌드에 링크), `run2.sh 이름 폴더 설정`(`ORB3_S3_DUMP`), `egstage`(본질 그래프 다시 풀기 · 첫 반복 단계), `s3stage ldlt|stage`. Rust 비교: `~/opt/orb3_cv/rs` 의 `--bin s3opt 덤프들…` · `--bin egst` · `--bin s3st`. 덤프가 없으면 실패.

## 포터 BA

### 상태

| 부분 | Rust | 비트 비교 |
|---|---|---|
| g2o 경로: SparseOptimizer · OptimizationAlgorithmLevenberg(Raul 멈춤 조건 · 시도 10 번) · BlockSolver<6,3> 슈어 소거 · LinearSolverEigen(SimplicialLDLT + AMD) | `slam::orb3ba::{solver, ldlt}` | **비트 동일**: 실제 LBA 호출 33 개(강의실 ds_1118_121652 9 · TUM desk2 12 · room 12) 전부 끝 정점 값 · 간선 χ² · 반복 수 같음 |
| 간선: EdgeSE3ProjectXYZ(단안) · EdgeStereoSE3ProjectXYZ · 우리 EdgeGravityUp(수치 미분) — 오차 · 야코비안 · 2차 형식 | `orb3ba::edges` | 첫 반복 단계 비교(오차 · χ² · 간선마다 야코비안 · 누적 블록 35 만 값) 전부 같음 |
| SE3Quat exp · oplus · 사원수 곱 · 회전 | `orb3ba::se3` | 무작위 3 만 건(작은 · 중간 · 큰 회전) 전부 같음 |
| 커널 없는 간선 경로(전역 BA: bRobust=false) — Hpl 이 Bᵀ·(AᵀΩ)ᵀ 순서 | `orb3ba::edges` | 강의실 9 호출을 커널 빼고 원본 g2o 로 다시 푼 결과(`NOROBUST=1 replay`)와 전부 같음 |
| g2o::Sim3 exp · log(PartialPivLU 3×3 풀이 포함) · 곱 · 역 · map | `orb3cv::g2osim3`(한 벌) | 무작위 4 만 건(회전 · 척도 0 · 아주 작게 · 작게 · 크게) 전부 같음 — CV 포터가 가져다 쓸 것 |
| LocalBundleAdjustment 그래프 만들기 · 이상치 정리 · 되써 넣기 | `orb3ba::lba`(통합 담당) | **비트 동일** 37 호출(통합 상태 절) |
| GlobalBundleAdjustemnt(루프 뒤 전역 BA, 커널 없음) 의 g2o 부분 | 같은 `optimize` | **비트 동일**: TUM room 전체(루프 검출 뒤) 실제 GBA 1 호출 — 정점 19,860 · 간선 93,821 · 반복 9, 끝 값 · χ² 전부 같음 |
| 합치기용 LBA(MergeLocal: 5 번 Huber → 이상치 level 1 · 커널 뺌 → 10 번) 의 g2o 부분 | 같은 `optimize`(두 번째는 level 0 간선만 넣고 delta=None) | 덤프 장치는 있음(마술 수 0x4142474d), 지도 합치기가 일어난 녹화가 아직 없어 미비교 |
| OptimizeEssentialGraph · OptimizeSim3 | **CV 포터(`opt/orb3-rs-cv`) 몫** | |
| LocalBundleAdjustment 그래프 만들기 · 이상치 정리 · 되써 넣기 | 남음(공통 자료 구조 `orb3rs::types` 받은 뒤) | |
| GlobalBundleAdjustemnt · OptimizeEssentialGraph(Sim3, BlockSolver<7,3>) · OptimizeSim3(BlockSolverX + 밀집) · 합치기용 BA | 남음 | |

### 쓰는 법

`orb3ba::Problem { vertices, edges, lambda_init }` 에 정점(자세 SE3Quat [x,y,z,w]+t · 점, 고정 여부, id) · 간선(원본 addEdge 순서, Huber δ)을 넣고 `orb3ba::optimize(&p, 10)` → 정점 값 · 간선마다 마지막 χ²(원본 `e->chi2()` 와 같은 값: 마지막으로 계산한 오차, 되돌린 시도의 것일 수 있음) · 반복 수. 원본 `pbStopFlag` 는 아직 받지 않음(덤프의 호출은 모두 멈춤 없이 끝남).

### 순서 규칙(원본과 같게)

- 활성 정점 = 활성 간선에 걸린 정점, id 순. 헤세 번호 = 고정 아닌 자세 먼저, 다음 점(주변화). 모든 정점이 고정인 간선은 빠짐.
- 간선은 추가 순서. Hpl 블록 · 슈어 블록 패턴은 buildStructure 와 같게, 슈어 행렬은 블록 열 · 블록 안 열 · 블록 행 순(fillCCS 위 삼각)으로 LDLT 에 넣음. AMD 순서는 첫 풀이 한 번(analyzePattern), 이후 수치 분해만.
- λ 초기값 = 1e-5 · 최대 |대각|(자세 다음 점), 관성 지도면 100(setUserLambdaInit).

### 부동소수 규칙(기준 빌드 = -ffp-contract=off)

- 컴파일러 축약은 없지만 Eigen 이 묶음 곱(`pmadd`)에 vfmadd 내장 함수를 직접 쓴다 → 고정 크기 행렬 곱 · 행렬×벡터에서 **묶음으로 계산되는 행만** 한 번 반올림(`se3::fused`): 3 행 결과면 0 · 1 행(2 행은 낱개, 합은 m0 + (m1 + m2)), 6 행 결과면 전부. 내적(`Aᵀ·x` 6 길이)은 SSE 2 묶음 세 개의 합 ((m2+m4)+m0) + ((m3+m5)+m1). 원소별 식 · 스칼라 식 · LDLT 는 소스 순서 그대로.
- 실측으로 확인한 자리: 2 차 형식(AᵀWA, AᵀWB, Aᵀω), 단안 야코비안(−projectJac·R, −projectJac·SE3deriv), 슈어(Bi·D⁻¹, BDinv·Bjᵀ, Bi·(D⁻¹b), D⁻¹·cl), exp 의 Ω² · V·υ. 단 χ² 의 Ω·e 는 축약 없음(그대로 맞음).
- 이전 포터의 FMA 빌드(자리마다 GCC 축약을 따라 함) 판은 이 기준으로 바꿨다(기계어 읽기 불필요).

### 하네스(저장소 밖, WSL `~/opt/orb3_ba_bg`)

- `build.sh`: 기준 빌드(`~/opt/orb3_ref/ORB_SLAM3`)에 링크, `mkpatch.py` 가 원본 Optimizer.cc 의 LBA 에 덤프를 끼운 `libbadump.so`(먼저 링크해 원본 함수를 가림) · `ds_run`(녹화 rgb.txt · depth.txt 로 원본 RGB-D, 매 장면 지도 스레드 기다림, imu 있으면 위쪽 벡터) · `replay`(덤프를 원본 g2o 로 다시 풀기 · 한 호출의 첫 반복 단계 덤프).
- 덤프(`mkpatch2.py`: LBA 0x4142414c · 전역 BA 0x41424247 · 합치기 LBA 0x4142474d, `ORB3_BA_ONLY=16진 마술 수` 로 한 종류만, 간선은 level 0 만): `ORB3_BA_DUMP=파일 ORB3_BA_MAX=개수 ./ds_run 단어장 설정 녹화폴더 장수 [imu.csv]`. 지금 쓰는 것: `~/orbba_ref/ds_lba.bin`(강의실 700 장, 4.6 MB), `tum_desk2.bin` · `tum_room.bin`(`tum.sh`, 앞 250 장), `room_gba.bin`(room 전체, 전역 BA 만, 16 MB). 큰 덤프 읽을 땐 `ulimit -c 0`.
- 시험(`t.sh 이름 [호출]`, 입력 없으면 실패): `full_calls`(ORB3BA_DUMP 의 모든 호출을 끝까지), `stage_first_iteration` · `stage_solve`(ORB3BA_STAGE = `replay 덤프 단계파일 호출` 출력, `st.sh 호출` 이 둘 다), `se3_exp_oplus`(ORB3BA_SE3 = `se3/se3ref` 출력).
- 원본 replay 자체가 현장 덤프와 χ² 1 개씩 다른 호출이 2 개 있음(정점은 같음) — Rust 는 현장 값과 같다.

### 공통 솔버 인터페이스(다른 포터용)

- `orb3ba::Problem { vertices: Vec<Vertex{id, fixed, est: Est::Pose{q,t} | Est::Point(p)}>, edges: Vec<Edge{kind, v0(점 또는 단항 자세), v1(자세), meas, info(열 우선), delta: Option<f64>}>, lambda_init }` → `orb3ba::optimize(&p, iters) -> Outcome { est, chi2, iterations }`.
- 낮은 층: `solver::Sys`(오차 · 선형화 · `build_system`), `Sys::schur_pattern` · `Sys::solve(&mut Schur, λ)`(슈어 + LDLT, 6 차 자세 · 3 차 점), `ldlt::Ldlt::{analyze, factorize, solve}`(차수 무관, 위 삼각 CCS 입력) — 7 차(Sim3) 블록 솔버는 이 LDLT 를 그대로 쓰면 된다.
- `orb3cv::g2osim3::G2oSim3 { r, t, s }`: `exp` · `log` · `mul` · `inverse` · `map`. VertexSim3Expmap::oplus 는 `_fix_scale` 이면 **풀이 벡터 x 의 7 번째 칸을 제자리에서 0 으로 바꾼다**(그 뒤 computeScale 이 바뀐 x 를 씀).
- 원본 χ² 비교 때 비활성 간선(정점 모두 고정)의 값은 초기화 안 된 메모리라 비교에서 뺀다.
- 덤프: `ORB3_BA_DUMP=파일 ORB3_BA_MAX=개수 ./ds_run 단어장 설정 녹화폴더 장수 [imu.csv]`. 지금 쓰는 것: `~/orbba_ref/ds_lba.bin`(강의실 700 장, 4.6 MB), `tum_desk2.bin` · `tum_room.bin`(`tum.sh`, 앞 250 장).
- 시험(`t.sh 이름 [호출]`, 입력 없으면 실패): `full_calls`(ORB3BA_DUMP 의 모든 호출을 끝까지), `stage_first_iteration` · `stage_solve`(ORB3BA_STAGE = `replay 덤프 단계파일 호출` 출력, `st.sh 호출` 이 둘 다), `se3_exp_oplus`(ORB3BA_SE3 = `se3/se3ref` 출력).
- 원본 replay 자체가 현장 덤프와 χ² 1 개씩 다른 호출이 2 개 있음(정점은 같음) — Rust 는 현장 값과 같다.

#### 전역 BA 전체(`orb3ba::gba`, 11-25)

- `global_bundle_adjustment(atlas, ext, map, n_iter, stop, loop_kf, robust) -> GbaOut` (= GlobalBundleAdjustemnt → BundleAdjustment, RGB-D). 키프레임 · 점은 번호 순(원본 std::set 자리), 관측은 `ObsOrder`(기본 번호 순).
  - 정점: 나쁘지 않은 키프레임(첫 키프레임만 고정), 점 id = 점 번호 + 최대 키프레임 번호 + 1, 관측이 하나도 안 걸린 점은 뺌. 왼쪽 칸 -1 관측은 간선 없이 nEdges 만 셈. Huber δ = √5.99 · √7.815(float), robust 거짓이면 커널 없음.
  - 되써 넣기: loop_kf 가 지도 원점 키프레임이면 바로(자세 · 점 · UpdateNormalAndDepth), 아니면 `GbaOut { kf_gba: mTcwGBA, mp_gba: mPosGBA }` 를 돌려줌(mnBAGlobalForKF = loop_kf, 옮기기는 루프 쪽). mTcwGBA = SE3d(q,t).cast<float>(): double 정규화 → float → 다시 정규화. 원본의 dist > 1 갈래는 세기만 해서 뺌.
- **비트 동일**: TUM room 전체, 루프 뒤 실제 전역 BA 1 호출(키프레임 279 · 점 19,155 · 간선 90,457 · 반복 8) — 그래프(정점 · 간선 순서 · 값) · 반복 수 · 모든 키프레임 mTcwGBA · 모든 점 mPosGBA 같음. 바로 쓰기 갈래(loop_kf = 원점)는 실제 입력이 없어 미비교(지역 BA 와 같은 SetPose · UpdateNormalAndDepth 를 씀).
- 하네스 `~/opt/orb3_gba`(`gpatch.py` · `gbaw.h` · `build.sh`): 원본 BundleAdjustment 에 지도 상태 앞 · 뒤 덤프. **원본은 전역 BA 그래프를 만드는 동안 지도 스레드가 관측을 계속 바꾼다**(원본 자체 경합) → 하네스는 함수 동안 mMutexMapUpdate 를 쥐어 추적을 막고, 그래프가 실제로 읽은 점 위치 · 관측을 점마다 따로 기록(`used`)해 그것을 입력으로 씀. 실행: `ORB3_BA_ONLY=41424247 ORB3_BA_DUMP=~/orbba_ref/room_gba2.bin ORB3_GBA_WORLD=~/orbba_ref/room_gba2_w.bin ORB3_BA_MAX=4 ./ds_run 단어장 TUM1.yaml room 2000`, 시험 `ORB3GBA=~/orbba_ref/room_gba2 cargo test --release -p slam --lib orb3ba::gba_tests`.

#### CV 포터가 쓸 것(Sim3 · 7 차 블록)

- `orb3ba::sim3::Sim3 { q, t, s }` — `exp(&[f64;7])` · `log()` · `mul` · `inverse` · `map`(기준 빌드와 무작위 4 만 건 비트 동일).
- `orb3ba::ldlt::Ldlt` — `analyze(n, ap, ai)`(위 삼각 CCS 패턴, 첫 풀이 때 한 번: AMD 순서 · 소거 나무) → `factorize(&values) -> bool`(대각 0 이면 false) → `solve(&b)`. 차수 무관, 7 차 블록도 fillCCS 순서(블록 열 · 블록 안 열 · 블록 행 · 블록 안 행, 대각 블록은 위 삼각만)로 넣으면 됨.
- Levenberg 틀: `solver::optimize_stop` 의 반복부 — λ 초기값(user 값 또는 1e-5 · 최대 |대각|), 시도마다 push → setLambda → solve → update → restoreDiagonal → 오차 → ρ = (χ²₀ − χ²)/(Σ x(λx + b) + 1e-3), 좋으면 λ *= max(1/3, min(2/3, 1 − (2ρ − 1)³)) · ni = 2, 나쁘면 λ *= ni · ni *= 2 · pop, 시도 10 번, (ini − cur)·1e3 < ini 가 3 번 연속이면 멈춤. 7 차 판은 이 순서를 그대로 쓰고 블록 솔버만 바꿈(지금 코드는 6 · 3 고정 — 일반화가 필요하면 BA 에 요청).
