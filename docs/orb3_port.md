# ORB-SLAM3 → Rust (BA 몫)

공통 원칙 · 인터페이스는 `opt/orb3-rust` 의 같은 파일. 이 브랜치(`opt/orb3-rs-ba`)는 아래 "포터 BA" 절만 쓴다 — 통합 때 그 절을 옮겨 붙인다.

## 포터 BA

### 상태

| 부분 | Rust | 비트 비교 |
|---|---|---|
| g2o 경로: SparseOptimizer · OptimizationAlgorithmLevenberg(Raul 멈춤 조건 · 시도 10 번) · BlockSolver<6,3> 슈어 소거 · LinearSolverEigen(SimplicialLDLT + AMD) | `slam::orb3ba::{solver, ldlt}` | **비트 동일**: 실제 LBA 호출 33 개(강의실 ds_1118_121652 9 · TUM desk2 12 · room 12) 전부 끝 정점 값 · 간선 χ² · 반복 수 같음 |
| 간선: EdgeSE3ProjectXYZ(단안) · EdgeStereoSE3ProjectXYZ · 우리 EdgeGravityUp(수치 미분) — 오차 · 야코비안 · 2차 형식 | `orb3ba::edges` | 첫 반복 단계 비교(오차 · χ² · 간선마다 야코비안 · 누적 블록 35 만 값) 전부 같음 |
| SE3Quat exp · oplus · 사원수 곱 · 회전 | `orb3ba::se3` | 무작위 3 만 건(작은 · 중간 · 큰 회전) 전부 같음 |
| 커널 없는 간선 경로(전역 BA: bRobust=false) — Hpl 이 Bᵀ·(AᵀΩ)ᵀ 순서 | `orb3ba::edges` | 강의실 9 호출을 커널 빼고 원본 g2o 로 다시 푼 결과(`NOROBUST=1 replay`)와 전부 같음 |
| g2o::Sim3 exp · log(PartialPivLU 3×3 풀이 포함) · 곱 · 역 · map | `orb3ba::sim3` | 무작위 4 만 건(회전 · 척도 0 · 아주 작게 · 작게 · 크게) 전부 같음 — CV 포터가 가져다 쓸 것 |
| LocalBundleAdjustment 그래프 만들기 · 이상치 정리 · 되써 넣기 | 남음(공통 자료 구조 `orb3rs::types` 받은 뒤) | |
| GlobalBundleAdjustemnt · 합치기용 LBA(MergeLocal) 의 g2o 부분 | 같은 `optimize` 로 풀림(같은 간선 · 블록 솔버). 실제 덤프 비교 진행 중 | |
| OptimizeEssentialGraph · OptimizeSim3 | **CV 포터(`opt/orb3-rs-cv`) 몫** | |

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
- 덤프: `ORB3_BA_DUMP=파일 ORB3_BA_MAX=개수 ./ds_run 단어장 설정 녹화폴더 장수 [imu.csv]`. 지금 쓰는 것: `~/orbba_ref/ds_lba.bin`(강의실 700 장, 4.6 MB), `tum_desk2.bin` · `tum_room.bin`(`tum.sh`, 앞 250 장).
- 시험(`t.sh 이름 [호출]`, 입력 없으면 실패): `full_calls`(ORB3BA_DUMP 의 모든 호출을 끝까지), `stage_first_iteration` · `stage_solve`(ORB3BA_STAGE = `replay 덤프 단계파일 호출` 출력, `st.sh 호출` 이 둘 다), `se3_exp_oplus`(ORB3BA_SE3 = `se3/se3ref` 출력).
- 원본 replay 자체가 현장 덤프와 χ² 1 개씩 다른 호출이 2 개 있음(정점은 같음) — Rust 는 현장 값과 같다.

### 공통 솔버 인터페이스(다른 포터용)

- `orb3ba::Problem { vertices: Vec<Vertex{id, fixed, est: Est::Pose{q,t} | Est::Point(p)}>, edges: Vec<Edge{kind, v0(점 또는 단항 자세), v1(자세), meas, info(열 우선), delta: Option<f64>}>, lambda_init }` → `orb3ba::optimize(&p, iters) -> Outcome { est, chi2, iterations }`.
- 낮은 층: `solver::Sys`(오차 · 선형화 · `build_system`), `Sys::schur_pattern` · `Sys::solve(&mut Schur, λ)`(슈어 + LDLT, 6 차 자세 · 3 차 점), `ldlt::Ldlt::{analyze, factorize, solve}`(차수 무관, 위 삼각 CCS 입력) — 7 차(Sim3) 블록 솔버는 이 LDLT 를 그대로 쓰면 된다.
- `orb3ba::sim3::Sim3 { q, t, s }`: `exp` · `log` · `mul` · `inverse` · `map`. VertexSim3Expmap::oplus 는 `_fix_scale` 이면 **풀이 벡터 x 의 7 번째 칸을 제자리에서 0 으로 바꾼다**(그 뒤 computeScale 이 바뀐 x 를 씀).
- 원본 χ² 비교 때 비활성 간선(정점 모두 고정)의 값은 초기화 안 된 메모리라 비교에서 뺀다.
