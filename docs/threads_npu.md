# ondevice 스레드별 입력 · 출력과 NPU 호출 항목

기준 코드: 브랜치 opt/npu-async (= opt/imu ebf7ebe). 보드에서 `ondevice` 하나를 실행하면 아래 스레드가 한 프로세스 안에서 돈다.

## 1. 스레드별 입력 → 출력

| 스레드 | 코드 위치 | 입력 | 하는 일 | 출력 (다음으로 넘기는 것) |
|---|---|---|---|---|
| 카메라 읽기 | slam/src/io.rs:258 | D435 USB 프레임 | 컬러 · 깊이를 받아 컬러에 맞춰 정렬 | 컬러 RGB 640×480 + 깊이 u16 (버퍼 풀) |
| imu | slam/src/imu.rs:564 | I2C 레지스터 (200 Hz) | 이상한 값 거르기, 시각 붙이기 | IMU 표본 → 링버퍼 (복사 없음) |
| imu-pose | slam/src/imu.rs:509 | 링버퍼 + 마지막 영상 자세 | 장면 사이 회전 적분 | 200 Hz 위치 → pose.json (메모리) |
| slam | slam/src/app.rs:1150 | 컬러 · 깊이 + IMU 링버퍼 | 위치 추적, 지도 그래프, 루프 클로저 | 장면 스트림(위치 · 컬러 · 깊이 · 그래프 노드 · 표시) → 메모리 큐 |
| reader | pipeline/src/main.rs:662 | 장면 스트림 | 위치를 바로 뷰어로, 분할 전처리(640 레터박스) | SegJob(입력 u8 640×640×3) |
| seg-infer | main.rs:689 | SegJob | **NPU 에 분할 올리기 (비동기)** | 작업 번호 |
| seg-collect | main.rs:693 | 작업 번호 | **NPU 결과 받기** | SegRaw(출력 10 개를 한 f32 버퍼에) |
| seg-post | main.rs:737 | SegRaw | 상위 300 고르기, 박스 · 마스크 해독 | 물체 목록(Det) 붙은 Frame → main |
| main | main.rs (run) | Frame | 마스크 × 깊이 → 3D 점, 같은 물체 합치기, 지도 갱신, 물체 사진 자르기 | 물체 기억, 물체 사진 → embed |
| embed | pipeline/src/embed.rs:313 | 물체 사진(Arc) | **NPU 에 SigLIP2 올리기 · 받기**, 정규화 | 사진 벡터 → 이름 · 검색용 파일 |
| pts | main.rs:1924 | 깊이 + 위치 | 점 지도(3D 맵) 조각 만들기 | 점 지도 조각 |
| rebuild | main.rs:1002 / 1038 (필요할 때) | 보관 장면 + 바뀐 위치 | 루프 클로저 뒤 지도 다시 쌓기 | 고친 지도 · 물체 |
| view | main.rs:2381 | 물체 기억 · 지도 | 방 나누기, view.json 만들기 | view.json (메모리) |
| writer | pipeline/src/out.rs:57 | 쓸 파일(PNG · PLY) | 파일로 쓰기 | 기억 폴더 |
| sgview (+ stream, http×N) | main.rs:2636, viewer/sgview/src/lib.rs | view.json · 점 지도 · pose.json | 웹 뷰어로 바뀐 부분만 전송 | 브라우저 |
| save | main.rs:2697 | 기억 | 주기적으로 디스크 저장 | 기억 폴더 |

## 2. NPU 호출 항목

### ObjectSAM (물체 분할) — 장면마다
| 단계 | 스레드 | 내용 | 데이터 |
|---|---|---|---|
| 전처리 | reader | 640 레터박스(빈 곳 회색 114), RGB | u8 [1, 640, 640, 3] = 1.2 MB |
| 올리기 | seg-infer | `dxrt_engine_run_async(e, 입력 포인터, …, &id)` — 입력 버퍼를 복사 없이 넘김 | 작업 번호 |
| 함께 올리기 | — | `--npu-jobs K` 로 K 장까지 NPU 에 겹쳐 올림 | |
| 받기 | seg-collect | `dxrt_engine_wait(e, id, 출력 버퍼)` — **DXRT 가 출력 전체를 우리 버퍼로 복사** | 출력 10 개: 머리 9 개(보폭 8/16/32 × 박스 4 · 점수 1 · 계수 32) + 기본 마스크 32×160×160 (≈ 3.3 MB) |
| 후처리 | seg-post | `heads_to_det`(문턱 · 상위 300 · 박스 해독), `decode`(계수 × 기본 마스크 → 마스크, 원본 크기로) | 물체 목록 |
| 시간 | | NPU 약 9 ms | |

### SigLIP2 (물체 사진 → 벡터) — 새 물체 · 더 크게 본 물체마다
| 단계 | 스레드 | 내용 | 데이터 |
|---|---|---|---|
| 전처리 | embed | 물체 사진을 32×32 조각 64 개로(`preprocess_into`), 입력 버퍼 2 개를 번갈아 | f32 [1, 64, 3072] = 786 KB |
| 올리기 | embed | `dxrt_engine_run_async` | 작업 번호 |
| 받기 | embed | 다음 사진 전처리 뒤 `dxrt_engine_wait` | 벡터(DIM) |
| 후처리 | embed | 정규화, 파일 쓰기, 단어 표와 비교해 이름 | 이름 · 확신도 |
| 시간 | | NPU 13~18 ms | |

## 3. 이미 적용된 것 (비동기 · 복사 줄이기)
- 분할: 전처리 → NPU → 후처리를 스레드 3 개로 나눠 겹침, NPU 는 비동기 올리기 + 별도 받기 스레드, 여러 장 동시 올리기(`--npu-jobs`).
- SigLIP2: 한 장 올려 둔 동안 다음 사진 전처리(버퍼 2 개 번갈아).
- 장면 버퍼 풀(bufpool), IMU 링버퍼(복사 없이 빌려 씀), 물체 사진은 Arc 로 공유.

## 4. 아직 남은 복사 · 대기 (최적화 후보)
1. **분할 출력 받기**: `wait` 가 출력 약 3.3 MB 를 장면마다 우리 버퍼로 복사 → DXRT 출력 버퍼를 직접 읽는 방법 검토.
2. **SigLIP2 받기**: 물체마다 결과용 Vec 를 새로 할당(`Vec::new` → `wait_into`) → 버퍼 재사용.
3. **물체 사진 저장**: 사진 · 마스크 · 깊이를 PNG 로 쓸 때 `clone()` 으로 복사.
4. **SigLIP2 결과 파일**: 벡터를 바이트로 바꾸는 Vec 새로 만듦.
5. **스레드 수**: writer · pts · view · save 처럼 가끔만 깨는 스레드는 합칠 수 있는지 측정 필요.

## 물체 벡터 메모리 표(`viewer/sgview/src/objtable.rs`)

검색이 질의마다 물체 벡터 파일을 하나씩 읽고 물체 × 단어장(1,230 개) 내적을 다시 하던 것을, 물체 벡터를 이어 붙인 f32 행렬 하나와
물체마다 미리 계산한 단어 점수(`SCALE · cos(단어, 사진)`)로 바꿨다. 물체 벡터가 들어오거나 바뀔 때만 그 줄을 계산하고(코어 여럿),
질의는 물체마다 내적 한 번(질의 · 사진)과 미리 계산한 점수의 log-sum-exp 만 한다. 단어장 점수는 선택 사항이라(단어를 주지 않으면 계산하지 않음)
sigmoid 확률로 바로 검색하는 경우에도 `scores` · `top_k` 만으로 쓴다.

| 공개 함수 | 하는 일 |
|---|---|
| `ObjTable::new(dim, scale)` | 빈 표 |
| `set_words(words)` | 단어 목록(글 벡터). 바뀌면 모든 물체의 단어 점수를 다시 계산 |
| `upsert(key, &v)` · `remove(&key)` | 물체 추가 · 갱신 · 삭제(삭제는 마지막 줄을 그 자리로) |
| `sync_bytes(&[(key, 파일 바이트)])` | 벡터 파일 바이트로 맞추기: 바뀐 것만 다시 계산, 목록에 없는 물체는 지움 |
| `row` · `vector` · `logits` | 줄 번호, 그 줄의 벡터 · 단어 점수 |
| `scores(&q)` · `top_k(&q, k)` | 질의 벡터와 모든 물체의 내적, 큰 순 k 개 |

```rust
let mut t: sgview::objtable::ObjTable<u32> = ObjTable::new(768, 111.83);
t.set_words(word_vectors);              // 선택: 단어장 점수까지 미리 계산
t.upsert(7, &picture_vector);           // 추가 · 갱신
let best = t.top_k(&query_vector, 5);   // [(물체 번호, 내적)] 큰 순
t.remove(&7);
```

내적은 `objtable::dot`: 보드(aarch64)는 NEON 4 칸 × 4 묶음, PC 는 차례대로 더함(같은 값이 기존 검색과 비트까지 같음, 보드는 상대 1e-7 안팎 차).

질의 한 번(PC WSL, 단어장 1,225 + NULLS 5, 질의 글 벡터는 미리 있음, 벡터 파일은 메모리(`--serve`), 5 번 평균):

| 물체 수 | 전(질의마다 파일 읽기 + 물체 × 단어 내적) | 후(메모리 표) | 벡터가 모두 바뀐 뒤 첫 질의 |
|---|---|---|---|
| 10 | 5.7 ms | 0.70 ms | 8.7 ms |
| 100 | 46.4 ms | 2.17 ms | 12.0 ms |
| 1,000 | 477 ms | 9.23 ms | 40.6 ms |

결과(순위 · 점수 · by)는 전 방식과 같다(시험 `same_as_per_query_scan`, 질의 4 개 × 두 번 + 벡터 파일 바뀜).
