# 동작 원리

## 처리 흐름

```
RealSense D435 (컬러 + 깊이, 640×480@30)
        ▼ slam_bridge (Rust, RTAB-Map 0.24.2 경로를 옮김)
  장면마다 위치 추적(F2F), 루프 클로저로 지도 노드 위치 다시 맞춤(BRIEF + GTSAM 방식)
  → 장면(컬러 + 컬러에 맞춘 깊이 + 위치 + 기준 노드) · 그래프 블록(움직인 노드)을 스트림으로
        ▼ memmap --stream (Rust)
  NPU 분할 (YOLO26s-seg)         → 물체 마스크·이름
  점 지도(점 작업 스레드): Potree 방식 옥트리 노드
  점유 지도: 바닥 5 cm 칸
  물체 트래커: 같은 자리·같은 이름이면 같은 물체로 합침, 안 보이면 "사라짐"
  병합 후처리(15 장마다): 따로 생긴 두 물체를 3D IoU·겹침 비율·이름 분포·색 분포·사진 벡터로 판단해 합침
  물체 사진 벡터(--embed, 따로 도는 스레드): 물체의 가장 좋은 사진 → SigLIP2 영상 인코더 → 768 차원 벡터
  루프 클로저 뒤 다시 쌓기(뒤 스레드): 보관한 장면으로 점유 지도·점 지도·물체를 새 위치로
        ▼
  씬그래프: 방 나누기(Hydra room finder 의 2D 판 — 벽까지 여유가 큰 곳부터 채워 가며 문처럼 좁은 목에서 끊음. 벽은 높이 1.3 m 넘는 막힘만 — 책상·의자는 방 안으로 본다) → 방→물체, 방↔방(문), 물체 위·옆 관계
  방 이름: 방 안 물체 이름으로 종류별 점수 → 확률·근거(예: 노트북·키보드·마우스 → office), 같은 종류는 번호(kitchen 2), 방 번호는 지도가 자라도 유지
        ▼
  기억 폴더(view.json · 점 노드 · 물체 사진 · 물체 벡터) → 웹 뷰어(변화를 SSE 로 밀어 줌, 검색어 → SigLIP2 글 인코더 → 물체 순위)
```

- 본 스레드는 위치·점유 지도·트래커만 하고, 점 지도·다시 쌓기·파일 쓰기는 각자 스레드에서 돈다(`--threads` 로 우선순위·코어 배치).
- 처리가 0.2 s 넘게 밀리면 쌓인 장면을 건너뛰고 최신 장면부터 처리한다. 그래프 블록은 버리지 않는다(`--no-drop` 이면 모두 처리).
- 분할 모델은 공통 인터페이스(`Segmenter`)로, 모델 파일이 `.onnx` 면 PC(ONNX Runtime), `.dxnn` 이면 보드 NPU(DXRT C API)에서 돈다.
- 스트림 입력의 분할은 세 단계(전처리 → 추론 → 후처리)를 스레드 셋이 장면을 한 장씩 차례로 넘기며 겹쳐 돌린다: 추론이 도는 동안 앞 장면의 마스크 해독과 다음 장면의 읽기·전처리가 돈다. 보드 NPU 에는 장면을 4 장까지 함께 올려(`--npu-jobs`, DXRT 비동기 실행) NPU 코어 여럿이 나눠 돌고, 결과는 올린 순서대로 받아 장면 순서는 그대로다(`--seg-pipe off` 는 한 스레드에서 차례로).
- 병합 기준(HOV-SG·ConceptGraphs 방식): 4 cm 칸 IoU ≥ 0.5, 또는 작은 쪽 기준 겹침 ≥ 0.6 이면서 이름 투표 코사인 ≥ 0.5 이거나 색 히스토그램 코사인 ≥ 0.8.
- 사진 벡터로 합치기: 두 물체의 사진 벡터 코사인 ≥ 0.88 이고 작은 쪽 굳은 칸의 25 % 이상이 큰 쪽 굳은 칸과 겹치면 이름이 달라도 합친다(같은 물체가 보는 방향에 따라 두 이름으로 남은 경우). 사진 벡터는 물체의 종류까지만 가린다 — 나란히 놓인 모니터 두 대도 코사인이 0.92 다. 그래서 자리가 겹칠 때만 쓰고, 같은 이름의 이웃 물체는 칸 겹침·맞닿음·전형 크기로 가른다. 벡터가 아직 없는 물체는 기하·이름 조건만 본다.
- 물체 사진 벡터: 물체 사진이 바뀔 때마다 계산 스레드(코어 2 개)가 `objects/O<id>_emb.bin`(f32 768 개, 길이 1)을 다시 쓴다. 밀리면 물체마다 가장 새 사진만 계산한다.
- 검색(`/api/search`, 뷰어 `search.rs`): 점수 = 1 − (1 − 이름 일치)(1 − 생김새 일치). 이름 일치는 검색어가 분할 모델의 이름 80 개 중 어느 것을 뜻하는지(글 벡터끼리, "의자" → chair), 생김새 일치는 검색어가 그 물체 사진을 이름들보다 잘 설명하는지(글 벡터와 사진 벡터, "마실 것" → 컵 사진)다. 글자가 이름과 겹치면 이름 일치 1.
- 연결 프로그램 `slam_bridge` 는 C++ 원본 `rtab_bridge` 와 PC 에서 출력이 바이트까지 같다. 스트림 형식·루프 클로저·다시 쌓기: [bridge.md](bridge.md), 보드 설치 [board_bridge.md](board_bridge.md).

## 폴더

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 기억을 만드는 프로그램(Rust, memmap) |
| `pipeline/src/stream.rs` | 장면 스트림 입력(`--stream`) |
| `pipeline/src/kfstore.rs` | 다시 쌓기용 장면 보관(압축) |
| `pipeline/src/maps.rs` | 누적 지도·물체 트래커·물체 병합 |
| `pipeline/src/embed.rs` | 물체 사진 → 검색용 벡터(SigLIP2 영상 인코더) |
| `pipeline/src/rooms.rs` | 방 나누기·문·방 번호 잇기·물체 배정·방 이름 |
| `pipeline/src/graph.rs` | 씬그래프 연결(방→물체, 방↔방, 물체↔물체) |
| `pipeline/src/seg.rs` | 분할 모델 실행(ONNX·NPU) |
| `slam/` | 연결 프로그램 `slam_bridge`(Rust, RealSense·TUM → 위치 추적·루프 클로저 → 장면 스트림), [slam/README.md](../slam/README.md) |
| `bridge/` | C++ 원본 `rtab_bridge`(RTAB-Map 라이브러리 링크, `slam_bridge` 비교 기준·지금 보드 실시간 스크립트) |
| `tools/` | 깊이→컬러 정렬 `fast_align`(Rust, rs2::align 과 비트 동일)과 도구 `rs_info`·`rs_rec`·`rs_align_check`·`align_sdk_check`·`bench_align`·`imu_probe`·`check_tum`, [realsense_input.md](realsense_input.md) |
| `viewer/sgview/` | 웹 뷰어(Rust 서버 + three.js) |
| `train/ObjectSAM/` | 자체 분할 모델 학습기(C++/CUDA) |
| `docs/` | 설계·측정 결과·평가 계획 |
| `models/`, `data/` | 모델·영상·기억 저장본(git 밖) |

## 개발 환경

- 저장소 경로는 공백·한글 없이(`C:\ondevice`). DEEPX 도구가 이런 경로에서 깨진다.
- 빌드 기능: PC `--features directml`(윈도우 그래픽카드) 또는 기본(CPU), 보드 `--no-default-features --features dxrt,cam`.
- 평가 표: `pipeline/target/release/bench [--quick]` — TUM 책상·사무실을 정해진 조건으로 돌려 위치 오차·실패·속도·물체 수를 한 표로.
- 다른 카메라 RGB-D 를 TUM 형식으로: `rgbd2tum <원본> <출력(이름에 freiburg3)> fx fy cx cy <깊이 1 m 값>` — TUM fr3 보정값 640×480 으로 다시 그려 RTAB-Map `rtabmap-rgbd_dataset`·`slam_bridge --tum` 과 memmap 이 같이 읽음. 다른 SLAM 위치로 돌리기: `memmap --tum <폴더> --gt <위치 파일>`, 정렬 후 위치 오차: `ate <추정> <정답>`(memmap 은 `poses_est.txt`·`poses_map.txt` 를 남김).
