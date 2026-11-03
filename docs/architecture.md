# 동작 원리

## 처리 흐름

```
카메라 한 장
  ├─ NPU ① 분할 (YOLO26s-seg)   → 물체 마스크·이름
  ├─ NPU ② 깊이 (YOLO26s-depth) → 화소마다 거리
  └─ NPU ③ 특징점 (SuperPoint)  → 앞 장면과 맞출 점
        ▼ CPU (Rust)
  위치 추정: 같은 특징점의 3D 위치를 여러 장면에 걸쳐 평균 → 카메라 위치·방향
        ▼
  누적 지도: 깊이를 카메라 위치로 옮겨 3D 점·점유 지도에 더함(새 점만)
  물체 트래커: 같은 자리·같은 이름이면 같은 물체로 합침, 안 보이면 "사라짐"
        ▼
  씬그래프: 빈 공간으로 방 나누기 → 방→물체, 물체 위·옆 관계
        ▼
  기억 폴더(view.json · 점 파일 · 물체 사진) → 웹 뷰어
```

- 작업 스레드가 다음 장면의 읽기와 모델 실행을 한 장 앞서 처리하고, 본 스레드가 위치·지도·트래커를 갱신한다. 실시간 모드에서는 파일 쓰기를 기록 스레드가 맡는다.
- 모델 셋은 공통 인터페이스(`Segmenter`·`DepthNet`·`FeatNet`)로 묶여, 모델 파일이 `.onnx` 면 PC(ONNX Runtime), `.dxnn` 이면 보드 NPU(DXRT C API)에서 돈다.
- 위치 추정과 지도 방법의 자세한 내용: [pose_and_map.md](pose_and_map.md).

## 폴더

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 기억을 만드는 프로그램(Rust, memmap) |
| `pipeline/src/vo.rs` | 위치 추정 |
| `pipeline/src/maps.rs` | 누적 지도·물체 트래커 |
| `pipeline/src/graph.rs` | 방 나누기·씬그래프 연결 |
| `pipeline/src/seg.rs` | 모델 실행(ONNX·NPU) |
| `viewer/sgview/` | 웹 뷰어(Rust 서버 + three.js) |
| `train/ObjectSAM/` | 자체 분할 모델 학습기(C++/CUDA) |
| `docs/` | 설계·측정 결과·평가 계획 |
| `models/`, `data/` | 모델·영상·기억 저장본(git 밖) |

## 개발 환경

- 저장소 경로는 공백·한글 없이(`C:\ondevice`). DEEPX 도구가 이런 경로에서 깨진다.
- 빌드 기능: PC `--features directml`(그래픽카드) 또는 기본(CPU), 보드 `--no-default-features --features dxrt,cam`.
