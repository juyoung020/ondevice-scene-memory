# YOLO26-seg 로 옮기기 (C++/CUDA 학습기)

지금 학습기(`src/`)는 YOLOv8s-seg 전용이다. 같은 CropFormer 라벨로 **YOLO26s-seg(1 클래스)** 를 직접 학습하려고 무엇이 같고 무엇을 새로 짜는지 정리한다.
이유: DX-M1 NPU 를 만든 DEEPX 가 YOLO26 을 공식 지원한다(Pi5 + M1 에서 YOLO26s-seg 처리량 95.7 FPS, dx-benchmark v2.4.0). COCO 기준으로 YOLOv8s-seg 보다 마스크 mAP 40.0 vs 36.8, 연산 34.5 vs 42.6 GFLOPs.

기준: Ultralytics 8.4.192 `ultralytics/cfg/models/26/yolo26-seg.yaml`, scale s, `end2end: True`, `reg_max: 1`.

## 신경망 (nc=1, 입력 640, 변수 11,434,463)

| # | 모듈 | 출력 (C×H×W) | 변수 | v8s 학습기에 |
|---|---|---|---|---|
| 0 | Conv 3×3 s2 | 32×320×320 | 928 | 있음 |
| 1 | Conv 3×3 s2 | 64×160×160 | 18,560 | 있음 |
| 2 | C3k2 (c3k=False, e=0.25) | 128×160×160 | 26,080 | C2f 와 같음(e 만 다름) |
| 3 | Conv 3×3 s2 | 128×80×80 | 147,712 | 있음 |
| 4 | C3k2 (c3k=False, e=0.25) | 256×80×80 | 103,360 | C2f 와 같음 |
| 5 | Conv 3×3 s2 | 256×40×40 | 590,336 | 있음 |
| 6 | C3k2 (c3k=True) | 256×40×40 | 346,112 | **C3k 새로** |
| 7 | Conv 3×3 s2 | 512×20×20 | 1,180,672 | 있음 |
| 8 | C3k2 (c3k=True) | 512×20×20 | 1,380,352 | C3k |
| 9 | SPPF (k5, n=3, shortcut) | 512×20×20 | 656,896 | 풀링 있음, **cv1 활성화 없음 + 잔차 새로** |
| 10 | C2PSA | 512×20×20 | 990,976 | **어텐션 새로** |
| 11–12 | Upsample + Concat(6) | 768×40×40 | — | 있음 |
| 13 | C3k2 (c3k=True) | 256×40×40 | 477,184 | C3k |
| 14–15 | Upsample + Concat(4) | 512×80×80 | — | 있음 |
| 16 | C3k2 (c3k=True) | 128×80×80 | 136,192 | C3k |
| 17–18 | Conv s2 + Concat(13) | 384×40×40 | 147,712 | 있음 |
| 19 | C3k2 (c3k=True) | 256×40×40 | 378,880 | C3k |
| 20–21 | Conv s2 + Concat(10) | 768×20×20 | 590,336 | 있음 |
| 22 | C3k2 (c3k=True, e=0.5, **attn=True**) | 512×20×20 | 1,843,712 | **Bottleneck + PSABlock 새로** |
| 23 | Segment26 | 아래 | 2,418,463 | **머리 대부분 새로** |

학습 때 출력: one2many·one2one 각각 boxes (4×8400), scores (1×8400), mask_coefficient (32×8400), proto (32×160×160), semantic (1×80×80).

## 그대로 쓰는 것

- Conv+BN+SiLU 노드(`CONVBN`), 채널 오프셋으로 쓰는 0-복사 Concat(`DestSpec`), 업샘플 후 Concat, SPPF 5×5 풀링 3번(`POOL`), Proto 의 전치합성곱(`CONVT`)
- C2f 배선(= C3k2 의 겉), Bottleneck, 잔차 더하기(`res`)
- TAL 할당, CIoU, 마스크 BCE(proto × 계수), GPU 증강, FTL 라벨 읽기, EMA, fsow 저장·읽기

## 새로 짜는 것

### 구조

| # | 무엇 | 쓰는 곳 | 구현 |
|---|---|---|---|
| S1 | Conv+BN **활성화 끔** | SPPF.cv1, Attention qkv·proj·pe, PSABlock ffn 두 번째 | `CONVBN` 에 act 플래그 |
| S2 | **깊이별(그룹) 합성곱** 3×3 | Attention.pe(g=C), cls 머리 DWConv | `ConvOp` 에 groups (cuDNN 그룹 conv 앞·뒤·wgrad) |
| S3 | C3k (= C3: cv1·cv2·cv3 + Bottleneck 3×3 e=1 ×2) | 6·8·13·16·19 | 조합 |
| S4 | SPPF 잔차 (`y + x`) | 9 | cv2 에 `res` |
| S5 | **Attention** — qkv 를 머리별로 나눠 S = softmax(qᵀk·scale) (N×N), out = v·Sᵀ + pe(v), proj | 10(C2PSA, c=256, 4머리, key 32, head 64, N=400), 22 | 새 노드 `ATTN`: cuBLAS strided batched GEMM(fp16, fp32 누적), softmax 앞·뒤 커널, NHWC↔머리 배치 전치 |
| S6 | PSABlock = x + attn(x), x + ffn(x) | 10, 22 | 조합(S5 + S1 + res) |
| S7 | Detect 머리(비legacy): box cv2 = Conv3·Conv3·1×1(4, reg_max=1), cls cv3 = [DW3+1×1]·[DW3+1×1]·1×1(1), mask cv4 = Conv3·Conv3·1×1(32) | 23 | `HEADCONV` 이름 규칙 바꾸기(`model.23.*`), S2 |
| S8 | **one2one 사본**(cv2·cv3·cv4), 입력·proto 를 **detach** | 23 | 같은 노드 한 벌 더, 넥으로 dgrad 안 흘림 |
| S9 | **Proto26**: P4·P5 를 1×1 conv 후 최근접 ×2·×4 업샘플해 P3 에 더함 → feat_fuse 3×3 → Proto(cv1·ConvT·cv2·cv3) | 23 | 업샘플-더하기 노드(×4 포함) |
| S10 | semseg 가지(학습 때만): Conv3·Conv3·1×1(nc=1) on 합친 P3 | 23 | 조합 |

### 손실 (`E2ELoss(SegmentLoss26)`)

| # | 무엇 | 내용 |
|---|---|---|
| L1 | 이중 할당 | one2many = TAL topk 10, one2one = TAL topk 7·topk2 1. 합 = o2m·w + o2o·(1−w), w 는 0.8 → 0.1 을 epoch 따라 선형 |
| L2 | 상자 | reg_max=1: 거리(ltrb) 4 개를 바로 예측. CIoU ×7.5 + **L1(거리)** ×6.0 (DFL 없음) |
| L3 | SegTaskAlignedAssigner | 겹침 = (상자 IoU + **예측 마스크 IoU**)/2, 마스크 IoU 는 gt 마다 상자 지표 상위 3·topk 쌍만 |
| L4 | 의미 분할 | semantic(1×80×80) vs 물체 마스크 합집합, BCE 0.5 + Dice 0.5 |
| L5 | 마스크·점수 | 지금과 같음(cls ×0.5) |

### 최적화기

공식 YOLO26s-seg 설정: **MuSGD**(2D 이상 가중치는 Muon — Newton-Schulz 직교화, 나머지는 SGD), lr0 0.00125, lrf 0.5, momentum 0.937, wd 7e-4, warmup 1, 50 epoch, close_mosaic 5, mixup 0.05, copy_paste 0.15.
순서: 먼저 지금의 SGD 로 옮긴 신경망을 검증하고, 그다음 MuSGD(GEMM 기반 Newton-Schulz 5회)를 더한다.

### 가중치

- 출발: `yolo26s-seg.pt`(COCO, 11,505,800 변수) → fsow. 클래스 출력층(cv3·one2one_cv3 마지막 1×1, semseg 마지막)만 80 → 1 로 새로 초기화(v21L 과 같은 방식).
- 내보내기: fsow → Ultralytics `.pt`(`pytools/fsow_to_pt.py` 확장) → ONNX → DX-COM(q-lite / q-pro) → `.dxnn`.

## 검증 순서

1. 블록별 정합성(fp32 빌드): Conv(act 끔)·DWConv·C3k·SPPF·Attention·PSABlock·Proto26·Segment26 을 PyTorch 와 앞·뒤(기울기) 비교 — `tests/parity.cpp` 를 블록 단위로 늘림
2. 전체 신경망 앞·뒤 정합성 (yolo26s-seg.pt 로)
3. 손실 정합성: 같은 배치·같은 출력으로 E2ELoss 값·기울기 비교
4. 한 배치 과적합 → 5 epoch 학습 → v21L 과 같은 uHumans2 S3 벤치마크
5. fsow → .pt → ONNX → .dxnn 변환, Pi5 + M1 에서 ms·INT8 손실

## 환경

- 학습 PC: RTX 5070 Ti 16 GB(sm_120). 원래 빌드는 리눅스(`/usr/local/cuda-12.8`, cuDNN, turbojpeg).
- 데이터: COCO train2017(약 19 GB) + `juyoung020/objectsam-labels` 릴리스 v1(`train2017.ftl` 1.2 GB, `.plan` 210 MB, `.siglip` 4 MB).
- 정합성 시험용 파이썬: Ultralytics 8.4.192 + torch(CUDA).
