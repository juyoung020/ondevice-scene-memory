# 모델 파일

모델은 저장소에 넣지 않고 GitHub 릴리스 두 개에 올려 둡니다. `scripts/setup.sh`(리눅스·WSL·보드) · `scripts/setup.ps1`(윈도우)이 받아서 릴리스의 `SHA256SUMS` 로 sha256 을 확인하고 `models/` 에 풉니다.

- [`models-v2`](https://github.com/juyoung020/ondevice-scene-memory/releases/tag/models-v2) (2026-11-17): 지금 보드가 쓰는 NPU 모델(NPU 만 도는 판), 그 컴파일 재료, 이름 표.
- [`models-v1`](https://github.com/juyoung020/ondevice-scene-memory/releases/tag/models-v1) (2026-11-14): PC 모델(ONNX), 학습용 가중치, 예전 보드 NPU 모델.

`setup.sh` 는 `objectsam-onnx` · `siglip2-onnx` 를 기본으로, `--board` 에 `board-npu-v2`, `--train` 에 `train-weights`, `--yolo` 에 `yolo26s-seg-onnx` 를 더 받습니다. `--board` 는 보드 모델을 `~/models` 에도 복사합니다(프로그램의 보드 기본 모델 폴더, 같은 이름의 다른 파일이 있으면 두고 알림).

## models-v2 묶음

| 묶음 | 크기 | 풀리는 곳 | 쓰는 곳 | 라이선스 | sha256 |
|---|---|---|---|---|---|
| `board-npu-v2.tar.gz` | 112,636,701 B | `models/board/siglip2-vision_npu.dxnn`, `models/board/objectsam-416_640.dxnn`, `models/board/README.md` | 프로그램(보드 NPU `--embed` · `--seg`) | Apache-2.0(SigLIP2), AGPL-3.0 + 비상업(ObjectSAM) | `4f43ede5c1edcd10ef2e115d5b573dc5830f338dddf8087c16a2cff820017ae3` |
| `siglip2-npu-src.tar.gz` | 359,677,387 B | `siglip2-npu-src/`(`vq.onnx` + `vq.onnx.data`, `compiler.log`, `diagnosis_report.html`, `README.md`) | 사진 벡터 dxnn 을 다시 만들거나 다른 NPU 로 옮길 때 | Apache-2.0 | `5d4e0546a821d437555efaa2e27903e2488d722b9f8d6919c52d5f842ad2d382` |
| `vocab.tar.gz` | 3,796,278 B | `vocab/`(저장소 `vocab/` 와 같은 `indoor` · `names` 의 `.txt` · `.voc` + `README.md`) | 다른 프로젝트에서 이름 표만 쓸 때(이 저장소는 `vocab/` 를 그대로 씀) | PolyForm Noncommercial 1.0.0, 이름 출처 LVIS v1(CC BY 4.0) · COCO | `37e78f34bd8a96b401268be44d02954dd92eee5146848f5686d536251bd7d31b` |

| 파일 | 내용 | 확인 값 | sha256 |
|---|---|---|---|
| `board/siglip2-vision_npu.dxnn` (132,117,685 B) | SigLIP2 영상 인코더, 입력 `patches` f32 [1,64,3072](32×32 조각 64 개) → `pooler_output` [1,768]. DX-COM 2.4.1(형식 8), 8 비트, 보정 `ema` 물체 사진 200 장, NPU 작업 하나 | 컴파일러 진단 코사인 0.9907, 보드 ↔ PC 코사인 평균 0.928 · 최소 0.847(사진 40 장), NPU 13.5 ms | `2c9f0bcc3a27a1224016300547bd624292ecd942d45ef8f817bc2a07dbe5b00d` |
| `board/objectsam-416_640.dxnn` (3,840,545 B) | ObjectSAM 머리 출력판, 입력 `images` u8 [1,640,640,3] → 머리 9 개 + 원형 `output1` [1,32,160,160]. DX-COM 2.4.1(형식 8), 8 비트, 보정 `ema` TUM 305 장, NPU 작업 하나 | PC ONNX 대비 물체 마스크 IoU 평균 0.861(fr1_desk 304 장), NPU 9 ms | `b3579a4bfa144aec88f49e26491c39575142d444eacdc0b7c7b4e5a91e405178` |
| `siglip2-npu-src/vq.onnx` · `vq.onnx.data` | 위 dxnn 을 컴파일한 원본 ONNX(가중치 따로) | 실수로 원래 인코더와 코사인 0.9999 | `5821eac5a0edda58f19c6a08ce70fdf910ac1f4b469e5aac68fcc567a2528e3b`(`vq.onnx`) |

입력 만들기 · 출력 풀기 · 다른 프로그램에서 DXRT 로 여는 법은 릴리스 본문과 각 묶음의 `README.md`, 만드는 순서는 [npu_models.md](npu_models.md). 두 dxnn 은 이 판의 코드(`pipeline/src/seg.rs` 의 머리 출력 풀기, `embed.rs` 의 조각 입력)가 있어야 읽힙니다.

## models-v1 묶음

| 묶음 | 풀리는 곳 | 쓰는 곳 | 라이선스 |
|---|---|---|---|
| `yolo26s-seg-onnx.tar.gz` | `models/yolo26s-seg.onnx` | 비교 · 학습용(`setup.sh --yolo`). 지금 프로그램의 분할은 클래스 없는 ObjectSAM 만 받음 | AGPL-3.0 |
| `objectsam-onnx.tar.gz` | `models/ObjectSAM-416_640.onnx` | 프로그램(PC 분할, `--seg` 기본값) | AGPL-3.0 + 비상업 |
| `siglip2-onnx.tar.gz` | `models/siglip2-base-patch32-256/` | 프로그램(`--embed` 영상 인코더, 뷰어 `--search-model` 글 인코더) | Apache-2.0 |
| `board-npu.tar.gz` | `models/board/siglip2-vision_v8.dxnn`, `models/board/objectsam-416_640.dxnn` | 예전 보드 판(모델 안 CPU 작업 포함, 지금 프로그램의 보드 분할은 열지 않음). 지금은 `board-npu-v2` | Apache-2.0, AGPL-3.0 + 비상업(ObjectSAM) |
| `train-weights.tar.gz` | `models/*.pt`, `models/*.fsow` | 학습용(`train/ObjectSAM`) | AGPL-3.0 (+ 비상업: ObjectSAM) |

## models-v1 파일

| 파일 | 용도 | 출처 | 라이선스 | sha256 |
|---|---|---|---|---|
| `yolo26s-seg.onnx` | 프로그램 | Ultralytics YOLO26s-seg(COCO) 의 ONNX 내보내기(Hugging Face `prithivMLmods/YOLO26-ONNX`) | AGPL-3.0 | `8420b5a95e54b285ae7577207567df794856727a93bca8fef7259326eb633c25` |
| `ObjectSAM-416_640.onnx` | 프로그램 | ObjectSAM v1.0(Ultralytics FastSAM-s 에서 미세조정한 1 클래스 분할, `train/ObjectSAM/archive`), 640 입력 ONNX | AGPL-3.0, 학습 데이터(ADE20K · BEHAVIOR-1K) 조건상 비상업 연구용 | `ec242882ddea4f19fccb919960b1279d3120387b336f9b23192243a08139518a` |
| `siglip2-base-patch32-256/vision_model.onnx` | 프로그램 | Google SigLIP2 base patch32-256(Hugging Face `google/siglip2-base-patch32-256`) 영상 인코더 ONNX | Apache-2.0 | `0ddb7507cfcb54c2cd503029949add9dc6586de67164bb69c2c09588cdcc9719` |
| `siglip2-base-patch32-256/text_model_fp16.onnx` | 프로그램 | 같은 모델의 글 인코더 ONNX(FP16) | Apache-2.0 | `3f8149272095410332433ca84f2c99611ed4679429eff43fbf3983ea15664ef3` |
| `siglip2-base-patch32-256/tokenizer.json` | 프로그램 | 같은 모델 | Apache-2.0 | `cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322` |
| `siglip2-base-patch32-256/tokenizer.model` · `*config.json` · `special_tokens_map.json` · `README.md` | 모델 설명·설정 | 같은 모델 | Apache-2.0 | 묶음 sha256 |
| `board/siglip2-vision_v8.dxnn` | 예전 보드 판(입력 `[1,3,256,256]`, 앞뒤 CPU 작업 포함) | 위 영상 인코더를 DX-COM 2.4.1 로 NPU 용 변환(파일 형식 8, DXRT 3.4)(만드는 법: `docs/performance_log.md` 의 SigLIP2 절) | Apache-2.0 | `3905f733b21726720170218bfe00b7582bb29081ecca360cc5c957da44c0224e` |
| `board/objectsam-416_640.dxnn`(v1) | 예전 보드 판(모델 안 CPU 작업 포함) | `ObjectSAM-416_640.onnx` 를 DX-COM 2.4.1 로 8 비트 변환(파일 형식 8, 보정 TUM fr1_desk · fr3_long_office 305 장). 보드에서 PC ONNX 와 물체 마스크 IoU 평균 0.86 | AGPL-3.0, 비상업 연구용 | `f92e857a8d331cd31fd57122334569cf72f2c32436dd26b04c640a7a9e44fa63` |
| `yolo26s-seg.pt` | 학습용 | Ultralytics YOLO26s-seg(COCO) | AGPL-3.0 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |
| `yolov8s-seg.pt` | 학습용(출발 가중치) | Ultralytics YOLOv8s-seg(COCO) | AGPL-3.0 | `0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413` |
| `ObjectSAM-416.pt` | 학습용 | ObjectSAM v1.0 가중치(`train/ObjectSAM/archive/NOTICE`) | AGPL-3.0, 비상업 연구용 | `2917144d060151f6ac0a7b7dcfeb55e4ee98148c51befb09bb8438ba028a5dd4` |
| `yolo26s-seg-nc1.fsow` | 학습용 | `yolo26s-seg.pt` 의 클래스 출력층을 1 종으로 바꾼 학습기 형식 | AGPL-3.0 | `2d88f3ec9827c955aa869343dcade73520ed072ddecc360113be2181c6e055e5` |
| `yolov8s-seg-nc1.fsow` | 학습용 | `yolov8s-seg.pt` 같은 방식 | AGPL-3.0 | `de976c3b0fd0bf5ca7f1bd1c8658ddc9c3b787a456740eb74d406c1588e3f25a` |
| `ref26.fsow` | 학습용(정합성 시험 기준값) | `yolo26s-seg-nc1.fsow` 로 계산한 정방향·역방향 기준(`train/ObjectSAM/docs/YOLO26_PORT.md`) | AGPL-3.0 | `c34a5eb7851672b29bf41d76bc2f3e82d4e8d0587b1ad0405517f7baf9fde402` |

## 라이선스 관계

- 코드는 `LICENSE.md`(PolyForm Noncommercial 1.0.0)를 따르고, 모델 파일은 위 표의 각 라이선스를 따릅니다. 모델은 코드와 별도 파일로 배포되며, 프로그램은 실행할 때 경로로 읽습니다.
- AGPL-3.0 모델(Ultralytics 계열 · ObjectSAM)은 AGPL-3.0 조건으로 다시 배포할 수 있습니다. 원본은 [Ultralytics](https://github.com/ultralytics/ultralytics), 학습 코드는 `train/ObjectSAM`(AGPL-3.0, `train/ObjectSAM/LICENSE` · `NOTICE`).
- ObjectSAM v1.0 은 비상업 조건의 데이터(ADE20K 이미지 · BEHAVIOR-1K)로 학습해 비상업 연구용입니다(`train/ObjectSAM/archive/NOTICE`).
- SigLIP2 와 그 변환본은 Apache-2.0 입니다(`siglip2-base-patch32-256/README.md`).
- 이름 표(`vocab.tar.gz`, 저장소 `vocab/`)의 이름 목록은 LVIS v1 범주 이름(CC BY 4.0) · COCO 에서 왔고, 글 벡터는 SigLIP2(Apache-2.0)로 계산했으며 파일은 저장소와 같은 PolyForm Noncommercial 1.0.0 입니다.
