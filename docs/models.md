# 모델 파일

모델은 저장소에 넣지 않고 GitHub 릴리스 [`models-v1`](https://github.com/juyoung020/ondevice-scene-memory/releases/tag/models-v1)에 올려 둡니다. `scripts/setup.sh`(리눅스·WSL·보드) · `scripts/setup.ps1`(윈도우)이 받아서 sha256 을 확인하고 `models/` 에 풉니다. 묶음 파일의 sha256 은 릴리스의 `SHA256SUMS` 에 있습니다.

## 묶음

| 묶음 | 풀리는 곳 | 쓰는 곳 | 라이선스 |
|---|---|---|---|
| `yolo26s-seg-onnx.tar.gz` | `models/yolo26s-seg.onnx` | 프로그램(PC 분할, `--seg` 기본값) | AGPL-3.0 |
| `objectsam-onnx.tar.gz` | `models/ObjectSAM-416_640.onnx` | 프로그램(1 클래스 분할 고르기, `--seg-names object`) | AGPL-3.0 + 비상업 |
| `siglip2-onnx.tar.gz` | `models/siglip2-base-patch32-256/` | 프로그램(`--embed` 영상 인코더, 뷰어 `--search-model` 글 인코더) | Apache-2.0 |
| `board-npu.tar.gz` | `models/board/siglip2-vision_v8.dxnn`, `models/board/objectsam-416_640.dxnn` | 프로그램(보드 NPU `--embed`, `--seg`) | Apache-2.0, AGPL-3.0 + 비상업(ObjectSAM) |
| `train-weights.tar.gz` | `models/*.pt`, `models/*.fsow` | 학습용(`train/ObjectSAM`) | AGPL-3.0 (+ 비상업: ObjectSAM) |

`setup.sh` 는 위 세 개를 기본으로, `--board` 에 `board-npu`, `--train` 에 `train-weights` 를 더 받습니다.

## 파일

| 파일 | 용도 | 출처 | 라이선스 | sha256 |
|---|---|---|---|---|
| `yolo26s-seg.onnx` | 프로그램 | Ultralytics YOLO26s-seg(COCO) 의 ONNX 내보내기(Hugging Face `prithivMLmods/YOLO26-ONNX`) | AGPL-3.0 | `8420b5a95e54b285ae7577207567df794856727a93bca8fef7259326eb633c25` |
| `ObjectSAM-416_640.onnx` | 프로그램 | ObjectSAM v1.0(Ultralytics FastSAM-s 에서 미세조정한 1 클래스 분할, `train/ObjectSAM/archive`), 640 입력 ONNX | AGPL-3.0, 학습 데이터(ADE20K · BEHAVIOR-1K) 조건상 비상업 연구용 | `ec242882ddea4f19fccb919960b1279d3120387b336f9b23192243a08139518a` |
| `siglip2-base-patch32-256/vision_model.onnx` | 프로그램 | Google SigLIP2 base patch32-256(Hugging Face `google/siglip2-base-patch32-256`) 영상 인코더 ONNX | Apache-2.0 | `0ddb7507cfcb54c2cd503029949add9dc6586de67164bb69c2c09588cdcc9719` |
| `siglip2-base-patch32-256/text_model_fp16.onnx` | 프로그램 | 같은 모델의 글 인코더 ONNX(FP16) | Apache-2.0 | `3f8149272095410332433ca84f2c99611ed4679429eff43fbf3983ea15664ef3` |
| `siglip2-base-patch32-256/tokenizer.json` | 프로그램 | 같은 모델 | Apache-2.0 | `cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322` |
| `siglip2-base-patch32-256/tokenizer.model` · `*config.json` · `special_tokens_map.json` · `README.md` | 모델 설명·설정 | 같은 모델 | Apache-2.0 | 묶음 sha256 |
| `board/siglip2-vision_v8.dxnn` | 프로그램(보드) | 위 영상 인코더를 DX-COM 2.4.1 로 NPU 용 변환(파일 형식 8, DXRT 3.4)(만드는 법: `docs/performance_log.md` 의 SigLIP2 절) | Apache-2.0 | `3905f733b21726720170218bfe00b7582bb29081ecca360cc5c957da44c0224e` |
| `board/objectsam-416_640.dxnn` | 프로그램(보드 `--seg`, 이름은 `--embed` 로) | `ObjectSAM-416_640.onnx` 를 DX-COM 2.4.1 로 8 비트 변환(파일 형식 8, 보정 TUM fr1_desk · fr3_long_office 305 장). 보드에서 PC ONNX 와 물체 마스크 IoU 평균 0.86 | AGPL-3.0, 비상업 연구용 | `f92e857a8d331cd31fd57122334569cf72f2c32436dd26b04c640a7a9e44fa63` |
| `yolo26s-seg.pt` | 학습용 | Ultralytics YOLO26s-seg(COCO) | AGPL-3.0 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |
| `yolov8s-seg.pt` | 학습용(출발 가중치) | Ultralytics YOLOv8s-seg(COCO) | AGPL-3.0 | `0bac0770b55e5eb5b76a61bc535673288dcec36c2bc0cd25ee0d584c632f3413` |
| `ObjectSAM-416.pt` | 학습용 | ObjectSAM v1.0 가중치(`train/ObjectSAM/archive/NOTICE`) | AGPL-3.0, 비상업 연구용 | `2917144d060151f6ac0a7b7dcfeb55e4ee98148c51befb09bb8438ba028a5dd4` |
| `yolo26s-seg-nc1.fsow` | 학습용 | `yolo26s-seg.pt` 의 클래스 출력층을 1 종으로 바꾼 학습기 형식 | AGPL-3.0 | `2d88f3ec9827c955aa869343dcade73520ed072ddecc360113be2181c6e055e5` |
| `yolov8s-seg-nc1.fsow` | 학습용 | `yolov8s-seg.pt` 같은 방식 | AGPL-3.0 | `de976c3b0fd0bf5ca7f1bd1c8658ddc9c3b787a456740eb74d406c1588e3f25a` |
| `ref26.fsow` | 학습용(정합성 시험 기준값) | `yolo26s-seg-nc1.fsow` 로 계산한 정방향·역방향 기준(`train/ObjectSAM/docs/YOLO26_PORT.md`) | AGPL-3.0 | `c34a5eb7851672b29bf41d76bc2f3e82d4e8d0587b1ad0405517f7baf9fde402` |

보드의 분할 모델 `yolo26-s-seg_640x640.dxnn` 은 DEEPX 가 배포하는 파일이라 이 릴리스에 넣지 않습니다. DEEPX dx-all-suite 의 모델 받기(`workspace/res/models/`)로 받습니다([dx_app](https://github.com/DEEPX-AI/dx_app)).

## 라이선스 관계

- 코드는 `LICENSE.md`(PolyForm Noncommercial 1.0.0)를 따르고, 모델 파일은 위 표의 각 라이선스를 따릅니다. 모델은 코드와 별도 파일로 배포되며, 프로그램은 실행할 때 경로로 읽습니다.
- AGPL-3.0 모델(Ultralytics 계열 · ObjectSAM)은 AGPL-3.0 조건으로 다시 배포할 수 있습니다. 원본은 [Ultralytics](https://github.com/ultralytics/ultralytics), 학습 코드는 `train/ObjectSAM`(AGPL-3.0, `train/ObjectSAM/LICENSE` · `NOTICE`).
- ObjectSAM v1.0 은 비상업 조건의 데이터(ADE20K 이미지 · BEHAVIOR-1K)로 학습해 비상업 연구용입니다(`train/ObjectSAM/archive/NOTICE`).
- SigLIP2 와 그 변환본은 Apache-2.0 입니다(`siglip2-base-patch32-256/README.md`).
