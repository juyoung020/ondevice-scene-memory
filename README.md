# 물체 공간 기억 — 손카메라로 훑으면 보드가 방을 3D로 기억한다

[변경 기록(CHANGELOG)](CHANGELOG.md)

RealSense D435 깊이 카메라를 손에 들고 방을 한 바퀴 훑으면, 라즈베리파이 5와 DEEPX DX-M1 NPU 보드가 다음 세 가지를 실시간으로 만들어 기억합니다.

- **3D 지도:** 3D 점 지도, 바닥 점유 지도
- **물체 목록:** 무엇이 어디에 있는지(3D 위치, 사진, 마지막으로 본 시각)
- **씬그래프:** 방 → 물체, 방 ↔ 방(문), 물체 ↔ 물체 관계(위·옆)

계산은 모두 보드 안에서, Rust 프로그램 하나(`ondevice`)가 합니다. 카메라 위치는 위치 프로그램(RTAB-Map 의 SLAM 경로를 Rust 로 옮긴 것)이, 물체 분할은 NPU가, 지도·물체·씬그래프는 물체 기억(memmap)이, 화면은 뷰어 서버가 맡고 넷이 한 프로세스에서 돕니다.

![손카메라로 책상 주위를 돌며 3D 지도·물체·방이 쌓이는 모습](docs/cover-slam.gif)

*공개 영상(TUM RGB-D 사무실)을 넣은 모습. 아래는 3D 지도와 손에 든 카메라(파란 원뿔), 가운데는 찾은 물체, 위는 방입니다. 고화질: [cover-slam.mp4](docs/cover-slam.mp4)*

## 구성

```
ondevice(한 프로세스): D435(컬러 + 깊이) → 위치(slam, 위치 추적·루프 클로저) → 물체 기억(memmap) → 뷰어 서버(sgview) → 브라우저
                                                               NPU 물체 분할 ┘
```

- **위치:** 위치 프로그램(크레이트 [slam/](slam/README.md), `ondevice --slam` · `ondevice slam`)은 RTAB-Map 0.24.2 에서 쓰는 경로(위치 추적·루프 클로저·GTSAM 그래프 최적화)를 Rust 로 옮긴 것으로, PC 에서 C++ 원본 `rtab_bridge` 와 출력이 바이트까지 같고 약 2.2 배 빠릅니다. 장면마다 카메라 위치를 구하고, 한 바퀴 돌아 아까 본 곳을 알아보면(루프 클로저) 지나온 위치를 다시 맞춥니다. 자세한 형식은 [docs/bridge.md](docs/bridge.md).
- **점 지도:** Potree 방식(옥트리 단계별 상세도 + 둥근 점 + EDL 음영)으로 저장해, 뷰어는 화면에 필요한 만큼만 받아 그립니다. 루프 클로저로 위치가 고쳐지면 보관한 장면으로 지도와 물체를 다시 쌓습니다.
- **실시간:** 처리가 밀리면 쌓인 장면을 건너뛰고 최신 장면부터 처리합니다(위치 정보는 버리지 않음). 지도 작업·다시 쌓기는 본 스레드 밖에서 돕니다.

## 되는 것

- 물체 찾기(COCO 80종: 의자·컵·노트북·책 …) — 보드 NPU
- 카메라 위치 추정과 루프 클로저 — 위치 프로그램(`ondevice --slam`)(RTAB-Map 방식: BRIEF 특징 + GTSAM 최적화)
- 3D 점 지도, 물체 기억(옮겨짐·사라짐 표시)
- 글로 물체 찾기: 지도에 chair 로 저장돼 있어도 "의자"·"앉을 것"으로 찾습니다(물체 사진 벡터 + 글 벡터, SigLIP2)
- 방 나누기(Hydra 방식: 문처럼 좁은 목에서 끊음)와 문 찾기, 방 이름(물체 구성으로: 사무실·주방·거실 …, 확률·근거 포함), 방↔물체·방↔방·물체↔물체 연결
- 보드 실시간(NPU 코어 여럿에 장면을 함께 올림), PC·폰·보드 화면 브라우저 뷰어(High/Low 화질)

## 성능 (보드 실측, 공개 영상을 실제 속도로 재생)

| 항목 | 값 |
|---|---|
| 지도 갱신 | 약 11 장면/초, 지연 0.2 초 안팎(최대 0.25–0.35 초) |
| 물체 분할 | 받은 장면의 90 % 이상(초당 약 10 번) |
| 카메라 위치 오차 | 책상 4 cm, 사무실 3 cm, 58 m 복도 24 cm |

측정 방법과 기록은 [docs/performance_log.md](docs/performance_log.md)에, 최적화별 전·후 요약은 [docs/optimization.md](docs/optimization.md)에 있습니다.

## 빠른 시작 — PC(Windows·WSL)

보드 없이 PC에서 같은 과정을 돌려 볼 수 있습니다.

**1. 모델 받기** — `models/` 폴더에 둡니다.

| 파일 | 하는 일 | 받는 곳 |
|---|---|---|
| `yolo26s-seg.onnx` | 물체 찾기(분할) | Hugging Face `prithivMLmods/YOLO26-ONNX` |

**2. 시험 영상 받기** — [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)에서 영상을 받아 `data/` 아래에 풉니다(D435 로 찍은 사무실 영상 `data/d435/rgbd_dataset_freiburg3_office_sync` 와 같은 TUM 형식). 쓴 데이터 전체 목록은 [data/README.md](data/README.md).

**3. 빌드하고 실행** — 저장소 맨 위가 Cargo 작업 공간 하나입니다(`common` · `slam` · `pipeline` · `viewer/sgview` · `tools` · `rsdriver`, 빌드 결과는 `target/release/` 한 곳). 실행 파일은 `ondevice` 하나로, 위치 추적(slam) → 지도·물체(memmap) → 뷰어 서버(sgview)를 한 프로세스에서 돌립니다. 장면은 메모리 큐로 복사 없이 넘어가고, 뷰어는 이 프로세스가 기억 폴더에 쓴 파일(view.json · pose.json · map.pgm · 물체 사진 · 점 파일 · 사진 벡터 …)을 쓴 바이트 그대로 메모리에서 받습니다. 기억 폴더는 결과물로 그대로 디스크에 쓰입니다. 명령줄은 처음에 한 번 읽어 세 부분의 설정으로 나누고, ONNX Runtime 환경도 하나를 함께 씁니다. 자세한 것은 [slam/README.md](slam/README.md)와 [docs/bridge.md](docs/bridge.md).

```bash
cargo build --release --features directml        # target/release/ondevice

# TUM 영상 → 위치 추적 → 점 지도 + 물체 → 뷰어(http://127.0.0.1:8091), 한 명령
target/release/ondevice --slam "--tum data/d435/rgbd_dataset_freiburg3_office_sync --config docs/rtabmap_pi.ini" \
    --every 2 --seg models/yolo26s-seg.onnx --embed models/siglip2-base-patch32-256/vision_model.onnx --vocab vocab/names.voc \
    --max-depth 3.5 --out data/memory/office \
    --serve "8091 --bind 127.0.0.1 --search-model models/siglip2-base-patch32-256 --vocab vocab/names.voc"
```

- `--slam "<위치 인자>"`: 위치 프로그램을 같은 프로세스의 스레드로 돌립니다. 인자·큐 규칙(`--queue`, `--block`/`--drop`, 그래프 블록은 버리지 않음)은 `ondevice slam` 과 같고, 기억 폴더는 그 장면 스트림을 `--stream` 으로 받았을 때와 바이트까지 같습니다.
- `--serve "<포트> [뷰어 인자]"`: 뷰어 서버를 같은 프로세스에서 띄웁니다. 입력이 끝나도 뷰어는 계속 떠 있고(Ctrl-C 로 끝냄), `--serve-exit` 를 주면 입력이 끝날 때 함께 끝납니다.
- 부분만 돌리기: `ondevice slam …`(위치만, 장면 스트림을 파일·표준출력으로), `ondevice view <기억 폴더> …`(뷰어만), `ondevice --stream <파일|-> …`(녹화한 장면 스트림으로 물체 기억).

검색 모델은 Hugging Face 의 SigLIP2 base patch32-256 을 ONNX 로 내보낸 것으로, `models/siglip2-base-patch32-256/` 에 `vision_model.onnx`(물체 기억이 물체 사진에), `text_model_fp16.onnx`·`tokenizer.json`(뷰어가 검색어에) 을 둡니다. `--embed`·`--search-model` 을 빼면 검색창은 저장된 이름의 글자만 비교합니다.

`vocab/names.voc` 는 이름 1,225 개(LVIS v1 의 범주 이름(CC BY 4.0) · COCO, ObjectSAM 라벨에 쓴 문 · 창 · 계단 · 바닥 · 벽 등, 목록 `vocab/names.txt`)의 글 벡터를 PC 에서 미리 계산해 둔 파일입니다. 물체 기억은 `--vocab` 으로 물체 사진이 가장 닮은 이름을 물체 이름으로 쓰고(tv → monitor, 몫 0.3 이상일 때), 뷰어는 같은 파일로 이름들의 글 벡터 계산을 건너뜁니다. 목록을 바꾸면 `ondevice view --search-model models/siglip2-base-patch32-256 --encode-vocab vocab/names.txt vocab/names.voc` 로 다시 만듭니다.

## 보드에서 실행 — Raspberry Pi 5 + DX-M1 + D435

보드도 같은 한 명령입니다: D435 → 위치 추적 → 물체 기억 → 뷰어가 한 프로세스. 위치 프로그램의 원본(C++ `rtab_bridge`)과의 바이트 동일 확인은 PC(x86) 기준으로 마쳤고, 보드(ARM)에서의 같은 비교는 남아 있습니다(방법: [docs/board_bridge.md](docs/board_bridge.md)).

```bash
cargo build --release --no-default-features --features dxrt   # target/release/ondevice (D435 입력은 rsdriver: V4L2 직접)
M=~/dx-all-suite_my/workspace/res/models

target/release/ondevice --slam "--realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file data/memory/live/reset.req" \
    --every 1 --live 1 --seg $M/yolo26-s-seg_640x640.dxnn --max-depth 3.5 --out data/memory/live \
    --serve "8091 --bind 0.0.0.0"     # 같은 네트워크의 PC·폰에서 http://<보드 주소>:8091
```

- **D435 입력 고르기:** 리눅스 빌드의 기본은 `rsdriver`(librealsense2 없이 V4L2·UVC 직접)입니다. librealsense2 판은 `--features slam-realsense` 로 빌드하고 `--slam "--realsense --driver librealsense …"` 로 고릅니다. rsdriver 의 장면 시각은 컬러 장치 시각에 첫 장면에서 한 번 정한 시스템 시계 차이를 더한 값이라(librealsense 의 global time 처럼 시계 차이를 계속 따라가지 않음) 두 판의 절대 시각은 다를 수 있고, 장면 간격은 같습니다. rsdriver 입력은 실제 카메라 확인이 남아 있습니다.

- **아무 말로 찾기(보드):** 물체 기억에 `--embed <SigLIP2 영상 인코더>.dxnn`(NPU), 뷰어에 `--search-model <text_model_fp16.onnx 와 tokenizer.json 이 든 폴더>` 를 더합니다. 글 인코더는 뷰어가 뜬 뒤 1 ~ 2 분 걸려 올라오고, 그 전에는 이름 글자만 비교합니다. NPU 용 인코더를 만드는 법과 정확도는 `docs/performance_log.md` 의 SigLIP2 절.
- **움직일 때 번지면:** `--slam "… --exposure 8 --gain 110"` 으로 컬러 노출을 8 ms 로 고정합니다(기본은 자동 노출).

- **보드 화면에서 볼 때:** `chromium --password-store=basic http://127.0.0.1:8091`(키링 암호 창이 브라우저를 막지 않게). 보드 GPU에서는 뷰어가 자동으로 Low 화질로 시작합니다.
- **NPU는 한 번에 한 프로그램만** 쓰세요. 두 개가 동시에 쓰면 NPU 연결이 끊겨 재부팅해야 할 수 있습니다. 새로 돌리기 전에 `pkill -f 'ondevice'`.

## 주요 옵션 (ondevice)

| 옵션 | 뜻 |
|---|---|
| `--slam "<위치 인자>"` | 위치 프로그램을 같은 프로세스에서 돌려 장면을 메모리로 받음(인자는 `ondevice slam` 과 같음) |
| `--stream -\|<파일>` | 녹화한 장면 스트림 입력(`ondevice slam` 이 쓴 파일, 표준입력) |
| `--serve "<포트> [뷰어 인자]"` · `--serve-exit` | 뷰어 서버를 같은 프로세스에서 띄움(기억 폴더에 쓴 파일은 메모리에서 보냄) / 입력이 끝나면 함께 끝냄 |
| `--rebuild-every S` / `--no-rebuild` | 루프 클로저 뒤 다시 쌓기 간격(초, 기본 5) / 끄기 |
| `--no-drop` | 밀려도 장면을 버리지 않음(재생 결과를 늘 같게) |
| `--threads auto\|none\|…` | 스레드 우선순위·코어 배치(기본 auto; 역할 main · reader · pts · rebuild · writer · infer · slam, 예 `slam=0:2-3` 은 위치 프로그램 풀을 코어 2·3 에) |
| `--npu-jobs K` | 스트림 분할 장면을 NPU 에 K 장까지 함께 올림(NPU 코어 여럿이 나눠 돎, 기본 4, `1` 은 한 장씩) |
| `--seg <모델>` / `--conf 0.35` | 물체 분할 모델 / 물체로 인정할 확신도 |
| `--embed <vision_model.onnx>` | 물체 사진마다 검색용 벡터를 남김(뷰어 검색, 물체 병합에 씀) |
| `--vocab <글 벡터 파일>` | 물체 사진이 가장 닮은 이름을 물체 이름으로 씀(`--embed` 와 함께, `vocab/names.voc`) |
| `--seg-names <이름,…>` | 분할 모델의 클래스 이름(기본 COCO 80 개). 물체만 자르는 1 클래스 모델(ObjectSAM)은 `object` |
| `--max-depth` | 지도에 넣을 최대 거리(m) |
| `--every N` | 받은 장면 N장마다 처리 |
| `--live 1` / `--loop` | 실제 속도로 재생하며 뷰어에 계속 갱신 / 끝나면 반복 |
| `--out <폴더>` | 기억을 저장할 폴더(뷰어가 이 폴더를 읽음) |

## 뷰어 보는 법

- **마우스:** 끌면 돌리기, 휠은 확대·축소, 오른쪽 끌기는 이동.
- **오른쪽 위 도구줄:** `Fit`(전체 맞추기) · `Top`(위에서 보기) · `Demo`(천천히 자동 회전) · `High/Low`(화질) · `Settings`(설정 패널).
- **위쪽 검색창(`/` 키):** 찾을 물체를 아무 말로나 칩니다. 맞는 물체가 점수 순으로 왼쪽에 뜨고 지도에 초록 윤곽으로 표시됩니다. Enter 는 가장 맞는 물체로 이동.
- **오른쪽 아래 새로고침 아이콘:** 리셋. 위치를 다시 잡고 지도를 처음부터 쌓습니다.
- **화질:** High는 PC용(점 섞기로 가까이서도 면처럼 보임), Low는 보드·폰용입니다. Low는 돌리는 동안 절반 해상도로 가볍게 그리고 멈추면 0.2초 뒤 원래 화질로 다시 그립니다.
- **화면 구성:** 색 점이 3D 점 지도, 파란 원뿔이 지금 카메라(위치를 놓친 동안은 빨간색), 위쪽 층이 물체와 방 노드입니다.
- **물체 색:** 초록 = 지금 보임, 주황 = 옮겨짐, 회색 = 그 자리에서 사라짐.
- **설정 패널에서 켜고 끄기:** `3D map points`(점 지도), `Object name labels`(이름표), `Show graph layers`(물체·방 층) 등.
- **시연할 때:** `Follow the handheld camera (behind)`를 켜면 시점이 손카메라 뒤에서 따라갑니다(마우스로 돌리면 꺼짐).

설명 문서: 연결·루프 클로저·다시 쌓기는 [docs/bridge.md](docs/bridge.md), 뷰어 세부는 [viewer/sgview/README.md](viewer/sgview/README.md).

## 라이선스

[PolyForm Noncommercial License 1.0.0](LICENSE.md) — 개인 공부·연구·교육·비영리 목적으로는 자유롭게 쓰고 고치고 나눌 수 있고, **상업적 이용은 허용하지 않습니다.** 나눌 때는 `LICENSE.md` 의 저작권 표시(Required Notice)를 함께 전합니다.

저장소에 함께 든 외부 코드(three.js, OrbitControls), `slam/` 에 Rust 로 옮긴 코드(RTAB-Map, OpenCV, GTSAM, FLANN, CCOLAMD — 출처는 [slam/README.md](slam/README.md)), `tools/src/rsref.rs` 에 시험 기준으로 옮긴 librealsense 정렬 코드(Apache-2.0), 연결해 쓰는 라이브러리·모델(RTAB-Map, librealsense, OpenCV, GTSAM, SigLIP2, YOLO26, DEEPX 런타임)은 각자의 라이선스를 따릅니다.
