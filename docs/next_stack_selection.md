# 다음 단계 부품 선정 — RGB-D SLAM · 깊이 카메라 · 분할 모델

조사일 2026-11-05. 새 구성은 **깊이 카메라 + 검증된 RGB-D SLAM(루프 클로저·재위치 찾기) + NPU 에서 분할만**이다. 우리 Rust 쪽(memmap)은 SLAM 위치를 받아 지도·물체·씬그래프·뷰어를 맡는다.
"확인 필요"는 공개 자료로 확인하지 못해 보드나 PC 실측으로 정할 항목이다.

## 결론

**깊이 카메라는 Intel RealSense D455 로 확정**(2026-11-05 확보). PC 시험은 같은 계열인 D435i 공개 데이터로 했다. 시작은 IMU 없이(D455 에는 IMU 가 있어 나중에 더할 수 있음). 위치는 RGB-D 만으로 추적하고, SLAM 은 RTAB-Map 의 `CameraRealSense2` 로 붙인다. Pi 5 에서는 librealsense 를 소스 빌드(RSUSB 백엔드, 커널 패치 없이)하고, 깊이-컬러 정렬은 CPU 에서 SDK 의 NEON 경로 또는 같은 결과를 내는 `bridge/align` 의 fast_align 으로 한다(640×480 PC 0.6–0.9 ms, Pi 환산 약 2 ms, [`d455_input.md`](d455_input.md)). 아래 카메라 비교는 조사 기록으로 둔다.

| 항목 | 추천 | 차선 |
|---|---|---|
| RGB-D SLAM | **RTAB-Map 0.24 (C++ 라이브러리, F2M 시각 오도메트리)**: BSD-3이고 지금도 활발히 관리된다. 루프 클로저 뒤 고쳐진 키프레임 위치를 API 하나로 꺼낼 수 있고, Orbbec·RealSense 카메라 드라이버가 들어 있다. | **ORB-SLAM3 (RGB-D / RGB-D-관성)**: TUM 정확도가 가장 높고(1.6 cm / 1.0 cm) 0.2.0 때 빌드해 본 경험이 있다. GPL-3.0이고 2022년 이후 멈췄으며, 키프레임을 꺼내려면 헤더 패치가 필요하다. |
| 깊이 카메라 | **Orbbec Gemini 335**: 최적 거리 0.26–3 m, 2 m에서 오차 ≤1.5 %. IMU가 있고 USB-C 하나로 전원·데이터를 받는다. SDK는 오픈소스이고 하드웨어 깊이-컬러 정렬을 지원한다. 국내 약 46만 원. | **Intel RealSense D435i**: SLAM 생태계가 가장 크다(ORB-SLAM3 예제 기본 카메라). 국내 55.5만 원(VAT 별도). Debian 13에서 동작하는지는 확인 필요. |
| 분할 모델(DX-M1) | **YOLO26-l-seg 640²**: DEEPX 공식 모델이다. 지금 해독기를 그대로 쓰고 mAP가 +4.1 오른다(35.0 → 39.2). | **Mask-RT-DETR-L 640²**: DEEPX 공식, Apache-2.0, mAP 45.4로 가장 높다. 새 해독기가 필요하고 출력 크기와 PCIe 전송 시간은 확인 필요. |

## 1. RGB-D SLAM

### 비교

| | RTAB-Map | ORB-SLAM3 | stella_vslam |
|---|---|---|---|
| TUM fr1/desk ATE | 2.9 cm (F2M), 1.9 cm (ORB2 오도메트리) [R1] | 1.6 cm (ORB-SLAM2 RGB-D 값, 같은 핵심) [O1] | 확인 필요 |
| TUM fr3/office (= long_office_household) ATE | 2.1 cm (F2M) [R1] | 1.0 cm [O1] | 확인 필요 |
| 우리 DVO(Kinect 깊이, 비교 기준) | 책상 11 cm · 사무실 23 cm | | |
| 속도 근거 | F2M 오도메트리 70 ms/장(i7-3770 **한 코어**) [R1]. 공식 위키: RPi3에서 설정을 줄여 오도메트리 5–6 Hz [R2] | 추적 25.6 ms/장, 지역 BA 197 ms, 루프 뒤 전체 BA 1.6 s (i7-4790, fr3_office) [O1]. Pi 5 평가 논문: 일부 TUM RGB-D 순서에서 6 FPS 미만(요약 기준, 전문 확인 필요) [O3] | 확인 필요 |
| Pi 5 실측 | 확인 필요 | 위 [O3] 외 확인 필요 | 확인 필요 |
| 루프 클로저 | 있음(외형 기반 + 근접 탐지, 그래프 최적화) | 있음(DBoW2 + 포즈 그래프 + 전체 BA) | 있음(FBoW + 포즈 그래프 + 전체 BA) |
| 재위치 찾기 | 있음(같은 그래프 안에서 루프로 다시 연결) | 있음. 놓치면 새 지도를 만든 뒤 나중에 합친다(Atlas) | 있음 |
| 질감 적은 실내 강건성 | 저질감 TUM Pioneer 순서에서 셋 중 가장 정확 [C1] | 같은 순서에서 잘못된 재위치로 지도가 엇갈려 RMSE 1 m 이상 [C1] | RTAB-Map 다음 [C1] |
| 위치를 꺼내는 API | `Rtabmap::process(SensorData, odomPose)` → `getLocalOptimizedPoses()`, `getGraph(poses, links, optimized=true, global=true)`, `getLoopClosureId()`, `getMapCorrection()` [R3] | `System::TrackRGBD()`가 지금 위치를 돌려준다. 키프레임은 `Atlas::GetAllKeyFrames()` + `KeyFrame::GetPose()`, 큰 변화는 `Atlas::GetLastBigChangeIdx()`로 알 수 있는데 `mpAtlas`가 private라 꺼내는 함수를 붙이는 패치가 필요 [O2] | `feed_RGBD_frame()`이 위치를 돌려주고, `map_publisher::get_keyframes()`로 키프레임 목록을 받는다 [S1] |
| 노드별 데이터 저장 | 노드마다 RGB·깊이를 SQLite DB에 저장(다시 쌓기에 바로 쓸 수 있음) | 키프레임에 특징점만 | 키프레임에 특징점만 |
| 카메라 드라이버 | 내장: `CameraOrbbecSDK`, `CameraRealSense2`, `CameraDepthAI` 등 [R4] | 입력은 `cv::Mat`. RealSense D435i 예제 포함(RGB-D, RGB-D-관성) [O2] | 입력은 `cv::Mat` |
| IMU | 중력 방향 제약에 쓸 수 있음(설정은 확인 필요) | RGB-D-관성 모드 있음 | 확인 필요 |
| aarch64·Debian 13 빌드 | 소스 빌드. 필수는 OpenCV·PCL·SQLite, 선택은 g2o·GTSAM·Qt(GUI). Debian 13 패키지 없음 [R2] | 소스 빌드. CMake가 `-std=c++11`, OpenCV 4.4, Pangolin(필수)을 요구 → 플래그·뷰어 패치 필요 [O2] | 소스 빌드(g2o, FBoW 등) |
| 유지보수 | 0.24.2 (2026-10-25), 마지막 푸시 2026-11 | 마지막 커밋 2022-02-10 | 마지막 커밋 2026-08-26 |
| 라이선스 | BSD-3-Clause | GPL-3.0 | BSD-2-Clause(일부 의존 g2o 확장 LGPL) |

ORB-SLAM3는 단일 코어 속도가 PC의 절반 이하인 Pi에서 지역 BA·전체 BA가 추적 스레드와 코어를 나눠 쓴다. RTAB-Map은 오도메트리(매 장)와 지도 갱신(기본 1 Hz, `Rtabmap/DetectionRate`)이 나뉘어 있고, 기억 관리로 갱신 시간을 상한 아래로 묶는다(`setTimeThreshold`). 그래서 4코어를 Rust 지도·NPU 전후처리와 나눠 쓰는 우리 조건에 맞는다.

### 제외·참고 후보

| 후보 | 판단 |
|---|---|
| ElasticFusion · BAD SLAM · BundleFusion · KinectFusion 계열 | 제외: CUDA 데스크톱 GPU 필요(Pi 5 GPU는 VideoCore VII) |
| DROID-SLAM · MASt3R-SLAM 등 딥러닝 SLAM | 제외: 큰 GPU 필요 |
| cuVSLAM (Isaac ROS) | 제외: NVIDIA GPU(Jetson) 전용 |
| DVO-SLAM | 제외: ROS 1 전용, 오래 멈춤 |
| OpenVINS · Basalt | 참고: VIO(단안·스테레오 + IMU)라 RGB-D 깊이 입력 경로가 따로 없다. 카메라 IMU를 붙이는 방법을 참고할 수 있다. |
| stella_vslam | 3순위: BSD-2이고 RGB-D·루프·재위치를 모두 갖췄다. TUM RGB-D 정확도·Pi 속도 공개 자료는 확인 필요 |

## 2. 깊이 카메라

| | Orbbec Gemini 335 | RealSense D435i | RealSense D455 | OAK-D Pro | Orbbec Gemini 2 |
|---|---|---|---|---|---|
| 방식 | 능동 스테레오, 기준선 50 mm | 능동 스테레오, 기준선 50 mm | 능동 스테레오, 기준선 95 mm | 능동 스테레오, 기준선 75 mm, 기기 안에서 깊이 계산 | 능동 스테레오 |
| 실내 거리·정확도 | 0.10 m–(최적 0.26–3 m), 2 m에서 ≤1.5 % [G1] | 동작 거리 약 0.3–3 m [D1], 정확도 수치는 확인 필요 | 최적 0.6–6 m, 4 m에서 <2 %, 최소 약 52 cm(최대 해상도) [D3] | 최적 0.8–12 m, 4 m 아래 <2 %, 최소 20–40 cm(확장 시차) [K1] | 확인 필요 |
| 깊이 해상도·속도 | 최대 1280×800@30, 컬러 1920×1080@30 [G1] | 깊이 1280×720@30 [D1] | 컬러 1280×800 글로벌 셔터 [D3] | 컬러 4K@30 [K1] | 확인 필요 |
| IMU | 있음 [G1] | 있음(BMI055/085) | 있음(BMI055) [D3] | 있음(BNO086) [K1] | 확인 필요 |
| 깊이-컬러 정렬 | SDK 하드웨어 정렬 예제 `hw_d2c_align`, 소프트웨어 정렬 필터 [G2] | SDK `rs2::align` | SDK `rs2::align` | 기기 안 정렬 [K1] | SDK v2 지원 |
| SDK·C/C++ | OrbbecSDK v2 (C/C++, v2.10.6, 2026-09) [G2] | librealsense (C/C++, v2.59.1, 2026-10) [D2] | 같음 | DepthAI v3 (C++) | OrbbecSDK v2 |
| Pi 5 · Debian 13 | 공식 시험 목록은 Jetson뿐이다. Pi 5 + Ubuntu 24.04에서 335L 동작 사례가 있고, Raspberry Pi OS에서는 소스 빌드 [G3]. Debian 13은 확인 필요 | 공식 지원: "마지막으로 동작 확인된 Pi OS는 Bookworm", libuvc(RSUSB) 백엔드로 소스 빌드 권장 [D4]. Debian 13 apt 저장소는 403 [D5] → 소스 빌드, 확인 필요 | D435i와 같음 | 확인 필요 | Gemini 335와 같음 |
| 전원 | USB-C 하나로 전원·데이터 [G1], 소비 전력 확인 필요 | USB 3, 소비 전력 확인 필요 | 같음 | 기본 2.5–3 W, Pro는 최대 15 W라 외부 전원 + Y 어댑터 권장 [K1] → 손에 들기 불편 | 확인 필요 |
| 국내 입수·가격 | 아이씨뱅큐 460,000원 [G4], 해외 $264 | 디바이스마트 555,000원(VAT 별도), 입고 예정일 미정 [D6] | 확인 필요 | 국내가 확인 필요(일본 78,430엔) | 아이씨뱅큐 400,000원 [G4] |
| SDK 라이선스 | MIT(일부 서드파티 별도) | Apache-2.0 | Apache-2.0 | MIT(확인 필요) | MIT |

- 제외: **Orbbec Femto Bolt**. 깊이 계산이 호스트 GPU(OpenGL 4.4)에서 돌기 때문이다. Pi 5에서 "Depth engine create and initialize failed"가 보고됐고, Orbbec 포럼 답변도 "Raspberry Pi 미지원"이다 [G5].
- 제외: **RealSense D405**. 근거리 전용(수십 cm)이라 방 훑기 거리와 어긋난다.
- **Pi 5 USB 전류:** 5 V 5 A 전원(공식 27 W)일 때만 USB 포트 전체에 1.6 A가 나가고, 그 밖에는 600 mA다 [P1]. 카메라·DX-M1 총 전력을 보드에서 확인한다.
- Gemini 335를 고른 이유: 실내 최적 거리(0.26–3 m)가 손카메라로 책상·방을 훑는 거리와 맞고, 국내 재고가 있고 가장 싸다. IMU가 있고 깊이-컬러 정렬을 하드웨어로 해 Pi CPU를 아낀다. RTAB-Map에 드라이버(`CameraOrbbecSDK`)가 이미 있다.

## 3. 분할 모델 (DX-M1)

DEEPX 공식 모델 저장소([dx-modelzoo](https://github.com/DEEPX-AI/dx-modelzoo), [dx_app 예제 카드](https://github.com/DEEPX-AI/dx_app/tree/main/src/cpp_example/instance_segmentation))의 인스턴스 분할 모델이다. mAP는 DEEPX가 같은 평가기로 잰 COCO 값(INT8 Q-Lite)이고, `.dxnn`은 DX-COM 2.5.0으로 컴파일됐다.

| 모델 | 연산량 | mAP (원본 → Q-Lite) | 라이선스 | 비고 |
|---|---|---|---|---|
| YOLO26-s-seg (지금) | 19.9 GMACs | 35.6 → 35.0 | AGPL-3.0 | 보드 NPU 54 ms + CPU 전후처리 16 ms ([performance_log.md](performance_log.md)) |
| YOLO26-m-seg | 68.7 GMACs | 38.4 → 38.1 | AGPL-3.0 | 속도가 모자랄 때 대안 |
| **YOLO26-l-seg** | 78.5 GMACs | 39.7 → 39.2 | AGPL-3.0 | m보다 연산 +14 %에 mAP +1.1. 출력 모양은 s와 같다 |
| YOLO26-x-seg | 173.5 GMACs | 40.6 → 40.3 | AGPL-3.0 | 연산이 l의 2.2배 |
| Mask-RT-DETR-S (HGNetv2) | 30.3 GOPs | 40.9 → 40.6 | Apache-2.0 | |
| Mask-RT-DETR-M | 38.3 GOPs | 42.5 → 41.8 | Apache-2.0 | |
| **Mask-RT-DETR-L** | 62.7 GOPs | 45.5 → 45.4 | Apache-2.0 | 쿼리 300개마다 마스크 한 장을 출력한다(NMS 없음). 출력 크기와 PCIe 1세대 전송 시간은 확인 필요 |
| YOLO11-l-seg / YOLOv8-l-seg | 74.6 / 113.1 GOPs | 43.0 → 37.5 / 44.0 → 37.1 | AGPL-3.0 | 양자화 손실이 커서(−5.5 / −6.9) 제외 |
| FastSAM-s 1024² | 57.2 GOPs | 13.0 → 12.0 (클래스 없는 분할) | Apache-2.0 | 열린 어휘의 마스크 단계로만 쓸 수 있다 |

- **NPU 속도:** DEEPX 공개 자료에서 모델별 DX-M1 지연 시간은 찾지 못했다(확인 필요). 연산량은 l이 s의 3.9배다. 깊이·SuperPoint가 빠지면서 NPU는 분할 하나만 돌리고, 분할은 키프레임마다(초당 2–5회)면 충분하므로 l을 받아들일 여유가 있다. 보드 PCIe 1세대에서 실측해 정한다.
- **열린 어휘:** 공식 저장소에 YOLOE·YOLO-World는 아직 없다. 대신 **CLIP 이미지·텍스트 인코더**(`clip-img_*`, `clip-text_resnet50_77x512_openai`)가 NPU용으로 있다. 다음 단계인 물체 검색("텀블러 어디 있어?")은 물체 사진마다 CLIP 이미지 임베딩을 저장하고 텍스트 임베딩과 비교하는 방식으로 붙일 수 있다(ConceptGraphs 방식). YOLOE를 고정 클래스로 내보내 DX-COM으로 컴파일하는 길도 있으나 연산자 지원은 확인 필요.
- **판단:** 분할 품질을 올릴 여지가 크다. YOLO26-l-seg는 지금 해독기 그대로 파일만 바꾸면 되므로 먼저 넣는다. Mask-RT-DETR-L은 mAP가 +10이고 라이선스도 더 자유롭지만, 출력 전송과 해독기 이식을 보드에서 재 본 뒤 고른다.

## 4. 우리 파이프라인 연결

```
[C++ slamd 프로세스]                                 [Rust memmap]
 카메라 SDK(RGB, 정렬된 깊이, IMU, 보정값)
   → RTAB-Map 오도메트리(매 장) ──── 공유 메모리 링 ──→ 장면: RGB · 깊이 · 위치 · 추적 상태 · 기준 노드
   → Rtabmap::process(키프레임)                         NPU 분할(키프레임) → 물체
   → 루프·재위치 → 최적화 그래프 ── 그래프 갱신 ──→ 키프레임 위치 표 → 고쳐진 키프레임만 다시 쌓기
```

### 위치를 Rust로 받는 방법: 별도 C++ 프로세스(추천)

- `slam/` 폴더(C++/CMake)에 `slamd`를 둔다. 카메라를 혼자 열고 SLAM을 돌린 뒤, 결과를 `/dev/shm`의 링 버퍼에 쓴다. 장면마다 시각, RGB, 정렬된 깊이(u16 mm), 위치 T_wc, 추적 상태, 기준 노드 id와 그 노드에서의 상대 위치를 담는다.
- 그래프가 바뀌면(루프 클로저 `getLoopClosureId() > 0`, 재위치, 최적화) `getLocalOptimizedPoses()`의 {노드 id → 위치} 표 전체를 버전 번호와 함께 한 블록에 쓴다. 키프레임 1,000개라도 64 KB다.
- Rust는 `--slam /dev/shm/slamd` 입력 모드로 이 링을 읽는다. 장면 형식이 TUM Kinect 입력(`--depth kinect`)과 같아 지도·물체 코드가 그대로 돈다. 보정값(fx, fy, cx, cy)은 SDK가 주므로 체커보드 보정이 필요 없다.
- 프로세스를 나누면 좋은 점: cargo 빌드는 Rust만으로 유지되고, SLAM이 죽어도 지도가 남는다. ORB-SLAM3(GPL)로 바꿔도 Rust 쪽은 그대로다.
- 대안 FFI(`cxx` 크레이트 + C 래퍼): 한 프로세스로 지연을 줄일 수 있지만, 빌드가 무거워지고 스레드가 섞인다. 링 지연이 문제가 될 때만 쓴다.

### 루프 클로저 뒤 지도·물체 다시 쌓기

1. **키프레임 저장소(Rust):** 노드 id마다 쌓은 시점의 위치, 깊이(320×240 u16, 약 150 KB), 작은 RGB, 분할 결과(물체별 마스크·이름·점수), 그 노드에서 나온 물체 관측 id를 둔다. 1,000개면 약 200 MB로 8 GB 안에 들어간다.
2. **지도 = 키프레임들의 합:** TSDF·점 지도는 키프레임(과 키프레임에 상대 위치로 매단 장면)만으로 쌓는다. 장면의 세계 위치는 `기준 노드 위치 × 상대 위치`로 계산해, 노드가 고쳐지면 장면도 따라 움직이게 한다.
3. **고쳐진 키프레임만 다시:** 새 그래프에서 위치가 1 cm·0.5° 넘게 바뀐 노드만 고른다. TSDF는 가중 평균이라 옛 위치로 빼고(de-integrate) 새 위치로 더한다. 점 지도는 영향받은 64 cm 구역만 비우고 그 구역을 본 키프레임으로 다시 채운다. 이 작업은 백그라운드 스레드에서 하고 끝나면 바꿔 끼운다.
4. **물체:** 물체는 관측(노드 id + 카메라 좌표 점)의 묶음으로 둔다. 세계 위치는 새 노드 위치로 다시 계산하고, 기존 병합 후처리(3D IoU·이름·색)를 한 번 돌린다. 점유 지도·방 나누기·씬그래프는 지금처럼 주기적으로 다시 만든다.
5. 뷰어에는 그래프 버전을 넘겨, 고쳐진 순간 지도가 "맞춰지는" 모습을 시연 장면으로 쓴다.

### 깊이 카메라 입력부

- 640×480(또는 848×480) 깊이 + 컬러 15–30 장/초, 깊이는 컬러에 하드웨어 정렬, IMU는 SDK 콜백으로 받는다.
- 깊이 후처리(구멍 메우기·시간 필터)는 카메라 안 또는 SDK 필터로 한다. 최대 거리는 지금처럼 `--max-depth`(기본 4 m)로 자른다.
- 지금 웹캠(ABKO)은 녹화·비교용으로 남는다.

## 5. 위험과 먼저 확인할 것 (보드 실측)

| 순위 | 위험 | 먼저 확인할 것 |
|---|---|---|
| 1 | **Pi 5에서 SLAM 속도**: Pi 5 RGB-D SLAM 공개 실측이 거의 없다(ORB-SLAM3 6 FPS 미만 보고, RTAB-Map은 RPi3 5–6 Hz만). Rust 지도·분할 전후처리·뷰어와 4코어를 나눠 쓴다. | RTAB-Map F2M 오도메트리 ms/장(640×480, 특징 500–1000개, `Odom/ImageDecimation` 1·2)을 분할을 함께 돌리며 잰다. 목표 10–15 장/초. 모자라면 특징 수 줄이기, 320×240 오도메트리, 코어 고정(SLAM 2코어 / Rust 2코어) 순으로 맞춘다. |
| 2 | **카메라 SDK × Pi 5 × Debian 13**: Orbbec은 Jetson만, RealSense는 Bookworm까지만 공식 확인됐다. 커널 6.12와 USB 3 안정성, USB 전류 600 mA 제한이 걸린다. | OrbbecSDK v2 소스 빌드 → 깊이+컬러+IMU 30 장/초로 10분 연속 스트림, 정렬 확인. `usb_max_current_enable`·5 A 전원 확인. 산 카메라로 RTAB-Map `CameraOrbbecSDK` 동작 확인. |
| 3 | **루프 클로저 뒤 다시 쌓기 시간**: 최적화 직후 키프레임 수백 개를 다시 넣으면 수 초가 걸릴 수 있다. | TUM 사무실(키프레임 약 200–300개)로 PC와 보드에서 재적분 시간을 잰다. 길면 영향 구역만 다시 쌓기와 백그라운드 교체로 줄인다. |
| 4 | 분할 모델 파일 호환·속도: 공식 `.dxnn`은 DX-COM 2.5.0으로 컴파일됐고 보드 런타임은 DXRT 3.4.2다. PCIe는 1세대다. | YOLO26-l-seg, Mask-RT-DETR-L을 보드에 올려 로드 여부·ms·출력 크기를 잰다. |
| 5 | 발열: SLAM + NPU + 카메라를 장시간 돌린다. | 시연 길이(10–15분) 동안 CPU 클럭·온도를 기록한다(능동 쿨러). |

## 6. 첫 단계 계획 — PC에서 TUM으로 SLAM 비교

목표: 같은 Kinect 깊이로 추천 SLAM이 우리 DVO(책상 0.11 m, 사무실 0.23 m)보다 얼마나 정확한지, 루프 클로저가 사무실 순서에서 언제 일어나 오차를 얼마나 줄이는지 숫자로 남긴다.

1. **빌드(PC, Linux/WSL2):** RTAB-Map 0.24.2 핵심 라이브러리와 도구(GUI 제외), ORB-SLAM3(`rgbd_tum`, 비교용).
2. **RTAB-Map 실행:** `rtabmap-rgbd_dataset` 도구 [R5]로 fr1/desk와 fr3/long_office_household를 두 설정으로 돌린다. 하나는 기본값(F2M), 다른 하나는 Pi 흉내(특징 500개, 4코어 제한). 최적화된 노드 위치와 장면별 오도메트리 위치를 TUM 형식으로 뽑는다.
3. **ORB-SLAM3 실행:** `rgbd_tum` + `TUM1.yaml`/`TUM3.yaml` → `CameraTrajectory.txt`.
4. **같은 잣대로 평가:** 우리 `bench`와 같은 정렬(SE(3) Horn)로 ATE를 잰다. 표에는 ATE, 루프 클로저 횟수와 일어난 장면 번호, 루프 끈 경우 대비 개선, 장면당 ms, 추적 실패 수를 넣는다. 기준 행은 우리 DVO다.
5. **우리 지도에 넣어 보기:** memmap에 SLAM 위치 파일 입력(TUM 형식, 정답 위치 자리)을 붙여 TSDF 메쉬를 만들고, `surfcmp`로 DVO 메쉬와 비교한다. 이어서 키프레임 그래프 갱신을 흉내 내 다시 쌓기 시간을 잰다.
6. 결과가 좋으면 보드로: RTAB-Map Pi 빌드 → TUM 재생 속도 → 카메라 도착 후 실시간 → `slamd` 공유 메모리 연결.

## 출처

- [R1] Labbé & Michaud, RTAB-Map 논문(JFR 2019), 표 6·실험 환경: https://arxiv.org/abs/2403.06341
- [R2] RTAB-Map 설치 위키(Raspberry Pi 절): https://github.com/introlab/rtabmap/wiki/Installation
- [R3] `Rtabmap.h`: https://github.com/introlab/rtabmap/blob/master/corelib/include/rtabmap/core/Rtabmap.h , 릴리스: https://github.com/introlab/rtabmap/releases
- [R4] 카메라 드라이버 목록: https://github.com/introlab/rtabmap/tree/master/corelib/include/rtabmap/core/camera
- [R5] TUM 형식 평가 도구: https://github.com/introlab/rtabmap/tree/master/tools/RgbdDataset
- [O1] Mur-Artal & Tardós, ORB-SLAM2 논문 표 III·IV(i7-4790): https://arxiv.org/abs/1610.06475
- [O2] ORB-SLAM3 저장소(`include/System.h`, `include/Atlas.h`, `CMakeLists.txt`, RealSense 예제): https://github.com/UZ-SLAMLab/ORB_SLAM3
- [O3] An In-depth Evaluation of ORB-SLAM3 on the Raspberry Pi 5 (IEEE): https://ieeexplore.ieee.org/document/11373149/
- [C1] Merzlyakov & Macenski, A Comparison of Modern General-Purpose Visual SLAM Approaches(2021): https://arxiv.org/abs/2107.07589
- [S1] stella_vslam: https://github.com/stella-cv/stella_vslam
- [G1] Orbbec Gemini 335 사양·가격: https://store.orbbec.com/products/gemini-335
- [G2] OrbbecSDK v2(지원 플랫폼·예제·라이선스): https://github.com/orbbec/OrbbecSDK_v2
- [G3] Gemini 335L Pi 5 동작 사례: https://github.com/dwifk0/orbbec-gemini335l-ros2
- [G4] 아이씨뱅큐 ORBBEC 기획전: https://www.icbanq.com/A05_templete/templeteList.do?catg_code=101106109&order=&t_idx=224
- [G5] Femto Bolt × Pi 5 깊이 엔진 실패: https://3dclub.orbbec3d.com/t/femto-bolt-raspberry-pi-5-depth-engine-create-and-initialize-failed/4285
- [D1] Intel D435i 사양: https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html
- [D2] librealsense: https://github.com/realsenseai/librealsense
- [D3] D455 사양: https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d455.html
- [D4] RealSense 지원 포럼(Pi 5 · Trixie): https://support.realsenseai.com/hc/en-us/community/posts/48193960122899
- [D5] Debian 13 apt 저장소 403: https://github.com/realsenseai/librealsense/issues/14277
- [D6] 디바이스마트 D435i: https://www.devicemart.co.kr/goods/view?no=13017225
- [K1] Luxonis OAK-D Pro: https://docs.luxonis.com/hardware/products/OAK-D%20Pro
- [P1] Raspberry Pi 5 USB 전원 백서: https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf
- DEEPX 분할 모델 mAP·연산량: dx_app 모델 카드 `src/cpp_example/instance_segmentation/*/README.md`, dx-modelzoo YAML

## PC 첫 시험 (2026-11-05, D435i 실데이터)

Digiteo seq2(RealSense D435i, 연구소 복도 58 m 한 바퀴, 1,658 장, 깊이 투사 켬)를 `rgbd2tum` 으로 640×480 으로 바꿔 RTAB-Map 0.24.2(Windows, 장면마다 노드)로 돌림. 정렬 후 위치 오차(`ate`, 정답 332 개):

| 위치 | 위치 오차 | 최대 |
|---|---|---|
| **RTAB-Map** | **0.37 m** | 0.58 m |
| 우리 DVO | 6.58 m | 12.4 m |

RTAB-Map 처리 155 초(PC, 1,658 장). 그 위치 + D435 깊이 + YOLO-seg 로 memmap 표면 지도: 복도 한 바퀴가 이중벽 없이 닫힘, 바닥은 D435 깊이 잡음으로 군데군데 비어 보임. 컬러가 흑백인 데이터라 물체는 6 개.

TUM 사무실(Kinect 실측 깊이, 물체 많은 장면, IMU 없음)에서 RTAB-Map(장면마다 노드): 위치 오차 **2.0 cm**(최대 5 cm). 이 위치로 만든 메쉬가 정답 위치 메쉬와 5 cm 안 99 % 겹침(`surfcmp` F 0.991, 중앙 0.9 cm), 물체 49 개. 비교할 땐 `ate <추정> <정답> 0.02 <정렬본>` 으로 SLAM 경로를 정답 좌표계로 옮겨 `memmap --gt <정렬본>`.

### 시연 설정 비용 (PC 코어 하나로 묶어 잼, TUM 사무실 2,488 장 ≈ 83 초)

| 설정 | 위치 추적(매 장) | 지도 갱신(1 초에 1 번) | 위치 오차 |
|---|---|---|---|
| A 기본(``Rtabmap/DetectionRate 1``) | 23 ms(95 % 27) | 38 ms(최대 116) | 1.9 cm |
| B 가볍게(+ ``Vis/MaxFeatures 400``, ``Kp/MaxFeatures 300``, ``Odom/ImageDecimation 2``, ``OdomF2M/MaxSize 1000``) | **11 ms**(95 % 14) | 29 ms(최대 76) | 3.6 cm |

Pi 5 추정(A76 코어가 이 PC 코어보다 2–3 배 느리다고 볼 때): A 15–20 장/초, B 30 장/초 근처. 지도 갱신은 따로 도는 스레드로 1 초에 약 100 ms. 보드 실측 필요.
