# 이어서 하기 (2026-11-19 밤 기준)

다른 컴퓨터 · 다른 세션이 바로 이어 받도록 지금 상태 · 다음 할 일 · 규칙을 모은 문서.

## 목표와 우선순위
- **IMU 판이 기본**. 1차 목표 = **방 하나를 겹침 · 중복 없이 쌓기**(위치 품질 먼저, 실시간으로 잘 돌 것).
- 고칠 땐 녹화로 숫자를 먼저 재고 하나씩 바꿔 비교한다(한 번에 한 가지). 나아진 것만 기본값에 넣는다.
- 설정은 실행 인자 말고 **코드 기본값**(`main.rs` · `app.rs` 의 기본값, `scripts/board_run.sh`)을 고친다. 다시 켜도 그대로여야 한다.

## 브랜치
| 브랜치 | 커밋 | 내용 |
|---|---|---|
| `opt/integrate` | `f94ca8e` | **시연판**. RTAB-Map 포팅 + IMU 융합. 보드에서 도는 판 |
| `slam/orb3` | `2101121` | ORB-SLAM3(적외선 스테레오 + IMU) 교체 실험. RTAB 포팅 삭제판. 실시간에서 RTAB 보다 나빠 보류 |

`opt/integrate` 최근 커밋(위가 최신):
- `f94ca8e` 지나온 길을 바닥에 늘 보이게(그래프 층과 따로) — 보드 빌드 성공, **뷰어 화면 확인은 아직**.
- `4de3a48` 위치가 확실한 장면만 지도에 넣음: 임시 위치(`PROVISIONAL`) · 불확실(`UNCERTAIN`) 장면은 카메라만 옮김(`pipeline/src/stream.rs` `no_map()`). 사용자 평: "많이 좋아졌다".
- `36bb96c` 놓친 뒤 새 지도는 이어질 때까지 맡아 둠(`imu_hold_wait` = 무한) · IMU 가 없으면 카메라 전용으로 계속.
- `8a850a7` IMU 모드(`GRAVITY`)에선 바닥 기울기 보정(`tilt_c`)을 끔: 가만히 있어도 지도가 기울며 겹치던 것.
- `e583ff6` 점 지도 깊이 4 m(D435 깊이 오차 약 6.7 cm @ 4 m) · 검색 · 뷰어 멈춤(처리 루프 33 ms 예산 + `serve_now`) · `SlamBackend` trait · 적외선 녹화.

## 보드
- Raspberry Pi 5 + DX-M1 NPU + D435(자체 V4L2 드라이버 `rsdriver`) + IMU ICM-42688-P(`/dev/i2c-1`, 주소 0x69).
- 접속: 랜선 직결 `ssh kau@kau.local`(169.254.167.99) 또는 Tailscale `kau@100.77.176.0`. 저장소 `~/ondevice-scene-memory`(= origin `opt/integrate`).
- 빌드(보드): `. ~/.cargo/env && cargo build --release --no-default-features --features dxrt` (약 5 분, 무거운 빌드 둘을 함께 돌리면 메모리 부족).
- 실행(실시간, IMU): `SLAM_EXTRA="--stats ~/live_stats_X.csv" bash scripts/board_run.sh > ~/live.log 2>&1` → 뷰어 `http://kau.local:8091/`.
- 녹화 재생(IMU 함께): `bash scripts/board_run.sh --rec ~/rec/<이름>`.
- **끌 때는 `pkill -INT -x ondevice`**. 그냥 `pkill`(SIGTERM/KILL)로 IMU 읽는 도중 끊으면 I2C SDA 가 낮게 잡혀 IMU 가 한동안 무응답("IMU 없음")이 된다. 원인은 I2C 버스 막힘(배선은 정상). 진단: 보드 `/tmp/whoami`(C, WHO_AM_I 0x47 응답 확인).
- `pkill -f` 는 자기 ssh 명령까지 죽이므로 `-x` 또는 PID 로.
- 장면별 통계 CSV 칸: 7 lost · 8 new_map · 9 map_corrected · 10 loop · 13 held · 15 uncertain · 16 method(5 = IMU 로 이어 붙임) · 23 no_map · 24 tilt_map. 로그(`live.log`)의 "루프 클로저 받음 … 옮겨진 노드 최대", "다시 쌓기: 노드 최대 … 움직임" 으로 보정 크기를 본다.
- 2026-11-19 밤 시연판이 켜진 채(옛 바이너리 PID 307231). 다음에 켤 땐 위 방법으로 끄고 다시 켜면 `f94ca8e` 가 뜬다.

## 녹화
- 보드 `~/rec/`: `ds_1118_121652`, `walk1`(v1), `ir_on1` · `ir_off1` · `ir_off2` · `ir_loop_off` · `ir_loop_on`(v2).
- v2 = 적외선 좌우(EuRoC 꼴 `mav0/cam0 · cam1`) + IMU 200 Hz(`imu0.csv`, `mav0/imu0/data.csv`) + 컬러 · 깊이. `ir_loop_*` 은 방 한 바퀴 돌고 처음 자리로 돌아옴(레이저 끔/켬). 녹화 도구 `scripts/rec_ir.sh`(slam/orb3).
- GitHub 릴리즈 `recordings-v1`(완료), `recordings-v2`: 2026-11-19 밤 노트북(`C:\Users\rlawn\rec\recordings-v2\upload.sh`)에서 남은 조각 올리는 중. 확인: `gh release view recordings-v2 --json assets -q '.assets[] | "\(.name) \(.size)"'` — 조각 8 개(`ir_loop_off.tar.part-00/01`, `ir_loop_on.tar.part-00/01`, `ir_off1/ir_off2/ir_on1.tar.part-00`) + `SHA256SUMS`. 빠진 게 있으면 노트북에서 `bash ~/rec/recordings-v2/upload.sh` 다시(같은 이름 덮어씀). 합치기: `cat X.tar.part-* > X.tar && sha256sum -c`.

## 남은 문제 (2026-11-19 실시간 6,876 장 기준)
- 위치 추적 이슈 전체(다시 붙지 않음 · 기울기 · 루프 보정 0 · 녹화 빈틈 · 멈춤)와 원인 · 할 일: [`pose_issues.md`](pose_issues.md)
- **"살짝 층"**: 새 지도는 2 번 생겼고 둘 다 1 초 안에 다시 이어짐(합치기는 됨). 원인은 **루프 보정 크기** — 방 안을 돌다 원래 자리를 알아봤을 때 노드를 0.61 m · 0.26 m 옮김(방 한 바퀴에 수십 cm 밀림). 점 지도 청크(`chunks.rs` `CHUNK_M` 0.6 m · `CHUNK_DEG` 25° · `CHUNK_KF` 30)가 노드 하나에 묶여 통째로 옮겨져, 덩어리 안의 다른 보정을 못 따라감.
- 물체 id `#100…`, `#500…` 은 다시 쌓기 때 붙는 번호(`mem.next + 100_000`)이지 중복 표시가 아님.

## 다음 할 일 (사용자가 고른 방향: VIO)
1. **기준 재기**: `ir_loop_off` · `ir_loop_on` 을 지금 시연판으로 재생(`board_run.sh --rec`)해 처음 자리로 돌아왔을 때 어긋남 · 루프 보정 크기 · 불확실 비율 · 물체 중복을 숫자로.
2. **VIO 후보를 같은 녹화로 하나씩**: 1순위 Basalt VIO(Pi 5 속도 · 추적 안정성), 다음 ORB-SLAM3 스테레오-관성(`slam/orb3`, 실시간에서 초기화가 불안정했던 원인을 녹화로). 숫자로 RTAB+IMU 보다 나을 때만 `SlamBackend` 구현으로 시연판에 붙임.
3. 그 밖에: 점 지도 청크 잘게(루프 보정 따라 휘게), 점 번짐(`--pts-obs 2` 기본값 여부, 사용자 결정 대기), `f94ca8e` 경로 표시 화면 확인, IMU 정상 종료(I2C 막힘 방지).
- 작업 범위는 IMU 판(사용자 지시).

## 규칙
- 코드는 Rust / C++ 만. 커밋 작성자 `juyoung020 <151780134+juyoung020@users.noreply.github.com>` 한 줄만. `CLAUDE.md` 의 CHANGELOG 규칙. 푸시는 origin 만.
- PC 의 GPU 는 다른 세션(강화학습)이 씀 → CPU 만. PC C: 여유가 빠듯함.
- 답은 한국어로.
