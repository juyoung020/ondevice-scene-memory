# 실행 결과

## ADVIO 15 (사무실 52 초, 아이폰 카메라 + IMU) — 포즈: ARKit, 2026-11-03

학습 없이 추론만: YOLO26s-seg(분할) + YOLO26s-depth(깊이, m) + 포즈 → 점유 지도 갱신 + 물체 기억, 뷰어 sgview.

- 위치 오차(정답 대비 ATE): 0.93 m (걸은 거리 21 m 의 4.4 %)
- 물체 45 개(의자 22·테이블 10·소파 6·화분 3·책 3·침대 1), 사라짐 판정 4
- 키프레임 311 개, 0.12 s/키프레임 (RTX 5070 Ti)

![요약](advio15_arkit_summary.png)

| 위에서 | 3D |
|---|---|
| ![top](advio15_arkit_top.png) | ![3d](advio15_arkit_3d.png) |
