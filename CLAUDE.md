# 작업 규칙 (ondevice-scene-memory)

## 버전 관리 — Keep a Changelog + Semantic Versioning

변경 기록은 [Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/) 모범 사례를, 버전 번호는 [Semantic Versioning 2.0.0](https://semver.org/lang/ko/)을 따른다.

- 루트 `CHANGELOG.md` 한 파일에 사람이 읽는 변경 기록을 남긴다. git 로그를 그대로 붙여넣지 않는다 — "무엇이 달라졌고 왜 중요한지"를 쓴다.
- 맨 위에 `## [Unreleased]` 를 두고, 작업 단위를 커밋할 때마다 해당 항목을 같은 커밋에 추가한다.
- 항목은 아래 6가지로만 묶는다(빈 묶음은 쓰지 않는다):
  - `Added` 새 기능 · `Changed` 기존 동작 변경 · `Deprecated` 곧 없앨 것 · `Removed` 없앤 것 · `Fixed` 버그 수정 · `Security` 보안
- 수치가 있는 결과(ATE, 지연, FPS, 물체 수 등)는 항목에 숫자로 남긴다. 예: `VIO ATE 0.64 m (ARKit 0.64 m, 같은 구간)`.
- 릴리스할 때 `[Unreleased]` 를 `## [X.Y.Z] - YYYY-MM-DD` (ISO 날짜) 로 바꾸고 새 빈 `[Unreleased]` 를 연다. 최신 버전이 위.
- 버전 규칙(1.0.0 전까지는 0.y.z):
  - MINOR(0.y.0): 파이프라인 단계가 새로 되거나 결과 형식(view.json·모델 입출력)이 바뀔 때 — 예: VIO 연결, NPU 포팅, 씬그래프 형식 변경
  - PATCH(0.y.z): 버그 수정·튜닝·문서·캡처
  - 1.0.0: Pi5 + DX-M1 온보드 엔드투엔드가 실제로 돌 때
- 릴리스마다 git 태그 `vX.Y.Z` 를 달고 태그도 푸시한다(`git push origin vX.Y.Z`).
- 파일 맨 아래에 버전별 비교 링크를 둔다: `[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/vX.Y.Z...HEAD`

## 커밋·푸시 단위

- 작은 수정마다 커밋하지 말고, **의미 있는 작업 단위로 크게 묶어** 커밋한다. 한 단위 = 하나의 목표가 끝까지 된 상태(코드 + 결과 캡처 + 문서 + CHANGELOG 항목).
  - 예: "VIO 시각 보정 찾고 고쳐서 ATE 측정까지" = 한 커밋, "뷰어 손카메라 표시 + 새 캡처" = 한 커밋.
- 단위가 끝나면 바로 `git push origin main`. 중간에 깨진 상태로 푸시하지 않는다.
- 커밋 메시지: 첫 줄에 무엇을 했는지(한국어 가능), 필요하면 본문에 이유·수치. Claude 공동작성자·세션 줄은 넣지 않는다.
- 커밋 작성자: `juyoung020 <151780134+juyoung020@users.noreply.github.com>`.
- 큰 파일(models/, data/, *.pt, *.onnx, *.dxnn)은 커밋하지 않는다(.gitignore). 결과 이미지는 `docs/results/` 에.
- 이 저장소는 비공개다. `train/ObjectSAM` 내용을 공개 ObjectSAM 저장소로 푸시하지 않는다.
