# 작업 규칙 (ondevice-scene-memory)

## 버전 관리 — Keep a Changelog + Semantic Versioning

변경 기록은 [Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/) 모범 사례를, 버전 번호는 [Semantic Versioning 2.0.0](https://semver.org/lang/ko/)을 따른다.

### CHANGELOG.md 쓰는 법 (keepachangelog.com/ko/1.1.0 원칙 그대로)
사이트의 지침 원칙: 사람을 위한 것이다 · 모든 버전에 항목이 있어야 한다 · 같은 종류의 변경은 묶는다 · 버전과 섹션은 링크할 수 있어야 한다 · 최신 버전이 먼저 · 각 버전에 출시일 · 시맨틱 버저닝을 따른다고 밝힌다.
사이트가 꼽는 나쁜 관행: 커밋 로그를 그대로 옮기기 · 폐기 예정(Deprecated)·삭제·호환 깨짐을 빠뜨리기 · 지역마다 다른 날짜 표기 · 중요한 변경을 일부만 적기(사소한 건 빼도 되지만 중요한 건 빠짐없이).

- 루트 `CHANGELOG.md` 한 파일. 머리말은 `# Changelog` + "이 프로젝트의 주요 변경 사항을 기록한다" + Keep a Changelog·Semantic Versioning 을 따른다는 문장.
- git 로그를 옮겨 적지 않고 "무엇이 달라졌고 왜 중요한지"를 쓴다. 호환이 깨지는 변경(view.json 형식, 명령 인자 기본값, 모델 교체)은 반드시 적는다.
- 맨 위는 항상 `## [Unreleased]`. 작업 단위를 커밋할 때 그 단위의 항목을 **같은 커밋 안에서** 여기에 추가한다.
- 버전 하나 안에서 같은 묶음 제목은 한 번만 쓴다(`### Added` 가 두 번 나오면 안 된다). 순서는 Added → Changed → Deprecated → Removed → Fixed → Security, 빈 묶음은 쓰지 않는다.
  - `Added` 새 기능 · `Changed` 기존 동작 변경 · `Deprecated` 곧 없앨 것 · `Removed` 없앤 것 · `Fixed` 버그 수정 · `Security` 보안
- 항목 하나 = 사용자가 체감하는 변화 하나. 같은 기능을 여러 번 고쳤으면 마지막 상태로 한 줄에 합친다(중간 시행착오는 쓰지 않는다).
- 수치 결과는 숫자로: `VIO ATE 0.64 m (ARKit 같은 구간 0.64 m)`, `물체 147 → 52`. 비교 그림은 `docs/results/` 경로를 단다.
- 심각한 버그로 거둬들인 버전은 지우지 않고 제목에 표시: `## [0.3.1] - 2026-11-05 [YANKED]`.
- 날짜는 ISO 8601(`YYYY-MM-DD`), 최신 버전이 위. 파일 맨 아래에 비교 링크:
  `[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/vX.Y.Z...HEAD`, `[X.Y.Z]: .../compare/vPREV...vX.Y.Z`.

### 버전 번호와 릴리스
- 1.0.0 전까지는 `0.y.z`.
  - MINOR(`0.y.0`): 파이프라인 단계가 새로 되거나 결과 형식(view.json·모델 입출력)이 바뀔 때 — VIO 연결, 새 데이터셋 입력, 씬그래프 층, 모델 교체, NPU 포팅.
  - PATCH(`0.y.z`): 버그 수정·튜닝·문서·캡처만.
  - `1.0.0`: Pi5 + DX-M1 온보드 엔드투엔드가 실제로 돌 때.
- 릴리스 시점: 사용자에게 결과를 보여 주고 한 단계가 정리됐을 때(대개 MINOR 하나). 그때
  1. `[Unreleased]` 내용을 `## [X.Y.Z] - 날짜` 로 옮기고 묶음을 정리, 새 빈 `[Unreleased]` 를 연다,
  2. 같은 커밋에 비교 링크 갱신,
  3. `git tag -a vX.Y.Z -m "vX.Y.Z"` 후 `git push origin main vX.Y.Z`.

## 커밋·푸시 단위 — 크게 묶는다

- **커밋은 작업 단위(목표 하나가 끝까지 된 상태)마다 한 번.** 한 단위 = 코드 + 결과 캡처 + 문서 + CHANGELOG 항목이 함께 들어간 커밋.
  - 예: "TUM 입력 + 깊이 모델 비교 + Depth Anything 연결" = 한 커밋, "씬그래프 층(방·엣지) + 뷰어 표시 + 캡처" = 한 커밋.
  - 하지 않는다: 캡처 한 장, 숫자 한 개 튜닝, 점 크기 조정, 오타 같은 것을 따로 커밋. 이런 것은 진행 중인 단위에 합쳐 두었다가 그 단위와 같이 커밋한다.
  - 사용자와 주고받으며 여러 번 고치는 중이면 커밋하지 말고, 사용자가 결과를 받아들이거나 다음 주제로 넘어갈 때 한 번에 커밋한다.
- 단위가 끝나면 바로 `git push origin main`. 깨진 상태(실행 안 되는 코드)로는 푸시하지 않는다.
- 커밋 메시지: 첫 줄 = 이 단위가 해낸 것(한국어), 본문 = 핵심 변경 목록·수치. Claude 공동작성자·세션 줄은 넣지 않는다.
- 커밋 작성자: `juyoung020 <151780134+juyoung020@users.noreply.github.com>`.
- 큰 파일(models/, data/, *.pt, *.onnx, *.dxnn)은 커밋하지 않는다(.gitignore). 결과 이미지는 `docs/results/` 에.
- 내가 만들지 않은 파일(예: 팀원이 둔 `board/`, `compile/`)은 묻지 않고 커밋하지 않는다.
- 이 저장소는 비공개다. `train/ObjectSAM` 내용을 공개 ObjectSAM 저장소로 푸시하지 않는다.
