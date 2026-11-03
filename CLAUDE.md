# 작업 규칙 (ondevice-scene-memory)

## 언어
- 코드(파이프라인·뷰어·학습기)는 Rust 또는 C++/CUDA.
- 외부 도구 자체가 다른 언어인 경우는 사용자 확인을 받은 뒤에 쓴다.

## 문서 쓰는 법
- 문서는 지금 상태만 적는다. 없어진 것·바뀌기 전 내용은 고칠 때 지운다(기록은 git 과 CHANGELOG 에 있다).
- 하는 것만 적는다. "하지 않는다", "~는 안 쓴다" 같은 문장은 쓰지 않는다.

## CHANGELOG.md — Keep a Changelog 1.1.0 + Semantic Versioning 2.0.0
- 머리말: `# Changelog` + "이 프로젝트의 주요 변경 사항을 모두 이 파일에 기록한다." + [Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/)·[Semantic Versioning 2.0.0](https://semver.org/lang/ko/) 을 따른다는 문장.
- 사람이 읽는 글이다. 항목 하나 = 사용자가 체감하는 변화 하나, **짧은 한 줄**(무엇이 생겼다/바뀌었다/고쳐졌다, 마침표). 방법·수치·그림은 `docs/` 에.
- 맨 위 `## [Unreleased]`, 버전은 `## [X.Y.Z] - YYYY-MM-DD`, 최신이 위.
- 묶음은 이 순서로 한 번씩: `Added` 새 기능 · `Changed` 기존 동작 변경 · `Deprecated` 곧 없앨 것 · `Removed` 없앤 기능 · `Fixed` 버그 수정 · `Security` 보안. 빈 묶음은 생략.
- 호환이 깨지는 변경(view.json 형식, 명령 인자 기본값, 모델 교체)은 `**호환 변경:**` 을 붙여 적는다.
- 같은 기능을 여러 번 고쳤으면 마지막 상태 한 줄로. 커밋 전에 `[Unreleased]` 전체를 다시 읽고 낡은 항목은 고치거나 지운다.
- 거둬들인 버전은 제목에 `[YANKED]`.
- 맨 아래 비교 링크: `[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/vX.Y.Z...HEAD`, `[X.Y.Z]: .../compare/vPREV...vX.Y.Z`.

### 버전 번호와 릴리스
- 1.0.0 전까지 `0.y.z`. MINOR = 파이프라인 단계가 새로 되거나 결과 형식이 바뀜, PATCH = 버그 수정·튜닝·문서. `1.0.0` = Pi5 + DX-M1 온보드 엔드투엔드 완성.
- 한 단계가 정리되면 릴리스: `[Unreleased]` → `## [X.Y.Z] - 날짜`, 비교 링크 갱신, `git tag -a vX.Y.Z -m "vX.Y.Z"`, `git push origin main vX.Y.Z`.

## 커밋·푸시
- 커밋은 작업 단위(목표 하나가 끝까지 된 상태)마다 한 번: 코드 + 결과 캡처 + 문서 + CHANGELOG 항목을 함께.
  - 예: "TUM 입력 + 깊이 모델 비교" = 한 커밋, "씬그래프 층 + 뷰어 표시 + 캡처" = 한 커밋.
  - 캡처·튜닝·오타 같은 작은 수정은 진행 중인 단위에 합친다. 사용자와 주고받으며 고치는 동안은 모아 두었다가 주제가 끝날 때 한 번에.
- 단위가 끝나면 `git push origin main`(실행되는 상태로).
- 커밋 메시지: 첫 줄 = 해낸 것(한국어), 본문 = 핵심 변경·수치. 작성자 `juyoung020 <151780134+juyoung020@users.noreply.github.com>`, 작성자 줄만.
- 커밋 대상은 내가 만든 파일. 큰 파일(models/, data/)은 .gitignore, 결과 이미지는 `docs/results/`.
- 푸시는 이 비공개 저장소(origin)로만.
