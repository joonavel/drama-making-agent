# Drama Making Agent

AI 기반 단편 드라마 자동 생성 시스템

## 개요

사용자의 간단한 아이디어로부터 32초 분량의 단편 드라마 영상을 자동으로 생성하는 AI 에이전트 시스템입니다. LangGraph 기반의 멀티 에이전트 워크플로우로 스토리 기획부터 영상 생성까지 전 과정을 자동화합니다.

## 주요 기능

- 📝 **스토리 기획**: 사용자 입력으로부터 스토리, 캐릭터, 비주얼 스타일 자동 생성
- 🎬 **연출 계획**: 5개의 키프레임과 4개의 샷 구성 자동 설계
- 🖼️ **이미지 생성**: Gemini 이미지 모델을 활용한 일관된 캐릭터 및 키프레임 생성
- 🎥 **비디오 생성**: 
  - Veo 3.1 API를 통한 고품질 비디오 생성
  - 실패 시 Kie API로 자동 폴백
- 🔄 **자동 후처리**: 4개의 비디오 클립을 하나의 완성된 영상으로 병합

## 시스템 구조

```
src/
├── config.py              # 전역 설정 및 유틸리티
├── prompts/               # 각 에이전트의 시스템/유저 프롬프트
└── workflows/
    ├── tasks.py           # Pydantic 데이터 모델 및 GraphState 정의
    ├── nodes.py           # 각 노드의 구현 (에이전트 로직)
    ├── graph.py           # LangGraph 워크플로우 정의
    └── main.py            # CLI 진입점
```

### 파일별 역할

#### `config.py`
- 프로젝트 전역 설정 (경로, API 키 등)
- 로거 설정
- 프롬프트 로딩 유틸리티
- LLM 인스턴스 생성 헬퍼 함수

#### `workflows/tasks.py`
- Pydantic 데이터 모델 정의
  - Bible 모델: `StoryBible`, `CharacterBible`, `StyleBible`
  - 출력 모델: `DirectorOutput`, `ImageEngineerOutput`, `VideoEngineerOutput`
  - `GraphState`: LangGraph 상태 관리 타입

#### `workflows/nodes.py`
- 각 워크플로우 노드의 구현
  - Bible 생성 노드: `generate_story_bible_node`, `generate_character_bible_node`, `generate_style_bible_node`
  - 계획 생성 노드: `generate_director_plan_node`, `generate_image_prompts_node`, `generate_video_prompts_node`
  - 생성 노드: `generate_assets_node`, `generate_frames_node`
  - GCS 업로드: `upload_assets_and_frames_to_gcs_node`
  - 비디오 생성: `generate_videos_node` (Veo 3.1), `generate_videos_with_kie_node` (Kie API)
  - 후처리: `postprocess_node`
  - 라우터: `route_after_veo_generation`

#### `workflows/graph.py`
- LangGraph 워크플로우 정의
  - `create_workflow()`: 전체 워크플로우 (Veo 3.1 + Kie API 폴백)
  - `create_workflow_kie()`: Kie API 전용 워크플로우
  - `create_assets_2_end()`: 에셋 생성부터 끝까지
  - `create_videos_2_end()`: 비디오 생성부터 끝까지

#### `workflows/main.py`
- CLI 진입점
- 세 가지 실행 모드 지원:
  - `full`: 전체 워크플로우 실행
  - `assets`: 에셋 생성부터 실행
  - `videos`: 비디오 생성부터 실행

## 워크플로우 로직

```
사용자 입력
    ↓
Story Bible 생성 → Character Bible 생성 → Style Bible 생성
    ↓
Director Plan 생성 (5 keyframes, 4 shots)
    ↓
Image Prompts 생성 ← Video Prompts 생성
    ↓                    ↓
Asset 생성 → Frames 생성 → GCS 업로드
                              ↓
                    Veo 3.1 비디오 생성
                        ↓       ↓
                   성공         실패
                    ↓           ↓
              Postprocess  Kie API 비디오 생성
                              ↓
                         Postprocess
                              ↓
                        최종 영상 출력
```

### 워크플로우 단계별 설명

1. **Bible 생성** (3단계)
   - Story Bible: 로그라인, 테마, 톤, 세계관 설정
   - Character Bible: 캐릭터 외형, 성격, 감정 아크
   - Style Bible: 촬영 스타일, 색상 팔레트, 조명

2. **계획 수립** (3단계)
   - Director Plan: 5개 키프레임 + 4개 샷 구성
   - Image Prompts: 이미지 생성용 프롬프트 최적화
   - Video Prompts: 비디오 생성용 프롬프트 최적화

3. **생성** (5단계)
   - Asset 생성: 캐릭터 레퍼런스 이미지
   - Frame 생성: 5개의 키프레임 이미지
   - GCS 업로드: Kie API 사용을 위한 GCS 업로드
   - Video 생성: Veo 3.1 또는 Kie API로 4개 비디오 생성
   - Postprocess: FFmpeg로 비디오 병합

### 출력 결과물

- `local_storage/`
  - `story_bible.json`: 스토리 설정
  - `character_bible.json`: 캐릭터 설정
  - `style_bible.json`: 비주얼 스타일 설정
  - `director_output.json`: 연출 계획
  - `image_engineer_output.json`: 이미지 프롬프트
  - `video_engineer_output.json`: 비디오 프롬프트
  - `imgs/assets/`: 캐릭터 레퍼런스 이미지
  - `imgs/assets/frames/`: 5개의 키프레임 이미지
  - `videos/`: 4개의 비디오 클립
  - `final_video.mp4`: 최종 병합 영상 (32초)

## 사전 요구사항

### 필수

1. **Python 3.11+**
2. **uv** (Python 패키지 매니저)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **FFmpeg** (비디오 병합용)
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

4. **Gemini API Key** (필수)
   - Google AI Studio에서 발급: https://aistudio.google.com/apikey
   - 이미지 생성 (`gemini-3-pro-image-preview`) 및 LLM 사용

### 선택 (Kie API 사용 시)

5. **Kie API Key** (선택)
   - Veo 3.1 API 일일 사용량 제한 시 자동 폴백
   - Kie API 계정: https://api.kie.ai/

6. **Google Cloud Storage** (Kie API 사용 시 필수)
   - GCS 버킷 생성
   - 서비스 계정 키 발급 (JSON 형식)
   - 버킷에 공개 읽기 권한 설정

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd drama-making-agent

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어서 API 키 입력
```

### 환경 변수 설정 (.env)

```bash
# 필수
GEMINI_API_KEY=your_gemini_api_key_here

# 선택 (Kie API 사용 시)
KIE_API_KEY=your_kie_api_key_here
GCS_BUCKET_NAME=your_gcs_bucket_name
GCS_SERVICE_ACCOUNT_KEY_PATH=path/to/service_account_key.json
```

## 사용법

### 전체 워크플로우 실행

```bash
uv run python -m src.workflows.main \
  --mode full \
  --input "Ghost Hunter Sarah investigates a haunted house. However, the Evil Ghost Gaspar attacks her and kidnaps her. Sarah must escape from the haunted house."
```

### 부분 실행 (이전 결과 활용)

#### 에셋 생성부터 실행
```bash
uv run python -m src.workflows.main \
  --mode assets \
  --input "placeholder"
```

#### 비디오 생성부터 실행
```bash
uv run python -m src.workflows.main \
  --mode videos \
  --input "placeholder"
```

### 실행 시간

- **전체 워크플로우**: 약 15-25분
  - Bible 생성: 2-3분
  - 계획 수립: 2-3분
  - 에셋/프레임 생성: 5-8분
  - 비디오 생성: 5-10분 (Veo 3.1) / 10-20분 (Kie API)
  - 후처리: 1초 미만

## 주요 기능 설명

### 자동 폴백 시스템

Veo 3.1 API가 일일 사용량 제한에 도달하거나 실패하면 자동으로 Kie API로 전환됩니다:

1. Veo 3.1 API 시도
2. 실패 시 `veo_failed` 플래그 설정
3. 라우터가 Kie API 노드로 전환
4. GCS에 업로드된 프레임 URL 사용
5. Kie API로 비디오 생성

### 일관성 있는 캐릭터 생성

- 첫 번째 캐릭터 생성
- 이후 캐릭터는 이전 캐릭터를 레퍼런스로 사용
- 모든 키프레임에 캐릭터 레퍼런스 전달
- 이전 프레임을 다음 프레임 생성 시 레퍼런스로 활용

### 자연스러운 비디오 연결

- 각 비디오의 마지막 프레임을 추출
- 다음 비디오의 첫 프레임으로 사용
- Frame interpolation으로 부드러운 전환

## 문제 해결

### Veo 3.1 API 할당량 초과
```
Error: Quota exceeded
```
→ Kie API 설정 후 자동 폴백 또는 `create_workflow_kie()` 직접 사용

### FFmpeg 명령어 대기
```
Overwrite? [y/N]
```
→ 이미 수정됨 (`-y` 플래그 추가), 기존 `final_video.mp4` 삭제 후 재실행

## 라이센스

MIT License

## 참고

- 프롬프트 디렉터리: `src/prompts/`