# Drama Creation Agent - Development Rules & Guidelines

## 1. Project Overview
**Drama Creation Agent**는 사용자의 테마 입력을 바탕으로 30초 분량의 쇼츠 드라마를 생성하는 Agentic Application입니다. Gemini Ecosystem(Gemini 3.0 Flash, Imagen 3, Veo 3.1)을 활용하여 일관성 있는 스토리와 영상을 생성합니다.

---

## 2. Tech Stack & Environment

### 2.1 Core Technologies
- **Language:** Python 3.10+
- **Dependency Management:** `uv` (An extremely fast Python package installer and resolver)
- **Containerization:** Docker
- **Cloud Services:** Google Cloud Platform (Vertex AI, GCS)
- **Key APIs:**
  - LLM: Gemini 3.0 Flash (Text Generation)
  - Image: Imagen 3.0 or Nano Banana (Frame Generation)
  - Video: Veo 3.1 (Video Generation)

### 2.2 Development Tools
- **Linter/Formatter:** `ruff` (Recommended for speed and compatibility with `uv`)
- **Type Checking:** `mypy`

---

## 3. Development Workflow

### 3.1 Dependency Management (`uv`)
본 프로젝트는 `uv`를 사용하여 의존성을 관리합니다.

- **프로젝트 초기화:** `uv init`
- **패키지 추가:** `uv add <package_name>`
- **개발 패키지 추가:** `uv add --dev <package_name>`
- **동기화 (install):** `uv sync`
- **스크립트 실행:** `uv run python main.py`

### 3.2 Docker Containerization
- **Base Image:** `python:3.10-slim` 또는 `python:3.10-alpine` (필요 라이브러리 호환성 확인 후 선택)
- **Multi-stage Build:** 빌드 아티팩트와 런타임 환경을 분리하여 이미지 크기 최적화.
- **Environment Variables:** 민감 정보(API Key 등)는 `.env` 파일로 관리하며 Docker 컨테이너 실행 시 주입. (이미지에 포함 금지)

---

## 4. Coding Conventions

### 4.1 Naming Conventions
- **Files/Modules:** `snake_case.py`
- **Variables/Functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_CASE`
- **Private Members:** `_snake_case` (prefix with underscore)

### 4.2 Type Hinting & Docstrings
- **Type Hints:** 모든 함수 시그니처에 Type Hint 사용을 권장합니다.
  def generate_shot_plan(story_bible: dict) -> List[Shot]:
      ...
  - **Docstrings:** Google Style Docstring을 따릅니다.
  def process_video(video_path: str) -> bool:
      """Processes the generated video for final export.

      Args:
          video_path (str): The local path to the video file.

      Returns:
          bool: True if processing succeeded, False otherwise.
      """
  ### 4.3 Error Handling
- **Exceptions:** 명시적인 예외 처리를 수행합니다. (`try-except Exception` 지양)
- **API Failures:** 외부 API 호출(Gemini, Imagen, Veo) 실패 시, 적절한 재시도(Retry) 로직과 백오프(Backoff) 전략을 구현합니다.
- **Safety Fallback:** 콘텐츠 정책 위반(Safety Filter) 발생 시, 사용자에게 명확한 피드백을 주거나 프롬프트를 수정하여 재시도하는 로직을 포함합니다.

---

## 5. Architecture & Pipeline Rules

### 5.1 Pipeline Stages (Strict Order)
파이프라인은 반드시 다음 순서를 따라야 하며, 이전 단계의 산출물(Artifact)이 다음 단계의 입력(Input)이 되어야 합니다.

1.  **User Input Analysis** -> Output: Theme, Metadata
2.  **Bible Generation** (Gemini) -> Output: `story_bible`, `character_bible`, `style_bible` (JSON)
3.  **Shot Planning** (Gemini) -> Output: 4-Shot Plan (8s/8s/8s/6s)
4.  **Frame Generation** (Imagen) -> Output: First/Last Frames (PNG)
5.  **Video Generation** (Veo) -> Output: 4 Video Clips (MP4)
6.  **Post-Processing** -> Output: Final Merged Video (MP4 + Audio)

### 5.2 Consistency Maintenance
- **Immutable Bibles:** 생성된 Bible(Story, Character, Style)은 Shot Planning 및 Frame Generation 단계에서 **변경되지 않는 불변의 기준(Source of Truth)**으로 사용되어야 합니다.
- **Prompt Injection:** 모든 프롬프트 생성 시 Bible의 내용을 System Prompt 또는 Context로 주입해야 합니다.

### 5.3 File & Data Management (GCS)
- **Path Structure:** `projects/{user_id}/{timestamp}/{stage}/...` 구조를 엄격히 준수합니다.
- **Intermediate Files:** 각 단계의 중간 산출물(JSON, PNG, MP4)은 반드시 GCS에 저장하여 추적 가능성(Audit Trail)을 확보합니다.

### 5.4 Async Operations
- **Long-Running Operations (LRO):** Veo 비디오 생성과 같이 시간이 오래 걸리는 작업은 비동기(Async) 처리 및 폴링(Polling) 방식으로 구현합니다. 메인 스레드를 차단(Block)하지 않도록 주의합니다.

---

## 6. Git & Version Control
- **Commit Messages:** [Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따릅니다.
  - `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`
- **Branching:** `main` (production), `develop` (integration), `feature/*` (development)

## 7. Testing
- **Unit Tests:** 핵심 로직(프롬프트 빌더, 파서 등)에 대한 단위 테스트 작성 (`pytest` 권장).
- **Integration Tests:** 실제 API 호출을 모킹(Mocking)하거나, 비용이 발생하지 않는 범위 내에서 통합 테스트 수행.