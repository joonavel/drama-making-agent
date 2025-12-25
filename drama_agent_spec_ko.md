# Drama Creation Agent - Project Specification

---

## 핵심 요약

**프로젝트 명:** Drama Creation Agent

**최종 목표:** 사용자가 테마 및 주제를 입력하면 짧은 드라마(5~10분), 유튜브 쇼츠 등을 생성하는 Agentic 애플리케이션을 만드는 것

**1차 목표:** 30초 정도의 쇼츠 생성(오디오 포함)

**타겟:** 쇼츠등의 유튜브 컨텐츠 생성을 통해 수익 창출을 노리는 일반인 > 사용이 쉽고 간편해야한다

**Technology Stack:** Gemini API(Gemini3, Imagen, Nano Banana, Veo3.1), Text-to-Speech model(오디오 생성이 어려운 경우)

---

## 1. Project Architecture Overview

### 1.1 Multi-Stage Pipeline

The Drama Creation Agent의 1차 프로토 타입은 총 **5-단계의 agentic workflow로 진행**:

```
User Input (Theme/Story)
    ↓
Bible & Guide 생성 (시나리오, 인물 설정서, 촬영 스타일 등)
    ↓
계획 수립 (5개의 프레임에 대한 계획, 30초)
    ↓
프레임 생성 (Imagen 사용, 개별 shot의 첫/마지막 프레임)
    ↓
영상 생성 (Veo: 8s/8s/8s/6s 클립 생성)
    ↓
후처리 (Assembly, Mixing, Export)
    ↓
최종 출력 (MP4)
```

위는 이상적인 workflow로 api로 해결이 불가할 시 사용자의 개입을 염두에 두고 workflow를 여러개로 나눌 수 있다.
- ex) 프레임 생성에 문제가 있을 경우
  - page1 -> 계획 출력, page2 -> 프레임으로 영상 생성

### 1.2 Key Design Principle: 일관성 유지

영상 생성시 Agent로 하여금 무작위 맥락으로 진행하지 않도록, 전 단계에서 시나리오, 설정서등의 Bible과 촬영 규칙 등의 Guide를 먼저 생성하게 하여 고정된 context로 활용하게 한다.


또한, Imagend의 이미지 참조 기능과 Veo3.1의 Frame2Frame 기능을 활용하여 영상 속 인물들의 일관성을 최대한 유지하도록한다.

이를 통해 아래 3가지 요소의 충족을 기대한다:
- **시각적 일관성** 4개의 클립(8s,8s,8s,6s)에서 설정상 동일한 인물은 일관성이 있어야한다.
- **서사석 일관성** 동일한 인물의 인격은 일관성이 있어야한다.
- **스타일 일관성** 4개의 클립은 특별한 이유가 없다면 장르, 분위기 등의 일관성이 있어야한다.

참고 링크
- 참조 이미지 사용: https://ai.google.dev/gemini-api/docs/image-generation?hl=ko#use-14-images
- Frame2Frame: https://ai.google.dev/gemini-api/docs/video?hl=ko&example=dialogue#using-first-and-last-video-frames

---

## 2. Input & Output Specification

### 2.1 User Input

**포맷:** 1~2 문장 정도의 입력, 장문의 입력도 가능

**Examples:**
- "비 오는 서울 밤, 검은 우산을 든 주인공이 골목을 달려 카페로 들어간다. 네온 누아르."
- "A mysterious stranger arrives in a small town, and a local discovers a hidden secret."
- "포장마차에서 만난 두 사람, 술 한잔으로 시작된 인연의 30초."
- "웃긴 영상 만들어줘"

**가이드라인 :**
- 캐릭터의 외형, 시나리오 등이 사용자 입력에 명시되어 있다면, 이를 반영해야 한다.

### 2.2 System Output

**Primary Deliverable:** MP4 video file (30 seconds)
- **해상도:** 1920×1080 (16:9 landscape)
- **프레임:** 30fps (standard for YouTube Shorts)
- **오디오:** Stereo, 48kHz
- **코덱:** H.264 (VP9 alternative for better compression)

---

## 3. 구현 전략

### 3.1 User Input Reception

**Input Type:** Web form / API endpoint
- 텍스트 입력
- Optional: 콘텐츠 체크 박스 (violence, sexual content, etc.)

**Output:** Normalized prompt string, user metadata (타임스탬프, 분석을 위한 유저 ID)

---

### 3.2 Stage 2: Bible & Guide 생성 (LLM - Gemini API)

**목적:** 일관성을 유지하기 위한 bible 및 guide 생성

**LLM Model:** `gemini-3.0-flash`

**Prompt Template Structure:**

```
You are a drama script and visual guidelines generator.

Given a user's story theme, generate a structured JSON with these sections:

1. STORY_BIBLE
   - logline (1-sentence summary)
   - tone_description (dramatic, comedic, noir, romantic, thriller, etc.)
   - world_rules (3-5 constraints on what's possible in this world)
   - ending_state (what the final frame will show - the "culmination")
   - negative_prompt (what MUST NOT appear: violence level, sexual content, etc.)

2. CHARACTER_BIBLE
   - name
   - age_appearance
   - fixed_visual_elements (hair style, clothing, accessories, distinguishing features)
   - personality_keywords (5 adjectives)
   - emotional_arc (starting emotion → ending emotion)
   - dialogue_voice (formal/casual/accent hints)

3. STYLE_BIBLE
   - cinematography (handheld/static/tracking shot preference)
   - color_palette (primary 3 colors in hex)
   - lighting (high-key/low-key/dramatic/natural)
   - film_grain (none/subtle/prominent)
   - artistic_style (photorealistic/cinematic/stylized/anime-inspired)
   - negative_visual_elements (what never appears in ANY frame)

IMPORTANT: All three sections must be consistent with each other and the user's input theme.
Return ONLY valid JSON, no markdown, no explanation.

User Input: "{user_input}"
```

**Expected Output:**
```json
{
  "story_bible": {
    "logline": "...",
    "tone_description": "...",
    "world_rules": [...],
    "ending_state": "...",
    "negative_prompt": "..."
  },
  "character_bible": {
    "name": "...",
    "age_appearance": "...",
    "fixed_visual_elements": "...",
    "personality_keywords": [...],
    "emotional_arc": "...",
    "dialogue_voice": "..."
  },
  "style_bible": {
    "cinematography": "...",
    "color_palette": [...],
    "lighting": "...",
    "film_grain": "...",
    "artistic_style": "...",
    "negative_visual_elements": "..."
  }
}
```

**Error Handling:**
- JSON 파싱 오류: 프롬프트 교체 + schema validation
- 정책 위반의 경우(성적이거나 폭력적인 컨텐츠): 유저 측에 정책 위반 에러 전달
- 애매모호한 입력: 멀티턴

---

### 3.3 Stage 3: Shot Planning (LLM - Gemini API)

**Purpose:** 30초 정도의 서사를 5개의 장면과 4개의 시나리오로 분해하여 계획 수립

**Constraint:** 총 길이 = 8s + 8s + 8s + 6s = 30 seconds (Veo 3.1의 생성 길이 제한 limits: 4/6/8 s)

**Prompt Template:**

```
You are a cinematography planner for short-form video.

Given a story bible and character bible, create a detailed shot plan for a 30-second video.

The video will have EXACTLY 4 shots with these durations:
- Shot 1: 8 seconds
- Shot 2: 8 seconds
- Shot 3: 8 seconds
- Shot 4: 6 seconds

For each shot, provide:
1. shot_objective (what narrative/emotional purpose does this shot serve?)
2. camera_movement (static/pan/zoom/tracking, direction)
3. character_action (what the character does during this shot)
4. first_frame_description (detailed visual description for image generation - 50-80 words)
5. last_frame_description (visual description of final frame - 50-80 words)
6. veo_prompt (cinematography instruction for video generation - 40-60 words)
7. dialogue (if any: speaker, text, emotional tone, delivery speed)
8. audio_hint (music mood, sound effects if any)

CRITICAL RULES:
- Character appearance MUST match the character bible in every shot
- Camera angles should vary (wide shot, close-up, medium shot, etc.)
- Dialogue timing MUST fit within the shot duration (assume average speech: ~150 words/minute)
- Each shot's last_frame should naturally transition to next shot's first_frame
- Respect the story arc: beginning (shot 1) → rising action (shots 2-3) → climax/resolution (shot 4)

Story Bible: {story_bible_json}
Character Bible: {character_bible_json}
Style Bible: {style_bible_json}

Return ONLY valid JSON, no markdown.
```

**Expected Output:**
```json
{
  "shot_plan": [
    {
      "shot_number": 1,
      "duration_seconds": 8,
      "shot_objective": "...",
      "camera_movement": "...",
      "character_action": "...",
      "first_frame_description": "...",
      "last_frame_description": "...",
      "veo_prompt": "...",
      "dialogue": {
        "speaker": "Character Name",
        "text": "...",
        "emotional_tone": "...",
        "speed": "natural"
      },
      "audio_hint": "..."
    },
    ...
  ]
}
```

---

### 3.4 Stage 4: Frame Generation (Imagen)

**Purpose:** 각 클립의 첫/마지막 프레임을 생성

**Model:** `imagen-3.0-generate-001` (text-to-image)

**Per-Shot Process:**

**5개의 이미지 생성(Imagen API calls)**

or

4개의 클립에 대응하는 각 **2 개의 이미지 생성(Imagen API calls)**:

#### 3.4.1 First Frame Generation

**Prompt Construction:**
```
{style_bible.cinematography} shot of {character_bible.fixed_visual_elements}, {character_bible.age_appearance}.
{shot[i].first_frame_description}
Style: {style_bible.artistic_style}. Lighting: {style_bible.lighting}. Color palette: {style_bible.color_palette}.
Negative: {style_bible.negative_visual_elements}, {story_bible.negative_prompt}.
```

**API Parameters:**
- `aspectRatio`: `"16:9"`
- `personGeneration`: `"allow"` (or `"dont_allow"`)
- `addWatermark`: `true` (for safe publication; set `false` if seed needed)
- `seed`: (optional, for reproducibility)
- `storageUri`: `gs://{project_bucket}/first_frames/shot_{i}.png`

**Output:** `gs://{project_bucket}/first_frames/shot_{i}.png`

#### 3.4.2 Last Frame Generation

**Prompt Construction:**
```
{style_bible.cinematography} shot of {character_bible.fixed_visual_elements}, {character_bible.age_appearance}.
{shot[i].last_frame_description}
Style: {style_bible.artistic_style}. Lighting: {style_bible.lighting}. Color palette: {style_bible.color_palette}.
Negative: {style_bible.negative_visual_elements}, {story_bible.negative_prompt}.
```

**API Parameters:** (천번쨰 프레임과 동일, output path만 다르게 `last_frames/shot_{i}.png`)

**Optimization Note:** Imagen batching is not officially supported, so calls must be sequential. Total time: ~10-15 seconds for 8 images (with parallel infrastructure on server side).

**Error Handling:**
- safety 필터에 거절당할 경우 (e.g., person generation policy): `personGeneration: "dont_allow"`로 재시도 or 생성 프롬프트에 변주

---

### 3.6 비디오 생성 (Veo 3.1 API)

**Purpose:** 첫프레임, 마지막 프레임, and cinematography 프롬프트를 이용해 클립 생성

**Model:** `veo-3.1-generate-001` or `veo-3.1-fast-generate-001`

**Per-Shot Video Generation:**

#### 3.6.1 API Request Structure

```json
{
  "instances": [
    {
      "prompt": "veo_prompt from shot plan",
      "image": "gs://.../first_frames/shot_1.png",
      "lastFrame": "gs://.../last_frames/shot_1.png"
    }
  ],
  "parameters": {
    "durationSeconds": 8,
    "aspectRatio": "16:9",
    "generateAudio": false,
    "resolution": "1080p",
    "storageUri": "gs://{project_bucket}/videos/"
  }
}
```

**Key Parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `durationSeconds` | 8 (shots 1-3), 6 (shot 4) | Veo 3 limit is 4/6/8 seconds |
| `aspectRatio` | "16:9" | Shorts landscape format |
| `generateAudio` | `true` | 반드시 true 여야 한다고 명시됨 |
| `resolution` | "1080p" | YouTube standard |
| `model` | "veo-3.1-generate-001" | Latest stable; "-fast-" variant for speed |

**API Call Details:**

Veo uses **long-running operations (LRO)**:

```python
# Pseudocode
operation = veo_api.predict_long_running(request)
operation_name = operation.name  # e.g., "projects/.../operations/12345"

# Poll until done
while True:
    operation = veo_api.get_operation(operation_name)
    if operation.done:
        break
    time.sleep(5)  # Check every 5 seconds

# Extract output
video_uri = operation.result.videos[0].gcsUri  # gs://.../sample_0.mp4
```

**Error Handling:**
- `INVALID_ARGUMENT`: 프롬프트 길이, 프레임 사이즈, 파라미터 범위 체크
- `FAILED_PRECONDITION`: 정책 위반 (person generation, violence, etc.)
  - 대응: 프롬프트 교체, 파라미터 변화 등

**Timeout Strategy:**
- Veo generation can take 30-120 seconds per 8-second video
- Total expected time for 4 shots: 3-8 minutes
- 사용자가 현재 무슨단계에 있고 예상 시간은 얼마인지 확인 가능하도록 구현

---

## 4. 기술 스택 및 서비스

### 4.1 구글 서비스

| Service | Component | Purpose |
|---------|-----------|---------|
| **API Service** | Gemini API | Bible & shot planning (LLM) |
| **API Service** | Imagen 3 | Frame image generation |
| **API Service** | Veo 3.1 | Video generation (8/6 sec clips) |
| **GCS** (Storage) | Bucket | Intermediate file storage (frames, videos, audio) |


### 4.2 로컬/엣지 서비스

| Service | Purpose |
|---------|---------|
| **Python 3.10+** | Orchestration script (prompt building, API calls, file management) |
| **FastAPI / Flask** | User-facing API / web interface (if REST endpoint needed) |

### 4.3 써드파티 (선택사항)

| Service | Use Case |
|---------|----------|
| **Runway Gen-3** | Alternative video generation (lower cost but slower) |
| **ElevenLabs** | Premium TTS (natural voices, emotion control) |
| **Airtable / Supabase** | User project history, credit management |

---

## 5. 데이터 흐름 및 저장

### 5.1 GCS 버킷 구조

```
gs://drama-agent-prod/
├── projects/
│   └── {user_id}/{timestamp}/
│       ├── 00_input/
│       │   └── user_prompt.txt
│       ├── 01_bible/
│       │   ├── story_bible.json
│       │   ├── character_bible.json
│       │   └── style_bible.json
│       ├── 02_shotplan/
│       │   └── shot_plan.json
│       ├── 03_frames/
│       │   ├── first_frames/
│       │   │   └── shot_{1,2,3,4}.png
│       │   └── last_frames/
│       │       └── shot_{1,2,3,4}.png
│       ├── 04_audio/
│       │   ├── dialogue_shot_{1,2,3,4}.wav
│       │   └── subtitles.srt
│       ├── 05_videos/
│       │   ├── shot_1.mp4
│       │   ├── shot_2.mp4
│       │   ├── shot_3.mp4
│       │   └── shot_4.mp4
│       └── 06_final/
│           ├── drama_final_30sec.mp4
│           ├── drama_final_30sec.srt
│           └── metadata.json
```

### 5.2 메타데이터 JSON (최종 배포물)

```json
{
  "project_id": "user_id/timestamp",
  "user_id": "user_id",
  "created_at": "2025-12-20T11:03:00Z",
  "original_prompt": "...",
  "duration_seconds": 30,
  "aspect_ratio": "16:9",
  "video_uri": "gs://drama-agent-prod/projects/.../drama_final_30sec.mp4",
  "subtitle_uri": "gs://drama-agent-prod/projects/.../drama_final_30sec.srt",
  "bibles": {
    "story": {...},
    "character": {...},
    "style": {...}
  },
  "shot_plan": [...],
  "models_used": {
    "llm": "gemini-2.0-flash",
    "image": "imagen-3.0-generate-001",
    "video": "veo-3.1-generate-001",
    "tts": "cloud-text-to-speech-chirp3"
  },
  "cost_estimate": {
    "gemini_tokens": 12000,
    "imagen_images": 8,
    "veo_seconds": 30,
    "tts_characters": 450,
    "total_usd": 0.35
  },
  "generation_time_seconds": 320
}
```

---

## 6. 안전성 및 정책 준수

### 6.1 콘텐츠 정책

1. **Person Generation:** 
   - Adult characters (18+): Allowed with `personGeneration: "allow"`
   - Minors: Restricted; requires additional consent verification
   
2. **Violence/Harm:**
   - Mild dramatization: Allowed
   - Graphic violence: Rejected by Veo/Imagen safety filters
   - Mitigation: Prompt carefully, use "fade to black" for violent transitions

3. **Sexual Content:**
   - Suggestive (fade-to-black): Allowed in rated content
   - Explicit: Not allowed
   
4. **Copyright/Likeness:**
   - Original characters/stories: Safe
   - Resemblance to real persons: May trigger rejection

### 6.2 프롬프트 보호

**Implemented Checks:**
- Pre-prompt validation: Reject or Modify the input if it contains known bad keywords
- Post-generation validation: Flag outputs for manual review if policies unclear
- User agreements: Terms of Service must cover generated content ownership

### 6.3 Audit Trail

Every generation logs:
- User ID + timestamp
- Original prompt + all intermediate artifacts
- Models used + parameter values
- Safety filter verdicts (passed/rejected/flagged)
- Final output reference

---

## 7. MVP 개발 로드맵

### Phase 1.0: Core 30-Second Pipeline (Target: Q4 2025 - Q1 2026)

**Deliverables:**
- [ ] Gemini prompt templates (Bible + Shot Planning)
- [ ] Imagen prompt engineering + frame generation pipeline
- [ ] Veo integration + LRO polling system
- [ ] Backend API (FastAPI) + async task queue
- [ ] Simple web UI (form input → video generation)
- [ ] Error handling + retry logic (graceful degradation)

**Testing:**
- 50+ test themes (diverse genres, tones, lengths)
- Quality assessment (visual consistency, narrative coherence, technical quality)
- Cost validation (actual spend vs. estimates)

---

### Phase 2.0: Features (Q1 2026+)

- **Variable Lengths:** 15s, 45s, 60s, 3min dramas
- **Vertical Format:** 9:16 for TikTok/Reels
- **Character Consistency:** Multi-shot character reuse (referenceImages in Veo)
- **Voice Selection:** Multiple character voices (multi-speaker dialogue)
- **Music Integration:** Licensed music library + automated sync
- **Subtitle Styles:** Custom fonts, colors, animations
- **User Editing UI:** Timeline editor for shot reordering, voice selection
- **Monetization:** Credit-based system, premium features (longer videos, faster generation)

---

### Phase 3.0: Advanced Features (2026+)

- **Lip-Sync:** Character mouth movement matching dialogue
- **Shot Stitching Intelligence:** Seamless transitions via object continuity
- **A/B Generation:** Multiple variations of same prompt (user voting)
- **Analytics:** User dashboard (videos created, views, revenue)
- **Community:** Gallery of generated videos, prompt sharing, voting

---

## 8. Key Decisions & Rationale

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| **Bible-first approach** | Ensures consistency before any visual generation | Prompt-per-frame (lower consistency) |
| **4 shots (8/8/8/6s)** | Fits Veo duration limit, optimal narrative pacing | Fewer longer shots, more shorter shots |
| **Imagen + Veo (not AI video synthesis end-to-end)** | Precise frame anchoring ensures coherent motion | Direct text→video (less controllable) |
| **Cloud TTS + SSML (not Veo's generateAudio)** | Better dialogue sync + word-level subtitles | Veo's auto-audio (lower accuracy) |
| **FFmpeg post-processing** | Precise control, cost-effective | Cloud Video AI (expensive, less flexible) |
| **GCS for intermediates** | Cheap storage, fast API integration, easy audit trail | Local file system (scaling issues) |
| **Async LRO polling** | Prevents blocking, scales horizontally | Synchronous Veo calls (slow) |

---

## 9. Appendix: Example End-to-End Flow

### User Input
```
"A tired office worker stumbles out into a rainy Seoul night, 
seeking solace in a quiet noodle shop. Neo-noir, melancholic, hope."
```

### Generated Story Bible (JSON excerpt)
```json
{
  "story_bible": {
    "logline": "Exhausted and disillusioned, a young professional finds a moment of peace 
                in an unexpected refuge, hinting at redemption.",
    "tone_description": "neo-noir, introspective, melancholic with subtle hope",
    "ending_state": "Close-up of character's face illuminated by warm noodle shop light, 
                    eyes closed, first genuine smile of the evening",
    "negative_prompt": "no violence, no explicit content, no unrealistic CGI"
  },
  "character_bible": {
    "fixed_visual_elements": "Dark suit, white dress shirt, loosened tie, tired eyes, 
                            slicked-back black hair, minimal jewelry",
    "emotional_arc": "Despair (beginning) → Curiosity → Comfort → Hope (end)"
  },
  "style_bible": {
    "cinematography": "slow tracking shots, medium close-ups, minimalist framing",
    "color_palette": ["#1a1a2e", "#16213e", "#ff6b35"],
    "lighting": "low-key, neon accent lights, warm tungsten",
    "artistic_style": "cinematic, photorealistic, film noir aesthetic"
  }
}
```

### Generated Shot Plan (excerpt - Shot 1)
```json
{
  "shot_1": {
    "shot_objective": "Establish character in despair, late night Seoul setting",
    "camera_movement": "slow tracking shot, left to right",
    "character_action": "walks slowly down rain-wet street, shoulders hunched, 
                       hands in pockets, no clear destination",
    "first_frame_description": "Wide shot of dark Seoul alley at night. 
      Neon signs reflected in wet pavement. Character in center foreground, 
      silhouetted against soft streetlight, suit drenched, tie loose. 
      Rain falling, puddles rippling. Color: deep blue and red neon. 
      Mood: lonely, tired, adrift.",
    "last_frame_description": "Medium close-up as character reaches alley corner. 
      Face partially lit by warm noodle shop glow (off-screen). Eyes beginning 
      to look up. Rain dripping from hair. First hint of relief in posture. 
      Same neon/wet aesthetic but warmer glow entering frame.",
    "veo_prompt": "Slow tracking shot of weary man walking through rainy Seoul alley. 
      Heavy downpour, neon signs reflecting in puddles. Neo-noir cinematography, 
      cool blue and red color grading. End with character turning toward warm 
      light source (off-screen).",
    "dialogue": {
      "speaker": "Character (internal monologue/voice-over)",
      "text": "Another day. Another night. Where do I go now?",
      "emotional_tone": "exhausted, questioning",
      "speed": "slow, deliberate"
    }
  }
}
```

### Generated Frames
(Imagen output: PNG images showing shot 1's first and last frames)

### Generated Video Clip
(Veo output: 8-second MP4 with motion between first→last frame)


### Final Output
```
drama_final_30sec.mp4 (30 seconds)
├─ Clip 1 (8s): Alley walk
├─ Clip 2 (8s): Entering shop entrance
├─ Clip 3 (8s): Sitting at counter, ordering
└─ Clip 4 (6s): First sip, comfort, hope
```

---

## 10. Success Metrics (Phase 1)

| Metric | Target |
|--------|--------|
| **Visual Consistency** | 80%+ of generated videos show consistent character appearance across 4 shots |
| **Narrative Coherence** | 75%+ of users report the generated video matches their original prompt intent |
| **Technical Quality** | 95%+ of videos export without errors; audio/video sync <100ms drift |
| **Generation Time** | <5 minutes from input to final download |
| **Cost Efficiency** | <$2.00 per 30-second video (including all API calls) |
| **User Satisfaction** | 4.0+/5.0 rating on generated video quality (subjective survey) |

---