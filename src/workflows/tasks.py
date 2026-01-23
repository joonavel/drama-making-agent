"""
Drama Making Agent - Pydantic Models for Workflow

이 파일은 workflow_1st_mvp_spec.md에 정의된 모든 Pydantic 모델을 포함합니다.
"""

from typing import List, Optional, Literal, Dict, Annotated, TypedDict
from pydantic import BaseModel, Field

HexColor = Annotated[
    str,
    Field(
        description="6자리 HEX 색상. '#' 포함/미포함 허용 (예: '1a1a2e' 또는 '#1a1a2e')."
    ),
]

# ==========================================
# Bible & Guide Generation Models
# ==========================================


class StoryBible(BaseModel):
    """
    Core Elements and information of the Story
    """

    logline: str = Field(description="1~2 sentence summary of the story")
    thematicCore: str = Field(description="underlying message/theme")
    toneDescription: str = Field(
        description="dramatic, comedic, noir, romantic, thriller 등 톤 설명"
    )
    worldRules: Optional[List[str]] = Field(
        default=None,
        description="the specific rules of the world if there are any(max 5)",
    )
    endingState: str = Field(description="the description of the final situation")
    negativePrompt: str = Field(
        description="the elements that should not appear in the frame"
    )


class EmotionalArc(BaseModel):
    """
    Emotional Arc of the Character in the Story
    """

    startEmotion: str = Field(
        description="Initial emotion of the character in the story"
    )
    endEmotion: str = Field(description="Final emotion of the character in the story")


class DialogueVoice(BaseModel):
    """
    Dialogue Voice of the Character
    """

    formality: Literal["formal", "casual", "mixed"] = Field(
        description="the level of formality of the dialogue"
    )
    accentHints: Optional[str] = Field(
        default=None, description="the hints of the accent/dialect/speech features"
    )


class CharacterInfo(BaseModel):
    """
    Character Information
    """

    name: str = Field(description="the name of the character")
    ageAppearance: str = Field(description="the age and appearance of the character")
    fixedVisualElements: List[str] = Field(
        description="the fixed visual elements(hair/clothes/accessories/identifying features). the core of consistency between shots"
    )
    personalityKeywords: List[str] = Field(
        description="the personality keywords(adjectives) of the character"
    )
    emotionalArc: EmotionalArc
    dialogueVoice: DialogueVoice


class CharacterBible(BaseModel):
    """
    Character Bible of the Story
    """

    CHARACTERINFO: List[CharacterInfo] = Field(
        description="the list of character information"
    )


class StyleBible(BaseModel):
    """
    Style Bible of the Story
    """

    cinematography: str = Field(
        description="the preferred cinematography(handheld/static/tracking etc.)"
    )
    colorPalette: List[HexColor] = Field(description="the representative colors(HEX)")
    lighting: str = Field(
        description="the preferred lighting(high-key/low-key/dramatic/natural etc.)"
    )
    filmGrain: Literal["none", "subtle", "prominent"] = Field(
        description="the degree of film grain"
    )
    artisticStyle: Literal[
        "photorealistic", "cinematic", "stylized", "anime-inspired"
    ] = Field(description="the artistic style")
    negativeVisualElements: List[str] = Field(
        description="the visual elements that should not appear in the frame"
    )


# ==========================================
# Shared Models for Planning & Prompts
# ==========================================


class Dialogue(BaseModel):
    """
    Information about dialogue and acting instructions
    """

    speaker: str = Field(description="Name of the character speaking")
    text: str = Field(description="The exact lines of dialogue")
    emotion: str = Field(
        description="Emotional state (e.g., 'Angry', 'Whispering', 'Joyful')"
    )
    acting_instruction: str = Field(
        description="Physical acting instruction for video generation (e.g., 'Slamming the table', 'Eyes widening in shock')"
    )
    order: int = Field(description="Order of the dialogue in the shot")


# ==========================================
# Director Agent Output Models
# ==========================================


class KeyframeDescription(BaseModel):
    """
    Description of a still moment at a specific time (0s, 8s, 16s...)
    """

    index: int = Field(description="Sequence index (0 to 4)")
    timecode: float = Field(
        description="Timeline position in seconds (e.g., 0.0, 8.0, 16.0, 24.0, 32.0)"
    )
    visual_description: str = Field(
        description="Detailed visual description of the scene. Focus on Subject, Background, Lighting. NO technical prompt terms."
    )
    mood: str = Field(description="Atmosphere/Lighting mood of this specific moment")


class ShotPlan(BaseModel):
    """
    Plan of the shot between keyframes
    """

    shot_index: int = Field(description="Shot index (1 to 4)")
    start_keyframe_index: int = Field(description="Index of the starting keyframe")
    end_keyframe_index: int = Field(description="Index of the ending keyframe")

    camera_movement: str = Field(
        description="Camera movement description (e.g., 'Slow dolly in', 'Tracking shot')"
    )
    character_action: str = Field(
        description="Action description between the keyframes"
    )
    dialogues: List[Optional[Dialogue]] = Field(
        default=[],
        description="Dialogue and acting instructions if the character speaks during this shot",
    )


class DirectorOutput(BaseModel):
    """
    Pure planning/direction data
    """

    project_title: str = Field(description="Title of the short drama")
    keyframes: List[KeyframeDescription] = Field(
        description="List of exactly 5 keyframes (0s, 8s, 16s, 24s, 32s)"
    )
    shots: List[ShotPlan] = Field(
        description="List of exactly 4 shots connecting the keyframes"
    )


# ==========================================
# Image Specialist Agent Output Models
# ==========================================


class KeyframePrompt(BaseModel):
    """
    Prompt for Keyframe Image
    """

    keyframe_index: int = Field(description="Index matching the Director's keyframe")
    prompt: str = Field(
        description="Optimized English prompt for Image Generation Model"
    )


class ReferencePrompt(BaseModel):
    """
    Prompt for Reference Image
    """

    name: str = Field(description="Name of the reference(Character, Object, etc.)")
    prompt: str = Field(
        description="Optimized English prompt for Image Generation Model"
    )


class ImageEngineerOutput(BaseModel):
    """
    The set of prompts for Image Generation
    """

    character_reference_prompt: List[ReferencePrompt] = Field(
        description="Prompt to generate a Reference image for consistency"
    )
    common_negative_prompt: str = Field(
        description="A shared negative prompt to prevent bad anatomy, text, low quality, etc."
    )
    keyframe_prompts: List[KeyframePrompt] = Field(
        description="List of 5 optimized image prompts corresponding to the Director's keyframes"
    )


# ==========================================
# Video Specialist Agent Output Models
# ==========================================


class VideoPrompt(BaseModel):
    """
    Prompt for Video Generation
    """

    shot_index: int = Field(description="Index matching the Director's shot")
    prompt: str = Field(
        description="Optimized English prompt for Video Generation Model (Veo 3.1). Includes visual descriptions, camera movement, and dialogue acting instructions."
    )


class VideoEngineerOutput(BaseModel):
    video_prompts: List[VideoPrompt] = Field(
        description="List of 4 optimized video prompts"
    )


# ==========================================
# LangGraph State Model
# ==========================================


class GraphState(TypedDict):
    """
    LangGraph State for the entire workflow
    """

    # User Input
    user_input: Optional[str]  # Initial user input for story generation
    
    # Config
    max_retries: int

    # Bible & Guide
    story_bible: Optional[StoryBible]  # 스토리 설정
    character_bible: Optional[CharacterBible]  # 캐릭터 설정
    style_bible: Optional[StyleBible]  # 촬영 스타일 설정

    # Planning & Prompts
    director_output: Optional[DirectorOutput]  # 바이블에 기반한 촬영 요소 및 계획
    image_engineer_output: Optional[
        ImageEngineerOutput
    ]  # 촬영 요소 및 계획에 기반한 이미지 생성 프롬프트
    video_engineer_output: Optional[
        VideoEngineerOutput
    ]  # 촬영 요소 및 계획에 기반한 비디오 생성 프롬프트

    # Asset, Frame, Video paths
    assets: Optional[Dict[str, str]]  # 에셋 파일 경로
    frames: Optional[List[str]]  # 프레임 파일 경로
    videos: Optional[List[str]]  # 비디오 파일 경로
    final_video: Optional[str]  # 최종 비디오 파일 경로

    # Error tracking
    errors: Optional[List[str]]  # 오류 메시지
