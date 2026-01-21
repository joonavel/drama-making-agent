"""
Workflows 모듈 - LangGraph 기반 워크플로우 구현
"""

# ==========================================
# Pydantic Models from tasks.py
# ==========================================
from .tasks import (
    # Bible & Guide Models
    StoryBible,
    EmotionalArc,
    DialogueVoice,
    CharacterInfo,
    CharacterBible,
    StyleBible,
    
    # Shared Models
    Dialogue,
    
    # Director Models
    KeyframeDescription,
    ShotPlan,
    DirectorOutput,
    
    # Image Specialist Models
    KeyframePrompt,
    ReferencePrompt,
    ImageEngineerOutput,
    
    # Video Specialist Models
    VideoPrompt,
    VideoEngineerOutput,
    
    # State Model
    GraphState,
    
    # Type Annotations
    HexColor,
)

# ==========================================
# Config from parent module
# ==========================================
from src.config import (
    load_prompt,
    get_llm,
    setup_logger,
    PROJECT_ROOT,
    PROMPTS_DIR,
    OUTPUT_DIR,
    ASSETS_DIR,
    FRAMES_DIR,
    VIDEOS_DIR,
    logger,
)

# ==========================================
# Future imports (uncomment when implemented)
# ==========================================
# from .nodes import (
#     bible_generation_node,
#     planning_node,
#     asset_generation_node,
#     frame_generation_node,
#     video_generation_node,
#     post_processing_node,
# )
#
# from .graph import create_workflow

__all__ = [
    # Models
    "StoryBible",
    "EmotionalArc",
    "DialogueVoice",
    "CharacterInfo",
    "CharacterBible",
    "StyleBible",
    "Dialogue",
    "KeyframeDescription",
    "ShotPlan",
    "DirectorOutput",
    "KeyframePrompt",
    "ReferencePrompt",
    "ImageEngineerOutput",
    "VideoPrompt",
    "VideoEngineerOutput",
    "GraphState",
    "HexColor",
    
    # Config utilities
    "load_prompt",
    "get_llm",
    "setup_logger",
    "PROJECT_ROOT",
    "PROMPTS_DIR",
    "OUTPUT_DIR",
    "ASSETS_DIR",
    "FRAMES_DIR",
    "VIDEOS_DIR",
    "logger",
    
    # Future exports
    # "bible_generation_node",
    # "planning_node",
    # "asset_generation_node",
    # "frame_generation_node",
    # "video_generation_node",
    # "post_processing_node",
    # "create_workflow",
]
