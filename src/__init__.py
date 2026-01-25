"""
Drama Making Agent - 드라마 쇼츠 영상 자동 생성 시스템
"""

__version__ = "0.1.0"

# ==========================================
# Main exports
# ==========================================
from .config import (
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

__all__ = [
    "__version__",
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
]
