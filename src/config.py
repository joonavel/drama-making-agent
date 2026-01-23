"""
전역 설정 및 유틸리티
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 환경 변수 로드
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

# 경로 상수 정의
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "local_storage"
ASSETS_DIR = OUTPUT_DIR / "imgs" / "assets"
FRAMES_DIR = ASSETS_DIR / "frames"  # 프레임 저장 기본 경로
VIDEOS_DIR = OUTPUT_DIR / "videos"  # 비디오 저장 기본 경로

# 디렉토리 생성
for directory in [PROMPTS_DIR, OUTPUT_DIR, ASSETS_DIR, FRAMES_DIR, VIDEOS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 포매터
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def load_prompt(path: str | Path) -> str:
    """
    프롬프트 파일을 로드합니다.

    Args:
        path: 프롬프트 파일 경로 (상대 경로 또는 절대 경로)

    Returns:
        str: 프롬프트 내용

    Example:
        >>> prompt = load_prompt("prompts/system/story_agent.prompt")
    """
    if isinstance(path, str):
        file_path = Path(path)
    elif isinstance(path, Path):
        file_path = path
    else:
        raise ValueError("path must be a string or a Path object")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_llm(
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:
    """
    LLM 인스턴스를 생성합니다.

    Args:
        model: 사용할 모델 이름
        temperature: 생성 온도 (0.0~2.0, Gemini 3.0+ 기본값 1.0)
        max_tokens: 최대 토큰 수
        timeout: 타임아웃 (초)
        max_retries: 최대 재시도 횟수

    Returns:
        ChatGoogleGenerativeAI: LLM 인스턴스

    Example:
        >>> llm = get_llm(model="gemini-3-flash-preview", temperature=1.0)
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


# 기본 로거
logger = setup_logger("workflows")


# 환경 변수 검증
def validate_env_vars():
    """필수 환경 변수 검증"""
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.warning("Please check your .env file")
    else:
        logger.info("All required environment variables are present")


# 모듈 로드 시 환경 변수 검증
validate_env_vars()
