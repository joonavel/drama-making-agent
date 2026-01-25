"""
LangGraph Nodes for Drama Making Workflow
"""

import json
import time
import os
import subprocess
import cv2
import requests
import datetime
from pathlib import Path
from PIL import Image
from typing import Any, Optional, Sequence
from google import genai
from google.genai import types
from google.cloud import storage
from src.config import (
    load_prompt,
    get_llm,
    OUTPUT_DIR,
    logger,
    PROMPTS_DIR,
    ASSETS_DIR,
    FRAMES_DIR,
    VIDEOS_DIR,
    GCS_BUCKET_NAME,
    GCS_SERVICE_ACCOUNT_KEY_PATH,
    KIE_API_KEY,
    KIE_API_BASE_URL,
)
from src.workflows.tasks import (
    StoryBible,
    CharacterBible,
    StyleBible,
    DirectorOutput,
    ImageEngineerOutput,
    VideoEngineerOutput,
    GraphState,
)


def generate_story_bible_node(state: GraphState) -> dict:
    """
    Story Bible 생성 노드

    Args:
        state: GraphState dict containing user_input

    Returns:
        dict: {"story_bible": StoryBible}
    """
    try:
        logger.info("Starting Story Bible generation...")

        # 1. state에서 user_input 추출
        user_input = state.get("user_input", "")
        max_retries = state.get("max_retries", 3)
        if not user_input:
            raise ValueError("user_input is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "story_agent.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "story_agent.prompt")
        user_prompt = user_prompt_template.format(user_input=user_input)

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(StoryBible)

        messages = [("system", system_prompt), ("user", user_prompt)]
        story_bible = None
        for idx in range(max_retries):
            try:
                logger.info(
                    f"Generating story bible (attempt {idx + 1}/{max_retries})..."
                )
                story_bible = structured_llm.invoke(messages)
            except Exception as e:
                logger.error(
                    f"Error in generating story bible (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)
                continue
            if story_bible is not None:
                break

        if story_bible is None:
            raise ValueError("Failed to generate story bible")

        # 4. JSON 파일로 저장
        output_path = OUTPUT_DIR / "story_bible.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(story_bible.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Story Bible saved to {output_path}")

        # 5. 결과 반환
        return {"story_bible": story_bible}

    except Exception as e:
        logger.error(f"Error in generate_story_bible_node: {e}")
        return {"errors": [f"Story Bible generation failed: {str(e)}"]}


def generate_character_bible_node(state: GraphState) -> dict:
    """
    Character Bible 생성 노드

    Args:
        state: GraphState dict containing user_input, story_bible

    Returns:
        dict: {"character_bible": CharacterBible}
    """
    try:
        logger.info("Starting Character Bible generation...")

        # 1. state에서 필요한 데이터 추출
        user_input = state.get("user_input", "")
        story_bible = state.get("story_bible")
        max_retries = state.get("max_retries", 3)
        if not user_input:
            raise ValueError("user_input is required")
        if not story_bible:
            raise ValueError("story_bible is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "character_agent.prompt")
        user_prompt_template = load_prompt(
            PROMPTS_DIR / "user" / "character_agent.prompt"
        )

        # story_bible을 JSON 문자열로 변환하여 프롬프트에 포함
        story_context = story_bible.model_dump_json(indent=2)
        user_prompt = user_prompt_template.format(
            story_context=story_context, user_input=user_input
        )

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(CharacterBible)

        messages = [("system", system_prompt), ("user", user_prompt)]

        character_bible = None
        for idx in range(max_retries):
            logger.info(
                f"Generating character bible (attempt {idx + 1}/{max_retries})..."
            )
            try:
                character_bible = structured_llm.invoke(messages)
            except Exception as e:
                logger.error(
                    f"Error in generating character bible (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)
                continue
            if character_bible is not None:
                break

        # 4. JSON 파일로 저장
        output_path = OUTPUT_DIR / "character_bible.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(character_bible.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Character Bible saved to {output_path}")

        # 5. 결과 반환
        return {"character_bible": character_bible}

    except Exception as e:
        logger.error(f"Error in generate_character_bible_node: {e}")
        return {"errors": [f"Character Bible generation failed: {str(e)}"]}


def generate_style_bible_node(state: GraphState) -> dict:
    """
    Style Bible 생성 노드

    Args:
        state: GraphState dict containing user_input, story_bible, character_bible

    Returns:
        dict: {"style_bible": StyleBible}
    """
    try:
        logger.info("Starting Style Bible generation...")

        # 1. state에서 필요한 데이터 추출
        user_input = state.get("user_input", "")
        story_bible = state.get("story_bible")
        character_bible = state.get("character_bible")
        max_retries = state.get("max_retries", 3)
        if not user_input:
            raise ValueError("user_input is required")
        if not story_bible:
            raise ValueError("story_bible is required")
        if not character_bible:
            raise ValueError("character_bible is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(
            PROMPTS_DIR / "system" / "visual_style_agent.prompt"
        )
        user_prompt_template = load_prompt(
            PROMPTS_DIR / "user" / "visual_style_agent.prompt"
        )

        # 이전 bible들을 JSON 문자열로 변환하여 프롬프트에 포함
        story_context = story_bible.model_dump_json(indent=2)
        character_context = character_bible.model_dump_json(indent=2)
        user_prompt = user_prompt_template.format(
            story_context=story_context,
            character_context=character_context,
            user_input=user_input,
        )

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(StyleBible)

        messages = [("system", system_prompt), ("user", user_prompt)]
        style_bible = None
        for idx in range(max_retries):
            logger.info(f"Generating style bible (attempt {idx + 1}/{max_retries})...")
            try:
                style_bible = structured_llm.invoke(messages)
            except Exception as e:
                logger.error(
                    f"Error in generating style bible (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)
                continue
            if style_bible is not None:
                break

        # 4. JSON 파일로 저장
        output_path = OUTPUT_DIR / "style_bible.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(style_bible.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Style Bible saved to {output_path}")

        # 5. 결과 반환
        return {"style_bible": style_bible}

    except Exception as e:
        logger.error(f"Error in generate_style_bible_node: {e}")
        return {"errors": [f"Style Bible generation failed: {str(e)}"]}


def generate_director_plan_node(state: GraphState) -> dict:
    """
    Director Plan 생성 노드

    Args:
        state: GraphState dict containing story_bible, character_bible, style_bible

    Returns:
        dict: {"director_output": DirectorOutput}
    """
    try:
        logger.info("Starting Director Plan generation...")

        # 1. state에서 필요한 데이터 추출
        story_bible = state.get("story_bible")
        character_bible = state.get("character_bible")
        style_bible = state.get("style_bible")

        max_retries = state.get("max_retries", 3)
        if not story_bible:
            raise ValueError("story_bible is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        if not style_bible:
            raise ValueError("style_bible is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "director.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "director.prompt")

        # 모든 bible을 JSON 문자열로 변환하여 프롬프트에 포함
        story_context = story_bible.model_dump_json(indent=2)
        character_context = character_bible.model_dump_json(indent=2)
        style_context = style_bible.model_dump_json(indent=2)

        user_prompt = user_prompt_template.format(
            story_bible=story_context,
            character_bible=character_context,
            style_bible=style_context,
        )

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(DirectorOutput)

        messages = [("system", system_prompt), ("user", user_prompt)]

        for idx in range(max_retries):
            logger.info(
                f"Generating director plan (attempt {idx + 1}/{max_retries})..."
            )
            try:
                director_output = structured_llm.invoke(messages)
                # 4. 유효성 검증
                if len(director_output.keyframes) != 5:
                    logger.warning(
                        f"Expected 5 keyframes, got {len(director_output.keyframes)}"
                    )
                    time.sleep(1)
                elif len(director_output.shots) != 4:
                    logger.warning(
                        f"Expected 4 shots, got {len(director_output.shots)}"
                    )
                    time.sleep(1)
                else:
                    break

            except Exception as e:
                logger.error(
                    f"Error in generating director plan (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        if director_output is None:
            raise ValueError("Failed to generate director plan")

        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "director_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(director_output.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Director Plan saved to {output_path}")
        logger.info(
            f"Generated {len(director_output.keyframes)} keyframes and {len(director_output.shots)} shots"
        )

        # 6. 결과 반환
        return {"director_output": director_output}

    except Exception as e:
        logger.error(f"Error in generate_director_plan_node: {e}")
        return {"errors": [f"Director Plan generation failed: {str(e)}"]}


def generate_image_prompts_node(state: GraphState) -> dict:
    """
    Image Prompts 생성 노드

    Args:
        state: GraphState dict containing director_output, character_bible, style_bible

    Returns:
        dict: {"image_engineer_output": ImageEngineerOutput}
    """
    try:
        logger.info("Starting Image Prompts generation...")

        # 1. state에서 필요한 데이터 추출
        director_output = state.get("director_output")
        character_bible = state.get("character_bible")
        style_bible = state.get("style_bible")

        max_retries = state.get("max_retries", 3)
        if not director_output:
            raise ValueError("director_output is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        if not style_bible:
            raise ValueError("style_bible is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "image_specialist.prompt")
        user_prompt_template = load_prompt(
            PROMPTS_DIR / "user" / "image_specialist.prompt"
        )

        # keyframes 정보만 추출 (image specialist는 keyframes만 필요)
        keyframes_context = json.dumps(
            [kf.model_dump() for kf in director_output.keyframes],
            indent=2,
            ensure_ascii=False,
        )
        character_context = character_bible.model_dump_json(indent=2)
        style_context = style_bible.model_dump_json(indent=2)

        user_prompt = user_prompt_template.format(
            keyframes_plan=keyframes_context,
            character_bible=character_context,
            style_bible=style_context,
        )

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(ImageEngineerOutput)

        messages = [("system", system_prompt), ("user", user_prompt)]

        image_engineer_output = None
        for idx in range(max_retries):
            logger.info(
                f"Generating image prompts (attempt {idx + 1}/{max_retries})..."
            )
            try:
                image_engineer_output = structured_llm.invoke(messages)
                # 4. 유효성 검증
                if len(image_engineer_output.keyframe_prompts) != 5:
                    logger.warning(
                        f"Expected 5 keyframe prompts, got {len(image_engineer_output.keyframe_prompts)}"
                    )
                    time.sleep(1)
                else:
                    break
            except Exception as e:
                logger.error(
                    f"Error in generating image prompts (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        if image_engineer_output is None:
            raise ValueError("Failed to generate image prompts")

        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "image_engineer_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                image_engineer_output.model_dump(), f, indent=2, ensure_ascii=False
            )

        logger.info(f"Image Prompts saved to {output_path}")
        logger.info(
            f"Generated {len(image_engineer_output.character_reference_prompt)} reference prompts and {len(image_engineer_output.keyframe_prompts)} keyframe prompts"
        )

        # 6. 결과 반환
        return {"image_engineer_output": image_engineer_output}

    except Exception as e:
        logger.error(f"Error in generate_image_prompts_node: {e}")
        return {"errors": [f"Image Prompts generation failed: {str(e)}"]}


def generate_video_prompts_node(state: GraphState) -> dict:
    """
    Video Prompts 생성 노드

    Args:
        state: GraphState dict containing director_output, character_bible, style_bible

    Returns:
        dict: {"video_engineer_output": VideoEngineerOutput}
    """
    try:
        logger.info("Starting Video Prompts generation...")

        # 1. state에서 필요한 데이터 추출
        director_output = state.get("director_output")
        character_bible = state.get("character_bible")
        style_bible = state.get("style_bible")

        max_retries = state.get("max_retries", 3)
        if not director_output:
            raise ValueError("director_output is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        if not style_bible:
            raise ValueError("style_bible is required")

        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "video_specialist.prompt")
        user_prompt_template = load_prompt(
            PROMPTS_DIR / "user" / "video_specialist.prompt"
        )

        # shots 정보만 추출 (video specialist는 shots만 필요)
        shots_context = json.dumps(
            [shot.model_dump() for shot in director_output.shots],
            indent=2,
            ensure_ascii=False,
        )
        character_context = character_bible.model_dump_json(indent=2)
        style_context = style_bible.model_dump_json(indent=2)

        user_prompt = user_prompt_template.format(
            shot_plan=shots_context,
            character_bible=character_context,
            style_bible=style_context,
        )

        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(VideoEngineerOutput)

        messages = [("system", system_prompt), ("user", user_prompt)]

        video_engineer_output = None
        for idx in range(max_retries):
            logger.info(
                f"Generating video prompts (attempt {idx + 1}/{max_retries})..."
            )
            try:
                video_engineer_output = structured_llm.invoke(messages)
                # 4. 유효성 검증
                if len(video_engineer_output.video_prompts) != 4:
                    logger.warning(
                        f"Expected 4 video prompts, got {len(video_engineer_output.video_prompts)}"
                    )
                    time.sleep(1)
                else:
                    break

            except Exception as e:
                logger.error(
                    f"Error in generating video prompts (attempt {idx + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        if video_engineer_output is None:
            raise ValueError("Failed to generate video prompts")

        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "video_engineer_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                video_engineer_output.model_dump(), f, indent=2, ensure_ascii=False
            )

        logger.info(f"Video Prompts saved to {output_path}")
        logger.info(
            f"Generated {len(video_engineer_output.video_prompts)} video prompts"
        )

        # 6. 결과 반환
        return {"video_engineer_output": video_engineer_output}

    except Exception as e:
        logger.error(f"Error in generate_video_prompts_node: {e}")
        return {"errors": [f"Video Prompts generation failed: {str(e)}"]}


def _generate_image_with_retry(
    api_key: str,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    reference_img_paths: Optional[Sequence[str | Path]] = None,
) -> Any:
    """
    이미지 생성 API 호출 및 재시도 로직

    Args:
        api_key: API 키
        model_name: 모델 이름
        prompt: 이미지 생성 프롬프트
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 대기 시간 (초)
        reference_img_paths: 레퍼런스 이미지 경로

    Returns:
        생성된 이미지 응답

    Raises:
        Exception: 최대 재시도 후에도 실패한 경우
    """
    last_error = None
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=180000)
    )

    if reference_img_paths:
        reference_imgs = [Image.open(path) for path in reference_img_paths]
        contents = [prompt] + reference_imgs
    else:
        contents = [prompt]

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating image (attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            return response
        except Exception as e:
            last_error = e
            logger.warning(
                f"Image generation failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise Exception(
        f"Image generation failed after {max_retries} attempts: {last_error}"
    )


def generate_assets_node(state: GraphState) -> dict:
    """
    에셋(레퍼런스 이미지) 생성 노드

    Args:
        state: GraphState dict containing image_engineer_output, style_bible

    Returns:
        dict: {"assets": {"Character Name": "path/to/asset.png", ...}}
    """
    try:
        logger.info("Starting Assets generation...")

        # 1. state에서 필요한 데이터 추출
        image_engineer_output = state.get("image_engineer_output")
        style_bible = state.get("style_bible")

        if not image_engineer_output:
            raise ValueError("image_engineer_output is required")
        if not style_bible:
            raise ValueError("style_bible is required")

        # 2. Gemini 클라이언트 초기화
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        model_name = "gemini-3-pro-image-preview"  # gemini-2.5-flash-image, gemini-3-pro-image-preview

        # 3. 각 캐릭터 레퍼런스 이미지 생성
        assets = {}
        latest_character_name = None

        for ref_prompt in image_engineer_output.character_reference_prompt:
            character_name = ref_prompt.name
            prompt = ref_prompt.prompt

            logger.info(f"Generating asset for: {character_name}")

            try:
                # API 호출 (재시도 로직 포함)
                if latest_character_name is not None:
                    reference_img_paths = [ASSETS_DIR / f"{latest_character_name}.png"]
                    full_prompt = f"Follow the style of the provided image as a strict reference.\n\n{prompt}\n\nNegative prompt: {image_engineer_output.common_negative_prompt}"
                else:
                    reference_img_paths = []
                    full_prompt = f"{prompt}\n\nNegative prompt: {image_engineer_output.common_negative_prompt}"
                response = _generate_image_with_retry(
                    api_key,
                    model_name,
                    full_prompt,
                    reference_img_paths=reference_img_paths,
                )

                # 이미지 저장
                # 캐릭터 이름에서 파일명 생성 (공백을 언더스코어로 치환)
                safe_name = character_name.replace(" ", "_").lower()
                latest_character_name = safe_name
                asset_path = ASSETS_DIR / f"{safe_name}.png"

                # 응답에서 이미지 추출 및 저장
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if (
                                hasattr(part, "inline_data")
                                and part.inline_data is not None
                            ):
                                image = part.as_image()
                                image.save(asset_path)
                                logger.info(f"Asset saved to {asset_path}")
                                assets[character_name] = str(asset_path)
                                break
                else:
                    logger.warning(
                        f"No image data found in response for {character_name}"
                    )

            except Exception as e:
                logger.error(f"Failed to generate asset for {character_name}: {e}")
                # 에러 발생 시 계속 진행 (다른 에셋 생성 시도)
                continue

        if not assets:
            raise ValueError("Failed to generate any assets")

        logger.info(f"Successfully generated {len(assets)} assets")

        # 4. 결과 반환
        return {"assets": assets}

    except Exception as e:
        logger.error(f"Error in generate_assets_node: {e}")
        return {"errors": [f"Assets generation failed: {str(e)}"]}


def generate_frames_node(state: GraphState) -> dict:
    """
    키프레임 이미지 생성 노드 (레퍼런스 이미지 활용)

    Args:
        state: GraphState dict containing image_engineer_output, assets, style_bible

    Returns:
        dict: {"frames": ["path/to/keyframe_0.png", ...]}
    """
    try:
        logger.info("Starting Frames generation...")

        # 1. state에서 필요한 데이터 추출
        image_engineer_output = state.get("image_engineer_output")
        assets = state.get("assets", {})
        style_bible = state.get("style_bible")

        if not image_engineer_output:
            raise ValueError("image_engineer_output is required")
        if not style_bible:
            raise ValueError("style_bible is required")

        # 2. Gemini 클라이언트 초기화
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        model_name = "gemini-3-pro-image-preview"  # gemini-2.5-flash-image, gemini-3-pro-image-preview

        # 3. 각 키프레임 이미지 생성
        frames = []

        # 유효성 검증
        if len(image_engineer_output.keyframe_prompts) != 5:
            raise ValueError(
                f"Expected 5 keyframe prompts, got {len(image_engineer_output.keyframe_prompts)}"
            )

        for idx, keyframe_prompt in enumerate(image_engineer_output.keyframe_prompts):
            keyframe_index = idx
            prompt = keyframe_prompt.prompt

            logger.info(f"Generating keyframe {keyframe_index}...")

            # 레퍼런스 이미지 정보를 프롬프트에 포함
            # (현재 Gemini API에서 레퍼런스 이미지를 프롬프트로 전달하는 방법 적용)
            reference_img_paths = []
            if assets:
                reference_img_paths = [
                    item for item in assets.values() if item is not None
                ]
                print(f"reference img paths: {reference_img_paths}")
            try:
                # API 호출 (재시도 로직 포함)
                if idx != 0:
                    # 이전 프레임 이미지를 레퍼런스로 추가
                    reference_img_paths = [
                        FRAMES_DIR / f"keyframe_{idx - 1}.png"
                    ] + reference_img_paths

                if reference_img_paths:
                    prompt = (
                        f"Use the provided images as a strict reference.\n\n{prompt}"
                    )
                else:
                    prompt = prompt

                response = _generate_image_with_retry(
                    api_key, model_name, prompt, reference_img_paths=reference_img_paths
                )

                # 이미지 저장
                frame_path = FRAMES_DIR / f"keyframe_{idx}.png"

                # 응답에서 이미지 추출 및 저장
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if (
                                hasattr(part, "inline_data")
                                and part.inline_data is not None
                            ):
                                # PIL Image로 변환하여 저장
                                image = part.as_image()
                                image.save(frame_path)
                                logger.info(f"Keyframe saved to {frame_path}")
                                frames.append(frame_path)
                                break
                else:
                    logger.warning(
                        f"No image data found in response for keyframe {keyframe_index}"
                    )

            except Exception as e:
                logger.error(f"Failed to generate keyframe {keyframe_index}: {e}")
                # 에러 발생 시 계속 진행 (다른 프레임 생성 시도)
                continue

        if len(frames) != 5:
            logger.warning(f"Expected 5 frames, but generated {len(frames)}")

        if not frames:
            raise ValueError("Failed to generate any frames")

        logger.info(f"Successfully generated {len(frames)} frames")

        # 4. 결과 반환
        return {"frames": frames}

    except Exception as e:
        logger.error(f"Error in generate_frames_node: {e}")
        return {"errors": [f"Frames generation failed: {str(e)}"]}


def _extract_last_frame_opencv(video_path: str | Path, output_path: str | Path) -> bool:
    """
    OpenCV를 사용하여 비디오의 마지막 프레임을 추출합니다.

    Args:
        video_path: 비디오 파일 경로
        output_path: 저장할 이미지 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Error: 동영상 파일을 열 수 없습니다: {video_path}")
            return False

        # 총 프레임 수 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 마지막 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # 프레임 읽기
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(str(output_path), frame)
            logger.info(f"마지막 프레임이 저장되었습니다: {output_path}")
            cap.release()
            return True
        else:
            logger.error(f"Error: 프레임을 읽을 수 없습니다: {video_path}")
            cap.release()
            return False

    except Exception as e:
        logger.error(f"Failed to extract last frame from {video_path}: {e}")
        return False


def _generate_video_with_interpolation(
    api_key: str,
    video_prompt: str,
    first_frame_path: str | Path,
    last_frame_path: str | Path,
    save_path: str | Path,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> bool:
    """
    Veo 3.1 API를 사용하여 프레임 보간으로 비디오를 생성합니다.

    Args:
        api_key: API 키
        video_prompt: 비디오 생성 프롬프트
        first_frame_path: 첫 번째 프레임 이미지 경로
        last_frame_path: 마지막 프레임 이미지 경로
        save_path: 저장할 비디오 파일 경로
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 대기 시간 (초)

    Returns:
        bool: 성공 여부
    """
    last_error = None
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=120000)
    )
    for attempt in range(max_retries):
        try:
            logger.info(f"Generating video (attempt {attempt + 1}/{max_retries})...")
            logger.info(f"Prompt: {video_prompt[:100]}...")

            # 프레임 이미지 로드
            first_frame = types.Image.from_file(location=str(first_frame_path))
            last_frame = types.Image.from_file(location=str(last_frame_path))
            # Veo 3.1 API 호출 (frame interpolation)
            operation = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=video_prompt,
                image=first_frame,
                config=types.GenerateVideosConfig(
                    last_frame=last_frame,
                ),
            )

            logger.info(f"Operation started: {operation.name}")

            # 비디오 생성 완료 대기 (폴링)
            while not operation.done:
                logger.info("Waiting for video generation to complete...")
                time.sleep(retry_delay)
                operation = client.operations.get(operation)

            # 생성 완료 확인
            if not operation.response:
                raise Exception("Video generation completed but no response received")

            if (
                not hasattr(operation.response, "generated_videos")
                or not operation.response.generated_videos
            ):
                raise Exception("No generated videos in response")

            # 비디오 다운로드 및 저장
            video = operation.response.generated_videos[0]
            if not video or not hasattr(video, "video") or not video.video:
                raise Exception("Invalid video object in response")

            client.files.download(file=video.video)
            video.video.save(str(save_path))

            logger.info(f"Generated video saved to {save_path}")
            return True

        except Exception as e:
            last_error = e
            logger.warning(
                f"Video generation failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    logger.error(f"Video generation failed after {max_retries} attempts: {last_error}")
    return False


def generate_videos_node(state: GraphState) -> dict:
    """
    Veo 3.1 API를 사용하여 4개의 비디오를 생성하는 노드
    실패시 veo_failed 플래그를 True로 설정합니다.

    Args:
        state: GraphState dict containing frames, video_engineer_output

    Returns:
        dict: {"videos": ["path/to/video_0.mp4", ...], "veo_failed": bool}
    """
    try:
        logger.info("Starting Videos generation with Veo 3.1...")

        # 1. state에서 필요한 데이터 추출
        frames = state.get("frames")
        video_engineer_output = state.get("video_engineer_output")

        if not frames or len(frames) != 5:
            raise ValueError(f"Expected 5 frames, got {len(frames) if frames else 0}")
        if not video_engineer_output:
            raise ValueError("video_engineer_output is required")
        if len(video_engineer_output.video_prompts) != 4:
            raise ValueError(
                f"Expected 4 video prompts, got {len(video_engineer_output.video_prompts)}"
            )

        # 2. Google GenAI Client 초기화
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        # 3. 각 비디오 생성 (4개)
        videos = []
        temp_frames_dir = VIDEOS_DIR / "temp_frames"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            video_prompt_obj = video_engineer_output.video_prompts[i]
            video_prompt = video_prompt_obj.prompt

            logger.info(
                f"Generating video {i + 1}/4 (shot {video_prompt_obj.shot_index}) with Veo 3.1..."
            )

            # 첫 번째 프레임 결정
            if i == 0:
                # 첫 번째 비디오: frames[0]을 그대로 사용
                first_frame_path = frames[0]
            else:
                # 두 번째 비디오 이후: 이전 비디오의 마지막 프레임 추출
                prev_video_path = videos[i - 1]
                first_frame_path = temp_frames_dir / f"temp_frame_{i}.png"

                logger.info(
                    f"Extracting last frame from previous video: {prev_video_path}"
                )
                if not _extract_last_frame_opencv(prev_video_path, first_frame_path):
                    raise Exception(f"Failed to extract last frame from video {i - 1}")

            # 마지막 프레임: frames[i+1]
            last_frame_path = frames[i + 1]

            # 비디오 저장 경로
            video_path = VIDEOS_DIR / f"video_{i}.mp4"

            # 비디오 생성 (재시도 로직 포함)
            success = _generate_video_with_interpolation(
                api_key=api_key,
                video_prompt=video_prompt,
                first_frame_path=first_frame_path,
                last_frame_path=last_frame_path,
                save_path=video_path,
                max_retries=3,
                retry_delay=10.0,
            )

            if not success:
                raise Exception(f"Failed to generate video {i} with Veo 3.1")

            videos.append(str(video_path))
            logger.info(f"Video {i + 1}/4 completed: {video_path}")

        if len(videos) != 4:
            raise ValueError(f"Expected 4 videos, but generated {len(videos)}")

        logger.info(f"Successfully generated {len(videos)} videos with Veo 3.1")

        # 4. 결과 반환 (성공)
        return {"videos": videos, "veo_failed": False}

    except Exception as e:
        logger.error(f"Error in generate_videos_node: {e}")
        logger.warning("Veo 3.1 failed, will try Kie API...")
        
        # 실패시 veo_failed 플래그를 True로 설정하고 반환
        return {"veo_failed": True, "errors": [f"Veo 3.1 video generation failed: {str(e)}"]}


def _merge_videos_ffmpeg(
    video_paths: Sequence[str | Path], output_path: str | Path
) -> bool:
    """
    FFmpeg를 사용하여 여러 비디오를 병합합니다.

    Args:
        video_paths: 병합할 비디오 파일 경로 리스트
        output_path: 저장할 최종 비디오 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        # 파일 목록을 담은 텍스트 파일 생성
        file_list_path = VIDEOS_DIR / "file_list.txt"
        with open(file_list_path, "w") as f:
            for path in video_paths:
                abs_path = os.path.abspath(path)
                f.write(f"file '{abs_path}'\n")

        logger.info(f"Created file list: {file_list_path}")

        # FFmpeg concat demuxer 실행
        cmd = [
            "ffmpeg",
            "-y",  # 기존 파일을 묻지 않고 덮어쓰기
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",  # 재인코딩 없이 스트림 복사
            str(output_path),
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 임시 파일 삭제
        os.remove(file_list_path)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False

        logger.info(f"Videos merged successfully to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to merge videos: {e}")
        return False


def postprocess_node(state: GraphState) -> dict:
    """
    FFmpeg를 사용하여 4개의 비디오를 병합하는 후처리 노드

    Args:
        state: GraphState dict containing videos

    Returns:
        dict: {"final_video": "path/to/final_video.mp4"}
    """
    try:
        logger.info("Starting Video Postprocessing...")

        # 1. state에서 비디오 목록 추출
        videos = state.get("videos")

        if not videos or len(videos) != 4:
            raise ValueError(f"Expected 4 videos, got {len(videos) if videos else 0}")

        # 2. 모든 비디오 파일이 존재하는지 확인
        for video_path in videos:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

        # 3. FFmpeg로 병합
        final_video_path = OUTPUT_DIR / "final_video.mp4"

        success = _merge_videos_ffmpeg(videos, final_video_path)

        if not success:
            raise Exception("Failed to merge videos with FFmpeg")

        logger.info(f"Final video saved to {final_video_path}")

        # 4. 결과 반환
        return {"final_video": str(final_video_path)}

    except Exception as e:
        logger.error(f"Error in postprocess_node: {e}")
        return {"errors": [f"Video postprocessing failed: {str(e)}"]}


# ==========================================
# GCS Upload Utilities & Node
# ==========================================


def _upload_to_gcs(
    local_file_path: str | Path,
    gcs_blob_name: str,
    bucket_name: str = GCS_BUCKET_NAME,
    service_account_key_path: str = GCS_SERVICE_ACCOUNT_KEY_PATH,
) -> str:
    """
    GCS에 파일을 업로드하고 공개 URL을 반환합니다.

    Args:
        local_file_path: 업로드할 로컬 파일 경로
        gcs_blob_name: GCS에 저장될 blob 이름 (경로 포함)
        bucket_name: GCS 버킷 이름
        service_account_key_path: 서비스 계정 키 파일 경로

    Returns:
        str: 업로드된 파일의 공개 URL
    """
    try:
        # 서비스 계정 키로 클라이언트 생성
        storage_client = storage.Client.from_service_account_json(
            service_account_key_path
        )

        # 버킷 및 Blob 선택
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_blob_name)

        # 업로드 실행
        logger.info(f"Uploading {local_file_path} to GCS as {gcs_blob_name}...")
        blob.upload_from_filename(str(local_file_path))
        
        # 공개 URL 생성 (Signed URL 대신 공개 URL 사용)
        public_url = f"https://storage.googleapis.com/{bucket_name}/{gcs_blob_name}"
        
        logger.info(f"Upload successful: {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to GCS: {e}")
        raise


def upload_assets_and_frames_to_gcs_node(state: GraphState) -> dict:
    """
    에셋 및 프레임을 GCS에 업로드하는 노드

    Args:
        state: GraphState dict containing assets, frames

    Returns:
        dict: {"gcs_urls": {"assets": [...], "frames": [...]}}
    """
    try:
        logger.info("Starting GCS upload for assets and frames...")

        # 1. state에서 에셋 및 프레임 추출
        assets = state.get("assets", {})
        frames = state.get("frames", [])

        if not frames:
            raise ValueError("No frames to upload")

        gcs_urls = {"assets": [], "frames": []}

        # 2. 에셋 업로드 (선택적)
        if assets:
            for character_name, asset_path in assets.items():
                safe_name = character_name.replace(" ", "_").lower()
                gcs_blob_name = f"assets/{safe_name}.png"
                
                try:
                    url = _upload_to_gcs(asset_path, gcs_blob_name)
                    gcs_urls["assets"].append(url)
                except Exception as e:
                    logger.warning(f"Failed to upload asset {character_name}: {e}")
                    # 에셋 업로드 실패는 치명적이지 않으므로 계속 진행

        # 3. 프레임 업로드
        for idx, frame_path in enumerate(frames):
            gcs_blob_name = f"assets/frames/keyframe_{idx}.png"
            
            try:
                url = _upload_to_gcs(frame_path, gcs_blob_name)
                gcs_urls["frames"].append(url)
            except Exception as e:
                logger.error(f"Failed to upload frame {idx}: {e}")
                raise  # 프레임 업로드 실패는 치명적

        logger.info(
            f"Successfully uploaded {len(gcs_urls['assets'])} assets and {len(gcs_urls['frames'])} frames to GCS"
        )

        # 4. 결과 반환
        return {"gcs_urls": gcs_urls}

    except Exception as e:
        logger.error(f"Error in upload_assets_and_frames_to_gcs_node: {e}")
        return {"errors": [f"GCS upload failed: {str(e)}"]}


# ==========================================
# Kie API Utilities & Node
# ==========================================


def _download_video_from_url(video_url: str, save_path: str | Path) -> bool:
    """
    URL에서 비디오를 다운로드하여 로컬에 저장합니다.

    Args:
        video_url: 다운로드할 비디오 URL
        save_path: 저장할 로컬 파일 경로

    Returns:
        bool: 성공 여부
    """
    try:
        logger.info(f"Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        # 디렉터리 생성
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Video downloaded successfully to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return False


def _check_kie_video_status(task_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """
    Kie API 비디오 생성 상태를 확인합니다.

    Args:
        task_id: Kie API 작업 ID
        api_key: Kie API 키

    Returns:
        dict: 완료된 경우 응답 데이터, 진행 중이면 None, 실패시 False
    """
    if api_key is None:
        api_key = KIE_API_KEY
    if not api_key:
        raise ValueError("KIE_API_KEY is not set")
    
    url = f"{KIE_API_BASE_URL}/record-info?taskId={task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        result = response.json()

        if response.ok and result.get("code") == 200:
            data = result["data"]
            success_flag = data["successFlag"]

            if success_flag == 0:
                logger.info("Kie API: Generating...")
                return None
            elif success_flag == 1:
                logger.info("Kie API: Generation successful!")
                return data
            else:
                logger.error(f"Kie API: Generation failed - {result.get('msg')}")
                return False
        else:
            logger.error(f"Kie API: Status check failed - {result.get('msg')}")
            return None

    except Exception as e:
        logger.error(f"Kie API: Status check error - {e}")
        return None


def _wait_for_kie_completion(task_id: str, api_key: Optional[str] = None, poll_interval: int = 30, max_retries: int = 8) -> Optional[dict]:
    """
    Kie API 비디오 생성이 완료될 때까지 대기합니다.

    Args:
        task_id: Kie API 작업 ID
        api_key: Kie API 키
        poll_interval: 폴링 간격 (초)

    Returns:
        dict: 완료된 경우 응답 데이터, 실패시 None
    """
    if api_key is None:
        api_key = KIE_API_KEY
    if not api_key:
        raise ValueError("KIE_API_KEY is not set")
    
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        result = _check_kie_video_status(task_id, api_key)
        if result is False:
            return None
        elif result is not None:
            return result
        logger.info(f"Kie API: Waiting for completion... (attempt {attempt}/{max_retries})")
        time.sleep(poll_interval)
    logger.error(f"Kie API: Failed to complete video generation after {max_retries} attempts")
    return None


def _generate_video_with_kie(
    prompt: str,
    first_frame_url: str,
    last_frame_url: str,
    save_path: str | Path,
    api_key: Optional[str] = None,
    model: str = "veo3_fast",
    aspect_ratio: str = "16:9",
) -> bool:
    """
    Kie API를 사용하여 비디오를 생성합니다.

    Args:
        prompt: 비디오 생성 프롬프트
        first_frame_url: 첫 번째 프레임 이미지 GCS URL
        last_frame_url: 마지막 프레임 이미지 GCS URL
        save_path: 저장할 비디오 파일 경로
        api_key: Kie API 키
        model: 모델 이름
        aspect_ratio: 화면 비율

    Returns:
        bool: 성공 여부
    """
    try:
        if api_key is None:
            api_key = KIE_API_KEY
        if not api_key:
            raise ValueError("KIE_API_KEY is not set")
        
        logger.info("Generating video with Kie API...")

        # 1. 비디오 생성 요청
        url = f"{KIE_API_BASE_URL}/generate"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "imageUrls": [first_frame_url, last_frame_url],
            "model": model,
            "aspect_ratio": aspect_ratio,
            "enableTranslation": True,
            "generationType": "FIRST_AND_LAST_FRAMES_2_VIDEO",
        }

        response = requests.post(url, json=payload, headers=headers)
        result = response.json()

        if not response.ok or result.get("code") != 200:
            logger.error(f"Kie API: Request failed - {result.get('msg')}")
            return False

        task_id = result["data"]["taskId"]
        logger.info(f"Kie API: Task submitted with ID {task_id}")

        # 2. 생성 완료 대기
        completion_result = _wait_for_kie_completion(task_id, api_key)

        if not completion_result:
            logger.error("Kie API: Video generation failed or timed out")
            return False

        # 3. 비디오 다운로드
        video_urls = completion_result["response"]["resultUrls"]
        if not video_urls:
            logger.error("Kie API: No video URLs in response")
            return False

        video_url = video_urls[0]
        success = _download_video_from_url(video_url, save_path)

        if success:
            logger.info(f"Kie API: Video saved to {save_path}")

        return success

    except Exception as e:
        logger.error(f"Kie API: Error during video generation - {e}")
        return False


def generate_videos_with_kie_node(state: GraphState) -> dict:
    """
    Kie API를 사용하여 4개의 비디오를 생성하는 노드

    Args:
        state: GraphState dict containing gcs_urls, video_engineer_output

    Returns:
        dict: {"videos": ["path/to/video_0.mp4", ...]}
    """
    try:
        logger.info("Starting video generation with Kie API...")

        # 1. state에서 필요한 데이터 추출
        gcs_urls = state.get("gcs_urls")
        video_engineer_output = state.get("video_engineer_output")

        if not gcs_urls or not gcs_urls.get("frames"):
            raise ValueError("GCS frame URLs not found")
        if not video_engineer_output:
            raise ValueError("video_engineer_output is required")
        if len(video_engineer_output.video_prompts) != 4:
            raise ValueError(
                f"Expected 4 video prompts, got {len(video_engineer_output.video_prompts)}"
            )

        frame_urls = gcs_urls["frames"]
        if len(frame_urls) != 5:
            raise ValueError(f"Expected 5 frame URLs, got {len(frame_urls)}")

        # 2. 각 비디오 생성 (4개)
        videos = []
        temp_frames_dir = VIDEOS_DIR / "temp_frames"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            video_prompt_obj = video_engineer_output.video_prompts[i]
            video_prompt = video_prompt_obj.prompt

            logger.info(
                f"Generating video {i + 1}/4 (shot {video_prompt_obj.shot_index}) with Kie API..."
            )

            # 첫 번째 프레임 URL 결정
            if i == 0:
                # 첫 번째 비디오: GCS의 첫 번째 프레임 URL 사용
                first_frame_url = frame_urls[0]
            else:
                # 두 번째 비디오 이후: 이전 비디오의 마지막 프레임 추출 후 GCS 업로드
                prev_video_path = videos[i - 1]
                temp_frame_path = temp_frames_dir / f"temp_frame_{i}.png"

                logger.info(
                    f"Extracting last frame from previous video: {prev_video_path}"
                )
                if not _extract_last_frame_opencv(prev_video_path, temp_frame_path):
                    raise Exception(f"Failed to extract last frame from video {i - 1}")

                # GCS에 업로드
                gcs_blob_name = f"assets/frames/temp_keyframe_{i}.png"
                first_frame_url = _upload_to_gcs(temp_frame_path, gcs_blob_name)

            # 마지막 프레임 URL: GCS의 다음 프레임
            last_frame_url = frame_urls[i + 1]

            # 비디오 저장 경로
            video_path = VIDEOS_DIR / f"video_{i}.mp4"

            # Kie API로 비디오 생성
            success = _generate_video_with_kie(
                prompt=video_prompt,
                first_frame_url=first_frame_url,
                last_frame_url=last_frame_url,
                save_path=video_path,
            )

            if not success:
                raise Exception(f"Failed to generate video {i} with Kie API")

            videos.append(str(video_path))
            logger.info(f"Video {i + 1}/4 completed: {video_path}")

        if len(videos) != 4:
            raise ValueError(f"Expected 4 videos, but generated {len(videos)}")

        logger.info(f"Successfully generated {len(videos)} videos with Kie API")

        # 3. 결과 반환
        return {"videos": videos}

    except Exception as e:
        logger.error(f"Error in generate_videos_with_kie_node: {e}")
        return {"errors": [f"Kie API video generation failed: {str(e)}"]}


# ==========================================
# Router Functions
# ==========================================


def route_after_veo_generation(state: GraphState) -> str:
    """
    Veo 3.1 비디오 생성 후 다음 노드를 결정하는 라우터

    Args:
        state: GraphState dict containing veo_failed

    Returns:
        str: 다음 노드 이름 ("postprocess" 또는 "kie_videos")
    """
    veo_failed = state.get("veo_failed", False)
    
    if veo_failed:
        logger.info("Routing to Kie API due to Veo 3.1 failure")
        return "kie_videos"
    else:
        logger.info("Routing to postprocess (Veo 3.1 success)")
        return "postprocess"
