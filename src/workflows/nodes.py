"""
LangGraph Nodes for Drama Making Workflow
"""

import json
import time
import os
import subprocess
import cv2
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional, List, Sequence
from google import genai
from google.genai import types
from src.config import load_prompt, get_llm, OUTPUT_DIR, logger, PROMPTS_DIR, ASSETS_DIR, FRAMES_DIR, VIDEOS_DIR
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
        if not user_input:
            raise ValueError("user_input is required")
        
        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "story_agent.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "story_agent.prompt")
        user_prompt = user_prompt_template.format(user_input=user_input)
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(StoryBible)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        story_bible = structured_llm.invoke(messages)
        
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
        
        if not user_input:
            raise ValueError("user_input is required")
        if not story_bible:
            raise ValueError("story_bible is required")
        
        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "character_agent.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "character_agent.prompt")
        
        # story_bible을 JSON 문자열로 변환하여 프롬프트에 포함
        story_context = story_bible.model_dump_json(indent=2)
        user_prompt = user_prompt_template.format(
            story_context=story_context,
            user_input=user_input
        )
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(CharacterBible)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        character_bible = structured_llm.invoke(messages)
        
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
        
        if not user_input:
            raise ValueError("user_input is required")
        if not story_bible:
            raise ValueError("story_bible is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        
        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "visual_style_agent.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "visual_style_agent.prompt")
        
        # 이전 bible들을 JSON 문자열로 변환하여 프롬프트에 포함
        story_context = story_bible.model_dump_json(indent=2)
        character_context = character_bible.model_dump_json(indent=2)
        user_prompt = user_prompt_template.format(
            story_context=story_context,
            character_context=character_context,
            user_input=user_input
        )
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(StyleBible)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        style_bible = structured_llm.invoke(messages)
        
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
            style_bible=style_context
        )
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(DirectorOutput)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        director_output = structured_llm.invoke(messages)
        
        # 4. 유효성 검증
        if len(director_output.keyframes) != 5:
            raise ValueError(f"Expected 5 keyframes, got {len(director_output.keyframes)}")
        if len(director_output.shots) != 4:
            raise ValueError(f"Expected 4 shots, got {len(director_output.shots)}")
        
        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "director_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(director_output.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Director Plan saved to {output_path}")
        logger.info(f"Generated {len(director_output.keyframes)} keyframes and {len(director_output.shots)} shots")
        
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
        
        if not director_output:
            raise ValueError("director_output is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        if not style_bible:
            raise ValueError("style_bible is required")
        
        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "image_specialist.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "image_specialist.prompt")
        
        # keyframes 정보만 추출 (image specialist는 keyframes만 필요)
        keyframes_context = json.dumps(
            [kf.model_dump() for kf in director_output.keyframes],
            indent=2,
            ensure_ascii=False
        )
        character_context = character_bible.model_dump_json(indent=2)
        style_context = style_bible.model_dump_json(indent=2)
        
        user_prompt = user_prompt_template.format(
            keyframes_plan=keyframes_context,
            character_bible=character_context,
            style_bible=style_context
        )
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(ImageEngineerOutput)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        image_engineer_output = structured_llm.invoke(messages)
        
        # 4. 유효성 검증
        if len(image_engineer_output.keyframe_prompts) != 5:
            raise ValueError(f"Expected 5 keyframe prompts, got {len(image_engineer_output.keyframe_prompts)}")
        
        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "image_engineer_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(image_engineer_output.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Image Prompts saved to {output_path}")
        logger.info(f"Generated {len(image_engineer_output.character_reference_prompt)} reference prompts and {len(image_engineer_output.keyframe_prompts)} keyframe prompts")
        
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
        
        if not director_output:
            raise ValueError("director_output is required")
        if not character_bible:
            raise ValueError("character_bible is required")
        if not style_bible:
            raise ValueError("style_bible is required")
        
        # 2. 프롬프트 로드
        system_prompt = load_prompt(PROMPTS_DIR / "system" / "video_specialist.prompt")
        user_prompt_template = load_prompt(PROMPTS_DIR / "user" / "video_specialist.prompt")
        
        # shots 정보만 추출 (video specialist는 shots만 필요)
        shots_context = json.dumps(
            [shot.model_dump() for shot in director_output.shots],
            indent=2,
            ensure_ascii=False
        )
        character_context = character_bible.model_dump_json(indent=2)
        style_context = style_bible.model_dump_json(indent=2)
        
        user_prompt = user_prompt_template.format(
            shot_plan=shots_context,
            character_bible=character_context,
            style_bible=style_context
        )
        
        # 3. LLM 호출 (with_structured_output)
        llm = get_llm()
        structured_llm = llm.with_structured_output(VideoEngineerOutput)
        
        messages = [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
        
        video_engineer_output = structured_llm.invoke(messages)
        
        # 4. 유효성 검증
        if len(video_engineer_output.video_prompts) != 4:
            raise ValueError(f"Expected 4 video prompts, got {len(video_engineer_output.video_prompts)}")
        
        # 5. JSON 파일로 저장
        output_path = OUTPUT_DIR / "video_engineer_output.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(video_engineer_output.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Video Prompts saved to {output_path}")
        logger.info(f"Generated {len(video_engineer_output.video_prompts)} video prompts")
        
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
    reference_img_paths: Optional[Sequence[str | Path]] = None
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
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180000))
    
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
            logger.warning(f"Image generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise Exception(f"Image generation failed after {max_retries} attempts: {last_error}")


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
        
        model_name = "gemini-2.5-flash-image"
        
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
                response = _generate_image_with_retry(api_key, model_name, full_prompt, reference_img_paths=reference_img_paths)
                
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
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                image = part.as_image()
                                image.save(asset_path)
                                logger.info(f"Asset saved to {asset_path}")
                                assets[character_name] = str(asset_path)
                                break
                else:
                    logger.warning(f"No image data found in response for {character_name}")
                    
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
        
        model_name = "gemini-2.5-flash-image" # gemini-2.5-flash-image, gemini-3-pro-image-preview
        
        # 3. 각 키프레임 이미지 생성
        frames = []
        
        # 유효성 검증
        if len(image_engineer_output.keyframe_prompts) != 5:
            raise ValueError(f"Expected 5 keyframe prompts, got {len(image_engineer_output.keyframe_prompts)}")
        
        for idx, keyframe_prompt in enumerate(image_engineer_output.keyframe_prompts):
            
            keyframe_index = idx
            prompt = keyframe_prompt.prompt
            
            logger.info(f"Generating keyframe {keyframe_index}...")
            
            # 레퍼런스 이미지 정보를 프롬프트에 포함
            # (현재 Gemini API에서 레퍼런스 이미지를 프롬프트로 전달하는 방법 적용)
            reference_img_paths = []
            if assets:
                reference_img_paths = [item for item in assets.values() if item is not None]
                print(f"reference img paths: {reference_img_paths}")
            try:
                # API 호출 (재시도 로직 포함)
                if idx != 0:
                    # 이전 프레임 이미지를 레퍼런스로 추가
                    reference_img_paths = [FRAMES_DIR / f"keyframe_{idx-1}.png"] + reference_img_paths
                    
                if reference_img_paths:
                    prompt = f"Use the provided images as a strict reference.\n\n{prompt}"
                else:
                    prompt = prompt
                    
                response = _generate_image_with_retry(api_key, model_name, prompt, reference_img_paths=reference_img_paths)
                
                # 이미지 저장
                frame_path = FRAMES_DIR / f"keyframe_{idx}.png"
                
                # 응답에서 이미지 추출 및 저장
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                # PIL Image로 변환하여 저장
                                image = part.as_image()
                                image.save(frame_path)
                                logger.info(f"Keyframe saved to {frame_path}")
                                frames.append(frame_path)
                                break
                else:
                    logger.warning(f"No image data found in response for keyframe {keyframe_index}")
                    
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
    retry_delay: float = 10.0
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
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120000))
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
            
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No generated videos in response")
            
            # 비디오 다운로드 및 저장
            video = operation.response.generated_videos[0]
            if not video or not hasattr(video, 'video') or not video.video:
                raise Exception("Invalid video object in response")
            
            client.files.download(file=video.video)
            video.video.save(str(save_path))
            
            logger.info(f"Generated video saved to {save_path}")
            return True
            
        except Exception as e:
            last_error = e
            logger.warning(f"Video generation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error(f"Video generation failed after {max_retries} attempts: {last_error}")
    return False


def generate_videos_node(state: GraphState) -> dict:
    """
    Veo 3.1 API를 사용하여 4개의 비디오를 생성하는 노드
    
    Args:
        state: GraphState dict containing frames, video_engineer_output
        
    Returns:
        dict: {"videos": ["path/to/video_0.mp4", ...]}
    """
    try:
        logger.info("Starting Videos generation...")
        
        # 1. state에서 필요한 데이터 추출
        frames = state.get("frames")
        video_engineer_output = state.get("video_engineer_output")
        
        if not frames or len(frames) != 5:
            raise ValueError(f"Expected 5 frames, got {len(frames) if frames else 0}")
        if not video_engineer_output:
            raise ValueError("video_engineer_output is required")
        if len(video_engineer_output.video_prompts) != 4:
            raise ValueError(f"Expected 4 video prompts, got {len(video_engineer_output.video_prompts)}")
        
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
            
            logger.info(f"Generating video {i+1}/4 (shot {video_prompt_obj.shot_index})...")
            
            # 첫 번째 프레임 결정
            if i == 0:
                # 첫 번째 비디오: frames[0]을 그대로 사용
                first_frame_path = frames[0]
            else:
                # 두 번째 비디오 이후: 이전 비디오의 마지막 프레임 추출
                prev_video_path = videos[i-1]
                first_frame_path = temp_frames_dir / f"temp_frame_{i}.png"
                
                logger.info(f"Extracting last frame from previous video: {prev_video_path}")
                if not _extract_last_frame_opencv(prev_video_path, first_frame_path):
                    raise Exception(f"Failed to extract last frame from video {i-1}")
            
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
                retry_delay=10.0
            )
            
            if not success:
                raise Exception(f"Failed to generate video {i}")
            
            videos.append(str(video_path))
            logger.info(f"Video {i+1}/4 completed: {video_path}")
        
        if len(videos) != 4:
            raise ValueError(f"Expected 4 videos, but generated {len(videos)}")
        
        logger.info(f"Successfully generated {len(videos)} videos")
        
        # 4. 결과 반환
        return {"videos": videos}
        
    except Exception as e:
        logger.error(f"Error in generate_videos_node: {e}")
        return {"errors": [f"Videos generation failed: {str(e)}"]}


def _merge_videos_ffmpeg(video_paths: Sequence[str | Path], output_path: str | Path) -> bool:
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
        with open(file_list_path, 'w') as f:
            for path in video_paths:
                abs_path = os.path.abspath(path)
                f.write(f"file '{abs_path}'\n")
        
        logger.info(f"Created file list: {file_list_path}")
        
        # FFmpeg concat demuxer 실행
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(file_list_path),
            '-c', 'copy',  # 재인코딩 없이 스트림 복사
            str(output_path)
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
