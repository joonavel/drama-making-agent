"""
LangGraph Nodes for Drama Making Workflow
"""

import json
from pathlib import Path
from typing import Dict, Any
from src.config import load_prompt, get_llm, OUTPUT_DIR, logger, PROMPTS_DIR
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
