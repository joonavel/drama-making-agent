"""
LangGraph Nodes for Drama Making Workflow
"""

import json
from pathlib import Path
from typing import Dict, Any
from src.config import load_prompt, get_llm, OUTPUT_DIR, logger, PROMPTS_DIR
from src.workflows.tasks import StoryBible, CharacterBible, StyleBible, GraphState


def generate_story_bible_node(state: GraphState) -> GraphState:
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


def generate_character_bible_node(state: GraphState) -> GraphState:
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


def generate_style_bible_node(state: GraphState) -> GraphState:
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
