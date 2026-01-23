from langgraph.graph import StateGraph, START, END
from src.workflows.nodes import (
    generate_story_bible_node,
    generate_character_bible_node,
    generate_style_bible_node,
    generate_director_plan_node,
    generate_image_prompts_node,
    generate_video_prompts_node,
    generate_assets_node,
    generate_frames_node,
    generate_videos_node,
    postprocess_node,
)
from src.workflows.tasks import GraphState


def create_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("story_bible", generate_story_bible_node)
    workflow.add_node("character_bible", generate_character_bible_node)
    workflow.add_node("style_bible", generate_style_bible_node)
    workflow.add_node("director_plan", generate_director_plan_node)
    workflow.add_node("image_prompts", generate_image_prompts_node)
    workflow.add_node("video_prompts", generate_video_prompts_node)
    workflow.add_node("assets", generate_assets_node)
    workflow.add_node("frames", generate_frames_node)
    workflow.add_node("videos", generate_videos_node)
    workflow.add_node("postprocess", postprocess_node)

    workflow.add_edge(START, "story_bible")
    workflow.add_edge("story_bible", "character_bible")
    workflow.add_edge("character_bible", "style_bible")
    workflow.add_edge("style_bible", "director_plan")
    workflow.add_edge("director_plan", "image_prompts")
    workflow.add_edge("image_prompts", "video_prompts")
    workflow.add_edge("video_prompts", "assets")
    workflow.add_edge("assets", "frames")
    workflow.add_edge("frames", "videos")
    workflow.add_edge("videos", "postprocess")
    workflow.add_edge("postprocess", END)

    return workflow.compile()


def create_assets_2_end():
    workflow = StateGraph(GraphState)

    workflow.add_node("assets", generate_assets_node)
    workflow.add_node("frames", generate_frames_node)
    workflow.add_node("videos", generate_videos_node)
    workflow.add_node("postprocess", postprocess_node)

    workflow.add_edge(START, "assets")
    workflow.add_edge("assets", "frames")
    workflow.add_edge("frames", "videos")
    workflow.add_edge("videos", "postprocess")
    workflow.add_edge("postprocess", END)

    return workflow.compile()


def create_videos_2_end():
    workflow = StateGraph(GraphState)

    workflow.add_node("videos", generate_videos_node)
    workflow.add_node("postprocess", postprocess_node)

    workflow.add_edge(START, "videos")
    workflow.add_edge("videos", "postprocess")
    workflow.add_edge("postprocess", END)

    return workflow.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.config import PROJECT_ROOT

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

    workflow = create_workflow()
    user_input = "A young woman discovers a secret about her father's past that leads to a dangerous journey."
    response = workflow.invoke(GraphState(user_input=user_input))

    # workflow = create_videos_2_end()
    # import json
    # with open("local_storage/image_engineer_output.json", "r") as f:
    #     image_engineer_output = json.load(f)
    # with open("local_storage/video_engineer_output.json", "r") as f:
    #     video_engineer_output = json.load(f)
    # with open("local_storage/story_bible.json", "r") as f:
    #     story_bible = json.load(f)
    # with open("local_storage/character_bible.json", "r") as f:
    #     character_bible = json.load(f)
    # with open("local_storage/style_bible.json", "r") as f:
    #     style_bible = json.load(f)
    # with open("local_storage/director_output.json", "r") as f:
    #     director_output = json.load(f)
    # frames = [FRAMES_DIR / f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")]
    # # print(f"frames: {frames}")
    # response = workflow.invoke(GraphState(
    #     image_engineer_output=ImageEngineerOutput(**image_engineer_output),
    #     video_engineer_output=VideoEngineerOutput(**video_engineer_output),
    #     story_bible=StoryBible(**story_bible),
    #     character_bible=CharacterBible(**character_bible),
    #     style_bible=StyleBible(**style_bible),
    #     director_output=DirectorOutput(**director_output),
    #     frames=frames,
    # ))
