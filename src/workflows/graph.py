from langgraph.graph import StateGraph, START, END
from src.workflows.nodes import *
from src.workflows.tasks import GraphState

def create_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("story_bible", generate_story_bible_node)
    workflow.add_node("character_bible", generate_character_bible_node)
    workflow.add_node("style_bible", generate_style_bible_node)
    workflow.add_node("director_plan", generate_director_plan_node)
    workflow.add_node("image_prompts", generate_image_prompts_node)
    workflow.add_node("video_prompts", generate_video_prompts_node)
    # workflow.add_node("assets", generate_assets_node)
    # workflow.add_node("frames", generate_frames_node)
    # workflow.add_node("videos", generate_videos_node)
    # workflow.add_node("final_video", generate_final_video_node)
    
    workflow.add_edge(START, "story_bible")
    workflow.add_edge("story_bible", "character_bible")
    workflow.add_edge("character_bible", "style_bible")
    workflow.add_edge("style_bible", "director_plan")
    workflow.add_edge("director_plan", "image_prompts")
    workflow.add_edge("image_prompts", "video_prompts")
    # workflow.add_edge("video_prompts", "assets")
    workflow.add_edge("video_prompts", END)

    return workflow.compile()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.config import PROJECT_ROOT
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

    workflow = create_workflow()
    user_input = "A young woman discovers a secret about her father's past that leads to a dangerous journey."
    response = workflow.invoke(GraphState(user_input=user_input))