import argparse
import os
from typing import cast
from dotenv import load_dotenv
from src.config import PROJECT_ROOT, FRAMES_DIR
from src.workflows.graph import (
    create_workflow,
    create_assets_2_end,
    create_videos_2_end,
)
from src.workflows.tasks import (
    GraphState,
    ImageEngineerOutput,
    VideoEngineerOutput,
    StoryBible,
    CharacterBible,
    StyleBible,
    DirectorOutput,
)


def main():
    parser = argparse.ArgumentParser(description="Drama Making Agent")
    parser.add_argument(
        "--input", type=str, required=True, help="User input for story generation"
    )
    parser.add_argument(
        "--mode", type=str, required=True, help="Mode: full, assets, videos"
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

    if args.mode == "full":
        workflow = create_workflow()
        response = workflow.invoke(cast(GraphState, {"user_input": args.input}))
    else:
        import json

        with open("local_storage/image_engineer_output.json", "r") as f:
            image_engineer_output = json.load(f)
        with open("local_storage/video_engineer_output.json", "r") as f:
            video_engineer_output = json.load(f)
        with open("local_storage/story_bible.json", "r") as f:
            story_bible = json.load(f)
        with open("local_storage/character_bible.json", "r") as f:
            character_bible = json.load(f)
        with open("local_storage/style_bible.json", "r") as f:
            style_bible = json.load(f)
        with open("local_storage/director_output.json", "r") as f:
            director_output = json.load(f)

        if args.mode == "assets":
            workflow = create_assets_2_end()
            response = workflow.invoke(
                cast(
                    GraphState,
                    {
                        "image_engineer_output": ImageEngineerOutput(
                            **image_engineer_output
                        ),
                        "video_engineer_output": VideoEngineerOutput(
                            **video_engineer_output
                        ),
                        "story_bible": StoryBible(**story_bible),
                        "character_bible": CharacterBible(**character_bible),
                        "style_bible": StyleBible(**style_bible),
                        "director_output": DirectorOutput(**director_output),
                    },
                )
            )
        elif args.mode == "videos":
            workflow = create_videos_2_end()
            frames = [
                FRAMES_DIR / f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")
            ]
            response = workflow.invoke(
                cast(
                    GraphState,
                    {
                        "image_engineer_output": ImageEngineerOutput(
                            **image_engineer_output
                        ),
                        "video_engineer_output": VideoEngineerOutput(
                            **video_engineer_output
                        ),
                        "story_bible": StoryBible(**story_bible),
                        "character_bible": CharacterBible(**character_bible),
                        "style_bible": StyleBible(**style_bible),
                        "director_output": DirectorOutput(**director_output),
                        "frames": frames,
                    },
                )
            )
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
