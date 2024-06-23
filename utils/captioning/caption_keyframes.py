import os
from typing import List
from PIL import Image
import pandas as pd
from pathlib import Path

from utils.captioning.model import CaptionModel


def get_column_values(df: pd.DataFrame, column_name: str) -> List[str]:
    if column_name in df.columns:
        return df[column_name].values.tolist()
    else:
        return None


def caption_images(model: CaptionModel, base_prompt: str, tasks, csv_df: pd.DataFrame, filename_column: str = 'Filename', directory: str = "videos/keyframes") -> str:
    """
    Caption the images using the provided model and tasks.

    Args:
        model (CaptionModel): The captioning model to use for generating captions.
        base_prompt (str): The base prompt to use for generating captions.
        tasks (dict): A dictionary mapping task names to task descriptions.
        csv_file (str, optional): The name of the CSV file to read and write captions to. Defaults to "extracted_keyframes.csv".
        filename_column (str, optional): The name of the column in the CSV file that contains the filenames. Defaults to 'Filename'.
        directory (str, optional): The directory where the images are located. Defaults to "videos/keyframes".

    Returns:
        str: The filepath of the CSV file with the generated captions.
    """

    extend_to_full_path = lambda filename: Path(directory) / filename
    full_filepaths = csv_df[filename_column].apply(extend_to_full_path).tolist()

    subtitle_list = get_column_values(csv_df, 'Subtitle')

    for task, description in tasks.items():
        task_outputs = []
        for i, filepath in enumerate(full_filepaths):
            print("\nfilepath: ", filepath)
            image = Image.open(filepath)
            enc_image = model.encode_image(image)
            if subtitle_list is not None:
                subtitle = subtitle_list[i]
                prompt = f" {description}"
            else:
                prompt = f"{description}"
            task_outputs.append(model.run_inference(enc_image, prompt))
        csv_df[task] = task_outputs

    csv_df.to_csv(csv_filepath, index=False)

    return csv_df


if __name__ == "__main__":

    csv_filepath = "extracted_keyframes.csv"

    filename_column = 'Filename'
    directory = os.path.join("/content","keyframes")
    directory = "/home/limin/Documents/programming/finding_scenes_in_learning_videos/awt-pj-ss24-finding_scenes-2/videos/Rust in 100 Seconds_keyframes"
    csv_filepath = Path(directory) / csv_filepath
    csv_df = pd.read_csv(csv_filepath)

    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = CaptionModel(model_id, revision=revision)

    tasks = {
        "CAPTION": "Caption the scene. Describe it with as much information as possible.",
        "INFORMATION": "Generate detailed information for this scene for this scene.",
        "LANGUAGE": "What is the language used in the video this keyframe was captured from?",
        "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, and the likes.",
    }

    prompt = f"""
    Given keyframe extracted from a scene and the corresponding SUBTITLES - the subtitles transcribed for this scene.
    Generate detailed information for this scene for TASK - instructions on what exactly to capture.
    Use both the image and the SUBTITLES to infer the information.
    If SUBTITLES is not provided, infer the information only from the keyframe image.
    If the TASK cannot be completed, then return "NONE".
    """.strip()

    csv_df = caption_images(
        model=model,
        base_prompt=prompt,
        tasks=tasks,
        directory=directory,
        csv_df=csv_df
    )

    csv_df.to_csv(csv_filepath, index=False)
