import os
from typing import List, Any
from PIL import Image
import pandas as pd
from pathlib import Path

from pandas import DataFrame
from transformers import AutoTokenizer

from utils.constants import VIDEO_DIR
from utils.captioning.model import CaptionModel


def get_column_values(df: pd.DataFrame, column_name: str) -> List[str]:
    if column_name in df.columns:
        return df[column_name].values.tolist()
    else:
        return None


def caption_images(
    model: CaptionModel, 
    base_prompt: str, 
    tasks, csv_df: pd.DataFrame,
    filename_column: str, 
    directory: str
) -> str:
    """
    Caption keyframe images using a given model and tasks.

    Args:
        model (CaptionModel): The captioning model to use for inference.
        base_prompt (str): The base prompt to use for captioning.
        tasks (dict): A dictionary of tasks to perform, where the keys are the task names and the values are descriptions.
        csv_df (pd.DataFrame): The DataFrame containing the image filenames and subtitles.
        filename_column (str): The name of the column in csv_df that contains the image filenames.
        directory (str): The directory where the image files are located.

    Returns:
        pd.DataFrame: The updated DataFrame with the captioning results.

    """

    print("Captioning keyframe images...")

    def extend_to_full_path(filename):
        return Path(directory) / filename
    
    full_filepaths = csv_df[filename_column].apply(extend_to_full_path).tolist()

    subtitle_list = get_column_values(csv_df, 'Subtitle')

    for task, description in tasks.items():
        print(f"\nTask: {task}")
        task_outputs = []
        for i, filepath in enumerate(full_filepaths):
            print("\nProcessing: ", filepath)
            image = Image.open(filepath)
            enc_image = model.encode_image(image)
            if subtitle_list is not None:
                subtitle = subtitle_list[i]
                prompt = f" {description}"
            else:
                prompt = f"{description}"
            task_outputs.append(model.run_inference(enc_image, prompt))
        csv_df[task] = task_outputs

    print("Captioning complete!")

    return csv_df


def caption_images_llava(
        model: Any,
        processor: Any,
        tasks,
        csv: str,
        filename_column: str,
        directory: str
) -> DataFrame:

    print("Captioning keyframe images...")

    csv_df = pd.read_csv(csv)

    def extend_to_full_path(filename):
        return Path(directory) / filename

    full_filepaths = csv_df[filename_column].apply(extend_to_full_path).tolist()

    subtitle_list = get_column_values(csv_df, 'Subtitle')

    for task, description in tasks.items():
        print(f"\nTask: {task}")
        task_outputs = []
        for i, filepath in enumerate(full_filepaths):
            print("\nProcessing: ", filepath)

            image = Image.open(filepath)
            prompt = f"[INST] <image>\n{description}[/INST]"
            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

            output = model.generate(**inputs, max_new_tokens=200)

            task_outputs.append(
                processor.decode(output[0], skip_special_tokens=True)
            )

        csv_df[task] = task_outputs

    print("Captioning complete!")
    csv_df.to_csv(csv, index=False)



if __name__ == "__main__":

    video_title = "Rust in 100 Seconds"
    csv_filepath = f"{video_title}_keyframes.csv"

    filename_column = 'Filename'
    directory = Path(VIDEO_DIR) / f"{video_title}_keyframes"

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
        filename_column=filename_column,
        csv_df=csv_df
    )

    csv_df.to_csv(csv_filepath, index=False)

