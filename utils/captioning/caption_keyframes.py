import os
from typing import List
from PIL import Image
import pandas as pd

from utils.captioning.model import CaptionModel


def get_column_values(csv_file: str, column_name: str) -> List[str]:
    df = pd.read_csv(csv_file)
    return df[column_name].values.tolist()


def get_filepaths_from_csv(csv_file: str, filename_column: str, directory: str = "videos/keyframes") -> List[str]:
    csv_filepath = os.path.join(directory, csv_file)
    filenames = get_column_values(csv_filepath, filename_column)
    full_filepaths = [f"{os.path.join(directory,''.join(filename.split('_')[:-1]),filename)}" for filename in filenames]
    return full_filepaths


def caption_images(model: CaptionModel, base_prompt: str, tasks, csv_file: str = "extracted_keyframes.csv", filename_column: str = 'Filename', directory: str = "videos/keyframes") -> str:
    """
    Caption the images using the provided model and prompt.

    Args:
        model (object): The image captioning model.
        prompt (str): The prompt to use for generating captions.
        csv_file (str, optional): The name of the CSV file to store the captions. Defaults to "extracted_keyframes.csv".
        filename_column (str, optional): The name of the column in the CSV file that contains the filenames. Defaults to 'Filename'.
        directory (str, optional): The directory where the keyframe images are located. Defaults to "videos/keyframes".

    Returns:
        None
    """

    csv_filepath = os.path.join(directory, csv_file)
    full_filepaths = get_filepaths_from_csv(csv_file, filename_column, directory)
    subtitle_list = get_column_values(csv_filepath, 'Subtitle')
    caption_list = []

    for filepath, subtitle in zip(full_filepaths, subtitle_list):
        image = Image.open(filepath)
        enc_image = model.encode_image(image)
        generated_captions = []
        for task in tasks:
            prompt = f"{base_prompt} \nSUBTITLES: {subtitle}\nTASK: {task}"
            prompt_response = model.run_inference(enc_image, prompt)
            generated_captions.append(prompt_response)
        caption_list.append(generated_captions)

    df = pd.read_csv(csv_filepath)
    df["Caption"] = caption_list
    df.to_csv(csv_filepath, index=True)

    return csv_filepath


if __name__ == "__main__":

    csv_filepath = "extracted_keyframes.csv"

    filename_column = 'Filename'
    directory = os.path.join("/content","keyframes")

    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = CaptionModel(model_id, revision)

    task_list = [
        "Generate detailed information for this scene for TASK - instructions on what exactly to capture.",
        "What is the language of the scene",
        "What kind of scene is this",
    ]

    prompt = f"""
    Given keyframe extracted from a scene and the corresponding SUBTITLES - the subtitles transcribed for this scene.
    Generate detailed information for this scene for TASK - instructions on what exactly to capture.
    Use both the image and the subtitles to infer the information.
    If the TASK cannot be completed, then return "NONE".
    """.strip()

    caption_images(
        model=model,
        base_prompt=prompt,
        tasks=task_list,
        directory=directory
    )

