import os
from typing import List

from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.captioning.model import Model


def get_column_values(csv_file: str, column_name: str) -> List[str]:
    df = pd.read_csv(csv_file)
    if column_name in df.columns:
        return df[column_name].values.tolist()
    else:
        return None


def get_filepaths_from_csv(csv_file: str, filename_column: str, directory: str = "videos/keyframes") -> List[str]:
    csv_filepath = os.path.join(directory, csv_file)
    filenames = get_column_values(csv_filepath, filename_column)
    full_filepaths = [f"{os.path.join(directory,''.join(filename.split('_')[:-1]),filename)}" for filename in filenames]
    return full_filepaths


def caption_images(model: Model, base_prompt: str, tasks, csv_file: str = "extracted_keyframes.csv", filename_column: str = 'Filename', directory: str = "videos/keyframes") -> None:
    full_filepaths = get_filepaths_from_csv(csv_file, filename_column, directory)
    subtitle_list = get_column_values(csv_file, 'SubtitleTEST')
    caption_list = []

    for i, filepath in enumerate(full_filepaths):
        image = Image.open(filepath)
        enc_image = model.encode_image(image)
        task_responses = {}
        for task in tasks:
            # Use subtitles if available, otherwise don't include them in the prompt
            if subtitle_list is not None:
                subtitle = subtitle_list[i]
                prompt = f"{base_prompt} \nSUBTITLES: {subtitle}\nTASK: {task}"
            else:
                prompt = f"{base_prompt} \nTASK: {task}"
            prompt_response = model.run_inference(enc_image, prompt)
            task_responses[task] = prompt_response
        caption_list.append(task_responses)

    csv_filepath = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_filepath)
    df["Caption"] = caption_list
    df.to_csv(csv_filepath, index=False)

    return csv_filepath


if __name__ == "__main__":

    csv_filepath = "extracted_keyframes.csv"

    filename_column = 'Filename'
    directory = os.path.join("/content","keyframes")

    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = Model(model_id, revision)

    task_list = [
        "Caption the scene. Describe it with as much information as possible."
        "Generate detailed information for this scene for this scene.",
        "What is the language used in the video this keyframe was captured from?",
        "What kind of video is this, is it a tutorial, a lecture, and the likes.",
    ]

    prompt = f"""
    Given keyframe extracted from a scene and the corresponding SUBTITLES - the subtitles transcribed for this scene.
    Generate detailed information for this scene for TASK - instructions on what exactly to capture.
    Use both the image and the SUBTITLES to infer the information.
    If SUBTITLES is not provided, infer the information only from the keyframe image.
    If the TASK cannot be completed, then return "NONE".
    """.strip()




    csv_filepath = caption_images(
        model=model,
        base_prompt=prompt,
        tasks=task_list,
        directory=directory,
        csv_file=csv_filepath
    )
