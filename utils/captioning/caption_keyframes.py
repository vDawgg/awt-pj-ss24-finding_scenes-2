import os
from typing import List, Any
from PIL import Image
import pandas as pd
from transformers import PreTrainedTokenizerFast

from utils.captioning.model import CaptionModel


def get_column_values(csv_file: str, column_name: str) -> List[str]:
    df = pd.read_csv(csv_file)
    if column_name in df.columns:
        return df[column_name].values.tolist()
    else:
        return None


def get_filepaths_from_csv(csv_file: str, filename_column: str, directory: str = "videos/keyframes") -> List[str]:
    csv_filepath = os.path.join(directory, csv_file)
    filenames = get_column_values(csv_filepath, filename_column)
    full_filepaths = [f"{os.path.join(directory, ''.join(filename.split('_')[:-1]), filename)}" for filename in
                      filenames]
    return full_filepaths


def caption_images(model: CaptionModel, base_prompt: str, tasks, csv_file: str = "extracted_keyframes.csv",
                   filename_column: str = 'Filename', directory: str = "videos/keyframes") -> str:
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
    df = pd.read_csv(csv_filepath)

    for task, description in tasks.items():
        task_outputs = []
        for i, filepath in enumerate(full_filepaths):
            image = Image.open(filepath)
            enc_image = model.encode_image(image)
            if subtitle_list is not None:
                subtitle = subtitle_list[i]
                prompt = f" {description}"
            
            else:
                prompt = f"{description}"
            task_outputs.append(model.run_inference(enc_image, prompt))
        df[task] = task_outputs

    df.to_csv(csv_filepath, index=False)

    return csv_filepath


def caption_images_idefics_2(model: Any, processor: PreTrainedTokenizerFast, tasks, csv_file: str = "extracted_keyframes.csv",
                   filename_column: str = 'Filename', directory: str = "videos/keyframes") -> str:
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
    df = pd.read_csv(csv_filepath)

    for task, description in tasks.items():
        task_outputs = []
        for i, filepath in enumerate(full_filepaths):
            image = Image.open(filepath)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{description}"},
                    ]
                },
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], padding=True,
                               return_tensors="pt")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            task_outputs.append(
                processor.batch_decode(
                    model.generate(**inputs, max_new_tokens=600),
                    skip_special_tokens=True
                )[0].split("Assistant: ")[-1]
            )
            """else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"{description}"},
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text",
                             "text": f"{task_outputs[-1]}"},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Given the context of the preceding scene, please complete the following task: {description}"},
                        ]
                    },
                ]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[Image.open(full_filepaths[i-1]), image], padding=True, return_tensors="pt")
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                task_outputs.append(
                    processor.batch_decode(
                        model.generate(**inputs, max_new_tokens=600),
                        skip_special_tokens=True
                    )[0].split("Assistant: ")[-1]
                )"""
        df[task] = task_outputs
    df.to_csv(csv_filepath, index=False)

    return csv_filepath


if __name__ == "__main__":

    csv_filepath = "extracted_keyframes.csv"

    filename_column = 'Filename'
    directory = os.path.join("/content","keyframes")

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

    csv_filepath = caption_images(
        model=model,
        base_prompt=prompt,
        tasks=tasks,
        directory=directory,
        csv_file=csv_filepath
    )
