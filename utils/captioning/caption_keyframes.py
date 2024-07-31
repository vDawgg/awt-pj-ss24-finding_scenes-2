import os
import pandas as pd
from PIL import Image
from typing import List, Any
from transformers import PreTrainedTokenizerFast


def get_column_values(csv_file: str, column_name: str) -> List[str]:
    """
    Get the values of a column from a CSV file.

    :param str csv_file: The path to the CSV file.
    :param str column_name: The name of the column to get the values from.
        
    :rtype: List[str]
    :return: A list of values from the specified column."""
    df = pd.read_csv(csv_file)
    if column_name in df.columns:
        return df[column_name].values.tolist()
    else:
        return None


def get_filepaths_from_csv(csv_file: str, filename_column: str, directory: str) -> List[str]:
    """ Get the full filepaths of the images from the CSV file.
    :param str csv_file: The path to the CSV file.
    :param str filename_column: The name of the column containing the filenames.
    :param str directory: The directory where the images are located.

    :rtype: List[str]
    :return: A list of full filepaths of the images.
    """

    csv_filepath = os.path.join(directory, csv_file)
    filenames = get_column_values(csv_filepath, filename_column)
    full_filepaths = [f"{os.path.join(directory, filename)}" for filename in
                      filenames]
    return full_filepaths


def caption_images_idefics_2(model: Any, processor: PreTrainedTokenizerFast, tasks, csv_file: str,
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
        str: The path to the CSV file containing the captions.
    """

    csv_filepath = os.path.join(directory, csv_file)
    full_filepaths = get_filepaths_from_csv(csv_file, filename_column, directory)
    subtitle_list = get_column_values(csv_filepath, 'Subtitle')
    df = pd.read_csv(csv_filepath)

    for task, description in tasks.items():
        task_outputs = []
        for _, filepath in enumerate(full_filepaths):
            image = Image.open(filepath)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ""},
                    ]
                },
            ]
            if task == "CAPTION" or task == "KEY-CONCEPTS" or task == "QUESTIONS" or task == "RESOURCES":
                messages[0]["content"][1]["text"] = f"{description}. Use the following list of subtitles taken from the video the image is from as additional context: {subtitle_list}"
            else:
                messages[0]["content"][1]["text"] = f"{description}"
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
        df[task] = task_outputs
    df.to_csv(csv_filepath, index=False)

    return csv_filepath

