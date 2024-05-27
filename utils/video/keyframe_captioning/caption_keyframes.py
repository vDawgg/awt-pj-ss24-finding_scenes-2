import os
from typing import List

from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.video.keyframe_captioning.model import Model



def get_column_values(csv_file: str, column_name: str) -> List[str]:
    df = pd.read_csv(csv_file)
    return df[column_name].values.tolist()


def get_filepaths_from_csv(csv_file: str, filename_column: str, directory: str = "videos/keyframes") -> List[str]:
    csv_filepath = os.path.join(directory, csv_file)
    filenames = get_column_values(csv_filepath, filename_column)
    full_filepaths = [f"{os.path.join(directory,''.join(filename.split('_')[:-1]),filename)}" for filename in filenames]
    return full_filepaths


def caption_images(model: Model, prompt: str, csv_file: str = "extracted_keyframes.csv", filename_column: str = 'Filename', directory: str = "videos/keyframes") -> None:
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

    full_filepaths = get_filepaths_from_csv(csv_file, filename_column, directory)

    caption_list = []

    for filepath in full_filepaths:
        image = Image.open(filepath)
        enc_image = model.encode_image(image)
        prompt_response = model.run_inference(enc_image, prompt)
        caption_list.append(prompt_response)

    csv_filepath = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_filepath)
    df["Caption"] = caption_list
    df.to_csv(csv_filepath, index=True)


if __name__ == "__main__":

    csv_filepath = "extracted_keyframes.csv"

    filename_column = 'Filename'
    directory = os.path.join("/content","keyframes")

    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model = Model(model_id, revision)
    prompt = "Describe the image with as much detail as possible. Generate as much information that can be turned into meta data as possible"

    caption_images(
        model=model,
        prompt=prompt,
        directory=directory
    )
