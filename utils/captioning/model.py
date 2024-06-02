from typing import Tuple
import os
from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    """
    A class representing a model for keyframe captioning.

    Attributes:
        model_id (str): The ID of the model.
        revision (str): The revision of the model.
        model (AutoModelForCausalLM): The pretrained model for caption generation.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The prompt for generating captions.

    Methods:
        __init__(model_id: str, revision: str): Initializes the Model object.
        init_model(model_id: str, revision: str, cache_dir: str = "./model_cache") -> Tuple[AutoModelForCausalLM, AutoTokenizer]: Initializes the model and tokenizer.
        encode_image(image) -> torch.Tensor: Encodes the input image.
        inference(enc_image: torch.Tensor, prompt: str) -> str: Performs inference to generate a caption.
    """

    def __init__(self, model_id: str, revision: str):
        """
        Initializes a Model object.

        Args:
            model_id (str): The ID of the model.
            revision (str): The revision of the model.
        """
        self.model_id = model_id
        self.revision = revision
        self.model, self.tokenizer = self.init_model(model_id, revision)
        self.prompt = "Describe the image with as much detail as possible. Generate as much information that can be turned into meta data as possible."

    def init_model(self, model_id: str, revision: str, cache_dir: str = "./model_cache") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Initializes the model and tokenizer.

        Args:
            model_id (str): The ID of the model.
            revision (str): The revision of the model.
            cache_dir (str, optional): The directory to cache the model. Defaults to "./model_cache".

        Returns:
            model (AutoModelForCausalLM): The pretrained model for caption generation.
            tokenizer (AutoTokenizer): The tokenizer for the model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_type = torch.float16 if device == "cuda" else torch.float32

        model_dir = os.path.join(cache_dir, model_id.replace('/', '_'), revision)

        # Create the directory if it does not exist
        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
            # Model is not saved locally, download and save it
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision,
                torch_dtype=torch_type
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        else:
            # Load the model and tokenizer from the local disk
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch_type
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

        model = model.to(device)
        return model, tokenizer

    def encode_image(self, image) -> torch.Tensor:
        """
        Encodes the input image.

        Args:
            image: The input image.

        Returns:
            enc_image: The encoded image.
        """
        enc_image = self.model.encode_image(image)
        return enc_image

    def run_inference(self, enc_image: torch.Tensor, prompt: str) -> str:
        """
        Performs inference to generate a caption.

        Args:
            enc_image: The encoded image.
            prompt (str): The prompt for generating captions.

        Returns:
            prompt_response: The generated caption.
        """
        prompt_response = self.model.answer_question(enc_image, prompt, self.tokenizer)
        return prompt_response


if __name__ == "__main__":
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    model = Model(model_id, revision)

    enc_image = model.encode_image(image)

    prompt_response = model.inference(enc_image, model.prompt)

    print(prompt_response)
