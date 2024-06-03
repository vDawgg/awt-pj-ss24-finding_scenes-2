from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model.model import Model


class CaptionModel(Model):
    """
    A class representing a model for keyframe captioning.

    Attributes:
        model_id (str): The ID of the model.
        revision (str): The revision of the model.
        model (AutoModelForCausalLM): The pretrained model for caption generation.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The prompt for generating captions.

    Methods:
        encode_image(image) -> torch.Tensor: Encodes the input image.
        inference(enc_image: torch.Tensor, prompt: str) -> str: Performs inference to generate a caption.
    """

    def encode_image(self, image) -> torch.Tensor:
        """
        Encodes the input image.

        Args:
            image: The input image.

        Returns:
            enc_image: The encoded image.
        """
        return self.model.encode_image(image)

    def run_inference(self, enc_image: torch.Tensor, prompt: str) -> str:
        """
        Performs inference to generate a caption.

        Args:
            enc_image: The encoded image.
            prompt (str): The prompt for generating captions.

        Returns:
            prompt_response: The generated caption.
        """
        return self.model.answer_question(enc_image, prompt, self.tokenizer)


if __name__ == "__main__":
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    model = CaptionModel(model_id, revision=revision)

    enc_image = model.encode_image(image)

    prompt_response = model.run_inference(enc_image, model.prompt)

    print(prompt_response)
