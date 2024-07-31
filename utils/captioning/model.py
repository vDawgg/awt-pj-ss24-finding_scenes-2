import torch
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

