from utils.model.model import Model


class LLMModel(Model):
    """
    A class representing an LLM for caption alignment and metadata generation.

    Attributes:
        model_id (str): The ID of the model.
        revision (str): The revision of the model.
        model (AutoModelForCausalLM): The pretrained model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompt (str): The prompt for aligning captions or generating metadata.

    Methods:
        run_inference: General purpose function, that tokenizes a given prompt and returns the generation of the model
    """

    def run_inference(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        General purpose function, that tokenizes a given prompt and returns the generation of the model

        :param prompt: The prompt for the model.
        :param max_new_tokens: The maximum number of tokens the model can generate (Needed to pass as some models are
        quite restricted by default).
        :return: the models output.
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        return self.model.generate(**tokenized_prompt, max_new_tokens=max_new_tokens)


if __name__ == '__main__':
    model = Model("mistralai/Mistral-7B-Instruct-v0.3")
