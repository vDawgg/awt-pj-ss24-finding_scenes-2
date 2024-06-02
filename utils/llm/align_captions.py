import pandas as pd

from utils.llm.model import LLMModel


def align_caption(model: LLMModel, caption: str, context: str) -> str:
    prompt = ("In the following you will be given a caption and preceding captions from an educational video. The current caption might not yet fit the preceding captions that well yet so your task is to make sure that the caption follows the red line from the preceding captions. If the caption already fits nicely in the context you do not have to make any adjustments. If the caption does not fit the context well yet, give me the adjusted caption. Do not write anything else"
              "Caption:"
              f"{caption}"
              f"Context:"
              f"{context}")
    return model.generate(prompt)


def align_video_captions(model: LLMModel, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    caption_list = df["Caption"]
    for i, caption in enumerate(caption_list):
        caption_list[i] = align_caption(model, caption, caption_list[:i])
        print(caption_list[i])
    df["Caption"] = caption_list
