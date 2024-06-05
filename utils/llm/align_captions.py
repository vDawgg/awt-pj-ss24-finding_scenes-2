import pandas as pd

from utils.llm.model import LLMModel


def align_caption(model: LLMModel, caption: str, context: str) -> str:
    prompt = ("In the following you will be given a caption and preceding captions from an educational video. The current caption might not yet fit the preceding captions that well yet so your task is to make sure that the caption follows the red line from the preceding captions. If the caption already fits nicely in the context you do not have to make any adjustments. If the caption does not fit the context well yet, give me the adjusted caption. Do not write anything else."
              "Caption:"
              f"{caption}"
              f"Context:"
              f"{context}"
              "Aligned Caption:\n")
    return model.run_inference(prompt)


def align_video_captions(model: LLMModel, csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    updated_caption_list = ['']
    caption_list = df["Caption"]
    for i, caption in enumerate(caption_list):
        updated_caption_list.append(
            # This uses the last caption in the list for now, ideally this should
            # be set as an HP further down the line
            model.tokenizer.decode(
                align_caption(model, caption, updated_caption_list[-1])[0]
            ).split("Aligned Caption:")[1]
        )
    df["Caption"] = updated_caption_list[1:]
