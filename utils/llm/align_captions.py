import pandas as pd

from utils.llm.model import LLMModel


def align_caption(model: LLMModel, caption: str, context: str) -> str:
    prompt = [
        {
            "role": "system",
            "content": "You are a bot whose job it is to make sure that imprecise captions for different parts of a video are well aligned with each other. To this end a user will supply you with a caption and the context for this caption, which are the preceding captions. You should then answer only with the caption, which is edited to properly fit the context. Remember that these captions should describe a video so remove any mentions of image descriptions, etc.",
        },
        {
            "role": "user",
            "content": f"Caption: '{caption}'; Context: {context}"
        }
    ]
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
                align_caption(model, caption, caption_list[:i])[0]
            ).split("<|assistant|>")[1]
        )
    df["Caption"] = updated_caption_list[1:]
    df.to_csv(csv_path)
