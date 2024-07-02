from utils.objects.metadata_object import MetaDataObject
from utils.video.video_extraction_with_pytube import YouTubeVideo
from utils.video.scenes import get_scenes
from utils.keyframe.keyframe_extraction import generate_keyframes_from_scenes
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.model import CaptionModel
from utils.captioning.caption_keyframes import caption_images_llava
from utils.llm.model import LLMModel
from utils.llm.align_captions import align_video_captions
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
import pandas as pd

from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file
import sys
if __name__ == '__main__':

    #  input_string = input("Enter  url of youtube video : ")
    #  print("You entered:", input_string)
    #  sys.stdout.flush()
    input_string="https://www.youtube.com/watch?v=5C_HPTJg5ek"
    downloader = YouTubeVideo(input_string)

    title, path, subtitles = downloader.download_video_and_subtitles()
    print(path)
    scene_csv = get_scenes(path)


    keyframe_csv, keyframe_dir = generate_keyframes_from_scenes(title)
    save_subtitle_in_csv(subtitles, keyframe_csv)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        use_flash_attention_2=True
    )

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    tasks = {
    "CAPTION": "Caption the scene and put it into the context of the video through the provided subtitles if possible. Describe it with as much information as possible. ",
    "LANGUAGE": "What is the language used in the video this keyframe was captured from? Only give the language as an answer.",
    "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc? Only answer with the type of the video.",
    "BACKGROUND": "What is the background of the scene Describe it in detail.",
    "OBJECTS": " Can you list all objects sperated by commas? Only give the list of objects as an answer.",
    }

    caption_images_llava(model=model, processor=processor, tasks=tasks, directory=keyframe_dir, csv=keyframe_csv, filename_column="Filename")
    #  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #  model = LLMModel(model_id, load_in_4bit=True)
    #  align_video_captions(model, "./videos/keyframes/extracted_keyframes.csv",tasks)

    scene_objects_with_extraction_data = get_metadata_from_scene_file(path_to_scene_csv=scene_csv)
    scene_objects_with_llm_data = get_metadata_from_keyframe_file(path_to_keyframes_csv=keyframe_csv ,scene_objects= scene_objects_with_extraction_data,tasks=tasks)
    metaDataObject=MetaDataObject(input_string, downloader.yt, scene_objects_with_llm_data)

    with open('./metadata.json', 'w') as outfile:
        outfile.write(metaDataObject.to_json())
