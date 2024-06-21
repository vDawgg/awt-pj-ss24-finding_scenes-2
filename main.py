from utils.objects.metadata_object import MetaDataObject
from utils.video.video_extraction_with_pytube import YouTubeVideo
from utils.video.scenes import get_scenes
from utils.video.keyframe_extraction import process_all_videos_in_csv, create_keyframes_csv
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.model import CaptionModel
from utils.captioning.caption_keyframes import caption_images, caption_images_florence_2
from transformers import AutoProcessor, AutoModelForCausalLM

from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file
import sys
if __name__ == '__main__':
    input_string="https://www.youtube.com/watch?v=5C_HPTJg5ek"
    downloader = YouTubeVideo(input_string)

    path, subtitles = downloader.download_video_and_subtitles()
    print(path)
    scene_csv = get_scenes(path)

    process_all_videos_in_csv(scene_csv, "./videos/keyframes")
    create_keyframes_csv("./videos/keyframes", scene_csv)
    save_subtitle_in_csv(subtitles, "videos/keyframes/extracted_keyframes.csv")

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

    directory = "./videos/keyframes"
    tasks = {
      "CAPTION": "<MORE_DETAILED_CAPTION>",
      "OBJECTS": "<OD>",
      "TEXT": "<OCR>",
    }

    caption_images_florence_2(model=model, processor=processor, tasks=tasks, directory=directory)
    #  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #  model = LLMModel(model_id, load_in_4bit=True)
    #  align_video_captions(model, "./videos/keyframes/extracted_keyframes.csv",tasks)

    scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

    print(scene_objects_with_extraction_data)

    scene_objects_with_llm_data=get_metadata_from_keyframe_file( path_to_keyframes_csv="./videos/keyframes/extracted_keyframes.csv" ,scene_objects= scene_objects_with_extraction_data,tasks=tasks)

    metaDataObject=MetaDataObject(input_string, downloader.yt, scene_objects_with_llm_data)

    with open('./metadata.json', 'w') as outfile:
        outfile.write(metaDataObject.to_json())
        print(metaDataObject.to_json())