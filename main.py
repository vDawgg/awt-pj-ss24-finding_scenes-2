from utils.objects.metadata_object import MetaDataObject
from utils.video.video_extraction_with_pytube import YouTubeVideo
from utils.video.scenes import get_scenes
from utils.video.keyframe_extraction import process_all_videos_in_csv, create_keyframes_csv
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.model import CaptionModel
from utils.captioning.caption_keyframes import caption_images
from utils.llm.model import LLMModel
from utils.llm.align_captions import align_video_captions

from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file
if __name__ == '__main__':

 downloader = YouTubeVideo("https://www.youtube.com/watch?v=04gH4fONZ7s")

 path, subtitles = downloader.download_video_and_subtitles()
 print(path)
 scene_csv = get_scenes(path)

#  process_all_videos_in_csv(scene_csv, "./videos/keyframes")
#  create_keyframes_csv("./videos/keyframes", scene_csv) 
#  save_subtitle_in_csv(subtitles, "videos/keyframes/extracted_keyframes.csv")
#  model_id = "vikhyatk/moondream2"
#  revision = "2024-05-20"
#  model = CaptionModel(model_id, revision=revision)
#  directory = "./videos/keyframes"
#  tasks = {
#     "CAPTION": "Caption the scene. Describe it with as much information as possible. ",
#     "INFORMATION": "Generate detailed information for this scene.",
#     "LANGUAGE": "What is the language used in the video this keyframe was captured from?",
#     "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc?",
#     "OBJECTS": "What objects are present in the scene? List as many as possible as an array of objects. Name just the objects, not the actions. And spereate them with commas.",
#  }

#  prompt = f"""
#  Given keyframe extracted from a scene and the corresponding SUBTITLES - the subtitles transcribed for this scene.
#  Generate detailed information for this scene for TASK - instructions on what exactly to capture.
#  Use both the image and the subtitles to infer the information.
#  If the TASK cannot be completed, then return "NONE".
#  """.strip()

#  caption_images(model=model, base_prompt=prompt, tasks=tasks, directory=directory)
#  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#  model = LLMModel(model_id, load_in_4bit=True)
#  align_video_captions(model, "./videos/keyframes/extracted_keyframes.csv",tasks)

 
 scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

 scene_objects_with_llm_data=get_metadata_from_keyframe_file( path_to_keyframes_csv="./videos/keyframes/extracted_keyframes.csv" ,scene_objects= scene_objects_with_extraction_data)

 metaDataObject=MetaDataObject("https://www.youtube.com/watch?v=5C_HPTJg5ek", downloader.yt, scene_objects_with_llm_data) 

 print(metaDataObject.to_json())