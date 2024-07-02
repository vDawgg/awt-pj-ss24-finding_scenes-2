import torch

from utils.llm.mistral_helper import create_scene_caption
from utils.objects.metadata_object import MetaDataObject
from utils.video.video_extraction_with_pytube import YouTubeVideo
from utils.video.scenes import get_scenes
from utils.video.keyframe_extraction import process_all_videos_in_csv, create_keyframes_csv
from utils.video.subtitles import save_subtitle_in_csv
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from utils.captioning.caption_keyframes import caption_images_idefics_2
from utils.llm.model import LLMModel
from utils.llm.align_captions import align_video_captions
import gc

from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file, set_new_content_for_metadata_attribute_for_sceneobjects

if __name__ == '__main__':
    input_string="https://www.youtube.com/watch?v=5C_HPTJg5ek"
    downloader = YouTubeVideo(input_string)

    path, subtitles = downloader.download_video_and_subtitles()
    print(path)
    scene_csv = get_scenes(path)

    process_all_videos_in_csv(scene_csv, "./videos/keyframes")
    create_keyframes_csv("./videos/keyframes", scene_csv)
    save_subtitle_in_csv(subtitles, "videos/keyframes/extracted_keyframes.csv")

    quantization_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_use_double_quant=True,
     bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForVision2Seq.from_pretrained(
     "HuggingFaceM4/idefics2-8b",
     torch_dtype=torch.float16,
     _attn_implementation="flash_attention_2",
     quantization_config=quantization_config
    )
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

    directory = "./videos/keyframes"
    tasks = {
       "CAPTION": "Caption the scene. Describe the contents and likely topics with as much detail as possible.",
       "LANGUAGE": "What is the language used in the video this keyframe was captured from",
       "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc",
       "BACKGROUND": "What is the background of the scene Describe it in detail.",
       "OBJECTS": " Can you list all objects sperated by commas",
     }

    prompt = f"""
    """.strip()

    caption_images_idefics_2(model=model, processor=processor, tasks=tasks, directory=directory)
    #  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #  model = LLMModel(model_id, load_in_4bit=True)
    #  align_video_captions(model, "./videos/keyframes/extracted_keyframes.csv",tasks)
    
    scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

    gc.collect()
    model=None
    torch.cuda.empty_cache()
    

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16,)
    tokenizer=AutoTokenizer.from_pretrained(model_id)

    create_scene_caption(model,tokenizer,subtitles, "./videos/keyframes/extracted_keyframes.csv","./videos/keyframes/llm_captions.csv")
    scene_objects_with_llm_data=set_new_content_for_metadata_attribute_for_sceneobjects( path_to_keyframes_csv="./videos/keyframes/llm_captions.csv" ,scene_objects= scene_objects_with_extraction_data, attribute="Caption")
    
    metaDataObject=MetaDataObject(input_string, downloader.yt, scene_objects_with_llm_data)

    with open('metadata_idefics.json', 'w') as outfile:
        outfile.write(metaDataObject.to_json())
        print(metaDataObject.to_json())