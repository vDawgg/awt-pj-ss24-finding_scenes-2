import gc
import torch

from utils.video.youtube import YouTubeVideo
from utils.objects.metadata_object import MetaDataObject
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from utils.video.scenes import get_scenes
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.caption_keyframes import caption_images_idefics_2
from utils.video.keyframe_extraction import process_all_videos_in_csv, create_keyframes_csv
from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file, set_new_content_for_metadata_attribute_for_scene_objects
from utils.llm.mistral_helper import create_key_concept_for_scene_with_audio_of_scene, create_lom_caption_with_just_scenes_List, create_scene_caption_with_audio_of_scene, create_video_caption

if __name__ == '__main__':

    input_string="https://www.youtube.com/watch?v=q0zmfNx7OM4"
    downloader = YouTubeVideo(input_string)

    subtitles = downloader.download_subtitles()
    path =downloader.download_video()
    scene_csv = get_scenes(path)

    process_all_videos_in_csv(scene_csv, "./videos/keyframes", no_of_frames_to_return=3)
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
        "KEY-CONCEPTS": "What are the key-concepts outlined in this scene?",
        "QUESTIONS": "Are there any questions or interactions addressed to the audience in this scene? If not simply answer 'NONE'",
        "TEXT": "Transcribe the text in this scene if there is any. Only answer with the text that is visible to you and nothing else. If there is no text do answer with 'NONE'",
        "RESOURCES": "Are there any additional resources mentioned in this scene? If not simply answer 'NONE'",
        "LANGUAGE": "What is the language used in the video this keyframe was captured from",
        "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc",
     }

    output_csv = caption_images_idefics_2(model=model, processor=processor, tasks=tasks, directory=directory)
    scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

    gc.collect()
    model=None
    torch.cuda.empty_cache()
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16,)
    tokenizer=AutoTokenizer.from_pretrained(model_id)

    scene_objects_with_llm_data=get_metadata_from_keyframe_file( path_to_keyframes_csv=output_csv ,scene_objects= scene_objects_with_extraction_data,tasks=tasks)
    create_scene_caption_with_audio_of_scene(model,tokenizer,subtitles, "./videos/keyframes/extracted_keyframes.csv","./videos/keyframes/llm_captions.csv",scene_csv)
    scene_objects_with_llm_data=set_new_content_for_metadata_attribute_for_scene_objects(path_to_keyframes_csv="./videos/keyframes/llm_captions.csv" ,scene_objects= scene_objects_with_extraction_data, attribute="Caption")
    create_key_concept_for_scene_with_audio_of_scene(model,tokenizer,subtitles, "./videos/keyframes/extracted_keyframes.csv","./videos/keyframes/llm_key_concepts.csv",scene_csv)
    scene_objects_with_llm_data=set_new_content_for_metadata_attribute_for_scene_objects(path_to_keyframes_csv="./videos/keyframes/llm_key_concepts.csv" ,scene_objects= scene_objects_with_extraction_data, attribute="KEY-CONCEPTS")
    description=create_video_caption(model,tokenizer, subtitles,"./videos/keyframes/llm_captions.csv")
    video_json=create_lom_caption_with_just_scenes_List(model,tokenizer, subtitles,"./videos/keyframes/llm_captions.csv")

    metaDataObject=MetaDataObject(input_string, downloader.yt, scene_objects_with_llm_data)
    metaDataObject.llm_description=description

    for key, value in video_json.items():
        setattr(metaDataObject, key.lower().replace(" ", "_"), value)

    with open('metadata_idefics.json', 'w') as outfile:
        outfile.write(metaDataObject.to_json())
        print(metaDataObject.to_json())

        