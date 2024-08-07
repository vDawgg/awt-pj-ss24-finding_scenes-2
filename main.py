import gc
import sys
from xmlrpc.client import Error
from pathlib import Path

from utils.constants import VIDEO_DIR, HF_TOKEN
from utils.video.youtube import YouTubeVideo
from utils.objects.metadata_object import MetaDataObject
from utils.video.scenes import get_scenes
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.caption_keyframes import caption_images_idefics_2
from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file, set_new_content_for_metadata_attribute_for_scene_objects
from utils.llm.mistral_helper import create_key_concept_for_scene_with_audio_of_scene, create_lom_caption_with_just_scenes_List, create_scene_caption_with_audio_of_scene, create_video_caption
from utils.keyframe.keyframe_extraction import generate_keyframes_from_scenes

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Error("Please enter a youtube link to run this script")
    input_string = sys.argv[1]

    downloader = YouTubeVideo(input_string)
    title = downloader.get_youtube_video_title()
    no_of_frames_to_return = 3

    if Path(f'{VIDEO_DIR}/{title}.json').is_file():
        print(f"Metadata for {title} has already been generated")
        sys.exit()

    subtitles = downloader.download_subtitles()
    path = downloader.download_video()
    scene_csv = get_scenes(path)

    keyframe_csv_path = generate_keyframes_from_scenes(
        video_name=title,
        no_of_frames_to_return=no_of_frames_to_return
    )

    print("Keyframes saved at:", keyframe_csv_path)

    save_subtitle_in_csv(subtitles, keyframe_csv_path)

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

    keyframes_dir = str(Path(VIDEO_DIR) / f"{title}_keyframes")

    tasks = {
        "CAPTION": "Caption the scene. Describe the contents and likely topics with as much detail as possible.",
        "KEY-CONCEPTS": "What are the key-concepts outlined in this scene?",
        "QUESTIONS": "Are there any questions or interactions addressed to the audience in this scene? If not simply answer 'NONE'",
        "TEXT": "Transcribe the text in this scene if there is any. Only answer with the text that is visible to you and nothing else. If there is no text do answer with 'NONE'",
        "RESOURCES": "Are there any additional resources mentioned in this scene? If not simply answer 'NONE'",
        "LANGUAGE": "What is the language used in the video this keyframe was captured from",
        "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc",
     }

    output_csv = caption_images_idefics_2(model=model,
                                          processor=processor,
                                          tasks=tasks,
                                          csv_file=keyframe_csv_path,
                                          directory=keyframes_dir)
    scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

    gc.collect()
    torch.cuda.empty_cache()
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            quantization_config=quantization_config,
                            attn_implementation="flash_attention_2",
                            torch_dtype=torch.float16,
                            token=HF_TOKEN
                        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    scene_objects_with_llm_data = get_metadata_from_keyframe_file(
                                        path_to_keyframes_csv=output_csv,
                                        scene_objects=scene_objects_with_extraction_data,
                                        tasks=tasks
                                    )
    create_scene_caption_with_audio_of_scene(
        model,
        tokenizer,
        subtitles,
        keyframe_csv_path,
        Path(keyframes_dir) / "llm_captions.csv",
        scene_csv
    )

    scene_objects_with_llm_data = set_new_content_for_metadata_attribute_for_scene_objects(
                                        path_to_keyframes_csv=f"{keyframes_dir}/llm_captions.csv",
                                        scene_objects= scene_objects_with_extraction_data,
                                        attribute="Caption"
                                    )
    create_key_concept_for_scene_with_audio_of_scene(
        model,
        tokenizer,
        subtitles,
        keyframe_csv_path,
        f"{keyframes_dir}/llm_key_concepts.csv",
        scene_csv
    )

    scene_objects_with_llm_data = set_new_content_for_metadata_attribute_for_scene_objects(
                                        path_to_keyframes_csv=f"{keyframes_dir}/llm_key_concepts.csv",
                                        scene_objects= scene_objects_with_extraction_data,
                                        attribute="KEY-CONCEPTS"
                                    )
    description = create_video_caption(model,tokenizer, subtitles,f"{keyframes_dir}/llm_captions.csv")
    video_json = create_lom_caption_with_just_scenes_List(model,tokenizer, subtitles, f"{keyframes_dir}/llm_captions.csv")

    metaDataObject = MetaDataObject(input_string, downloader.yt, scene_objects_with_llm_data)
    metaDataObject.llm_description = description

    for key, value in video_json.items():
        setattr(metaDataObject, key.lower().replace(" ", "_"), value)

    with open(f'{VIDEO_DIR}/{downloader.yt.title}.json', 'w') as outfile:
        outfile.write(metaDataObject.to_json())
        print(metaDataObject.to_json())

        
