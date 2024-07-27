from Katna.config import Video
from nltk.corpus.reader import titles

from utils.constants import VIDEO_DIR

import gc
import json
import torch
import subprocess
import pandas as pd
from pathlib import Path
from utils.video.youtube import YouTubeVideo
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from utils.video.youtube import YouTubeVideo
from utils.objects.metadata_object import MetaDataObject
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


from utils.video.scenes import get_scenes
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.caption_keyframes import caption_images_idefics_2
from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file, set_new_content_for_metadata_attribute_for_scene_objects
from utils.llm.mistral_helper import create_key_concept_for_scene_with_audio_of_scene, create_lom_caption_with_just_scenes_List, create_scene_caption_with_audio_of_scene,create_video_caption



app = FastAPI()

    
def file_exists(file_path: str) -> bool:
    return Path(file_path).exists()


# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return HTMLResponse(content=open("utils/assets/index.html").read(), status_code=200)


@app.post("/video")
def download_video(url: str):
    downloader = YouTubeVideo(url)
    path = downloader.download_video()
    subtitles= downloader.download_subtitles()
    print(path)
    return {"path": path, "subtitles": subtitles}

@app.get("/scenes")
def scenes(url: str):
    try:
        title = YouTubeVideo(url).get_youtube_video_title()
        print(title)
        path = str(Path(VIDEO_DIR) / f"{title}.mp4")
        print(path)
        scene_csv = get_scenes(path)
        print(scene_csv)
        return {"scene_csv": scene_csv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/keyframes")
def extract_keyframes(url: str):
    try:
        title = YouTubeVideo(url).get_youtube_video_title()
        scene_csv_path = Path(VIDEO_DIR) / f"{title}_scenes" / 'scene_list.csv'

        command = ["python", "utils/video/keyframe_extraction.py", scene_csv_path, "3"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Output:", result.stdout)
        else:
            print("Error:", result.stderr)
        
        keyframes_csv = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'
        keyframes_data = pd.read_csv(keyframes_csv)
        
        return {
            "message": "Keyframes extracted successfully",
            "keyframes_data": keyframes_data.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption")
def get_caption(url: str):
    try:
        downloader = YouTubeVideo(url)
        subtitles = downloader.download_subtitles()
        csv_path = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'
        save_subtitle_in_csv(subtitles, csv_path)
        keyframes_csv = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'
        keyframes_data = pd.read_csv(keyframes_csv)
        
        return {
            "message": "Subtitles extracted successfully",
            # "keyframes_data": keyframes_data.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/frame_caption")
def get_frame_caption():

    gc.collect()
    torch.cuda.empty_cache()

    try:
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

        return {"message": "Captioning completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def generate_metadata(url: str):
    gc.collect()
    torch.cuda.empty_cache()

    try:
        downloader = YouTubeVideo(url)
        subtitles = downloader.download_subtitles()

        tasks = {
            "CAPTION": "Caption the scene. Describe the contents and likely topics with as much detail as possible.",
            "KEY-CONCEPTS": "What are the key-concepts outlined in this scene?",
            "QUESTIONS": "Are there any questions or interactions addressed to the audience in this scene? If not simply answer 'NONE'",
            "TEXT": "Transcribe the text in this scene if there is any. Only answer with the text that is visible to you and nothing else. If there is no text do answer with 'NONE'",
            "RESOURCES": "Are there any additional resources mentioned in this scene? If not simply answer 'NONE'",
            "LANGUAGE": "What is the language used in the video this keyframe was captured from",
            "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc",
        }

        title = YouTubeVideo(url).get_youtube_video_title()
        output_csv = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'
        scene_csv = Path(VIDEO_DIR) / f"{title}_scenes" / 'scene_list.csv'

        
        scene_objects_with_extraction_data=get_metadata_from_scene_file(path_to_scene_csv=scene_csv)
        
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


        metaDataObject=MetaDataObject(url, downloader.yt, scene_objects_with_llm_data)
        metaDataObject.llm_description=description

        for key, value in video_json.items():
          setattr(metaDataObject, key.lower().replace(" ", "_"), value)

        with open(f'{VIDEO_DIR}/{title}.json', 'w') as outfile:
            outfile.write(metaDataObject.to_json())
            print(metaDataObject.to_json())

        return {"message": "Metadata extraction completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline")
def run_pipeline(url: str):
    title = YouTubeVideo(url).get_youtube_video_title()

    if not file_exists(str(Path(VIDEO_DIR) / f"{title}.mp4")):
        keyframes_csv = (Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv')
        metadata_json = Path(Path(VIDEO_DIR) / f"{title}.json")
        
        keyframes_csv.unlink(missing_ok=True)
        metadata_json.unlink(missing_ok=True)
        download_video(url)

    if not file_exists(str(Path(VIDEO_DIR) / f"{title}_scenes" / 'scene_list.csv')):
        keyframes_csv = (Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv')
        metadata_json = Path(Path(VIDEO_DIR) / f"{title}.json")

        keyframes_csv.unlink(missing_ok=True)
        metadata_json.unlink(missing_ok=True)
        scenes(url)
        extract_keyframes(url)
        get_caption(url)

    if not file_exists(str(f'{VIDEO_DIR}/{title}.json')):
        get_frame_caption()
        generate_metadata(url)

    with open(f'{VIDEO_DIR}/{title}.json', 'r') as file:
        metadata = json.load(file)

    return {
        "message": "Metadata extraction completed successfully",
        "metadata": metadata
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)