import subprocess


from pytube import YouTube
from pathlib import Path
from fastapi import FastAPI, HTTPException
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from utils.objects.metadata_object import MetaDataObject
from utils.video.video_extraction_with_pytube import YouTubeVideo
from utils.video.scenes import get_scenes
from utils.video.keyframe_extraction import process_all_videos_in_csv, create_keyframes_csv
from utils.video.subtitles import save_subtitle_in_csv
from utils.captioning.caption_keyframes import caption_images_idefics_2
from utils.llm.model import LLMModel
from utils.llm.align_captions import align_video_captions

from utils.metadata.metadata_function import get_metadata_from_scene_file, get_metadata_from_keyframe_file
from utils.constants import VIDEO_DIR

app = FastAPI()


def get_youtube_video_title(url: str) -> str:
    try:
        video = YouTube(url)
        return video.title
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error retrieving video title"


@app.get("/")
async def root():
    return {"message": "Advanced Web Technologies Project - Finding Scenes in Learning Videos"}


@app.post("/video")
def download_video(url: str):
    downloader = YouTubeVideo(url)
    path, subtitles = downloader.download_video_and_subtitles()
    print(path)
    return {"path": path, "subtitles": subtitles}


@app.get("/scenes")
def scenes(url: str):
    try:
        title = get_youtube_video_title(url)
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
        title = get_youtube_video_title(url)
        path = str(Path(VIDEO_DIR) / f"{title}.mp4")
        scene_csv = get_scenes(path)

        command = ["python", "utils/video/keyframe_extraction.py", scene_csv, "3"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Output:", result.stdout)
        else:
            print("Error:", result.stderr)
        
        return {"message": "Keyframes extracted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption")
def get_caption(url: str):
    try:
        downloader = YouTubeVideo(url)
        _, subtitles = downloader.download_video_and_subtitles()
        csv_path = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'
        save_subtitle_in_csv(subtitles, csv_path)

        return {"message": "Subtitles extracted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/frame_caption")
def get_frame_caption():
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


@app.post("/metadata")
def convert_metadata(url: str):
    try:
        title = get_youtube_video_title(url)
        downloader = YouTubeVideo(url)

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model = LLMModel(model_id, load_in_4bit=True)

        tasks = {
            "CAPTION": "Caption the scene. Describe the contents and likely topics with as much detail as possible.",
            "KEY-CONCEPTS": "What are the key-concepts outlined in this scene?",
            "QUESTIONS": "Are there any questions or interactions addressed to the audience in this scene? If not simply answer 'NONE'",
            "TEXT": "Transcribe the text in this scene if there is any. Only answer with the text that is visible to you and nothing else. If there is no text do answer with 'NONE'",
            "RESOURCES": "Are there any additional resources mentioned in this scene? If not simply answer 'NONE'",
            "LANGUAGE": "What is the language used in the video this keyframe was captured from",
            "VIDEO_TYPE": "What kind of video is this, is it a tutorial, a lecture, etc",
        }
        output_csv = Path(VIDEO_DIR) / "keyframes" / 'extracted_keyframes.csv'

        align_video_captions(model, output_csv, tasks)

        scene_csv = Path(VIDEO_DIR) / f"{title}_scenes" / 'scene_list.csv'
        scene_objects_with_extraction_data = get_metadata_from_scene_file(path_to_scene_csv=scene_csv)

        print(scene_objects_with_extraction_data)

        scene_objects_with_llm_data = get_metadata_from_keyframe_file(
            path_to_keyframes_csv=output_csv,
            scene_objects=scene_objects_with_extraction_data,
            tasks=tasks
        )

        metaDataObject = MetaDataObject(url, downloader.yt, scene_objects_with_llm_data)

        with open('metadata_idefics.json', 'w') as outfile:
            outfile.write(metaDataObject.to_json())
            print(metaDataObject.to_json())

        return {"message": "Metadata extraction completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def generate_metadata(url: str):
    download_video(url)
    scenes(url)
    extract_keyframes(url)
    get_caption(url)
    get_frame_caption()
    convert_metadata(url)
    return {"message": "Metadata extraction completed successfully"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)