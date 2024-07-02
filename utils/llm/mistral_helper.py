import gc
import os
import csv
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.llm.model import LLMModel  # Ensure you have this custom module
from utils.model.model import Model  # Ensure you have this custom module
from utils.video.video_extraction_with_pytube import YouTubeVideo  # Ensure you have this custom module

def create_prompt_for_scene5(scene_object, audio_script, caption: str, subtitle: str) -> str:
    """Creates a detailed description prompt for a scene using keyframe descriptions and corresponding transcripts."""
    
    # Gather keyframe descriptions with corresponding audio transcripts
    key_frame_descriptions = "\n".join(
        [f" Keyframe Description : {description_caption}\nAudio Transcript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object[caption], scene_object[subtitle]), start=1)]
    )
    
    # Use the provided audio script for additional context
    subtitles = parse_srt(audio_script)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Prepare the prompt in the specified format
    prompt = (
    "You are a highly advanced AI whose task is to generate a detailed description of a scene based primarily on provided keyframe descriptions and their corresponding audio transcripts.\n\n"
    "You will be given detailed descriptions of keyframes along with their corresponding audio transcripts. Additionally, you will receive an audio script for the entire video for contextual support, but your primary sources of information should be the keyframe descriptions and their audio transcripts.\n\n"
    "Please generate a coherent and accurate description of the scene based on the keyframe descriptions and audio transcripts. Use the audio script from the whole video sparingly for additional context if needed, but focus mainly on the keyframe information provided.\n\n"
    "Keyframe Descriptions with Audio Transcripts:\n"
    f"{key_frame_descriptions}\n\n"
    "Audio Script from the Whole Video for Additional Context:\n"
    f"{subtitle_context}\n\n"
    "Generate a detailed description of the scene:")

    
    return prompt

def create_prompt_for_scene_with_context_of_different_scene(scene_object, audio_script, caption: str, subtitle: str) -> str:
    """Creates a detailed description prompt for a scene using keyframe descriptions and corresponding transcripts."""
    
    # Gather keyframe descriptions with corresponding audio transcripts
    key_frame_descriptions = "\n".join(
        [f" Keyframe Description : {description_caption}\nAudio Transcript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object[caption], scene_object[subtitle]), start=1)]
    )
    
    # Use the provided audio script for additional context
    subtitles = parse_srt(audio_script)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Prepare the prompt in the specified format
    prompt = (
    "You are a highly advanced AI whose task is to generate a detailed description of a scene based primarily on provided keyframe descriptions and their corresponding audio transcripts.\n\n"
    "You will be given detailed descriptions of keyframes along with their corresponding audio transcripts. Additionally, you will receive an audio script for the entire video for contextual support, but your primary sources of information should be the keyframe descriptions and their audio transcripts.\n\n"
    "Please generate a coherent and accurate description of the scene based on the keyframe descriptions and audio transcripts. Use the audio script from the whole video sparingly for additional context if needed, but focus mainly on the keyframe information provided.\n\n"
    "Keyframe Descriptions with Audio Transcripts:\n"
    f"{key_frame_descriptions}\n\n"
    "Audio Script from the Whole Video for Additional Context:\n"
    f"{subtitle_context}\n\n"
    "Generate a detailed description of the scene:")

    
    return prompt



def create_prompt_for_scene4(scene_object, srt_content, caption: str, subtitle: str) -> str:
    """Creates a detailed description prompt for a scene using keyframe descriptions and corresponding transcripts."""
    
    # Gather keyframe descriptions with corresponding audio transcripts
    key_frame_descriptions = "\n".join(
        [f"{i}: {description_caption,}\nAudio Transcript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object[caption], scene_object[subtitle]), start=1)]
    )
    # Parse subtitles
    subtitles = parse_srt(srt_content)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Prepare the prompt in the specified format
    prompt = (
        "You are a highly advanced AI whose task is to generate a detailed description of a scene based on provided keyframe descriptions and corresponding audio transcripts. "
        "You will be given descriptions of keyframes along with their corresponding audio transcripts, as well as some context from preceding subtitles. "
        "Use the keyframe descriptions and their audio transcripts as your primary sources of information to create a coherent and accurate scene description. "
        "If available, use the subtitle context to enrich your description, but prioritize the keyframe descriptions and their audio transcripts."
        "\n\nKeyframe Descriptions with Audio Transcripts:\n"
        f"{key_frame_descriptions}"
        "\n\nContext from Subtitles:\n"
        f"{subtitle_context}"
        "\n\nGenerate a detailed description of the scene:"
    )
    
    return prompt


def create_prompt_for_scene3(scene_object, srt_content,caption:str, subtitle:str) -> str:
    """Creates a detailed description prompt for a scene, including context from subtitles in the specified format."""
    
    # Gather key frame descriptions
    key_frame_descriptions = "\n".join(
        [f"{i}: {description_caption,}\nAudio Transcript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object[caption], scene_object[subtitle]), start=1)]
    )
    
    # Parse subtitles
    subtitles = parse_srt1(srt_content)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in subtitles])
    else:
        subtitle_context = "No subtitles available"
    
    # Prepare the prompt in the specified format
    prompt = (
        "You are a bot whose job is to create a caption for a scene based on keyframe descriptions and context from subtitles. "
        "A user will supply you with descriptions of keyframes and the context from preceding subtitles. "
        "Your task is to generate a coherent and accurate caption for the scene using the provided information. "
        "The keyframe descriptions should be used as the primary source of information, supplemented by the context from the audio transcript of the whole video"
        "Keyframe Descriptions: "
        f"{key_frame_descriptions}"
        "\nContext from Subtitles: "
        f"{subtitle_context}"
        "\nCreate a Caption for the Scene:"
    )
    
    return prompt


def create_prompt_for_scene2(scene_object, srt_content) -> list:
    """Creates a detailed description prompt for a scene, including context from subtitles in the specified format."""
    
    # Gather key frame descriptions
    key_frame_descriptions = "\n".join([f"{i}: {description}" for i, description in enumerate(scene_object.values(), start=1)])
    
    # Parse subtitles
    subtitles = parse_srt(srt_content)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Prepare the prompt in the specified format
    prompt = [

        {
            "role": "user",
            "content": "You are a bot whose job is to make sure that imprecise captions for different parts of a video are well aligned with each other. "
                       "To this end, a user will supply you with a caption and the context for this caption, which are the preceding captions. "
                       "Your task is to edit the caption to properly fit the context. Remove any mentions of image descriptions, etc., and ensure that the caption accurately describes the video."
                       f"Caption: '{key_frame_descriptions}'; Context: {subtitle_context}"
        },
      
    ]
    
    return prompt


def get_content_of_column_by_source_and_column_names(filepath, column_names: List[str]) -> dict:
    dict_list = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_filename = row['Source Filename']
            for column_name in column_names:
                if source_filename not in dict_list:
                    dict_list[source_filename] = {}
                if column_name not in dict_list[source_filename]:
                    dict_list[source_filename][column_name] = []
                dict_list[source_filename][column_name].append(row[column_name])
    return dict_list

def get_scene_caption_from_csv(filepath, column_name):
    descriptions = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            caption = row[column_name]
            descriptions.append(caption)
    
    return descriptions

def create_prompt_for_video(scenes_captions: List[str], srt_context: str) -> str:
    prompt = (
        "You are a highly intelligent AI with the ability to generate rich and comprehensive descriptions of videos. "
        "Below are detailed captions of key frames from various scenes. Your task is to synthesize these into a cohesive and vivid description of the entire video:\n"
    )

    subtitles = parse_srt(srt_context)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Add captions of each scene to the prompt
    for scene_number, scene_caption in enumerate(scenes_captions, start=1):
        prompt += f"\nScene {scene_number} Caption {scene_caption}:\n"

    prompt += (
        "\nUsing the above key frame descriptions, please generate a comprehensive and detailed narrative of the video.\n"
        "Additionally, consider the following Audio script context while crafting your description:\n"
        f"{subtitle_context}"
    )
    
    return prompt


def parse_srt(srt_content):
    """Parses SRT file content and returns a list of subtitle texts."""
    subtitles = []
    srt_blocks = srt_content.strip().split('\n\n')
    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            subtitle_text = ' '.join(lines[2:])
            subtitles.append(subtitle_text)
    return subtitles

def parse_srt1(srt_content):
    """Parses SRT file content and returns a list of tuples containing (occurrence, subtitle text)."""
    subtitles = []
    srt_blocks = srt_content.strip().split('\n\n')
    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            occurrence = lines[0].strip()
            subtitle_text = ' '.join(lines[2:])
            subtitles.append((occurrence, subtitle_text))
    return subtitles




def get_values(data, key):
    values = []
    for obj_name, obj in data.items():
        if key in obj:
            if isinstance(obj[key], list):
                values.extend(obj[key])
            else:
                values.append(obj[key])
    return values


def save_data_to_csv(filepath, data, header=['Source Filename', 'Caption']):
    """Saves data to a CSV file or creates a new file if it doesn't exist."""
    file_exists = os.path.isfile(filepath)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        for row in data:
            writer.writerow(row)

def create_scene_caption(model,tokenizer, subtitles,keyframes_path,output_path):
    # Check if the output path exists
    if os.path.exists(output_path):
        # Delete the existing CSV file
        os.remove(output_path) 
    dict_list = get_content_of_column_by_source_and_column_names(keyframes_path, ["CAPTION","Subtitle"])
    for source_filename, column_data in dict_list.items():
        print(source_filename)
        llm_prompt_for_scene = create_prompt_for_scene5(column_data, subtitles, "CAPTION", "Subtitle")
        print(llm_prompt_for_scene)
        encodeds =  tokenizer(llm_prompt_for_scene, return_tensors="pt").to("cuda")
        prompt_length = encodeds['input_ids'].shape[1]
        generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=600, do_sample=True)
        decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
        caption = decoded.replace('\n', ' ').replace('\r', ' ')
        save_data_to_csv(output_path, [{"Source Filename": source_filename, "Caption": caption}])
    return  output_path  

def create_video_caption(model,tokenizer, subtitles,input_path):
    scene_dict=get_scene_caption_from_csv(input_path, "Caption")
    prompt=create_prompt_for_video(scene_dict,subtitles)
    print(prompt)
    encodeds =  tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = encodeds['input_ids'].shape[1]
    generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=100, do_sample=True, )
    decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
    caption = decoded.replace('\n', ' ').replace('\r', ' ')

    return caption

    



if __name__ == '__main__':

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    input_string = "https://www.youtube.com/watch?v=5C_HPTJg5ek"
    downloader = YouTubeVideo(input_string)
    path, subtitles = downloader.download_video_and_subtitles()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    # tokenizer=AutoTokenizer.from_pretrained(model_id)
    # create_scene_caption(model,tokenizer, subtitles, "C:/uni/awt-pj-ss24-finding_scenes-2/videos/keyframes/extracted_keyframes.csv","./test2.csv")
    # gc.collect()
    # model=None
    # torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16,device_map = 'auto')
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    caption=create_video_caption(model,tokenizer, subtitles,"./test2.csv")
    print(caption)
    
