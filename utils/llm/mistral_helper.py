import gc
import os
import csv
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.llm.model import LLMModel  # Ensure you have this custom module
from utils.model.model import Model  # Ensure you have this custom module
from utils.video.subtitles import search_subtitle_for_scene
import json
from utils.video.youtube import YouTubeVideo  # Ensure you have this custom module


def create_prompt_of_scene_for_caption_using_scene_subtitles(scene_object, scene_subtitles: str) -> str:
    """Creates a detailed description prompt for a scene using keyframe descriptions and corresponding subtitles of the scene."""
    
    # Gather keyframe descriptions with corresponding subtitles
    key_frame_descriptions = "\n".join(
        [f" Keyframe Description : {description_caption}\nAudio transscript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object["CAPTION"], scene_object["Subtitle"]), start=1)]
    )
    
    # Prepare the prompt in the specified format
    prompt = (
    "You are a highly advanced AI tasked with generating a detailed description of a scene. You will be provided with keyframe descriptions and their corresponding subtitles, along with the full audio transcript of the scene for additional context.\n\n"
    "Instructions:\n\n"
    "1. **Primary Sources**: Use the keyframe descriptions and their corresponding audio transscript as your main sources of information.\n"
    "2. **Secondary Source**: Utilize the full audio transcript of the scene only for additional context if necessary.\n"
    "3. **Focus**: Concentrate on creating a coherent and accurate depiction of the scene based primarily on the keyframe information provided.\n\n"
    "Scene Information:\n\n"
    "Keyframe Descriptions with Corresponding Audio transscript:\n"
    f"{key_frame_descriptions}\n\n"
    "Full Scene Audio Transcript for Context:\n"
    f"{scene_subtitles}\n\n"
    "Task:\n"
    "Generate a detailed and vivid description of the scene based on the provided keyframe descriptions and audio transscript. Do not explain the scene in detail; simply describe it.\n"
    "Description of the Scene:"
    )
    
    return prompt

def create_prompt_of_scene_for_key_concepts(scene_object, scene_subtitles: str) -> str:
    """Creates a detailed description prompt for identifying key concepts in a scene using keyframe descriptions and corresponding subtitles of the scene."""
    
    # Gather keyframe concepts with corresponding subtitles
    key_frame_concepts = "\n".join(
        [f"Keyframe Concept: {concept_caption}\n Audio transscript: {concept_subtitle}" for i, (concept_caption, concept_subtitle) in enumerate(zip(scene_object["KEY-CONCEPTS"], scene_object["Subtitle"]), start=1)]
    )
    
    # Prepare the prompt in the specified format
    prompt = (
    "You are a highly advanced AI tasked with identifying the key concepts outlined in a scene. You will be provided with key concepts and their corresponding audio transscript, along with the full audio transscript of the scene for additional context.\n\n"
    "Instructions:\n\n"
    "1. **Primary Sources**: Use the key concepts and their corresponding audio transscript as your main sources of information.\n"
    "2. **Secondary Source**: Utilize the full scene audio transscript only for additional context if necessary.\n"
    "3. **Focus**: Concentrate on identifying and listing the key concepts based primarily on the keyframe information provided.\n\n"
    "Scene Information:\n\n"
    "Key Concepts with Corresponding audio transscript:\n"
    f"{key_frame_concepts}\n\n"
    "Full Scene audio transscript for Context:\n"
    f"{scene_subtitles}\n\n"
    "Task:\n"
    "Identify and list the key concepts outlined in the scene based on the provided key concepts and audio transscript.\n"
    "Be short and concise in your response.\n\n"
    "Keyframe Concepts: "
    )
    
    return prompt

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



def create_prompt_for_video_description(scenes_captions: List[str], srt_context: str) -> str:
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
        "\nOutput the description in a clear and engaging manner, focusing on the key elements and themes of the video."
        "Just describe the video; don't explain it in detail.\n\n"
        "Give me the description of the video:"
    )
    
    return prompt


def create_lom_prompt_for_video(scenes_captions: List[str], srt_context: str) -> str:
    prompt = (
        "You are a highly intelligent AI capable of generating detailed Learning Object Metadata (LOM) for educational videos. "
        "Below are detailed captions of key frames from various scenes. Your task is to synthesize these into a comprehensive LOM object for the video, "
        "focusing on the educational attributes:\n"
    )

    subtitles = parse_srt(srt_context)
    if subtitles:
        subtitle_context = "\n".join([f"{i}: {subtitle}" for i, subtitle in enumerate(subtitles, start=1)])
    else:
        subtitle_context = "No subtitles available"
    
    # Add captions of each scene to the prompt
    for scene_number, scene_caption in enumerate(scenes_captions, start=1):
        prompt += f"\nScene:{scene_number} Caption: {scene_caption}:\n"

    prompt += ( 
        "\nUsing the above key frame descriptions, please generate a detailed Learning Object Metadata (LOM) object for the video. "
        "Ensure to include the following educational attributes:\n"
        "- **Learning Resource Type**: The type of learning resource (e.g., lecture, tutorial, demonstration).\n"
        "- **Interactivity Type**: The type of interactivity (e.g., passive, active, mixed).\n"
        "- **Interactivity Level**: The degree of interactivity (e.g., low, medium, high).\n"
        "- **Intended End User Role**: The primary user role (e.g., student, teacher).\n"
        "- **Context**: The educational context (e.g., higher education, vocational training).\n"
        "- **Difficulty Level**: The difficulty level of the content (e.g., beginner, intermediate, advanced).\n"
        "- **Typical Learning Time**: The typical time required to complete the learning from the video.\n"
        "- **Discipline**: The academic or professional discipline relevant to the video content.\n"
        "- **Educational Level**: The education level targeted by the video (e.g., undergraduate, professional development).\n"
        "Additionally, consider the following audio script context while crafting your description:\n"
        f"{subtitle_context}"
        "\nOutput the LOM object in the following JSON structure:\n"
        "{\n"
        "  \"LearningResourceType\": \"\",\n"
        "  \"InteractivityType\": \"\",\n"
        "  \"InteractivityLevel\": \"\",\n"
        "  \"IntendedEndUserRole\": \"\",\n"
        "  \"Context\": \"\",\n"
        "  \"DifficultyLevel\": \"\",\n"
        "  \"TypicalLearningTime\": \"\",\n"
        "  \"Discipline\": \"\",\n"
        "  \"EducationalLevel\": \"\"\n"
        "}\n"
    )
    
    return prompt

from typing import List

def create_lom_prompt_for_video_with_transcript(audio_transcript: List[str]) -> str:
    prompt = (
        "You are a highly intelligent AI capable of generating detailed Learning Object Metadata (LOM) for educational videos. "
        "Below is a detailed transcript from the audio of the video. Your task is to synthesize this into a comprehensive LOM object for the video, "
        "focusing on the educational attributes:\n"
    )
    audio_transcript = parse_srt(audio_transcript)
    transcript_context = "\n".join([f"{i}: {line}" for i, line in enumerate(audio_transcript, start=1)])
    
    prompt += ( 
        f"\nAudio Transcript:\n{transcript_context}\n"
        "\nUsing the above transcript, please generate a detailed Learning Object Metadata (LOM) object for the video. "
        "Ensure to include the following educational attributes:\n"
        "- **Learning Resource Type**: The type of learning resource (e.g., lecture, tutorial, demonstration).\n"
        "- **Interactivity Type**: The type of interactivity (e.g., passive, active, mixed).\n"
        "- **Interactivity Level**: The degree of interactivity (e.g., low, medium, high).\n"
        "- **Intended End User Role**: The primary user role (e.g., student, teacher).\n"
        "- **Context**: The educational context (e.g., higher education, vocational training).\n"
        "- **Difficulty Level**: The difficulty level of the content (e.g., beginner, intermediate, advanced).\n"
        "- **Typical Learning Time**: The typical time required to complete the learning from the video.\n"
        "- **Discipline**: The academic or professional discipline relevant to the video content.\n"
        "- **EducationalLevel**: The education level targeted by the video (e.g., undergraduate, professional development).\n"
        "- **TargetAudienceAge**: The average age of the target audience for the video (e.g., 25 years old, mid-30s).\n"
        "Output the LOM object in the following JSON structure:\n"
        "{\n"
        "  \"LearningResourceType\": \"\",\n"
        "  \"InteractivityType\": \"\",\n"
        "  \"InteractivityLevel\": \"\",\n"
        "  \"IntendedEndUserRole\": \"\",\n"
        "  \"Context\": \"\",\n"
        "  \"DifficultyLevel\": \"\",\n"
        "  \"TypicalLearningTime\": \"\",\n"
        "  \"Discipline\": \"\",\n"
        "  \"EducationalLevel\": \"\",\n"
        "  \"TargetAudienceAge\": \"\",\n" "}\n"
    )
    
    return prompt

from typing import List

def create_lom_prompts_for_video_with_transcript_iterate(audio_transcript: List[str]) -> List[str]:

    base_prompt = (
       "Below is a detailed transcript from the audio of the video. Your task is to synthesize this into a short comprehensive Bullet Point \n\n"
       "For following Task: "
    )
    
    # Parse the transcript assuming it's in subtitle format (like SRT)
    audio_transcript = parse_srt(audio_transcript)
    transcript_context = "\n".join([f"{i}: {line}" for i, line in enumerate(audio_transcript, start=1)])
    
    # LOM attributes with their descriptions
    
    lom_attributes = {
        "Learning Resource Type": "Provide the type of learning resource (e.g., lecture, tutorial, demonstration).",
        "Interactivity Type": "Specify the interactivity type (e.g., passive, active, mixed).",
        "Interactivity Level": "Specify the degree of interactivity (e.g., low, medium, high).",
        "Intended End User Role": "Describe the primary user role (e.g., student, teacher).",
        "Context": "Describe the educational context (e.g., higher education, vocational training).",
        "Difficulty Level": "Indicate the difficulty level of the content (e.g., beginner, intermediate, advanced).",
        "Typical Learning Time": "State the typical time required to complete the learning from the video (e.g., 10-20 hours, 20 mins).",
        "Discipline": "Specify the academic or professional discipline relevant to the video content (e.g., Computer Science, Sociology, Mathematics, Psychology).",
        "Educational Level": "Specify the education level targeted by the video (e.g., undergraduate, professional development).",
        "Target Audience Age": "Provide the average age of the target audience for the video (e.g., 25 years old, mid-30s)."
    }
    
    prompts = {}

    # Iterate over each attribute and generate the prompt
    for attribute, description in lom_attributes.items():
        prompt = (
            base_prompt +
            f"{description}\n\n"
            "Focus on providing the content only, WITHOUT explanations or additional details.\n\n"
            "Use Following captions for from scenes for context:\n\n"
             f"{transcript_context}\n\n"
             f"{attribute}:"
             
        )
        prompts[attribute] = prompt   
        
    return prompts

def create_lom_prompts_for_video_with_scenes_iterate(scenes_captions: List[str]) -> List[str]:

    base_prompt = (
        "Below are captions from scenes of the video. Your task is to synthesize these into short comprehensive keywords.\n\n"
        "Just give the words without any addiontal Context\n\n"
        "For the following Task:\n"
    )
    # LOM attributes with their descriptions    
    lom_attributes = {
        "Learning Resource Type": "Provide the type of learning resource (e.g., lecture, tutorial, demonstration).",
        "Intended End User Role": "Describe the primary user role (e.g., student, teacher).",
        "Context": "Describe the educational context (e.g., higher education, vocational training).",
        "Difficulty Level": "Indicate the difficulty level of the content (e.g., beginner, intermediate, advanced).",
        "Typical Learning Time": "State the typical time required to complete the learning from the video (e.g., 10-20 hours, 20 mins).",
        "Discipline": "Specify the academic or professional discipline relevant to the video content (e.g., Computer Science, Sociology, Mathematics, Psychology).",
        "Educational Level": "Specify the education level targeted by the video (e.g., undergraduate, professional development).",
        "Target Audience Age": "Provide the average age of the target audience for the video (e.g., 25 years old, mid-30s)."
    }
    
    prompts = {}

    # Iterate over each attribute and generate the prompt
    for attribute, description in lom_attributes.items():
        prompt = (
           # base_prompt +
            f"{description}\n\n"
            "Focus on providing the content only, WITHOUT explanations or additional details.\n\n"
            "Use Following Captions of Scenes of the video for context:\n\n"    
        )
        for scene_number, scene_caption in enumerate(scenes_captions, start=1):
            prompt += f"\nScene:{scene_number} Caption: {scene_caption}:\n"
        prompt+= "Use maximum 5 words in your output\n\n"
        prompt += f"{attribute}:"   
        prompts[attribute] = prompt 
      
        
    return prompts

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


def create_scene_caption_with_audio_of_scene(model,tokenizer, subtitles,keyframes_path,output_path,scene_path=""):
    # Check if the output path exists
    if os.path.exists(output_path):
        # Delete the existing CSV file
        os.remove(output_path)
   
    dict_list = get_content_of_column_by_source_and_column_names(keyframes_path, ["CAPTION","Subtitle"])
    subtiles_dict = search_subtitle_for_scene(subtitles,scene_path)
    for source_filename, column_data in dict_list.items():
        print(source_filename)
        subtitles_of_scene = subtiles_dict[source_filename]
        llm_prompt_for_scene = create_prompt_of_scene_for_caption_using_scene_subtitles(column_data, subtitles_of_scene)
        print(llm_prompt_for_scene)
        encodeds =  tokenizer(llm_prompt_for_scene, return_tensors="pt").to("cuda")
        prompt_length = encodeds['input_ids'].shape[1]
        generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=600, do_sample=True)
        decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
        caption = decoded.replace('\n', ' ').replace('\r', ' ')
        save_data_to_csv(output_path, [{"Source Filename": source_filename, "Caption": caption}])
    return  output_path  

def create_key_concept_for_scene_with_audio_of_scene(model,tokenizer, subtitles,keyframes_path,output_path,scene_path=""):
    if os.path.exists(output_path):
        # Delete the existing CSV file
        os.remove(output_path)

    dict_list = get_content_of_column_by_source_and_column_names(keyframes_path, ["KEY-CONCEPTS","Subtitle"])
    print(dict_list)
    subtiles_dict = search_subtitle_for_scene(subtitles,scene_path)
    for source_filename, column_data in dict_list.items():
        print(source_filename)
        subtitles_of_scene = subtiles_dict[source_filename]
        llm_prompt_for_scene = create_prompt_of_scene_for_key_concepts(column_data, subtitles_of_scene)
        print(llm_prompt_for_scene)
        encodeds =  tokenizer(llm_prompt_for_scene, return_tensors="pt").to("cuda")
        prompt_length = encodeds['input_ids'].shape[1]
        generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=100, do_sample=True)
        decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
        caption = decoded.replace('\n', ' ').replace('\r', ' ')
        save_data_to_csv(output_path, [{"Source Filename": source_filename, "KEY-CONCEPTS": caption}], ['Source Filename', 'KEY-CONCEPTS'])
    return  output_path  


def create_scene_caption_with_audio_of_whole_video(model,tokenizer, subtitles,keyframes_path,output_path,):
    # Check if the output path exists
    if os.path.exists(output_path):
        # Delete the existing CSV file
        os.remove(output_path) 
    dict_list = get_content_of_column_by_source_and_column_names(keyframes_path, ["CAPTION","Subtitle",""])
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
    prompt=create_prompt_for_video_description(scene_dict,subtitles)
    print(prompt)
    encodeds =  tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = encodeds['input_ids'].shape[1]
    generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=200, do_sample=True, )
    decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
    caption = decoded.replace('\n', ' ').replace('\r', ' ')

    return caption

def create_lom_caption(model,tokenizer, subtitles,input_path):
    scene_dict=get_scene_caption_from_csv(input_path, "Caption")
    prompt=create_lom_prompt_for_video(scene_dict,subtitles)
    print(prompt)
    encodeds =  tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = encodeds['input_ids'].shape[1]
    generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=20, do_sample=True, )
    decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
    caption = decoded.replace('\n', ' ').replace('\r', ' ')

    return caption

def create_lom_caption_with_just_subtitle(model,tokenizer, subtitles,input_path):
    prompt=create_lom_prompt_for_video_with_transcript(subtitles)
    print(prompt)
    encodeds =  tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = encodeds['input_ids'].shape[1]
    generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=100, do_sample=True, )
    decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
    caption = decoded.replace('\n', ' ').replace('\r', ' ')

    return caption


def create_lom_caption_with_just_subtitle_List(model,tokenizer, subtitles,input_path):
    prompts=create_lom_prompts_for_video_with_transcript_iterate(subtitles)
    captions = {}
    for attribute, prompt in prompts.items():
        print(prompt)
        encodeds = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = encodeds['input_ids'].shape[1]
        generated_ids = model.generate(encodeds['input_ids'], max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,do_sample=True,num_return_sequences=1)
        decoded = tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
        caption = decoded.replace('\n', ' ').replace('\r', ' ')
        captions[attribute] = caption
    return captions

def create_lom_caption_with_just_scenes_List(model,tokenizer, subtitles,input_path):
    scene_dict=get_scene_caption_from_csv(input_path, "Caption")
    prompts=create_lom_prompts_for_video_with_scenes_iterate(scene_dict)
    captions = {}
    for attribute, prompt in prompts.items():
        print(prompt)
        encodeds = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = encodeds['input_ids'].shape[1]
        generated_ids = model.generate(encodeds['input_ids'], max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,do_sample=True,num_return_sequences=1)
        decoded = tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
        caption = decoded.replace('\n', ' ').replace('\r', ' ')
        captions[attribute] = caption
    return captions


if __name__ == '__main__':

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    input_string = "https://www.youtube.com/watch?v=d56mG7DezGs"
    downloader = YouTubeVideo(input_string)
    path, subtitles = downloader.download_video_and_subtitles()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    # tokenizer=AutoTokenizer.from_pretrained(model_id)
    # create_scene_caption(model,tokenizer, subtitles, "C:/uni/awt-pj-ss24-finding_scenes-2/videos/keyframes/extracted_keyframes.csv","./test2.csv")
    # gc.collect()
    # model=None
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    caption=create_lom_caption_with_just_subtitle_List(model,tokenizer, subtitles,"C:/uni/awt-pj-ss24-finding_scenes-2/videos/keyframes/llm_captions.csv")
    # Print the JSON
    print(caption)

    
