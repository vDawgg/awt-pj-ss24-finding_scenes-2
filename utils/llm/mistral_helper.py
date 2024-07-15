import os
import csv
from typing import List
from utils.video.subtitles import search_subtitle_for_scene

def create_prompt_of_scene_for_caption_using_scene_subtitles(scene_object:any, scene_subtitles: str) -> str:
    """Creates a detailed description prompt for a scene using keyframe descriptions and corresponding subtitles of the scene.

    :param str scene_object: The scene object containing keyframe descriptions and subtitles.
    :param str scene_subtitles: The full audio transcript of the scene.

    :rtype: str
    :returns: The prompt for generating a detailed description of the scene.
    """
    # Gather keyframe descriptions with corresponding subtitles
    key_frame_descriptions = "\n".join(
        [f" Keyframe Description : {description_caption}\nAudio transscript: {description_subtitle}" for i, (description_caption, description_subtitle) in enumerate(zip(scene_object["CAPTION"], scene_object["Subtitle"]), start=1)]
    )# Prepare the prompt in the specified format
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

def create_prompt_of_scene_for_key_concepts(scene_object:any, scene_subtitles: str) -> str:
    """Creates a detailed description prompt for identifying key concepts in a scene using keyframe descriptions and corresponding subtitles of the scene.

    :param str scene_object: The scene object containing keyframe descriptions and subtitles.
    :param str scene_subtitles: The full audio transcript of the scene.

    :rtype: str
    :returns: The prompt for generating a detailed description of the scene.
    """
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

def get_content_of_column_by_source_and_column_names(filepath, column_names: List[str]) -> dict:
    """
    Extracts the content of specified columns from a CSV file and organizes it by source filename.

    :param str filepath: The path to the CSV file.
    :column_names: The list of column names to extract content from.

    :rtype: dict
    :returns: A dictionary containing the content organized by source filename.

    """
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

def get_scene_caption_from_csv(filepath:str, column_name:str)-> List[str]:

    """
    Extracts the scene captions from a CSV file.
    
    :param str filepath: The path to the CSV file.
    :param str column_name: The name of the column containing the scene captions.
    
    :rtype: List[str]
    :returns: A list of scene captions.
    """

    descriptions = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            caption = row[column_name]
            descriptions.append(caption)
    
    return descriptions

def create_prompt_for_video_description(scenes_captions: List[str], srt_context: str) -> str:
    """
    Creates a prompt for generating a comprehensive description of a video based on key frame descriptions and subtitles.

    :param List[str] scenes_captions: A list of captions for key scenes in the video.
    :param str srt_context: The full audio transcript of the video.

    :rtype: str
    :returns: The prompt for generating a comprehensive description of the video.

    """

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

def create_lom_prompts_for_video_with_scenes_iterate(scenes_captions: List[str]) -> List[str]:

    """
    Creates a set of prompts for generating LOM metadata attributes based on scene captions.

    :param List[str] scenes_captions: A list of captions for key scenes in the video.

    :rtype: dict
    :returns: A dictionary containing the LOM attributes as keys and the corresponding prompts as values.
    
    """
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

def parse_srt(srt_content: str)-> List[str]:
    """
    Parses SRT file content and returns a list of subtitle texts.

    :param str srt_content: The content of the SRT file.

    :rtype: List[str]
    :return: A list of subtitle texts.

    """
    subtitles = []
    srt_blocks = srt_content.strip().split('\n\n')
    for block in srt_blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            subtitle_text = ' '.join(lines[2:])
            subtitles.append(subtitle_text)
    return subtitles

def save_data_to_csv(filepath: str, data:any, header: List[str]=['Source Filename', 'Caption']):
    """Saves data to a CSV file or creates a new file if it doesn't exist.

       :param str filepath: The path to the CSV file.
       :param list data: The data to be saved in the CSV file.
       :param list header: The header of the CSV file.

       :rtype: None
       :returns: None
    """
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
    """Generates captions for scenes using keyframes and subtitles.

    :param model: The language model to generate captions.
    :param tokenizer: The tokenizer for the language model.
    :param subtitles: The subtitles of the video.
    :param keyframes_path: The path to the keyframes CSV file.
    :param output_path: The path to save the generated captions.
    :param scene_path: The path to the scene CSV file.
    
    :rtype: str
    :returns: The path to the saved CSV file."""

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

def create_key_concept_for_scene_with_audio_of_scene(model:any,tokenizer:any, srt_subtitles:str,keyframes_path:str,output_path:str,scene_path:str="")->str:
    """ Generates key concepts for scenes using keyframes and subtitles

    :param any model: The language model to generate key concepts.
    :param any tokenizer: The tokenizer for the language model.
    :param str subtitles: The subtitles of the video.
    :param str keyframes_path: The path to the keyframes CSV file.
    :param str output_path: The path to save the generated key concepts.
    :param str scene_path: The path to the scene CSV file.

    :rtype: str
    :returns: The path to the saved CSV file.
    """

    if os.path.exists(output_path):
        # Delete the existing CSV file
        os.remove(output_path)

    dict_list = get_content_of_column_by_source_and_column_names(keyframes_path, ["KEY-CONCEPTS","Subtitle"])
    print(dict_list)
    subtiles_dict = search_subtitle_for_scene(srt_subtitles,scene_path)
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

def create_video_caption(model :any,tokenizer:any, srt_subtitles:str,input_path:str)-> str:
    """
    Generates captions for the video using keyframes and subtitles

    :param model: The language model to generate captions.
    :param tokenizer: The tokenizer for the language model.
    :param subtitles: The subtitles of the video.
    :param input_path: The path to the keyframes CSV file.

    :rtype: str
    :returns: The caption generated
    """

    scene_dict=get_scene_caption_from_csv(input_path, "Caption")
    prompt=create_prompt_for_video_description(scene_dict,srt_subtitles)
    print(prompt)
    encodeds =  tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = encodeds['input_ids'].shape[1]
    generated_ids = model.generate(encodeds['input_ids'],max_new_tokens=200, do_sample=True, )
    decoded = tokenizer.decode(generated_ids[0][prompt_length:],skip_special_tokens=True)
    caption = decoded.replace('\n', ' ').replace('\r', ' ')
    return caption

def create_lom_caption_with_just_scenes_List(model,tokenizer, subtitles,input_path):
    """
    Generates LOM metadata attributes for the video using the scene captions

    :param model: The language model to generate LOM metadata attributes.
    :param tokenizer: The tokenizer for the language model.
    :param subtitles: The subtitles of the video.
    :param input_path: The path to the keyframes CSV file.

    :rtype: dict
    :returns: A dictionary containing the LOM metadata attributes and their corresponding values.
    """
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

    
