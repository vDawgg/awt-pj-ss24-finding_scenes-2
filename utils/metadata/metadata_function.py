import csv
from typing import List, Union
from utils.objects.scene_object import SceneObject




def get_metadata_from_scene_file(path_to_scene_csv: str) -> List[SceneObject]:
    scene_objects = []
    with open(path_to_scene_csv, "r") as f:
        reader = csv.DictReader(f)
        for scene in reader:
           scene_objects.append(SceneObject(
            duration=scene['Length (seconds)'],
            scene_start=scene['Start Time (seconds)'],
            scene_end=scene['End Time (seconds)'],
            title=scene['file_name'],
        ))
    return scene_objects

def get_metadata_from_keyframe_file(path_to_keyframes_csv: str, scene_objects: List[SceneObject]) -> List[SceneObject]:
    with open(path_to_keyframes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for scene_object in scene_objects:
                if row['Source Filename'] == scene_object.title:
                    scene_object.language = row['LANGUAGE']
                    scene_object.objects.append(row['INFORMATION'])
                    scene_object.actions.append(row['VIDEO_TYPE'])
                    scene_object.captions.append(row['CAPTION'])

    return scene_objects
        
