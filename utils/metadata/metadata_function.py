from collections import namedtuple
import csv
from typing import List, Union
from utils.objects.scene_object import SceneObject


def get_metadata_from_scene_file(path_to_scene_csv: str) -> List[SceneObject]:
    scene_objects = []
    with open(path_to_scene_csv, "r") as f:
        reader = csv.DictReader(f)
        for scene in reader:
            scene_object=SceneObject()
            scene_object.__setattr__("duration", scene['Length (seconds)'])
            scene_object.__setattr__("scene_start", scene['Start Time (seconds)'])
            scene_object.__setattr__("scene_end", scene['End Time (seconds)'])
            scene_object.__setattr__("title", scene['file_name'])
            scene_objects.append(scene_object)
            
    return scene_objects

def get_metadata_from_keyframe_file(path_to_keyframes_csv: str, scene_objects: List[SceneObject], tasks:dict) -> List[SceneObject]:
    with open(path_to_keyframes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for scene_object in scene_objects:
                if row['Source Filename'] == scene_object.title:
                    for task in tasks.keys():
                        scene_object.__setattr__(task.lower().strip().replace(' ', '_').replace('(','').replace(')',''), row[task])
    return scene_objects

def set_new_content_for_metadata_attribute_for_sceneobjects(path_to_keyframes_csv: str, scene_objects: List[SceneObject], attribute) -> List[SceneObject]:
    with open(path_to_keyframes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for scene_object in scene_objects:
                if row['Source Filename'] == scene_object.title:
                    setattr(scene_object, attribute.lower(), row[attribute])
    return scene_objects
       
