import csv
from typing import List
from utils.objects.scene_object import SceneObject

from utils.general.word_counter import most_frequent_noun

def get_metadata_from_scene_file(path_to_scene_csv: str) -> List[SceneObject]:
    """ 
     This function reads the scene csv file and extracts the metadata for each scene object.
     :param path_to_scene_csv: The path to the scene csv file.
     :rtype: List[SceneObject]
     :returns: A list of SceneObject objects with the metadata extracted from the scene
    """
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
    """ This function reads the keyframes csv file and extracts the metadata for each scene object.

     :param path_to_keyframes_csv: The path to the keyframes csv file.
     :param scene_objects: A list of SceneObject objects.
     :param tasks: A dictionary of tasks and their corresponding attributes.

     :rtype: List[SceneObject]
     :returns: A list of SceneObject objects with the metadata extracted from the keyframes csv file."""
    
    with open(path_to_keyframes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for scene_object in scene_objects:
                if row['Source Filename'] == scene_object.title:
                    for task in tasks.keys():
                        attr_name = task.lower().strip().replace(' ', '_').replace('(','').replace(')','')
                        attr_value = row[task]
                        if task in ["QUESTIONS", "TEXT"]:
                        # Check if the attribute already exists
                           if hasattr(scene_object, attr_name):
                            # Get the existing attribute value (which should be a list) and append the new value
                            existing_value = getattr(scene_object, attr_name)
                            setattr(scene_object, attr_name, attr_value+","+existing_value)
                           else:
                            # Create a new attribute with the value as a list
                            setattr(scene_object, attr_name, attr_value)
                        elif task in["LANGUAGE", "VIDEO_TYPE"]:
                           if hasattr(scene_object, attr_name):
                            # Get the existing attribute value (which should be a list) and append the new value
                              existing_value = getattr(scene_object, attr_name)
                              setattr(scene_object, attr_name, attr_value+" "+existing_value)
                           else:
                            # Create a new attribute with the value as a list
                              setattr(scene_object, attr_name, attr_value)
                        else:
                            setattr(scene_object, attr_name, attr_value)
        get_attribute_by_frequency(scene_objects, "video_type")
        get_attribute_by_frequency(scene_objects, "language")
        
    return scene_objects

def get_attribute_by_frequency(scene_objects: List[SceneObject], attribute: str) -> List[SceneObject]:
    """ This function gets the most frequent noun from a list of strings and sets it as the attribute value for each SceneObject in the list.

     :param scene_objects: A list of SceneObject objects.
     :param attribute: The attribute to set.

     :rtype: List[SceneObject]
     :returns: A list of SceneObject objects with the attribute set to the most frequent noun."""
    for scene_object in scene_objects:
        if hasattr(scene_object, attribute):
            setattr(scene_object, attribute, most_frequent_noun(getattr(scene_object, attribute)))
    return scene_objects

def set_new_content_for_metadata_attribute_for_scene_objects(path_to_keyframes_csv: str, scene_objects: List[SceneObject], attribute:str) -> List[SceneObject]:
    """ This function reads the keyframes csv file and extracts the metadata for each scene object.

     :param path_to_keyframes_csv: The path to the keyframes csv file.
     :param scene_objects: A list of SceneObject objects.
     :param attribute: The attribute to set.

     :rtype: List[SceneObject]
     :returns: A list of SceneObject objects with the metadata extracted from the keyframes csv file.
    """
    with open(path_to_keyframes_csv, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for scene_object in scene_objects:
                if row['Source Filename'] == scene_object.title:
                    setattr(scene_object, attribute.lower(), row[attribute])
    return scene_objects
       
