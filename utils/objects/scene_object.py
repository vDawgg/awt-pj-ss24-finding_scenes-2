
from typing import List


class SceneObject:
    def __init__(self, actions:List[str], objetcs:List[str],
                 description:str, text_in_scene:List[str],
                 duration:int=0,scene_start:int=0,
                 scene_end:int=0):
        
        self.description = description
        self.actions : List[str] = actions
        self.objects: List[str] = objetcs
        self.text_in_scene: List[str] = text_in_scene

        self.duration= duration
        self.scene_start=scene_start
        self.scene_end=scene_end
