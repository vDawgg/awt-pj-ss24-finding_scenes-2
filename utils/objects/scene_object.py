import json
class SceneObject:
    """A class to represent a scene object.

    Methods:
        to_json: Returns the scene object as a JSON string.
    """
  
    def to_json(self):
        return json.dumps(self.__dict__)
