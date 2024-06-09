
import json
from typing import List
class SceneObject:
  
    def to_json(self):
        return json.dumps(self.__dict__)
