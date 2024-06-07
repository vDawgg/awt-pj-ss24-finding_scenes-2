from typing import List




def get_metadata_from_vlm_captions(self, keys_to_check: List[str], my_dict: dict) -> Union[str, None]:
    for key in keys_to_check:
        if key in my_dict:
            return key  # Return the first key found
    return None  # Return None if no key is found

def get_metadata_from_llm_captions(self, keys_to_check: List[str], my_dict: dict) -> Union[str, None]:
    for key in keys_to_check:
        if key in my_dict:
            return key  # Return the first key found
    return None  # Return None if no key is found
        
