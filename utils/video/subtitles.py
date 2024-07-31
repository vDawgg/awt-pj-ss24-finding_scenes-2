import pysrt
import unicodedata
import pandas as pd
from typing import Union

def time_to_seconds(time_object: pysrt.SubRipItem)-> int:
    """ 
    Convert a pysrt.SubRipItem object to seconds.

    :param pysrt.SubRipItem: A pysrt.SubRipItem object representing a timestamp.
    :rtype: int
    :returns: The timestamp in seconds.
    """
    return time_object.hours * 3600 + time_object.minutes * 60 + time_object.seconds + time_object.milliseconds / 1000

def time_to_milliseconds(time_object: pysrt.SubRipItem) -> int:
    """ 
    Convert a pysrt.SubRipItem object to milliseconds.

    :param pysrt.SubRipItem: A pysrt.SubRipItem object representing a timestamp.

    :rtype: int
    :returns: The timestamp in milliseconds.
    """
    return (time_object.hours * 3600 * 1000 + time_object.minutes * 60 * 1000 + time_object.seconds * 1000 + time_object.milliseconds)


def search_subtitle(srt_string: str, timestamp_seconds: int)-> Union[str, None]:
    """Search for a subtitle by timestamp in seconds in a SRT string.

    :param str srt_string: A string containing the subtitles in SubRip (SRT) format.
    :param int timestamp_seconds: The timestamp in seconds to search for.

    :rtype: Union[str, None]
    :returns: The subtitle text if found, otherwise None.
    
    """
    subs= pysrt.from_string(srt_string)

    for sub in subs:
        start_time = time_to_milliseconds(sub.start)
        end_time = time_to_milliseconds(sub.end)
        
        if start_time <= timestamp_seconds <= end_time:
            # Remove non-printable Unicode characters
            clean_text = ''.join(c for c in sub.text if unicodedata.category(c)[0] != 'C')
            return clean_text

    return None     

def search_subtitle_for_scene(srt_string: str, csv_file: str) -> dict[str, str]:
    """Extract subtitles for each scene based on timestamps.

    :param str srt_string: A string containing the subtitles in SubRip (SRT) format.
    :param str csv_file: The file path of the CSV file containing scene timestamps.

    :rtype: dict[str, str]
    :returns: A dictionary containing the subtitles for each scene based on timestamps.
    """
    
    # Read scenes from the CSV file
    scenes_df = pd.read_csv(csv_file)
    scenes = scenes_df.to_dict('records')
    
    # Parse the subtitles
    subs = pysrt.from_string(srt_string)
    subtitles_by_scene = {}
    
    for scene in scenes:
        scene_start = scene['Start Time (seconds)'] * 1000
        scene_end = scene['End Time (seconds)'] * 1000
        scene_subtitles = []

        for sub in subs:
            start_time = time_to_milliseconds(sub.start)
            end_time = time_to_milliseconds(sub.end)
            
            if scene_start <= start_time <= scene_end or scene_start <= end_time <= scene_end:
                # Remove non-printable Unicode characters
                clean_text = ''.join(c for c in sub.text if unicodedata.category(c)[0] != 'C')
                scene_subtitles.append(clean_text)
        
        subtitles_by_scene[scene['file_name']] = '\n'.join(scene_subtitles)
    
    return subtitles_by_scene

def save_subtitle_in_csv(srt_string:str,input_path_csv:str)-> str:
   """
    Adds subtitles to the corresponding timestamps in a CSV file and saves the updated CSV.

    This function takes a string containing subtitles in SubRip (SRT) format and the file path
    of a CSV file. It matches the subtitles to the corresponding timestamps in the CSV file
    and saves the updated CSV with the added subtitle information.

    :param srt_string: A string containing the subtitles in SubRip (SRT) format.
    :param input_path_csv: The file path of the CSV file to which subtitles need to be added.
    :rtype: void
   """

   df = pd.read_csv(input_path_csv)
   subtitles_list = []

   # Iterate through each row and find the corresponding subtitle
   for _, row in df.iterrows():
     global_timestamp_ms = row['Timestamp Global (ms)']
     subtitle = search_subtitle(srt_string, global_timestamp_ms)
     subtitles_list.append(subtitle)

   # Add the subtitles list as a new column to the DataFrame
   df['Subtitle'] = subtitles_list

   # Print the updated DataFrame
   print(df)
   # Put subtitles in csv
   df.to_csv(input_path_csv, index=False)

   return input_path_csv
 

       