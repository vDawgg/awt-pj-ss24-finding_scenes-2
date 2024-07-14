import pysrt
import unicodedata
import pandas as pd
from typing import Union
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

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

def create_video_with_subtitles(subtitles_srt: str,path: str,fontsize:int=24, font: str='Arial', color: str='yellow', debug :bool = False, bg_color:str="black")-> str:
    """
    This function takes a SRT subtitles string, a path to a video file, and optional parameters for subtitle appearance. 
    It overlays the subtitles onto the video and returns the path of the resulting video file.

    :param str subtitles_srt: A string containing the subtitles in SubRip (SRT) format.
    :param str path: The file path of the video file to which subtitles need to be added.
    :param str output_path: (optional) The directory path to save the output video file. 
    
    :param int fontsize: (optional) The font size of the subtitles. Default is 24.
    :param str font: (optional) The font family of the subtitles. Default is 'Arial'.
    :param str color: (optional) The color of the subtitles. Default is 'yellow'.
    :param bool debug: (optional) A flag indicating whether to enable debug mode. Default is False.

    :rtype: str
    :returns: The path of the output video file.
    """
    # Parse subtitles from SRT string
    subtitles=pysrt.from_string(subtitles_srt)
    video = VideoFileClip(path)     
    begin,_= path.split(".mp4")

    output_video_path = begin+'_subtitled'+".mp4"

    # Get video size
    video_size=video.size

    # List to store subtitle clips
    subtitle_clips = []

    # Iterate over subtitles and add them to the video
    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        video_width, video_height = video_size
        
        # Create TextClip for subtitle
        text_clip = TextClip(subtitle.text, fontsize=fontsize, font=font, color=color, bg_color = bg_color,size=(video_width*3/4, None), method='caption').set_start(start_time).set_duration(duration)

        # Calculate position of subtitle
        x_position_of_subtitle = 'center'
        y_position_of_subtitle = video_height* 4/ 5 
        text_position = (x_position_of_subtitle, y_position_of_subtitle )
        
        # Set position of subtitle clip
        subtitle_clips.append(text_clip.set_position(text_position))
    
    # Create final video with subtitles
    final_video = CompositeVideoClip([video] + subtitle_clips)

    # Write output video file
    final_video.write_videofile(output_video_path)

    return output_video_path

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
 

       