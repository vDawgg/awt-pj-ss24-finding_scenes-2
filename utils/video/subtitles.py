import pysrt
from typing import Union
import pandas as pd
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from video_extraction_with_pytube import YouTubeVideo


def time_to_seconds(time_object: pysrt.SubRipItem)-> int:
    return time_object.hours * 3600 + time_object.minutes * 60 + time_object.seconds + time_object.milliseconds / 1000

def time_to_milliseconds(time_object: pysrt.SubRipItem) -> int:
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
    """Search for a subtitle by timestamp in seconds in a SRT string ."""
    subs= pysrt.from_string(srt_string)

    for sub in subs:
        start_time = time_to_milliseconds(sub.start)
        end_time = time_to_milliseconds(sub.end)
        
        if start_time <= timestamp_seconds <= end_time:
            return sub.text

    return None     

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
   for index, row in df.iterrows():
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

    
if __name__ == "__main__":
   downloader = YouTubeVideo("https://www.youtube.com/watch?v=2s6mIboARCM")
   path, subtitles = downloader.download_video_and_subtitles()
   # Scene extraction
   # Keyframe extraction
   save_subtitle_in_csv(subtitles,"videos/keyframes/extracted_keyframes.csv")
   df = pd.read_csv("videos/keyframes/extracted_keyframes.csv")
   for index, row in df.iterrows():
     global_timestamp_ms = row['Timestamp Global (ms)']
     global_timestamp_hhmmss = row['Timestamp Global (hh:mm:ss.SSS)']
     filename = row['Filename']
     subtitle=search_subtitle(subtitles,global_timestamp_ms)
     print(f"Row {index}:")
     print(f"  Filename: {filename}")
     print(f"  Global Timestamp (ms): {global_timestamp_ms}")
     print(f"  Global Timestamp (hh:mm:ss.SSS): {global_timestamp_hhmmss}")
     print(f"  Subtitle: {subtitle}\n")
       