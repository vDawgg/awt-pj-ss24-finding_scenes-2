import os
import tempfile
import pysrt
from datetime import timedelta

from keyframe_extraction import process_all_videos_in_csv,create_keyframes_csv

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def open_srt_from_string(srt_string: str) -> pysrt.SubRipFile:
    # Create a temporary file and write the SRT string data to it
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(srt_string)

    # Close the file to release the resources
    temp_file.close()

    # Open the temporary file with pysrt
    subtitles = pysrt.open(temp_file.name)

    # Delete the temporary file
    os.unlink(temp_file.name)

    return subtitles



def time_to_seconds(time_object: pysrt.SubRipItem)-> int:
    return time_object.hours * 3600 + time_object.minutes * 60 + time_object.seconds + time_object.milliseconds / 1000


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
    subtitles=open_srt_from_string(subtitles_srt)
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

def search_subtitle(subtitles_srt, timestamp_seconds):
    """Search for a subtitle by timestamp in seconds in an SRT file."""
    subs= open_srt_from_string(subtitles_srt)

    for sub in subs:
        start_time = time_to_seconds(sub.start)
        end_time = time_to_seconds(sub.end)
        
        if start_time <= timestamp_seconds <= end_time:
            return sub.text

    return None     


if __name__ == "__main__":
    process_all_videos_in_csv(
        csv_file_path="videos/video_scenes/scene_list.csv",
        output_dir="videos/keyframes",
        no_of_frames_to_return=1
    )

    create_keyframes_csv(
        keyframes_csv_input_dir="videos/keyframes",
        scenes_csv_input_file="videos/video_scenes/scene_list.csv",
        output_file="videos/keyframes/extracted_keyframes.csv"
    )