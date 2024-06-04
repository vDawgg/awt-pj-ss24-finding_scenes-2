import os
import csv
import glob
import time
from typing import Optional

from Katna.writer import KeyFrameDiskWriter
from katna_custom.custom_writer import TimeStampDiskWriter
from katna_custom.custom_video_extraction import CustomVideo

from utils.keyframe.csv_creation import create_csv
from utils.video.constants import VIDEO_DIR


def get_column_values_from_csv(
    csv_file_path: str, 
    column_name: str
) -> list:
    """
    Get all the values in a column from a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        column_name (str): The name of the column to extract the values from.

    Returns:
        list: A list of values from the specified column.
    """
    column_values = []

    with open(csv_file_path, "r") as f:
        reader = csv.DictReader(f)
        column_values = [row[column_name] for row in reader]

    return column_values


# TODO currently not supporting Windows OS multiprocessing
def extract_keyframe(
    video_file_path: str,
    output_dir: str,
    no_of_frames_to_return: int = 1,

) -> None:
    """
    Extracts keyframes from a video file and saves them to the specified output directory.

    Args:
        video_file_path (str): The path to the video file.
        output_dir (str): The directory to save the keyframes. If not provided, keyframes will be saved in the same directory as the video file.
        no_of_frames_to_return (int, optional): The number of keyframes to extract. Defaults to 1.

    Returns:
        None
    """

    # Initialize video module
    vd = CustomVideo()

    # Initialize diskwriter to save data at desired location
    frame_diskwriter = KeyFrameDiskWriter(location=output_dir)
    timestamp_diskwriter = TimeStampDiskWriter(location=output_dir)
    
    # Extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_return,
        file_path=video_file_path,
        frame_writer=frame_diskwriter,
        timestamp_writer=timestamp_diskwriter
    )


def process_all_videos(
    video_files: list,
    output_dir: str, 
    no_of_frames_to_return: int = 1
):
    """
    Process all videos in a csv file and extract keyframes from each video.

    Args:
    csv_file_path (str): The path to the csv file containing the video files.
    output_dir (str): The directory to save the keyframes.

    Returns:
    None
    """

    # Apply keyframe_extraction to each video file
    for video_file in video_files:
        input_video_dir = os.path.join(VIDEO_DIR, "video_scenes")
        extract_keyframe(
            video_file_path=os.path.join(input_video_dir, video_file),
            output_dir=output_dir,
            no_of_frames_to_return=no_of_frames_to_return,
        )


def generate_keyframes_from_scenes(
    scenes_csv_input_file: str,
    keyframes_dir: str = None,
    no_of_frames_to_return: int = 1
):
    """
    Generate keyframes from scenes in a csv file.

    Args:
        scenes_csv_input_file (str): The path to the csv file containing the scene information.
        keyframes_dir (str, optional): The directory to save the keyframes. If not provided, a directory will be created based on the video name.
        no_of_frames_to_return (int, optional): The number of keyframes to return for each scene. Defaults to 1.

    Returns:
        str: The path to the generated keyframe CSV file.
    """

    video_files = get_column_values_from_csv(scenes_csv_input_file, "file_name")

    video_name = video_files[0].split("-Scene")[0]
    
    if keyframes_dir is None:
        keyframes_dir = os.path.join(VIDEO_DIR, f"{video_name}_keyframes")

    os.makedirs(keyframes_dir, exist_ok=True)

    # Extract keyframes from all videos in the scene list
    process_all_videos(
        video_files=video_files,
        output_dir=keyframes_dir,
        no_of_frames_to_return=no_of_frames_to_return
    )

    # Create a new csv file with the keyframe information
    keyframe_csv_path = create_csv(
        keyframes_csv_input_dir=keyframes_dir,
        scenes_csv_input_file=scenes_csv_input_file,
        output_csv_filepath=os.path.join(keyframes_dir, "keyframes.csv")
    )
    
    return keyframe_csv_path


if __name__ == "__main__":

    start_time = time.time()




    keyframe_csv_path = generate_keyframes_from_scenes(
        scenes_csv_input_file="videos/video_scenes/scene_list.csv",
        no_of_frames_to_return=1
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")