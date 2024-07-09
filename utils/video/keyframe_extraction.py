import os
import sys
import csv
import glob
from typing import Optional

from Katna.writer import KeyFrameDiskWriter
from katna_custom.custom_writer import TimeStampDiskWriter
from katna_custom.custom_video_extraction import CustomVideo

from utils.video.create_keyframes_csv import create_keyframes_csv

# TODO currently not supporting Windows OS multiprocessing
def keyframe_extraction(
    video_file_path: str,
    no_of_frames_to_return: int = 12,
    output_dir: Optional[str] = None
) -> None:
    """
    Extracts keyframes from a video file and saves them to the specified output directory.

    Args:
        video_file_path (str): The path to the video file.
        no_of_frames_to_return (int, optional): The number of keyframes to extract. Defaults to 12.
        output_dir (str, optional): The directory to save the keyframes. If not provided, keyframes will be saved in the same directory as the video file.

    Returns:
        None
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_file_path)

    video_keyframes_dir = os.path.join(
        output_dir, 
        os.path.basename(video_file_path).split(".")[0]
    )
    os.makedirs(video_keyframes_dir, exist_ok=True)

    # Initialize video module
    vd = CustomVideo()

    # Initialize diskwriter to save data at desired location
    frame_diskwriter = KeyFrameDiskWriter(location=video_keyframes_dir)
    timestamp_diskwriter = TimeStampDiskWriter(location=video_keyframes_dir)
    
    print(f"Input video file path = {video_file_path}")

    # Extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_return,
        file_path=video_file_path,
        frame_writer=frame_diskwriter,
        timestamp_writer=timestamp_diskwriter
    )


def process_all_videos_in_csv(
    csv_file_path: str, 
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

    input_video_dir = os.path.dirname(csv_file_path)

    video_files = []
    # Read the csv file
    with open(csv_file_path, "r") as f:
        reader = csv.DictReader(f)
        video_files = [row["file_name"] for row in reader]

    # Apply keyframe_extraction to each video file
    for video_file in video_files:
        keyframe_extraction(
            video_file_path=os.path.join(input_video_dir, video_file),
            no_of_frames_to_return=no_of_frames_to_return,
            output_dir=output_dir
        )


def process_all_videos_in_directory(
        directory: str, 
        output_dir: str, 
        no_of_frames_to_return: int = 1
):
    """
    Process all videos in a directory and extract keyframes from each video.

    Args:
        directory (str): The directory containing the video files.
        output_dir (str): The directory to save the keyframes.

    Returns:
        None
    """
    # Get a list of all mp4 files in the directory
    video_files = glob.glob(os.path.join(directory, "*.mp4"))

    # Apply keyframe_extraction to each video file
    for video_file in video_files:
        keyframe_extraction(
            video_file_path=video_file,
            no_of_frames_to_return=no_of_frames_to_return,
            output_dir=output_dir
        )


def main():
    csv_file_path = sys.argv[1]
    no_of_frames_to_return = int(sys.argv[2])

    process_all_videos_in_csv(csv_file_path=csv_file_path, output_dir="./videos/keyframes", no_of_frames_to_return=no_of_frames_to_return)
    create_keyframes_csv(scenes_csv_input_file=csv_file_path)


if __name__ == "__main__":
    main()
