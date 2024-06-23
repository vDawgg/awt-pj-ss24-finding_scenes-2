import os
import csv
import glob
import time
from typing import Optional
from pathlib import Path
import pandas as pd


from Katna.writer import KeyFrameDiskWriter
from katna_custom.custom_writer import TimeStampDiskWriter
from katna_custom.custom_video_extraction import CustomVideo
from utils.keyframe.utils import time_string_to_milliseconds, milliseconds_to_time_string

# from utils.video.constants import VIDEO_DIR
from utils.constants import VIDEO_DIR


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
    video_name: str,
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
        input_video_dir = os.path.join(VIDEO_DIR, f"{video_name}_scenes")

        extract_keyframe(
            video_file_path=os.path.join(input_video_dir, video_file),
            output_dir=output_dir,
            no_of_frames_to_return=no_of_frames_to_return,
        )


def generate_keyframes_from_scenes(
    video_name: str,
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

    scenes_csv_input_file = Path(VIDEO_DIR) / f"{video_name}_scenes" / "scene_list.csv"
    video_files = get_column_values_from_csv(scenes_csv_input_file, "file_name")

    keyframes_dir = Path(VIDEO_DIR) / f"{video_name}_keyframes"

    os.makedirs(keyframes_dir, exist_ok=True)

    # Extract keyframes from all videos in the scene list
    process_all_videos(
        video_name=video_name,
        video_files=video_files,
        output_dir=keyframes_dir,
        no_of_frames_to_return=no_of_frames_to_return
    )

    # Merge scene csv files into one
    keyframe_csv_path = merge_csv(video_name=video_name)

    return keyframe_csv_path


def generate_csv_file_paths(
        keyframes_csv_input_dir: str = "videos/keyframes"
) -> list:
    """
    Generate a list of all CSV files in the specific keyframe folders

    Args:
    keyframes_csv_input_dir (str): The directory containing the keyframe CSV files.

    Returns:
    list: A list of all CSV files in the specific keyframe folders.
    """
    # Get a list of all CSV files in the specific keyframe folders
    csv_files = glob.glob(f"{keyframes_csv_input_dir}/*.csv")

    return csv_files


def merge_csv(
        video_name: str,
):
    """
    Creates a CSV file containing keyframe data by combining multiple CSV files and merging them with scene information.

    Args:
        keyframes_csv_input_dir (str): The directory path where the keyframe CSV files are located. Default is "videos/keyframes".
        scenes_csv_input_file (str): The file path of the scene list CSV file. Default is "videos/video_scenes/scene_list.csv".
        output_file (str): The file path of the output CSV file. Default is "videos/keyframes/extracted_keyframes.csv".

    Returns:
        str: The file path of the output CSV file.

    """

    csv_dir = Path(VIDEO_DIR) / f"{video_name}_keyframes"
    merged_csv_filepath = Path(VIDEO_DIR) / f"{video_name}_keyframes" / f"{video_name}_keyframes.csv"
    scenes_csv_input_file = Path(VIDEO_DIR) / f"{video_name}_scenes" / "scene_list.csv"

    print("Merge CSV files from:", csv_dir)

    csv_files = generate_csv_file_paths(csv_dir)

    # Initialize an empty list to hold dataframes
    dfs = []

    # Loop over the list of CSV files
    for csv_file in csv_files:
        print(f"Reading {csv_file}")
        # Read each CSV file into a DataFrame and append it to the list
        df = pd.read_csv(csv_file)
        dfs.append(df)
        # Delete the CSV file
        os.remove(csv_file)

    # Concatenate all the dataframes in the list into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort the combined dataframe based on 'Filename' column
    combined_df = combined_df.sort_values('Filename')

    # Read the scenes CSV file into a pandas DataFrame
    scenes_df = pd.read_csv(scenes_csv_input_file)

    scenes_df.rename(columns={'file_name': 'Source Filename'}, inplace=True)

    # Merge the combined DataFrame with the scenes DataFrame on the 'Filename' column
    merged_df = pd.merge(combined_df, scenes_df, on='Source Filename')

    # Calculate the global timestamp by adding the local timestamp to the scene start time
    timestring_scene = merged_df["Start Timecode"].astype(str).apply(time_string_to_milliseconds)
    merged_df["Timestamp Global (ms)"] = merged_df["Timestamp Local (ms)"] + timestring_scene
    merged_df["Timestamp Global (hh:mm:ss.SSS)"] = merged_df["Timestamp Global (ms)"].apply(milliseconds_to_time_string)

    # Select only the desired columns from the merged DataFrame and save it to a new CSV file
    final_df = merged_df[['Filename', 'Source Filename','Timestamp Local (ms)', 'Timestamp Local (hh:mm:ss.SSS)','Timestamp Global (ms)', 'Timestamp Global (hh:mm:ss.SSS)']]
    final_df.to_csv(merged_csv_filepath, index=False, columns=['Filename', 'Source Filename', 'Timestamp Local (ms)', 'Timestamp Local (hh:mm:ss.SSS)','Timestamp Global (ms)', 'Timestamp Global (hh:mm:ss.SSS)'])

    print("Merged CSV file saved at:", f"{video_name}_keyframes.csv")

    return merged_csv_filepath


if __name__ == "__main__":

    start_time = time.time()

    video_name = "Rust in 100 Seconds"

    keyframe_csv_path = generate_keyframes_from_scenes(
        video_name=video_name,
        no_of_frames_to_return=1
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")