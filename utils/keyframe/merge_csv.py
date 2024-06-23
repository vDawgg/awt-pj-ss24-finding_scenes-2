import os
import glob

import pandas as pd
from pathlib import Path

from utils.constants import VIDEO_DIR

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


def time_string_to_milliseconds(time_str):
    """
    Converts a time string in the format 'hh:mm:ss.sss' to milliseconds.

    Parameters:
    time_str (str): The time string to be converted.

    Returns:
    int: The total number of milliseconds.

    Example:
    >>> time_string_to_milliseconds('01:23:45.678')
    5025678
    """
    # Split the time string into hours, minutes, seconds, and milliseconds
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split('.')
    
    # Convert each component to milliseconds and sum them up
    total_milliseconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)
    
    return total_milliseconds


def milliseconds_to_time_string(milliseconds):
    """
    Converts milliseconds to a formatted time string in the format hh:mm:ss.SSS.

    Args:
        milliseconds (int): The number of milliseconds to convert.

    Returns:
        str: The formatted time string in the format hh:mm:ss.SSS.
    """
    # Convert milliseconds to seconds
    total_seconds = milliseconds / 1000
    
    # Extract hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    
    # Format the components into hh:mm:ss.SSS format
    time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds * 1000):03d}"
    
    return time_string


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

    csv_file_paths = Path(VIDEO_DIR) / f"{video_name}_keyframes"
    merged_csv_filepath = Path(VIDEO_DIR) / f"{video_name}_keyframes" / f"{video_name}_keyframes.csv"
    scenes_csv_input_file = Path(VIDEO_DIR) / f"{video_name}_scenes" / "scene_list.csv"

    csv_files = generate_csv_file_paths(csv_file_paths)

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

    return merged_csv_filepath
