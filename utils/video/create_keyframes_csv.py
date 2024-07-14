import glob
import pandas as pd

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
    csv_files = glob.glob(f"{keyframes_csv_input_dir}/*/*.csv")

    return csv_files


def time_string_to_milliseconds(time_str: str)-> int:
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


def milliseconds_to_time_string(milliseconds: int)-> str:
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


def create_keyframes_csv(
    keyframes_csv_input_dir: str = "videos/keyframes",
    scenes_csv_input_file: str = "videos/video_scenes/scene_list.csv",
    output_file: str = "videos/keyframes/extracted_keyframes.csv"
)-> None:
    """
    Create a CSV file containing the keyframe data for each scene.

    :param keyframes_csv_input_dir: The directory containing the keyframe CSV files.
    :param scenes_csv_input_file: The file containing the scene data.
    :param output_file: The output CSV file to save the keyframe data.

    :rtype: None
    :returns: None
    """
    csv_files = generate_csv_file_paths(keyframes_csv_input_dir)

    # Initialize an empty list to hold dataframes
    dfs = []

    # Loop over the list of CSV files
    for csv_file in csv_files:
        # Read each CSV file into a DataFrame and append it to the list
        dfs.append(pd.read_csv(csv_file))

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
    final_df.to_csv(output_file, index=False, columns=['Filename', 'Source Filename', 'Timestamp Local (ms)', 'Timestamp Local (hh:mm:ss.SSS)','Timestamp Global (ms)', 'Timestamp Global (hh:mm:ss.SSS)'])
