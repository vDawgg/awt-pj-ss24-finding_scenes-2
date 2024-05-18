import glob
import pandas as pd
import datetime

def create_keyframes_csv(
    keyframes_csv_input_dir: str = "videos/keyframes",
    scenes_csv_input_file: str = "videos/video_scenes/scene_list.csv",
    output_file: str = "videos/keyframes/extracted_keyframes.csv"
):
        
    # Get a list of all CSV files in the specific keyframe folders
    csv_files = glob.glob(f"{keyframes_csv_input_dir}/*/*.csv")

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

    # Reset the index of the sorted dataframe
    combined_df = combined_df.reset_index(drop=True)

    # Write the sorted dataframe with reset index to a new CSV file in the keyframe directory
    # combined_df.to_csv('videos/keyframes/extracted_keyframes.csv')

    # Function to convert timecode to seconds
    def timecode_to_seconds(timecode):
        h, m, s = map(float, timecode.split(':'))
        return datetime.timedelta(hours=h, minutes=m, seconds=s).total_seconds()

    # Read the scenes CSV file into a pandas DataFrame
    df1 = pd.read_csv(scenes_csv_input_file)

    # Convert 'Start Timecode' to seconds
    df1['Start Time (seconds)'] = df1['Start Timecode'].apply(timecode_to_seconds)
    df1.rename(columns={'file_name': 'Origin Filename'}, inplace=True)
    # Merge the combined DataFrame with the scenes DataFrame on the 'Filename' column
    merged_df = pd.merge(combined_df, df1, on='Origin Filename')

    # Calculate the global timestamp
    merged_df['Global Timestamp (s)'] = merged_df['Start Time (seconds)'] + merged_df['Timestamp Local (s)']

    # Select only the desired columns from the merged DataFrame
    final_df = merged_df[['Index', 'Filename', 'Origin Filename', 'Timestamp Local', 'Timestamp Local (s)', 'Global Timestamp (s)','Start Time (seconds)']]
    # Write the final DataFrame to a new CSV file
    final_df.to_csv(output_file, index=False, columns=['Filename', 'Origin Filename', 'Timestamp Local', 'Timestamp Local (s)', 'Global Timestamp (s)', 'Start Time (seconds)'])