import os
import csv

import datetime

from Katna.writer import Writer

def milliseconds_to_time_string(milliseconds):
    # Convert milliseconds to seconds
    total_seconds = milliseconds / 1000
    
    # Extract hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    
    # Format the components into hh:mm:ss.SSS format
    time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds * 1000):03d}"
    
    return time_string


class TimeStampDiskWriter(Writer):
    def __init__(self, location, file_ext=".csv"):
        self.output_dir_path = location
        self.file_ext = file_ext
        self._create_dir(location)

    def generate_output_filename(self, filepath):
        file_name_arr = []
        input_file_name = self._generate_filename_from_filepath(filepath)
        file_name_arr.append(input_file_name)
        filename = "_".join(file_name_arr)
        return filename

    def save_timestamp_data_to_disk(self, timestamps, file_name):
        file_full_path = os.path.join(self.output_dir_path, file_name + self.file_ext)
        with open(file_full_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['Filename', 'Source Filename', 'Timestamp Local', 'Timestamp Local (s)'])
            # for i, timestamp in enumerate(timestamps):
            #     writer.writerow([f"{file_name}_{i}.jpeg", f"{file_name}.mp4", timestamp, timestamp / 1000])
            writer.writerow(['Filename', 'Source Filename', 'Timestamp Local (ms)', 'Timestamp Local (hh:mm:ss.SSS)'])
            for i, timestamp in enumerate(timestamps):
                timestamp_str = milliseconds_to_time_string(timestamp)
                writer.writerow([f"{file_name}_{i}.jpeg", f"{file_name}.mp4", int(timestamp), timestamp_str])

    def write(self, filepath, data):
        output_filename = self.generate_output_filename(filepath)
        self.save_timestamp_data_to_disk(data, file_name=output_filename)
