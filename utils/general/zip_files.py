import os
import zipfile


def zip_files(directory:str, output_zip:str)-> None:
    """Zip all files in a directory into a single zip file.
    :param str directory: The directory containing the files to zip.
    :param str output_zip: The path to the output zip file.
    :rtype: None
    :return: None
    """
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))


def unzip_files(zip_file, output_directory):
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        zipf.extractall(output_directory)


# Usage example
directory = '/home/limin/Documents/programming/finding_scenes_in_learning_videos/awt-pj-ss24-finding_scenes-2/videos/keyframes'
output_zip = './keyframes.zip'  # Replace with the desired output zip file path

zip_files(directory, output_zip)

