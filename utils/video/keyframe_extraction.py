import os
import glob

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from typing import Optional
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


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
    vd = Video()

    # Initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=video_keyframes_dir)

    print(f"Input video file path = {video_file_path}")

    # Extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_return, file_path=video_file_path,
        writer=diskwriter
    )


def process_all_videos_in_directory(directory: str, output_dir: str):
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
            no_of_frames_to_return=12,
            output_dir=output_dir
        )


if __name__ == "__main__":
    process_all_videos_in_directory(
        directory="videos/video_scenes",
        output_dir="videos/keyframes"
    )