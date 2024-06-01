from scenedetect import (detect, ContentDetector, AdaptiveDetector, split_video_ffmpeg, open_video)
from scenedetect.scene_manager import SceneManager, write_scene_list
import os
from pathlib import Path
from utils.video.constants import VIDEO_DIR
import pandas as pd


def get_scenes(video_path: str) -> str:
    """
    This function detects scenes from a given video file and returns the directory the scenes and scene_list.csv
    are written to.

    :param video_path: path to the video file
    :return: directory the scenes are written to
    """
    print("Starting scene extraction")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # Create scene dir
    scene_dir = os.path.join(VIDEO_DIR, f'{video.name}_scenes')
    Path(scene_dir).mkdir(exist_ok=True)

    # Split video into scenes and write to scene_dir
    split_video_ffmpeg(video_path, scene_list, Path(scene_dir), show_progress=True)
    scene_file_list = sorted([file for file in os.listdir(scene_dir) if file.endswith('.mp4')])
    
    # Create scene_list.csv, which contains the scene start time, end time, duration and file_name
    write_scene_list(open(os.path.join(scene_dir, 'scene_list.csv'), 'w'), scene_list, False)

    # Add file_name column to scene_list.csv
    scene_list_csv = os.path.join(scene_dir, 'scene_list.csv')
    df = pd.read_csv(scene_list_csv)
    df['file_name'] = scene_file_list
    df.to_csv(scene_list_csv, index=False)

    print("Done\n")
    return scene_list_csv


if __name__ == '__main__':
    get_scenes(os.path.join(VIDEO_DIR, 'Rust in 100 Seconds.mp4'))
