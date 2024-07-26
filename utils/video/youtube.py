from utils.constants import VIDEO_DIR

import os
import yt_dlp
from pytubefix import YouTube
from typing import List, Union

class YouTubeVideo:

    def __init__(self, url: str, output_path: str = VIDEO_DIR):
        """Initializes a YouTubeVideo object.

        :param str url: The URL of the YouTube video.
        :param str output_path (optional): The directory path to save the downloaded video and subtitles. Default is VIDEO_DIR.
        """
        self.url = url
        self.output_path = output_path
        self.yt = YouTube(self.url)

    def get_first_existing_key(self, keys_to_check: List[str], my_dict: dict) -> Union[str, None]:
        """Returns the first key found in a dictionary from a list of keys.

        :param List[str] keys_to_check: A list of keys to check in the dictionary.
        :param dict my_dict: The dictionary to check.
        :rtype: Union[str, None]
        :returns: The first key found in the dictionary from the list of keys. If no key is found, returns None.
        """
        for key in keys_to_check:
            if key in my_dict:
                return key  # Return the first key found
        return None  # Return None if no key is found

    def download_video(self) -> str:
        """Downloads a YouTube video.
        :rtype: str
        :returns: The path of the downloaded video file."""
        title = ''.join(c for c in self.yt.title if c.isalnum() or c.isspace()) 
        path = os.path.join(self.output_path, title+'.mp4')
        ydl_opts = {
            "format": "mp4",
            "outtmpl": path,
        }           
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(self.url)

        return path

    def download_subtitles(self) -> Union[str, None]:
        """Fetches the subtitles of a YouTube video if available.

        :rtype: Union[str, None]
        :returns: Subtitles in SRT format as a string. If no subtitles are found, returns None.
        """
        try:
            yt = self.yt
            #yt.bypass_age_gate()
            priority_languages = ['en', 'a.en', 'de', 'a.de']
            caption_key = self.get_first_existing_key(priority_languages, yt.captions.keys())

            if not caption_key:
                print("Video has no captions in English or German")
                return None

            subtitles = yt.captions[caption_key]

            if subtitles:
                return subtitles.generate_srt_captions()

        except Exception as e:
            print(f"Error fetching subtitles: {e}")
            return None

    def get_youtube_video_title(self) -> str:
        try:
            allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            title = ''.join(e for e in self.yt.title if e in allowed_chars)

            return title
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Error retrieving video title"