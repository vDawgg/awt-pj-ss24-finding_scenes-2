from typing import List, Tuple, Union
from pytube import YouTube
import os
from utils.objects.metadata_object import MetaDataObject
from utils.constants import VIDEO_DIR
import yt_dlp

class YouTubeVideo:
    def __init__(self, url: str, output_path: str = VIDEO_DIR):
        self.url = url
        self.output_path = output_path
        self.yt = YouTube(self.url)

    def get_first_existing_key(self, keys_to_check: List[str], my_dict: dict) -> Union[str, None]:
        for key in keys_to_check:
            if key in my_dict:
                return key  # Return the first key found
        return None  # Return None if no key is found

    def download_video_and_subtitles(self) -> Union[Tuple[str, str], Tuple[str, None]]:
        """Downloads a YouTube video and its subtitles if available.

        :rtype: Union[Tuple[str, str], Tuple[str, None]]
        :returns: A tuple containing the path of the downloaded video and, if available, the subtitles in SRT format. If no subtitles are found, the second element of the tuple is None.
        """
        try:
            yt = self.yt
            yt.bypass_age_gate()
            video = yt.streams.filter(progressive=True, file_extension='mp4').first()
            if video:
                title = ''.join(c for c in yt.title if c.isalnum() or c.isspace())
                print(f"Downloading '{title}'...")
                #video.download(self.output_path, filename=title +".mp4")
                print("Download completed!")

                priority_languages = ['en', 'a.en', 'de', 'a.de']
                caption_key = self.get_first_existing_key(priority_languages, yt.captions.keys())

                if not caption_key:
                    print("Video has no captions in English or German")
                    return self.output_path + title + ".mp4", None

                subtitles = yt.captions[caption_key]

                if subtitles:
                    return "", subtitles.generate_srt_captions()

            else:
                print("No video found with mp4 format.")
        except Exception as e:
            print(f"Error downloading video: {e}")

    def download_video(self) -> str:
       
       title = ''.join(c for c in self.yt.title if c.isalnum() or c.isspace()) 
       path = os.path.join(self.output_path, title+'.mp4')
       ydl_opts = {
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
            yt.bypass_age_gate()
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

if __name__ == "__main__":
    downloader = YouTubeVideo("https://www.youtube.com/watch?v=L0koqAJe4lc")            
    subtitles = downloader.download_subtitles()
    print(subtitles)

                 
    
    
