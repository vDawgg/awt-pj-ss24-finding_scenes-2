from typing import List, Tuple, Union
from pytube import YouTube
from utils.video.constants import VIDEO_DIR


def get_first_existing_key(keys_to_check: List[str], my_dict: dict) -> Union[str, None]:
    for key in keys_to_check:
        if key in my_dict:
            return key  # Return the first key found
    return None  # Return None if no key is found


def download_video_and_subtitles(url: str, output_path=VIDEO_DIR) -> Union[Tuple[str, str], Tuple[str, None]]:
    """This function takes a YouTube video URL and an optional output path. It downloads the video and, if available, its subtitles in English or German.

    :param str url: The URL of the YouTube video.
    :param str output_path: (optional) The directory path to save the downloaded video. Default is './video/'.
    :rtype: Union[Tuple[str, str], Tuple[str, None]]
    :returns: A tuple containing the path of the downloaded video and, if available, the subtitles in SRT format. If no subtitles are found, the second element of the tuple is None.
     """
    try:
        yt = YouTube(url)
        yt.bypass_age_gate()
        video = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if video:
            print(f"Downloading '{yt.title}'...")
            video.download(output_path)
            print("Download completed!")

            priority_languages = ['en', 'a.en', 'de', 'a.de']
            
            caption_key = get_first_existing_key(priority_languages, yt.captions.keys())

            if not caption_key:
                print("video has no captions in english or german")
                return output_path+video.default_filename, None
        
            subtitles = yt.captions[caption_key]

            if subtitles:
                return output_path+video.default_filename, subtitles.generate_srt_captions()
        else:
            print("No video found with mp4 format.")
    except Exception as e:
        print(f"Error downloading video: {e}")


if __name__ == "__main__":
    path, subtitles = download_video_and_subtitles("https://www.youtube.com/watch?v=5C_HPTJg5ek")
    print(subtitles)

