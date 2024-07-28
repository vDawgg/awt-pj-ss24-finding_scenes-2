from pytube import YouTube


def get_youtube_video_title(url: str) -> str:
    try:
        video = YouTube(url)
        title = video.title
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        title = ''.join(e for e in title if e in allowed_chars)

        return title
    except Exception as e:
        print(f"Error retrieving video title: {e}")
        return None

