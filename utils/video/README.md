## Installation

To ensure compatibility and avoid unresolved issues present in the standard version of `pytube`, it's recommended to install the library directly from the GitHub repository of the fork that contains fixes.

```bash
python -m pip install git+https://github.com/Rasmy000/pytube.git
```

By installing `pytube` from this GitHub repository, you'll be using a forked version that has addressed some bugs and may contain improvements not yet merged into the main release.

## Usage

After installing the forked version of `pytube`, you can use the `download_video_and_subtitles` function from the `video_extraction_with_pytube.py` file to download videos from YouTube along with their subtitles.

```python
from video_extraction_with_pytube import download_video_and_subtitles

# Example usage
url = "https://www.youtube.com/watch?v=your_video_id"
output_path = "./videos/"
video_path, subtitles = download_video_and_subtitles(url, output_path)

if subtitles:
    with open("subtitles.srt", "w", encoding="utf-8") as f:
        f.write(subtitles)
    print("Subtitles downloaded successfully.")
else:
    print("No subtitles found.")
```

Make sure to replace `"your_video_id"` with the ID of the YouTube video you want to download.

## Requirements

- Python 3.x
- `pytube` package from the forked GitHub repository
