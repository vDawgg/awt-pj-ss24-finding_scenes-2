import os
from pathlib import Path

# VIDEO_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'videos')
VIDEO_DIR = Path(__file__).resolve().parent.parent / 'videos'

HF_TOKEN = ""
