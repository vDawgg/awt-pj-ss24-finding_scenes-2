import json
from typing import List
from utils.objects.scene_object import SceneObject
from pytube import YouTube


class MetaDataObject:
    def __init__(self, url, youtube_object: YouTube, SceneObjects: List[SceneObject] = []):
        self.youtube_title: str = youtube_object.title if youtube_object else ""
        self.youtube_description: str = youtube_object.description if youtube_object else ""
        self.published_date: str = str(youtube_object.publish_date) if youtube_object else ""
        self.youtube_video_id: str = youtube_object.video_id if youtube_object else ""
        self.youtube_thumbnail_url: str = youtube_object.thumbnail_url if youtube_object else ""
        self.youtube_rating: str = youtube_object.rating if youtube_object else ""
        self.youtube_views: str = youtube_object.views if youtube_object else ""
        self.youtube_age_restricted: str = youtube_object.age_restricted if youtube_object else ""
        self.youtube_keywords: List[str] = youtube_object.keywords if youtube_object else []
        self.youtube_author: str = youtube_object.author if youtube_object else ""
        self.youtube_channel_id: str = youtube_object.channel_id if youtube_object else ""
        self.youtube_length: str = youtube_object.length if youtube_object else ""
        self.url: str = url
        self.scene_objects: List[SceneObject] =SceneObjects

    def to_json(self):
           return json.dumps(self.__dict__)
