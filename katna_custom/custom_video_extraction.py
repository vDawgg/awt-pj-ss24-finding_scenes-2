import functools
import operator
from multiprocessing import Pool

from Katna.video import Video
from Katna.decorators import FileDecorators
import Katna.config as config
import Katna.helper_functions as helper

from katna_custom.custom_frame_extractor import CustomFrameExtractor as FrameExtractor
from katna_custom.custom_image_selector import CustomImageSelector as ImageSelector


class CustomVideo(Video):
    def __init__(self):
        super().__init__()

    @FileDecorators.validate_file_path
    def extract_video_keyframes(
        self,
        no_of_frames,
        file_path,
        frame_writer,
        timestamp_writer
    ):
        """Returns a list of best key images/frames from a single video.

        :param no_of_frames: Number of key frames to be extracted
        :type no_of_frames: int, required
        :param file_path: video file location
        :type file_path: str, required
        :param writer: Writer object to process keyframe data
        :type writer: Writer, required
        :return: List of numpy.2darray Image objects
        :rtype: list
        """

        # get the video duration
        video_duration = self._get_video_duration_with_cv(file_path)

        # duration is in seconds
        if video_duration > (config.Video.video_split_threshold_in_minutes * 60):
            print("Large Video (duration = %s min), will split into smaller videos " % round(video_duration / 60))
            top_frames, top_timestamps = self.extract_video_keyframes_big_video(no_of_frames, file_path)
        else:
            top_frames, top_timestamps =  self._extract_keyframes_from_video(no_of_frames, file_path)

        frame_writer.write(file_path, top_frames)
        timestamp_writer.write(file_path, top_timestamps)

        print("Completed processing for : ", file_path)


    def _extract_keyframes_from_video(self, no_of_frames, file_path):
            """Core method to extract keyframe for a video

            :param no_of_frames: [description]
            :type no_of_frames: [type]
            :param file_path: [description]
            :type file_path: [type]
            """
            # Creating the multiprocessing pool
            self.pool_extractor = Pool(processes=self.n_processes)
            # Split the input video into chunks. Each split(video) will be stored
            # in a temp
            if not helper._check_if_valid_video(file_path):
                raise Exception("Invalid or corrupted video: " + file_path)

            # split videos in chunks in smaller chunks for parallel processing.
            chunked_videos = self._split(file_path)
            frame_extractor = FrameExtractor()

            # Passing all the clipped videos for  the frame extraction using map function of the
            # multiprocessing pool
            with self.pool_extractor:
                extracted_candidate_frames, extracted_candidate_timestamps = self.pool_extractor.map(
                    frame_extractor.extract_candidate_frames, chunked_videos
                )[0]

            self._remove_clips(chunked_videos)
            image_selector = ImageSelector(self.n_processes)

            top_frames, top_frames_timestamps = image_selector.select_best_frames(
                extracted_candidate_frames, no_of_frames, extracted_candidate_timestamps
            )

            del extracted_candidate_frames
            del extracted_candidate_timestamps

            return top_frames, top_frames_timestamps

    def extract_video_keyframes_big_video(self, no_of_frames, file_path):
        """

        :param no_of_frames:
        :type no_of_frames:
        :param file_path:
        :type file_path:
        :return:
        :rtype:
        """

        # split the videos with break point at 20 min
        video_splits = self._split_large_video(file_path)
        print("Video split complete.")

        all_top_frames_split = []
        all_top_timestamps_split = []

        # call _extract_keyframes_from_video
        for split_video_file_path in video_splits:
            top_frames_split, top_timestamps_split = self._extract_keyframes_from_video(no_of_frames, split_video_file_path)
            all_top_frames_split.append(top_frames_split)
            all_top_timestamps_split.append(top_timestamps_split)

        # collect and merge keyframes to get no_of_frames
        self._remove_clips(video_splits)
        image_selector = ImageSelector(self.n_processes)

        # list of list to 1d list
        extracted_candidate_frames = functools.reduce(operator.iconcat, all_top_frames_split, [])
        extracted_candidate_timestamps = functools.reduce(operator.iconcat, all_top_timestamps_split, [])

        # top frames with timestamps
        top_frames, top_timestamps = image_selector.select_best_frames(
            extracted_candidate_frames, no_of_frames, extracted_candidate_timestamps
        )

        return top_frames, top_timestamps