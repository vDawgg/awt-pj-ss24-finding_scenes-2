import cv2
import numpy as np
from scipy.signal import argrelextrema

from Katna.frame_extractor import FrameExtractor


class CustomFrameExtractor(FrameExtractor):
    def __init__(self):
        super().__init__()

    def __calculate_frame_difference(self, curr_frame, prev_frame):
        """Function to calculate the difference between current frame and previous frame

        :param frame: frame from the video
        :type frame: numpy array
        :param curr_frame: current frame from the video in LUV format
        :type curr_frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :return: difference count and frame if None is empty or undefined else None
        :rtype: tuple
        """

        if curr_frame is not None and prev_frame is not None:
            # Calculating difference between current and previous frame
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            return count
        return None

    def extract_candidate_frames(self, videopath):
        """Extracts candidate key-frames from an input video.

        This function takes an input video path and returns a list of candidate key-frames.

        :param videopath: The path of the input video.
        :type videopath: str
        :return: A tuple containing a list of OpenCV Image objects representing the candidate key-frames and a list of corresponding timestamps.
        :rtype: tuple
        """

        extracted_candidate_key_frames = []
        extracted_candidate_timestamps = []

        # Get all frames from video in chunks using python Generators
        frame_extractor_from_video_generator = self.__extract_all_frames_from_video__(
            videopath
        )

        # Loop over every frame in the frame extractor generator object and calculate the
        # local maxima of frames
        for frames, frame_diffs, timestamps in frame_extractor_from_video_generator:

            extracted_candidate_key_frames_chunk = []
            extracted_candidate_timestamps_chunk = []
            if self.USE_LOCAL_MAXIMA:

                # Getting the frame with maximum frame difference
                extracted_candidate_key_frames_chunk, extracted_candidate_timestamps_chunk = self.__get_frames_in_local_maxima__(
                    frames, frame_diffs, timestamps
                )
                extracted_candidate_key_frames.extend(
                    extracted_candidate_key_frames_chunk
                )
                extracted_candidate_timestamps.extend(
                    extracted_candidate_timestamps_chunk
                )

        return extracted_candidate_key_frames, extracted_candidate_timestamps

    def __get_frames_in_local_maxima__(self, frames, frame_diffs, timestamps):
        """ Internal function for getting local maxima of key frames 
        This function returns a single image with the strongest change from its vicinity of frames 
        (vicinity defined using window length)

        :param frames: list of frames to do local maxima on
        :type frames: list of images
        :param frame_diffs: list of frame difference values 
        :type frame_diffs: list of images
        :param timestamps: list of timestamps for the frames
        :type timestamps: list of floats
        :return: extracted key frames and their timestamps
        :rtype: tuple (list of images, list of floats)
        """
        extracted_key_frames = []
        extracted_timestamps = []
        diff_array = np.array(frame_diffs)
        # Normalizing the frame differences based on window parameters
        sm_diff_array = self.__smooth__(diff_array, self.len_window)

        # Get the indexes of those frames which have maximum differences
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

        for frame_index in frame_indexes:
            extracted_key_frames.append(frames[frame_index - 1])
            extracted_timestamps.append(timestamps[frame_index - 1])

        del frames[:]
        del sm_diff_array
        del diff_array
        del frame_diffs[:]
        del timestamps[:]

        return extracted_key_frames, extracted_timestamps

    def __process_frame(self, frame, prev_frame, frame_diffs, frames, timestamp, timestamps):
        """Function to calculate the difference between current frame and previous frame

        :param frame: frame from the video
        :type frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :param frame_diffs: list of frame differences
        :type frame_diffs: list of int
        :param frames: list of frames
        :type frames: list of numpy array
        :return: previous frame and current frame
        :rtype: tuple
        """

        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        # Calculating the frame difference for previous and current frame
        frame_diff = self.__calculate_frame_difference(curr_frame, prev_frame)
        
        if frame_diff is not None:
            #count, frame = frame_diff
            frame_diffs.append(frame_diff)
            frames.append(frame)
            timestamps.append(timestamp)
        del prev_frame
        prev_frame = curr_frame
        
        return prev_frame, curr_frame


    def __extract_all_frames_from_video__(self, videopath):
        """Generator function for extracting frames from a input video which are sufficiently different from each other, 
        and return result back as list of opencv images in memory

        :param videopath: inputvideo path
        :type videopath: `str`
        :return: Generator with extracted frames in max_process_frames chunks and difference between frames
        :rtype: generator object with content of type [numpy.ndarray, numpy.ndarray] 
        """
        cap = cv2.VideoCapture(str(videopath))

        ret, frame = cap.read()
        i = 1
        chunk_no = 0
        
        while ret:
            curr_frame = None
            prev_frame = None

            frame_diffs = []
            frames = []
            timestamps = []  # List to store timestamps
            for _ in range(0, self.max_frames_in_chunk):
                if ret:
                    # Get the timestamp of the current frame
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                    # Calling process frame function to calculate the frame difference and adding the difference 
                    # in **frame_diffs** list and frame to **frames** list
                    prev_frame, curr_frame = self.__process_frame(frame, prev_frame, frame_diffs, frames, timestamp, timestamps)
                    i = i + 1
                    ret, frame = cap.read()
                    # print(frame_count)
                else:
                    cap.release()
                    break
            chunk_no = chunk_no + 1


            yield frames, frame_diffs, timestamps  # Yield timestamps along with frames and differences
        cap.release()
