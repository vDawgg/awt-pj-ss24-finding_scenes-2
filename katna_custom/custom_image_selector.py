from __future__ import print_function

import numpy as np
import itertools  
from multiprocessing import Pool
import Katna.config as config

from Katna.image_selector import ImageSelector


class CustomImageSelector(ImageSelector):
    def __init__(self, n_processes):
        super().__init__(n_processes)

    def select_best_frames(self, input_key_frames, number_of_frames, timestamps = None):
            """[summary] Public function for Image selector class: takes list of key-frames images and number of required
            frames as input, returns list of filtered keyframes

            :param object: base class inheritance
            :type object: class:`Object`
            :param input_key_frames: list of input keyframes in list of opencv image format 
            :type input_key_frames: python list opencv images
            :param number_of_frames: Required number of images 
            :type: int   
            :return: Returns list of filtered image files 
            :rtype: python list of images
            """


            self.nb_clusters = number_of_frames

            filtered_key_frames = []
            filtered_images_list = []
            filtered_timestamps_list = []
            # Repeat until number of frames 
            min_brightness_values = np.arange(config.ImageSelector.min_brightness_value, -0.01, -self.brightness_step)
            max_brightness_values = np.arange(config.ImageSelector.max_brightness_value, 100.01, self.brightness_step)
            min_entropy_values = np.arange(config.ImageSelector.min_entropy_value, -0.01, -self.entropy_step)
            max_entropy_values = np.arange(config.ImageSelector.max_entropy_value, 10.01, self.entropy_step)

            for (min_brightness_value, max_brightness_value, min_entropy_value, max_entropy_value) in itertools.zip_longest(min_brightness_values, max_brightness_values, min_entropy_values, max_entropy_values): 
                if min_brightness_value is None:
                    min_brightness_value = 0.0
                if max_brightness_value is None:
                    max_brightness_value = 100.0
                if min_entropy_value is None:
                    min_entropy_value = 0.0
                if max_entropy_value is None:
                    max_entropy_value = 10.0
                self.min_brightness_value = min_brightness_value
                self.max_brightness_value = max_brightness_value
                self.min_entropy_value = min_entropy_value
                self.max_entropy_value = max_entropy_value
                filtered_key_frames, filtered_timestamps = self.__filter_optimum_brightness_and_contrast_images__(
                    input_key_frames, timestamps
                )

                if len(filtered_key_frames) >= number_of_frames:
                    break

            # Selecting the best images from each cluster by first preparing the clusters on basis of histograms 
            # and then selecting the best images from every cluster
            if len(filtered_key_frames) >= self.nb_clusters:
                files_clusters_index_array = self.__prepare_cluster_sets__(filtered_key_frames)
                selected_images_index = self.__get_best_images_index_from_each_cluster__(
                    filtered_key_frames, files_clusters_index_array
                )

                for index in selected_images_index:
                    img = filtered_key_frames[index]
                    timestamp = filtered_timestamps[index]
                    filtered_images_list.append(img)
                    filtered_timestamps_list.append(timestamp)
            else:
                # if number of required files are less than requested key-frames return all the files
                for img in filtered_key_frames:
                    filtered_images_list.append(img)
                for timestamp in filtered_timestamps:
                    filtered_timestamps_list.append(timestamp)

            return filtered_images_list, filtered_timestamps_list

    def __filter_optimum_brightness_and_contrast_images__(self, input_img_files, timestamps=None):
            """ Internal function for selection of given input images with following parameters :optimum brightness and contrast range ,
            returns array of image files which are in optimum brigtness and contrast/entropy range.
    
            :param object: base class inheritance
            :type object: class:`Object`
            :param files: list of input image files 
            :type files: python list of images
            :return: Returns list of filtered images  
            :rtype: python list of images 
            """

            n_files = len(input_img_files)

            # -------- calculating the brightness and entropy score by multiprocessing ------
            pool_obj = Pool(processes=self.n_processes)

            # self.pool_obj_entropy = Pool(processes=self.n_processes)
            with pool_obj:
                brightness_score = np.array(
                    pool_obj.map(self.__get_brightness_score__, input_img_files)
                )

                entropy_score = np.array(
                    pool_obj.map(self.__get_entropy_score__, input_img_files)
                )

            # -------- Check if brightness and contrast scores are in the min and max defined range ------
            brightness_ok = np.where(
                np.logical_and(
                    brightness_score > self.min_brightness_value,
                    brightness_score < self.max_brightness_value,
                ),
                True,
                False,
            )
            contrast_ok = np.where(
                np.logical_and(
                    entropy_score > self.min_entropy_value,
                    entropy_score < self.max_entropy_value,
                ),
                True,
                False,
            )

            # Returning only those images which are have good brightness and contrast score
            filtered_key_frames = [
                    input_img_files[i]
                    for i in range(n_files)
                    if brightness_ok[i] and contrast_ok[i]
                ]

            if timestamps is not None:
                filtered_timestamps = [
                    timestamps[i]
                    for i in range(n_files)
                    if brightness_ok[i] and contrast_ok[i]
                ]

            return filtered_key_frames, filtered_timestamps
