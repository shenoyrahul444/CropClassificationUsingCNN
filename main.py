"""
Goal: Build a pipeline to deliver imagery to such a model

## Objectives
1. Extract and format information from raw images.

2. Prepare data for ingestion by a machine learning model.
                The model is a convolutional neural network with input dimension (224, 224, 3).
3. Create an easily reproduceable pipeline along with documenation.
4. Visualize geolocation data.

Observations:
Images are stored to their respective class folders.
"""

# todo: Normalizing Image inputs
# Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network.
# todo: Data augmentation
import numpy as np
import pandas as pd
from tqdm import tqdm   #Shows progress bar for looping status
import os
import wand
import collections.abc
from wand.image import Image
import exifread
import piexif
from random2 import shuffle
from GPSPhoto import gpsphoto

class ImageDataPipeline:
    def __init__(self,source: str,destination: str, TRAIN_TEST_VAL_PROPORTIONS: dict):
        self.__source = source
        self.__destination = destination
        self.__TRAIN_TEST_VAL_PROPORTIONS = TRAIN_TEST_VAL_PROPORTIONS
        self.__crops_geo_data = {}

    def _create_output_directories(self, crop_folder_names: list):
        outer_folders = ['train', 'test', 'validation']

        os.mkdir(self.__destination)
        for outer_folder_name in outer_folders:
            outer_folder_location = "{}\{}".format(self.__destination, outer_folder_name)
            os.mkdir(outer_folder_location)

            for inner_folder_name in crop_folder_names:
                inner_folder_location = "{}\{}".format(outer_folder_location, inner_folder_name)
                os.mkdir(inner_folder_location)

    def extract_information_from_source(self):
        ttv_props = self.__TRAIN_TEST_VAL_PROPORTIONS
        image_source_path = self.__source
        crops_geo_data = self.__crops_geo_data

        crop_folders_list = [name for name in os.listdir(image_source_path) if name != ".DS_Store"]

        # Create Train Test Val folders with subfolders with crop names
        self._create_output_directories(crop_folders_list)


        for folder_name in tqdm(crop_folders_list):
            folder_location = "{}\{}".format(image_source_path,folder_name)
            file_name_ordered = [name for name in os.listdir(folder_location) if name != ".DS_Store"]
            # print(file_names)
            file_names = file_name_ordered[:]
            shuffle(file_names)           # Does In-place random shuffling on the copy of list
            print(file_names)
            # print(os.listdir(crop_folders_path))
            # print(len(os.listdir(folder_location)))
            # print(os.listdir(folder_location))
            # print(np.random.shuffle(print(os.listdir(folder_location))))
            # for file_name in os.listdir(folder_location):
            #     if folder_name not in crops_geo_data:
            #         crops_geo_data[folder_name] = []
            #
            #     if file_name == ".DS_Store":
            #         continue
            #
            #     file_location = "{}\{}".format(folder_location,file_name)

                # Get the data from image file and return a dictionary
        #         data = gpsphoto.getGPSData(file_location)
        #         if data:        #Some Images dont have exif data
        #             crops_geo_data[folder_name].append((data['Latitude'],data['Longitude']))
        # return crops_geo_data






if __name__ == '__main__':


    #Constants
    WORKING_DIRECTORY = os.getcwd()
    IMAGE_DIRECTORY = "images"
    OUTPUT_DIRECTORY = "output"         # Train Test and Validation Images resized to (224, 224, 3) for CNN

    TRAIN_TEST_VAL_PROPORTIONS = {'train':70,'test':15,'validation':15}

    # Data Variables
    crops_geo_data = None
    source = "{}\{}".format(WORKING_DIRECTORY, IMAGE_DIRECTORY)
    destination = "{}\{}".format(WORKING_DIRECTORY, OUTPUT_DIRECTORY)

    gip = ImageDataPipeline(source,destination,TRAIN_TEST_VAL_PROPORTIONS)

    # Obejctive 1. Extract and format information from raw images.
    # Extracting Geo Location from crop images for plotting later on
    crops_geo_data = gip.extract_information_from_source()

    # Obejctive 2. Prepare data for ingestion by a machine learning model.
    """
    This involves resizing images
    """


    # print(crops_geo_data)


    # gip.extract_information()

