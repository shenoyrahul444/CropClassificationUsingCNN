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

from tqdm import tqdm   #Shows progress bar for looping status
import os
import wand
import collections.abc
from wand.image import Image
import exifread
import piexif

from GPSPhoto import gpsphoto

class ImageDataPipeline:
    def __init__(self,source: str,destination: str):
        self.__source = source
        self.__destination = destination
        self.__crops_geo_data = {}


    def extract_information_from_source(self):
        crops_geo_data = self.__crops_geo_data
        PROJECT_PATH = self.__source

        for folder_name in tqdm(os.listdir(PROJECT_PATH)):
            if folder_name == ".DS_Store":
                continue
            folder_location = "{}\{}".format(PROJECT_PATH,folder_name)

            for file_name in os.listdir(folder_location):
                if folder_name not in crops_geo_data:
                    crops_geo_data[folder_name] = []

                if file_name == ".DS_Store":
                    continue

                file_location = "{}\{}".format(folder_location,file_name)

                # Get the data from image file and return a dictionary
                data = gpsphoto.getGPSData(file_location)
                if data:
                    crops_geo_data[folder_name].append((data['Latitude'],data['Longitude']))
        return crops_geo_data


if __name__ == '__main__':


    #Constants
    WORKING_DIRECTORY = os.getcwd()
    IMAGE_DIRECTORY = "images"
    OUTPUT_DIRECTORY = "output"
    source = "{}\{}".format(WORKING_DIRECTORY,IMAGE_DIRECTORY)
    destination = "{}\{}".format(WORKING_DIRECTORY,OUTPUT_DIRECTORY)

    # Data Variables
    crops_geo_data = None

    gip = ImageDataPipeline(source,destination)

    # Obejctive 1. Extract and format information from raw images.
    # Extracting Geo Location from crop images for plotting later on
    crops_geo_data = gip.extract_information_from_source()

    # Obejctive 2. Prepare data for ingestion by a machine learning model.
    """
    This involves resizing images
    """


    print(crops_geo_data)


    # gip.extract_information()

