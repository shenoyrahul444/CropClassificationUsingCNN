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

class ImageDataPipeline:
    def __init__(self,source: str,destination: str):
        self.__source = source
        self.__destination = destination
        self.__crops_geo_data = {}

    def get_image_details(self,file_location):
        with Image(filename=file_location) as img:
            print("***** Printing Image Metadata *****")
            print('width =', img.width)
            print('height =', img.height)
            print('Metadata = ',img.metadata)
            # print([(k, v) for k, v in img.metadata.items()])

        # with Image(filename='1460533143_94615607b7_z.jpg') as img:
        #     print('width =', img.width)
        #     print('height =', img.height)
            # print(dir(img))

        # exif = {}
        # with Image(filename='1460533143_94615607b7_z.jpg') as image:
        #     exif.update((k[5:], v) for k, v in image.metadata.items()
        #                            if k.startswith('exif:'))
        #    print(exif)

    def extract_information_from_source(self):
        PROJECT_DATA_STRUCTURE = {}
        PROJECT_PATH = self.__source

        for folder_name in tqdm(os.listdir(PROJECT_PATH)):
            if folder_name == ".DS_Store":
                continue
            folder_location = "{}\{}".format(PROJECT_PATH,folder_name)

            for file_name in os.listdir(folder_location):
                if folder_name not in PROJECT_DATA_STRUCTURE:
                    PROJECT_DATA_STRUCTURE[folder_name] = []

                if file_name == ".DS_Store":
                    continue

                file_location = "{}\{}".format(folder_location,file_name)

                self.get_image_details(file_location)
                # print(file_name,file_location)
                # PROJECT_DATA_STRUCTURE[folder_name].append(file_name)
        # print(PROJECT_DATA_STRUCTURE)


if __name__ == '__main__':

    WORKING_DIRECTORY = os.getcwd()

    SOURCE = "{}\images".format(WORKING_DIRECTORY)
    DESTINATION = "{}\output".format(WORKING_DIRECTORY)

    gip = ImageDataPipeline(SOURCE,DESTINATION)
    gip.extract_information_from_source()

    # gip.extract_information()

