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
from random2 import shuffle
from GPSPhoto import gpsphoto
import cv2

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

    def _resize_and_store(self, category_type: str, folder_name: str, file_names_list: list,image_output_dimensions:tuple):
        for file_name in file_names_list:
            original_image_location = "{}\{}\{}".format(self.__source,folder_name, file_name)
            processed_image_storage_location = "{}\{}\{}\{}".format(self.__destination, category_type, folder_name, file_name)

            img = cv2.imread(original_image_location)
            imcp = img.copy()   # Creating copy for processing and storage

            # Resizing the images to Expected Dimensions for the CNN
            final_image = cv2.resize(imcp, image_output_dimensions, interpolation=cv2.INTER_AREA)

            # Writing the resized image to output location
            cv2.imwrite(processed_image_storage_location, final_image)

    """STEP 1: Extract Information from the source"""
    def extract_information_from_images(self):      # Time Complexity : O(n) where n is the number of images in all the folders
        image_source_path = self.__source
        crops_geo_data = self.__crops_geo_data

        crop_folders_list = [name for name in os.listdir(image_source_path) if name != ".DS_Store"]

        for folder_name in tqdm(crop_folders_list):
            folder_location = "{}\{}".format(image_source_path,folder_name)
            file_names_list = [name for name in os.listdir(folder_location) if name != ".DS_Store"]

            for file_name in file_names_list:
                if folder_name not in crops_geo_data:
                    crops_geo_data[folder_name] = []

                file_location = "{}\{}".format(folder_location,file_name)

                # Get the data from image file and return a dictionary
                data = gpsphoto.getGPSData(file_location)
                if data:        #Some Images dont have exif data
                    crops_geo_data[folder_name].append((data['Latitude'],data['Longitude']))
        return crops_geo_data

    """STEP 2: Process and Organize images for CNN"""
    def process_and_store_images(self, image_output_dimensions: tuple):
        ttv_props = self.__TRAIN_TEST_VAL_PROPORTIONS
        image_source_path = self.__source
        crop_folders_list = [name for name in os.listdir(image_source_path) if name != ".DS_Store"]

        # # Create Train Test Val folders with subfolders with crop names
        self._create_output_directories(crop_folders_list)

        for folder_name in tqdm(crop_folders_list):
            folder_location = "{}\{}".format(image_source_path, folder_name)
            file_names_list = [name for name in os.listdir(folder_location) if name != ".DS_Store"]

            file_names_random = file_names_list[:]
            n = len(file_names_random)
            shuffle(file_names_random)  # Does In-place random shuffling on the copy of list
            training_images_count = int(n * (ttv_props['train'] / 100))
            testing_images_count = int(n * (ttv_props['test'] / 100))


            training_images_list = file_names_random[:training_images_count]
            testing_images_list = file_names_random[training_images_count:training_images_count + testing_images_count]
            validation_images_list = file_names_random[training_images_count + testing_images_count:]

            data_categories = {
                "train": training_images_list,
                "test": testing_images_list,
                "validation": validation_images_list
            }

            for category_type, category_file_names in data_categories.items():
                self._resize_and_store(category_type, folder_name, category_file_names,image_output_dimensions)


if __name__ == '__main__':

    #Constants
    WORKING_DIRECTORY = os.getcwd()
    IMAGE_DIRECTORY = "images"
    OUTPUT_DIRECTORY = "output"         # Train Test and Validation Images resized to (224, 224, 3) for CNN
    FINAL_IMAGE_DIMENSIONS = {"height":224,"width":224,"channel":3}
    TRAIN_TEST_VALIDATION_PROPORTIONS = {'train':70,'test':15,'validation':15}

    # Data Variables
    crops_geo_data = None
    source = "{}\{}".format(WORKING_DIRECTORY, IMAGE_DIRECTORY)
    destination = "{}\{}".format(WORKING_DIRECTORY, OUTPUT_DIRECTORY)

    idp = ImageDataPipeline(source,destination,TRAIN_TEST_VALIDATION_PROPORTIONS)

    # Obejctive 1. Extract and format information from raw images.
    # Extracting Geo Location from crop images for plotting later on
    crops_geo_data = idp.extract_information_from_images()

    # Obejctive 2. Prepare data for ingestion by a machine learning model.
    # This involves resizing images. Since channgel for all images is '3', they are not altered during the processing phase
    image_output_dimensions= (FINAL_IMAGE_DIMENSIONS["height"],FINAL_IMAGE_DIMENSIONS["width"])
    idp.process_and_store_images(image_output_dimensions)

