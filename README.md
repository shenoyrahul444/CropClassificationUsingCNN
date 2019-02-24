# CropClassificationUsingCNN

### Objectives
    1. Extract and format information from raw images.
        GeoLocation in exif
    2. Prepare data for ingestion by a machine learning model.
    3. Create an easily reproduceable pipeline along with documenation.
    4. Visualize geolocation data.

My choice for the quick prototyping is Python. 
There are many image processing libraries having power and learning-curve tradeoff.
I will be using 'Wand' package for its ease of use.

Observation:
Data consists of JPEG Images of 5 different types of crops in their respective folders.
The Images dont have a uniform aspect ratio. 

"""

Single Label Multi-Class

80/10/10

Iris

Create Batches

cv2.imresize() -> Standard
CenterCrop/RandomCrop

Batching

GPU Advantage of 

Bottle necks in training pipeline

Full Resize is better as per Micros