# CropClassificationUsingCNN

## Objectives
#### PART 1 - Data preparation and consolidation (Timeline - 12 hours)
    1. Extract and format information from raw images.
        GeoLocation in exif
    2. Prepare data for ingestion by a machine learning model.
    3. Create an easily reproduceable pipeline along with documenation.
    4. Visualize geolocation data.
#### PART 2 - Building CNN Classifier (Pending)



My choice for the quick prototyping is Python 3.6. 
There are many image processing libraries having power and learning-curve tradeoff. So choices will be made considering the project timeline.

#### Observation:
Data consists of JPEG Images of 5 different types of crops in their respective folders.
The Images dont have a uniform aspect ratio. 
Single Label Multi-Class Classification


# PROJECT SETUP GUIDELINES FOR TESTING 

        Python 3.6 is used and the dependencies are mentioned in the 'requirements.txt' file.
        Download and install Python 3.6
        
        Then in commandline/terminal 
        > python install -r requirements.txt
        
        Input images are stored in 'images' folder (Looked suitable)
        Crop Icons folder is for the Map Visualization(Crops_Plot.html) created after running the program.
        Open the file in browser after creation. The program should also create output directory with files stored as might be required by the CNN.
        Normally deep learning frameworks like Keras and TensorFlow accept the image data in this fashion.


### Project Walkthrough and thought process 

        The image data for various crops are present in jpg format with both the dimensions greater than required (224,224). So no blurring on resize mostly.
        Maintaining aspect ratio can be priority in Image Classification problems, but Resizing is also considered to good results. Cropping randomly can be tricky
        as losing information in certain cases may not be possible.

        I have gone ahead with resizing using opencv-python package as cv2, as it is quite popular for image manipulation with wide range of functionalities.
        Other packages explored were Wand as a part of ImageMagick, but cv2 was easier to setup and use.

        Extracting Geolocation is done using GPSPhoto package instead of writing custom programs. (No reinventing the wheel)
        For larger amount of images, performing parallelization on GPU's or utilizing Spark cluster for distributed process with PySpark could have been a possibility.
        Also, involving S3 could be done for storage.

        Although certain tasks could be merged to based on utilizing the individual directory traversal, however, it also looked like a requirement to keep the processes separate
        while creating the data pipeline as work is done in stages such that the stages dont effect each other. So reducing the coupling was a priority.

        The code is object oriented to oraganize, consolidate, and represent the logic in a good way.

        Random shuffling of the filenames is to remove any bias coming from sequence of photos.
        Folium is used for creating maps and plotting geolocations on a html page. Factors considered are ease of use, documentation, and learning curve.
        Other packages explored for visualization were Bokeh, matplotlib, and plotly.



# General Information Extraction Process thoughts 


 ## 1> Training data acquisition and storage
 
        The data for multi-label image classification will be images which can vary in dimensions, color-scale, format, size, etc.
        The first step would be to just store the data in their respective labelled folders representing their category/label

        Since the scale of the data size can go anywhere from a few MegaBytes to GigaBytes,Terabytes, PetaBytes  to Exabytes in some cases, it is essential to store the acquired
        data from multiple channels on a scalable cloud storage services. Also, automating the acquisition and storage of data from different sources is done using ETL and make use of
        technologies like Kafka, RabbitMQ Messaging Queues to provide a real-time publish-consume mechanism. The data can be stored in powerful Datawarehouses like Redshift and Postgrest with their cluster in place.

        Images are mostly store in Amazon S3 or EBS like Scalable services. Cost, Security, reliability, availability(varying based on policies and plans), easy of migration, future-oriented growth of service,
        and ease of accessibility are the dominant factors making this decision.

 ## 2> Dataset curation
-       
        Once the is acquired, it needs to processed and transformed to the form and in a method which is useful for analysis.
        This may vary based on kind of analyses. A batched way of feeding the data into neural networks and other deep learning algorithms is mostly the case with image and object detection use-cases
        The preparation and curation process can be optimized by using Cloud Based Powerful Distributed computing services. AWS EC2, Lambda, EMR, Elastic Beanstalk are ideal for these computation tasks
        and are ready to scale.

 ## 3> Model development
-   
        Developing the model using large amount of data to train, validate, and Test the Machine Learning algorithms is a very compute intensive task.
        Generally ML and other statistical algorithms are used based on what the use case is.

        There are great deeplearning libraries like TensorFlow, Pytorch, Theano, Keras that provide efficient frameworks to pre-process the data, set up and configure the network,
        and measure its performace. They are designed to make use of GPUs to distribute and parallelize the computations. The model is thoroughly tested to handle the load and deliver speed.
        Once the accepted threshold is satisfied, it is then moved onto production servers and integrated within products and applications.

 ## 4> Moving models to production
        After the Model is created and found to be stable and accurate, it is moved into production run on the backend where the data is fed continuously through the model and the predictions are presented to the clients.
        This is a crucial moment when the the work is put to actual client facing environment and any problem here can be a make or break deal costing significant revenue.
        The models speed and performance is monitored. Retraining the model is done timely to avoid model drift and allow to make the right decisions with time.

 ## 5> Serving predictions to the client
        Internal Clients are presented the findings using web and desktop applications.TensorBoard, Web-based Dashboards(using D3, Highcharts, etc), Tableau, PowerBI, and other visualization software can consume the performance data and create great visualiztions.

        The user facing side is able to see the results in the accuracy of predictions, quality of recommendations, etc through the user facing side of the applications.
        With a good model performing well on the data, the company can make huge profit, cut down costs, and greatly improve the quality of the product/service.
