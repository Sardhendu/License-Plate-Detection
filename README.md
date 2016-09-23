# Licence-Plate-Detection
The code is an approach to Detect licence plate of Indian vehicles with extensive use of Machine Learning Algorithms and Image Processing
The Data-set is however not provided due to security reasons. 

Below are list of important functions.

Bld_FeatureCrps:
This file does the feature extraction from the license plate images. The paths of stashed images (both license plate and non-license plate) are provided, the code extractes the features from the images and store each image as a set of feature in the disk with their respective label. In short the code attempts to create the training sample.
Note: The training and test sample of licence plate images are manually cropped images, only the licence plate part not the whole image of the vehicle.

Bld_Model:
This file contains set of models on which the training data is trained. The trained models are then stored in the disk for the crossvaldation and test data.

Classfy:
This module takes input an image, use set of morphological operation, extracts all the contours from an image and then classifies all the rectangles. The rectangle classified with a high threshold are identified as license plate 
