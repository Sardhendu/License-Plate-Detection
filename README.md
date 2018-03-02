# Licence-Plate-Detection
The code is an approach to Detect licence plate of vehicles with use of Machine Learning Algorithms and Image Processing techniques


## Get Started:

* You would want to start with the module **LP_Detect_main.py**. Let go one by one 
 
   #### create_feature_matrix: 
   * Calls a function of the class **CrtFeatures**: inside the module **Bld_FeatureCrps**. This module is aimed to extract features from a license plate/non license plate and store the features into disk (Training Features). It basically uses two directories 
        * Train_data1 : Manually extracted license plates (cropped form vehicles image).
        * Train_data2 : Some random images (Non License plate)
     -- see config-local.conf file to get to know the directories name
     
   #### train_model: 
   * Makes call to function of class **Model** inside module **BldModel**. Now that we have extracted training features as discussed above, we would want algorithm to learn the features of a license plate and a non-license plate. This is achieved by train_model.
        * It fetches the saved features and the corresponding label (license_plate or non-license plate) and sends it to the SVM model.
        * It also stores the learned model into the disk so that while cross validation and testing we can invoke the model and classify. 
             
   #### valid and run_cross_valid:
   * These function makes use of the above modules to extract features of a manually cropped license plate/non-license plate and use the saved model to classify the images.
   
   #### Extract_lisenceplate:
   * This is the most important function that is provided with the actual directory where your images (vehicles/non-vehicles) are provided. 
      * It extracts all contours (rectangles, circles polygon defined with intense edges), applies some morphological operations and
      * Then the extracted contours are send to the feature extractor where features for each contours are extracted.
      * These features are then classified as license-plate and non-license plate.
      * A high probability indicates a contour to be the license plate. 
      * Finally the contoured classified as a license plates (high probability) are stashed in a directory.
      
      -- Look at the "config-local.conf" to get a understanding of the directory name.
      
      
## Snapshot of the process

#### Train: 
Training data is manually created by taking a vehicle's image and cropping out the license plate image. Another 
simple way would be to extract all possible contours form the image (look at /Code/Classify.py) and save them in a 
directory. Then manually go an select the license plates. Below are some images of License plates used for Training.

##### License Plates:

<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_train/Licence-Plate/yes%20(33).jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_train/Licence-Plate/yes%20(32).jpg" width="200" height="100"> 

##### Non License Plates

<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_train/Not-Licence-Plate/no%20(6).jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_train/Not-Licence-Plate/no%20(8).jpg" width="200" height="100"> 


#### Test:
For Testing, an image containing a vehicle image with its license plate was provides. Given this image we first 
extracted all the contours and then classify each contours as a license plate or not a license plate.


##### Test Images:
<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify
/Foreign_cars/image_2_classify (9).jpg" width="400" height="300"> <img src="https://github
.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/Foreign_cars/image_2_classify (4).jpg" 
width="400" height="300"> 

##### Contours Extracted (Region of Interest)  
<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/contoured_images_roi/roi_images0038.jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/contoured_images_roi/roi_images0036.jpg" 
width="200" height="100"> <img src="https://github
.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify
/contoured_images_roi/roi_images0041.jpg" width="200" height="100">

##### Region of interest classified as License Plate
<img src="https://github
.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify
/contoured_images_roi/roi_images0041.jpg" width="200" height="100"> <img src="https://github
.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify
/contoured_images_roi/roi_images0036_28" width="200" height="100">

A sample of Dataset is provided: Look under the directory folder "/DataSet" to get a sense of the dataset.


  
