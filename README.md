# Licence-Plate-Detection
The code is an approach to Detect/Extract licence plate of vehicles use of Machine Learning  and Image Processing techniques


## Get Started:

* You would want to start with the module **LP_Detect_main.py**. Lets go one by one 
 
   ### [Create Features](https://github.com/Sardhendu/License-Plate-Detection/blob/master/Code/Bld_FeatureCrps.py): 
   * This module is aimed to extract features from a license plate/non license plate and store the features into disk (Training Features). 
   * One could manually create a small dataset by cropping out License plates from vehicle images and small set of random images (non-license plates)
   * The module employs **Histogram of Oriented Gradients (HOG)** as a features extraction technique. In a nutshell, given an image (say 32x32x3) **HOG** would create a feature vector for any Machine Leaning model to consume. You could even experiment with simple features extraction technique such as **Edges 32x32 = 1024x1**, **Flattening the image (32x32x3) into 3072x1** and use these vectors as an input to Machine learning Models.
   
       #### Module
       * Gets license plates and non-license plate from respective directories, extract features and stores the feature vectors into the disk
            * [Sample License Plate data can be found](https://github.com/Sardhendu/License-Plate-Detection/tree/master/DataSet/Data-Files/images_train/Licence-Plate)
            * [Sample Non-License plate data can be found](https://github.com/Sardhendu/License-Plate-Detection/tree/master/DataSet/Data-Files/images_train/Not-Licence-Plate)
     
   ### [Train Model](https://github.com/Sardhendu/License-Plate-Detection/blob/master/Code/BldModel.py): 
   * From step 1 we already have our features, now all we have to do is send these features to a machine learning model to learn patterns to distinguish License plates and Non-License plates.
   * In our case, the data is not very big, so we use **Support Vector Machines** as our machine learning model. SVM's are awesome with marginal data size and are robust to overfitting.
   
        ##### Module:
        * Fetches the saved features and the corresponding label (license_plate or non-license plate) and sends it to the SVM model.
        * Stores the learned model into the disk to be used while cross validation and testing. 
            * [CSV files containing the features](https://github.com/Sardhendu/License-Plate-Detection/tree/master/DataSet/Feature-Model)
             
   ### [Cross Validation](https://github.com/Sardhendu/License-Plate-Detection/blob/master/Code/LP_Detect_main.py):
   * Now that we have a model in place we would want to validate the model to predict if a rectangular region is license plate or non-license plate
   * **Note** here we do not provide the whole image, but use manually extract license plates and random images to get a sense of the model performance
   
   ### [Extract License plate](https://github.com/Sardhendu/License-Plate-Detection/blob/master/Code/LP_Detect_main.py):
   * We have every thing set up, and a good model too. Now the aim is to extract all the license plate given a image containing vehicles. 
   * We know that a license plate has visible edges. So, we extract all region of interests **(ROI)** from a image, i.e (rectangles, circles polygon defined with intense edges).
   * We know that license plate are rectangular in shape, so we reshape/extend the **ROI** as rectangles. Note for a given image we can have 100's or **ROI** of which only 2-3 would be license plate. So we have to classify all the **ROI** using our **SVM** model.   
   * All the **ROI's** are send to the feature extractor module.
  * The generated features for each **ROI** are then send to the **SVM** classifier for classification.
  * A high probability (say >90%) indicates a **ROI** to be the license plate. 
  * Finally we stash the **ROI** that have high probability of being a license plate into the disk for manual examination.
      
  **Look HERE** to get a sense of the directory name, so that you can run the model for yourself.
      
      
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
For testing, an image containing a vehicle image with its license plate was provided. Given the image we first 
extracted all the **ROI's** and then classify each **ROI's** as a license plate or not a license plate.


##### Test Images:
<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/Foreign_cars/image_2_classify%20(9).jpg" width="400" height="300"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/Foreign_cars/image_2_classify%20(4).jpg" 
width="400" height="300"> 

##### Contours Extracted (Region of Interest)  
<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/contoured_images_roi/roi_images0038.jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/contoured_images_roi/roi_images0039.jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/contoured_images_roi/roi_images0041.jpg" width="200" height="100">

##### Region of interest classified as License Plate
<img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/extracted_licenceplate_image/roi_images0042_33.jpg" width="200" height="100"> <img src="https://github.com/Sardhendu/License-Plate-Detection/blob/master/DataSet/Data-Files/images_classify/extracted_licenceplate_image/roi_images0036_28.jpg" width="200" height="100">

A sample of Dataset is provided: Look under the directory folder "/DataSet" to get a sense of the dataset.


  
