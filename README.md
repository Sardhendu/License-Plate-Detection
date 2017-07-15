# Licence-Plate-Detection
The code is an approach to Detect licence plate of vehicles with use of Machine Learning Algorithms and Image Processing techniques

A sample of Dataset is provided: Look under the directory folder to get a sense of the dataset.


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
  
