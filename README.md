# Licence-Plate-Detection
The code is an approach to Detect licence plate of Indian vehicles with extensive use of Machine Learning Algorithms and Image Processing
The Data-set is however not provided due to security reasons. 

The code is quite straight forward, 

c_create_dataset:
The create feature method of the package c_create_dataset accepts the path array of the folder that contains the licence plate images and random images that doesnot pertain to licence plates.
The create feature method therefore creates the complete dataset with HOG features labeling the licence plate images as 1 and non-licence plate images as 0
Note: The licence plate images are cropped images, only the licence plate part not the whole image.

d_train_n_pred_licenceplayte:
This package perform the complete algorithm by calling the methods of b_Logistic_regression. 
It performs the algorithm on the training set finds the optimal parameter theta and use the theta value on the test data set. 

e_image_segment_into_contours:
It takes all the images which contains a licence plate one by one and finds all the possible contours.
After finding the contours it calls the d_train_n_pred_licenceplate package to find the contoured image that has the licence plate.

f_main_function:
This is the main function it makes call to method of other packages and does the entire job.
