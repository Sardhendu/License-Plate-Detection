#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     08/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

'''
About: This code snippet prepares the features. It goes to the file where all the pictures of licence plates and non licence plate are stashed, it them gets each picture one by one
       , It binarizes the images, resize them, perform morphological operation. Then it calls the HOG method creates the histogram of gradient for the each image which in our
       case is the feature to our logistic regression classifier.
       This code snippet also prepares the class 1 or 0. 1 for the licence plate images and 0 for the non licence plate images.
'''
import sys
sys.path.insert(0, 'C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing')
import imutils

import glob
import cv2
import numpy as np
import mahotas.thresholding
import gradient_calculation
import a_HOG_computation
#import hist_features_HOG




def create_features(path):
        # We first instantiate HOG
        hog= a_HOG_computation.HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(3,3), normalize=True)
        feature_arr=[]
        class_arr=[]
        for folder in path:
            #print folder
            for file in glob.glob(folder):
                    #print file
                    image_orig=cv2.imread(file)

                    # Convert the coloured image into grayscale image
                    image_resized=imutils.resize(image_orig, width=300)
                    image_gray=cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

                    # Do thresholding
                    image_thresh=image_gray
                    T=mahotas.thresholding.otsu(image_gray) # will find an optimal value of T from the image
                    image_thresh[image_thresh>T]=255 # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is white
                    image_thresh[image_thresh<T]=0   # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is Black
                    image_thresh = cv2.bitwise_not(image_thresh)


                    # Perform dialation (Morphological operation)
                    kernel = np.ones((2,2),np.uint8)
                    image_eroded = cv2.erode(image_thresh,kernel,iterations = 1)


                    # We resize the numberplate Since we are using only cars we would resize it to 60*120 pixels
                    image_resized= imutils.resize(image_eroded,width=90, height=30)


                    # We use HOG for licence plate detection
                    hist=hog.describe(image_resized)

                    '''# The below loop will actually create my dataset
                    list_train_data=[]
                    total_magnitude=0
                    for x_axis in range(0,image_resized.shape[0]):
                        for y_axis in range(0,image_resized.shape[1]):
                            # Here we will calculate gradient for each pixels
                            total_magnitude=total_magnitude+gradient_calculation.call_magnitude(image_resized, x_axis, y_axis)
                            list_train_data.append(image_resized[x_axis][y_axis])

                    print file, "    ", total_magnitude
                    list_train_data.append(total_magnitude)
                    feature_arr.append(list_train_data)
                    '''
                    feature_arr.append(hist)


                    # We prepare the class for the photos that are actual licence plate as 1 and non licence plate as 0
                    if folder==path[0]:
                        class_arr.append([1])
                    else:
                        class_arr.append([0])


        return feature_arr,class_arr
