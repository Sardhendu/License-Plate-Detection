#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     11/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import mahotas
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing')
import imutils
import HOG_computation


def create_histogram_for_image(image):
    hog= HOG_computation.HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(3,3), normalize=True)

    # Convert the coloured image into grayscale image
    image_resized=imutils.resize(image, width=300)
    image_gray=cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Do thresholding
    image_thresh=image_gray
    T=mahotas.thresholding.otsu(image_gray) # will find an optimal value of T from the image
    image_thresh[image_thresh>T]=255 # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is white
    image_thresh[image_thresh<T]=0   # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is Black
    image_thresh = cv2.bitwise_not(image_thresh)


    # Perform dialation
    kernel = np.ones((2,2),np.uint8)
    image_eroded = cv2.erode(image_thresh,kernel,iterations = 1)


    # We resize the numberplate Since we are using only cars we would resize it to 60*120 pixels
    image_resized= imutils.resize(image_eroded,width=90, height=30)


    # We use HOG for licence plate detection
    hist=hog.describe(image_resized)

    return hist