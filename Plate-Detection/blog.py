#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     15/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

"""
import cv2
import sys
sys.path.insert(0, 'C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing')
import imutils

image=cv2.imread("C:\\Users\\sardhendu_mishra\\Desktop\\ahha\\DSC_0023.jpg")

image_resize=imutils.resize(image,width=700)

image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

image_blurr=cv2.GaussianBlur(image_gray, (3,3), 0)

image_thresh=cv2.adaptiveThreshold(image_blurr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\ahha\\23.jpg", image_thresh)
cv2.imshow("image_orig",image_thresh)
cv2.waitKey()


"""
import numpy as np


f=np.fromfile("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data_sets\\For_blog\\ex4x.dat", dtype='float64')
fa=np.fromfile("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data_sets\\For_blog\\ex4y.dat", dtype='float64')

a=f.read()

"""
x=np.array(feature_array,dtype="float64")
y=np.array(class_array,dtype="float64")

# Add the intercept value 1 to the first column. the 0 indicates first column and 1 indicates the intercept term
x=np.insert(x,0,1,axis=1)
(m,n)=x.shape
# Calculate mean and standard deviation
mn=np.mean(x, axis=0)  # axis=0 will perform column wise mean
sd=np.std(x, axis=0)

"""