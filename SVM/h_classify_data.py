# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:08:39 2015

@author: Sardhendu_Mishra
"""

import c_pre_processing
import e_image_segment_into_contours
import cv2

import sys
sys.path.insert(0, 'C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing')
import imutils

#==============================================================================
#   Making prediction on test data using the model
#==============================================================================


def classify_new_instance (image_to_classify_path, classifier):

    image_to_classify=cv2.imread(image_to_classify_path)
    contours=e_image_segment_into_contours.return_all_contours_in_image(image_to_classify)
    '''
    In the below code we loop all the contoured images, extract the features by the use of create_dataset. After the features are extracted the features are multiplied
    to the theta value obtained by the training dataset. The images (the number plate images are then put into the folder Countered_image_classified
    '''
    roi_name_array=[]   
    pred_classify=[]
    pred_image_name=[]
    pred_classify_all= []
    #roi_probability_array=[]
    count=0
    for (c,_) in contours:
        x,y,w,h=cv2.boundingRect(c)
        if w>=50 and h>=10:
            region_of_interest=image_to_classify[y:y+h,x:x+w]
            roi_name_array.append(region_of_interest)   # roi stands for region of interest, we store the array of each image in an stack array to later extract the licence plate from given index
            cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\photo_research\\photo_classify\\contoured_images_roi\\roi_images%04i.jpg" %count, region_of_interest)             
            path_classify_roi=["C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\photo_research\\photo_classify\\contoured_images_roi\\roi_images%04i.jpg" %count]
            # We take the entire path because our c_create_dataset.py accepts only path as input''
            feature_array_pred, class_pred=c_pre_processing.create_features(path_classify_roi)                   
            pred_classify=classifier.predict(feature_array_pred) # path_classify will basically contain one image
            pred_image_name=("roi_images%04i.jpg" %count)
            pred_classify_all.append([pred_classify,pred_image_name])
            
            count=count+1
        #cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\Countered_image_licenceplate\\licenceplate.jpg", list.licence_plate.count(1)
    #pred_classify_all=[[pred_classify,pred_image_name]]
    return pred_classify_all


#####################################################################################################
#####################################################################################################


"""
image=cv2.imread("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\photo_research\\segregate\\DSC_0055.jpg")
#image=cv2.imread("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\photo_research\\photo_classify\\photos_to_classify_indian_cars\\2.jpg")


image_resized=imutils.resize(image, height=600)
cv2.imshow("or", image_resized)
cv2.waitKey(0)
contours=e_image_segment_into_contours.return_all_contours_in_image(image_resized)

count=0
for (c,_) in contours:
    x,y,w,h=cv2.boundingRect(c)
    if w>50 and h>=10:
        print "I am here"
        region_of_interest=image_resized[y:y+h,x:x+w]
        cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\photo_research\\segregated_contours\\testimage%04i.jpg" %count, region_of_interest)             
        count=count+1
"""        