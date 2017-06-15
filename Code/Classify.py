# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:08:39 2015

@author: Sardhendu_Mishra
"""
import cv2
from collections import OrderedDict
import Configuration
from Bld_FeatureCrps import CrtFeatures


#==============================================================================
#   Making prediction on test data using the model
#==============================================================================

class Classify():

    def __init__(self):
        self.conf = Configuration.get_datamodel_storage_path()

    def extract_contours(self, image_to_classify):
        # For the sake of simplicity we resize the image
        image_gray = cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2GRAY)
        image_blurr = cv2.GaussianBlur(image_gray, (5,5), 0)
        image_edged = cv2.Canny(image_blurr, 0,10)    # Study of canny edge detection is important before assuming the thresholds
    # 7 im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        (__, cnts, _) = cv2.findContours(image_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_gray, cnts, -1,(0,255,0),2)
        # cv2.imshow("Contoured image", image_gray)
        # cv2.waitKey(0)
        # Fetch the rectangle coordinates around the contours
        cnts=sorted([(c, cv2.boundingRect(c) [0]) for c in cnts], key=lambda x:x[1])
        return cnts


    def classify_new_instance (self, image_to_classify, classifier):
        contours = self.extract_contours(image_to_classify)
        '''
        In the below code we loop through all the contoured images, extract the features by the use of create_dataset. After the features are extracted the features are multiplied to the theta value obtained by the training dataset. The images (the number plate images are then put into the folder Countered_image_classified
        '''
        roi_name_array=[]   
        pred_classify=[]
        pred_image_name=[]
        pred_classify_all= OrderedDict()
        #roi_probability_array=[]
        count=0
        for (c,_) in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w>=50 and h>=10: # We do not go through all the contours but rectangles that have more probability of being a lisence plate
                region_of_interest = image_to_classify[y:y+h,x:x+w]
                roi_name_array.append(region_of_interest)   
                # roi stands for region of interest, we store the array of each image in an stack array to later extract the licence plate from given index
                # We save the region of interest into a folder, retrive it and then classify it             
                cv2.imwrite(self.conf['Regions_of_Intrest']+"roi_images%04i.jpg" %count, region_of_interest) 
                path_classify_roi = [self.conf['Regions_of_Intrest']+"roi_images%04i.jpg" %count]
                # We now create feature set for the Region of Interest as we did formt he training set
                roi_feature_array, _ = CrtFeatures().create_features(path_classify_roi) 
                pred_classify = classifier.predict_proba(roi_feature_array) # path_classify will basically contain one image
                pred_image_name=("roi_images%04i.jpg" %count)
                pred_classify_all[pred_image_name] = pred_classify[0]
                count += 1
        return pred_classify_all


