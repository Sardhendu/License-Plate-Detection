# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:07:10 2015

@author: Sardhendu_Mishra
"""
import cPickle
import numpy as np
import glob
import c_pre_processing


path_test_1="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\photo_research\\2photus_crossvalid\\licence_plate\\*.jpg"

path_test_2="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\photo_research\\2photus_crossvalid\\not_licence_plate\\*.jpg"



#==============================================================================
#   Opening the saved model and making prediction on test data using the model
#==============================================================================

def test_main(classifier):
            
    x=np.array(feature_array_test, dtype="float64")
    
    test_classify=classifier.predict(x)

    return test_classify
#accuracy=accuracy_score(labels_test, pred)


#####################################################################################################
#####################################################################################################

path_test=[path_test_2]#, path_test_2]
# Prepare the test dataset


feature_array_test, class_test=c_pre_processing.create_features(path_test)

folder="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\models\\*.csv"


for files in glob.glob(folder):
    print files       
    classifier= open(files).read()
    classifier=cPickle.loads(classifier) 
    test_classify=test_main(classifier)
    print classifier
    print test_classify