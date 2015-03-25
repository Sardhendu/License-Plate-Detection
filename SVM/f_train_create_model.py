# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:40:18 2015

@author: Sardhendu_Mishra
"""

import cPickle
import numpy as np
from sklearn.svm import SVC

from sklearn.svm import LinearSVC
import c_pre_processing





path_train_1="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\photo_research\\1photus_train\\licence_plate\\*.jpg"
path_train_2="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\photo_research\\1photus_train\\not_licence_plate\\*.jpg"


#path_new_instance="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\photos_to_classify\\*.jpg"
image_to_classify_path="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\photo_research\\3photo_classify\\photos_to_classify_indian_cars\\4.jpg"
#image_to_classify_path="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\photos_to_classify\\IMG_0382.jpg"






#==============================================================================
#  Seperate the data into training and test dataset and see the accuracy of  
#  classifier while classifing the learned models with the segregated test set.
#  The segregation will be done only once. So the accuracies can be compared.
#==============================================================================





#==============================================================================
#   Main Call
#============================================================================== 


#path_classify=[path_new_instance]
path_train=[path_train_1,path_train_2]    
# Prepare the training dataset
feature_train, class_train=c_pre_processing.create_features(path_train) 
feature_train=np.array(feature_train,dtype="float64")
class_train=np.array(class_train,dtype="float64")


   



#==============================================================================
# Classifier for Linear SVC  
#==============================================================================
model=LinearSVC(random_state=42)
model.fit(feature_train, class_train)
f=open("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\models\\model_linear_svc.csv" , "w")
f.write(cPickle.dumps(model))
f.close()




#==============================================================================
#   Classifier for poly  
#==============================================================================
"""
degree_range=[1,2,3,4]
kernel="poly"

for degree in degree_range:
    clf=SVC(kernel=kernel, degree=degree)
    classifier=clf.fit(feature_train,class_train)
    print classifier                                                                                                                             
    f=open("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\models\\model_svm_poly%i.csv" %degree , "w")
    f.write(cPickle.dumps(classifier))
    f.close()


"""

#==============================================================================
#   Classifier for rbf
#==============================================================================

c_range = [1, 100.0, 1000.0]            
gamma_range = [0.1, 1, 10.0]  
kernel="linear"

for c in c_range:
    for gamma in gamma_range:
            clf=SVC(kernel=kernel, C=c, gamma=gamma)
            classifier=clf.fit(feature_train,class_train)
            print classifier                                                                                                                             
            f=open("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\Data mining and analysis\\image_processing\\Research_1\\models\\model_svm_rbf_%i_%i.csv" %(int(c),int(gamma)) , "w")
            f.write(cPickle.dumps(classifier))
            f.close()
            #classifier,test=test_classify=test_main(path_test) 
            #pred=classify_new_instance(image_to_classify_path)
       
