# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:07:10 2015

@author: Sardhendu_Mishra
"""
import cPickle
import numpy as np
import glob

import Configuration
from Bld_FeatureCrps import CrtFeatures
from BldModel import SvmModel




#==============================================================================
# Model 1: Opening the saved model and making prediction on test data using the model
#==============================================================================


class Test():
    def __init__(self):
        self.conf = Configuration.get_datamodel_storage_path()

    def test_main(self, classifier, feature_array_test):
        test_classify = classifier.predict(feature_array_test)
        return test_classify
        #accuracy=accuracy_score(labels_test, pred)

    def test_using_models(self, feature_test):   
        # Testing the test data precision with all the models with different value of paramter
        for files in glob.glob(self.conf['Models']):   
            print files       
            classifier = open(files).read()
            classifier = cPickle.loads(classifier) 
            test_classify = test_main(classifier, feature_test)
            print classifier
            #print test_classify
            print "count of 1 is %d" %sum(test_classify==1)
            print "count of 0 is %d" %sum(test_classify==0)

    def test_using_model_s(self, feature_test):   
       feature_train = np.genfromtxt(Data_feature_dir, delimiter=',') 
       theta = np.genfromtxt(Theta_val_dir, delimiter=',') 

       sigma=10
       feature_test_kernel_cnvrtd = SvmModel().build_kernel(feature_test,feature_train,sigma)
       feature_test_kernel_cnvrtd = np.insert(feature_test_kernel_cnvrtd,0,1,axis=0)
       feature_test_kernel_cnvrtd = np.transpose(feature_test_kernel_cnvrtd)
       prediction = SvmModel().cal_sigmoid(feature_test_kernel_cnvrtd, theta)  
       np.savetxt(self.conf['Data_feature_KernelCnvtd_tst_dir'], feature_test_kernel_cnvrtd, delimiter=",")
       print prediction
 
    def create_test_features(self, model = 'model_s'):
        tst_path = [self.conf['Test_data1_dir']]#, self.conf['Test_data2_dir']]
        feature_tst, labels_tst = CrtFeatures().create_features(tst_path)
        feature_matrix_test = np.array(feature_tst, dtype="float64")
        labels_array_test = np.array(class_test)
        if model_s:
            self.test_using_model_s(feature_matrix_test)
        else:
            self.test_using_model_s(feature_matrix_test)

            
    
