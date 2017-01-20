# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:07:10 2015

@author: Sardhendu_Mishra
"""
import os
import glob
import numpy as np
from six.moves import cPickle as pickle 


import Configuration
from Bld_FeatureCrps import CrtFeatures
from BldModel import SvmModel



#==============================================================================
# Model 1: Opening the saved model and making prediction on test data using the model
#==============================================================================


class Eval():
    def __init__(self):
        self.conf = Configuration.get_datamodel_storage_path()
        
    def test_using_model_rbf(self, feature_test):   
        # Testing the test data precision with all the models with different value of paramter
        dict_models = {}
        for files in glob.glob(self.conf['Models']):   
            file_name = os.path.basename(files)     
            classifier = pickle.load(open(files, 'rb')) 
            test_classify = classifier.predict(feature_test)
            dict_models[file_name] = test_classify
        return dict_models
            

    def test_using_model_self(self, feature_test):   
       feature_train = np.genfromtxt(Data_feature_dir, delimiter=',') 
       theta = np.genfromtxt(Theta_val_dir, delimiter=',') 

       sigma=10
       feature_test_kernel_cnvrtd = SvmModel().build_kernel(feature_test,feature_train,sigma)
       feature_test_kernel_cnvrtd = np.insert(feature_test_kernel_cnvrtd,0,1,axis=0)
       feature_test_kernel_cnvrtd = np.transpose(feature_test_kernel_cnvrtd)
       prediction = SvmModel().cal_sigmoid(feature_test_kernel_cnvrtd, theta)  
       np.savetxt(self.conf['Data_feature_KernelCnvtd_tst_dir'], feature_test_kernel_cnvrtd, delimiter=",")
       print (prediction)


            
    
