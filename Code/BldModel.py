# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:40:18 2015

@author: Sardhendu_Mishra
"""
from __future__ import division

import copy
from six.moves import cPickle as pickle 
import numpy as np
from sklearn.svm import SVC
import Configuration

from sklearn.svm import LinearSVC
from Bld_FeatureCrps import CrtFeatures


#==============================================================================
#  Seperate the data into training and test dataset and see the accuracy of  
#  classifier while classifing the learned models with the segregated test set.
#  The segregation will be done only once. So the accuracies can be compared.
#==============================================================================
class SvmModel():

    def understand_gaussian_kernel (self,x,y,sigma):
        euclidean_distance=pow((x-y),2)
        z=euclidean_distance/(2*(pow(sigma,2)))
        f=np.exp(-z)
        return f

    def build_kernel(self,x_new,x_orig,sigma):
        # using gaussian kernel or rbf would create a m*m dimentional training set so we take one
        (m,n)=x_orig.shape
        (m_new, n_new)= x_new.shape

        f=np.zeros(shape=(m,m_new))
        f_each_row=np.zeros(m)
        for i in range(0,m_new):
            #print i
            l=x_new[i]
            #print l
            for j in range(0,m):
                p=x_orig[j]
                euclidean_distance=0
                for k in range(0,n):
                    a=l[k]
                    b=p[k]   
                    euclidean_distance=euclidean_distance+(pow((b-a),2))
                z=euclidean_distance/(2*(pow(sigma,2)));
                f_each_row[j]=np.exp(-z);    
                f[j,i]=f_each_row[j];
        return f
                
    def cal_sigmoid(self, x,theta):
        z=np.dot(x,theta)  # For matrix multiplication x is [80*3], theta is [3*1]
        h= 1 / (1 + np.exp(- z))
        return h

    def cal_cost(self, h, y, m):
        j_theta= (-np.transpose(y).dot(np.log(h)) - (np.transpose(1-y).dot(np.log(1-h))))
        return j_theta
        
    def cal_grad(self, x,y,h,m):
        error =h-y
        grad=  (np.dot(np.transpose(x), error)) # For matrix multiplication x transpose is [3*80] and y is [80*1]
        return grad    

    def main_call(self, sigma, x, y,max_iter, alpha, c):
        X = self.build_kernel(x,x,sigma)
        X = np.insert(X,0,1,axis=1)
        (M,N) = X.shape
        
        theta = np.zeros(shape=(N ,1), dtype='float64')
        j_theta = np.zeros(shape=(max_iter,1), dtype='float64')
        for num_iter in range(0,max_iter):
            h = self.cal_sigmoid(X, theta)
            j_reg_term = np.subtract((np.transpose(theta).dot(theta)),pow(theta[0],2))/2
            j_theta[num_iter] = (float(c) * self.cal_cost(h,y,M)) + j_reg_term
            
            grad = self.cal_grad(X,y,h,M)
            
            grad = c * grad
            reg_grad = copy.deepcopy(theta)  # After taking the derivative the regularized term become theta
            reg_grad[0] = 0
            grad = grad+reg_grad
            
            theta = np.subtract(theta, (np.multiply(alpha , grad)))
            #print theta
        return X, j_theta,theta




#==============================================================================
# Classifier for Linear SVC  
#==============================================================================
class Models():

    def __init__(self, type = 'rbf'):
        self.type = type
        self.conf = Configuration.get_datamodel_storage_path()
        self.c_range = [0.1,1.0, 10.0, 100.0]#, 1000.0]            
        self.gamma_range = [0.1, 1, 10.0]  
        self.kernel="rbf"

    def fit(self, features, labels):
        if self.type == 'linear':
            # run_linear_SVM(features, labels):
            model = LinearSVC(random_state=42)
            model.fit(features, labels)
            f = open(self.conf["Linear_SVC_dir"] , "wb")
            pickle.dumps(model, f)
            f.close()

        elif self.type == 'rbf':
            c_range = self.c_range
            gamma_range = self.gamma_range
            kernel = self.kernel

            for c in c_range:
                for gamma in gamma_range:
                    clf = SVC(kernel=kernel, C=c, gamma=gamma, probability=True)
                    classifier = clf.fit(features,labels)
                    #print classifier                                                                                                                         
                    f = open(self.conf["SVM_RFB_dir"]%(int(c),int(gamma)) , "wb")
                    pickle.dump(classifier, f)
                    f.close()

