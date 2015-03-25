# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:04:09 2015

@author: Sardhendu_Mishra
"""


'''
About: This code snippet is the logistic regression with Newton's method with regularization, 
       Given the training data set this code snippet find the best parameter or the weights (thetas) for each
       feature of the data set. This is operated for 2nd degree polynomial.
       
       The Newtons method consists of a regularization term. For some simple data-set like 2D dataset
       the newtons method would just work fine without regularization. But wilh many features or dimensions
       the cost function for the logistic regression using newtons methods might be abrupt and non converging.
       To avoid this situation it is advicable to use the regularization term always with newtons method.
'''

import numpy as np


def cal_feature_scalling(x,n,mn,sd):
    for mnsd in range (1,n):  # We dont feature scale the first column column because its 1 . the range would go from 1 to n-1 
                              # 1,n-1 because in numpy array the first column is 1, second is 2 and so on
        x[:,mnsd]=np.divide(np.subtract(x[:,mnsd] , mn[mnsd]), sd[mnsd])

    return x

#hh=[]
def cal_sigmoid(x,theta):
    z=np.dot(x,theta)  # For matrix multiplication x is [80*3], theta is [3*1]
    h= 1 / (1 + np.exp(- z))
    return h

def cal_cost(h, y, m):
    j_theta= (-np.transpose(y).dot(np.log(h)) - (np.transpose(1-y).dot(np.log(1-h))))/m
    return j_theta

def cal_grad(x,y,h,m):
    error =h-y

    
    grad=  (np.dot(np.transpose(x), error))/m # For matrix multiplication x transpose is [3*80] and y is [80*1]
    return grad


def cal_hessian(x,h,m):
    diag_h=np.diag(h.ravel())
    diag_1_min_h=np.diag((1-h).ravel())
    hessian= ( np.dot (np.transpose(x), (np.dot((diag_h*diag_1_min_h) , x))) )/m
    return hessian


def main_call(x,y, max_iter,m,n, lambda_val):
    #print n
    j_theta=np.zeros(shape=(max_iter,1))
    theta=np.zeros(shape=(n,1))
    
    for num_iter in range (0,max_iter):
        h=cal_sigmoid(x,theta)

        j_theta[num_iter]=cal_cost(h, y, m)

        grad=cal_grad(x,y,h,m)
        reg_grad = (float(lambda_val)/float(m)) * theta
        reg_grad[1]= 0                                # We dont do regularization for the intercept column
        grad=grad+reg_grad
        
        hessian=cal_hessian(x,h,m)
        reg_hessian= (float(lambda_val)/float(m)) * np.eye(n)
        reg_hessian[1]= 0
        hessian= hessian+reg_hessian
        
        
        # Calculate the theta
        hessian_inv_grad=np.linalg.solve(hessian, grad) 
        theta=np.subtract(theta,hessian_inv_grad)

    return j_theta, theta
