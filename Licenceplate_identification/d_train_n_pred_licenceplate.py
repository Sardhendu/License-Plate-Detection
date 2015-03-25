#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     10/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
'''
About: This code snippet is just a caller to the logistic regression code snippet. And after fetching the weights id calculates the class for the new instance.
'''

import b_logistic_regression
import c_create_dataset


import numpy as np
import matplotlib.pyplot as plt



'''
   The below code will create the feature set by calling the create_features method of c_create_dataset package for our training data,
   and call methods of the  b_logistic_regression package to calculate the theta value.
   x is the feature set created of second order polynomial degree
   y id is the binary class for each training data instances
   theta is the optimal theta value calculated
'''
def train_dataset(alpha, max_iter, path):

    feature_array, class_array=c_create_dataset.create_features(path)

    # Convert the array into numpy arrays
    x=np.array(feature_array,dtype="float64")
    y=np.array(class_array,dtype="float64")


    # Doing polynomial for 3rd degree, we create a nuw array with 3rd degree and combined it to the main array
    x_ordered_2=pow(x,2)
    x=np.concatenate((x,x_ordered_2), axis=1)

    # Add the intercept value 1 to the first column. the 0 indicates first column and 1 indicates the intercept term
    x=np.insert(x,0,1,axis=1)
    (m,n)=x.shape

    # Calculate mean and standard deviation
    mn=np.mean(x, axis=0)  # axis=0 will perform column wise mean
    sd=np.std(x, axis=0)

    # Do the feature scalling of the data set
    x=b_logistic_regression.cal_feature_scalling(x,n,mn,sd)

    # Calculate the cost function, gradient and theta
    j_theta, theta=b_logistic_regression.main_call(x,y,alpha,max_iter,m,n)

    # Mean and standard deviation are also returned because our new instances will be needing them to predict the outcome
    return x, j_theta, theta, mn, sd




'''
   The below code will create the feature set for the test data instances by calling create_features method of c_create_dataset and then predict the output for each new instance.
   It also calls few method of the b_logistic_regression package to perform feature scalling on the datasets.
   x_pred is the feature set for our new instances that we have to classify
   pred_data is the prediction probability
'''
def test_dataset(mn,sd, theta, path):

    feature_array_pred, class_array=c_create_dataset.create_features(path)

    # Convert the test data into np arrays
    x_pred=np.array(feature_array_pred, dtype="float64")

    # The theta we get from the training dataset is the result of second degree polynomial of x, therefore we must convert the new instance into 2nd order to multiply it to theta
    # Doing polynomial for 2nd degree, we create a nuw array with 2nd degree and combined it to the main array
    x_pred_ordered_2=pow(x_pred,2)
    x_pred=np.concatenate((x_pred,x_pred_ordered_2), axis=1)

    # Add the intercept value 1 to the first column. the 0 indicates first column and 1 indicates the intercept term
    x_pred=np.insert(x_pred,0,1,axis=1)
    (m,n)=x_pred.shape

    x_pred=b_logistic_regression.cal_feature_scalling(x_pred,n,mn,sd)

    # Now we do the prediction for our new instances
    pred_data=b_logistic_regression.cal_sigmoid(x_pred,theta)


    return x_pred,pred_data

