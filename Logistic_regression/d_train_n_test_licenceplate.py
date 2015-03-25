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

import b_logistic_regression_gradient
import b_logistic_regression_newtons


import numpy as np


#==============================================================================
#  Training the training data
#==============================================================================

'''
    The below code will train your training data and find the minimum value using both Gradient descent
    and newtons method.
    
    The data is feature scales and is raised to the power 2 (i.e second degree polynomial is used)
    
    Note:
    If we dont feature scale the data and run gradient descent then most of the output of cost function would be Nan
    however, Newton method would still work without feature scalling
'''
def train_dataset(feature_array,class_array, alpha, lambda_val, max_iter_GD, max_iter_NM):

    #feature_array, class_array=c_create_dataset.create_features(path)

    # Convert the array into numpy arrays
    x=np.array(feature_array,dtype="float64")
    y=np.array(class_array,dtype="float64")


    # Doing polynomial for 2nd degree, we create a nuw array with 2nd degree and combined it to the main array
    x_ordered_2=pow(x,2)
    x=np.concatenate((x,x_ordered_2), axis=1)

    # Add the intercept value 1 to the first column. the 0 indicates first column and 1 indicates the intercept term
    x=np.insert(x,0,1,axis=1)
    (m,n)=x.shape
    # Calculate mean and standard deviation
    mn=np.mean(x, axis=0)  # axis=0 will perform column wise mean
    sd=np.std(x, axis=0)
    #print n


    # Do the feature scalling of the data set
    x=b_logistic_regression_gradient.cal_feature_scalling(x,n,mn,sd)



    # Calculate the cost function, gradient and theta
    ''' For gradient descent '''
    j_theta_GD, theta_GD=b_logistic_regression_gradient.main_call(x,y,alpha,max_iter_GD,m,n)
    ''' For Newtons method '''
    j_theta_NM, theta_NM=b_logistic_regression_newtons.main_call(x,y,max_iter_NM,m,n,lambda_val)

    # Mean and standard deviation are also returned because our new instances will be needing them to predict the outcome
    return x, j_theta_GD, theta_GD, j_theta_NM, theta_NM, mn, sd
#


#x_fs, j_theta_GD, theta_GD, j_theta_NM, theta_NM, mn, sd=train_dataset(x,y,0.03,5,1000,10)




#==============================================================================
#  See the output of test data 
#==============================================================================
'''
   The below code will create the feature set for the test data instances by calling create_features method of c_create_dataset and then predict the output for each new instance.
   It also calls few method of the b_logistic_regression package to perform feature scalling on the datasets.
   x_pred is the feature set for our new instances that we have to classify
   pred_data is the prediction probability
'''
def test_dataset(feature_array_pred, class_array,mn,sd, theta):

    #feature_array_pred, class_array=c_create_dataset.create_features(path)

    # Convert the test data into np arrays
    x_pred=np.array(feature_array_pred, dtype="float64")

    # The theta we get from the training dataset is the result of second degree polynomial of x, therefore we must convert the new instance into 2nd order to multiply it to theta
    # Doing polynomial for 2nd degree, we create a nuw array with 2nd degree and combined it to the main array
    x_pred_ordered_2=pow(x_pred,2)
    x_pred=np.concatenate((x_pred,x_pred_ordered_2), axis=1)

    # Add the intercept value 1 to the first column. the 0 indicates first column and 1 indicates the intercept term
    x_pred=np.insert(x_pred,0,1,axis=1)
    (m,n)=x_pred.shape

    x_pred=b_logistic_regression_gradient.cal_feature_scalling(x_pred,n,mn,sd)

    # Now we do the prediction for our new instances
    pred_data=b_logistic_regression_gradient.cal_sigmoid(x_pred,theta)


    return x_pred,pred_data

