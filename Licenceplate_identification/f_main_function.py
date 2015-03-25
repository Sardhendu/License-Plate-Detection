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

import d_train_n_pred_licenceplate
import e_image_segment_into_contours

import numpy as np
import cv2
import glob

path_train_1="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photus_train\\licence_plate\\*.jpg"
path_train_2="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photus_train\\not_licence_plate\\*.jpg"

path_test_1="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\Machine_learning\\photo_research\\photus_crossvalid\\licence_plate\\*.jpg"
path_test_2="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\Machine_learning\\photo_research\\photus_crossvalid\\not_licence_plate\\*.jpg"

path_new_instance="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\photos_to_classify\\*.jpg"
#path_new_instance="C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\5.jpg"


##################################################  The below code will train the model on the given cars  ######################################################################

alpha=0.03
max_iter=1000


path_train=[path_train_1,path_train_2]
path_test=[path_test_1]
#path_classify=[path_classify_instances]


# Train the dataset and create a model
x, j_theta, theta, mn, sd=d_train_n_pred_licenceplate.train_dataset(alpha, max_iter, path_train)

# Use the model to predict on test data set and note the accuracy
x_pred, pred_data=d_train_n_pred_licenceplate.test_dataset(mn,sd,theta,path_test)


#################################################################  The below code is to classify the new data   #################################################################
# Use the model to classify the new instances
#path_new_instance_arr=[path_new_instance]
#for folder in path_new_instance_arr:
#print folder
for file in glob.glob(path_new_instance):
        print file

        image_to_classify=cv2.imread(file)
        #cv2.imshow("original image", image_to_classify)
        #cv2.waitKey()

        # Take the new isntance and find the possible contours
        contours=e_image_segment_into_contours.return_all_contours_in_image(image_to_classify)


        '''
        In the below code we loop all the contoured images, extract the features by the use of create_dataset. After the features are extracted the features are multiplied
        to the theta value obtained by the training dataset. The images (the number plate images are then put into the folder Countered_image_classified
        '''
        roi_name_array=[]
        roi_probability_array=[]
        count=0
        for (c,_) in contours:
            x,y,w,h=cv2.boundingRect(c)
            #cv2.imshow("all contoured images",image_to_classify[y:y+h,x:x+w])
            #cv2.waitKey(0)
            #We create a rectangle arround the digits
            #cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if w>=50 and h>=10:
                region_of_interest=image_to_classify[y:y+h,x:x+w]

                #cv2.imshow("regions extracted",region_of_interest)
                #cv2.waitKey(0)

                ## The below lines of code will write all the reigons of interest extrated as contours from the image and then copy it to the location mentioned below,
                ## classify all the images that falls under region of interest and puts the probablity of each being an licence plate into the "roi_probability_array"
                #"roi_images%04i.jpg" %count,
                roi_name_array.append(region_of_interest)   # roi stands for region of interest, we store the array of each image in an stack array to later extract the licence plate from given index
                cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\contoured_images_roi\\roi_images%04i.jpg" %count, region_of_interest)

                path_classify_roi=["C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\contoured_images_roi\\roi_images%04i.jpg" %count]
                # We take the entire path because our c_create_dataset.py accepts only path as input'''
                x_classify, roi_probability=d_train_n_pred_licenceplate.test_dataset(mn,sd,theta,path_classify_roi)  # path_classify will basically contain one image
                roi_probability_array.append(roi_probability)

                count=count+1

        # We convert the arrays into multidimensional numpy array, so that the two arrays can be compared and the image of the licence plate can be extracted based on the index of
        # largest probability
        roi_name_array=np.reshape(roi_name_array,(len(roi_name_array),1))
        roi_probability_array=np.array(roi_probability_array)


        # Now we find the indices of the maximum probability
        indices_max_prob=np.argmax(roi_probability_array)
        licence_plate=roi_name_array[indices_max_prob]
        cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\Countered_image_licenceplate\\licenceplate.jpg", licence_plate[0])
        #cv2.imshow("Licenceplate", licence_plate[0])
        #cv2.waitKey()
        #cv2.destroyAllWindows()









'''
if float(output_data[0])==np.max(output_data, axis=0):      # Numpy array cell starts from 0 hence we provide 0 here
  cv2.imwrite("C:\\Users\\sardhendu_mishra\\Desktop\\StudyHard\\Machine_learning\\photo_research\\photo_classify\\Countered_image_classified\\image%04i.jpg" %count, region_of_interest)
  cv2.imshow("Licence plate", region_of_interest)
  cv2.waitKey()
            cv2.destroyAllWindows()
'''
# Now we go to the location where our contoured images are stored and then classify them all



'''
# Plot the j_theta graph
plt.plot(np.arange(max_iter), j_theta[:,0], '-b')
plt.xlabel("Number of Iteration")
plt.ylabel("Cost Function")
plt.show()
'''



'''
x=array([[ 1.        ,  0.89963014,  0.95851419, ...,  0.96278206,
         1.18425028,  1.10126107],
       [ 1.        ,  0.89963014,  0.95851419, ..., -0.09841206,
        -0.43739138, -0.12077956],
       [ 1.        ,  0.89963014,  0.95851419, ...,  1.15896921,
         1.18425028,  1.10126107],
       ...,
       [ 1.        , -1.04460526,  0.80813562, ...,  1.15896921,
         1.18425028,  1.10126107],
       [ 1.        ,  0.89963014,  0.95851419, ...,  1.15896921,
         1.18425028,  1.10126107],
       [ 1.        , -1.17476328, -1.17184887, ..., -1.1150182 ,
        -1.08783006, -0.41780333]])



x_pred=array([[ 1.        , -1.17476328, -1.17184887, ..., -0.95450144,
        -1.08783006, -1.06276921],
       [ 1.        , -0.6215917 , -1.17184887, ...,  1.05195804,
         0.24868779, -0.71482709],
       [ 1.        , -1.17476328, -1.17184887, ...,  1.10546363,
        -0.57995328, -1.06276921],
       ...,
       [ 1.        , -1.17476328, -1.17184887, ...,  1.15896921,
         1.18425028,  1.10126107],
       [ 1.        , -1.17476328, -1.17184887, ..., -1.1150182 ,
        -1.08783006, -1.06276921],
       [ 1.        , -1.17476328, -1.17184887, ..., -1.1150182 ,
        -1.08783006, -1.06276921]])
>>>




'''