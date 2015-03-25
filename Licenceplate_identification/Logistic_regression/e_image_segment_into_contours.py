#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     11/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

'''
For extraction of digits we would have to perform blurr and then canny edge detection
. Finally after all these plot the contour and extract the rectangular region arround the digit.
'''

import cv2

def return_all_contours_in_image(image_to_classify):

    #print "aaaaaaaa"

    # For the sake of simplicity we resize the image
    #image_resized=imutils.resize(image_class, height=200)
    image_gray=cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2GRAY)
    image_blurr=cv2.GaussianBlur(image_gray, (5,5), 0)
    image_edged= cv2.Canny(image_blurr, 30,150)    # Study of canny edge detection is important before assuming the thresholds

    (cnts, _) = cv2.findContours(image_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_gray, cnts, -1,(0,255,0),2)
    cv2.imshow("Contoured image", image_gray)
    cv2.waitKey(0)
    # Now lets find the rectangle coordinates around the contours
    cnts=sorted([(c, cv2.boundingRect(c) [0]) for c in cnts], key=lambda x:x[1])
    #print cnts[0]


    return cnts