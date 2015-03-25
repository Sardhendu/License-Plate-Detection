#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     10/02/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2

def translate(image, x, y):
    M=np.float32([[1,0,x],[0,1,y]])
    # M is defined as the floating point array because cv2 expects the matrix to be in floting point array
    # The first array [1,0,x] indicates the  number of pixels to shift right, A negative x would shift x pixels left
    # The second array [0,1,y] indicates the number of pixels to shift down, A negative y would shift the image y pixels up
    shifted=cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # The first argument to the warpAffine function is the image
    # The second argument to the warpAffine function is the array by which the image has to shift
    # The third argument is the dimension in (height and width), by default the image is stored in (weidth * height) but opencv takes dimension in (height * weidth) format
    return shifted

def rotate(image, angle, center = None, scale=1.0):
    # The arguments are :
    '''
    1. The image
    2. the angle by which you want to rotate
    3. The center from which you want to rotate, The default is None, when None then we define the center
    4. scale=1.0 by default, It states that
    '''
    (h,w)= image.shape[:2]

    # When the center is None then we define the center from which the rotation is to be done
    if center is None:
        center=(w/2, h/2)

    M=cv2.getRotationMatrix2D(center, angle, scale)
    rotated= cv2.warpAffine(image, M, (w,h))

    return rotated


def resize(image, width= None, height = None, inter=cv2.INTER_AREA):
    dim=None
    (h,w)= image.shape[:2]  # numpy array stores images in (height, width) array, but cv2 uses images in order (width, height) order

    if width is None and height is None:  # when no resizing occur
        return image

    if width is None:       # when resized height is passed and width is not then we calculate the aspect ratio of the weidth
        r= height / float(h)
        dim=(int(w * r), height)      # height is the resized hieght
    elif height is None:    # When resized width is passed and hieght is not then we calculate the aspect ratio for the height
        r= width / float(w)
        dim=(width , int(h * r))
    else:                   # when both width and height ratio are provided
        dim=(width, height)

    resized= cv2.resize(image, dim, interpolation=inter)
    # the third argument hold an algorithm in cv2 defined to resize the image
    # we can also use other algorithm like cv2.INTER_LINEAR, cv2.INTER_CUBIC, and cv2.INTER_NEAREST.

    return resized



