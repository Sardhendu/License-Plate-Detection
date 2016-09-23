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
About: This code snippet is written to calculate the gradient of each pixel, every pixel of an image call this code snippet and the code snippet \
       returns the magnitude of the very pixel.
'''

import math
import cv2



# Not needed in our case because we take care of the pixel coordinate in generate_pixel function ,, anyway this checks if the pixel in process is in range of the pixel size
def check_pixel_range (image, r, c):    #c is the column and r is the row
    total_size_xaxis=image.shape[1]
    total_size_yaxis=image.shape[0]
    return c>0 and c<total_size_xaxis-1 and r>0 and r<total_size_yaxis-1



# Calculate the dy (the change in y axis) we need the vertical neighbours to calculate dy
def cal_dy (image, r,c, default_delta=1.0):

    if not check_pixel_range(image,r,c):
         return default_delta
    vertical_neighbor_intensity_change=image[r-1,c]- image[r+1,c]# we have taken (r-1,c) and (r+1,c) because in openCV an image point is read as (r,c) not (c,r)
    dy=vertical_neighbor_intensity_change

    if dy==0:
        return default_delta
    else:
        return float(dy)



# Calculate the dx (the change in x axis) we need the horizontal neighbors to calculate dx
def cal_dx (image, r, c, default_delta=1.0):

    if not check_pixel_range(image,r,c):
         return default_delta
    horizontal_neighbor_intensity_change=image[r,c-1]- image[r,c+1] # we have taken (r-1,c) and (r+1,c) because in openCV an image point is read as (r,c) not (c,r)
    dx=horizontal_neighbor_intensity_change

    if dx==0:
        return default_delta
    else:
        return float(dx)


# Calculate the magnitude of the gradient
def cal_gradient_magnitude(dx, dy):
    magnitude=math.sqrt(math.pow(dx,2) + math.pow(dy,2))
    return magnitude


def call_magnitude(image, r,c):
    dy=cal_dy(image, r,c)
    dx=cal_dx(image,r,c)
    magnitude=int(cal_gradient_magnitude(dx,dy))

    return magnitude
