# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:05:19 2015

@author: Sardhendu_Mishra
"""

import cv2.cv as cv	
# interface opencv versi 2
import numpy as np
import cv2


def calc_hog(im,numorient=9,cellSize=(8,8)):
    ihog = calc_hog_base(im, numorient)
    return calc_hog_cells(ihog, numorient, cellSize)
    
    
def preprocess(img):
    # gamma correction 1/4 followed by contrast boost (histogram equalization)
    return cv2.equalizeHist((np.power(img/255., 0.25)*255).astype(np.uint8))    
    
def calc_hog_base(im, numorient=9):
    # calculate gradient using sobel operator
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
 
    gray = preprocess(gray) #preprocess gray image before calculating gradient
 
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
     
    # calculate gradient magnitude
    """
    #mag = np.sqrt(gx*gx+gy*gy)
    mag
    """
 
    # calculate gradient orientation and discretize to numorient values
    mid = numorient/2
    
    hog = np.zeros((im.shape[0], im.shape[1], numorient))
    
    
    """
    mid = numorient/2
    for y in xrange(0, im.shape[0]-1):
        for x in xrange(0, im.shape[1]-1):
            angle = int(round(mid*np.arctan2(gy[y,x], gx[y,x])/np.pi))+mid
            magnitude = np.sqrt(gx[y,x]*gx[y,x]+gy[y,x]*gy[y,x])
            hog[y,x,angle] += magnitude    
    
    
    """
    magor = mid+(mid*np.arctan2(gy, gx)/np.pi).astype(np.uint32)
    
    #print (magor.min(), magor.max()+1)
     
    # calculate hog in pixels by grouping each pixel based on gradient orientation
    
    for orien in xrange(magor.min(), magor.max()+1):
        mask = magor==orien
        hog[:,:,orien][mask] = mag[mask] + magor[mask]
        
 
    # calculate integral hog
    #ihog = cv2.integral(hog)
     
    return gray,gx,gy,mag,magor,hog        
    
gray,gx,gy,mag,magor,hog=calc_hog_base(image) 

cv2.imshow("gray", gray)
cv2.imshow("gx", gx)
cv2.imshow("gy", gy)
cv2.waitKey(0)   