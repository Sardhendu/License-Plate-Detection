#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      Sardhendu_Mishra
#
# Created:     04/03/2015
# Copyright:   (c) Sardhendu_Mishra 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
'''
About: The below code is the HOG class that instantiates HOG. HOG is used to create the features that is sent into a logistic regression machine.
       With the use of HOG feature we develop a model or find the optimal theta value which is again used for a new instance to predict the output.
       If the image is a number plate or not number plate
'''



from skimage import feature

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8),cellsPerBlock = (3, 3), normalize = False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize

    def describe(self, image):
        hist = feature.hog(image,
                            orientations = self.orienations,
                            pixels_per_cell = self.pixelsPerCell,
                            cells_per_block = self.cellsPerBlock,
                            normalise = self.normalize)

        return hist