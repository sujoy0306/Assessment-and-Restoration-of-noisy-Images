import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

#create blank array size equal 257 columns
countSet = np.array([], dtype=np.int16).reshape(0,257)

#set folder names and corresponding labels
folders = ['Gauss', 'NonGauss']
label   = [1,2]

for i,folder in enumerate(folders):

    #get names of all image file in Frames folder
    imageFiles = glob(folder+'/*')

    #iterate for each image
    for images in imageFiles:

        #read image as grayscale
        imArr = cv2.imread(images,cv2.COLOR_BGR2GRAY)

        #count unique pixel
        #NOTE: some color maynot exist
        unique,colCounts = np.unique(imArr, return_counts=True)

        #pack unique and count values together
        uniqueDict = dict(zip(unique,colCounts))
        counts = np.arange(256)
        #make sure the count size is 256
        for pixVal in uniqueDict.keys():
            counts[pixVal] = uniqueDict[pixVal]
    
        #append the label
        counts = np.append(counts,[label[i]])
    
        #add it to dataset
        countSet = np.vstack((countSet, counts))

#save as dataset
np.savetxt('dataset.csv', countSet.astype(int),fmt='%i', delimiter=',')
