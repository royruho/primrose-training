# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:05:51 2019

@author: royru
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 13:57:52 2019

@author: royru
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


filename = (r'C:\Users\royru\Desktop\primrose\image_processing\facecrop.jpg')
amit = cv2.imread(filename,1)
px = amit[100,100]
print ("pixel 100X100 : ",px)
bluepx = px[0]
print ("blue: ",bluepx)
# accessing BLUE value
print (amit.item(100,100,0))
# modifying BLUE value
amit.itemset((100,100,0),255)
amit.itemset((101,101,0),255)
amit.itemset((100,101,0),255)
amit.itemset((101,100,0),255)
print ('modify blue pixel' , amit.item(100,100,0))
print ("shape:",amit.shape)
print ("size:",amit.size)
print ("type:",amit.dtype)

#split file to RGB
b,g,r = cv2.split(amit)
g = g // 2
r = r // 2
b = b + 10
# merge giving blue a boost
blueish = cv2.merge((b,g,r))

cv2.imshow('blueish',blueish)
cv2.imshow('Amit',amit)
replicate = cv2.copyMakeBorder(amit,15,15,15,15,cv2.BORDER_REPLICATE)
BORDERWRAP = cv2.copyMakeBorder(amit,15,15,15,15,cv2.BORDER_WRAP)
cv2.imshow('replicate',replicate)
cv2.imshow('BORDER_WRAP',BORDERWRAP)
cv2.waitKey(0)
cv2.destroyAllWindows()

# get center of image

center = amit[40:140, 40:105]
cv2.imshow('center',center)
#read image in gray mode
grey = cv2.cvtColor(amit,cv2.COLOR_BGR2GRAY)
cv2.imshow('grey',grey)

# make binary image with different threshholds
ret,binary50 =  cv2.threshold(grey,50,255,cv2.THRESH_BINARY)
ret,binary =  cv2.threshold(grey,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
cv2.imshow('binary50',binary50)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blur image using gaussianblur
gaussian_blured1 = cv2.GaussianBlur(amit, (51,1), 2)
gaussian_blured2 = cv2.GaussianBlur(amit, (1,51), 2)
cv2.imshow('Amit',amit)
cv2.imshow('gaussian_blured1',gaussian_blured1)
cv2.imshow('gaussian_blured2',gaussian_blured2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# use 2dfilter
filter2d1 = cv2.filter2D(amit, -1,(-.1,1,.1))
filter2d2 = cv2.filter2D(amit, -1, (-10,0,10))
filter2d3 = cv2.filter2D(amit, -1, (10,0,-10))
cv2.imshow('Amit',amit)
cv2.imshow('filter2d1',filter2d1)
cv2.imshow('filter2d2',filter2d2)
cv2.imshow('filter2d3',filter2d3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#resize image using pyramid up and down and resize
pyrmidup = amit
pyrmidup = cv2.pyrUp(amit,pyrmidup,(len(amit[1,:])*2, len(amit[:,1])*2))
pyrmiddown = pyrmidup
pyrmiddown = cv2.pyrDown(pyrmidup,pyrmiddown,(len(pyrmidup[1,:])//2, len(pyrmidup[:,1])//2))
resizeimg = cv2.resize(pyrmidup,(len(amit[1,:]),len(amit[:,1])))
cv2.imshow('Amit',amit)
cv2.imshow('pyrmidup',pyrmidup)
cv2.imshow('pyrmiddown',pyrmiddown)
cv2.imshow('resizeimg',resizeimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rotate image using rotataion matrix
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
Mr = cv2.getRotationMatrix2D((100,300), -45,.8) 
wrapM = cv2.warpAffine(amit, M,((len(amit[1,:])*2,len(amit[:,1])*2)))
wrapMr = cv2.warpAffine(amit, Mr,((len(amit[1,:])*2,len(amit[:,1])*2)))
cv2.imshow('Amit',amit)
cv2.imshow('wrapM',wrapM)
cv2.imshow('wrapMr',wrapMr)
cv2.waitKey(0)
cv2.destroyAllWindows()




