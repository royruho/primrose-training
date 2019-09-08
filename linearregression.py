# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:22:49 2019

@author: royru
"""

import numpy as np
randint10 = np.random.randint(-100,101, size=10) # 1.a vector 10 random integers range [-100-100]
randfloat10 = np.random.uniform(-100.0, 100.0, size=10) # 1.b vector 10 random floats range [-100-100)
randint3mult = 3*(np.random.randint(-10,11, size=5)) #1.c vector 5 random integers range [-30-30] multiplys of 3
array1slope = 0.5 #decides slope of line
v1 = np.random.uniform(-10,10, size=10)
firstarray = np.array([array1slope*v1]) #2.a line y=ax
noisearray = np.random.normal(-1.0,1.0, size=10) # 2.b create noise
firstarray = firstarray + noisearray # 2.b add noise
v2 = np.random.uniform(-10,11, size=10) # 2.c 
array2slope=.5 #
b2=10 # added value to line
secondarray = np.array([array2slope*v2+b2]) #2.c line y=ax+b
noisearray = np.random.normal(-5.0,5.0, size=10)
secondarray = secondarray + noisearray
v3 = np.random.uniform(-100,101, size=20)
third_array = v3*v3 + b2 #2.d parabula line y=x^2 + b
matrix1 = np.zeros((4,4))
matrix1 += np.arange(1,5) #3.a
matrix2 = np.diag(([1,1,1,1])) #3.a
matrix1 = matrix1 + matrix2
multmatrix = matrix1@matrix2 #3.b
trasposedmatrix = multmatrix.T #3.c
inversedmatrix = np.linalg.inv(multmatrix) #3.c

###############################   4        ####################################
#v1 - x featurs
#firstarray - y 
v1 = np.reshape(v1,(10,1))
#stack = np.column_stack((v1,v1))
#test = np.linalg.inv(v1.T@v1)@v1.T
h_firstarray = np.linalg.inv(v1.T@v1)@v1.T@firstarray.T

###############################   5        ####################################

v2array = np.array([np.ones(v2.shape),v2]).T
h_secondarray = np.linalg.inv(v2array.T@v2array)@v2array.T@secondarray.T


import matplotlib.pyplot as plt
#       4 - plot firstarray
line1X = (np.linspace(-10,10,10))
line1Y = np.reshape(line1X*h_firstarray,10,1)
line1X = np.reshape(line1X,(10,1))
plt.figure(1)
plt.xlabel('v1')
plt.ylabel('firstarray.T')
plt.title('firstarray ={}v1'.format(array1slope))
plt.plot(line1X,line1Y, 'b-',v1,firstarray.T, 'ro')
#plt.figure(2)
#plt.plot(v1,firstarray.T, 'ro')
plt.show()

#                6 - plot secondarray
line2X = (np.linspace(-10,10,10))
line2Y = np.reshape(line2X*h_secondarray[1]+h_secondarray[0],10,1)
line2X = np.reshape(line2X,(10,1))
plt.figure(1)
plt.xlabel('v2')
plt.ylabel('seconarray.T')
plt.title('secondarray ={}v2 + {}'.format(array2slope,b2))
plt.plot(line2X,line2Y, 'b-',v2,secondarray.T, 'ro')
#plt.figure(2)
#plt.plot(v1,firstarray.T, 'ro')
plt.show()

###############################  7        ####################################

#calculate parameters for thirdarray
ss = 20 #sample size
ts = 20 #tests size
noisearray2 = np.random.randn(ss)
b3= 1
v3 = np.random.uniform(-10.0,10.0,size=ss)
#v3=v3+noisearray2
v3sqrxpararameter = 0.1
v3xparameter = 0.3
third_array = v3sqrxpararameter*v3*v3 +v3xparameter*v3+ b3 + noisearray2
v3 = np.reshape(v3,(ss,1))
third_array = np.reshape(third_array,(ss,1))
third_arrayb= np.ones(third_array.shape)
third_arrayx = v3
third_arrayy= v3*v3
stack = np.column_stack((third_arrayb,third_arrayx,third_arrayy))
h_thirdarray = np.linalg.inv(stack.T@stack)@stack.T@third_array

#plot thirdarray
import matplotlib.pyplot as plt
line3X = (np.linspace(-10,10,ts))
line3Y = np.reshape((line3X*h_thirdarray[1]+line3X*line3X*h_thirdarray[2]+h_thirdarray[0]),(ts,1))
line3X = np.reshape(line3X,(ts,1))
plt.figure(3)
plt.xlabel('v3')
plt.ylabel('thirdarray')
plt.title('thirdarray = {}v2*v2 + {}v2 + {}'.format(v3sqrxpararameter,v3xparameter,b3))
plt.plot(line3X,line3Y, 'b',line3X,line3Y, 'bo',v3,third_array, 'ro')
plt.show()

###############################  8        ####################################

#question 8 y=a(e^(bx^2)+cx)
import numpy as np
import matplotlib.pyplot as plt
xq8 = ([0.08750722, 0.01433097, 0.30701415, 0.35099786, 0.80772547, 0.16525226, 0.46913072, 0.69021229, 0.84444625, 0.2393042, 0.37570761, 0.28601187, 0.26468939, 0.54419358, 0.89099501, 0.9591165, 0.9496439 ,0.82249202, 0.99367066, 0.50628823])
yq8 = ([4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882])
lnyq8 = np.log(yq8)
xq8 = np.reshape(xq8,20,1)
ss = 20 #sample size
#ts = 20 #tests size
#noisearray2 = np.random.randn(ss)
#b3= 1
#v3 = np.random.uniform(-10.0,10.0,size=ss)
##v3=v3+noisearray2
#v3sqrxpararameter = 0.1
#v3xparameter = 0.3
third_array = lnyq8 #v3sqrxpararameter*v3*v3 +v3xparameter*v3+ b3 + noisearray2
v3 = np.reshape(xq8,(ss,1))
lnyq8 = np.reshape(lnyq8,(ss,1))
q8a= np.ones(lnyq8.shape)
q8c = xq8
q8b= xq8*xq8
stack = np.column_stack((q8a,q8c,q8b))

plt.figure(4)
plt.xlabel('xq8')
plt.ylabel('lnyq8')
plt.title('y=a(e^(bx^2)+cx) parameters are , a={},b={},c={}'.format(np.exp(h_q8[0]),h_q8[2],h_q8[1]))
testlinex= ([0.00001,0.1,0.2,0.3,0.4,0.5,1,1.5,2])
testliney= h_q8[0]+testlinex*h_q8[1]+h_q8[2]*testlinex*testlinex
plt.plot(xq8,lnyq8, 'ro',testlinex,testliney,'b-')
plt.show()
