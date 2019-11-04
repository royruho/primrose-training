# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:37:50 2019

@author: royru
"""
import numpy as np

epsilon = 10**(-5)

def sigmoid(x): # Activation function used to map any real value between 0 and 1
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x): 
    return (sigmoid(x)*(1-sigmoid(x)))

def loss(yp,label):
    return 1/2*(yp-label)**2

def loss_derv(yp,label):
    return yp-label

learning_rate = 0.3

inputs = np.array([ 
[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]])

labels = np.array([
[0],
[1],
[1],
[0]])

    
# 1 layer
print ("********************************\n 1 layer:")
x=inputs
y= labels
weights = np.random.uniform(-1,1,(3,1))
w2= weights
for i in range(100000):
    layer_2 = x@w2
    layer_2_x_derv = w2.T
    layer_2_w_derv = x.T
    layer_2_sig_activation = sigmoid(layer_2)
    layer_2_sig_derv = sigmoid_derv(layer_2)
    sum_loss = np.sum(loss(layer_2_sig_activation,y))
    error = layer_2_sig_activation-y
    delta_w2 = layer_2_w_derv@(error*layer_2_sig_derv)
    w2 = w2 - delta_w2*learning_rate
    if (i % 10000) == 0:
        print (sum_loss)
   
# 2 layers
print ("********************************\n 2 layers:")
x=inputs
y= labels
weights1 = np.random.uniform(-1,1,(3,4))
w1 = weights1
weights2 = np.random.uniform(-1,1,(4,1))
w2= weights2
for i in range(100000):
    layer_1 = x@w1
    layer_1_x_derv = w1.T
    layer_1_w_derv = x.T
    layer_1_sig_activation = sigmoid(layer_1)
    layer_1_sig_derv = sigmoid_derv(layer_1)
    layer_2 = layer_1_sig_activation@w2
    layer_2_x_derv = w2.T
    layer_2_w_derv =  layer_1_sig_activation.T
    layer_2_sig_activation = sigmoid(layer_2)
    layer_2_sig_derv = sigmoid_derv(layer_2)
    sum_loss = np.sum(loss(layer_2_sig_activation,y))
    error = layer_2_sig_activation-y
    delta_w2 = layer_2_w_derv@(error*layer_2_sig_derv)
    delta_w1 =layer_1_w_derv@(layer_1_sig_derv*((error*layer_2_sig_derv)@layer_2_x_derv))
    w2 = w2 - delta_w2*learning_rate
    w1 = w1 - delta_w1*learning_rate
    if (i % 10000) == 0:
        print (sum_loss)
        

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
iris = datasets.load_iris()
x = iris.data[:,:4]
y = np.reshape(iris.target,(150,1))
hot_encoded_labels = LabelBinarizer()
hot_encoded_labels.fit([0,1,2])
binary_labels = hot_encoded_labels.transform(y)
for i in range (len(y)) :
    if y[i] == 2:
        y[i] = 1



print ("********************************\n 2 layers on IRIS data set:")

weights1 = np.random.uniform(-1,1,(4,2))
w1 = weights1
weights2 = np.random.uniform(-1,1,(2,1))
w2= weights2
for i in range(100000):
    layer_1 = x@w1
    layer_1_x_derv = w1.T
    layer_1_w_derv = x.T
    layer_1_sig_activation = sigmoid(layer_1)
    layer_1_sig_derv = sigmoid_derv(layer_1)
    layer_2 = layer_1_sig_activation@w2
    layer_2_x_derv = w2.T
    layer_2_w_derv =  layer_1_sig_activation.T
    layer_2_sig_activation = sigmoid(layer_2)
    layer_2_sig_derv = sigmoid_derv(layer_2)
    sum_loss = np.sum(loss(layer_2_sig_activation,y))
    error = layer_2_sig_activation-y
    delta_w2 = layer_2_w_derv@(error*layer_2_sig_derv)
    delta_w1 =layer_1_w_derv@(layer_1_sig_derv*((error*layer_2_sig_derv)@layer_2_x_derv))
    w2 = w2 - delta_w2*learning_rate
    w1 = w1 - delta_w1*learning_rate
    if (i % 10000) == 0:
        print (sum_loss)
        
# 3 layers
print ("********************************\n 3 layers on IRIS data set:")

weights1 = np.random.uniform(-1,1,(4,10))
w1 = weights1
weights2 = np.random.uniform(-1,1,(10,10))
w2 = weights2
weights3 = np.random.uniform(-1,1,(10,1))
w3= weights3
learning_rate = 0.3

for i in range(100000):
    # forward layer 1
    layer_1 = x@w1
    # remember  derivatives layer 1
    layer_1_x_derv = w1.T
    layer_1_w_derv = x.T
    #activation layer 1
    layer_1_sig_activation = sigmoid(layer_1)
    #remember sigmoid derivative
    layer_1_sig_derv = sigmoid_derv(layer_1)
    
    layer_2= layer_1_sig_activation@w2
    layer_2_x_derv = w2.T
    layer_2_w_derv = layer_1_sig_activation.T
    layer_2_sig_activation = sigmoid(layer_2)
    layer_2_sig_derv = sigmoid_derv(layer_2)
    layer_3 = layer_2_sig_activation@w3
    layer_3_x_derv = w3.T
    layer_3_w_derv = layer_2_sig_activation.T
    layer_3_sig_activation = sigmoid(layer_3)
    layer_3_sig_derv = sigmoid_derv(layer_3)
    sum_loss = np.sum(loss(layer_3_sig_activation,y))
    error = layer_3_sig_activation-y
    delta_w3 = layer_3_w_derv@(error*layer_3_sig_derv)
    delta_w2 = layer_2_w_derv@(layer_2_sig_derv*((error*layer_3_sig_derv)@layer_3_x_derv))
    delta_w1 = layer_1_w_derv@(layer_1_sig_derv*((layer_2_sig_derv*layer_3_x_derv*(error*layer_3_sig_derv))@layer_2_x_derv))
    w3 = w3 - delta_w3*learning_rate
    w2 = w2 - delta_w2*learning_rate
    w1 = w1 - delta_w1*learning_rate
    if (i % 10000) == 0:
        learning_rate = learning_rate *.9
        print (sum_loss)
        
# 3 layers
print ("********************************\n 3 layers 3 labels on IRIS data set:") # not working

weights1 = np.random.uniform(-1,1,(4,10))
w1 = weights1
weights2 = np.random.uniform(-1,1,(10,10))
w2 = weights2
weights3 = np.random.uniform(-1,1,(10,3))
w3= weights3
learning_rate = 0.3

for i in range(100000):
    # forward layer 1
    layer_1 = x@w1
    # remember  derivatives layer 1
    layer_1_x_derv = w1.T
    layer_1_w_derv = x.T
    #activation layer 1
    layer_1_sig_activation = sigmoid(layer_1)
    #remember sigmoid derivative
    layer_1_sig_derv = sigmoid_derv(layer_1) 
    layer_2= layer_1_sig_activation@w2
    layer_2_x_derv = w2.T
    layer_2_w_derv = layer_1_sig_activation.T
    layer_2_sig_activation = sigmoid(layer_2)
    layer_2_sig_derv = sigmoid_derv(layer_2)
    layer_3 = layer_2_sig_activation@w3
    layer_3_x_derv = w3.T
    layer_3_w_derv = layer_2_sig_activation.T
    layer_3_sig_activation = sigmoid(layer_3)
    layer_3_sig_derv = sigmoid_derv(layer_3)
    sum_loss = np.sum(loss(layer_3_sig_activation,y))
    error = layer_3_sig_activation-binary_labels
    delta_w3 = layer_3_w_derv@(error*layer_3_sig_derv)
    delta_w2 = layer_2_w_derv@(layer_2_sig_derv*((error*layer_3_sig_derv)@layer_3_x_derv))
    delta_w1 = layer_1_w_derv@(layer_1_sig_derv*((layer_2_sig_derv*layer_3_x_derv*(error*layer_3_sig_derv))@layer_2_x_derv))
    w3 = w3 - delta_w3*learning_rate
    w2 = w2 - delta_w2*learning_rate
    w1 = w1 - delta_w1*learning_rate
    if (i % 10000) == 0:
        learning_rate = learning_rate *.9
        print (sum_loss)
        
        
    