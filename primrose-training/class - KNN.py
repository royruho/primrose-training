# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statistics
class DataSet:
    def __init__(self,Data):# input is dataframe
        self.X= Data.iloc[:, :-1].values
        self.y= Data.iloc[:, 4].values
    def Split(self,ts=0.20): # parse iris array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(self.X, self.y,test_size=ts))
            
class KNNclass:
    def __init__(self,data):# input is dataframe
        self.Xtest = data[1]
        self.ytest = data[3]
        self.X= data[0]
        self.y= data[2]
        self.neighborlist = []
        self.labels = []

    def Calacdistance(self,index1,index2): # calc distance of pont2 points on train model
        return np.sqrt(np.sum((self.X[index1]-self.X[index2])**2))

    def Sortneighbors(self,index,K): # sort on train model 
        self.neighborlist = []
        for i in range (0,len(self.X)):
            self.neighborlist.append((i,self.Calacdistance(index,i)))
            self.neighborlist.sort(key=lambda tup: tup[1])
        return (self.neighborlist[0:K])

    def Predict(self,index,K): # predict on train model
        self.labels = []
        neighbors = self.Sortneighbors(index,K)
        for i in range(0,K):
            self.labels.append(self.y[neighbors[i][0]])
        return (statistics.mode(self.labels))

    def accuracy(self,K): # accuracy of train model on train model
        counter = 0
        for i in range (0,len(self.X)):
            if self.Predict(i,K) == self.y[i]:
                counter = counter +1
        return ((counter/(i+1)*100))

    def Predictest(self,i,K): #return predicted label of Xtest point using Xtrain model
        self.labels = []
        neighbors = []
        neighbors = self.testSortneighbors(i,K)
        for i in range(0,K):
            self.labels.append(self.y[neighbors[i][0]])
        try:
            return (statistics.mode(self.labels))
        except :
            return  (self.labels[0])
    
    def testmodel(self,K): # return accuracy of predictions on test-set using Xtrain model
        counter = 0
        for i in range (0,len(self.Xtest)):
            if self.Predictest(i,K) == self.ytest[i]:
                counter = counter +1
        self.testaccuracy = ((counter/(i+1)*100))
        return ((counter/(i+1)*100))
    
    def testCalacdistance(self,index1,index2): #distance betwin test point and train point set
        return np.sqrt(np.sum((self.Xtest[index1]-self.X[index2])**2))
    
    def testSortneighbors(self,index,K): # sort neares neighbors of xtest point from xtrain set
        self.neighborlist = []
        for i in range (0,len(self.X)):
            self.neighborlist.append((i,self.testCalacdistance(index,i)))
            self.neighborlist.sort(key=lambda tup: tup[1])
        return (self.neighborlist[0:K])
#########################################################################################
if __name__ == "__main__": 
    iris0 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    iris1 = iris0.copy()# load data frame from file
    iris1.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
    irisData = DataSet(iris1)
    Kmodel2= KNNclass(irisData.Split(0.2))
    print (Kmodel2.testmodel(3)) # should be almost 100% accurate
    Kmodel1= KNNclass(irisData.Split(0.95))
    print (Kmodel1.testmodel(3)) # should be less accurate - smaller train data

