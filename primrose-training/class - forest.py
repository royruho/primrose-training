# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:58:34 2019

@author: royru
"""
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statistics
from sklearn import impute 
import random

def branch_node(dataset0,X_index,threshold): # split data in node 
        datasetL = dataset0[dataset0[X_index] <= threshold ]
        datasetR = dataset0[dataset0[X_index] > threshold ]
        return (datasetL, datasetR)
def get_split(dataset0): # get split parameters for data on node
    threshvector = np.mean(dataset0,axis=0)
    X_index = None
    X_threshold = None
    bestgini = 100000
    for feature in range (0,len(dataset0.columns)-1):
        sorteddata = dataset0.sort_values(feature).copy(deep = True)
        for index in range (0,len(sorteddata.iloc[:,feature])-1):
            L,R = branch_node(dataset0,feature,(sorteddata.iloc[index,feature]+sorteddata.iloc[index+1,feature])/2)
            if bestgini > get_gini(L['L'],R['L']):
                X_index = feature
                X_threshold = threshvector[feature]
                bestgini = get_gini(L['L'],R['L'])
    print (X_index,X_threshold,bestgini)
    return (X_index,X_threshold)

def forest_split(dataset0): # get split parameters for data on node
    threshvector = np.mean(dataset0,axis=0)
    X_index = None
    X_threshold = None
    bestgini = 100000
    for i in range (0,int((len(dataset0.columns)-1)**.5)):
        featurelist = random.sample(range((len(dataset0.columns)-1)), int((len(dataset0.columns)-1)**.5))
        sorteddata = dataset0.sort_values(featurelist[i]).copy(deep = True)
        for index in range (0,len(sorteddata.iloc[:,featurelist[i]])-1):
            L,R = branch_node(dataset0,featurelist[i],(sorteddata.iloc[index,featurelist[i]]+sorteddata.iloc[index+1,featurelist[i]])/2)
            if bestgini > get_gini(L['L'],R['L']):
                X_index = featurelist[i]
                X_threshold = threshvector[featurelist[i]]
                bestgini = get_gini(L['L'],R['L'])
#    print (X_index,X_threshold,bestgini)
    return (X_index,X_threshold)
        
def get_gini(dataL,dataR): # caclculate gini index
    lenL =len(dataL)
    lenR =len(dataR)
    countL1 = np.sum(dataL)
    countL0 = lenL-countL1
    countR1 = np.sum(dataR)     
    countR0 = lenR-countR1
    if lenL == 0 :
        leftgini = 0
    else:
        leftgini = 1-(countL0/lenL)**2-(countL1/lenL)**2
    if lenR == 0 :
        rightgini = 0
    else:
        rightgini = 1-(countR0/lenR)**2-(countR1/lenR)**2
    combinedgini = (leftgini*lenL+rightgini*lenR)/(lenL+lenR)
    return (combinedgini)





def Split(data,ts=0.20): # parse array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(data,test_size=ts))
                                 
def accuracy(tree,test):
    count = 0
    for i in range (0,len(test)):
        if tree.predict(test.iloc[i]) == test.iloc[i]['L']:
            count += 1
    print ('accuracy of tree on test is: ', count/len(test)*100)
    count = 0
    for i in range (0,len(train)):
        if tree.predict(train.iloc[i]) == train.iloc[i]['L']:
            count += 1
    print ('accuracy of tree on train is: ', count/len(train)*100)

def preparewdcdata(url):
    data = pd.read_csv(url,header=None)
    data = data.iloc[:,1:]
    datalabel = data[1].copy(deep=True)
    data = data.iloc[:,1:]
    labelatend = pd.concat([data, datalabel], axis=1)    
    def labelto01(row):
        if row[1] == 'M':
            val = 1
        else:
            val = 0
        return val
    labelatend[1] = labelatend.apply(labelto01, axis=1)
    labelatend.columns = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 'L']
    return labelatend

def preparepimadata(url):
    pima = pd.read_csv(url,header=None)
    pima.columns = [0,  1,  2,  3,  4,  5,  6,  7 ,'L']
    imputer = impute.SimpleImputer(missing_values = 0, strategy = 'mean')
    imputer.fit(pima.iloc[:,1:8])
    pima_imputed = pima.copy(deep=True)
    pima_imputed.iloc[:,1:8] = imputer.transform(pima.iloc[:,1:8])
    return pima_imputed



def forestaccuracy(forest,test):
    count = 0
    for i in range (0,len(test)):
        if forest.fpredict(test.iloc[i]) == (test.iloc[i]['L']):
            count += 1
    print ('accuracy of forest model on test is: ', count/len(test)*100)
    count = 0
    for i in range (0,len(train)):
        if forest.fpredict(train.iloc[i]) == train.iloc[i]['L']:
            count += 1
    print ('accuracy of forest on train is: ', count/len(train)*100)
    
class node:
    def __init__(self,indata,maxdeep,deep=0):
        self.isbranch = False
        self.indata = indata
        self.deep = deep
        if self.deep == maxdeep:
            if np.sum(self.indata['L']) > len(self.indata['L'])/2:
                self.value = 1
                return
            else :
                self.value = 0
                return
            return
        else :
            if get_gini(self.indata['L'], self.indata['L']) == 0 :
                if np.sum(self.indata['L']) > len(self.indata['L'])/2:
                    self.value = 1
                    return
                else :
                   self.value = 0
                   return
                return
            else : 
                self.isbranch = True
                self.splitXfeature, self.splitXvalue = get_split(self.indata)
                self.left , self.right = branch_node(self.indata,self.splitXfeature, self.splitXvalue)
                self.leftnode = node(self.left,maxdeep,self.deep +1)
                self.rightnode= node(self.right,maxdeep,self.deep +1)
            return      
    def predict(self,value):
        if self.isbranch:
            if value[self.splitXfeature] < self.splitXvalue:
                return self.leftnode.predict(value)
            else:
                return self.rightnode.predict(value)
        else:
            return self.value 
class forest :
    def __init__(self,indata,maxdeep,numberOfTrees):
        self.trees = [()]
        for i in range (0, numberOfTrees):
            newtree = forestnode(indata,maxdeep)
            self.trees.append(newtree)
    def fpredict(self,value):
        fvalue = 0
        for i in range (1,len(self.trees)):
            if self.trees[i].predict(value) == 1 :
                fvalue = fvalue +1
        if fvalue >= len(self.trees)/2:
            return 1
        return 0
    
class forestnode:
    def __init__(self,indata,maxdeep,deep=0):
        if deep == 0:
            print ('new tree')
        self.isbranch = False
        self.indata =  indata.iloc[np.random.randint(len(indata),size=len(indata))]
        self.deep = deep
        if self.deep == maxdeep:
            print (self.deep)
            if np.sum(self.indata['L']) >= len(self.indata['L'])/2:
                self.value = 1
                return
            else :
                self.value = 0
                return
            return
        else :
            if get_gini(self.indata['L'], self.indata['L']) == 0 :
                if np.sum(self.indata['L']) >= len(self.indata['L'])/2:
                    self.value = 1
                    return
                else :
                   self.value = 0
                   return
                return
            else : 
                self.isbranch = True
                self.splitXfeature, self.splitXvalue = forest_split(self.indata)
                self.left , self.right = branch_node(self.indata,self.splitXfeature, self.splitXvalue)
                self.leftnode = forestnode(self.left,maxdeep,self.deep +1)
                self.rightnode= forestnode(self.right,maxdeep,self.deep +1)
            return
    def predict(self,value):
        if self.isbranch:
            if value[self.splitXfeature] < self.splitXvalue:
                return self.leftnode.predict(value)
            else:
                return self.rightnode.predict(value)
        else:
#            print (' leaf value is :', self.value)
            return self.value
        
if __name__ == "__main__": 
    
#    pimainputdata = preparepimadata('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
#    train,test = Split(pimainputdata,0.2)
#    pimaforest = forest(train,3,15)
#    forestaccuracy (pimaforest,test)
#    
#    pimainputdata = preparepimadata('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
#    train,test = Split(pimainputdata,0.2)
#    pimatree = node(train,3)
#    accuracy(pimatree,test)       

    wdcinputdata = preparewdcdata('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
    train,test = Split(wdcinputdata,0.2)
    wdcforest = forest(train,3,10)      
    forestaccuracy (wdcforest,test)
    wdctree = node(train,3)
    accuracy(wdctree,test) 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
#    skforest = RandomForestClassifier(n_estimators=3, max_depth=10,min_samples_leaf = 15)
#    skforest.fit(train.iloc[:,:-1],train.iloc[:,-1])
#    skforest.score(train.iloc[:,:-1],train.iloc[:,-1])
#    skforest.score(test.iloc[:,:-1],test.iloc[:,-1])


    pimainputdata = preparepimadata('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
    train,test = Split(pimainputdata,0.2)
    skforest = RandomForestClassifier(n_estimators=1000, max_depth=5,min_samples_leaf = 10)
    skforest.fit(train.iloc[:,:-1],train.iloc[:,-1])
    skforest.score(train.iloc[:,:-1],train.iloc[:,-1])
    skforest.score(test.iloc[:,:-1],test.iloc[:,-1])