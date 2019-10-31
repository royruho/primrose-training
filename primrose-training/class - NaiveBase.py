# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statistics
from sklearn import impute 


def calculateProbability(x, mean, stdev): 
        exponent  = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return  (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
class Dataset:
    def __init__(self,Data):# input is dataframe
        self.X= Data.iloc[:, :-1].values
        self.y= Data.iloc[:, -1].values
    def Split(self,ts=0.20): # parse iris array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(self.X, self.y,test_size=ts))
                 
class Naivebase:
    def __init__(self,data):# input is dataframe
        self.Xtest = data[1]
        self.ytest = data[3]
        self.X= data[0]
        self.y= data[2]
        self.label1X = self.X[self.y==1]
        self.label0X = self.X[self.y==0]
        self.numberOfFeatueres= len(self.X[0])
        self.p1 = len(self.label1X)/(len(self.label0X)+len(self.label1X))
        self.p0 = len(self.label0X)/(len(self.label0X)+len(self.label1X))
    def calcMean(self,data): # calc mean of vector
        return (np.mean(data,axis=0))
    def calcStd(self,data): # calc std of vector
        return (np.std(data,axis=0))
    def totalMeanStd(self): # stacks mean and std vector to an array
        self.totalMeanStd = np.column_stack((self.calcMean(self.Xtest),self.calcStd(self.Xtest)))
    def Label1MeanStd(self): # stacks mean and std for data with labe 1
        self.label1MeanStd = np.column_stack((self.calcMean(self.label1X),self.calcStd(self.label1X)))
    def Label0MeanStd(self): # stacks mean and std for data with labe 0
        self.label0MeanStd = np.column_stack((self.calcMean(self.label0X),self.calcStd(self.label0X)))
    def Classprob1(self,x,col): # calculate probabity for sample to have label 1
        return calculateProbability(x, self.label1MeanStd[col][0], self.label1MeanStd[col][1])
    def Classprob0(self,x,col): # calculate probabity for sample to have label 0
        return calculateProbability(x, self.label0MeanStd[col][0], self.label0MeanStd[col][1])
    def Calcprobabilties(self): # calculate porbabilty of smaple to be 0 or 1 and precicts label with higher probabilty
            posprob = self.p1
            negprob = self.p0
            for i in range (0,self.numberOfFeatueres):
                posprob = posprob*self.Classprob1(1,i)
                negprob = negprob*self.Classprob0(1,i)
            return 1 if posprob > negprob else 0
#            if posprob > negprob:
#                return 1
#            else :
#                return 0
    def getPredictions(self): # create prdictions for test data
        predictions = []
        for i in range(len(self.ytest)):
            result = self.Calcprobabilties()
            predictions.append(result)
            self.predictions = predictions
    def getAccuracy(self): # count correct prdiction over test data
        self.getPredictions()
        correct = 0
        for x in range(len(self.ytest)):
            if self.ytest[x] == self.predictions[x]:
                correct += 1
        return (correct/float(len(self.ytest))) * 100.0
            
            
            
#########################################################################################
if __name__ == "__main__": 
    pima = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',header=None)
    pima.columns = ['preg num', 'plasma gluc','blood pressure','Triceps skin','serum insulin','Body mass index','Diabetes pedigree function','age','y']
    imputer = impute.SimpleImputer(missing_values = 0, strategy = 'mean')
    imputer.fit(pima.iloc[:,1:8])
    pima_imputed = pima.copy(deep=True)
    pima_imputed.iloc[:,1:8] = imputer.transform(pima.iloc[:,1:8])
    pimaset = Dataset(pima_imputed)
    pimabasemodel = Naivebase(pimaset.Split())
    pimabasemodel.totalMeanStd()
    pimabasemodel.Label0MeanStd()
    pimabasemodel.Label1MeanStd()
    prediction = pimabasemodel.Calcprobabilties()
    print (pimabasemodel.getAccuracy())


#    test =([1,2,3,4,1],[5,6,7,8,0],[9,10,11,12,0],[13,14,15,16,0],[2,1,2,1,0])
#    testdf = pd.DataFrame(list(test), columns=[0,1,2,3,4], index=[0,1,2,3,4])  
#    testset = Dataset(testdf)
#    testmodel = Naivebase(testset.Split())
#    testmodel.totalMeanStd()
#    testmodel.Label0MeanStd()
#    testmodel.Label1MeanStd()
#    testmodel.Calcprobabilties()
#    calculateProbability(10,1,1)
