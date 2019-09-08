# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statistics
from sklearn import impute 
import seaborn as sns

np.seterr(all='ignore')

def prepareiris(data):
    iris = pd.read_csv(data)
    iris.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
    def f(row):
        if row['class'] == 'Iris-setosa':
            val = 1
        elif  row['class'] == 'Iris-versicolor':
            val = 2
        else :
            val = 3
        return val
    iris['label'] = iris.apply(f, axis=1)
    iris = iris[['sepal_length', 'sepal_width','label']]
    iris.columns = [1,2,'L']
    corr = iris.corr()
    ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=201),
    square=True)
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right');
    plt.show()
    return (iris) # an array 
    
def Split(data,ts=0.20): # parse array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(data[:,:-1],data[:,-1],test_size=ts))

#class center:
#    def __init__(self,dataX,center):
#        self.dataX = dataX
#        self.center = center
#        self.distances = [()]
#        for i in range (0,len(dataX)):
#            self.distances.append(self.dataX[i])     
                              
class Kfold:
    def __init__(self,dataX,Ksize):# input is  X,K
        self.X=dataX
        self.Ksize=Ksize
        self.dict = {}
        self.krange = np.arange(1,self.Ksize+1)
        self.Kcenters = [()]
        self.initializeK()
        for i in self.X:
            mindist = []
            for icenter in range (1,self.Ksize+1):
                center = self.dict[icenter][0]
                mindist.append(self.calcdist(center,i))
                print (self.calcdist(center,i))
            print('min', mindist.index(min(mindist)))

        
    def initializeK(self):
        randomindex = np.random.choice(self.X.shape[0], self.Ksize , replace=False)
        counter = 1
        for i in randomindex:
            self.Kcenters.append(self.X[i])
            self.dict.update({counter: [self.X[i]]})
            counter += 1

    def calcdist(self,point,center):
        return np.sqrt(np.sum((point-center)**2))
        

        
            
        

    
def predict(fit,x):
    if fit.new_teta.T@x > 0:
        return 1
    else:
        return -1
        
def accuracy(fit,data,label):
    counter = 0 
    for i in range (0,len(data)):
        predicted = predict(fit,data[i])
        if predicted == label[i]:
            counter = counter+1
    print ('accuraccy is' , counter , ' of ' , len(data), ' ', counter/len(data)*100)
    



#########################################################################################
if __name__ == "__main__":
    irisdata = prepareiris('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    irisdata.plot.scatter(x=1, y=2, c='L', cmap ='rainbow')
    plt.show()
    irisdata = irisdata.to_numpy()
    trainX,testX,trainy,testy = Split(irisdata)
    kfit = Kfold(trainX,3)
    
    
    
