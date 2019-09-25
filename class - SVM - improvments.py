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

def prepareiris(data): # return data with first 2 features, class label set to 1 , -1  and normalize featues
    iris = pd.read_csv(data)
    iris.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
    def f(row):
        if row['class'] == 'Iris-setosa':
            val = 1
        else:
            val = -1
        return val
    iris['label'] = iris.apply(f, axis=1)
    iris = iris[['sepal_length', 'sepal_width','label']]
    iris.columns = [1,2,'L']
    for i in range (1,len(iris.columns)):    
        normalized = (iris[i]-iris[i].mean())/iris[i].std() # normlize data
        iris[i]=normalized
    corr = iris.corr() # print corr matrix
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
    return (iris) # pandas array 
    
def Split(data,ts=0.20): # parse array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=ts))
                                 
class SVM:
    def __init__(self,dataX,datay,input_learning_rate=0.3,irange=10):# input is  X,y
        self.X=dataX
        self.y=datay     
        self.theta0=teta0
        self.learning_rate = input_learning_rate
        self.irange = irange
        self.listoflos = ([])
        self.new_teta = self.theta0
        self.gradient = 0
        self.shape = (1,len(self.theta0))
        sumofloss =0
        for iteration in range (0,self.irange):
            if iteration % 50 == 0 :
                print ('#',iteration,' sumofloss',sumofloss, ' learning rate and slack features value decreased')
                self.learning_rate = self.learning_rate*0.75       
            sumofloss =0
            sumgradient = 0
            for i in range (0, len(self.X)):
                sumofloss = sumofloss + self.hindgeloss(i)
                sumgradient = sumgradient + self.calcgradient(i)
#                print (' self.calcgradient(i)',  self.calcgradient(i))
#                print ('sumgradient',sumgradient)
            self.listoflos.append(sumofloss/i)
            self.gradient = sumgradient / i
            self.new_teta = self.updateteta(iteration)
        print ('finel teta is :' , self.new_teta, 'theta norma = ' ,np.linalg.norm(self.new_teta[1:]))

            
#        print ('final teta:',new_teta)
#        self.prediction = (probability(self.finaltheta,self.testX)).flatten()
#        self.predicetedy= (self.prediction >= 0.5).astype(int)
#        self.accuracy= np.mean(self.predicetedy ==  self.testy)*100

    def hindgeloss(self,index):
        loss = float(np.dot(np.reshape(self.X[index],self.shape),self.new_teta))
#        print ('loss' , loss)
        loss = loss *self.y[index]
#        print ('self.y[index]' , self.y[index])
        return np.maximum(0, 1-loss)
    def calcgradient(self,index):
        if self.y[index]*float(np.dot(np.reshape(self.X[index],self.shape),self.new_teta))<1: #*self.learning_rate:
            return -self.y[index]*self.X[index]
        else:
            return 0
    def updateteta(self,iteration):
        theta = self.new_teta
        return (np.reshape(theta.T - (self.learning_rate*self.gradient),np.shape(self.theta0)))         
        
    def model_line(self): #create model line y = ax+b - adds it to  a plot
        final_teta = self.new_teta
        modelline = -final_teta[2]/final_teta[6]
        xline= np.arange(-5,5)
        yline = xline*modelline-final_teta[0]/final_teta[6]
        plt.plot(xline,yline)

    def model_linei(self): #create model line y = ax+b - adds it to  a plot
        final_teta = self.new_teta
        modelline = -final_teta[1]/final_teta[2]
        xline= np.arange(-5,5)
        yline = xline*modelline-final_teta[0]/final_teta[2]
        plt.plot(xline,yline)

        
    def plot_sumoflos(self):
        _plot = np.reshape(self.listoflos,(len(self.listoflos),1))
        _iterration = np.reshape(np.arange(0,self.irange),(self.irange,1))
#        _plot = np.column_stack((_iterration,_plot))
        plt.plot(_iterration,_plot)
        plt.show()
        
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
    
def randomize_theta(features): #create a radom theta0 array
    return (10*np.random.rand(features,1))

def preparepima(data):
    pima = pd.read_csv(data,header=None)
    pima.columns = ['preg num', 'plasma gluc','blood pressure','Triceps skin','serum insulin','Body mass index','Diabetes pedigree function','age','y']
    imputer = impute.SimpleImputer(missing_values = 0, strategy = 'mean')
    imputer.fit(pima.iloc[:,1:8])
    pima_imputed = pima.copy(deep=True)
    pima_imputed.iloc[:,1:8] = imputer.transform(pima.iloc[:,1:8])
#    corr = pima_imputed.corr()
#    ax = sns.heatmap(
#    corr, 
#    vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200),
#    square=True
#    )
#    ax.set_xticklabels(
#    ax.get_xticklabels(),
#    rotation=45,
#    horizontalalignment='right'
#    );
    pima_imputed.columns = [1,2,3,4,5,6,7,8,'y']
    for i in range (1,9):    
        normalized = (pima_imputed[i]-pima_imputed[i].mean())/pima_imputed[i].std()
        pima_imputed[i]=normalized
    pima_imputed.loc[pima_imputed['y'] == 0,['y']] = -1
    corr1 = pima_imputed.corr()
    ax = sns.heatmap(
    corr1, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=201),
    square=True
    )
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
    );
    plt.show()
    return (pima_imputed) # an array 


#########################################################################################
if __name__ == "__main__":
    irisdata = prepareiris('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    trainX,testX,trainy,testy = Split(irisdata)
    trainX = np.c_[np.ones((trainX.shape[0], 1)), trainX]
    testX = np.c_[np.ones((testX.shape[0], 1)), testX]
    trainy = trainy.to_numpy()
    testy = testy.to_numpy()
    teta0 = randomize_theta(3)
    fitmodel = SVM(trainX,trainy,0.8,501)
    fitmodel.plot_sumoflos()
    accuracy(fitmodel,trainX,trainy)
    accuracy(fitmodel,testX,testy)
    irisdata.plot.scatter(x=1, y=2, c='L', cmap ='rainbow')
    fitmodel.model_linei()
    plt.show()
    
    pimadata = preparepima('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
    trainX,testX,trainy,testy = Split(pimadata)
    trainX = np.c_[np.ones((trainX.shape[0], 1)), trainX]
    testX = np.c_[np.ones((testX.shape[0], 1)), testX]
    trainy = trainy.to_numpy()
    testy = testy.to_numpy()
    teta0 = randomize_theta(9)
    fitmodel = SVM(trainX,trainy,0.8,501)
    fitmodel.plot_sumoflos()
    accuracy(fitmodel,trainX,trainy)
    accuracy(fitmodel,testX,testy)
    pimadata.plot.scatter(x=2, y=6, c='y', cmap ='rainbow')
    fitmodel.model_line()
    plt.show()