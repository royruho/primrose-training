# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""
#
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
#iris.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
#print (iris)
#def f(row):
#    if row['class'] == 'Iris-setosa':
#        val = 1
#    else:
#        val = 0
#    return val
#iris['label'] = iris.apply(f, axis=1)
#Isetosa = iris.loc[iris['label'] == 1]
#Notsetosa = iris.loc[iris['label'] == 0]
#
#plt.scatter(Isetosa['sepal_length'], Isetosa['sepal_width'], label='Isetosa')
#plt.scatter(Notsetosa['sepal_length'], Notsetosa['sepal_width'], label='Notsetosa')
#plt.legend()
#plt.show()
#iris.plot.scatter(x='sepal_length', y='sepal_width', c='label')
#print (iris)



#Secondly, creating the model:.
#1. Create a function which calculates the logit function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all='ignore')

def plot_graph(_figure,_xlabel,_ylabel,_XX):
    plt.figure(_figure)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_figure)
    plt.plot(_XX[:,0],_XX[:,1],'o')
    plt.show()

      
def parse_data(_iris,_frac): # parse iris array to train and test 
    DF=_iris.copy(deep=True)
    DF.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
    def f(row):
        if row['class'] == 'Iris-setosa':
            val = 1
        else:
            val = 0
        return val
    DF['label'] = DF.apply(f, axis=1)
    train=DF.sample(n=None,frac=_frac)
    test=DF.drop(train.index)
    Xtrain = train[['sepal_length', 'sepal_width']].copy()
    ytrain = train[['label']].copy()
    Xtest =  test[['sepal_length', 'sepal_width']].copy()
    ytest = test[['label']].copy()
    Xtrain = np.c_[np.ones((Xtrain.shape[0], 1)), Xtrain]
    Xtest = np.c_[np.ones((Xtest.shape[0], 1)), Xtest]
    ytrain = np.reshape((ytrain),(len(Xtrain),1))
    ytest = np.reshape((ytest),(len(Xtest),1))
    ytrain= ytrain.to_numpy()
    ytest= ytest.to_numpy()
    return (Xtrain,ytrain,Xtest,ytest) # an array 
    

def sigmoid(x): # Activation function used to map any real value between 0 and 1
    return 1. / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0] #number of samples
    total_cost = -(1 / m) * np.sum(
            y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x))) #mean log of loss over all samples 
    return total_cost
def compute_gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,x))-y)

def update_teta_logR(_old_teta,_learning_rate,_gradient): #update teta vector in gradient descent model
    return  _old_teta-_learning_rate*_gradient

class FitOutput:
    def __init__(self,Data,theta0,input_learning_rate=0.1,irange=100):# input is  X,y,theta0
        self.testX=Data[2]
        self.testy=Data[3].flatten()
        self.X=Data[0]
        self.y=Data[1]        
        self.theta0=theta0
        self.learning_rate = input_learning_rate
        self.irange = irange
        self.listoflos = ([])
        learning_rate = input_learning_rate # LR0.01
        new_teta = self.theta0
        gradient = compute_gradient(new_teta,self.X,self.y)
        for i in range (0,self.irange):
            gradient = compute_gradient(new_teta,self.X,self.y)
            new_teta = update_teta_logR(new_teta,learning_rate,gradient)
            self.listoflos.append(cost_function(new_teta,self.X,self.y))
        print ('final teta:',new_teta)
        self.finaltheta = new_teta
        self.prediction = (probability(self.finaltheta,self.testX)).flatten()
        self.predicetedy= (self.prediction >= 0.5).astype(int)
        self.accuracy= np.mean(self.predicetedy ==  self.testy)*100
        
        
    def model_line(self,_iris): #create model line y = ax+b - adds it to  a plot
        final_teta = self.finaltheta
        modelline = -final_teta[1,0]/final_teta[2,0]
        xline= np.arange(0,10)
        yline = xline*modelline-final_teta[0,0]/final_teta[2,0]
        plt.plot(xline,yline)
        DF=_iris.copy(deep=True)
        DF.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
        def f(row):
            if row['class'] == 'Iris-setosa':
                val = 1
            else:
                val = 0
            return val
        DF['label'] = DF.apply(f, axis=1)
        Isetosa = DF.loc[DF['label'] == 1]
        Notsetosa = DF.loc[DF['label'] == 0]
        plt.scatter(Isetosa['sepal_length'], Isetosa['sepal_width'], label='Isetosa')
        plt.scatter(Notsetosa['sepal_length'], Notsetosa['sepal_width'], label='Notsetosa')
        plt.legend()
        plt.show()
        
    def plot_sumoflos(self):
        _plot = np.reshape(self.listoflos,(len(self.listoflos),1))
        _iterration = np.reshape(np.arange(0,self.irange),(self.irange,1))
        _plot = np.column_stack((_iterration,_plot))
        plot_graph(1,'iterattion number','Cost',_plot)

def randomize_theta(): #create a radom theta0 array
    return (10*np.random.rand(3,1))


#########################################################################################
if __name__ == "__main__": 
    iris0 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    iris1 = iris0.copy()# load data frame from file
    Data = parse_data(iris1,.8) # parse data to train 0.8 / test 0.2 fractions
    shortrun = FitOutput(Data,randomize_theta(),0.1,25) # create innstance of shrot(25) iterrations
    shortrun.plot_sumoflos() #print sum pf loss against iterrations plot
    shortrun.model_line(iris1) # print model against original data
    print('test accuracy:',shortrun.accuracy,'\n') #print accuracy of short model 
    finalrun = FitOutput(Data,randomize_theta(),0.1,250) # create innstance of long (250) iterrations
    finalrun.model_line(iris1)  # print model against original data
    print('test accuracy:',finalrun.accuracy) # print accuracy of longer itteration model
    





    
 
#########
#2. Create a function which calculates the negative log likelihood function (which is the
#cost/loss function )
#3. Create a function which calculates the gradient :
#4. Create a function for train-test split by a percentage
#5. Create the training function (fit function) for the logistic regression model, which starts
#from some random values for the parameters of theta, and performs gradient descent to
#minimize the loss.
#6. Train your classifier with enough iterations and see if it manages to classify the data
#correctly (try different learning rates, epochs and initializations for theta), split your data
#into train and test sets and print out the precision of prediction on each after training.
#7. Print the best train and test accuracy.
#
