# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""
import copy
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
    iris = iris[['sepal_length', 'sepal_width','petal_length',' petal_width','label']]
    iris.columns = [1,2,3,4,'L']
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


class Kfold:
    def __init__(self,dataX,Ksize):# input is  X,K
        self.X=dataX
        self.Ksize=Ksize
        self.dict = {} #key calues are classes, each key paired with a list of related intances
        self.class_center = {} # dic of class center
        self.initializeK() # assign K random insatces as first K centers
#        self.class_center = copy.deepcopy(self.dict) 
        self.old_center = copy.deepcopy(self.class_center)
        self.clasify_instances()
        self.update_centers()

                
    def fit(self):
        loop = 0
        while self.compare_center() and loop < 50 :
            self.old_center = copy.deepcopy(self.class_center)
            print ('round', loop)
            self.clear_dict()
            self.clasify_instances()
            self.update_centers()
            loop +=1
        print (self.class_center)

    def compare_center(self): # return true if center was updated in the last loop
        for i in range (0,self.Ksize):
            if ((self.class_center[i]) != (self.old_center[i])).all() :
                print('true')
                return True
#        print (self.class_center[i][0])
#        print (self.dict[i][0])
        return False
    
    def update_centers(self): # update self.class_center with mean of class features
        center_counter = 0
        for i in self.dict :
            new_center = []
            if len(self.dict[i]) > 1:
                new_center.append(np.mean(self.dict[i],axis=0))
            print ('i', i, ':', new_center)
            self.class_center.update({center_counter: np.asarray(new_center)})
            center_counter += 1
        
    def clasify_instances(self): # itirates each instace and assign it to nearest self.class_center
        self.clear_dict()
        for i in range (0,len(self.X)):
            mindist = []
            for icenter in range (0,self.Ksize):
                k_center = self.class_center[icenter]
                mindist.append(self.calcdist(k_center,self.X[i]))
            classindex = mindist.index(min(mindist))
#            print (classindex)
#            print(self.X[i])
            if len(self.dict[classindex]) == 0: # if empty - initilize np.array
                self.dict.update({classindex: np.asarray(self.X[i])})
            else:                
                self.dict[classindex] = np.vstack([self.dict[classindex], self.X[i]]) # else - append instace to array
        print ('number of instances in each class are:') 
        for i in range (0,self.Ksize):
            print ('key {} : {}'.format(i,len(self.dict[i])))
    def clear_dict(self): # clears self.dict from old instances
        for key_index in range (0,self.Ksize):
            self.dict.update({key_index: []})
            
    def initializeK(self):
        randomindex = np.random.choice(self.X.shape[0], self.Ksize , replace=False)
        key_index = 0
        for i in randomindex:
            self.class_center.update({key_index: np.asarray(self.X[i])})
            key_index += 1
#        print (self.dict)

    def calcdist(self,point,center): # returns distance 
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
    irisdata_numpy = irisdata.to_numpy()
    trainX,testX,trainy,testy = Split(irisdata_numpy,0)
    kfit = Kfold(trainX,3)
    kfit.fit()
### visualize on iris : 
    ### plot classes as classified by kfit    
    class0 = kfit.dict[0]
    class1 = kfit.dict[1]
    class2 = kfit.dict[2]
    center0 = kfit.class_center[0][0][:-2] # center calculated by model
    center1 = kfit.class_center[1][0][:-2]
    center2 = kfit.class_center[2][0][:-2]
    plt.plot(class0[:,0],class0[:,1], 'ro')
    plt.plot(class1[:,0],class1[:,1], 'bo')
    plt.plot(class2[:,0],class2[:,1], 'go')
    plt.plot(center0[0],center0[1], 'rD',  markersize=12)
    plt.plot(center1[0],center1[1], 'bD',  markersize=12)
    plt.plot(center2[0],center2[1], 'gD',  markersize=12)
    plt.show()     
    
    # plot data centers on original data :
    irisdata.plot.scatter(x=1, y=2, c='L', cmap ='rainbow')     
    plt.plot(center0[0],center0[1], 'rD',  markersize=12)
    plt.plot(center1[0],center1[1], 'bD',  markersize=12)
    plt.plot(center2[0],center2[1], 'gD',  markersize=12)
    plt.show()
#    
#    di = {}
#    k=5
#    for i in range(k):
#        di[i] = i
#    
#    k = [1,2,3,4]