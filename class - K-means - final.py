# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru
"""
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#np.seterr(all='ignore')

def prepareiris(data): # return Iris data as pandas array with class labels transformed from string to  1/2/3
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
    return (iris) # an array 
    
def Split(data,ts=0.20): # parse array to train and test 
            from sklearn.model_selection import train_test_split
            return(train_test_split(data[:,:-1],data[:,-1],test_size=ts))


class Kmeans:
    def __init__(self,data,k_size,plot_class = False):# input is  X,K
        self.X=data
        self.Ksize=k_size
        self.dict = {} #key calues are classes, each key paired with a list of related intances
        self.class_center = {} # dic of class center
        self.initialize() # assign K random insatces as first K centers
        self.old_center = copy.deepcopy(self.class_center)
        self.clasify_instances()
        self.update_centers()
        self.variance = None
        self.fit(plot_class)
        
    def fit(self,plot_class = False):
        loop = 0
        while self.compare_center() and loop < 50 :
            self.old_center = copy.deepcopy(self.class_center)
            print ('round', loop)
            self.clear_dict()
            self.clasify_instances()
            self.update_centers()
            loop +=1
            if plot_class:
                self.plot_classes_2d()
                print (self.class_center)
                print (np.sum(self.total_distance()))
                print ("press enter to view next step")
                input ()
        self.variance = np.sum(self.total_distance())
        
    def compare_center(self): # return true if center was updated in the last loop
        for i in range (0,self.Ksize):
            return ((self.class_center[i]) != (self.old_center[i])).all()
    
    def update_centers(self): # update self.class_center with mean of class features
        center_counter = 0
        for i in self.dict :
            new_center = []
            if len(self.dict[i]) > 1:
                new_center.append(np.mean(self.dict[i],axis=0))
            print ('i', i, ':', new_center)
            self.class_center.update({center_counter: np.asarray(new_center)})
            center_counter += 1
        
    def clasify_instances(self): #itirates each instace and assign it to nearest self.class_center
        self.clear_dict()
        for i in range (0,len(self.X)):
            mindist = []
            for icenter in range (0,self.Ksize):
                k_center = self.class_center[icenter]
                mindist.append(self.calcdist(k_center,self.X[i]))
            classindex = mindist.index(min(mindist))
            if len(self.dict[classindex]) == 0: # if empty - initilize np.array
                self.dict.update({classindex: np.asarray(self.X[i])})
            else:                
                self.dict[classindex] = np.vstack([self.dict[classindex], self.X[i]]) # else - append instace to array
#        print ('number of instances in each class are:') 
#        for i in range (0,self.Ksize):
#            print ('key {} : {}'.format(i,len(self.dict[i])))
                
    def clear_dict(self): # clears self.dict from old instances
        for key_index in range (0,self.Ksize):
            self.dict.update({key_index: []})
            
    def initialize(self):
        randomindex = np.random.choice(self.X.shape[0], self.Ksize , replace=False) # choose random K insatces as initial centers
        key_index = 0
        for i in randomindex:
            self.class_center.update({key_index: np.asarray(self.X[i])})
            key_index += 1

    def calcdist(self,point,center): # returns distance 
        return np.sqrt(np.sum((point-center)**2))
        
    def total_distance(self): # calculate sum of point distances in each cluster from cluster center
        sum_distances = []
        for kcenter in self.class_center:
            class_distance = 0 
            for i in range (0,len(self.dict[kcenter])):
                class_distance = class_distance + self.calcdist(self.dict[kcenter][i],self.class_center[kcenter]) #sum distnaces for each instance in class
                sum_distances.append(class_distance) # add sum of distaces of cluster to list
        return sum_distances # return a list with sum of distances

    def plot_classes_2d(self):
        ax = plt.gca()
        for i in range (0,self.Ksize): # for every K print 2 features of data and centers
            color = next(ax._get_lines.prop_cycler)['color'] # get color of next plot inorder for marker and line to be with same color
            plt.plot(self.dict[i][:,0],self.dict[i][:,1],'o',color = color) #plot data
            plt.plot(self.class_center[i][0][0],self.class_center[i][0][1],'D' ,color = color, markersize=12) # plot Kmean 
        plt.show()    
                    
class Kmeans_train(Kmeans):
    ''' train a number of instances of Kmeans calss and saves
    the instance with smallest variance '''
    def __init__(self,dataX,number_of_instances,Ksize):
        super().__init__(dataX,Ksize) 
        kmeans_array = []
        self.total_sum_array = []
        for i in range (number_of_instances): # for each instance
            kmeans_array.append(Kmeans(dataX,Ksize)) #create KMeans
            self.total_sum_array.append(kmeans_array[i].variance) # add Kmeans.varince to list
        self.best_fit_index =  self.total_sum_array.index(min(self.total_sum_array)) # return index of kmenas instance with lowest variance
        self.best_kmenas = kmeans_array[self.best_fit_index]
        self.dict = self.best_kmenas.dict
        self.class_center = self.best_kmenas.class_center
        self.variance = self.best_kmenas.variance
    def get_kmenas(self):
        return self.best_kmenas
    def get_centers(self):
        return self.best_kmenas.class_center
        



#########################################################################################
if __name__ == "__main__":
    irisdata = prepareiris('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    irisdata_numpy = irisdata.to_numpy()
    trainX,testX,trainy,testy = Split(irisdata_numpy,0)
    kfit = Kmeans(trainX,6,plot_class=True) # create kmenas instace plot each step = true
    train = Kmeans_train(trainX,5,3) # train 5 kmenas instances
    best_kmenas = train #  return the best kmenas parameters from kmenas_train
    best_kmenas.plot_classes_2d() #plot classes

### visualize  3 centers on iris : 
    center =  train.get_centers()
    center0 = center[0][0]
    center1 = center[1][0]
    center2 = center[2][0]
    # plot data centers on original iris data labels :
    irisdata.plot.scatter(x=1, y=2, c='L', cmap ='rainbow')     
    plt.plot(center0[0],center0[1], 'rD',  markersize=12)
    plt.plot(center1[0],center1[1], 'bD',  markersize=12)
    plt.plot(center2[0],center2[1], 'gD',  markersize=12)
    plt.show()


