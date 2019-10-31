# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:56:50 2019

@author: royru


is there a point in unassigned data list list?
is point core? 
    move to class list
    move nighbor points to class list
    check if neighbor points are new cors
        assign and check till no more new core points
    assign all points in list to class
    remove points from unassigned data list
back to start
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#from sklearn import impute



def prepareiris(data): # returns iris data as dataframe
    iris = pd.read_csv(data)
    iris.columns = ['sepal_length', 'sepal_width','petal_length',' petal_width','class']
    iris.columns = [1,2,3,4,'L']
    return (iris) # an array 

#def preparepima(data): # returns pima data as dataframe
#    pima = pd.read_csv(data,header=None)
#    pima.columns = ['preg num', 'plasma gluc','blood pressure','Triceps skin','serum insulin','Body mass index','Diabetes pedigree function','age','y']
#    imputer = impute.SimpleImputer(missing_values = 0, strategy = 'mean')
#    imputer.fit(pima.iloc[:,1:8])
#    pima_imputed = pima.copy(deep=True)
#    pima_imputed.iloc[:,1:8] = imputer.transform(pima.iloc[:,1:8])
#    pima_imputed.columns = [1,2,3,4,5,6,7,8,'L']
#    return (pima_imputed) # an array 
    
def Split(data,ts=0.20): # parse array to trainX testX train_y test_y 
            from sklearn.model_selection import train_test_split
            return(train_test_split(data[:,:-1],data[:,-1],test_size=ts))


class DBscan:
    ''' input: unlabeled data array, epsilon , min core plots)
        output: none '''
    def __init__(self,data,epsilon,core_number):# input is  X,K
        self.X=data
        self.epsilon=epsilon
        self.core_number = core_number  
        self.labels = [0]*len(self.X) # hold labels of data. -1 Noise, 0 unasigned yet, class_number
        self.class_count = 0
        self.initilize_db_scan()
        self.stack_labels() # add labels to data array
        
    def is_core_point(self,point): # chech if point is a core point in dataset
        epsilon_distnace_points = 0
        for i in (self.X):
            if self.calcdist(i,point) <= self.epsilon and self.calcdist(i,point) != 0 :
                epsilon_distnace_points +=1
        if epsilon_distnace_points >= self.core_number:
            return True
        else:
            return False
        
    def calcdist(self,point,center): # returns distance between 2 points
        return np.sqrt(np.sum((point-center)**2))
      
    def fetch_epsilon_indexes(self,point):
        ''' return index number in self.X of epsilon neigbors
        '''
        epsilon_points_indexes = []
        for i in range (len(self.X)):
            if self.calcdist(self.X[i],point) <= self.epsilon and self.calcdist(self.X[i],point) != 0 :
                epsilon_points_indexes.append(i)
        return epsilon_points_indexes
               
    def initilize_db_scan(self):
        for p in range(0, len(self.X)):
            if self.labels[p] != 0 :
                continue
            else:
                if not self.is_core_point(self.X[p]):
                    self.labels[p] = -1
                else: 
                    self.class_count += 1
                    self.grow_cluster(p)
                    
    def grow_cluster(self,seed_index):
        p = seed_index
        self.labels[p] = self.class_count
        self.neighbor_pts = self.fetch_epsilon_indexes(self.X[p])
        i = 0
        while i < (len(self.neighbor_pts)): # still points to calssifei in current class
            next_point_index = self.neighbor_pts[i]
            if self.labels[next_point_index] == -1: # class points classified earlier as noise
                self.labels[next_point_index] = self.class_count
            elif self.labels[next_point_index] == 0:
                self.labels[next_point_index] = self.class_count
                if self.is_core_point(self.X[next_point_index]): # if next point is core - add its nigbors to class
                    self.neighbor_pts.extend(self.fetch_epsilon_indexes(self.X[next_point_index]))
            i +=1        
            
    def stack_labels(self):
        self.labels_arr = np.reshape(self.labels,(len(self.labels),1))
        self.labeld_data = np.column_stack((self.X,self.labels_arr))

    def plot_classes(self,three_d=False): 
        ''' input 3d plot or 2d plot'''
        x = self.labeld_data[:,0]
        y = self.labeld_data[:,1]
        z = self.labeld_data[:,2]
        labels = self.labeld_data[:,-1]
        if three_d :
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(x,y,z, c =  labels , cmap = plt.viridis())
            fig.colorbar(img)
            plt.show()
        else:
            img = plt.figure(figsize=(9,6))
            plt.scatter(x,y, c =labels,cmap ='rainbow') #plot data
            plt.show()           

            
        
        
  

#########################################################################################
if __name__ == "__main__":
    irisdata = prepareiris('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    irisdata_numpy = irisdata.to_numpy()
    trainX,testX,trainy,testy = Split(irisdata_numpy,0) # split data - for unsupervised - test size - 0 
    three_d = True # print 3d plot of clasified data
    iadbscan_classifier_fit = DBscan(trainX,0.2,5) # DBscan clf object epislon length =0.2 min_points = 5
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.4,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.6,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.8,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    three_d = False
    iadbscan_classifier_fit = DBscan(trainX,0.2,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.4,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.6,5)
    iadbscan_classifier_fit.plot_classes(three_d)
    iadbscan_classifier_fit = DBscan(trainX,0.8,5)
    iadbscan_classifier_fit.plot_classes(three_d)
#    pimadata = preparepima('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
#    pimadata = pimadata.to_numpy()
#    trainX,testX,trainy,testy = Split(pimadata,0)
#    trainX,testX,trainy,testy = Split(pimadata)
#    padbscan_classifier_fit = DBscan(trainX,4,5)
#    padbscan_classifier_fit.plot_classes(three_d)
#    padbscan_classifier_fit = DBscan(trainX,25,5)
#    padbscan_classifier_fit.plot_classes(three_d)
#    padbscan_classifier_fit = DBscan(trainX,75,3)
#    padbscan_classifier_fit.plot_classes(three_d)
#    padbscan_classifier_fit = DBscan(trainX,100,5)
#    padbscan_classifier_fit.plot_classes(three_d)
#    
