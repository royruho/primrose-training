# -*- coding: utf-8 -*-
"""
Created on Tue sep 24 12:56:50 2019

@author: royru
"""


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
import kmeans as km
     

class GMM:
    def __init__(self,k,data,init = False): # init - for initiating mu list
        self.k = k # number of clusters
        self.data = data
        self.instances_weights = np.full((len(data),k),1/k) 
        self.phi_sum = list(np.zeros(self.k)) # sum of intances associated with gaussian
        self.mu_list = self.init_mu_list() # init list of k clusters mu's
        if init != False: # init with kmeans cluster centers if given
            for k in range (self.k):
                self.mu_list[k] = init[k][0]
        self.phi_list = np.ones(self.k)/self.k # proportion of instances associated with gaussian
        self.sigma_list = self.init_sigma_list() # init sigma with cov of data
        self.log_liklihood = [] # list of log liklihood after each expection
        
    def init_mu_list(self):
        self.mu_list = []
        for i in range (self.k):
            random_insatnce = np.random.choice(
                    self.data.shape[0], replace=False) # choose random K insatces as initial centers
            self.mu_list.append(self.data[random_insatnce])
        return self.mu_list
    
    def update_mu_list(self): # list of k clusters mu's
        mu_list = self.mu_list
        for k in range (self.k): # set mu list to 0
            mu_list[k] = mu_list[k] - mu_list[k]
        for i in range (len(self.data)): # add mu associated with gaussian over for every instace 
            for k in range (self.k):
                mu_list[k] += (self.instances_weights[i,k]*self.data[i]/(self.phi_sum[k]))
        self.mu_list = mu_list
        
    def update_phi_sum(self): # sum of intances associate with gaussian
        for  k in range (self.k):
            self.phi_sum[k] = np.sum(self.instances_weights[:,k],axis = 0)
            
    def update_phi_list(self): # proportion of instances associated with gaussian
        for  k in range (self.k):
            self.phi_list[k] = self.phi_sum[k]/len(self.data)
        return self.phi_list
    
    def init_sigma_list(self): # init sigma with cov of data
        self.sigma_list = []
        for k in range (self.k):
            self.sigma_list.append(np.cov(self.data.T))
        return self.sigma_list
        
    def calc_sigma(self,k): # return sigma matrix
        x = (self.data-self.mu_list[k]) # center data
        w = np.sqrt(self.instances_weights[:,k]) # sqrt so when multuplyed become ^1
        w = np.reshape(w,(len(self.data),1))
        wx = np.multiply(w,x) # multiply element wise => vector x stays the same size
        wx = (1/(self.phi_sum[k]**0.5))*(wx) # sqrt so when multuplyed become ^1 
        sigma = wx.T@wx
        return sigma
    
    def update_sigma_list(self): # update sigma matrix for every k
        for k in range (self.k):
            self.sigma_list[k] = self.calc_sigma(k)

    def probability(self,datum,mu,sigma): # return liklihood of datum from gaussian 
        var = multivariate_normal(mu,sigma,allow_singular =True)
        return var.pdf(datum)
              
    def expectation(self): # updates instances_weights and add a log_liklihood entry
        log_liklihood = 0
        for i in range (len(self.data)):
            for k in range (self.k):
                self.instances_weights[i,k] = self.phi_list[k]*self.probability(
                        self.data[i],self.mu_list[k],self.sigma_list[k])
            log_liklihood += -np.log(np.sum(self.instances_weights[i])) # - log to get positive numbers
            self.instances_weights[i] = self.instances_weights[i]/np.sum(self.instances_weights[i])
        self.log_liklihood.append(log_liklihood)
    
    def maximization(self): # maximization step
        self.update_phi_sum()
        self.update_mu_list()
        self.update_phi_list()
        self.update_sigma_list()
            
    def fit(self,iterration,delta = 0.0001): # run expectation maximization
        self.expectation()
        self.maximization()
        delta_log_liklihood = self.log_liklihood[0]
        i = 0
        while i < iterration and delta_log_liklihood > delta: #  iterrate until stop criteria reached 
            self.expectation()
            self.maximization()
            delta_log_liklihood = self.log_liklihood[i] - self.log_liklihood[i+1] # eache step decreases value of -log liklihood
            print (i,self.log_liklihood[i+1], delta_log_liklihood)
            i = i +1
                            
    def label_data(self,datum): # return (k) probabiltis of instance association to each k
        prob = []
        for k in range (self.k):
            prob.append(self.probability(datum,self.mu_list[k],self.sigma_list[k]))
        return (prob)
    
    def plot_classes_2d(self,labels): # plot self.data colored by labels
        cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple' }
        for i in range (len(labels)):
            plt.plot(self.data[i,0],self.data[i,3],'o',c=cdict[labels[i]]) #plot data
        for k in range (self.k): # for every K print 2 features of data and centers
            plt.plot(self.mu_list[k][0],self.mu_list[k][3],'D' ,c = cdict[k],markersize=15) # plot Kmean 
        plt.show()    

def predict(gmm_model,data): # predict data according to gmm_model
    proba = []
    label = []
    for i in range (len(data)):
        a = gmm_model.label_data(data[i])
        proba.append(a)
        label.append(proba[i].index(max(proba[i])))
    return label,proba # label and tuple with all k-clusters probabolitis for eacg instance
        
def get_precision(true_labels,predicted_labels): # works only with supervised
    counter = 0
    for i in range (len(true_labels)):
        if true_labels[i] == predicted_labels[i] :
            counter += 1
    print ('precision over {} samples is {}%'.format(i,counter/i*100))
    
            
########################################################################################
if __name__ == "__main__": 
    iris = datasets.load_iris()
    X = iris.data[:,:4]  # take 4 first features
#    X = scale(X,axis = 0) # scale data mean = 0 , std = 1
    y = iris.target # get labels 
    kmeans_cls = km.Kmeans_train(X,5,3) # train 5 k=3 k-means models and pick the best 1
    init_mu_with_kmeans = kmeans_cls.get_centers() # get centroids of k-means model
    g0 = GMM(3,X) # 3 clusters GMM clasiffier
    g0.fit(1000) # maximum 1000 itirations
    g1 = GMM(3,X,init_mu_with_kmeans) # 3 clusters GMM clasiffier initiated with k-means class centers
    g1.fit(1000) # maximum 1000 itirations
    p0 = predict(g0,X) # labels with g0 model 
    p1 = predict(g1,X) # labels with g1 model 
    print ('predicted labels:')
    g0.plot_classes_2d(p0[0])
    print ('predicted labels with init from kmeans:')
    g1.plot_classes_2d(p1[0])
    print ('true labels:')
    g1.plot_classes_2d(y)

