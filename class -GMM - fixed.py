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


class Gaussian:
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    def probability(self,datum):
        var = multivariate_normal(self.mu, self.sigma)
        return var.pdf(datum)
        
        

class GMM:
    def __init__(self,k,data):
        self.k = k
        self.data = data
        self.instances_weights = np.full((len(data),k),1/k)
        self.phi_sum = list(np.zeros(self.k))
        self.mu_list = self.init_mu_list()
        self.phi_list = np.ones(self.k)/self.k
        self.sigma_list = self.init_sigma_list()
        self.log_liklihood = []
        
    def init_mu_list(self):
        self.mu_list = []
        for i in range (self.k):
            random_insatnce = np.random.choice(
                    self.data.shape[0], replace=False) # choose random K insatces as initial centers
            self.mu_list.append(self.data[random_insatnce])
#        print (self.mu_list)
        return self.mu_list
    
    def update_mu_list(self):
        mu_list = self.mu_list
        for k in range (self.k): # set mu list to 0
            mu_list[k] = mu_list[k] - mu_list[k]
        for i in range (len(self.data)):
            for k in range (self.k):
#                print ('mu ',mu_list[k], self.instances_weights[i,k], self.data[i]/self.phi_sum[k])
                mu_list[k] += (self.instances_weights[i,k]*self.data[i]/(self.phi_sum[k]))
#                print (mu_list[k])
        self.mu_list = mu_list
#        print (mu_list[0], mu_list[1])
        
    def update_phi_sum(self):
        for  k in range (self.k):
            self.phi_sum[k] = np.sum(self.instances_weights[:,k],axis = 0)
            
    def update_phi_list(self):
        for  k in range (self.k):
            self.phi_list[k] = self.phi_sum[k]/len(self.data)
        return self.phi_list
    
    def init_sigma_list(self):
        self.sigma_list = []
        for k in range (self.k):
            self.sigma_list.append(np.cov(self.data.T))
        return self.sigma_list
        
    def calc_sigma(self,k):
        x = (self.data-self.mu_list[k])
        w = np.sqrt(self.instances_weights[:,k])
        w = np.reshape(w,(len(self.data),1))
        wx = np.multiply(w,x) # element wise - vector x stay the same size
        wx = (1/(self.phi_sum[k]**0.5))*(wx) 
        sigma = wx.T@wx
        return sigma
    def update_sigma_list(self):
        for k in range (self.k):
            self.sigma_list[k] = self.calc_sigma(k)

    def probability(self,datum,mu,sigma):
        var = multivariate_normal(mu,sigma,allow_singular =True)
        return var.pdf(datum)
              
    def expectation(self):
        log_liklihood = 0
        for i in range (len(self.data)):
            for k in range (self.k):
                self.instances_weights[i,k] = self.phi_list[k]*self.probability(
                        self.data[i],self.mu_list[k],self.sigma_list[k])
#                if i%100 == 0 :
#                    print ('i',i,'proba ', self.probability(
#                        self.data[i],self.mu_list[k],self.sigma_list[k]), 'data',self.data[i] ,'mu', self.mu_list[k])
            log_liklihood += -np.log(np.sum(self.instances_weights[i]))
            self.instances_weights[i] = self.instances_weights[i]/np.sum(self.instances_weights[i])
        self.log_liklihood.append(log_liklihood)
    
    def maximization(self):
        self.update_phi_sum()
        self.update_mu_list()
        self.update_phi_list()
        self.update_sigma_list()
            
    def fit(self,iterration,delta = 0.0001):
        self.expectation()
        self.maximization()
        delta_log_liklihood = self.log_liklihood[0]
        i = 0
        while i < iterration and delta_log_liklihood > delta:
            self.expectation()
            self.maximization()
            delta_log_liklihood = self.log_liklihood[i] - self.log_liklihood[i+1]
            print (self.log_liklihood[i+1], delta_log_liklihood)
            i = i +1
            
                
    def label_data(self,datum):
        prob = []
        for k in range (self.k):
            prob.append(self.probability(datum,self.mu_list[k],self.sigma_list[k]))
        return (prob)
    
    def plot_classes_2d(self,labels):
        cdict = {0: 'red', 1: 'blue', 2: 'green'}
        for i in range (len(labels)):
            plt.plot(self.data[i,0],self.data[i,3],'o',c=cdict[labels[i]]) #plot data
        for k in range (self.k): # for every K print 2 features of data and centers
            plt.plot(self.mu_list[k][0],self.mu_list[k][3],'D' ,c = cdict[k],markersize=15) # plot Kmean 
        plt.show()    

def predict(gmm_model,data):
    proba = []
    label = []
    for i in range (len(data)):
        a = gmm_model.label_data(data[i])
        proba.append(a)
        label.append(proba[i].index(max(proba[i])))
    return label,proba
        
            
        
            
        
    
        
        
        
        
        
    




#########################################################################################
if __name__ == "__main__": 
    iris = datasets.load_iris()
    X = iris.data[:,:4]  # we only take the first two features.
#    X = scale(X,axis = 0)
    y = iris.target
#    cov = X.T@X
#    inv = np.linalg.inv(cov)
    g1 = GMM(3,X)
    g1.fit(1000)
    ap = predict(g1,X[:])
    print ('predicted labels:')
    g1.plot_classes_2d(ap[0])
    print ('true labels:')
    g1.plot_classes_2d(y)

#    g1.expectation()
#    g1.maximization()
#    g1.expectation()
#    g1.maximization()


    
