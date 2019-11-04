# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:52:10 2019

@author: royru
"""

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


def p_distance(xi,xj,sigma):
    distance=np.exp(-np.linalg.norm(xi-xj)**2/(2*sigma**2))
    return distance
    
def q_distance(yi,yj):
    distance= (1 + np.linalg.norm(yi-yj)**2)**-1
    return distance

def update_p_table_i(p_table,index,sigma):
        for j in range (len(p_table[0,:])):
            if index != j :
                p_table [index,j] = p_distance(X[index],X[j],sigma)
        sum_over_i = np.sum(p_table[index])
        p_table[index] = p_table[index]/sum_over_i
        return p_table
        
def compute_perplexity(p_table,index):
    h = 0
    for i in range (len(p_table)):
        if index != i:
            h += -(p_table[index,i])*np.log2(p_table[index,i]+epsilon)
    return 2**h

def binary_sigma_search(p_table,index):
    min_sigma = 0
    max_sigma = 10**3
    sigma = 1
    i = 0
    while (abs(compute_perplexity(p_table,index)-PERPLEXITY) > epsilon):
        i = i+1
        if i >1000 :
            break
        else :
            if compute_perplexity(p_table,index) > PERPLEXITY :
                    max_sigma = sigma
                    sigma = (max_sigma+min_sigma)/2
            else :
                min_sigma = sigma
                sigma = (min_sigma+max_sigma)/2
        p_table = update_p_table_i(p_table,index,sigma)
#        print (compute_perplexity(p_table,index))
#        print (sigma)
    return p_table

    
class Tsne:
    def __init__(self,k,data):
        self.k = k
        self.data = data

def plot_tsne_2d(y,labels):
    cdict = {0: 'red', 1: 'blue', 2: 'green'}
    for i in range (len(labels)):
        plt.plot(y[i,0],y[i,1],'o',c=cdict[labels[i]]) #plot data 
    plt.show()    

def update_y(q_table,p_table,y):
    for i in range (len(p_table[:,0])):
        delta_i = 0
        for j in range (len(p_table[0,:])):
            delta_i = delta_i+(p_table[i,j]-q_table[i,j])*(y[i]-y[j])*(q_distance(y[i],y[j]))
        delta_i = 4*delta_i
        y[i]=y[i]-LR*delta_i
    return(y)
    
def update_q_table(q_table,y):
    for i in range (len(q_table[:,0])):
        for j in range (len(q_table[0,:])):
            if i != j :
                q_table [i,j] = q_distance(y[i],y[j])
        sum_over_i = np.sum(q_table[i])
        q_table[i] = q_table[i]/sum_over_i
    return q_table

def calc_cost(q_table,p_table):
    cost = 0
    for i in range (len(q_table[:,0])):
        for j in range (len(q_table[0,:])):
            if i != j :
                cost = cost + (p_table[i,j])*np.log((p_table[i,j]/q_table[i,j])+epsilon)
    return (cost)

#########################################################################################
if __name__ == "__main__": 
    iris = datasets.load_iris()
    X = iris.data[:,:4]  
    labels = iris.target[:]
    epsilon = 10**(-6)
    PERPLEXITY=5
    sigma=1
    EPOCHS=10
    LR= .02
    MOMENTUM=0.99

# create and normlize p_ table (p - higher dimension)
    # initilize p_table with sigma 1
    p_table = np.zeros((X.shape[0],X.shape[0]))
    for i in range (len(p_table[:,0])):
        for j in range (len(p_table[0,:])):
            if i != j :
                p_table [i,j] = p_distance(X[i],X[j],sigma)
        sum_over_i = np.sum(p_table[i])
        p_table[i] = p_table[i]/sum_over_i
    # update sigma for each index of p to normlize peprlexity between indexes     
    for i in range (len(p_table[:,0])):
       p_table = binary_sigma_search(p_table,i)
    p_table = (p_table+p_table.T)/2*i
    
    
    # create and normlize q_table (q - lower dimention)
    initialize_y = np.random.normal(0,10**(-4),(len(X[:,0]),2))
    y = np.copy(initialize_y)
    q_table =  np.zeros((y.shape[0],y.shape[0]))
    q_table = update_q_table(q_table,y)
    
    # update y
    plot_tsne_2d(initialize_y,labels)
    cost = calc_cost(q_table,p_table)
    print (cost)
    for i in range (EPOCHS):
        y = update_y(q_table,p_table,y)
        q_table = update_q_table(q_table,y)
        cost = calc_cost(q_table,p_table)
        if (i%2) == 0 :
            plot_tsne_2d(y,labels)
            print (cost)
            LR = LR*0.9
        




    






