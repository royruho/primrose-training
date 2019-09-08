# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:20:00 2019

@author: royru
"""

import numpy as np
older_sibling_age = np.reshape(([31,22,40,26]),(4,1))
yunger_sibling_age = np.reshape(([22,21,37,25]),(4,1))
times_talked=np.reshape(([2,3,8,12]),(4,1))

# 1.a
XX = np.column_stack((older_sibling_age,yunger_sibling_age))
Y = times_talked
W1a = np.linalg.inv(XX.T @ XX ) @ XX.T @ Y
sumoflose1a = abs((W1a.T@XX.T)-Y.T)
print ('sumoflose1a',sumoflose1a.sum())
#1.b - The age difference between the siblings is linearly dependent of sibling age
#1.c square of the age difference is not linearly dependent so it might benefit the model
sqr_age_difference = np.reshape(np.square(older_sibling_age-yunger_sibling_age),(4,1))
XX = np.column_stack((older_sibling_age,yunger_sibling_age,sqr_age_difference))
W1c = np.linalg.inv(XX.T @ XX ) @ XX.T @ Y
sumoflose1c = abs((W1c.T@XX.T)-Y.T)
print ('sumoflose1c',sumoflose1c.sum())
#1.d  vector of ones, - new featrue gives abetter optimization of linear regression model (y = a1x1+a2x2 +b instead of y=a1x1+a2x2 in 1.a model)
vectorofones= np.ones(older_sibling_age.shape)
XX = np.column_stack((vectorofones,older_sibling_age,yunger_sibling_age))
W1d = np.linalg.inv(XX.T @ XX ) @ XX.T @ Y
sumoflose1d = abs((W1d.T@XX.T)-Y.T)
print ('sumoflose1d',sumoflose1d.sum())
#1.e - cannot get better prediction unless using higher polynom


#2.a 
import numpy as np

def sumoflose_gradinet(_XX,_teta,_Y): #average absulote sum of error for specific teta
    return (1/len(_Y))*(abs(_XX@_teta-_Y).sum())

def compute_gradient(_XX,_teta,_Y):  #computes gradient  vector from specific teta
    return (1/len(_Y))*(_XX.T@((_XX@_teta)-_Y))

def update_teta_GD(_old_teta,_learning_rate,_gradient): #update teta vector in gradient descent model
    return  _old_teta-_learning_rate*_gradient

def calc_momentum_vector(_momentum,_momentum_vector,_learning_rate,_gradient):
    return ((_momentum*_momentum_vector-_learning_rate*_gradient))

def compute_stochastic_gradient(_XX,_Y,_random_index,_teta):  #computes gradient  vector from specific teta
    return (_XX[_random_index]*(_XX[_random_index]@_teta-_Y[_random_index]))


def plot_graph(_figure,_xlabel,_ylabel,_listoflos,_learning_rate,i):
    import matplotlib.pyplot as plt
    plt.figure(_figure)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title('learning rate,{}'.format(_learning_rate))
    _plotlistoflos=np.reshape(_listoflos,(i+1,2))
    plt.plot(_plotlistoflos[:,0],_plotlistoflos[:,1],'b')
    plt.show()
    print('learning rate',learning_rate,listoflos[i],'\n teta:',new_teta)
##############################################################################
if __name__ == "__main__": 
##    ##############check on large array
#    ss = 1000 #sample size
#    noisearray2 = np.random.randn(ss)
#    b3= 1
#    Xvector = np.random.randint(-2.0,2.0,size=ss)
#    #v3=v3+noisearray2
#    v3sqrxpararameter =1
#    v3xparameter = 1
#    Yvector = v3sqrxpararameter*Xvector*Xvector +v3xparameter*Xvector+ b3 + noisearray2
#    x=Xvector #([0,1,2,])#3,4,5,6]) # features vector
#    y=Yvector #([1,3,7])#,12,21,31,43]) # label vector
#########
    x=([-2,-1,0,1,2,3,4])#,3,4]) # features vector
    y=([3,1,1,3,7,13,21])#,12,21]) # label vector
    X=np.reshape((x),(len(x),1))
    Y=np.reshape((y),(len(x),1))
    teta0=np.reshape(([2,2,0]),(3,1)) 
    input_learning_rate = 0.001
    vectorofones= np.ones(Y.shape)
    Xsqr=np.square(X)
    XX = np.column_stack((vectorofones,X,Xsqr)) # features matrix
    irange = 50 # number of iteration
    
    print ('= = = = = = = = = = = = = = = =')
### gradient descent 0.01 model
    print('teta solved:',(np.linalg.inv(XX.T@XX)@XX.T@Y))
    
    learning_rate = input_learning_rate # LR0.01
    new_teta = teta0
    listoflos = ([])
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
    plot_graph(1,'iterration','sum of loss',listoflos,learning_rate,i)
    
### gradient descent 0.1 model
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate*10    # LR0.1
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
    plot_graph(2,'iterration','sum of loss',listoflos,learning_rate,i)
    
## gradient descent 1.0 model
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate*100    # LR0 1.0
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
    plot_graph(3,'iterration','sum of loss',listoflos,learning_rate,i)
    
# 2.b - learning rate = 0.01 succeeded in reducing loss inn each step. step size is small enough 
# 2.b - learning rate = 0.1 failed in reducing loss inn each step. step size is too big
# 2.b - learning rate = 1 failed in reducing loss inn each step. step size is too big
### 2.c reducing learing rate on the fly :
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate # LR 1
    learning_rate_adjuster = 0.999   # learning rate decrese in each itteration
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
        learning_rate = learning_rate *learning_rate_adjuster # adjusting learning_rate on the fly
    plot_graph(4,'leaning rate decreased in each iterration','sum of loss',listoflos,learning_rate,i)
    
### 2.c increasing learing rate on the fly :
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate # LR 1
    learning_rate_adjuster = 1.01   # learning rate increase in each itteration
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
        learning_rate = learning_rate *learning_rate_adjuster # adjusting learning_rate on the fly
    plot_graph(5,'leaning rate increased in each iterration','sum of loss',listoflos,learning_rate,i)
        
#2.c**    #####                 momentum vector
    
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate    # LR0 0.01
    momentum = 0.9
    momentum_vector= np.zeros(teta0.shape)
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,new_teta,Y)
        momentum_vector = calc_momentum_vector(momentum,momentum_vector,learning_rate,gradient)
        #print ('i:',i,'mv:',momentum_vector)
        new_teta = new_teta + momentum_vector 
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
    plot_graph(6,'momentom iterration','sum of loss',listoflos,learning_rate,i)
    
    
    
#2.d**    #####                 NESTROV_momentum vector
    
    new_teta = teta0
    listoflos = ([])
    learning_rate = input_learning_rate    # LR0 0.01
    momentum = 0.9
    momentum_vector= np.zeros(teta0.shape)
    gradient = compute_gradient(XX,teta0,Y)
    for i in range (0,irange):
        gradient = compute_gradient(XX,(new_teta+momentum_vector),Y)
        momentum_vector = calc_momentum_vector(momentum,momentum_vector,learning_rate,gradient)
        new_teta = new_teta + momentum_vector
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
    plot_graph(7,'nestrov iterration','sum of loss',listoflos,learning_rate,i)
    
# 2.f*** stochastic descent
    
    new_teta = teta0
    listoflos = ([])
    learning_rate_adjuster = 1
    learning_rate = input_learning_rate    # LR0.01
    for i in range (0,irange):
        random_index = np.random.randint(0,len(x))
        gradient = compute_stochastic_gradient(XX,Y,random_index,new_teta)
        new_teta = update_teta_GD(new_teta,learning_rate,gradient)
        listoflos.append((i,sumoflose_gradinet(XX,new_teta,Y)))
        learning_rate=input_learning_rate
    plot_graph(8,'stochastic iterration','sum of loss',listoflos,learning_rate,i)
    

##############check on large array










#import numpy as np
#
#
#def  cal_cost(theta,X,y):
#    '''
#    
#    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
#    theta = Vector of thetas 
#    X     = Row of X's np.zeros((2,j))
#    y     = Actual y's np.zeros((2,1))
#    
#    where:
#        j is the no of features
#    '''
#    
#    m = len(y)
#    
#    predictions = X.dot(theta)
#    cost = (1/2*m) * np.sum(np.square(predictions-y))
#    return cost
#
#def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
#    '''
#    X    = Matrix of X with added bias units
#    y    = Vector of Y
#    theta=Vector of thetas np.random.randn(j,1)
#    learning_rate 
#    iterations = no of iterations
#    
#    Returns the final theta vector and array of cost history over no of iterations
#    '''
#    m = len(y)
#    cost_history = np.zeros(iterations)
#    theta_history = np.zeros((iterations,2))
#    for it in range(iterations):
#        
#        prediction = np.dot(X,theta)
#        
#        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
#        theta_history[it,:] =theta.T
#        cost_history[it]  = cal_cost(theta,X,y)
#        
#    return theta, cost_history, theta_history
#
#X = 2 * np.random.rand(100,1)
#y = 4 +3 * X+np.random.randn(100,1)
#lr =0.01
#n_iter = 1000
#
#theta = np.random.randn(2,1)
#
#X_b = np.c_[np.ones((len(X),1)),X]
#theta,cost_history,theta_history = gradient_descent(X_b,y,theta,lr,n_iter)
#
#
#print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
#print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))