# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:37:50 2019

@author: royru
"""
import numpy as np

epsilon=10 ** (-5)


def sigmoid(x):  # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def sigmoid_derv(x):
    return (sigmoid(x) * (1 - sigmoid(x)))


def loss(yp, label):
    return 1 / 2 * (yp - label) ** 2


def loss_derv(yp, label):
    return yp - label

class  Fully_connected: # fully connectoed linear layer
    def __init__(self, prev_layer_size, curr_layer_size):
        self.weights = np.atleast_2d(np.random.uniform(-1, 1, (prev_layer_size, curr_layer_size))) # x derv
        self.bias = np.atleast_1d(np.random.uniform(-1, 1, curr_layer_size))
    def prop(self,prev_layer):
        self.prev_layer = np.atleast_2d(prev_layer) # w derv
        self.curr_layer = prev_layer @ self.weights + self.bias # bias derv
    def back_prop(self,error,lr=0.1):
        self.error = np.atleast_2d(error)
        self.back_prop_error = self.error @ self.weights.T
        self.bias_update = error
        self.weights_update = self.prev_layer.T @ self.error
        self.weights = self.weights - lr*self.weights_update
        self.bias = self.bias - lr*self.bias_update

class  Non_linear: # non linear sigmoid activation layer

    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = sigmoid(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = sigmoid_derv(self.prev_layer)*self.error

if __name__ == "__main__":
    learning_rate = -0.3

    inputs=np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]])

    labels=np.array([
        [0],
        [0],
        [1],
        [1]])

    labels_2d=np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])

    # 2  fully connected layers
    print("********************************\n 1 layer:")
    x = inputs
    y = labels_2d
    fc1 = Fully_connected(3,4)
    sig1 = Non_linear()
    fc2 = Fully_connected(4,2)
    sig2 = Non_linear()
    print_shape = False

    epochs = 2
    for epoch in range (epochs):
        sum_loss = 0
        for inst in range (x.shape[0]):
            fc1.prop(x[inst])
            sig1.prop(fc1.curr_layer)
            fc2.prop(sig1.curr_layer)
            sig2.prop(fc2.curr_layer)
            sum_loss = sum_loss + loss(sig2.curr_layer, y[inst])
            error = np.reshape((sig2.curr_layer - y[inst]),(1,2))
            sig2.back_prop(error)
            fc2.back_prop(sig2.back_prop_error)
            sig1.back_prop(fc2.back_prop_error)
            fc1.back_prop(sig1.back_prop_error)
            if print_shape == True :
                print("fc1.curr_layer.shape", fc1.curr_layer.shape)
                print("sig1.curr_layer.shape", sig1.curr_layer.shape )
                print("fc2.curr_layer.shape", fc2.curr_layer.shape)
                print("sig2.curr_layer.shape", sig2.curr_layer.shape)
                print ("error" , error.shape)
                print ("sig2.back_prop_error", sig2.back_prop_error.shape)
                print ("fc2.back_prop_error",fc2.back_prop_error.shape)
                print ("sig2.back_prop_error",sig2.back_prop_error.shape)
                print ("fc1.back_prop_error",fc1.back_prop_error.shape)
        if (epoch % 1) == 0:
            print(sum_loss)



    # weights = np.random.uniform(-1, 1, (3, 1))
    # w2 = weights
    # for i in range(10):
    #     for sample in range(inputs.shape[0]):
    #         print (i, sample)

    #     layer_2=x @ w2
    #     layer_2_x_derv=w2.T
    #     layer_2_w_derv=x.T
    #     layer_2_sig_activation=sigmoid(layer_2)
    #     layer_2_sig_derv=sigmoid_derv(layer_2)
    #     sum_loss=np.sum(loss(layer_2_sig_activation, y))
    #     error=layer_2_sig_activation - y
    #     delta_w2=layer_2_w_derv @ (error * layer_2_sig_derv)
    #     w2=w2 - delta_w2 * learning_rate
    #     if (i % 10000) == 0:
    #         print(sum_loss)
    #
    # # 2 layers
    # print("********************************\n 2 layers:")
    # x=inputs
    # y=labels
    # weights1=np.random.uniform(-1, 1, (3, 4))
    # w1=weights1
    # weights2=np.random.uniform(-1, 1, (4, 1))
    # w2=weights2
    # for i in range(100000):
    #     layer_1=x @ w1
    #     layer_1_x_derv=w1.T
    #     layer_1_w_derv=x.T
    #     layer_1_sig_activation=sigmoid(layer_1)
    #     layer_1_sig_derv=sigmoid_derv(layer_1)
    #     layer_2=layer_1_sig_activation @ w2
    #     layer_2_x_derv=w2.T
    #     layer_2_w_derv=layer_1_sig_activation.T
    #     layer_2_sig_activation=sigmoid(layer_2)
    #     layer_2_sig_derv=sigmoid_derv(layer_2)
    #     sum_loss=np.sum(loss(layer_2_sig_activation, y))
    #     error=layer_2_sig_activation - y
    #     delta_w2=layer_2_w_derv @ (error * layer_2_sig_derv)
    #     delta_w1=layer_1_w_derv @ (layer_1_sig_derv * ((error * layer_2_sig_derv) @ layer_2_x_derv))
    #     w2=w2 - delta_w2 * learning_rate
    #     w1=w1 - delta_w1 * learning_rate
    #     if (i % 10000) == 0:
    #         print(sum_loss)
    #
    # from sklearn import datasets
    # from sklearn.preprocessing import LabelBinarizer
    #
    # iris=datasets.load_iris()
    # x=iris.data[:, :4]
    # y=np.reshape(iris.target, (150, 1))
    # hot_encoded_labels=LabelBinarizer()
    # hot_encoded_labels.fit([0, 1, 2])
    # binary_labels=hot_encoded_labels.transform(y)
    # for i in range(len(y)):
    #     if y[i] == 2:
    #         y[i]=1
    #
    # print("********************************\n 2 layers on IRIS data set:")
    #
    # weights1=np.random.uniform(-1, 1, (4, 2))
    # w1=weights1
    # weights2=np.random.uniform(-1, 1, (2, 1))
    # w2=weights2
    # for i in range(100000):
    #     layer_1=x @ w1
    #     layer_1_x_derv=w1.T
    #     layer_1_w_derv=x.T
    #     layer_1_sig_activation=sigmoid(layer_1)
    #     layer_1_sig_derv=sigmoid_derv(layer_1)
    #     layer_2=layer_1_sig_activation @ w2
    #     layer_2_x_derv=w2.T
    #     layer_2_w_derv=layer_1_sig_activation.T
    #     layer_2_sig_activation=sigmoid(layer_2)
    #     layer_2_sig_derv=sigmoid_derv(layer_2)
    #     sum_loss=np.sum(loss(layer_2_sig_activation, y))
    #     error=layer_2_sig_activation - y
    #     delta_w2=layer_2_w_derv @ (error * layer_2_sig_derv)
    #     delta_w1=layer_1_w_derv @ (layer_1_sig_derv * ((error * layer_2_sig_derv) @ layer_2_x_derv))
    #     w2=w2 - delta_w2 * learning_rate
    #     w1=w1 - delta_w1 * learning_rate
    #     if (i % 10000) == 0:
    #         print(sum_loss)
    #
    # # 3 layers
    # print("********************************\n 3 layers on IRIS data set:")
    #
    # weights1=np.random.uniform(-1, 1, (4, 10))
    # w1=weights1
    # weights2=np.random.uniform(-1, 1, (10, 10))
    # w2=weights2
    # weights3=np.random.uniform(-1, 1, (10, 1))
    # w3=weights3
    # learning_rate=0.3
    #
    # for i in range(100000):
    #     # forward layer 1
    #     layer_1=x @ w1
    #     # remember  derivatives layer 1
    #     layer_1_x_derv=w1.T
    #     layer_1_w_derv=x.T
    #     # activation layer 1
    #     layer_1_sig_activation=sigmoid(layer_1)
    #     # remember sigmoid derivative
    #     layer_1_sig_derv=sigmoid_derv(layer_1)
    #
    #     layer_2=layer_1_sig_activation @ w2
    #     layer_2_x_derv=w2.T
    #     layer_2_w_derv=layer_1_sig_activation.T
    #     layer_2_sig_activation=sigmoid(layer_2)
    #     layer_2_sig_derv=sigmoid_derv(layer_2)
    #     layer_3=layer_2_sig_activation @ w3
    #     layer_3_x_derv=w3.T
    #     layer_3_w_derv=layer_2_sig_activation.T
    #     layer_3_sig_activation=sigmoid(layer_3)
    #     layer_3_sig_derv=sigmoid_derv(layer_3)
    #     sum_loss=np.sum(loss(layer_3_sig_activation, y))
    #     error=layer_3_sig_activation - y
    #     delta_w3=layer_3_w_derv @ (error * layer_3_sig_derv)
    #     delta_w2=layer_2_w_derv @ (layer_2_sig_derv * ((error * layer_3_sig_derv) @ layer_3_x_derv))
    #     delta_w1=layer_1_w_derv @ (
    #                 layer_1_sig_derv * ((layer_2_sig_derv * layer_3_x_derv * (error * layer_3_sig_derv)) @ layer_2_x_derv))
    #     w3=w3 - delta_w3 * learning_rate
    #     w2=w2 - delta_w2 * learning_rate
    #     w1=w1 - delta_w1 * learning_rate
    #     if (i % 10000) == 0:
    #         learning_rate=learning_rate * .9
    #         print(sum_loss)
    #
    # # 3 layers
    # print("********************************\n 3 layers 3 labels on IRIS data set:")  # not working
    #
    # weights1=np.random.uniform(-1, 1, (4, 10))
    # w1=weights1
    # weights2=np.random.uniform(-1, 1, (10, 10))
    # w2=weights2
    # weights3=np.random.uniform(-1, 1, (10, 3))
    # w3=weights3
    # learning_rate=0.3
    #
    # for i in range(100000):
    #     # forward layer 1
    #     layer_1=x @ w1
    #     # remember  derivatives layer 1
    #     layer_1_x_derv=w1.T
    #     layer_1_w_derv=x.T
    #     # activation layer 1
    #     layer_1_sig_activation=sigmoid(layer_1)
    #     # remember sigmoid derivative
    #     layer_1_sig_derv=sigmoid_derv(layer_1)
    #     layer_2=layer_1_sig_activation @ w2
    #     layer_2_x_derv=w2.T
    #     layer_2_w_derv=layer_1_sig_activation.T
    #     layer_2_sig_activation=sigmoid(layer_2)
    #     layer_2_sig_derv=sigmoid_derv(layer_2)
    #     layer_3=layer_2_sig_activation @ w3
    #     layer_3_x_derv=w3.T
    #     layer_3_w_derv=layer_2_sig_activation.T
    #     layer_3_sig_activation=sigmoid(layer_3)
    #     layer_3_sig_derv=sigmoid_derv(layer_3)
    #     sum_loss=np.sum(loss(layer_3_sig_activation, y))
    #     error=layer_3_sig_activation - binary_labels
    #     delta_w3=layer_3_w_derv @ (error * layer_3_sig_derv)
    #     delta_w2=layer_2_w_derv @ (layer_2_sig_derv * ((error * layer_3_sig_derv) @ layer_3_x_derv))
    #     delta_w1=layer_1_w_derv @ (
    #                 layer_1_sig_derv * ((layer_2_sig_derv * layer_3_x_derv * (error * layer_3_sig_derv)) @ layer_2_x_derv))
    #     w3=w3 - delta_w3 * learning_rate
    #     w2=w2 - delta_w2 * learning_rate
    #     w1=w1 - delta_w1 * learning_rate
    #     if (i % 10000) == 0:
    #         learning_rate=learning_rate * .9
    #         print(sum_loss)
    #
    #
