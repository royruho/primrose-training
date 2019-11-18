##########################################################################
# recurrent neural network - trained on binary sum of 2 numbers          #
##########################################################################
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.datasets import load_digits
Epsilon = 10**-6

######################################### #
#activation functions and its derivatives: #
###########################################
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def softmax(x): # soft mac activation function
    """Compute softmax values for each sets of scores in x."""
    e_x=np.exp(x)
    e_x = e_x / (Epsilon+e_x.sum())
    return e_x

def softmax_derv(x): # when using log_loss
    """Compute softmax values for each sets of scores in x."""
    return 1

def relu(x):
        return x * (x > 0)
def relu_derv(x):
        return 1. * (x > 0)

##############################
# loss functions             #
##############################
def loss(yp,label): # squred error loss
    return abs(yp-label)

def log_loss(yp,label): #Soft_max loss function
    loss = label*np.log(yp+Epsilon)
    return -np.sum(loss)

def loss_derv(yp,label): # for squred root loss and for softmax using log lossv
    return yp-label

################################
# Misc functions               #
################################
def predict(bin_generator,fc1,rnn1,sig2,fc2,sig3):
# predicts target using forward prop
    for digit in range(len(bin_generator.first_bin) - 1, -1, -1):
        x=(bin_generator.first_bin[digit], bin_generator.second_bin[digit])
        fc1.prop(x)
        rnn1.prop(fc1.curr_layer)
        sig2.prop(rnn1.curr_layer)
        fc2.prop(sig2.curr_layer)
        sig3.prop(fc2.curr_layer)
        print ("digit: {}, {} + {} = {} , prediction = {} , error = {}".format(digit, bin_generator.first_bin[digit],
                                                    bin_generator.second_bin[digit],
                                                    bin_generator.sum_bin[digit],sig3.curr_layer,
                                                    (sig3.curr_layer-bin_generator.sum_bin[digit])))

####################################
#   Layers as Classes              #
####################################
class  Fully_connected: # fully connectoed linear layer
    def __init__(self, prev_layer_size, curr_layer_size):
        self.weights = 2*np.atleast_2d(np.random.rand(prev_layer_size, curr_layer_size))-1 # x derv
        self.weights = self.weights / np.sqrt(self.weights.shape[0])
        self.bias = np.atleast_2d(np.random.uniform(-1, 1, curr_layer_size))
        self.bias_update = np.zeros(self.bias.shape)
        self.weights_update = 0
        self.list_prev_layer = []
        self.back_prop_index = 0
    def prop(self,prev_layer):
        self.prev_layer = np.atleast_2d(prev_layer) # w derv
        self.curr_layer = prev_layer @ self.weights + self.bias # bias derv
        self.list_prev_layer.append(self.prev_layer)
    def back_prop(self,error, batch_update, lr=0.1):
        self.back_prop_index += 1
        self.error = np.atleast_2d(error)
        self.back_prop_error = self.error @ self.weights.T
        self.bias_update += error
        self.weights_update += self.list_prev_layer[-self.back_prop_index].T @ self.error
        if batch_update:
            self.weights = self.weights - lr*self.weights_update
            self.bias = self.bias - lr*self.bias_update
            self.back_prop_index = 0
            self.list_prev_layer = []

class  Identity: # Identity
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = self.prev_layer
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = 1*self.error

class  Sigmoid_activation: # non linear sigmoid activation layer
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = sigmoid(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = sigmoid_derv(self.prev_layer)*self.error
class  Relu_activation: # non linear Relu activation layer
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = relu(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = relu_derv(self.prev_layer)*self.error


class Binary_Data: # generates binary sum examples
    def __init__(self,bit_size):
        self.binary_dict = {}
        binary_dim=  bit_size
        self.largest_number=pow(2, binary_dim) #maximal number represented by bit size
        self.binary=np.unpackbits(
            np.array([range(self.largest_number)], dtype=np.uint8).T, axis=1)
        for i in range(self.largest_number):
            self.binary_dict[i]=self.binary[i]
    def int2binary(self,integer):
        return self.binary_dict[integer]
    def generate_data(self):
        self.first_int = np.random.randint(self.largest_number/2)
        self.second_int = np.random.randint(self.largest_number/2)
        self.sum_int = self.first_int + self.second_int
        self.first_bin=self.binary_dict[self.first_int]
        self.second_bin=self.binary_dict[self.second_int]
        self.sum_bin=self.binary_dict[self.sum_int]

class  RNN: # RNN layer
    def __init__(self, prev_layer_size, curr_layer_size):
        self.cur_layer_size = curr_layer_size
        self.weights = np.atleast_2d(np.random.uniform(-1, 1, (prev_layer_size, curr_layer_size))) # x derv
        self.weights = self.weights / np.sqrt(self.weights.shape[0])
        self.bias = np.atleast_1d(np.random.uniform(-1, 1, curr_layer_size))
        self.old_curr_layer = np.atleast_2d(np.zeros(curr_layer_size))
        self.bias_update = 0
        self.weights_update = 0
        self.list_current_layer = []
        self.back_prop_index = 0
    def prop(self,prev_layer):
        self.prev_layer = (np.atleast_2d(prev_layer)) # w derv
        self.curr_layer = self.old_curr_layer @ self.weights + self.prev_layer # +self.bias bias derv
        self.list_current_layer.append(self.curr_layer)
    def back_prop(self,error, batch_update, lr=0.1):
        self.back_prop_index += 1
        self.error = np.atleast_2d(error)
        self.back_prop_error = self.error @ self.weights.T
        self.bias_update += np.sum(error)
        self.weights_update += self.list_current_layer[-self.back_prop_index].T @ self.error
        self.old_curr_layer = copy.deepcopy(self.curr_layer)
        if batch_update:
            self.weights = self.weights - lr*self.weights_update
            self.bias = self.bias - lr*self.bias_update
            self.list_current_layer = []
            self.back_prop_index = 0

if __name__ == "__main__":
    ##############################
    # create layers architecture #
    #############################
    fc1 = Fully_connected(2,16)
    # sig1 = Sigmoid_activation() # sigmoid activation
    rnn1 = RNN(16,16)
    sig2 = Sigmoid_activation()
    fc2=Fully_connected(16, 1)
    sig3 = Sigmoid_activation()
    ############################
    # set hyper parameters     #
    ############################
    binary_dim=8
    initial_lr = 0.0001 # initial learning rate
    epochs = 100
    print_loss = 20 # print loss after print_loss epochs
    LR_update = 0.99 # update learning rate each print_loss
    #################
    # generate data #
    #################
    bin_generator = Binary_Data(8)
    LR = initial_lr
    error = 0
    error_sum = 0
    bin_generator.generate_data()
    #####################
    # initialize run    #
    #####################
    for epoch in range (epochs): # epoch loop
        loss_var = 0
        weight_update = False
        # bin_generator.generate_data()
        error = []
        for digit in range(len(bin_generator.first_bin)-1, -1, -1):
            x = (bin_generator.first_bin[digit], bin_generator.second_bin[digit])
            fc1.prop(x)
            rnn1.prop(fc1.curr_layer)
            sig2.prop(rnn1.curr_layer)
            fc2.prop(sig2.curr_layer)
            sig3.prop(fc2.curr_layer)
            loss_var = loss_var + loss(sig3.curr_layer, bin_generator.sum_bin[digit])
            error.append(sig3.curr_layer - bin_generator.sum_bin[digit])
            error_sum = error_sum + error[-1]
            # print (error)
        sig3.back_prop(error[-1])
        fc2.back_prop(sig3.back_prop_error, weight_update, lr=LR)
        sig2.back_prop(fc2.back_prop_error)
        rnn1.back_prop(sig2.back_prop_error, weight_update, lr=LR)
        fc1.back_prop(rnn1.back_prop_error,weight_update, lr=LR)
        for digit in range(len(bin_generator.first_bin) - 2):
            rnn_previous_back_prop = rnn1.back_prop_error
            sig3.back_prop(error[-digit])
            fc2.back_prop(sig3.back_prop_error, weight_update, lr=LR)
            sig2.back_prop(fc2.back_prop_error)
            rnn1.back_prop(sig2.back_prop_error+rnn_previous_back_prop, weight_update, lr=LR)
            fc1.back_prop(rnn1.back_prop_error, weight_update, lr=LR)
        weight_update = True
        if weight_update:
            rnn_previous_back_prop=rnn1.back_prop_error
            sig3.back_prop(error[-digit-1])
            fc2.back_prop(sig3.back_prop_error, weight_update, lr=LR)
            sig2.back_prop(fc2.back_prop_error)
            rnn1.back_prop(sig2.back_prop_error+rnn_previous_back_prop, weight_update, lr=LR)
            fc1.back_prop(rnn1.back_prop_error,weight_update, lr=LR)
            error_sum = 0
            Batch_update = False
        if (epoch % print_loss) == 0:
            LR =  LR * LR_update
            # print (LR)
            print("epoch: {}, loss: {}".format(epoch, loss_var))
            predict(bin_generator, fc1, rnn1, sig2, fc2, sig3)


    predict(bin_generator,fc1,rnn1,sig2,fc2,sig3)