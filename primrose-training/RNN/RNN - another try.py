##########################################################################
# recurrent neural network - trained on binary sum of 2 numbers          #
##########################################################################
import numpy as np
import copy
import matplotlib.pyplot as plt

######################################### #
#activation functions and its derivatives: #
###########################################
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x):
    return (sigmoid(x)*(1-sigmoid(x)))

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



if __name__ == "__main__":

    ############################
    # set hyper parameters     #
    ############################
    binary_dim=2
    initial_lr = 0.01 # initial learning rate
    epochs = 5
    print_loss = 1 # print loss after print_loss epochs
    LR_update = 0.99 # update learning rate each print_loss
    #################
    # generate data #
    #################
    bin_generator = Binary_Data(8)
    LR = initial_lr
    bin_generator.generate_data()
    #####################
    # initialize run    #
    #####################
    w0 = (2*np.random.rand(2,16)-1)/np.sqrt(32)
    wh = (2*np.random.rand(16,16)-1)/np.sqrt(168)
    w2 = (2*np.random.rand(16,1)-1)/np.sqrt(16)
    for epoch in range (epochs): # epoch loop
        delta_w2 = []
        delta_wh = []
        delta_w0 = []
        layer2_backprop_error = []
        list_layer0 = []
        list_hidden = []
        list_layer2 = []
        list_x = []
        list_hidden.append(np.zeros((1,16)))
        # bin_generator.generate_data()
        error = []
        ### forward prop ###
        for digit in range(binary_dim): 
            x = (bin_generator.first_bin[digit], bin_generator.second_bin[digit])
            list_x.append(np.atleast_2d(x))
            layer0 = np.atleast_2d(x@w0)
            hidden_layer = list_hidden[-1] @ wh
            hidden_layer = sigmoid(np.atleast_2d(layer0+hidden_layer))
            list_hidden.append(hidden_layer)
            layer_2 = hidden_layer@w2
            yp = sigmoid(layer_2)
            error.append(yp - bin_generator.sum_bin[digit])
            layer2_backprop_error.append(sigmoid_derv(sigmoid_derv(error[-1])*w2.T))
            
        ### back prop ###
        for digit in range(binary_dim):
            delta_w2.append(np.atleast_2d(sigmoid_derv(error[binary_dim - digit-1]).T@list_hidden[-digit-1]))
            delta_wh.append(np.atleast_2d(list_hidden[-digit-2].T@np.sum(layer2_backprop_error[-digit-1:],axis=0)))
            delta_w0.append(np.atleast_2d(list_x[-digit-1].T@np.sum(layer2_backprop_error[-digit-1:],axis=0)))
        
        ### update weights ###
        w2 = w2 - LR*np.sum(delta_w2, axis = 0).T
        wh = wh - LR*np.sum(delta_wh, axis = 0)
        w0 = w0 - LR*np.sum(delta_w0, axis = 0)
        if (epoch % print_loss) == 0:
            LR =  LR * LR_update
            # print (LR)
            print ("epoch: {}, loss: {}".format(epoch, np.sum(np.abs(error))))
            for digit in range(binary_dim):
                print ("digit: {}, {} + {} = {} , prediction = {} , error = {}".format(digit, bin_generator.first_bin[digit],
                                         bin_generator.second_bin[digit],
                                         bin_generator.sum_bin[digit],yp,
                                         error[digit]))

