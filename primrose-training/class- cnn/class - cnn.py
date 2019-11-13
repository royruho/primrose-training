import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.datasets import load_digits
Epsilon = 10**-6

def sigmoid(x): # Activation function used to map any real value between 0 and 1
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def softmax(x): # soft mac activation function
    """Compute softmax values for each sets of scores in x."""
    e_x=np.exp(x)
    e_x = e_x / e_x.sum()
    # print ("sm",e_x)
    return e_x

def softmax_derv(x): #TODO not sure if correct
    """Compute softmax values for each sets of scores in x."""
    return 1

def hot_encode_labels(target): # encodes target to binary array size 10
    label = np.zeros(10)
    label[target] = 1
    return label

def loss(yp,label): # squred error loss
    return 1/2*(yp-label)**2

def loss_derv(yp,label): # squred root loss
    return yp-label

def log_loss(yp,label): #TODO log loss gives error because values are negative
    loss = label*np.log(yp)
    # print ("exp",np.log(yp),"loss",loss)
    return -np.sum(loss)


def zero_pad_image(image,size): # pads image with zeros
    return np.pad(image, (size,), 'constant', constant_values=0)

def add_layers_to_image(image, number_of_kernels):
    # create a 3D array with layers of image so multiple filters can convolt
    multilayred_image=np.zeros((number_of_kernels, image.shape[0], image.shape[1]))
    for i in range (number_of_kernels):
        multilayred_image[i]=image
    return multilayred_image

class  Fully_connected: # fully connectoed linear layer
    def __init__(self, prev_layer_size, curr_layer_size):
        self.weights = np.atleast_2d(np.random.uniform(-1, 1, (prev_layer_size, curr_layer_size))) # x derv
        self.weights = self.weights / np.sqrt(self.weights.shape[0])
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

class  Sigmoid_activation: # non linear sigmoid activation layer
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = sigmoid(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = sigmoid_derv(self.prev_layer)*self.error

class  Soft_max: # non linear soft_max activation layer
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = softmax(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = softmax_derv(self.prev_layer)*self.error

class Conv:
    def __init__(self, img_size, kernel_size):
        self.channels = np.zeros((kernel_size[0],img_size[0],img_size[1])) - 1
        self.kernel = np.random.uniform(-1, 1, kernel_size)
        self.kernel = self.kernel / (self.kernel.shape[1])
    def convolution(self, image, kernel):
        channels=np.zeros((image.shape[0], image.shape[1] - (kernel.shape[1]-1), image.shape[2] -(kernel.shape[2]-1))) - 1
        # print ("channels", channels.shape)
        # print("image", image.shape)
        # print("kernel", kernel.shape)
        for k_index in range(channels.shape[0]):
            for rows in range(0, channels.shape[1]):
                for columns in range(0, channels.shape[2]):
                    part_of_img=image[k_index, rows:rows + (kernel.shape[1]), columns:columns + (kernel.shape[2])]
                    conv=np.sum(part_of_img * kernel)
                    channels[k_index, rows - 1, columns - 1]=(conv/kernel.size)
                    # print (rows - 1,columns - 1,"/n",channels[:, rows - 1, columns - 1])
        return channels
    def prop(self,prev_layer):
        self.prev_layer = prev_layer
        self.padded_prev_layer =  zero_pad_image(self.prev_layer,1)
        self.multi_layered_prev_layer =  add_layers_to_image(self.padded_prev_layer, self.kernel.shape[0])
        self.curr_layer = self.convolution(self.multi_layered_prev_layer,self.kernel)
    def back_prop(self,error, lr=0.1):
        self.back_prop_kernal = np.reshape(error,(self.channels.shape[0],self.channels.shape[1],self.channels.shape[2]))
        self.back_prop_error = self.convolution(self.multi_layered_prev_layer,self.back_prop_kernal)
        self.kernel = self.kernel - self.back_prop_error*lr

class Conv_2d:
    def __init__(self, img_size, kernel_size):
        self.channels = np.zeros((kernel_size[0],img_size[0],img_size[1])) - 1
        self.kernel = np.random.uniform(-1, 1, kernel_size)
        self.kernel = self.kernel / (self.kernel.shape[1])
    def convolution(self, image, kernel):
        channels=np.zeros((image.shape[0], image.shape[1] - (kernel.shape[1]-1), image.shape[2] -(kernel.shape[2]-1))) - 1
        for i in range (image.shape[0]):
            channels[i] = signal.convolve2d(image[i], kernel[i],mode = "valid")
        return channels
    def prop(self,prev_layer):
        self.prev_layer = prev_layer
        self.padded_prev_layer =  zero_pad_image(self.prev_layer,1)
        self.multi_layered_prev_layer =  add_layers_to_image(self.padded_prev_layer, self.kernel.shape[0])
        self.curr_layer = self.convolution(self.multi_layered_prev_layer,self.kernel)
    def back_prop(self,error, lr=0.1):
        self.back_prop_kernal = np.reshape(error,(self.channels.shape[0],self.channels.shape[1],self.channels.shape[2]))
        self.back_prop_error = self.convolution(self.multi_layered_prev_layer,self.back_prop_kernal)
        self.kernel = self.kernel - self.back_prop_error*lr

def predict(image,conv,fc1,sm1): #,fc2,sm2):#  forward prop and argmax
    normlized = image / np.max(image)
    conv.prop(normlized)
    x=conv_layer.curr_layer
    x=x.flatten()
    fc1.prop(x)
    sm1.prop(fc1.curr_layer)
    # fc2.prop(sm1.curr_layer)
    # sm2.prop(fc2.curr_layer)
    return np.argmax(sm1.curr_layer)

if __name__ == "__main__":
    digits=load_digits()
    digits.data
    conv_layer = Conv((8,8),(6,3,3)) # 6 3X3 filters
    fc1 = Fully_connected(384,10)
    sm1 = Soft_max() # soft max activation
    initial_lr = .5 # initial learning rate
    LR = initial_lr  # active learning rate variable
    epochs = 2000
    print_shape = False # print shape of layers
    low_index = 0 # index limit of digits array
    high_index = 5 # index limit of digits array
    batch_size = 1 # batch size
    batch_index = 0
    error = 0
    for epoch in range (epochs):
        loss_var = 0
        for img_index in range (low_index,high_index):
            batch_index += 1
            normlized = (digits.images[img_index]-np.mean(digits.images[img_index]))\
                        /np.max(digits.images[img_index]) #normalize input
            conv_layer.prop(normlized)
            x = conv_layer.curr_layer
            x = x.flatten()
            fc1.prop(x)
            sm1.prop(fc1.curr_layer)
            # fc2.prop(sm1.curr_layer)
            # sm2.prop(fc2.curr_layer)
            loss_var = loss_var + log_loss(sm1.curr_layer, hot_encode_labels(digits.target[img_index]))
            error = error + (sm1.curr_layer - hot_encode_labels(digits.target[img_index]))
            if batch_index == batch_size:
                # print (error)
                sm1.back_prop(error/batch_size)
                fc1.back_prop(sm1.back_prop_error,lr=LR)
                # sm1.back_prop(fc2.back_prop_error,lr=LR)
                # fc1.back_prop(sm1.back_prop_error,lr=LR)
                conv_layer.back_prop(fc1.back_prop_error,lr=LR)
                error = 0
                batch_index = 0
            if print_shape == True :
                print("fc1.curr_layer.shape", fc1.curr_layer.shape)
                print("sm1.curr_layer.shape", sm1.curr_layer.shape )
                print ("error" , error.shape)
                print ("sm1.back_prop_error",sm1.back_prop_error.shape)
                print ("fc1.back_prop_error",fc1.back_prop_error.shape)
        if (epoch % 500) == 0:
            LR =  LR * 0.95
            # print (LR)
            print("epoch:",epoch," loss:",loss_var)

    for i in range (low_index,high_index):
        print ("predict", predict(digits.images[i],conv_layer,fc1,sm1), digits.target[i])


    a = digits.images[0]
    conv_layer.prop(a)
    plt.gray()
    plt.matshow(a)
    for i in range (conv_layer.channels.shape[0]):
        plt.matshow(conv_layer.curr_layer[i])
        plt.show(block=False)

