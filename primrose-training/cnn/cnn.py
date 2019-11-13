import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def sigmoid(x): # Activation function used to map any real value between 0 and 1
    return 1/(1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derv(x):
    """Compute softmax values for each sets of scores in x."""
    return x

def hot_encode_labels(target):
    label = np.zeros(10)
    label[target] = 1
    return  label

def sigmoid_derv(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def loss(yp,label):
    return 1/2*(yp-label)**2

def loss_derv(yp,label):
    return yp-label

def zero_pad_image(image,size):
    return np.pad(image, (size,), 'constant', constant_values=0)

def add_layers_to_image(image, number_of_kernels):
    multilayred_image=np.zeros((number_of_kernels, image.shape[0], image.shape[1]))
    for i in range (number_of_kernels):
        multilayred_image[i]=image
    return multilayred_image

def convolution(image, kernel):
    channels=np.zeros((image.shape[0],image.shape[1]-2,image.shape[2]-2)) - 1
    for rows in range(1, image.shape[1] - 1):
        for columns in range(1, image.shape[2] - 1):
            part_of_img=image[:, rows - 1:rows + 2, columns - 1:columns + 2]
            conv=np.sum(part_of_img * kernel, axis=(1, 2))
            channels[:, rows - 1, columns - 1]=conv
    return channels


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

class  Non_linear: # non linear activation layer
    def prop(self,prev_layer):
        self.prev_layer = prev_layer # w derv
        self.curr_layer = softmax(self.prev_layer)
    def back_prop(self,error,lr=0.1):
        self.error = error
        self.back_prop_error = softmax_derv(self.prev_layer)*self.error
if __name__ == "__main__":
    digits=load_digits()
    digits.data
    img_index = 15
    number_of_kernels = 6
    a = zero_pad_image(digits.images[img_index],1)
    multi_layred_image = add_layers_to_image(a,number_of_kernels)
    padded_shape = a.shape
    kernel= np.random.uniform(-1,1,(6,3,3))
    # kernel[0:2] = kernel[0:2] - ([[[1, 0, 1], [1, 0, 1], [1, 0, 1]], [[0, 1, 0], [1, -5, 1], [0, 1, 0]]])
    conv_output=convolution(multi_layred_image, kernel)
    x = conv_output.flatten()
    w2 = np.random.uniform(-1,1,(384,10))
    layer_2=x @ w2
    layer_2_x_derv=w2.T
    layer_2_w_derv=x.T
    layer_2_softmax_activation= softmax(layer_2)
    layer_2_softmax_derv = softmax_derv(layer_2)
    loss_var = np.sum(loss(layer_2_softmax_activation,hot_encode_labels(digits.target[img_index])))
    error = layer_2_softmax_activation - hot_encode_labels(digits.target[img_index])
    delta_w2=layer_2_w_derv @ (error * layer_2_softmax_derv)

    plt.gray()
    plt.matshow(a)
    plt.matshow(conv_output[5])
    plt.show(block=False)
