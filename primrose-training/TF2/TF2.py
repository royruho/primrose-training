import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.executing_eagerly()

node1 = tf.constant([1,2,3,4,5])
node2 = tf.constant([1,1,2,3,5])
node3 = tf.math.multiply(node1,node2)
node4 = tf.reduce_sum(node3)

Wa = tf.Variable(7)
ba = tf.Variable(8)
Wb = tf.math.add(Wa,ba)

tf.print (node3)
tf.print (node4)
tf.print ("wb", Wb)

class Lin_model:
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, X):
        multi = tf.math.multiply(X, self.W)
        output = tf.math.add(multi, self.b)
        return output

    def loss(self, yp, target):
        loss_var = tf.reduce_mean(tf.square(yp-target))
        return loss_var

    def train(self, inputs, targets, learning_rate):
        with tf.GradientTape() as calc_gradient:
            current_loss = self.loss(model(inputs), targets)
        dW, db = calc_gradient.gradient(current_loss, [model.W, model.b])
        model.W.assign_sub(learning_rate * dW)
        model.b.assign_sub(learning_rate * db)

model = Lin_model()
assert (model(3.0) == 15) # check call function
assert (model.loss(3.0, 1.0) == 4) # check loss function

data = pd.read_csv(r"C:\Users\royru\Desktop\primrose\github\primrose-training\primrose-training\TF\data_for_linear_regression_tf").astype(np.float32)
inputs = data['x']
targets = data['y']

#           ###
# train model #
#           ###
W_hist, b_hist = [], []
learning_rate = 0.1
epochs = range(5000)
plt.figure()
plt.scatter(inputs, targets, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()
print('Current loss: %1.6f' % model.loss(model(inputs), targets).numpy())
for epoch in epochs:
    W_hist.append(model.W.numpy())
    b_hist.append(model.b.numpy())
    current_loss = model.loss(model(inputs), targets)
    model.train(inputs, targets,learning_rate)
plt.figure()
plt.scatter(inputs, targets, c='b')
plt.scatter(inputs, model(inputs), c='r')
print('Current loss: %1.6f' % model.loss(model(inputs), targets).numpy())
plt.show()
plt.figure()
plt.plot(epochs, W_hist, 'r',
         epochs, b_hist, 'b')
plt.legend(['W', 'b'])
plt.show()