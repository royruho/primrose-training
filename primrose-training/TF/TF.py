import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

node1 = tf.constant([1,2,3,4,5])
node2 = tf.constant([1,1,2,3,5])
node3 = tf.math.multiply(node1,node2)
node4 = tf.reduce_sum(node3)

x = tf.placeholder(tf.float32, shape=[1,None], name = "x_plaveholder")
y = tf.placeholder(tf.float32, shape=[1,None], name = "y_placeholder")
Wa = tf.Variable(7)
ba = tf.Variable(8)
Wb = tf.math.add(Wa,ba)
xysum = tf.math.add(x,y)

sess = tf.Session()
sess.run(Wa.initializer)
sess.run(ba.initializer)
print(sess.run(Wb))
print(sess.run(xysum, feed_dict={x: [[3]], y: [[4]]}))
print(sess.run(node3))
print(sess.run(node4))

# Calc optimizer
tf.reset_default_graph()
sess = tf.Session()

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
w=tf.Variable(22.0)
b=tf.Variable(24.0)

sess.run(tf.global_variables_initializer())

linear_model=w*x+b
sqr_error=tf.square(y-linear_model)
loss=tf.reduce_sum(sqr_error)

data = pd.read_csv(r"C:\Users\royru\Desktop\primrose\github\primrose-training\primrose-training\TF\data_for_linear_regression_tf").astype(np.float32)
pdinputs = data['x'].to_numpy()
pdtargets = data['y'].to_numpy()
x_set = pdinputs
y_set = pdtargets

optimizer=tf.train.GradientDescentOptimizer(0.01)
mini=optimizer.minimize(loss)


print("w={}, b={}, loss={}:".format(sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))

for i in range(2):
    sess.run(mini,{x:x_set, y:y_set})

print("epoch={}, w={}, b={}, loss={}:".format(i,sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))
sess = tf.Session()



tf.reset_default_graph()
sess = tf.Session()
class Lin_model:
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, X):
        multi = tf.math.multiply(X, self.W)
        output = tf.math.add(multi, self.b)
        return output

    def loss(self,yp,target):
        loss_var = tf.reduce_mean(tf.square(yp-target))
        return loss_var

    def train(self, inputs, targets, learning_rate):
        sess.run(
            with tf.GradientTape() as calc_gradient:
                current_loss = self.loss(model(inputs), targets)
            dW, db = calc_gradient.gradient(current_loss, [model.W, model.b])
            model.W.assign_sub(learning_rate * dW)
            model.b.assign_sub(learning_rate * db))

model = Lin_model()
sess.run(tf.global_variables_initializer())
assert (sess.run(model(3.0)) == 15) # check call function
assert ((sess.run(model.loss(3.0, 1.0))) == 4) # check loss function

data = pd.read_csv(r"C:\Users\royru\Desktop\primrose\github\primrose-training\primrose-training\TF\data_for_linear_regression_tf").astype(np.float32)
pdinputs = data['x'].to_numpy()
pdtargets = data['y'].to_numpy()

inputs_placeholder = tf.placeholder(tf.float32, shape=[1,None], name = "inputs")
targets_placeholder = tf.placeholder(tf.float32, shape=[1,None], name = "targets")

ts_inputs = tf.constant(pdinputs)
ts_targets = tf.constant(pdtargets)

#           ###
# train model #
#           ###
W_hist, b_hist = [], []
learning_rate = tf.constant([0.1])
epochs = range(5)
plt.figure()
plt.scatter(pdinputs, pdtargets, c='b')
plt.scatter(pdinputs, sess.run(model(pdinputs)), c='r')
plt.legend(['targets', 'predictions'])
plt.show()
print('Current loss: %1.6f' % sess.run(model.loss(model(ts_inputs), pdtargets)))
for epoch in epochs:
    W_hist.append(sess.run(model.W))
    b_hist.append(sess.run(model.b))
    current_loss = sess.run(model.loss(model(ts_inputs), pdtargets))
    sess.run(model.train(ts_inputs, ts_targets, learning_rate))
plt.figure()
plt.scatter(pdinputs, pdtargets, c='b')
plt.scatter(pdinputs, sess.run(model(ts_inputs)), c='r')
print('Current loss: %1.6f' % sess.run(model.loss(model(ts_inputs), pdtargets)))
plt.show()
plt.figure()
plt.plot(epochs, W_hist, 'r',
         epochs, b_hist, 'b')
plt.legend(['W', 'b'])
plt.show()