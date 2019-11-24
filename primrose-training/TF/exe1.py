# exe1 - tensorflow
# eran bamani
# 4.3.18
# -------------------------------------------------------
import numpy as np
import tensorflow as tf
# -------------------------------------------------------
#3
node1 = tf.constant(5)
node2 = tf.constant(11)
#4
node3 = tf.add(node1, node2)
#5
sess = tf.Session()
print(sess.run(node3))
#6
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = x+y
print(sess.run(z,{x:5.0, y: 6.0}))
#7
W = tf.get_variable('w1', initializer=tf.constant(7))
b = tf.get_variable("b1", initializer=tf.constant(8))
wb = tf.add(W, b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(wb))
#8
# Variables are used to store the state of a graph. Variables need a value to be initialized while
# declaring it. To declare a variable we use tf.Variable() command and initialize
# them before running the graph in a session.
# b = tf.Variable([.5],dtype=tf.float32)

# Placeholders are used to feed external data into a TensorFlow graph.
# It allows a value to be assigned later i.e. a place
# in the memory where we’ll store a value later on.
# To define a Placeholder we use tf.placeholder() command.

#9
linear_model = W*x+b

#10
print(sess.run(liner_model,feed_dict={x:[1, 2, 3, 4]}))

# 11
y = tf.placeholder(tf.int32)
sqr_error = tf.square(y-linear_model)
sess.run(sqr_error,feed_dict={x:[1,2,3,4],y:[6,9,10,12]})

#12
loss=tf.reduce_sum(sqr_error)

#13
sess.run(loss,feed_dict={y:[0,-1,-2,-3],x:[1,2,3,4]})

#14
sess.run(tf.assign(w,-1))
sess.run(tf.assign(b,1))

#15
tf.reset_default_graph()
optimizer = tf.train.GradientDescentOptimizer(0.01)

#16
mini = optimizer.minimize(loss)
x_set = [1.0, 2.0, 3.0, 4.0]
y_set = [3.0, -1.0, -2.0, -3.0]
print("w={}, b={}, loss={}:".format(sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))

#17
for i in range(1000):
    sess.run(mini,{x:x_set, y:y_set})

#18
print("epoch={}, w={}, b={}, loss={}:".format(i,sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))

