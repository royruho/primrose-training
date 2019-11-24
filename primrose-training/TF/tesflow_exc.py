# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

tf.reset_default_graph()

node1= tf.constant(5) 
node2= tf.constant(11)

node3=node1+node2

sess = tf.Session() 

sess.run(node3)

print (sess.run(node3))

x=tf.placeholder(tf.int32)
y=tf.placeholder(tf.int32)
z=x+y
sess = tf.Session() 

print (sess.run(z,feed_dict= {x:5.0,y:6.0}))


w=tf.get_variable("w1",initializer=tf.constant(22))

b=tf.get_variable("b1",initializer=tf.constant(24))

sess = tf.Session() 

sess.run(tf.global_variables_initializer())

sess.run(w+b)

linear_model=w*x+b

sess.run(linear_model,feed_dict={x:[1,2,3,4]})

sqr_error=tf.square(y-linear_model)

sess.run(sqr_error,feed_dict={x:[1,2,3,4],y:[6,9,10,12]})

loss=tf.reduce_sum(sqr_error)

sess.run(loss,feed_dict={y:[0,-1,-2,-3],x:[1,2,3,4]})


sess.run(tf.assign(w,-1))
sess.run(tf.assign(b,1))

sess.run(loss,feed_dict={y:[0,-1,-2,-3],x:[1,2,3,4]})


# Calc optimizer
tf.reset_default_graph()

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
w=tf.get_variable("w2",initializer=tf.constant(22.0,dtype=tf.float32),dtype=tf.float32)
b=tf.get_variable("b2",initializer=tf.constant(24.0,tf.float32),dtype=tf.float32)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

linear_model=w*x+b
sqr_error=tf.square(y-linear_model)
loss=tf.reduce_sum(sqr_error)

optimizer=tf.train.GradientDescentOptimizer(0.01)

mini=optimizer.minimize(loss)

x_set = [1.0, 2.0, 3.0, 4.0]
y_set = [3.0, -1.0, -2.0, -3.0]

print("w={}, b={}, loss={}:".format(sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))

for i in range(1000):
    sess.run(mini,{x:x_set, y:y_set})

print("epoch={}, w={}, b={}, loss={}:".format(i,sess.run(w), sess.run(b), sess.run(loss,{x:x_set, y:y_set})))

# quadric equation

tf.reset_default_graph()

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
a=tf.Variable(2.0,name="a",dtype=tf.float32)
b=tf.Variable(4.0, name="b",dtype=tf.float32)
c=tf.Variable(1.0, name="c",dtype=tf.float32)

sess = tf.Session() 
sess.run(tf.initialize_all_variables())

solution=a*tf.square(x)+b*x+c
eq_error=tf.square(y-solution)
loss=tf.reduce_sum(eq_error)

optimizer=tf.train.GradientDescentOptimizer(0.001)

mini=optimizer.minimize(loss)

x_set = [1.0, 0.0, 4.0, -1.0]
y_set = [8.0, 2.0, 86.0, 6.0]

print("a={}, b={}, c={}, loss={}:".format(sess.run(a), sess.run(b), sess.run(c), sess.run(loss,{x:x_set, y:y_set})))

for i in range(1000):
    sess.run(mini,{x:x_set, y:y_set})
print("epoch={},a={}, b={}, c={}, loss={}:".format(i,sess.run(a), sess.run(b), sess.run(c), sess.run(loss,{x:x_set, y:y_set})))

