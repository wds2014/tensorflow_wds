# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:46:58 2018

@author: dana
"""

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
    
def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
    
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))
    
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    

train_data=np.load('/media/dana/Workplace/Data/mnist/npy1/train_data.npy').reshape(-1,28,28)
train_labels=np.load('/media/dana/Workplace/Data/mnist/npy1/train_labels.npy')
test_data=np.load('/media/dana/Workplace/Data/mnist/npy1/test_data.npy').reshape(-1,28,28)
test_labels=np.load('/media/dana/Workplace/Data/mnist/npy1/test_labels.npy')

sess=tf.InteractiveSession()

w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])


input_data=tf.placeholder('float',shape=[None,28,28])
input_label=tf.placeholder('float',shape=[None,10])
keep_prob=tf.placeholder("float")

h_conv1=tf.nn.relu(conv2d(tf.reshape(input_data,[-1,28,28,1]),w_conv1)+b_conv1)
h_pool1=max_pool_2(h_conv1)
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2(h_conv2)
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

cross_entropy=-tf.reduce_sum(input_label*tf.log(y_conv))
train_op=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(input_label,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))    
tf.global_variables_initializer().run()
j=0
for i in range(20000):
    for start,end in zip(range(0,len(train_data),128),range(128,len(train_data)+1,128)):
        train_op.run(feed_dict={input_data:train_data[start:end],input_label:train_labels[start:end],keep_prob:0.5})
        j+=1
        if j%50==0:
            train_accuracy=accuracy.eval(feed_dict={input_data:test_data,input_label:test_labels,keep_prob:1.0})
            print(i,train_accuracy)
    #print(i,np.mean(np.argmax(test_labels,axis=1)==tf.arg_max(sess.run(predit_op,feed_dict={input_data:test_data}),axis=1)))
    