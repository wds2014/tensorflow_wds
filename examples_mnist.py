import tensorflow as tf
import numpy as np
import sys
import glob
import os.path
base_root='/home/dana/caffe_wds1'

def get_label(label,total_num):
    y=np.zeros((total_num))
    for i in range(total_num):
        if i == int(label):
            y[i]=1
    return y

def read_images(img_txt):
    X=[]
    Y=[]
    with open(img_txt) as f:
        for i, lines in enumerate(f.readlines()):
            img_path=lines.split()
            image=cv2.imread(img_path[0])
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            label=get_label(img_path[1],10)
            #label=int(img_path[1])
            X.append(image)
            Y.append(label)
    X=np.array(X,dtype=np.float32)/255
    Y=np.array(Y,dtype=np.float32)
    return X,Y

def model(w1,w2,x):
    h=tf.nn.sigmoid(tf.matmul(x,w1))
    return tf.matmul(h,w2)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))


    
if __name__=='__main__':
    train_data=np.load("/media/dana/Workplace/Data/mnist/npy1/train_data.npy").reshape(60000,784)
    train_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/train_labels.npy")
    test_data=np.load("/media/dana/Workplace/Data/mnist/npy1/test_data.npy").reshape(10000,784)
    test_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/test_labels.npy")
    input_data=tf.placeholder("float",[None,784])
    input_labels=tf.placeholder("float",[None,10])
    w1=init_weights([784,625])
    w2=init_weights([625,10])

    nn_label=model(w1,w2,input_data)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_label,labels=input_labels))
    train_op=tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    predict_op=tf.argmax(nn_label,1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100):
            for start,end in zip(range(0,len(train_data),128),range(128,len(train_data)+1,128)):
                sess.run(train_op,feed_dict={input_data:train_data[start:end],input_labels:train_labels[start:end]})
            print(i,np.mean(np.argmax(test_data,axis=1)==sess.run(predict_op,feed_dict={input_data:test_data})))