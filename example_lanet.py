import tensorflow as tf
import numpy as np

train_data=np.load("/media/dana/Workplace/Data/mnist/npy1/train_data.npy")
train_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/train_labels.npy")
test_data=np.load("/media/dana/Workplace/Data/mnist/npy1/test_data.npy")
test_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/test_labels.npy")


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def init_bias(shape):
    return tf.Variable(tf.constant(0.0,shape=shape))

#def lenet_model(train_data,train_labels,test_data,test_labels):
input_data=tf.placeholder("float",shape=[None,28,28])
input_label=tf.placeholder("float",shape=[None,10])
def lenet_model(train_data,train_labels,test_data,test_labels):
    with tf.name_scope('conv1') as scope:
        
        weights=init_weights([5,5,1,20])

        conv1_out=tf.nn.conv2d(tf.reshape(input_data,[-1,28,28,1]),weights,strides=[1,1,1,1],padding='SAME')
        pool1=tf.nn.max_pool(conv1_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('conv2') as scope:
        
        weights=init_weights([5,5,20,50])

        conv2_out=tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')
        pool2=tf.nn.max_pool(conv2_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        pool2_flat=tf.reshape(pool2,[-1,7*7*50])

    with tf.name_scope('fc1') as scope:
        weights=init_weights([7*7*50,500])
        bias=init_bias([500])
        relu1=tf.nn.relu(tf.matmul(pool2_flat,weights)+bias)

    with tf.name_scope('fc2') as scope:
        weights=init_weights([500,10])
        bias=init_bias([10])
        y=tf.nn.softmax(tf.matmul(relu1,weights)+bias)


    cross_entropy=-tf.reduce_sum(input_label*tf.log(y))
    train_op=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(input_label,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))  
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #saver.restore(sess,'./sanp/56_model.skpt')
        #saver=tf.train.Saver()
        j=0
        for i in range(100):
            for start, end in zip(range(0,len(train_data),128),range(128,len(train_data)+1,128)):
                train_op.run(feed_dict={input_data:train_data[start:end],input_label:train_labels[start:end]})
                j+=1
                if j%50==0:
                    train_accurary=accuracy.eval(feed_dict={input_data:test_data,input_label:test_labels})
                    print(i,train_accurary)
            path='./sanp/'+str(i)+'_model.skpt'
            saver_file=saver.save(sess,path)
            print("sucussful:",saver_file)



# train_data=np.load("/media/dana/Workplace/Data/mnist/npy1/train_data.npy")
# train_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/train_labels.npy")
# test_data=np.load("/media/dana/Workplace/Data/mnist/npy1/test_data.npy")
# test_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/test_labels.npy")

lenet_model(train_data,train_labels,test_data,test_labels)
