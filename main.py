from lenet import lenet_net
import numpy as np
import tensorflow as tf

if __name__=='__main__':
    train_data=np.load("/media/dana/Workplace/Data/mnist/npy1/train_data.npy")
    train_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/train_labels.npy")
    test_data=np.load("/media/dana/Workplace/Data/mnist/npy1/test_data.npy")
    test_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/test_labels.npy")
    lenet_net=lenet_net()
    lenet_net.lenet_model()
    lenet_net.train('./snap/6_model.skpt',train_data,train_labels,test_data,test_labels)