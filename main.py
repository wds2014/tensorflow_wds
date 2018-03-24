from lenet import lenet_net
import numpy as np
import tensorflow as tf
import cv2

if __name__=='__main__':
    train_data=np.load("/media/dana/Workplace/Data/mnist/npy1/train_data.npy")
    train_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/train_labels.npy")
    test_data=np.load("/media/dana/Workplace/Data/mnist/npy1/test_data.npy")
    test_labels=np.load("/media/dana/Workplace/Data/mnist/npy1/test_labels.npy")
    lenet_net=lenet_net()
    lenet_net.lenet_model()
    #lenet_net.train('./snap/6_model.skpt',train_data,train_labels,test_data,test_labels)
    image=cv2.imread('./pics/wds1.png')
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image1=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)
    
    image=cv2.imread('./pics/wds2.png')
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image2=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

    image=cv2.imread('./pics/wds3.png')
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image3=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)
    #print(image1.shape,image2.shape,image3.shape)
    image_data=[]

    image_data.append(image1)
    image_data.append(image2)
    image_data.append(image3)
    input_image=np.array(image_data)
    print(input_image.shape)
    result=lenet_net.test(input_image,'./snap/3_model.skpt')
    print(result)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 100,100)
    for i in range(3):
        cv2.imshow('Image',input_image[i])
        cv2.waitKey(2000)