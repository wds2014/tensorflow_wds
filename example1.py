import tensorflow as tf
import numpy as np

a=tf.placeholder("float")
b=tf.placeholder("float")
y=tf.multiply(a,b)

with tf.Session() as sess:
    print(sess.run(y,feed_dict={a:3,b:5}))


trx=np.linspace(-1,1,101)
print(trx)
trY=2*trx+np.random.randn(*trx.shape)*0.33
x=tf.placeholder("float")
y=tf.placeholder("float")
def model(x,w):
    return tf.multiply(x,w)

w=tf.Variable(0.0,name="weights")
y_model=model(x,w)
cost = tf.square(y-y_model)
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for (xi,yi) in zip(trx,trY):
            sess.run(train_op,feed_dict={x:xi,y:yi})
    print(sess.run(w))

for start,end in zip(range(0,1000,128),range(128,1001,128)):
    print(start,end)