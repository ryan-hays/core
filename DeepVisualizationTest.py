#focus on visualizing filters, default in format of 5,5,20(3 sets of convo + pooling layers)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

#import data, set up graph and placeholders
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


with tf.device("/gpu:0"):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name="x-in")
    true_y = tf.placeholder(tf.float32, [None,10], name="y-in")
    keep_prob = tf.placeholder("float")


    #network architecture specifications
    x_image = tf.reshape(x,[-1,28,28,1])

    hidden_1 = slim.conv2d(x_image, 8, [5,5])
    pool_1 = slim.max_pool2d(hidden_1, [2,2])

    hidden_2 =  slim.conv2d(pool_1,16,[5,5])
    pool_2 = slim.max_pool2d(hidden_2,[2,2])

    hidden_3 = slim.conv2d(pool_2,32,[5,5])
    hidden_3_dropped = slim.dropout(hidden_3,keep_prob)
    out_y = slim.fully_connected(slim.flatten(hidden_3_dropped),10,activation_fn=tf.nn.softmax)

    cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))
    correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    train_step =  tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#train network
batchSize = 200
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(15000):
	batch = mnist.train.next_batch(batchSize)
	sess.run(train_step, feed_dict = {x:batch[0],true_y:batch[1], keep_prob:0.5})
	if i % 100 == 0 and i != 0:
		trainAccuracy = sess.run(accuracy,feed_dict={x:batch[0],true_y:batch[1], keep_prob:1.0})
		print("step %d, training accuracy %g"%(i,trainAccuracy))


		
testAccuracy =  sess.run(accuracy, feed_dict={x:mnist.test.images, true_y:mnist.test.labels,keep_prob:1.0})
print("test accuracy %g"%(testAccuracy))



#functions for visualizing accuracy
def getActivations(layer,stimuli,i):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units,i)
	

def plotNNFilter(units,j):
    plt.clf()
    filters = units.shape[3]
    plt.figure(1, figsize=(40,40))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1

   
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.imshow(units[0,:,:,i],interpolation="nearest",cmap="gray")
    
    fig = plt.gcf()
    fig.set_size_inches(4,4)
    
  #  fig_size = plt.rcParams["figure.figsize"]
  #  print(fig_size)

    figNm = 'NewImages/image' + str(j) + '.png'
    
    if(j % 5 == 1) | (j % 5 == 2):
        fig.savefig(figNm, dpi=128)
    elif(j % 5 == 3) | (j % 5 == 4):
        fig.savefig(figNm, dpi=64)
    else:
        fig.savefig(figNm, dpi=32)
    
for i in range(10):
    imageToUse = mnist.test.images[i]
    getActivations(hidden_1,imageToUse,i*5 + 1)
    getActivations(pool_1,imageToUse,i*5 + 2)
    getActivations(hidden_2,imageToUse,i*5 + 3)
    getActivations(pool_2,imageToUse, i*5 + 4)
    getActivations(hidden_3, imageToUse, i*5 + 5)