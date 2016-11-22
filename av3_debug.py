import time
import tensorflow as tf
import numpy as np
from av3_input import launch_enqueue_workers


W = tf.Variable(tf.zeros([8000, 1]))
b = tf.Variable(tf.zeros([1]))

sess = tf.Session()







#print "initialized all variables"
#print "weights:", sess.run(W)
#print "bises:", sess.run(b)

image_queue = launch_enqueue_workers(sess=sess, pixel_size=1, side_pixels=20, num_workers=1, batch_size=100,database_index_file_path="train_set.csv", num_epochs=False)

y_, x_image_batch = image_queue.dequeue_many(100)

print "image batch dimensions", x_image_batch.get_shape()
# formally adding one depth dimension to the input
x_image_with_depth = tf.reshape(x_image_batch, [-1, 8000])




y_conv = tf.matmul(x_image_with_depth, W) + b



train_squared_error = tf.reduce_mean(tf.square(y_ - y_conv))
train_step_run = tf.train.AdamOptimizer(1e-4).minimize(train_squared_error)

init = tf.initialize_all_variables()
sess.run([init])

print "image_batch", sess.run(x_image_batch)
print "image with depth", sess.run(x_image_with_depth)
print "y conv:", sess.run(y_conv)

i = 0
while True:
    i+=1
    print "train step:", sess.run([train_squared_error,train_step_run]),i
