import time,os, math
import tensorflow as tf
import numpy as np
from av3_input import launch_enqueue_workers


# telling tensorflow how we want to randomly initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def conv_layer(layer_name, input_tensor, filter_size, strides=[1, 1, 1, 1, 1], padding='SAME'):
  """makes a simple convolutional layer"""
  input_depth = filter_size[3]
  output_depth = filter_size[4]
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      W_conv = weight_variable(filter_size)
      variable_summaries(W_conv, layer_name + '/weights')
    with tf.name_scope('biases'):
      b_conv = bias_variable([output_depth])
      variable_summaries(b_conv, layer_name + '/biases')
    h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) + b_conv
    tf.histogram_summary(layer_name + '/pooling_output', h_conv)
    print layer_name,"output dimensions:", h_conv.get_shape()
    return h_conv

def relu_layer(layer_name,input_tensor,act=tf.nn.relu):
  """makes a simple relu layer"""
  with tf.name_scope(layer_name):
    h_relu = act(input_tensor, name='activation')
    tf.histogram_summary(layer_name + '/relu_output', h_relu)
    tf.scalar_summary(layer_name + '/sparsity', tf.nn.zero_fraction(h_relu))

  print layer_name, "output dimensions:", h_relu.get_shape()
  return h_relu

def pool_layer(layer_name,input_tensor,ksize,strides=[1, 1, 1, 1, 1],padding='SAME'):
  """makes a simple pooling layer"""
  with tf.name_scope(layer_name):
    h_pool = tf.nn.max_pool3d(input_tensor,ksize=ksize,strides=strides,padding=padding)
    tf.histogram_summary(layer_name + '/pooling_output', h_pool)
    print layer_name, "output dimensions:", h_pool.get_shape()
    return h_pool

def fc_layer(layer_name,input_tensor,output_dim):
  """makes a simple fully connected layer"""
  input_dim = int((input_tensor.get_shape())[1])

  with tf.name_scope(layer_name):
    weights = weight_variable([input_dim, output_dim])
    variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      h_fc = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/fc_output', h_fc)
    print layer_name, "output dimensions:", h_fc.get_shape()
    return h_fc


def max_net(x_image_batch,keep_prob):
    "making a simple network that can receive 20x20x20 input images. And output 2 classes"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        # reshaping 8000 vector into 20*20*20 image
        x_image = tf.reshape(x_image_batch, [-1, 20, 20, 20, 1])


    #making different regions
    inp_tensor = x_image
    max_no_layers = 20
    no_layers = 0
    inp_size = 1
    out_size = 30
    current_matrix_edge_size = 20
    while no_layers < max_no_layers:
        h_conv = conv_layer(layer_name='conv'+str(no_layers)+'_2x2x2', input_tensor=inp_tensor, filter_size=[2, 2, 2, inp_size, out_size])
        inp_tensor = relu_layer(layer_name='relu'+str(no_layers), input_tensor=h_conv)
        if no_layers < 3:
                inp_tensor = pool_layer(layer_name='pool'+str(no_layers)+'_2x2x2', input_tensor=inp_tensor, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])
                current_matrix_edge_size = max(int(current_matrix_edge_size+1)/2, 1)
        inp_size = out_size
        out_size = out_size + 15
        no_layers += 1

    with tf.name_scope("flatten_layer"):
        h_pool4_flat = tf.reshape(inp_tensor, [-1, current_matrix_edge_size*current_matrix_edge_size*current_matrix_edge_size * inp_size])

    h_fc1 = fc_layer(layer_name="fc1", input_tensor=h_pool4_flat, output_dim=1024)
    h_fc1_relu = relu_layer(layer_name="fc1_relu", input_tensor=h_fc1)

    with tf.name_scope("dropout"):
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)
    return y_conv


def compute_weighted_cross_entropy_mean(logits, labels,batch_size):
    """computes weighted cross entropy mean for a two class classification.
    Applies tf.nn.weighted_cross_entropy_with_logits
    accepts "labels" instead of "targets" as in
    tf.nn.sparse_softmax_cross_entropy_with_logits"""

    with tf.name_scope('weighted_cross_entropy_mean'):  # TODO (multiclass) now hardcoded to two classes

        # first column is inverted labels, second column is labels
        batch_class_zero = tf.reshape(-labels + 1, [batch_size, 1])
        batch_class_one = tf.reshape(labels, [batch_size, 1])
        batch_y = tf.concat(1, (batch_class_zero, batch_class_one))
        weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits, batch_y, pos_weight=5000, name=None)

        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy, name='cross_entropy')
        return cross_entropy_mean


#def unbalanced_sparse_softmax_cross_entropy_with_logits(logits,targets,class_weights=[1,1]):
#    entropy = tf.reduce_sum(tf.mul(tf.to_float(class_weights),targets) * -tf.log(tf.nn.softmax(logits)),reduction_indices=1)
#    return entropy

#def unbalanced_sparse_softmax_cross_entropy_with_logits(logits,labels,class_weights=[1,1]):
##
#
#    batch_size = int(logits.get_shape()[0])
#    num_classes = int(logits.get_shape()[1])
#    print "batch size:",batch_size
#    print "num classes",num_classes
#    left = tf.cast(tf.range(0,batch_size),tf.int64)
#    right = tf.cast((labels),tf.int64)
#    indices = tf.pack((left,right),axis=1)

#    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size,dtype=tf.float32),shape=[batch_size,num_classes])
#    targets = tf.sparse_tensor_to_dense(sparse_targets)
#    entropy = tf.reduce_sum(tf.mul(tf.to_float(class_weights),tf.to_float(targets)) * -tf.log(tf.nn.softmax(logits)),reduction_indices=1)
#    print "shape of the entropy", entropy.get_shape()
#    return entropy,logits





def unbalanced_sparse_softmax_cross_entropy_with_logits(logits,labels,class_weights=[1,1]):
    # convert labels to targets first
    batch_size = int(logits.get_shape()[0])
    num_classes = int(logits.get_shape()[1])
    # because the default format of labels is float32, it needs to be converted to int64
    labels = tf.cast(labels,dtype=tf.int32)

    print("calc softmax")
    indices = tf.cast(tf.pack((tf.range(0,batch_size),labels),axis=1),dtype=tf.int64)

    sparse_targets = tf.SparseTensor(indices=indices, values=tf.ones(batch_size,dtype=tf.float32),shape=[batch_size,num_classes])
    targets = tf.sparse_tensor_to_dense(sparse_targets)

    # formula:
    # ent = targets * -log(softmax(logits)) = targets * -log(softmax(x))
    # ent = targets * -log(e**x/sum(e**x))
    # ent = targets * -(x - log(sum(e**x))
    # ent = targets * -(x - soft_maximum_x)

    # stable way of soft_maximum_x = log(sum e**x))
    # if we shift by some constant K
    # log(e**x1 + e**x2 + e**x3...) = K + log((e**x1)/K + (e**x2)/K + (e**x3)/K)
    # K + log (e**(x1-K) + e**(x2-K) + e**(x3-K)+...)
    # if K = max(x), there is no overflow since all (x-K) are negative
    # log(sum(e**x)) = max + log(e**(x1-max) + e**(x2-max) + e**..)

    max_logits = tf.reduce_max(logits, reduction_indices=1)
    soft_maximum_x = max_logits + tf.log(tf.reduce_sum(tf.exp(logits - tf.tile(tf.reshape(max_logits,shape=[batch_size,1]),multiples=[1,num_classes])),1))
    simple_entropy = targets * -(logits - tf.tile(tf.reshape(soft_maximum_x,shape=[batch_size,1]),multiples=[1,num_classes]))
    unbalanced_entropy = tf.reduce_sum(class_weights * simple_entropy,reduction_indices=1)
    return unbalanced_entropy

#def imbalanced_dequeue(queue, keepprob):
#    count=0
#    y_ = tf.zeros(FLAGS.batch_size)
#    x_image_batch = tf.tile(tf.zeros(8000), FLAGS.batch_size)
#    while count < FLAGS.batch_size:
	#assuming queue has been randomized
#	y, x, _,_ = train_image_queue.dequeue()
#	add = True
#	if tf.equal(y, tf.constant(0)) and math.rand() < keepprob:
#	   add = False

#	if add:
#	    y_sparse = tf.SparseTensor(count, y, tf.shape(y_))
#	    y_ = tf.add(y_, tf.sparse_tensor_to_dense(y_sparse))
#	    indices = tf.add(tf.tile(count, 8000), tf.range(0, 8000))
#	    x_sparse = tf.SparseTensor(indices,x,tf.shape(x_image_batch))
#	    x_image_batch = tf.add(x_image_batch, tf.sparse_tensor_to_dense(x_sparse))
#	    count+=1
#    return (y_,x_image_batch)

def train():
    "train a network"
    print("opening")
    # create session since everything is happening in one
    sess = tf.Session()
    print("b4 LE")
    train_image_queue,filename_coordinator = launch_enqueue_workers(sess=sess, pixel_size=FLAGS.pixel_size, side_pixels=FLAGS.side_pixels, num_workers=FLAGS.num_workers, batch_size=FLAGS.batch_size,
                                         database_index_file_path=FLAGS.train_set_file_path, num_epochs=FLAGS.num_epochs)
    print("after launch enqueue")
    y_, x_image_batch,_,_ = train_image_queue.dequeue_many(FLAGS.batch_size)
    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(x_image_batch, keep_prob)

    # todo
    #cross_entropy_mean = compute_weighted_cross_entropy_mean(y_conv, y_, FLAGS.batch_size) 
    cross_entropy = unbalanced_sparse_softmax_cross_entropy_with_logits(y_conv,y_,[1,50])
    
    cross_entropy_mean = tf.reduce_sum(cross_entropy) / FLAGS.batch_size
    y_opp = tf.reverse_sequence(y_,1,0)
    cross_entropy_mean_opp = tf.reduce_sum(unbalanced_sparse_softmax_cross_entropy_with_logits(y_conv,y_opp,[1,50]))
    cross_entropy_mean_diff = tf.sub(cross_entropy_mean - cross_entropy_mean_opp)

    with tf.name_scope('train'):
        tf.scalar_summary('unbalanced cross entropy mean', cross_entropy_mean)
        train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean_diff)


    with tf.name_scope('evaluate_predictions'):

#        # first: evaluate error when labels are randomly shuffled
#        # randomly shuffle along one of the dimensions:
        shuffled_y_ = tf.random_shuffle(y_)
        shuffled_cross_entropy_mean = tf.reduce_sum(unbalanced_sparse_softmax_cross_entropy_with_logits(y_conv,shuffled_y_,[1,50])) / FLAGS.batch_size


    # many small subroutines that are needed to save network state,logs, etc.

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    # merge all summaries
    merged_summaries = tf.merge_all_summaries()
    # create a _log writer object
    train_writer = tf.train.SummaryWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)
    # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test' + str(FLAGS.run_index))

    print("var init")
    # initialize all variables
    sess.run(tf.initialize_all_variables())

    batch_num = 0
    print("here")
    while not filename_coordinator.stop:
        print("init")
	start = time.time()
        training_error,_ = sess.run([cross_entropy_mean,train_step_run],feed_dict={keep_prob:0.5})
        print "step:", batch_num, "run error:", training_error,\
            "examples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))
        # once in a hundred batches calculate correct predictions
        if (batch_num % 1000 == 999):
            # evaluate and print a few things
            print "eval:-------------------------------------------------------------------------------------"
            shuffled_training_error,training_error,train_summary = sess.run([shuffled_cross_entropy_mean,cross_entropy_mean,merged_summaries],feed_dict={keep_prob:0.5})
            print "step:", batch_num, "run error:",training_error, "shuffled run error:", shuffled_training_error
            train_writer.add_summary(train_summary, batch_num)

            saver.save(sess,FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)

        # exit the loop in case there is something wrong with the setup and model diverged into inf
        assert not np.isnan(training_error), 'Model diverged with loss = NaN'
        batch_num+=1



class FLAGS:

    # important model parameters
    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 1
    # size of the box around the ligand in pixels
    side_pixels = 20
    # number of times each example in the dataset will be read
    num_epochs = 20

    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 50
    # number of background processes to fill the queue with images
    num_workers = 16

    # data directories
    # path to the csv file with names of images selected for training
    train_set_file_path = '../../common/datasets/labeled_pdb/train_set_short.csv'
    # path to the csv file with names of the images selected for testing
    test_set_file_path = '../../common/datasets/labeled_pdb/test_set_short.csv'
    # directory where to write variable summaries
    summaries_dir = './summaries'






def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start"""
    summaries_dir = os.path.join(FLAGS.summaries_dir)
    # FLAGS.run_index defines when
    FLAGS.run_index = 1
    while ((tf.gfile.Exists(summaries_dir + "/"+ str(FLAGS.run_index) +'_train' ) or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_test' ))
           or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate') or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_logs')) and FLAGS.run_index < 1000:
        FLAGS.run_index += 1
    else:
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_train' )
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_test')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_netstate')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) + '_logs')
    train()

if __name__ == '__main__':
    tf.app.run()
