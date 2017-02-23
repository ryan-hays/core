import time,os
import tensorflow as tf
import numpy as np
from av4_input import index_the_database_into_queue,image_and_label_queue
from av4_networks import *
from av4_DCGAN_model import DCGAN

# telling tensorflow how we want to randomly initialize weights

def train():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
              sess,
              input_width=FLAGS.side_pixels,
              input_height=FLAGS.side_pixels,
              input_depth = FLAGS.side_pixels,
              output_width=FLAGS.side_pixels,
              output_height=FLAGS.side_pixels,
              output_depth = FLAGS.side_pixels,
              batch_size=FLAGS.batch_size,
              c_dim=FLAGS.num_channels,
              is_crop=FLAGS.is_crop,
              checkpoint_dir=FLAGS.checkpoint_dir,
              sample_dir=FLAGS.sample_dir)


    "train a network"
    # it's better if all of the computations use a single session
    sess = tf.Session()

    # create a filename queue first
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)
    # create labels

    # create an epoch counter
    batch_counter = tf.Variable(0)
    batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))
    epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)

    # create a custom shuffle queue
    _,current_epoch,label_batch,sparse_image_batch = image_and_label_queue(batch_size=FLAGS.batch_size, pixel_size=FLAGS.pixel_size,
                                                                          side_pixels=FLAGS.side_pixels, num_threads=FLAGS.num_threads,
                                                                          filename_queue=filename_queue, epoch_counter=epoch_counter)

    image_batch = tf.sparse_tensor_to_dense(sparse_image_batch,validate_indices=False)

    # keep_prob = tf.placeholder(tf.float32)


    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
        .minimize(dcgan.d_loss, var_list=dcgan.d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
        .minimize(dcgan.g_loss, var_list=dcgan.g_vars)

    try:
      sess.run(tf.global_variables_initializer())
    except:
        sess.run(tf.initialize_all_variables())



    dcgan.g_sum = tf.summary.merge([dcgan.z_sum, dcgan.d__sum,
        dcgan.G_sum, dcgan.d_loss_fake_sum, dcgan.g_loss_sum])
    dcgan.d_sum = tf.summary.merge(
        [dcgan.z_sum, dcgan.d_sum, dcgan.d_loss_real_sum, dcgan.d_loss_sum])
    dcgan.writer = tf.summary.FileWriter("./logs", sess.graph)

    batch_z = np.random.uniform(-1, 1, size=(dcgan.sample_num , dcgan.z_dim))
    
    print "sample number", dcgan.sample_num


   
    


    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:
        start = time.time()
        batch_num = sess.run(batch_counter_increment)

        batch_images = sess.run([image_batch])
        print "images shape ",len(batch_images)
        print "image shape ",batch_images[0].shape
        # don't know why it return a list
        sess.run([dcgan.d_loss, dcgan.g_loss],feed_dict={dcgan.inputs:batch_images[0],dcgan.z: batch_z})
        # Update D network
        _, summary_str = self.sess.run([d_optim, dcgan.d_sum],
            feed_dict={ self.inputs: batch_images, dcgan.z: batch_z })
        dcgan.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, dcgan.g_sum],
            feed_dict={ dcgan.z: batch_z })
        dcgan.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, dcgan.g_sum],
            feed_dict={ dcgan.z: batch_z })
        dcgan.writer.add_summary(summary_str, counter)
          
        errD_fake = dcgan.d_loss_fake.eval({ dcgan.z: batch_z })
        errD_real = dcgan.d_loss_real.eval({ self.inputs: batch_images })
        errG = dcgan.g_loss.eval({dcgan.z: batch_z})

        if (batch_num % 1000 == 999):
            sess.run()
            # once in a while save the network state and write variable summaries to disk
            try:
                samples, d_loss, g_loss = self.sess.run(
                [dcgan.sampler, dcgan.d_loss, dcgan.g_loss],
                feed_dict={
                    dcgan.z: sample_z,
                    dcgan.inputs: sample_inputs,
                },
              )
                save_images(samples, [40, 40,40],
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
                print("one pic error!...")

    assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'


class FLAGS:
    """important model parameters"""

    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 0.5
    # size of the box around the ligand in pixels
    side_pixels = 40
    # Total number of channels for the data
    num_channels = 14
    # weights for each class for the scoring function
    # number of times each example in the dataset will be read
    num_epochs = 50000 # epochs are counted based on the number of the protein examples
    # usually the dataset would have multiples frames of ligand binding to the same protein
    # av4_input also has an oversampling algorithm.
    # Example: if the dataset has 50 frames with 0 labels and 1 frame with 1 label, and we want to run it for 50 epochs,
    # 50 * 2(oversampling) * 50(negative samples) = 50 * 100 = 5000
    # num_classes = 2
    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 10
    # number of background processes to fill the queue with images
    num_threads = 512
    # data directories

    #whether or not we want to crop input images according to output dimensions
    is_crop = True

    #directory to save the checkpoints
    checkpoint_dir = "checkpoint"

    #directory to save the image samples
    sample_dir = "samples"

    # path to the csv file with names of images selected for training
    database_path = "/home/ubuntu/common/data/labeled_av4"
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None#'./summaries/36_netstate/saved_state-23999'
    learning_rate = 0.0002
    beta1 = 0.5

def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start"""
    summaries_dir = os.path.join(FLAGS.summaries_dir)
    # FLAGS.run_index defines when
    FLAGS.run_index = 1
    while ((tf.gfile.Exists(summaries_dir + "/"+ str(FLAGS.run_index) +'_train' ) or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_test' ))
           or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate') or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_logs')) and FLAGS.run_index < 1000:
        FLAGS.run_index += 1
    else:
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_train' )
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_test')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_logs')
    
    train()

if __name__ == '__main__':
    tf.app.run()
