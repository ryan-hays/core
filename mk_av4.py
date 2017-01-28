import time,os
import tensorflow as tf
import numpy as np
from av4_input import image_and_label_queue




def train():
    "train a network"
    # with the current setup all of the TF's operations are happening in one session
    sess = tf.Session()

    current_idx, dense_image, final_transition_matrix = image_and_label_queue(sess=sess,batch_size=FLAGS.batch_size,
                                                pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                num_threads=FLAGS.num_threads,database_path=FLAGS.database_path,
                                                                  num_epochs=FLAGS.num_epochs)

    #current_epoch,label_batch,image_batch = image_and_label_queue(sess=sess,batch_size=FLAGS.batch_size,
    #                                            pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
    #                                            num_threads=FLAGS.num_threads,database_path=FLAGS.database_path,
    #                                                              num_epochs=FLAGS.num_epochs)
    # TODO: write atoms in layers of depth
    # floating is temporary
    #float_image_batch = tf.cast(image_batch,tf.float32)

    #keep_prob = tf.placeholder(tf.float32)
    #predicted_labels= max_net(float_image_batch,keep_prob)

    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predicted_labels,label_batch)
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #tf.summary.scalar('cross entropy mean', cross_entropy_mean)

    # randomly shuffle along the batch dimension and calculate an error
    #shuffled_labels = tf.random_shuffle(label_batch)
    #shuffled_cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predicted_labels,shuffled_labels))
    #tf.summary.scalar('shuffled cross entropy mean', shuffled_cross_entropy_mean)

    # Adam optimizer is a very heart of the network
    #train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # merge all summaries and create a file writer object
    #merged_summaries = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)

    # create saver to save and load the network state
    #saver = tf.train.Saver()
    #if not FLAGS.saved_session is None:
    #    print "Restoring variables from sleep. This may take a while..."
    #    saver.restore(sess, FLAGS.saved_session)

    # initialize all variables (two thread variables should have been initialized in av4_input already)
    sess.run(tf.global_variables_initializer())

    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_num = 0
    while True:
        #start = time.time()
        my_idx,my_dense_image,my_final_matrix = sess.run([current_idx,dense_image,final_transition_matrix])
        print "idx:",my_idx
        print "dense_matrix:",my_final_matrix
        print "dense image:",my_dense_image
        time.sleep(5)
    #    epo,c_entropy_mean,_ = sess.run([current_epoch[0],cross_entropy_mean,train_step_run], feed_dict={keep_prob: 0.5})
    #    print "epoch:",epo,"global step:", batch_num, "\tcross entropy mean:", c_entropy_mean,
    #    print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

    #    if (batch_num % 100 == 99):
    #        # once in a while save the network state and write variable summaries to disk
    #        c_entropy_mean,sc_entropy_mean,summaries = sess.run(
    #            [cross_entropy_mean, shuffled_cross_entropy_mean, merged_summaries], feed_dict={keep_prob: 1})
    #        print "cross entropy mean:",c_entropy_mean, "shuffled cross entropy mean:", sc_entropy_mean
    #        train_writer.add_summary(summaries, batch_num)
    #        saver.save(sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)
    #
        batch_num += 1
    #assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'


class FLAGS:
    """important model parameters"""

    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 0.5
    # size of the box around the ligand in pixels
    side_pixels = 40
    # weights for each class for the scoring function
    # number of times each example in the dataset will be read
    num_epochs = 5000 # epochs are counted based on the number of the protein examples
    # usually the dataset would have multiples frames of ligand binding to the same protein
    # av4_input also has an oversampling algorithm.
    # Example: if the dataset has 50 frames with 0 labels and 1 frame with 1 label, and we want to run it for 50 epochs,
    # 50 * 2(oversampling) * 50(negative samples) = 50 * 100 = 5000

    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 200  # number of background processes to fill the queue with images
    num_threads = 512
    # data directories
    # path to the csv file with names of images selected for training
    database_path = "../../datasets/labeled_pdb_av4/**/"
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None


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