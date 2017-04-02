import time,re
import tensorflow as tf
import numpy as np
from av4_input import image_and_label_queue,index_the_database_into_queue,read_receptor_and_ligand,convert_protein_and_ligand_to_image
from av4_main import FLAGS
from av4_networks import max_net
from av4_utils import generate_exhaustive_affine_transform,deep_affine_transform,affine_transform


FLAGS.saved_session = './summaries/2_netstate/saved_state-999'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/unlabeled_av4'
FLAGS.num_epochs = 10
FLAGS.top_k = FLAGS.num_epochs


# av4 lanscape visualizer makes visualizaition of the predicted energy landscape in VMD

# take a single ligand;
# generate affine transform
# enqueue all of the ligands into a queue without shuffling (I don't think I need a queue)
# take the ligands ; collect the energy terms


class minimizer_object:

    def __init__(self,side_pixels=FLAGS.side_pixels,pixel_size=FLAGS.pixel_size,batch_size=FLAGS.batch_size,num_threads=FLAGS.num_threads):
        self.affine_transform_queue = tf.FIFOQueue(capacity=10000,dtypes=tf.float32,shapes=[4,4])
        self.exhaustive_affine_transform = generate_exhaustive_affine_transform()
        self.ligand_elements = tf.Variable([0,0],trainable=False)
        self.ligand_coords = tf.Variable([[0.0,0.0,0.0],[0.0,0.0,0.0]],trainable=False)
        self.receptor_elements = tf.Variable([0,0,0],trainable=False)
        self.receptor_coords = tf.Variable([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],trainable=False)

        self.transformed_ligand_coords,_ = affine_transform(self.ligand_coords,self.affine_transform_queue.dequeue())

        self.complex_image,_,_ = convert_protein_and_ligand_to_image(self.ligand_elements,
                                                                     self.transformed_ligand_coords,
                                                                     self.receptor_elements, self.receptor_coords,
                                                                     side_pixels, pixel_size)

        self.image_batch = tf.train.batch([self.complex_image], batch_size=batch_size,
                                           num_threads=num_threads, capacity=batch_size * 5,
                                           shapes=[[side_pixels, side_pixels, side_pixels]])

        self.keep_prob = tf.placeholder(tf.float32)
        y_conv = max_net(self.image_batch, self.keep_prob, batch_size)

        # compute softmax over raw predictions
        self.predictions_batch = tf.nn.softmax(y_conv)[:,1]



    def evaluate_all_positions(self,sess,my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords):

        sess.run(tf.global_variables_initializer())
        sess.run(self.affine_transform_queue.enqueue_many(self.exhaustive_affine_transform))

        tf.assign(self.ligand_elements,my_ligand_elements,validate_shape=False)
        tf.assign(self.ligand_coords,my_ligand_coords,validate_shape=False)
        tf.assign(self.receptor_elements,my_receptor_elements,validate_shape=False)
        tf.assign(self.receptor_coords,my_receptor_coords,validate_shape=False)

        try:
            while True:
                start = time.time()
                my_predictions = sess.run(self.predictions_batch,feed_dict={self.keep_prob :1},options=tf.RunOptions(timeout_in_ms=1000))
                print "\tprediction averages:", np.mean(my_predictions),
                print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        except tf.errors.DeadlineExceededError:
            print "evaluation finished"




#print "right before minimizer"
#time.sleep(1)
minimizer1 = minimizer_object()
#print "right after minimizer "
#time.sleep(1)

def evaluate_on_train_set():
    "train a network"



    with tf.name_scope("epoch_counter"):
        batch_counter = tf.Variable(0)
        batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((100000*FLAGS.num_epochs)/FLAGS.batch_size)))
        epoch_counter = tf.div(batch_counter*FLAGS.batch_size,1000000)



    # create session to compute evaluation
    sess = tf.Session()

    # create a filename queue first
    filename_queue,examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    # there is an additional step with variable initialization in order to get the name of "count up to" in the graph
    batch_counter = tf.Variable(0,trainable=False)
    # sess.run(tf.global_variables_initializer())

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,epoch_counter=tf.constant(0))


    # create saver to save and load the network state

    #print "right before saver"
    #time.sleep(1)
    sess.run(tf.global_variables_initializer())
    saver= tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))

    #print "right after saver"
    #time.sleep(1)

    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess,FLAGS.saved_session)
        print "unitialized vars:", sess.run(tf.report_uninitialized_variables())
        #time.sleep(1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)

    batch_num = 0

    #print "before the while loop"
    #time.sleep(1)
    while True or not coord.should_stop():
        start = time.time()

        print sess.run(ligand_file)

        my_ligand_file,my_current_epoch,my_label,my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords = \
            sess.run([ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords])
        minimizer1.evaluate_all_positions(sess,my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords)

        print "batch_num:",batch_num
        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))


evaluate_on_train_set()
print "All Done"