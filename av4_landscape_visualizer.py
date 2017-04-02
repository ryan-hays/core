import time,re
import tensorflow as tf
import numpy as np
from av4_input import image_and_label_queue,index_the_database_into_queue,read_receptor_and_ligand,convert_protein_and_ligand_to_image
from av4_main import FLAGS
from av4_networks import max_net
from av4_utils import generate_exhaustive_affine_transform,deep_affine_transform


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
    affine_transform_queue = tf.FIFOQueue(capacity=10000,dtypes=tf.float32,shapes=[4,4])
    exhaustive_affine_transform = generate_exhaustive_affine_transform()
    ligand_elements_var = tf.Variable([0,0,0],trainable=False)
    ligand_coords_var = tf.Variable([[0,0,0],[0,0,0]],trainable=False,name="dd1")
    receptor_elements_var = tf.Variable([0,0,0],trainable=False,name="dd2")
    receptor_coords_var = tf.Variable([[0,0,0],[0,0,0]],trainable=False)


    def find_minima(self,sess):
        sess.run(self.affine_transform_queue.enqueue_many(self.exhaustive_affine_transform))

        try:
            while True:
                print sess.run(self.affine_transform_queue.dequeue_many(100),options=tf.RunOptions(timeout_in_ms=1000))
        except tf.errors.DeadlineExceededError:
            print "evaluation finished"


minimizer1 = minimizer_object()



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
    #sess.run(tf.global_variables_initializer())

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,epoch_counter=tf.constant(0))

    #ligand_file = tf.convert_to_tensor(ligand_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)

    #print sess.run(tf.report_uninitialized_variables())
    #time.sleep(100)

    print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


    # create saver to save and load the network state

    saver= tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))
    #saver = tf.train.Saver()
    #var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    #sess.run(tf.global_variables_initializer())
    #print "tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):",tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #print "variables_to_restore()", sess.run(tf.contrib.framework.get_variables_to_restore(include=[tf.GraphKeys.TRAINABLE_VARIABLES]))
    #time.sleep(1000)

    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess,FLAGS.saved_session)
        # print tf.report_uninitialized_variables()
        # time.sleep(100)

    batch_num = 0

    while True or not coord.should_stop():
        start = time.time()

        print sess.run(ligand_file)

        #minimizer1.find_minima(sess=sess)

        print "batch_num:",batch_num
        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))


evaluate_on_train_set()
print "All Done"