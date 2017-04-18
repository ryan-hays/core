import time,re
import tensorflow as tf
import numpy as np
import av4_input
from av4_main import FLAGS
import av4_conformation_sampler

minimizer1 = av4_conformation_sampler.SearchAgent()


def evaluate_on_train_set():
    "train a network"

    with tf.name_scope("epoch_counter"):
        batch_counter = tf.Variable(0)
        batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((100000*FLAGS.num_epochs)/FLAGS.batch_size)))
        epoch_counter = tf.div(batch_counter*FLAGS.batch_size,1000000)


    # create session to compute evaluation
    sess = FLAGS.main_session

    # create a filename queue first
    filename_queue,examples_in_database = av4_input.index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    # there is an additional step with variable initialization in order to get the name of "count up to" in the graph
    batch_counter = tf.Variable(0,trainable=False)

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = av4_input.read_receptor_and_ligand(filename_queue,epoch_counter=tf.constant(0))

    # create saver to save and load the network state

    sess.run(tf.global_variables_initializer())
    saver= tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer")
                                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")
                                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))


    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess,FLAGS.saved_session)
        print "unitialized vars:", sess.run(tf.report_uninitialized_variables())
        time.sleep(1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)

    batch_num = 0


    while True or not coord.should_stop():
        start = time.time()

        print "print taking next ligand:", sess.run(ligand_file)
        my_ligand_file,my_current_epoch,my_label,my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords = \
            sess.run([ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords])

        print "obtained coordinates trying to minimize the ligand "
        print "ligand elements:", len(my_ligand_elements)
        print "ligand coords", len(my_ligand_coords[:,0])
        print "receptor element:", len(my_receptor_elements)
        print "receptor coords:", len(my_receptor_coords[:,0])
        minimizer1.grid_evaluate_positions(my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords)
        #minimizer1.grid_evaluate_positions(np.array([0]),np.array([[0,0,0]]), my_receptor_elements,
        #                                   my_receptor_coords)


        # minimizer should return many images
        # for simplicity of visualization of what is going on, the images will be in python (and for error notifications)
        # images can be enqueud into the main queue



        print "batch_num:",batch_num
        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))


evaluate_on_train_set()
print "All Done"
