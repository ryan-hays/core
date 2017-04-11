import time,re
import tensorflow as tf
import numpy as np
from av4_input import image_and_label_queue,index_the_database_into_queue,read_receptor_and_ligand,convert_protein_and_ligand_to_image
from av4_main import FLAGS
from av4_networks import max_net
from av4_utils import generate_exhaustive_affine_transform,deep_affine_transform,affine_transform,generate_identity_matrices


FLAGS.saved_session = './summaries/4_netstate/saved_state-82099'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/holdout_av4'
FLAGS.num_epochs = 2
FLAGS.top_k = FLAGS.num_epochs


# av4 lanscape visualizer makes visualizaition of the predicted energy landscape in VMD

# take a single ligand;
# generate affine transform
# enqueue all of the ligands into a queue without shuffling
# take the ligands ; collect the energy terms


class minima_search_agent:

    class evaluations_container:
        """" groups together information about the evaluated positions in a memory-efficient way.
        ligand pose transformations stores (groups of) affine transformation matrices """
        labels = np.array([])
        predictions = np.array([])
        ligand_pose_transformations = np.array([]).reshape([0,4,4])
        cameraviews = np.array([]).reshape([0,4,4])

        def add_batch(self,prediction_batch,ligand_pose_transformation_batch,cameraview_batch,label_batch=[]):
            "adds batch of predictions for different positions and cameraviews of the ligand"

            self.labels = np.append(self.labels,label_batch, axis=0)
            self.predictions = np.append(self.predictions,prediction_batch, axis=0)
            self.ligand_pose_transformations = np.append(self.ligand_pose_transformations,ligand_pose_transformation_batch,axis=0)
            self.cameraviews = np.append(self.cameraviews,cameraview_batch, axis=0)

            return len(self.predictions)


#        # example 1
#        # take all (do not remove middle) dig a hole in the landscape
#        # batch is a training example
        def convert_into_training_batches(self,positives_in_batch=10,negatives_in_batch=90,num_batches=10000):
            """takes predictions for which the label * prediction is the highest
            (label * prediction is equal to the cost to the network)
            constructs num_batches of bathes with a single positive example
            returns a list of batches
            """
            # TODO: variance option for positives and negatives
            # TODO: clustering
            # TODO: add VDW possibilities
            # TODO: RMSD + the fraction of native contacts

            # generate labels first
            # in this case positives labels are simply the examples with identity affine transform
            # everything that is not the initial conformation is called negatives
            self.labels = np.asarray(map(lambda affine_transform: np.all(affine_transform == np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),self.ligand_pose_transformations))


            # sort in descending order by predictions
            order = np.argsort(self.predictions)
            print "order:",order
            print "labels:",self.labels
            self.labels = self.labels[order]
            print "reordered labels:",self.labels

            self.predictions = self.predictions[order]
            self.ligand_pose_transformations = self.ligand_pose_transformations[order]
            self.cameraviews = self.cameraviews[order]

            # take hardest positive and negative labels;
            positives_idx = (np.arange(len(self.labels))[self.labels])[-positives_in_batch:]
            negatives_idx = (np.arange(len(self.labels))[-self.labels])[:negatives_in_batch]

            print "positives idx:", positives_idx
            print "negatives idx:", negatives_idx
            print "labels:", self.labels

            print "all positives:", np.arange(len(self.labels))[self.labels], len(np.arange(len(self.labels))[self.labels])
            print "all negatives:", np.arange(len(self.labels))[-self.labels], len(np.arange(len(self.labels))[-self.labels])


            #batch_positives_idx = positives_idx[-(1+np.arange(positives_in_batch)+ i*positives_in_batch)]
            #batch_negatives_idx = negatives_idx[np.arange(negatives_in_batch)+i*negatives_in_batch]
            #batch_idx = np.concatenate([batch_positives_idx,batch_negatives_idx],axis=0)

            #print "positives:",positives_idx,len(positives_idx)
            #print "negatives:",negatives_idx,len(negatives_idx)
            #print "batch positives:", batch_positives_idx
            #print "batch negatives:", batch_negatives_idx








            #print "predictions for correct pose:", self.predictions[identity_idx],

            #print "min:",np.min(self.predictions[identity_idx]),"max:",np.max(self.predictions[identity_idx])

            #print "correct pose rank:", np.sum(np.asarray((self.predictions[-identity_idx] > np.average(self.predictions[identity_idx])),dtype=np.int32))

















#            # TODO(maksym): it can be an interesting idea to cluster the results for training

#        # example 2
#        # remove semi-correct conformations
#        # batch is a training example
#        def convert_into_training_batches(self,batch_size,num_batches=2):
#            return 999

#       # example 3
#       # remove semi-correct conformations
#       # take many images of positive
#       # image is a training example




    def __init__(self,side_pixels=FLAGS.side_pixels,pixel_size=FLAGS.pixel_size,batch_size=FLAGS.batch_size,num_threads=FLAGS.num_threads,sess = FLAGS.main_session):

        # generate a single image from protein coordinates, and ligand coordinates + transition matrix
        self.sess = sess
        self.ligand_pose_transformations_queue = tf.FIFOQueue(capacity=80000,dtypes=tf.float32,shapes=[4,4])
        self.ligand_pose_transformations = tf.concat(0,[generate_identity_matrices(num_frames=1000),generate_exhaustive_affine_transform()])
        self.ligand_elements = tf.Variable([0],trainable=False,validate_shape=False)
        self.ligand_coords = tf.Variable([[0.0,0.0,0.0]],trainable=False,validate_shape=False)
        self.receptor_elements = tf.Variable([0],trainable=False,validate_shape=False)
        self.receptor_coords = tf.Variable([[0.0,0.0,0.0]],trainable=False,validate_shape=False)

        transformed_ligand_coords,ligand_pose_transformation = affine_transform(self.ligand_coords,self.ligand_pose_transformations_queue.dequeue())

        # TODO: create fast reject for overlapping atoms of the protein and ligand
        complex_image,_,cameraview = convert_protein_and_ligand_to_image(self.ligand_elements,
                                                                     transformed_ligand_coords,
                                                                     self.receptor_elements, self.receptor_coords,
                                                                     side_pixels, pixel_size)

        # create and enqueue images in many threads, and deque and score images in a main thread
        self.coord = tf.train.Coordinator()
        self.minimizer_queue = tf.FIFOQueue(capacity=batch_size*5,dtypes=[tf.float32,tf.float32,tf.float32],shapes=[[side_pixels,side_pixels,side_pixels],[4,4],[4,4]])
        self.minimizer_queue_enqueue = self.minimizer_queue.enqueue([complex_image,ligand_pose_transformation,cameraview])
        self.queue_runner = tf.train.QueueRunner(self.minimizer_queue,[self.minimizer_queue_enqueue]*num_threads)
        self.image_batch,self.ligand_pose_transformation_batch,self.cameraview_batch = self.minimizer_queue.dequeue_many(batch_size)
        self.keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope("network"):
            y_conv = max_net(self.image_batch, self.keep_prob, batch_size)
        self.predictions_batch = tf.nn.softmax(y_conv)[:,1]

        # create a pipeline to convert ligand transition matrices + cameraviews into images again


    def evaluate_all_positions(self,my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords):

        self.sess.run(self.ligand_pose_transformations_queue.enqueue_many(self.ligand_pose_transformations))
        print "enqueue affine transform:", time.sleep(1)
        self.enqueue_threads = self.queue_runner.create_threads(self.sess, coord=self.coord, start=False, daemon=True)
        print "launch threads:",time.sleep(1)


        self.sess.run(tf.assign(self.ligand_elements,my_ligand_elements,validate_shape=False))
        self.sess.run(tf.assign(self.ligand_coords,my_ligand_coords,validate_shape=False))
        self.sess.run(tf.assign(self.receptor_elements,my_receptor_elements,validate_shape=False))
        self.sess.run(tf.assign(self.receptor_coords,my_receptor_coords,validate_shape=False))

        # re-initialize the evalutions class
        self.evaluated = self.evaluations_container()


        print "print queue size -3:", self.sess.run(self.minimizer_queue.size())
        for tr in self.enqueue_threads:
            print tf
            tr.start()
        print "print queue size -2:", self.sess.run(self.minimizer_queue.size())
        time.sleep(0.5)
        print "print queue size -1:", self.sess.run(self.minimizer_queue.size())

        print "start evaluations...."
        time.sleep(0.5)

        try:
            while True:
                start = time.time()
                my_prediction_batch,my_image_batch,my_ligand_pose_transformation_batch,my_cameraview_batch = \
                    self.sess.run([self.predictions_batch,self.image_batch,self.ligand_pose_transformation_batch,self.cameraview_batch],
                                  feed_dict = {self.keep_prob:1},options=tf.RunOptions(timeout_in_ms=1000))

                # save the predictions and cameraviews into the memory
                positions_evaluated = self.evaluated.add_batch(my_prediction_batch,my_ligand_pose_transformation_batch,my_cameraview_batch)

                print "ligand_atoms:",np.sum(np.array(my_image_batch >7,dtype=np.int32)),
                print "\tpositions evaluated:",positions_evaluated,
                print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        except tf.errors.DeadlineExceededError:

            # create training examples for the main queue
            self.evaluated.convert_into_training_batches(positives_in_batch=20,negatives_in_batch=200,num_batches=5)

            # empty the queue and continue
            try:
                while True:
                    print self.sess.run(self.minimizer_queue.dequeue(),options=tf.RunOptions(timeout_in_ms=1000))
            except tf.errors.DeadlineExceededError:
                print "evaluations finished; moving to the next protein"

        return 0



















minimizer1 = minima_search_agent()


def evaluate_on_train_set():
    "train a network"


    with tf.name_scope("epoch_counter"):
        batch_counter = tf.Variable(0)
        batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((100000*FLAGS.num_epochs)/FLAGS.batch_size)))
        epoch_counter = tf.div(batch_counter*FLAGS.batch_size,1000000)


    # create session to compute evaluation
    sess = FLAGS.main_session

    # create a filename queue first
    filename_queue,examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    # there is an additional step with variable initialization in order to get the name of "count up to" in the graph
    batch_counter = tf.Variable(0,trainable=False)

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,epoch_counter=tf.constant(0))

    # create saver to save and load the network state

    sess.run(tf.global_variables_initializer())
    saver= tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network") +
                             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))


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
        minimizer1.evaluate_all_positions(my_ligand_elements,my_ligand_coords,my_receptor_elements,my_receptor_coords)

        # minimizer should return many images
        # for simplicity of visualization of what is going on, the images will be in python (and for error notifications)
        # images can be enqueud into the main queue



        print "batch_num:",batch_num
        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))


evaluate_on_train_set()
print "All Done"










            #self.sess.run(self.minimizer_queue.close(cancel_pending_enqueues=True))
            #self.coord._stop_event.set()
            #self.coord.clear_stop()

#            for tr in self.enqueue_threads:
#                print tf
#                tr.stop()

#            self.coord.clear_stop()
#            print "stop has been cleared"
#            self.sess.run(self.affine_transform_queue.enqueue_many(self.exhaustive_affine_transform))


#            self.minimizer_queue_enqueue = self.minimizer_queue.enqueue(self.complex_image)
#            self.queue_runner = tf.train.QueueRunner(self.minimizer_queue, [self.minimizer_queue_enqueue] * 8)
#
 #           self.enqueue_threads = self.queue_runner.create_threads(self.sess, coord=self.coord, start=False,
 #                                                                   daemon=True)
#
#
#            print "print queue size 1:", self.sess.run(self.minimizer_queue.size())
#            print "starting enqueue threads again"
 #           for tr in self.enqueue_threads:
 #               print tf
 #              tr.start()
  #          print "enqueue threads have started again"
   #         print "queue size 2:", self.sess.run(self.minimizer_queue.size())
   #         time.sleep(5)
   #         print "queue size 3", self.sess.run(self.minimizer_queue.size())
   #         time.sleep(2)

#            #self.sess.run(self.image_batch, options=tf.RunOptions(timeout_in_ms=1000))
#            time.sleep(1)
#            self.coord.join(threads=self.enqueue_threads,stop_grace_period_secs=5)
#
#            # Clearning stop !
#
#
#            print "joined threads:", time.sleep(1)
#            self.sess.run(self.image_batch, options=tf.RunOptions(timeout_in_ms=1000))
 #           print "one more image batch done!",time.sleep(1)
#
#            print "evaluation finished"
#            return 0