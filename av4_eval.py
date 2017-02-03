import time,os
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from av4_input import image_and_label_queue,index_the_database_into_queue
from av4_main import FLAGS
from av4_networks import intuit_net
from collections import defaultdict


FLAGS.saved_session = './summaries/36_netstate/saved_state-23999'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/unlabeled_av4'
FLAGS.num_epochs = 25
FLAGS.top_k = FLAGS.num_epochs

class store_predictions:
    '''
    store add of the prediction results
    :return:
    '''

    raw_predictions = defaultdict(list)
    processed_predictions = defaultdict(list)

    def add_batch(self, ligand_file_paths, batch_current_epoch, batch_predictions):
        ligand_file_name = map(lambda filename:os.path.basename(filename).split('.')[0],ligand_file_paths)

        for ligand,current_epoch,prediction in zip(ligand_file_name, batch_current_epoch, batch_predictions):
            #   if in_the_range:
            self.raw_predictions[ligand].append(prediction)

    def reduce(self):
        '''
        if a ligand has more than one predictions
        use mean as final predictions
        '''
        for key, value in self.raw_predictions.items():

            if len(value) > 1:
                predictions_size = map(lambda x: len(x), value)
                if len(set(predictions_size)) > 1:
                    raise Exception(key, " has different number of predictions ", set(predictions_size))
                self.processed_predictions[key].append(np.mean(value, axis=0))
            else:
                self.processed_predictions[key].append(value)

    def final_predictions(self, predictions_list):
        length = min(len(predictions_list, 10))
        return np.mean(predictions_list[:length])

    def fill_na(self):
        for key in self.raw_predictions.keys():
            value_len = len(self.raw_predictions[key])
            if value_len>FLAGS.top_k:
                print "{} have more predictions than expected, {} reuqired {} found.".format(key,FLAGS.top_k,value_len)
            else:
                for i in range(FLAGS.top_k-value_len):
                    self.raw_predictions[key]

    def save_multiframe_predictions(self):
        records = []
        for key, value in self.raw_predictions.items():
            value_len = len(self.raw_predictions[key])
            if value_len>FLAGS.top_k:
                print "{} have more predictions than expected, {} reuqired {} found.".format(key,FLAGS.top_k,value_len)
                records.append([key]+value[:FLAGS.top_k])
            else:
                records.append([key]+value)

        submission_csv = pd.DataFrame(records, columns=['Id']+[ 'Predicted_%d'%i for i in range(1,len(records[0]))])
        submission_csv.to_csv(FLAGS.predictions_file_path + '_multiframe_submission.csv', index=False)

    def save_average(self):
        '''
        take average of multiple predcition
        :return:
        '''
        records = []
        for key,value in self.raw_predictions.items():
            records.append([key,np.mean(np.array(value))])

        submission_csv = pd.DataFrame(records,columns=['ID','Predicted'])
        submission_csv.to_csv(FLAGS.predictions_file_path+'_average_submission.csv',index=False)

    def save_max(self):

        records = []
        for key,value in self.raw_predictions.items():
            records.append([key, np.max(np.array(value))])

        submission_csv = pd.DataFrame(records, columns=['ID', 'Predicted'])
        submission_csv.to_csv(FLAGS.predictions_file_path + '_max_submission.csv', index=False)

    def save(self):
        self.save_average()
        self.save_max()
        self.save_multiframe_predictions()


def evaluate_on_train_set():
    "train a network"

    # create session which all the evaluation happens in
    sess = tf.Session()

    # create a filename queue first
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    # there is an additional step with variable initialization in order to get the name of "count up to" in the graph
    batch_counter = tf.Variable(0)
    sess.run(tf.global_variables_initializer())
    batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))
    batch_counter_var_name = sess.run(tf.report_uninitialized_variables())

    epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)

    # create a custom shuffle queue
    ligand_file,current_epoch,label_batch,sparse_image_batch = image_and_label_queue(batch_size=FLAGS.batch_size, pixel_size=FLAGS.pixel_size,
                                                                          side_pixels=FLAGS.side_pixels, num_threads=FLAGS.num_threads,
                                                                          filename_queue=filename_queue, epoch_counter=epoch_counter)

    image_batch = tf.sparse_tensor_to_dense(sparse_image_batch,validate_indices=False)

    keep_prob = tf.placeholder(tf.float32)
    y_conv = intuit_net(image_batch,keep_prob,FLAGS.batch_size)

    # compute softmax over raw predictions
    predictions = tf.nn.softmax(y_conv)[:,1]

    # restore variables from sleep
    saver = tf.train.Saver()
    saver.restore(sess,FLAGS.saved_session)

    # use
    sess.run(tf.contrib.framework.get_variables_by_name(batch_counter_var_name[0])[0].initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)

    # create an instance of a class to store predictions
    all_predictios = store_predictions()
    print "starting evalution..."

    try:
        while True or not coord.should_stop():
            start = time.time()
            batch_num = sess.run([batch_counter_increment])
            test_ligand, test_epoch, test_predictions = sess.run([ligand_file, current_epoch, predictions],
                                                                 feed_dict={keep_prob: 1})
            print "current_epoch:", test_epoch[0], "batch_num:", batch_num,
            print "\tprediction averages:",np.mean(test_predictions),
            print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))
            all_predictios.add_batch(test_ligand, test_epoch, test_predictions)

    except Exception:
        print "exiting the loop"

    all_predictios.save()

evaluate_on_train_set()
print "All Done"
