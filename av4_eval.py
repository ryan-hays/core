import time, os
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from av4_eval_input import image_and_label_shuffle_queue,index_the_database_into_queue
from av4 import FLAGS, max_net
from collections import defaultdict
FLAGS.test_set_path = '/home/ubuntu/common/data/new_kaggle/test/unlabeled_av4'
FLAGS.saved_session = './summaries/3_netstate/saved_state-1199'
FLAGS.top_k=10
FLAGS.predictions_file_path = re.sub("netstate", "logs", FLAGS.saved_session)


class store_predictions:
    '''
    store add of the prediction results
    :return:
    '''

    raw_predictions = defaultdict(list)
    processed_predictions = defaultdict(list)

    def add_batch(self, batch_in_the_range, ligand_file_paths, batch_predictions):
        ligand_file_name = map(lambda filename:os.path.basename(filename).split('.')[0],ligand_file_paths)

        for in_the_range,ligand,prediction in zip(batch_in_the_range, ligand_file_name, batch_predictions):
            if in_the_range:
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
                records.append( [key]+value )


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
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=False)

    # create an epoch counter
    batch_counter = tf.Variable(0)
    batch_counter_increment = tf.assign(batch_counter, tf.Variable(0).count_up_to(
        np.round((examples_in_database * FLAGS.num_epochs) / FLAGS.batch_size)))
    epoch_counter = tf.div(batch_counter * FLAGS.batch_size, examples_in_database)

    current_epoch, batch_ligand_filename,batch_in_the_range, y_, x_image_batch = image_and_label_shuffle_queue(batch_size=FLAGS.batch_size,
                                                                     pixel_size=FLAGS.pixel_size,
                                                                     side_pixels=FLAGS.side_pixels,
                                                                     num_threads=FLAGS.num_threads,
                                                                     filename_queue=filename_queue,
                                                                     epoch_counter=epoch_counter,
                                                                     evaluation=True)

    float_image_batch = tf.cast(x_image_batch, tf.float32)
    batch_size = tf.shape(x_image_batch)

    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(float_image_batch, keep_prob)

    # compute softmax over raw predictions
    predictions = tf.nn.softmax(y_conv)[:, 1]
    # restore variables from sleep
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.saved_session)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # create a variable to store all predicitons
    all_predictios = store_predictions()
    counter = 0
    print "start eval..."
    while True or not coord.should_stop():
        counter +=1
        batch_num = sess.run(batch_counter_increment)
        test_current_epoch,test_ligand,test_in_the_range ,test_predictions = sess.run([current_epoch,batch_ligand_filename,batch_in_the_range ,predictions],
                                                              feed_dict={keep_prob: 1})
        all_predictios.add_batch(test_in_the_range,test_ligand, test_predictions)

        print "counter ",counter
        print "batch num", batch_num,
        print "current epoch"
        print test_current_epoch

        if min(test_current_epoch)>FLAGS.top_k:
            break;

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=5)


    all_predictios.save()


evaluate_on_train_set()
print "All Done"

