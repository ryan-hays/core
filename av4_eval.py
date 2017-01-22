import time, os
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from av4_eval_input import image_and_label_queue,index_the_database,read_receptor_and_ligand
from av4 import FLAGS, max_net
from collections import defaultdict

FLAGS.saved_session = '/home/ubuntu/xiao/code/core/summaries/5_netstate/saved_state-2299'

FLAGS.predictions_file_path = re.sub("netstate", "logs", FLAGS.saved_session)



class store_predictions:
    '''
    store add of the prediction results
    :return:
    '''

    raw_predictions = defaultdict(list)
    processed_predictions = defaultdict(list)

    def add_batch(self, in_the_range,ligand_file_paths, frame_ids , batch_predictions):
        ligand_file_name = map(lambda filename:os.path.basename(filename).split('.')[0],ligand_file_paths)

        for mark,ligand,frame_id,prediction in zip(in_the_range,ligand_file_name,frame_ids,batch_predictions):
            if mark:
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


    def save_predictions(self):
        records = []
        for key, value in self.raw_predictions.items():
            records.append([key]+ value)

        submission_csv = pd.DataFrame(records, columns=['Id']+[ 'Predicted_%d'%i for i in range(1,len(records[0]))])
        submission_csv.to_csv(FLAGS.predictions_file_path + '_submission.csv', index=False)

    def save(self):
        self.fill_na()
        self.save_predictions()

def evaluate_on_train_set():
    sess = tf.Session()

    current_epoch,filename_batch,in_the_ranges ,label_batch, image_batch = image_and_label_queue(sess=sess, batch_size=FLAGS.batch_size,
                                                                    pixel_size=FLAGS.pixel_size,
                                                                    side_pixels=FLAGS.side_pixels,
                                                                    num_threads=FLAGS.num_threads,
                                                                    database_path=FLAGS.test_set_path,
                                                                    num_epochs=FLAGS.num_epochs)

    float_image_batch = tf.cast(image_batch,tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    predicted_labels = max_net(float_image_batch,keep_prob)

    positive_labels = tf.nn.softmax(predicted_labels)[:,1]

    batch_size = tf.shape(label_batch)

    saver = tf.train.Saver()
    saver.restore(sess,FLAGS.saved_session)

    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_local_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # create a variable to store all predicitons
    all_predictions = store_predictions()
    batch_num = 0
    print "start eval..."

    while not coord.should_stop():
        batch_shape, mark,test_ligands, test_frame_ids, test_predictions = sess.run(
            [batch_size, in_the_ranges,filename_batch, current_epoch, positive_labels], feed_dict={keep_prob: 1})
        all_predictions.add_batch(mark,test_ligands, test_frame_ids, test_predictions)
        batch_num += 1
        print "batch num", batch_num,
        print "\tbatch size", batch_shape[0]
        print test_ligands[0], test_frame_ids[0], test_predictions[0]
        print test_frame_ids
        print mark
        print "min epoch",min(test_frame_ids)
        if min(test_frame_ids)>FLAGS.top_k:
            break;

    print "+++++++++++++++++++++++++++++++++++over++++++++++++++++++++++++"
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=5)

    all_predictions.save()


evaluate_on_train_set()
print "All Done"


