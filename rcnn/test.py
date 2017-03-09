import time,os                                                                                                                                                                     
import tensorflow as tf                                                                                                                                                            
import numpy as np                                                                                                                                                                 
from av4_input import index_the_database_into_queue,image_and_label_queue   
from av4_main import FLAGS

def train():                                                                                                                                                                       
    "train a network"                                                                                                                                                              
    # it's better if all of the computations use a single session                                                                                                                  
    sess = tf.Session()                                                                                                                                                            

    # create a filename queue first                                                                                                                                                
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)                                                                        

    # create an epoch counter                                                                                                                                                      
    batch_counter = tf.Variable(0)                                                                                                                                                  
    batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))                              
    epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)                                                                                                    
 
    # create a custom shuffle queue                                                                                                                                                 
    _,current_epoch,label_batch,sparse_image_batch,_,_ = image_and_label_queue(batch_size=FLAGS.batch_size, pixel_size=FLAGS.pixel_size,                                            
                                                                          side_pixels=FLAGS.side_pixels, num_threads=FLAGS.num_threads,                                             
                                                                          filename_queue=filename_queue, epoch_counter=epoch_counter)                                               
    
    print sparse_image_batch

    image_batch = tf.sparse_tensor_to_dense(sparse_image_batch,validate_indices=False)                                                                                              

train()
