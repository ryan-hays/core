import numpy as np
import tensorflow as tf
import prody as pd
from av4_atomdict import atom_dictionary
from av3_input import *
from av4 import FLAGS,max_net
import sys

i = sys.argv[1]

inv_ATM = {v: k for k, v in atom_dictionary.ATM.iteritems()}

print inv_ATM

def get_batch():
        f = open('pdbs.txt','a')
	sess = tf.Session()
        sess.run(tf.global_variables_initializer())                                                                                                                                 
        print "! Session started"
        
        a = image_and_label_queue(sess=sess,batch_size=1,
                                                                 pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                                 num_threads=FLAGS.num_threads,database_path=FLAGS.database_path)                                                 
        _,y_,x_image_batch = a[2]
        l_name = a[0]
        r_name = a[1]
        print "! Batch loaded"                                                                                                                                                      
        float_image_batch = tf.cast(x_image_batch,tf.float32) 
        keep_prob = tf.placeholder(tf.float32)
        #print sess.run([l_name,r_name])
        y_conv = max_net(float_image_batch,keep_prob)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        
        with tf.name_scope('train'):
                tf.summary.scalar('weighted cross entropy mean', cross_entropy_mean)
                train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print "! Queue loaded"
        batch = sess.run(float_image_batch)
        #f.write(str(i)+' '+str(sess.run([l_name,r_name]))+'\n')
        f.close()
        #print batch
        print batch[0].shape
        print "! Batch evaluated"
        return batch[0]

def read_array(input_tensor):
	ligand_names,ligand_coords = [],[]
	receptor_names,receptor_coords = [],[]
        a = input_tensor.shape
	x,y,z = a[0],a[1],a[2]
	for i in range(x):
		for j in range(y):
			for k in range(z):
				val = input_tensor[i][j][k]
				if val != 0 and val < 10:
					atom_type = inv_ATM[val]
					pos = [float(i),float(j),float(k)]
					receptor_names.append(atom_type)
					receptor_coords.append(pos)
                                elif val > 10:
                                        val = val-10
                                        atom_type = inv_ATM[val]
                                        pos = [float(i),float(j),float(k)]
                                        ligand_names.append(atom_type)
                                        ligand_coords.append(pos)

	return ligand_names,ligand_coords,receptor_names,receptor_coords

#A,B = read_array(np.arange(8).reshape(2,2,2))

def convert_to_pdb(names,coords,filename):
	structure = pd.AtomGroup('Structure')
	structure.setNames(names)
	structure.setCoords(coords)
        structure.setResnames([filename[:-4] for i in coords]) 
	print [atom for atom in structure.iterAtoms()] 
        pd.writePDB(filename,structure)

def convert_to_vmd(l_coords,r_coords,idx):
        f = open('complex{0}.txt'.format(idx),'w')
        for position in l_coords:
                x,y,z = str(position[0])+' ',str(position[1])+' ',str(position[2])
                f.write('draw sphere {'+x+y+z+'} radius 0.3 resolution 100\n')
        f.write('draw color red\n')
        for position in r_coords:
                x,y,z = str(position[0])+' ',str(position[1])+' ',str(position[2])                                                                                              
                f.write('draw sphere {'+x+y+z+'} radius 0.3 resolution 100\n')    
        f.close()

def test(i):
        batch = get_batch()
        print "Batched"
        l_names,l_coords,r_names,r_coords = read_array(batch)
        print "Read"
        convert_to_vmd(l_coords,r_coords,i)
        print "Converted"
        return True

test(i)
