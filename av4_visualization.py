import tensorflow as tf
import numpy as np
import time,os
from av4_input import single_dense_image_example
from av4_atomdict import atom_dictionary
from av4_utils import affine_transform

def split_dense_image(dense_image):
    '''With input of a numpy array, splits ligand and receptor coordinates
    and names.'''
    inv_ATM = {v: k for k, v in atom_dictionary.ATM.iteritems()}
    ligand_names,ligand_coords = [],[]
    receptor_names,receptor_coords = [],[]
    for (x,y,z),atom_value in np.ndenumerate(dense_image):
        if atom_value > 0:
            if atom_value > 10:
                atom_value -= 10
                ligand_names.append(inv_ATM[atom_value])
                ligand_coords.append([float(x),float(y),float(z)])
            else:
                receptor_names.append(inv_ATM[atom_value])
                receptor_coords.append([float(x),float(y),float(z)])
    return ligand_names,ligand_coords,receptor_names,receptor_coords

def convert_to_vmd(filename,r_coords,l_coords):
    '''Converts a list of ligand and receptor coordinates to a tcl script
    that can be run in VMD. Ligand will appear in red, receptor in blue.'''
    f = open(filename+'.tcl','w')
    # write receptor coordinates
    for position in r_coords:
        x,y,z = str(position[0])+' ',str(position[1])+' ',str(position[2])
        f.write('draw sphere {'+x+y+z+'} radius 0.3 resolution 100\n')
    
    # switch color
    f.write('draw color red\n')

    # write ligand coordinates
    for position in l_coords:
        x,y,z = str(position[0])+' ',str(position[1])+' ',str(position[2])                                                                                              
        f.write('draw sphere {'+x+y+z+'} radius 0.3 resolution 100\n')    
    
    f.close()

def visualize_complex():
    '''writes a VMD output file so you can see the complex'''

    # with the current setup all of the TF's operations are happening in one session
    sess = tf.Session()

    current_filename,ligand_com,final_transition_matrix,current_idx,label,dense_image = single_dense_image_example(sess=sess,batch_size=FLAGS.batch_size,
                                                pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                num_threads=FLAGS.num_threads,database_path=FLAGS.database_path,
                                                num_epochs=FLAGS.num_epochs)
    sess.run(tf.global_variables_initializer())

    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_num = 0
    
    my_filename,my_idx,my_dense_image,my_final_matrix,my_com = sess.run([current_filename,current_idx,dense_image,final_transition_matrix,ligand_com])
    print "filename",my_filename
    print "idx:",my_idx
    print "affine_matrix:",my_final_matrix
    print "center of mass", my_com

    # given a filename, extract the 4-letter PDB code (makes file naming easier later)
    def get_protein_id(filename):
        underscores = [idx for idx, letter in enumerate(my_filename) if letter == '_']
        underscore_after_id = underscores[-2]
        protein_id = my_filename[underscore_after_id-4:underscore_after_id]
        return protein_id
    my_protein_id = get_protein_id(my_filename)

    # invert the affine transformation matrix
    inverted_matrix = np.linalg.inv(my_final_matrix).astype('float32')

    _,l_coords,_,r_coords = split_dense_image(my_dense_image)
    
    # apply the inverse affine transform
    l_coords_,_ = affine_transform(l_coords,inverted_matrix)
    r_coords_,_ = affine_transform(r_coords,inverted_matrix)
    l_coords_inv = sess.run(l_coords_)
    r_coords_inv = sess.run(r_coords_)

    # invert the translation by ligand's center of mass (performed in av4_input)
    for (i,j), coordinate in np.ndenumerate(l_coords):
        l_coords[i][j] = (l_coords[i][j] - (FLAGS.side_pixels * 0.5) + 0.5) + my_com[j] 
    for (i,j), coordinate in np.ndenumerate(r_coords):
        r_coords[i][j] = (r_coords[i][j] - (FLAGS.side_pixels * 0.5) + 0.5) + my_com[j]

    convert_to_vmd('./complexes/'+my_protein_id,r_coords,l_coords)

    time.sleep(5)


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
    batch_size = 1  # number of background processes to fill the queue with images
    num_threads = 1
    # data directories
    # path to the csv file with names of images selected for training
    database_path = "../datasets/labeled_av4/**/"
    # directory where to write variable summaries
    output_dir = './complexes'
    # optional saved session: network from which to load variable states
    saved_session = None


def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start"""
    output_dir = os.path.join(FLAGS.output_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    visualize_complex()

if __name__ == '__main__':
    tf.app.run()
