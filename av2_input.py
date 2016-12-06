import re,random,time,prody,threading
import numpy as np
import tensorflow as tf
from av2_atomdict import atom_dictionary

# todo solve the problem when many atoms are coming into the same box
# todo add from rdkit or pytraj or openeye to support mol files
# todo add logger
# todo update dictionary
# todo add some way to generate PDBs from vectors

# script currently works the following way:
# 1 ) ligand atom coordinates are read into the matrix
# 2 ) ligand atom coordinates are centered on zero
# 3 ) box is rotated ( in reality, it was easier to transform ligand cordinates) until all ligand atoms can fit
# 4 ) protein is read and transformed with the same affine transform
# 5 ) 3D tensor of zeros is generated
# 6 ) Coordinates of every atom are rounded  (ceiled coordinates)
# 7 ) number from atom vocabulary is written as pixel in place of 0



def generate_random_transition_matrix(shift_range=10,rotation_range=np.pi*2):
    """returns a random transition matrix
    rotation range - determines random rotations along any of X,Y,Z axis
    shift_range determines allowed shifts along any of X,Y,Z axis """

    # randomly shift along X,Y,Z
    xyz_shift_matrix = np.matrix([[1 ,0 , 0, random.random()*shift_range],
                                 [0, 1, 0, random.random()*shift_range],
                                 [0, 0, 1, random.random()*shift_range],
                                 [0, 0, 0, 1]])

    ## randomly rotate along X
    rand_x = random.random()*rotation_range
    x_rotate_matrix = np.matrix([[1, 0, 0, 0],
                               [0, np.cos(rand_x), - np.sin(rand_x), 0],
                               [0, np.sin(rand_x), np.cos(rand_x), 0],
                                [0, 0, 0, 1]])

    ## randomly rotate along Y
    rand_y = random.random()*rotation_range
    y_rotate_matrix = np.matrix([[np.cos(rand_y), 0, np.sin(rand_y), 0],
                                [0, 1, 0, 0],
                                [-np.sin(rand_y), 0, np.cos(rand_y), 0],
                                [0, 0 ,0 ,1]])

    ## randomly rotate along Z
    rand_z = random.random()*rotation_range
    z_rotate_matrix = np.matrix([[np.cos(rand_z), -np.sin(rand_z), 0, 0],
                               [np.sin(rand_z), np.cos(rand_z), 0, 0],
                               [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    # combining all matrices into one by multiplication
    # nice property of the matrix
    xyz_rotate_shift_matrix =  xyz_shift_matrix * x_rotate_matrix * y_rotate_matrix * z_rotate_matrix

    return xyz_rotate_shift_matrix

#def affine_transform(coordinate_array,transition_matrix):
#    """applies transition to every point in array"""
#
#    def transform_one_point(one_point,transition_matrix=transition_matrix):
#        return np.sum(transition_matrix * np.matrix(np.append(one_point,1)).transpose(),axis=1)[0:3]
#
#    transformed_coordinates = np.array(map(transform_one_point,coordinate_array[:,0:3])).reshape(-1,3)
#    return transformed_coordinates

def affine_transform(coordinate_array,transition_matrix):
    """applies transition to every point of the array"""
    #print "FLAG1"
    matrix_coord = np.matrix(coordinate_array)    # np.matrix has different defaults compared to np.array
    #print "matrix_coord", matrix_coord
    #print "FLAG2"
    raw_of_ones = np.matrix(np.ones(len(matrix_coord[:,0])))
    #print "raw_of_ones",raw_of_ones
    #print "FLAG3"
    transformed_coord = (transition_matrix * np.vstack((matrix_coord.transpose(),raw_of_ones)))[0:3,:].transpose()
    return np.array(transformed_coord)



def generate_image(ligand_filename,receptor_filename,pixel_size=1,side_pixels=20,num_attempts=1000):
    """generates 3d tensor from file"""

    def if_in_the_box(atom_coord,box_edge = pixel_size * side_pixels):
        "checks if the atom will fall into the cubic box of size box_edge around [0 0 0]"

        if (-0.5*box_edge < atom_coord[0] < 0.5*box_edge) \
                and (-0.5*box_edge < atom_coord[1] < 0.5*box_edge) \
                and (-0.5*box_edge < atom_coord[2] < 0.5*box_edge):
            return True
        else:
            return False

    prody_ligand = prody.parsePDB(ligand_filename)
    # center the coordinates of the ligand around [ 0 0 0]

    ligand_center_of_mass = prody.calcCenter(prody_ligand)
    prody_ligand.setCoords(prody_ligand.getCoords() - ligand_center_of_mass)

    # find transition matrix that could fit all the ligand atoms
    final_transition_matrix = False
    for attempt in range(num_attempts):
        print "attempt:",attempt
        # make a random transition matrix
        random_transition_matrix = generate_random_transition_matrix()
        # affine transform
        transformed_ligand_coord = affine_transform(prody_ligand.getCoords(),random_transition_matrix)
        # see if all ligand atoms are in the box
        if all(map(if_in_the_box,transformed_ligand_coord[:,0:3])):
            final_transition_matrix = random_transition_matrix
            break

    # check if attempt to generate the transition matrix was successful
    # if not, exit procedure
    if isinstance(final_transition_matrix, bool):
        print "reached maximum number of attempts and failed", num_attempts
        return False
    # if script goes past this point, it means transition matrix for the box to fit all ligand atoms was generated
    # transform ligand to the new coordinate system (only do it once)
    prody_ligand.setCoords(affine_transform(prody_ligand.getCoords(), final_transition_matrix))
    # parse the protein

    # instead of parsing receptor (old slow version), script will be reading numpy arrays
    # prody_receptor = prody.parsePDB(receptor_filename)
    receptor_filename = re.sub('.pdb$', '', receptor_filename)

    #print receptor_filename + '_coor.npy'
    #print np.load(receptor_filename + '_coor.npy')
    #print np.load(receptor_filename + '_elem.npy')
    prody_receptor = prody.AtomGroup()
    prody_receptor.setCoords(np.load(receptor_filename + '_coor.npy'))
    prody_receptor.setElements(np.load(receptor_filename + '_elem.npy'))
    # now prody_receptor is the same prody object as prody_ligand


    # shift coordinates of the protein the same way as for thr ligand
    prody_receptor.setCoords(prody_receptor.getCoords() -ligand_center_of_mass)
    # transform the protein (only do it once)
    prody_receptor.setCoords(affine_transform(prody_receptor.getCoords(), final_transition_matrix))
    # write protein and ligand into 3d tensor
    # round coordinates before mapping to cells
    ceiled_ligand_coord = np.asarray(np.round(prody_ligand.getCoords()/pixel_size - 0.5 + side_pixels/2),dtype=int)
    ceiled_receptor_coord = np.asarray(np.round(prody_receptor.getCoords() / pixel_size - 0.5 + side_pixels / 2),dtype=int)
    # map to cells in 3D tensor
    tensor_3d = np.zeros((side_pixels,)*3)

    def mark_atom_on_tensor(ceiled_atom_coord,atomname,if_ligand,tensor_3d=tensor_3d):
        """receives atomnames and ceiled coordinate. 1) Converts atomname into number 2) writes
        this number into corresponding bit in 3D tensor"""

        return np.zeros(20,20,20)
        #if if_ligand:
            #atomic_tag_number = atom_dictionary.LIG[atomname.lower()]
            #tensor_3d[zip(np.ndarray.tolist(ceiled_atom_coord))] = atomic_tag_number
        #else:
            # check if the coordinate is in the box
            #if (0 < ceiled_atom_coord[0] < 20) \
            #        and (0 < ceiled_atom_coord[1] < 20) \
            #        and (0 < ceiled_atom_coord[2] < 20):
                    #len(tensor_3d[0][0][:])):
                # if it is, add atom tag to the tensor
                #atomic_tag_number = 1 #atom_dictionary.REC[atomname.lower()]
                #tensor_3d[zip(np.ndarray.tolist(ceiled_atom_coord))] = atomic_tag_number


    # map ligand on 3d tensor first
    #map(mark_atom_on_tensor, ceiled_ligand_coord[:, 0:3],prody_ligand.getElements(),np.ones(prody_ligand.numAtoms()))
    # map the protein on 3d tensor
    #map(mark_atom_on_tensor, ceiled_receptor_coord[:, 0:3], prody_receptor.getElements(), np.zeros(prody_receptor.numAtoms()))

    return tensor_3d




    # test and write transformed coordinates to the pdb file
    # print "ligand:",len(prody_ligand.getCoords()[:,0])
    # print "ceiled:",len(ceiled_ligand_coord[:,0])

    # prody_ligand.setCoords(np.asarray(ceiled_ligand_coord*pixel_size,dtype=float))
    # prody_receptor.setCoords(np.asarray(ceiled_receptor_coord*pixel_size,dtype=float))
    # prody.writePDB("debug_ligand",prody_ligand)
    # prody.writePDB("debug_protein",prody_receptor)




def launch_enqueue_workers(sess,coord,data_dir,pixel_size,side_pixels,num_workers,batch_size):
    """launch many threads to fill the queue in the backgrund. Gracefully close after specified amount of batches
    maintains two separate queues 1 - Image queue 2 - Filename queue """

    # launch one thread in the background to read the database index file

    database_index_file = tf.train.string_input_producer(["train_set.csv"])                                            # todo - change name
    reader = tf.TextLineReader()
    _, one_file = reader.read(database_index_file)
    col1, col2, col3 = tf.decode_csv(one_file, record_defaults=[[0], ["some_string"], ["some_string"]])



    filename_queue = tf.FIFOQueue(num_workers*5, [tf.int32, tf.string, tf.string], shapes=[[], [], []])
    enqueue_filename_op = filename_queue.enqueue([col1, col2, col3])
    label,ligand_filepath,protein_filepath = filename_queue.dequeue()

    def filename_queue_worker(coord):
        while not coord.should_stop():
            sess.run([col1, col2, col3, enqueue_filename_op])

    filename_enqueue_thread = threading.Thread(target=filename_queue_worker, args=(coord,))
    filename_enqueue_thread.start()

    # now, when background thread is running, I can generate image tensors
    # file location for multiple image_workers is pulled from filename_queue

    image_queue = tf.FIFOQueue(5, [tf.int32,tf.float32], shapes=[[],[side_pixels,side_pixels,side_pixels]])   #todo add _batch_size

    feed_label = tf.placeholder(tf.int32)
    feed_image = tf.placeholder(tf.float32,shape=[side_pixels,side_pixels,side_pixels])
    enqueue_image_op = image_queue.enqueue([feed_label,feed_image])

    def image_queue_worker(coord,worker_name):
        while not coord.should_stop():
            mylabel,myligand_filepath,myreceptor_filepath = sess.run([label,ligand_filepath,protein_filepath])

            myimage = generate_image(ligand_filename=myligand_filepath, receptor_filename=myreceptor_filepath,
                                   pixel_size=pixel_size, side_pixels=side_pixels)

            print "worker:",worker_name,"added image to the queue"
            sess.run([enqueue_image_op],feed_dict={feed_label:mylabel,feed_image:myimage})

            if False:                                                                                            # todo add stop_rule to all threads
                coord.request_stop()

    #with tf.device("/cpu:1"):                                                                                  #todo testing many CPUs
                                                                                                                # todo visualization of many queue workers
    image_enqueue_threads = [threading.Thread(target=image_queue_worker,args=(coord,worker_name,)) for worker_name in xrange(num_workers)]
    for t in image_enqueue_threads:
        print "thread started:",t
        t.start()
        time.sleep(0.05)
    return image_queue






sess = tf.Session()
coord = tf.train.Coordinator()


image_queue = launch_enqueue_workers(sess=sess,coord=coord,data_dir="home",pixel_size=1,side_pixels=20,num_workers=16,batch_size=100)
threads = tf.train.start_queue_runners(coord=coord, sess=sess)                                          # todo: move the thread coordinator further away
dequeue_image_op = image_queue.dequeue_many(100)




while True:
    start_time = time.time()
    sess.run([dequeue_image_op])
    speed_test = str(("dequeue speed test:","%.2f" % (100 / (time.time() - start_time)),"\n"))

    my_log_file = open("./av2_log", "a")
    my_log_file.writelines(speed_test)
    my_log_file.close()
    print speed_test
    print "--------------------------------------------------------------------------------------------------------"





print "done !"