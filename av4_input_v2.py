import tensorflow as tf
from glob import glob
import os,time
import numpy as np
from av4_utils import generate_deep_affine_transform,affine_transform
# How many frame appear for each example
ligands_frame_num = 100


def index_the_database_into_queue(database_path, shuffle):
    """Indexes av4 database and returns two lists of filesystem path: ligand files, and protein files.
    Ligands are assumed to end with _ligand.av4, proteins should be in the same folders with ligands.
    Each protein should have its own folder named similarly to the protein name (in the PDB)."""
    # TODO controls epochs here
    ligand_file_list = []
    receptor_file_list = []
    # for the ligand it's necessary and sufficient to have an underscore in it's name
    print "number of ligands:", len(glob(os.path.join(database_path + '/**/**/', "*[_]*.av4")))

    for ligand_file in glob(os.path.join(database_path + '/**/**/', "*[_]*.av4")):
        receptor_file = "/".join(ligand_file.split("/")[:-1]) + "/" + ligand_file.split("/")[-1][:4] + '.av4'
        if os.path.exists(receptor_file):
            ligand_file_list.append(ligand_file)
            receptor_file_list.append(receptor_file)
        else:

            # TODO: remove another naming system from Xiao's scripts                #
            receptor_file = os.path.join(os.path.dirname(ligand_file),
                                         os.path.basename(ligand_file).split("_")[0] + '.av4')
            if os.path.exists(receptor_file):  # remove later
                ligand_file_list.append(ligand_file)  #
                receptor_file_list.append(receptor_file)  #

    def crystal_path(ligand_path):
        receptor = os.path.basename(ligand_path).split('_')[0]
        crystal_file_path = os.path.join('/home/ubuntu/xiao/data/newkaggle/dude/crystal/crystal_ligands',receptor,receptor+'_crystal.av4')
        return crystal_file_path

    crystal_file_list = map(crystal_path,ligand_file_list)
    index_list = range(len(ligand_file_list))
    examples_in_database = len(index_list)

    if examples_in_database == 0:
        raise Exception('av4_input: No files found in the database path:', database_path)
    print "Indexed ligand-protein pairs in the database:", examples_in_database

    # create a filename queue (tensor) with the names of the ligand and receptors
    index_tensor = tf.convert_to_tensor(index_list, dtype=tf.int32)
    ligand_files = tf.convert_to_tensor(ligand_file_list, dtype=tf.string)
    crystal_files = tf.convert_to_tensor(crystal_file_list,dtype=tf.string)
    receptor_files = tf.convert_to_tensor(receptor_file_list, dtype=tf.string)

    filename_queue = tf.train.slice_input_producer([index_tensor, ligand_files,crystal_files ,receptor_files], num_epochs=None,
                                                   shuffle=shuffle)
    return filename_queue, examples_in_database


def read_receptor_and_multiframe_ligand(filename_queue, epoch_counter):
    def decode_av4(serialized_record):
        # decode everything into int32
        tmp_decoded_record = tf.decode_raw(serialized_record, tf.int32)
        # first four bytes describe the number of frames in a record
        number_of_frames = tf.slice(tmp_decoded_record, [0], [1])
        # labels are saved as int32 * number of frames in the record
        labels = tf.slice(tmp_decoded_record, [1], number_of_frames)
        # elements are saved as int32 and their number is == to the number of atoms
        number_of_atoms = ((tf.shape(tmp_decoded_record) - number_of_frames - 1) / (3 * number_of_frames + 1))
        elements = tf.slice(tmp_decoded_record, number_of_frames + 1, number_of_atoms)

        # coordinates are saved as a stack of X,Y,Z where the first(vertical) dimension
        # corresponds to the number of atoms
        # second (horizontal dimension) is x,y,z coordinate of every atom and is always 3
        # third (depth) dimension corresponds to the number of frames

        coords_shape = tf.concat(0, [number_of_atoms, [3], number_of_frames])
        tmp_coords = tf.slice(tmp_decoded_record, number_of_frames + number_of_atoms + 1,
                              tf.shape(tmp_decoded_record) - number_of_frames - number_of_atoms - 1)
        multiframe_coords = tf.bitcast(tf.reshape(tmp_coords, coords_shape), type=tf.float32)

        return labels, elements, multiframe_coords

    idx = filename_queue[0]
    ligand_file = filename_queue[1]
    serialized_ligand = tf.read_file(ligand_file)
    serialized_crystal = tf.read_file(filename_queue[2])
    serialized_receptor = tf.read_file(filename_queue[3])


    ligand_labels, ligand_elements, multiframe_ligand_coords = decode_av4(serialized_ligand)
    _,_,crystal_coords = decode_av4(serialized_crystal)
    receptor_labels, receptor_elements, multiframe_receptor_coords = decode_av4(serialized_receptor)

    
    
    # select some frame of ligands, if don't have enough frame, repeat it
    size = tf.maximum(ligands_frame_num, tf.shape(ligand_labels)[0])
    select_range = tf.range(0, size)
    select_frame = tf.mod(select_range,tf.shape(ligand_labels)[0])
    multiple_ligand_coords = tf.gather(tf.transpose(multiframe_ligand_coords,perm=[2,0,1]),select_frame)
    labels = tf.gather(ligand_labels,select_frame)

    return ligand_file, tf.squeeze(epoch_counter), tf.squeeze(labels), ligand_elements, tf.squeeze(
        multiple_ligand_coords), tf.squeeze(crystal_coords),receptor_elements, tf.squeeze(multiframe_receptor_coords)

def convert_protein_and_multiple_ligand_to_image(ligand_elements,multiple_lgiand_coords,crystal_ligand_coords,receptor_elements,receptor_coords,side_pixels,pixel_size):

    max_num_attempts = 1000
    affine_transform_pool_size = 10000

    ligand_center_of_mass =tf.reduce_mean(crystal_ligand_coords,reduction_indices=0)
    centered_crystal_ligand = crystal_ligand_coords - ligand_center_of_mass
    centered_multiple_ligand_coords = multiple_lgiand_coords - ligand_center_of_mass
    centered_receptor_coords = receptor_coords - ligand_center_of_mass
    
    box_size = (tf.cast(side_pixels, tf.float32)* pixel_size)
    epsilon = tf.constant(0.999, dtype=tf.float32)


    def generate_transition_matrix(attempt, transition_matrix, batch_of_transition_matrices):
        transition_matrix = tf.gather(batch_of_transition_matrices,
                                      tf.random_uniform([],minval=0,maxval=affine_transform_pool_size,
                                                        dtype=tf.int32))
        attempt +=1
        return attempt, transition_matrix,batch_of_transition_matrices
    
    def not_enough_in_the_box(attempt, transition_matrix, batch_of_transition_matrices,centered_crystal_ligand=centered_crystal_ligand,
                           multiple_ligand_coords = centered_multiple_ligand_coords,box_size= box_size,
                           max_num_attempts=max_num_attempts):
        def affine_multiple_transform(index,multiple_coordinates=multiple_ligand_coords, transition_matrix=transition_matrix):
            # use map_fn to transform multiple frames
            coordinates = tf.gather(multiple_coordinates,index)
            transformed_coordinates,_ = affine_transform(coordinates,transition_matrix)
            return transformed_coordinates

        # if crystal ligand in the box
        #
        #
        transformed_crystal_coords,transition_matrix = affine_transform(centered_crystal_ligand,transition_matrix)
        not_all = tf.cast(
            tf.reduce_max(tf.cast(tf.square(box_size * 0.5) - tf.square(transformed_crystal_coords) < 0, tf.int32)), tf.bool)
        within_iteration_limit = tf.cast(tf.reduce_sum(tf.cast(attempt < max_num_attempts, tf.float32)), tf.bool)

        # if docked ligands in the box
        #
        #
        # transformed_coords.shape [n_frame,n_atoms,3]
        transformed_coords = tf.map_fn(affine_multiple_transform,tf.range(tf.shape(multiple_ligand_coords)[0]),dtype=tf.float32,parallel_iterations=1)
        # out_of_box_atoms.shape [ n_frame, n_atoms]
        #out_of_box_atoms = tf.squeeze(tf.reduce_sum(tf.cast(tf.square(box_size*0.5) - tf.cast(tf.square(transformed_coords),tf.float32)<0,tf.int32),reduction_indices=-1))
        # out_of_box_frame [n_frame]
        #out_of_box_frame = tf.squeeze(tf.cast(tf.reduce_sum(out_of_box_atoms,reduction_indices=-1)>0,tf.int32))
        #in_the_box_frame = tf.ones(tf.shape(out_of_box_frame),tf.int32) - out_of_box_frame

        # another approch
        #
        #
        #out_of_box_atoms = tf.squeeze(tf.reduce_sum(
        #    tf.cast(tf.less(box_size*0.5,tf.cast(tf.abs(transformed_coords),tf.float32)), tf.int32),
        #    reduction_indices=-1))
        #out_of_box_frame = tf.squeeze(tf.cast(tf.reduce_sum(out_of_box_atoms, reduction_indices=-1) > 0, tf.int32))
        #in_the_box_frame = tf.ones(tf.shape(out_of_box_frame), tf.int32) - out_of_box_frame

        # Third approch
        #
        #

        ceiled_ligand_coords = tf.cast(
            tf.round((tf.constant(-0.5, tf.float32) + (tf.cast(side_pixels, tf.float32) / 2.0) + (
                transformed_coords / pixel_size)) * epsilon),
            tf.int64)
        top_filter = ceiled_ligand_coords  >= side_pixels
        bottom_filter = ceiled_ligand_coords < 0
        out_of_box_atoms = tf.squeeze(tf.reduce_sum(tf.cast(tf.logical_or(top_filter, bottom_filter),tf.int32),reduction_indices=-1))
        out_of_box_frame = tf.squeeze(tf.reduce_sum(tf.cast(out_of_box_atoms>0,tf.int32),reduction_indices=-1))

        in_the_box_frame = tf.ones(tf.shape(out_of_box_frame), tf.int32) - out_of_box_frame

        return tf.logical_and(tf.logical_and(within_iteration_limit, not_all), tf.less(tf.reduce_sum(in_the_box_frame), ligands_frame_num))

    attempt = tf.Variable(tf.constant(0,shape=[1]))
    batch_of_transition_matrices = tf.Variable(generate_deep_affine_transform(affine_transform_pool_size))

    transition_matrix = tf.gather(batch_of_transition_matrices,
                                  tf.random_uniform([],minval=0,maxval=affine_transform_pool_size,dtype=tf.int64))

    last_attempt, final_transition_matrix, _ = tf.while_loop(not_enough_in_the_box,generate_transition_matrix,
                                                             [attempt,transition_matrix,batch_of_transition_matrices],
                                                             parallel_iterations=1)

    def affine_multiple_transform_1(index, multiple_coordinates=centered_multiple_ligand_coords,
                                  transition_matrix=final_transition_matrix):
        coordinates = tf.gather(multiple_coordinates, index)
        transformed_coordinates, _ = affine_transform(coordinates, transition_matrix)
        return transformed_coordinates

    rotatated_ligand_coords = tf.map_fn(affine_multiple_transform_1,tf.range(tf.shape(centered_multiple_ligand_coords)[0]),dtype=tf.float32,parallel_iterations=1)
    rotated_receptor_coords, _ = affine_transform(centered_receptor_coords, final_transition_matrix)



    #out_of_box_atoms = tf.squeeze(
    #    tf.reduce_sum(tf.cast(tf.square(box_size * 0.5) - tf.cast(tf.square(rotatated_ligand_coords),tf.float32) < 0, tf.int32),
    #                  reduction_indices=-1))

    #out_of_box_atoms = tf.squeeze(tf.reduce_sum(
    #    tf.cast(tf.less(box_size * 0.5, tf.cast(tf.abs(rotatated_ligand_coords), tf.float32)), tf.int32),
    #    reduction_indices=-1))

    #out_of_box_frame = tf.squeeze(tf.cast(tf.reduce_sum(out_of_box_atoms, reduction_indices=-1) > 0, tf.int32))

    #in_the_box_frame = tf.ones(tf.shape(out_of_box_frame),tf.int32) - out_of_box_frame

    ceiled_ligand_coords = tf.cast(
        tf.round((tf.constant(-0.5, tf.float32) + (tf.cast(side_pixels, tf.float32) / 2.0) + (
            rotatated_ligand_coords / pixel_size)) * epsilon),
        tf.int64)
    top_filter = ceiled_ligand_coords >= side_pixels
    bottom_filter = ceiled_ligand_coords < 0
    out_of_box_atoms = tf.squeeze(
        tf.reduce_sum(tf.cast(tf.logical_or(top_filter, bottom_filter), tf.int32), reduction_indices=-1))
    out_of_box_frame = tf.squeeze(tf.reduce_sum(tf.cast(out_of_box_atoms > 0,tf.int32), reduction_indices=-1))

    in_the_box_frame = tf.ones(tf.shape(out_of_box_frame), tf.int32) - out_of_box_frame

    inbox_ligand_coords = tf.gather(ceiled_ligand_coords, in_the_box_frame)

    def set_elements_coords_zero(): return  tf.constant([0],dtype=tf.int32),tf.constant([0], dtype=tf.int32), tf.zeros([1,1,3], dtype=tf.int64)
    def keep_elements_coords(): return tf.constant([1],dtype=tf.int32), tf.cast(ligand_elements,tf.int32), inbox_ligand_coords


    # transformed label 1 when rotate success 0 when failed
    transformed_label,ligand_elements,select_ligands_coords = tf.case({tf.less(tf.reduce_sum(in_the_box_frame), ligands_frame_num):set_elements_coords_zero},
                                                                      keep_elements_coords)




    epsilon = tf.constant(0.999, dtype=tf.float32)

    #ceiled_ligand_coords = tf.cast(
    #    tf.round((half_side_pixels+scalar_ligand_coords) * epsilon),
    #    tf.int64)

    ceiled_receptor_coords = tf.cast(
        tf.round((tf.constant(-0.5,tf.float32) + (tf.cast(side_pixels, tf.float32) /2.0) + (rotated_receptor_coords / pixel_size)) * epsilon),
        tf.int64)

    top_filter = tf.reduce_max(ceiled_receptor_coords, reduction_indices=1) < side_pixels
    bottom_filter = tf.reduce_min(ceiled_receptor_coords, reduction_indices=1) > 0
    retain_atoms = tf.logical_and(top_filter, bottom_filter)
    cropped_receptor_coords = tf.boolean_mask(ceiled_receptor_coords, retain_atoms)
    cropped_receptor_elements = tf.boolean_mask(receptor_elements, retain_atoms)


    multiple_cropped_receptor_coords = tf.ones([tf.shape(ceiled_ligand_coords)[0],1,1],tf.int64)*cropped_receptor_coords
    complex_coords = tf.concat(1, [select_ligands_coords, multiple_cropped_receptor_coords])
    complex_elements = tf.concat(0, [ligand_elements + 7, cropped_receptor_elements])

    # for each frame assign the 4th dimention as 0,1,2...100
    dimention_4th = tf.cast(tf.reshape(tf.range(tf.shape(complex_coords)[0]),
                                       [tf.shape(complex_coords)[0],1,1]),tf.int32)
    base = tf.ones([tf.shape(complex_coords)[0],tf.shape(complex_coords)[1],1],tf.int32)
    ligand_frame_depth =tf.cast( dimention_4th*base,tf.int64)
    # repeat complex_elements 100 time, used to generate SparseTensor
    multiple_complex_elements =tf.reshape( base*tf.reshape(complex_elements,[1,tf.shape(complex_elements)[0],1]),[-1])
    complex_coords_4d =tf.reshape( tf.concat(2,
                                  [complex_coords,ligand_frame_depth]),[-1,4])
    sparse_image_4d = tf.SparseTensor(indices=complex_coords_4d, values = multiple_complex_elements,
                                      shape=[side_pixels, side_pixels, side_pixels, ligands_frame_num])
    

    def generate_sparse_image_4d(depth,complex_coords=complex_coords,complex_elements=complex_elements):
        single_complex_coords = tf.gather(complex_coords,depth)
        complex_coords_4d = tf.concat(1,
                                      [single_complex_coords,
                                       tf.reshape(tf.ones(tf.shape(complex_elements),tf.int64)*tf.cast(depth,tf.int64) ,[-1, 1])])
        sparse_image_4d = tf.SparseTensor(indices=complex_coords_4d, values=complex_elements,
                                          shape=[side_pixels, side_pixels, side_pixels, ligands_frame_num])
        return sparse_image_4d

    

   # sparse_images = tf.map_fn(generate_sparse_image_4d,tf.range(tf.shape(complex_coords)[0]))
   # sparse_image_4d = tf.sparse_concate(-1,sparse_images)
    return sparse_image_4d,ligand_center_of_mass,final_transition_matrix,transformed_label


def image_and_label_queue(batch_size, pixel_size, side_pixels, num_threads, filename_queue, epoch_counter):

    ligand_file, current_epoch, labels, ligand_elements, multiple_ligand_coords, crystal_coords,receptor_elements, receptor_coords = read_receptor_and_multiframe_ligand(
        filename_queue,epoch_counter=epoch_counter)

    sparse_image,_,_ ,transformed_label= convert_protein_and_multiple_ligand_to_image(ligand_elements,multiple_ligand_coords,crystal_coords,receptor_elements,receptor_coords,
                                                           side_pixels,pixel_size)

    '''
    frames = []
    masks = []

    for i in range(multiframe_num):
        sparse_image,_,_,indicate = convert_protein_and_ligand_to_image(ligand_elements, multiple_ligand_coords, receptor_elements,
                                                                receptor_coords, side_pixels, pixel_size,i)
        frames.append(sparse_image)
        masks.append(indicate)



    #multiframe_batch = tf.boolean_mask(tensor=frames,mask=masks)
    #image_4d = tf.sparse_concat(-1,multiframe_batch)
    concate_sparse = lambda :tf.sparse_concate(-1,frames)
    '''

    label  = tf.cast(tf.greater(tf.reduce_sum(labels),0),tf.int32)
    final_label = label*transformed_label



    return ligand_file,current_epoch,final_label,sparse_image
    #return ligand_file,current_epoch,label,frames,masks
