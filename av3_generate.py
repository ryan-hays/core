import numpy as np
import os,re, random,prody
from itertools import chain
from atom_dict import atom_dictionary


def generate_random_transition_matrix(shift_range=20,rotation_range=np.pi*2):
    """returns a random transition matrix
    rotation range - determines random rotations along any of X,Y,Z axis
    shift_range determines allowed shifts along any of X,Y,Z axis """
    # REMARK: shift range is hard coded to 10A because that's how the proteins look like

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

def affine_transform(coordinate_array):
    """applies transition to every point of the array"""

    ligand_center_of_mass = np.average(coordinate_array, axis=0)
    relative_coords = coordinate_array - ligand_center_of_mass



    relative_matrix = np.matrix(relative_coords)
    raw_of_ones = np.matrix(np.ones(len(relative_matrix[:, 0])))

    transition_matrix = generate_random_transition_matrix()

    transformed_coord = (transition_matrix * np.vstack((relative_matrix.transpose(), raw_of_ones)))[0:3, :].transpose()

    final_coord = transformed_coord+ligand_center_of_mass

    return np.array(final_coord)

def generate(ligand_path,rotate_num=10000):

    destPath = '/home/ubuntu/xiao/data/random_gen/ligands'
    ligand_name =os.path.basename(ligand_path)
    pdb_name = ligand_name.split('_')[0]
    destFilepath = os.path.join(destPath,pdb_name)
    if not os.path.exists(destFilepath):
        os.mkdir(destFilepath)
    try:
        prody_ligand = prody.parsePDB(ligand_path)
    except Exception as e:
        print e

    ligand_atom_to_number = lambda atomname:atom_dictionary.LIG[atomname.lower()]

    try:
        atom_numbers = map(ligand_atom_to_number,prody_ligand.getElements())
        coordinates = prody_ligand.getCoords()
        for i in range(rotate_num):
            transformed_coords = affine_transform(coordinates)
            coordinates_and_atoms = np.hstack((transformed_coords,np.reshape(atom_numbers,(-1,1))))
            save_name = re.sub('.pdb$','_'+str(i),ligand_name)
            np.save(os.path.join(destFilepath,save_name),coordinates_and_atoms)
            print i
    except Exception as e:
        print e

def preprocess_receptors(folder_path):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if re.search('.pdb$', filename):
                receptor_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                prody_receptor = prody.parsePDB(receptor_file_path)

                def receptor_atom_to_number(atomname):
                    atomic_tag_number = atom_dictionary.REC[atomname.lower()]
                    return atomic_tag_number
                try:
                    atom_numbers = map(receptor_atom_to_number, prody_receptor.getElements())
                    coordinates_and_atoms = np.hstack((prody_receptor.getCoords(), np.reshape(atom_numbers, (-1, 1))))
                    np.save(re.sub('.pdb$', '', receptor_file_path), coordinates_and_atoms)
                except Exception:
                    pass

generate('/home/ubuntu/maksym/old/dataset_dec04/labeled_pdb/crystal_ligands/2pw3/2pw3_653_ligand.pdb')

preprocess_receptors('/home/ubuntu/xiao/data/random_gen/receptors')