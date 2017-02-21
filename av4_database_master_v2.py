import tensorflow as tf
import numpy as np
import time, os, re, prody
from av4_atomdict import atom_dictionary
from glob import glob

class stats:
    start = time.time()
    ligands_parsed = 0
    ligands_failed = 0


# crawls the database with crystal ligands
# looks which of them have docked ligands - if can acquire (?) and have pdb(?) - if can acquire
# moves all of them in a single folder with protein together
# writes statistics
# TODO make exceptions more informative

def convert_database_to_av4(database_path, positives_folder, decoys_folder, receptors_folder):
    """Crawls the folder (receptors in this case) and saves every PDB it finds
    into .npy array with 1) coordinates 2) mapped to the atom name number """

    # make a directory where the av4 form of the output will be written
    output_path = str(database_path + '_av4')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save_av4(filepath, labels, elements, multiframe_coords):
        labels = np.asarray(labels, dtype=np.int32)
        elements = np.asarray(elements, dtype=np.int32)
        multiframe_coords = np.asarray(multiframe_coords, dtype=np.float32)

        if not (int(len(multiframe_coords[:, 0]) == int(len(elements)))):
            raise Exception('Number of atom elements is not equal to the number of coordinates')

        if multiframe_coords.ndim == 2:
            if not int(len(labels)) == 1:
                raise Exception('Number labels is not equal to the number of coordinate frames')
        else:
            if not (int(len(multiframe_coords[0, 0, :]) == int(len(labels)))):
                raise Exception('Number labels is not equal to the number of coordinate frames')

        number_of_examples = np.array([len(labels)], dtype=np.int32)
        av4_record = number_of_examples.tobytes()
        av4_record += labels.tobytes()
        av4_record += elements.tobytes()
        av4_record += multiframe_coords.tobytes()
        f = open(filepath + ".av4", 'w')
        f.write(av4_record)
        f.close()

    file_list = glob(os.path.join(database_path,'*','*.pdb'))
    for filepath in file_list:
        path_to_positive = filepath
        ligand_output_path = path_to_positive.replace('.pdb','')

        try:
            # prody_receptor = prody.parsePDB(path_to_receptor)
            prody_positive = prody.parsePDB(path_to_positive)

            # ligand_cords will store multiple frames
            multiframe_ligand_coords = prody_positive.getCoords()
            labels = np.array([1])

        except Exception as e:
            print e
            stats.ligands_failed += 1
            print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed
            continue

        def atom_to_number(atomname):
            atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
            return atomic_tag_number

        ligand_elements = map(atom_to_number, prody_positive.getElements())
        save_av4(ligand_output_path, labels, ligand_elements, multiframe_ligand_coords)





convert_database_to_av4(database_path="/home/ubuntu/xiao/data/newkaggle/dude/crystal/crystal_ligands", positives_folder="crystal_ligands",
                        decoys_folder="docked_ligands", receptors_folder='receptors')

