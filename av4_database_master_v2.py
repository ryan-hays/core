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

def convert_dude_database_to_av4(database_path,positives_folder=None,decoys_folder=None,receptors_folder=None):
    output_path = str(database_path+'_av4')
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

    for receptor in os.listdir(database_path):
        receptor_folder = os.path.join(database_path,receptor)
        for ligand in os.listdir(receptor_folder):
            try:
                ligand_folder = os.path.join(os.path.join(receptor_folder,ligand))
                receptor_path = os.path.join(os.path.dirname(database_path),'receptors',receptor+'.pdb')
                frames = os.listdir(ligand_folder)
                frame_num = len(frames)
                prody_receptor = prody.parsePDB(receptor_path)

                for i in range(1,frame_num+1):
                    prody_positive = prody.parsePDB(os.path.join(ligand_folder, ligand + '_' + i + '.pdb'))
                    if i == 1:
                        multiframe_ligand_coords = prody_positive.getCoords()
                        labels = np.array([1])
                        ligand_elements = np.asarray(prody_positive.getElements())
                    else:
                        multiframe_ligand_coords = np.dstack(
                                    (multiframe_ligand_coords, prody_positive.getCoords()))
                        labels = np.concatenate((labels,[0]))

                        if not all(np.assarray(prody_positive.getElements())==ligand_elements):
                            raise Exception('attempting to add ligand with different order of atoms')
            except Exception as e:
                print e
                stats.ligands_failed +=1
                print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed
                continue

            stats.ligands_parsed+=1
            print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed

            dest_folder = os.path.join(output_path,receptor)
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)

            def atom_to_number(atomname):
                atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
                return atomic_tag_number

            receptor_elements = map(atom_to_number, prody_receptor.getElements())
            ligand_elements = map(atom_to_number, ligand_elements)

            save_av4(os.path.join(dest_folder,receptor), [0], receptor_elements, prody_receptor.getCoords())
            save_av4(os.path.join(dest_folder,ligand), labels, ligand_elements, multiframe_ligand_coords)


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





#convert_database_to_av4(database_path="/home/ubuntu/xiao/data/newkaggle/dude/crystal/crystal_ligands", positives_folder="crystal_ligands",
#                        decoys_folder="docked_ligands", receptors_folder='receptors')
convert_dude_database_to_av4(database_path="/home/ubuntu/xiao/data/newkaggle/dude/")
