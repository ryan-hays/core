import numpy as np
import os,re, random,prody
from itertools import chain
from av2_atomdict import atom_dictionary

# database master
# 1) crawls directories and writes a human-readable database_index.csv file
# 2) splits the data into training and testing sets

# 3) parses proteins and writes useful data in them into .npy formatted arrays
# 3) later it's much faster to read from npy arrays

# todo there is a problem with multiframe PDBs that is supposed to be fixed with Xiao's update of the database
# todo some error handling and loggig
# todo add time.time

def preprocess_PDB_to_npy(database_path):
    """crawls the folder (receptors in this case) and saves every PDB it finds
    into .npy array with 1) coordinates 2) mapped to the atom name number """

    def preprocess_ligands(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if re.search('.pdb$', filename):
                    ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                    try:
                        prody_ligand = prody.parsePDB(ligand_file_path)
                    except Exception:
                        pass
                    def ligand_atom_to_number(atomname):
                        atomic_tag_number = atom_dictionary.LIG[atomname.lower()]
                        return atomic_tag_number

                    try:

                        atom_numbers = map(ligand_atom_to_number, prody_ligand.getElements())
                        coordinates_and_atoms = np.hstack((prody_ligand.getCoords(), np.reshape(atom_numbers, (-1, 1))))
                        np.save(re.sub('.pdb$', '', ligand_file_path), coordinates_and_atoms)
                    except Exception:
                        pass


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

    preprocess_ligands(database_path + "/crystal_ligands/")
    preprocess_ligands(database_path + "/docked_ligands/")
    preprocess_receptors(database_path + "/receptors/")




def assign_label(ligand_file_path):                                                                         # todo
    """assigns label to be used for training. Returns "NONE" when the molecule should not be used for training"""
    # 1 for the crystal structure
    # 0 for bad fast docking
    # NONE for good fast docking
    label = None
    if re.search('crystal_ligands', ligand_file_path):
        if label == None:
            label = 1
        else:
            raise Exception('can not assign two labels to one example')

    if re.search('docked_ligands',ligand_file_path):
        if label == None:
            label = 0
        else:
            raise Exception('can not assign two labels to one example')

    if re.search('/ligands/',ligand_file_path):
        if label == None:
            label = random.randint(0,1)
    if label == None:
        raise Exception("can not assign label")
    else:
        return label


    # TODO if both have been assigned
    # check if one of the labels has been assigned

    #random_label = int(round(random.random()))
    #return random_label



def write_database_index_file(database_path,database_index_path):
    """crowls the database of the structure we are using and indexes it"""
    database_index_file = open(database_index_path + "/database_index.csv", "w")
    step = 0

    # walking across the ligand folders
    for dirpath, dirnames, filenames in chain(os.walk(database_path +"/crystal_ligands"),os.walk(database_path +"/docked_ligands")):

        for filename in filenames:
            # first look if what we found is a ligand
            pdb_name = dirpath.split("/")[-1]

            if re.search('.npy$',filename):
                # second see if we can find a corresponding protein
                ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)
                label = assign_label(ligand_file_path)
                if label is not None:
                    print "added to database",step,"label:",label
                    step +=1
                    receptor_file_path = os.path.abspath(str(database_path + "/receptors/" + pdb_name + ".npy"))
                
                    if os.path.exists(receptor_file_path):
                        database_index_file.write(str(label) + "," + ligand_file_path + "," + receptor_file_path + "\n")
                    else:
                        print "WARNING: missing receptor"

    database_index_file.close()


def split_into_train_and_test_sets(database_index_path,train_set_div):
    """splits the database textfile into training and testing sets. Takes into account that some of the PDBs
    are multimers. All multimers go i"""
    # create a unique pdb list, and randomly assign every pdb to the training or to the testing set
    database_index_lines = open(database_index_path + "/database_index.csv").readlines()

    def return_pdb(string):
        return string.split("/")[-1].strip("\n")

    pdb_names = np.array(map(return_pdb,database_index_lines))
    unique_pdb_names = np.unique(pdb_names)

    # now split according to the group
    test_set = []
    train_set = []

    def randomly_assign_and_retrieve_example(unique_pdb_name,pdb_names=pdb_names,train_set_div=train_set_div,database_index_lines=database_index_lines):
        # Given a pdb name randomly assign it either to the training or to the testing set keeping the probability.
        # Retrieve all of the examples with this pdb name from the database and write into training/testing file

        for index in np.where(pdb_names == unique_pdb_name)[0]:
            if (random.random() < train_set_div):
                test_set.append(database_index_lines[index])
            else:
                train_set.append(database_index_lines[index])

    map(randomly_assign_and_retrieve_example,unique_pdb_names)

    random.shuffle(train_set)
    open(database_index_path + "/test_set.csv", "w").writelines(train_set)
    random.shuffle(test_set)
    open(database_index_path + "/train_set.csv", "w").writelines(test_set)

def prepare_database_for_av3(database_path,train_set_div,convert_to_npy,write_index,split):
    if convert_to_npy:
        preprocess_PDB_to_npy(database_path)
    if write_index:
        write_database_index_file(database_path,database_path)
    if split:
        split_into_train_and_test_sets(database_path,train_set_div)

prepare_database_for_av3(database_path='../datasets/labeled_npy',train_set_div=0.8,convert_to_npy=False,write_index=True,split=True)

# Index test_set with out lab
prepare_database_for_av3(database_path='../datasets/unlabeled_npy',train_set_div=1.0,convert_to_npy=False,write_index=False,split=False)