import tensorflow as tf
import numpy as np
import time,os,re,prody
from av4_atomdict import atom_dictionary

class stats:
    start = time.time()
    ligands_parsed = 0
    ligands_failed = 0

# crawls the database with crystal ligands
# looks which of them have docked ligands - if can acquire (?) and have pdb(?) - if can acquire
# moves all of them in a single folder with protein together
# writes statistics
# TODO make exceptions more informative

def convert_database_to_av4(database_path,positives_folder,decoys_folder,receptors_folder):
    """Crawls the folder (receptors in this case) and saves every PDB it finds
    into .npy array with 1) coordinates 2) mapped to the atom name number """

    # make a directory where the av4 form of the output will be written
    output_path = str(database_path+'_av4')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save_av4(filepath,labels,elements,multiframe_coords):
        labels = np.asarray(labels,dtype=np.int32)
        elements = np.asarray(elements,dtype=np.int32)
        multiframe_coords = np.asarray(multiframe_coords,dtype=np.float32)

        if not (int(len(multiframe_coords[:,0]) == int(len(elements)))):
            raise Exception('Number of atom elements is not equal to the number of coordinates')

        if multiframe_coords.ndim==2:
            if not int(len(labels))==1:
                raise Exception ('Number labels is not equal to the number of coordinate frames')
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


    for dirpath,dirnames,filenames in os.walk(database_path +"/"+ positives_folder):
        for filename in filenames:
            if re.search('.pdb$', filename):

                path_to_positive = str(os.path.abspath(dirpath) + "/" + filename)

                # find path to folder with decoys
                path_to_decoys = re.sub(positives_folder,decoys_folder,str(os.path.abspath(dirpath)))
                # find path to the file with receptor
                path_to_receptor = re.sub(positives_folder,receptors_folder,str(os.path.abspath(dirpath))+'.pdb')

                try:
                    prody_receptor = prody.parsePDB(path_to_receptor)
                    prody_positive = prody.parsePDB(path_to_positive)

                    # ligand_cords will store multiple frames
                    multiframe_ligand_coords = prody_positive.getCoords()
                    labels = np.array([1])

                    # find all decoys
                    if os.path.exists(path_to_decoys):
                        for decoyname in os.listdir(path_to_decoys):
                            if re.match(re.sub('.pdb', '', path_to_positive.split("/")[-1]),re.sub('.pdb', '', decoyname)):
                                prody_decoy = prody.parsePDB(path_to_decoys + "/" + decoyname)

                                # see if decoy is same as the initial ligand
                                if not all(np.asarray(prody_decoy.getElements()) == np.asarray(prody_positive.getElements())):
                                    raise Exception('attempting to add ligand with different order of atoms')

                                multiframe_ligand_coords = np.dstack((multiframe_ligand_coords,prody_decoy.getCoords()))


                                labels = np.concatenate((labels,[0]))


                    if multiframe_ligand_coords.ndim==2:
                        raise Exception('no decoy molecules found')

                except Exception as e:
                    print e
                    stats.ligands_failed+=1
                    print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed
                    continue

                stats.ligands_parsed += 1
                print "ligands parsed:", stats.ligands_parsed, "ligands failed:", stats.ligands_failed

                # create an output path to write binaries for protein and ligands
                path_to_pdb_subfolder = output_path + "/" + str(os.path.abspath(dirpath)).split("/")[-1]

                if not os.path.exists(path_to_pdb_subfolder):
                    os.makedirs(path_to_pdb_subfolder)

                # convert atomnames to tags and write the data to disk
                def atom_to_number(atomname):
                    atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
                    return atomic_tag_number

                receptor_elements = map(atom_to_number,prody_receptor.getElements())
                ligand_elements = map(atom_to_number,prody_positive.getElements())

                receptor_output_path = path_to_pdb_subfolder + "/" +str(os.path.abspath(dirpath)).split("/")[-1]
                save_av4(receptor_output_path,[0],receptor_elements,prody_receptor.getCoords())
                ligand_output_path = path_to_pdb_subfolder + "/" + path_to_positive.split("/")[-1].split(".")[0]
                save_av4(ligand_output_path,labels,ligand_elements,multiframe_ligand_coords)




convert_database_to_av4(database_path="../datasets/labeled_pdb",positives_folder="crystal_ligands",decoys_folder="docked_ligands",receptors_folder='receptors')































#    def preprocess_ligands(folder_path):
#        for dirpath, dirnames, filenames in os.walk(folder_path):
#            for filename in filenames:
#                if re.search('.pdb$', filename):
#                    ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)
#                    try:
#                        prody_ligand = prody.parsePDB(ligand_file_path)
#                    except Exception:
#                        pass
#
#                    def ligand_atom_to_number(atomname):
#                        atomic_tag_number = atom_dictionary.LIG[atomname.lower()]
#                        return atomic_tag_number
#
#                    try:
#
#                        atom_numbers = map(ligand_atom_to_number, prody_ligand.getElements())
#                        coordinates_and_atoms = np.hstack(
#                            (prody_ligand.getCoords(), np.reshape(atom_numbers, (-1, 1))))
#                        np.save(re.sub('.pdb$', '', ligand_file_path), coordinates_and_atoms)
#                    except Exception:
#                        pass





"""
def convert_npy_to_av3(database_path,output_folder):
    #crawls the folder (receptors in this case) and saves every PDB it finds
    #into .npy array with 1) coordinates 2) mapped to the atom name number

    def preprocess_ligands_in_folder(docked_dir,cryst_dir):
        for dirpath, dirnames, filenames in os.walk(cryst_dir):
            for filename in filenames:
                if re.search('.pdb$', filename):

                    docked_file_path = str(os.path.abspath(dirpath) + "/" + filename)
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
                    except Exception:
                pass


                    crystal_ligand_file_path = str(os.path.abspath(dirpath) + "/" + filename)

                    # walk through the folder with docked ligands
                    pdb_name = dirpath.split("/")[-1]
                    crystal_ligand_name = filename.split(".")[0]

                    multiframe_ligand = np.array([])
                    for dirpath2, dirnames2, filenames2 in os.walk(docked_dir + "/" + pdb_name):
                        for filename2 in filenames2:
                            if re.search(crystal_ligand_name, filename2) and re.search('.npy$', filename2) and re.search('_fast',filename2):
                                # calculate the number of atoms
                                docked_ligand_file_path = str(os.path.abspath(dirpath2) + "/" + filename2)
                                docked_ligand = molecule_class(docked_ligand_file_path)
                                crystal_ligand = molecule_class(crystal_ligand_file_path)


                                # collect all of the docked ligands together in one array (numpy at the moment)
                                if multiframe_ligand.size:
                                    multiframe_ligand = np.hstack((multiframe_ligand,docked_ligand.coords))
                                else:
                                    multiframe_ligand = np.hstack((docked_ligand.elements.reshape(-1,1),docked_ligand.coords))
                                print "next ligand"


                    # save the stack of ligands to disk
                    # also save the protein
                    # the name should be same to the crystal ligand
                    np.save(output_folder +"/"+ crystal_ligand_name,multiframe_ligand)
                    print "saved ligand:", crystal_ligand_name
                    test_load = np.load(output_folder +"/"+ crystal_ligand_name + ".npy")
                    print "loaded ligand", test_load.shape


                            #docked_ligand_file_path = os.path.abspath(str(crystal_ligand_file_path + "/" + pdb_name + ".npy"))

                    #try:
                    #    docked_ligand_molecule = molecule_class(docked_ligand_file_path)
                    #    crystal_ligand_molecule = molecule_class(crystal_ligand_file_path)
                    #    print docked_ligand_file_path,crystal_ligand_file_path
                    #except Exception:
                     #   print "EXCEPTION"
                     #   pass

    preprocess_ligands_in_folder(docked_dir=database_path + "/docked_ligands",cryst_dir=database_path + "/crystal_ligands")

"""



#    class av4_container:
#        """Stores the information about the interacting protein and ligand. Writes itself to disk.
#        Raises exception when several proteins are added in one record.
#        Raises exception when ligand with the wrong atom order/number is added to the record
#        """
#        receptor_elements = np.array([],dtype=np.float32)
#        receptor_coords = np.array([[]],dtype=np.float32)
#        # ligands is stored as list of elements,
#        # and coordinate frames are stacked horizontally
#        ligand_elements = np.array([],dtype=np.float32)
#        ligand_coordinates = np.array([[]], dtype=np.float32)
#        labels = np.array([], dtype=np.float32)
#
#        def add_ligand(self,prody_ligand,label):
#            if not labels:
#                self.ligand_elements = prody_ligand.getCords()
#                self.ligand_coords = prody_ligand.
#            if not all(np.asarray(self.ligand.getElements()) == np.asarray(prody_ligand.getElements())):
#                raise Exception('attempting to add ligand with different order of atoms')
#            self.frames = np.hstack((self.frames,prody_ligand.getCords))
 #           self.labels = np.append(self.labels, label)
#
 #       def save(self, folder_path):
  #          pass



#database_path = "../datasets/labeled_npy"
#output_folder = "../datasets/experimental_format"
# preprocess_receptors_to_npy(rec_dir=database_path + "/receptors")
#convert_npy_to_av3(database_path,output_folder)


def prepare_labeled_pdb():
    preprocess_PDB_to_npy(database_path='../datasets/test_hydro')
    write_database_index_file(database_path='../datasets/test_hydro',database_index_path='../datasets/test_hydro',lig_dirs=["crystal_ligands","docked_ligands"])
    split_into_train_and_test_sets(database_index_path='../datasets/test_hydro',train_set_div=0.95,replicate_positives=20)