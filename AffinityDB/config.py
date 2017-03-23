import os

#base folder for all the output
base = '/home/ubuntu/xiao/data/AffinityDB'
# folder to store pdb download from Protein DataBank
pdb_download_path = os.path.join(base,'data','downloads')
# folder to store splited receptor
splited_receptors_path = os.path.join(base,'data','receptors')
# folder to store splited ligands
splited_ligands_path = os.path.join(base,'data','ligands')
# folder to store log files
log_folder = os.path.join(base,'log')
# folder to store docked ligands
docked_ligand_path = os.path.join(base,'data','docked')

# only ligands' atom number above threshold will saved
heavy_atom_threshold = 0

# path of smina binary file
smina = ''
# number of process running at the same time
process_num = 4

# pdb_target_list
target_list_file = os.path.join(os.path.abspath('.'),'target_list','main_pdb_target_list.txt')