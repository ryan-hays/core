import os
import sys

#base folder for all the output
base = '/n/scratch2/xl198/affinity/affinityDB'
# ligands download from Protein DataBank
lig_download_path = os.path.join(base,'data','lig_downloads')
# pdb download from Protein DataBank
pdb_download_path = os.path.join(base,'data','downloads')
# splited receptor
splited_receptors_path = os.path.join(base,'data','receptors')
# splited ligands
splited_ligands_path = os.path.join(base,'data','ligands')
# log files
log_folder = os.path.join(base,'log')
# docked ligands
docked_ligand_path = os.path.join(base,'data','docked')

# docked ligands by smina scoring function: vinardo
vinardo_docked_path = os.path.join(base, 'data','vinardo')

# ligans output by smina, keep the same atom order as docked result
smina_std_path = os.path.join(base, 'data', 'smina_std_ligands')
# only ligands' atom number above threshold will saved
heavy_atom_threshold = 0

# path of smina binary file
smina = ''
# number of process running at the same time
process_num = 4

# pdb_target_list
target_list_file = os.path.join(sys.path[0],'target_list','main_pdb_target_list.txt')

# ligand_target_list
lig_target_list_file = os.path.join(sys.path[0], 'target_list', 'main_ligands_target_list.txt')
