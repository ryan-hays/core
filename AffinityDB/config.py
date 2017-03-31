import os
import sys


"""
Parameter
"""

# only ligands' atom number above threshold will saved
heavy_atom_threshold = 0
# number of process running at the same time
process_num = 4


# RENAMES:
# tanimoto_cutoff    # minimum Tanimoto similarity score to be considered 
# clash_size_cutoff  # percentage_of_overlapping_atoms # no cutoff here!
# base               # database_root
# lig_download_path  # why ?

# downloads          # raw_pdbs
# receptors          # raw_receptors
# raw_ligands
# docked_ligand_path # SMINA_DOCKED ?? 
# docking parameters

# target_list_file     list_of_PDBs_to_download
# lig_target_list_file (Not needed)

# add folder for repaired ligands and proteins (with H)
# add folder for minimized hydrogens on ligands and proteins (How does the hydrogen addition happen)
# think how we can deal with multiple docking parameters


"""
overlap detaction constant
"""
tanimoto_cutoff = 0.75  # an exact Tanimoto similarity score should be recorded in the file
clash_cutoff_A = 4      # 
clash_size_cutoff = 0.3 # __ an exact value should be recorded


"""
Folders
"""
#base folder for all the output
base = '/home/xander/affinityDB/test'
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

# docked ligands by smina scoring function: vinardo                       # will c
vinardo_docked_path = os.path.join(base, 'data','vinardo')

# ligans output by smina, keep the same atom order as docked result
smina_std_path = os.path.join(base, 'data', 'smina_std_ligands')


"""
File Path 
"""

# path of smina binary file
smina = '/home/xander/Program/smina/smina.static'


# pdb_target_list
#target_list_file = os.path.join(sys.path[0],'target_list','main_pdb_target_list.txt')
target_list_file = '/home/xander/affinityDB/target_list/main_pdb_target_list.txt'
# ligand_target_list
lig_target_list_file = os.path.join(sys.path[0], 'target_list', 'main_ligands_target_list.txt')

# example scoring
scoring_terms = os.path.join(sys.path[0], 'scoring', 'smina.score')

