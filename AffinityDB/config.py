import os
import sys
from collections import namedtuple

"""
Parameter
"""

# only ligands' atom number above threshold will saved
heavy_atom_threshold = 0
# number of process running at the same time
process_num = 4

# RENAMES:
# [ ] tanimoto_cutoff    # minimum Tanimoto similarity score to be considered 
# [ ] clash_size_cutoff  # percentage_of_overlapping_atoms # no cutoff here!
# [x] base               # database_root
# [x] lig_download_path  # why ?

# [x] downloads          # raw_pdbs
# [x] receptors          # raw_receptors
# [x] raw_ligands
# [-] docked_ligand_path # SMINA_DOCKED ?? 
# [x] docking parameters
 
# [ ] target_list_file     list_of_PDBs_to_download
# [x] lig_target_list_file (Not needed)
 
# [x] add folder for repaired ligands and proteins (with H)
# [ ] add folder for minimized hydrogens on ligands and proteins (How does the hydrogen addition happen)
# [x] think how we can deal with multiple docking parameters




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
database_root = '/home/xander/affinityDB/test'

# pdb download from Protein DataBank
pdb_download_path = os.path.join(database_root,'data','row_pdb')
# splited receptor
splited_receptors_path = os.path.join(database_root,'data','row_receptors')
# splited ligands
splited_ligands_path = os.path.join(database_root,'data','row_ligands')
# log files
log_folder = os.path.join(database_root,'log')
# docked ligands
docked_ligand_path = os.path.join(database_root,'data','docked')

# ligands docked by smina , scoring function: default
smina_docked_path = os.path.join(database_root, 'data','smina')
# ligands docked by smina , scoring function: vinardo                       # will c
vinardo_docked_path = os.path.join(database_root, 'data','vinardo')

# ligans output by smina, keep the same atom order as docked result
smina_std_path = os.path.join(database_root, 'data', 'smina_std_ligands')


"""
File Path 
"""

# path of smina binary file
smina = '/home/xander/Program/smina/smina.static'


# pdb_target_list
#target_list_file = os.path.join(sys.path[0],'target_list','main_pdb_target_list.txt')
target_list_file = '/home/xander/affinityDB/target_list/main_pdb_target_list.txt'

# example scoring
scoring_terms = os.path.join(sys.path[0], 'scoring', 'smina.score')


"""
docking para
"""

vinardo_pm = {
    'arg': [],
    'kwarg' : {
    'autobox_add':12,
    'num_modes':400,
    'exhaustiveness':64,
    'scoring':'vinardo',
    'cpu':1
    }
}

smina_pm = {
    'arg':[],
    'kwarg':{
    'num_modes':400,
    'cpu':1
    }
}