import os
#base folder for all the output
base = './'
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

# when select ligand only the one have no less than 10 atoms remain
heavy_atom_threshold = 10

# path of smina binary file
smina = ''
# number of process running at the same time
process_num = 4