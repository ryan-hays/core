import os,sys
import config
from util import log,mkdir
from glob import glob


def dock_ligand(ligand_path):
    receptor_name = os.path.basename(ligand_path).split('_')[0]
    receptor_path = os.path.join(config.splited_receptors_path, receptor_name + '.pdb')

    docked_ligand_name = os.path.basename(ligand_path).replace('ligand','ligand_docked')
    docked_ligand_path = os.path.join(config.docked_ligand_path, receptor_name, docked_ligand_name)

    if not os.path.exists(docked_ligand_path):
        cmd = '{} -r {} -l {} --autobox_ligand {} -o {} --num_modes=1000 --energy_range=100 --cpu=1 '\
            .format(config.smina,receptor_path,ligand_path,ligand_path,docked_ligand_path)

        mkdir(os.path.dirname(docked_ligand_path))
        os.system(cmd)



def docking():
    ligands_list = glob(config.splited_ligands_path)
    map(dock_ligand,ligands_list)
