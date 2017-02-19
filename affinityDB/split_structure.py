import os,sys
import prody
import getopt
import numpy as np
import multiprocessing
from glob import glob
import config
from util import log,mkdir

def split_structure(pdb_path):
    pdb_name = os.path.basename(pdb_path).split('.')[0].lower()

    try:
        parsed = prody.parsePDB(pdb_path)
    except Exception as e:
        log('parse_failed.log','{},{}\n'.format(pdb_name,str(e)))
        return

    hetero = parsed.select(
        '(hetero and not water) or resname ATP or resname ADP or sesname AMP or resname GTP or resname GDP or resname GMP')
    receptor = parsed.select('protein or nucleic')
    if receptor is None:
        log("select_failed.log","{},doesn't have receptor.\n".format(pdb_name))
        return
    if hetero is None:
        log("select_failed.log","{},doesn't have ligand.\n".format(pdb_name))
        return


    # write ligand into file
    ligand_flags = False
    for each in prody.HierView(hetero).iterResidues():
        if each.select('not hydrogen').numAtoms() < config.heavy_atom_threshold:
            continue
        else:
            ligand_flags = True
            ResId = each.getResindex()
            ligand_path = os.path.join(config.splited_ligand_folder, pdb_name, "{}_{}_ligand.pdb".format(pdb_name, ResId))
            mkdir(os.path.dirname(ligand_path))
            prody.writePDB(ligand_path, each)

    # if have valid ligand, write down receptor
    if ligand_flags:
        receptor_path = os.path.join(config.splited_receptor_folder, pdb_name + '.pdb')
        prody.writePDB(receptor_path, receptor)
    else:
        log("threshold_failed.log","{}, no ligand above threshold {}.\n".format(pdb_name,config.heavy_atom_threshold))


def split():
    pdb_list = glob(config.pdb_download_path)
    map(split_structure,pdb_list)
