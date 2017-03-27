'''
load PDB download from RCSB
split it into receptor and hetero
if hetero's heavy atom more than threshold
save the receptor and hetero ligand
'''

import os,sys
import prody
import getopt
import numpy as np
import multiprocessing
from glob import glob
import config
from utils import log,mkdir



def split_structure(pdb_path):
    pdb_name = os.path.basename(pdb_path).split('.')[0].lower()

    try:
        parsed = prody.parsePDB(pdb_path)
    except Exception as e:
        log('parse_failed.log','{},{}'.format(pdb_name,str(e)))
        return
    
    try:
        header = prody.parsePDBHeader(pdb_path)
        log('resolution.txt','{},{}'.format(pdb_name,header['resolution']))
    except Exception as e:
        log('parse_header_failed.log','{},{}'.format(pdb_name, str(e)))

    hetero = parsed.select(
        '(hetero and not water) or resname ATP or resname ADP or resname AMP or resname GTP or resname GDP or resname GMP')
   
    receptor = parsed.select('protein or nucleic')
    if receptor is None:
        log("select_failed.log","{},doesn't have receptor.\n".format(pdb_name))
        return
    if hetero is None:
        log("select_failed.log","{},doesn't have ligand.\n".format(pdb_name))
        return

    # write ligand into file
    for each in prody.HierView(hetero).iterResidues():
        ResId = each.getResindex()
        ResName = each.getResname()
        ligand_path = os.path.join(config.splited_ligands_path, pdb_name, "{}_{}_{}_ligand.pdb".format(pdb_name, ResName, ResId))
        mkdir(os.path.dirname(ligand_path))
        prody.writePDB(ligand_path, each) 

    receptor_path = os.path.join(config.splited_receptors_path, pdb_name + '.pdb')
    prody.writePDB(receptor_path, receptor)
    log('success_ligand.log','{} success'.format(pdb_name))

def split(target_list):

    mkdir(config.splited_receptors_path)
    pool = multiprocessing.Pool(config.process_num)
    pool.map_async(split_structure,target_list)
    pool.close() 
    pool.join()
   
    
    #map(split_structure,target_list)

if __name__ == '__main__':

    target_list = glob(os.path.join(config.pdb_download_path,'*.pdb'))
    print "target ",len(target_list)
    args = sys.argv
    if len(args)>2:
        a = int(args[1])
        b = int(args[2])
        parts = np.linspace(0,len(target_list),int(a)+1).astype(int)
        target_list = target_list[parts[b-1]:parts[b]]

    split(target_list)
