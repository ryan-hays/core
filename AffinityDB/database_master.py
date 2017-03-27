"""
Processing data
"""

import multiprocessing
import os
import sys
import argparse
from glob import glob
import pandas
from utils import mkdir, log
import openbabel
import numpy as np

import config


FLAGS = None

def download_pdb(pdb_name):
    """
    Download pdb file from rcsb
    """
    if not os.path.exists(os.path.join(config.pdb_download_path, pdb_name + '.pdb')):
        download_address = 'https://files.rcsb.org/download/{}.pdb'.format(pdb_name)
        os.system('wget -P {}  {}'.format(config.pdb_download_path, download_address))
        print "Download ", pdb_name

def split_structure(pdb_path):
    """
    Split downloaded pdb into receptor and ligand
    Record pdb's resolution
    Record ligand's heavy atom num
    """
    pdb_name = os.path.basename(pdb_path).split('.')[0].lower()

    try:
        parsed = prody.parsePDB(pdb_path)
    except Exception as e:
        log('parse_failed.log', '{},{}'.format(pdb_name, str(e)))
        return

    try:
        header = prody.parsePDBHeader(pdb_path)
        log('resolution.csv', '{},{}'.format(pdb_name, header['resolution']))
    except Exception as e:
        log('parse_header_failed.log', '{},{}'.format(pdb_name, str(e)))

    hetero = parsed.select(
        '(hetero and not water) or resname ATP or resname ADP or resname AMP or resname GTP or resname GDP or resname GMP')

    receptor = parsed.select('protein or nucleic')
    if receptor is None:
        log("select_failed.log", "{},doesn't have receptor.\n".format(pdb_name))
        return
    if hetero is None:
        log("select_failed.log", "{},doesn't have ligand.\n".format(pdb_name))
        return

    # write ligand into file
    for each in prody.HierView(hetero).iterResidues():
        ResId = each.getResindex()
        ResName = each.getResname()
        ligand_path = os.path.join(config.splited_ligands_path, pdb_name, "{}_{}_{}_ligand.pdb".format(pdb_name, ResName, ResId))
        mkdir(os.path.dirname(ligand_path))
        prody.writePDB(ligand_path, each)
        atom_num = each.select('not hydrogen').numAtoms()
        log('atom_num.csv', "{}_{}_{},{}".format(pdb_name, ResName, ResId, atom_num))



    receptor_path = os.path.join(config.splited_receptors_path, pdb_name + '.pdb')
    prody.writePDB(receptor_path, receptor)
    log('success_ligand.log', '{} success'.format(pdb_name))


def vinardo_dock_ligand(ligand_path):
    """
    docking ligand by smina with vinardo
    reorder the ligands' atom by smina
    """
    ligand_path = ligand_path.strip()
    receptor_name = os.path.basename(ligand_path).split('_')[0]
    receptor_path = os.path.join(config.splited_receptors_path, receptor_name + '.pdb')

    docked_ligand_name = os.path.basename(ligand_path).replace('ligand', 'vinardo')
    docked_ligand_path = os.path.join(config.vinardo_docked_path, receptor_name, docked_ligand_name)

    std_ligand_path = ligand_path.replace(config.splited_ligands_path, config.smina_std_path)

    if not os.path.exists(std_ligand_path):
        cmd = '{} -r {} -l {} --scoring vinardo --score_only -o {}'\
            .format(config.smina, receptor_path, ligand_path, std_ligand_path)
        mkdir(os.path.dirname(std_ligand_path))
        os.system(cmd)

    if not os.path.exists(docked_ligand_path):
        cmd = '{} -r {} -l {} --autobox_ligand {} --autobox_add 12 -o {} --num_modes=400 --exhaustiveness 64 --scoring vinardo --cpu=1 '\
            .format(config.smina, receptor_path, ligand_path, ligand_path, docked_ligand_path)
        #print cmd
        mkdir(os.path.dirname(vinardo_docked_path))
        os.system(cmd)

def count_rotable_bond(ligand_path):
    """
    count the number of rotable bond in ligand
    """
    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]

    obConversion = openbabel.OBConversion()
    OBligand = openbabel.OBMol()
    rot_bond = 0

    if not obConversion.ReadFile(OBligand, ligand_path):
        log("rotbond_failed.log", "{}, cannot parse by openbabel".format(ligand_name))
        return

    for bond in openbabel.OBMolBondIter(OBligand):
        if  not bond.IsSingle() or bond.IsAmide() or bond.IsInRing():
            continue
        elif bond.GetBeginAtom().GetValence() == 1 or bond.GetEndAtom().GetValence() == 1:
            continue
        else:
            rot_bond += 1

    log("rotbond.log","{},{}".format(ligand_name, rot_bond))

def run(target_list, func):
    
    pool = multiprocessing.Pool(config.process_num)
    pool.map_async(func, target_list)
    pool.close()
    pool.join()



def main():

    if FLAGS.download:
        print "Downloading pdb from rcsb..."
        download_list = open(config.target_list_file).readline().strip().split(', ')
        mkdir(config.pdb_download_path)
        run(download_list, download_pdb)
    if FLAGS.split:
        print "Spliting receptor and ligand..."
        structure_list = glob(os.path.join(config.pdb_download_path,'*.pdb'))
        run(structure_list, split_structure)
    if FLAGS.vinardo:
        print "Smina vinardo docking..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, vinardo_dock_ligand)

    if FLAGS.rotbond:
        print "Counting rotable bonds..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, count_rotable_bond)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Database Master Option")

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='stroe_true')
    parser.add_argument('--vinardo', action='store_true')
    parser.add_argument('--rotbond', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    main()
