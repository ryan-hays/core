"""
Processing data
"""

import multiprocessing
import os
import sys
import argparse
from glob import glob
from functools import partial
from utils import mkdir, log
import openbabel
import numpy as np
import prody
import config
import subprocess
import re
import pandas
import time

FLAGS = None

ligand_name_of = lambda x:os.path.basename(x).split('.')[0]
receptor_of = lambda x:os.path.basename(x).split('_')[0]
receptor_path_of = lambda x:os.path.join(config.splited_receptors_path,receptor_of(x)+'.pdb')

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
        log("select_failed.log", "{},doesn't have receptor.".format(pdb_name))
        return
    if hetero is None:
        log("select_failed.log", "{},doesn't have ligand.".format(pdb_name))
        return

    # write ligand into file
    for each in prody.HierView(hetero).iterResidues():
        ResId = each.getResindex()
        ResName = each.getResname()
        ligand_path = os.path.join(config.splited_ligands_path, pdb_name, "{}_{}_{}_ligand.pdb".format(pdb_name, ResName, ResId))
        mkdir(os.path.dirname(ligand_path))
        prody.writePDB(ligand_path, each)
        atom_num = each.select('not hydrogen').numAtoms()

        log('atom_num.csv', "{}_{}_{},{}".format(pdb_name, ResName, ResId, atom_num), head='ligand,atom_num')

    receptor_path = os.path.join(config.splited_receptors_path, pdb_name + '.pdb')
    prody.writePDB(receptor_path, receptor)

    log('success_split_pdb.log', '{},success'.format(pdb_name), head='pdb,status')


def reorder_ligand(ligand_path):
    """
    use smina to read and write ligands, 
    make sure the atom order is the same as docking result
    """

    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]
    receptor_name = os.path.basename(ligand_path).split('_')[0]
    receptor_path = os.path.join(config.splited_receptors_path, receptor_name + '.pdb')

    std_ligand_path = ligand_path.replace(config.splited_ligands_path, config.smina_std_path)

    if not os.path.exists(std_ligand_path):
        cmd = '{} -r {} -l {} --custom_scoring {} --score_only -o {}'\
            .format(config.smina, receptor_path, ligand_path, config.scoring_terms, std_ligand_path)
        mkdir(os.path.dirname(std_ligand_path))
        #print cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        cont = cl.communicate()[0].strip().split('\n')

        terms = [line.strip().split(' ')[2:] for line in cont if line.startswith('##')]
        head = '\t'.join(['ligand', 'position'] + terms[0])
        data = [ligand_name, '0'] + terms[1]
        log('crystal_ligands_term_score.tsv', '\t'.join(data), head=head)


def vinardo_dock_ligand(ligand_path):
    """
    docking ligand by smina with vinardo
    """
    ligand_path = ligand_path.strip()
    receptor_name = os.path.basename(ligand_path).split('_')[0]
    receptor_path = os.path.join(config.splited_receptors_path, receptor_name + '.pdb')

    docked_ligand_name = os.path.basename(ligand_path).replace('ligand', 'vinardo')
    docked_ligand_path = os.path.join(config.vinardo_docked_path, receptor_name, docked_ligand_name)

    if not os.path.exists(docked_ligand_path):
        cmd = '{} -r {} -l {} --autobox_ligand {} --autobox_add 12 -o {} --num_modes=400 --exhaustiveness 64 --scoring vinardo --cpu=1 '\
            .format(config.smina, receptor_path, ligand_path, ligand_path, docked_ligand_path)

        mkdir(os.path.dirname(docked_ligand_path))
        os.system(cmd)

def crystal_ligand_for_same_receptor(ligand_path):
    """
    get the ligands splited from same pdb
    note ligands should be reordered by smina
    """
    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_ligand = os.path.join(config.smina_std_path, receptor, '_'.join([receptor, lig, resid, 'ligand.pdb']))

    ligands_list = glob(os.path.join(os.path.dirname(crystal_ligand), '*.pdb'))


    other_ligands = list(set(ligands_list) - set([crystal_ligand]))
    return crystal_ligand, other_ligands

def get_similar_ligands(ligand_path, ligands_for_same_receptor):
    """
    calculate tanimoto similarity
    return the ligands with high tanimoto similarity
    """
    similar_ligands = []
    for lig_path in ligands_for_same_receptor:
        cmd = 'babel -d {} {} -ofpt '.format(ligand_path, lig_path)
        ls = os.popen(cmd).read()
        tanimoto_similarity = re.split('=|\n', ls)[2]
        log('tanimoto_similarity.csv', '{},{},{}'.format(ligand_name_of(ligand_path), ligand_name_of(lig_path), tanimoto_similarity), head='lig_a,lig_b,tanimoto')
        if tanimoto_similarity > config.tanimoto_cutoff:
            similar_ligands.append(lig_path)

    return similar_ligands

def calculate_clash(docked_ligand, crystal_ligand):
    """
    for all position in docked_ligand
    calculate if they are clash with crystal ligand
    """

    try:
        docked = prody.parsePDB(docked_ligand).getCoordsets()
    except Exception as e:
        log("overlap_failed.log", "{},{}".format(ligand_name_of(docked_ligand), str(e)), head='ligand,exception')
        return 
    try:        
        crystal = prody.parsePDB(crystal_ligand).getCoords()
    except Exception as e:
        log("overlap_failed.log", "{},{}".format(ligand_name_of(crystal_ligand), str(e)), head='ligand,exception')

    expanded_docked = np.expand_dims(docked, -2)
    diff = expanded_docked - crystal
    distance = np.sqrt(np.sum(np.power(diff, 2), axis=-1))
    all_clash = (distance < config.clash_cutoff_A).astype(float)
    atom_clash = (np.sum(all_clash, axis=-1) > 0).astype(float)
    position_clash = np.mean(atom_clash, axis=-1) > config.clash_size_cutoff

    clash_ratio = sum(position_clash.astype(float))/len(position_clash)
    clash = position_clash.astype(int).astype(str)
    log("single_overlap.tsv", '\t'.join([ligand_name_of(docked_ligand), ligand_name_of(crystal_ligand), str(clash_ratio), ','.join(clash)]), head='\t'.join(['docked_ligand','crystal_ligand','overlap_ratio','overlap_status']))
    return position_clash



def detect_overlap(ligand_path):
    """
    if docked result overlap with other crystal ligands
    splited from the same receptor
    note crystal ligand should reordered by smina
    """

    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]

    crystal_ligand, other_ligands = crystal_ligand_for_same_receptor(ligand_path)

    if len(other_ligands) == 0:
        log("overlap_receptor_skip.log","{}, has only one ligand".format(receptor_of(ligand_path)), head='receptor,exception')
        return
    similar_ligands = get_similar_ligands(crystal_ligand, other_ligands)
    if len(similar_ligands) == 0:
        log("overlap_ligand_skip.log","{}, doesn't have similar ligands".format(ligand_name), head='ligand,exception')

    position_clash = None
    for lig_path in similar_ligands:
        cur_position_clash = calculate_clash(ligand_path, lig_path)
        if position_clash == None:
            position_clash = cur_position_clash
        else:
            position_clash += cur_position_clash

    clash = list(position_clash.astype(int).astype(str))
    clash_ratio = sum(position_clash.astype(float))/len(position_clash)

    log("overlap.tsv", '\t'.join([ligand_name, str(clash_ratio), ','.join(clash)]), head='\t'.join(['ligand','overlap_ratio','overlap_status']))



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
        log("rotbond_failed.log", "{}, cannot parse by openbabel".format(ligand_name), head='ligand,exception')
        return

    for bond in openbabel.OBMolBondIter(OBligand):
        if  not bond.IsSingle() or bond.IsAmide() or bond.IsInRing():
            continue
        elif bond.GetBeginAtom().GetValence() == 1 or bond.GetEndAtom().GetValence() == 1:
            continue
        else:
            rot_bond += 1

    log("rotbond.csv","{},{}".format(ligand_name, rot_bond), head='ligand,rotbond')


def calculate_rmsd(ligand_path):
    """
    calculate rmsd between docked ligands and crystal ligand
    note when calculate rmsd, crystal ligand should be reordered by smina
    """

    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_ligand_path =os.path.join(config.smina_std_path, receptor, '_'.join([receptor, lig, resid, 'ligand.pdb']))

    try:
        docked_coords = prody.parsePDB(ligand_path).getCoordsets()
        crystal_coord = prody.parsePDB(crystal_ligand_path).getCoords()
        rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=-1), axis=-1))

        rmsd_str = ','.join(rmsd.astype(str))
        log('rmsd.tsv', '\t'.join([ligand_name, rmsd_str]))

    except Exception as e:
        log('rmsd_failed.csv','{},{}'.format(ligand_name, str(e)), head='ligand,exception')
        return


def calculate_native_contact(mutex, ligand_path):
    """
    calculate native contact between the ligand and receptor
    """
    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    
    receptor_path = receptor_path_of(ligand_path)

    try:
        parsed_ligands = prody.parsePDB(ligand_path).select('not hydrogen')
        parsed_receptor = prody.parsePDB(receptor_path).select('not hydrogen')
        
        ligand_atom_num = parsed_ligands.numAtoms()

        ligands_coords = parsed_ligands.getCoordsets()
        receptor_coord = parsed_receptor.getCoords()
        
        exp_ligands_coords = np.expand_dims(ligands_coords, -2)
        diff = exp_ligands_coords - receptor_coord
        distance = np.sqrt(np.sum(np.square(diff),axis=-1))

        num_native_contact = None
        ratio_native_contact = None
        start = time.time()
        
        for threshold in np.linspace(4,8,9):
            contact = ( distance < threshold ).astype(int)
            num_contact = np.sum(contact, axis=(-1,-2))
            ratio_contact = 1.0 * num_contact / ligand_atom_num

            if num_native_contact is None:
                num_native_contact = num_contact
            else:
                num_native_contact = np.dstack((num_native_contact, num_contact))
            if ratio_native_contact is None:
                ratio_native_contact = ratio_contact
            else:
                ratio_native_contact = np.dstack((ratio_native_contact, ratio_contact))

        # after dstack shape become [1, x, y]
        num_native_contact = num_native_contact[0]
        ratio_native_contact = ratio_native_contact[0]
        mutex.acquire()
        
        head = ['ligand','heavy_atom_num', 'position'] + list(np.linspace(4,8,9).astype(str))
        data = []
        for i in range(len(num_native_contact)):
            data.append(','.join([ligand_name, str(ligand_atom_num)  ,str(i+1)]+ list(num_native_contact[i].astype(int).astype(str)))+'\n')
            #log("native_contact.csv", ','.join([ligand_name, str(ligand_atom_num)  ,str(i+1)]+ list(num_native_contact[i].astype(int).astype(str))), head=','.join(head))
            #log("ratio_native_contact.csv", ','.join([ligand_name, str(i+1)]+ list(ratio_native_contact[i].astype(str))), head=','.join(head))
        log('native_contact.csv', data, head= ','.join(head))
        print time.time() - start
        mutex.release()
            
    except Exception as e:
        mutex.acquire()
        log('native_contact_failed.csv','{},{},{}'.format(receptor_of(ligand_name), ligand_name, str(e)), head=','.join(['receptor','ligand','exception']))
        mutex.release()


def run(target_list, func):
    
    pool = multiprocessing.Pool(config.process_num)
    pool.map_async(func, target_list)
    pool.close()
    pool.join()



def main():
    
    m = multiprocessing.Manager()
    l = m.Lock()

    if FLAGS.download:
        print "Downloading pdb from rcsb..."
        download_list = open(config.target_list_file).readline().strip().split(', ')
        mkdir(config.pdb_download_path)
        run(download_list, download_pdb)
    if FLAGS.split:
        print "Spliting receptor and ligand..."
        structure_list = glob(os.path.join(config.pdb_download_path, '*.pdb'))
        mkdir(config.splited_receptors_path)
        run(structure_list, split_structure)
    if FLAGS.reorder:
        print "Reorder ligands atom..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, reorder_ligand)
    if FLAGS.vinardo:
        print "Smina vinardo docking..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, vinardo_dock_ligand)

    if FLAGS.rotbond:
        print "Counting rotable bonds..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, count_rotable_bond)

    if FLAGS.overlap:
        print "Overlap detecting..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path, '*', '*.pdb'))
        #ligands_list = ['/home/xander/affinityDB/data/vinardo/1a0b/1a0b_ZN_117_vinardo.pdb']
        run(ligands_list, detect_overlap)

    if FLAGS.rmsd:
        print "Calculating rmsd..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path, '*', '*.pdb'))
        run(ligands_list, calculate_rmsd)

    if FLAGS.contact:
        print "Calculating native contacting..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path, '*', '*.pdb'))
        run(ligands_list, partial(calculate_native_contact, l))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Database Master Option")

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--vinardo', action='store_true')
    parser.add_argument('--rotbond', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--rmsd', action='store_true')
    parser.add_argument('--contact', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    main()
