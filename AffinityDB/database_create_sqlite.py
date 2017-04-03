"""
Processing data
"""

import multiprocessing
import os
import sys
import argparse
from glob import glob
from functools import partial
from utils import log, smina_param, timeit, count_lines
import openbabel
import numpy as np
import prody
import config
from config import lock
import subprocess
import re
import pandas
import time
import sqlite3
from profilehooks import profile
from db import database

FLAGS = None

db = database()


mkdir = lambda path: os.system('mkdir -p {}'.format(path))
ligand_name_of = lambda x:os.path.basename(x).split('.')[0]
receptor_of = lambda x:os.path.basename(x).split('_')[0]
receptor_path_of = lambda x:os.path.join(config.splited_receptors_path, receptor_of(x)+'.pdb')




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
        data = [pdb_name, 0, str(e)]
        data = [data]
        db.insert_or_replace('split_state',data)

        #print 'insert'
        log('parse_failed.log',
            '{},{}'.format(pdb_name, str(e)))
        return

    try:
        
        header = prody.parsePDBHeader(pdb_path)
        data = [ pdb_name, header['experiment'], header['resolution']]
        data = [data]
        db.insert_or_replace('resolution',data)
        log('resolution.csv',
            '{},{}'.format(pdb_name, header['resolution']),
            head='pdb,resolution')

        
    except Exception as e:
        data = [pdb_name, 0, str(e)]
        data = [data]
        db.insert_or_replace('split_state',data, head)
        log('parse_header_failed.log',
           '{},{}'.format(pdb_name, str(e)))

    hetero = parsed.select(
        '(hetero and not water) or resname ATP or resname ADP or resname AMP or resname GTP or resname GDP or resname GMP')

    receptor = parsed.select('protein or nucleic')
    if receptor is None:
        data = [pdb_name, 0, "No receptor found"]
        data = [data]
        db.insert_or_replace('split_state',data)
        log("select_failed.log", "No receptor found")
        return
    if hetero is None:
        data = [pdb_name, 0, "No ligand found"]
        data = [[data]]
        db.insert_or_replace('SPLIT_STATE',data, head)
        log("select_failed.log", "No ligand found")
        return

    # write ligand into file
    for each in prody.HierView(hetero).iterResidues():
        ResId = each.getResindex()
        ResName = each.getResname()
        ligand_path = os.path.join(config.splited_ligands_path,
                                   pdb_name,
                                   "{}_{}_{}_ligand.pdb".format(pdb_name, ResName, ResId))
        mkdir(os.path.dirname(ligand_path))
        prody.writePDB(ligand_path, each)
        atom_num = each.select('not hydrogen').numAtoms()

        data = ["{}_{}_{}".format(pdb_name, ResName, ResId), 'ligand', atom_num]
        data = [data]
        db.insert_or_replace('atom_num',data)
        log('atom_num.csv',
            "{}_{}_{},{}".format(pdb_name, ResName, ResId, atom_num),
            head='ligand,atom_num')

    receptor_path = os.path.join(config.splited_receptors_path, pdb_name + '.pdb')
    prody.writePDB(receptor_path, receptor)
    
    receptor_atom_num = receptor.select('not hydrogen').numAtoms()
    data = [pdb_name, 'receptor', receptor_atom_num]
    data = [data]
    db.insert_or_replace('atom_num', data)

    data = [pdb_name, 1, 'success']
    data = [data]
    db.insert_or_replace('split_state',data)
    log('success_split_pdb.log',
        '{},success'.format(pdb_name),
        head='pdb,status')

def reorder_ligand(input_dir, output_dir, smina_pm, ligand_path):
    """
    smina output file change the order of atom,
    inorder to get correct rmsd value, we just read the
    crystal ligand and write it by smina to keep the 
    atom order the same as docking result
    """

    lignad_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    receptor_path = receptor_path_of(ligand_path)
    receptor_name = receptor_of(ligand_path)

    out_ligand_path = ligand_path.replace(input_dir, output_dir)

    kw = {
        'receptor':receptor_path,
        'ligand'  :ligand_path,
        'out'     :out_ligand_path
    }

    if not os.path.exists(out_ligand_path):
        mkdir(os.path.dirname(out_ligand_path))
        cmd = smina_pm.make_command(**kw)
        
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        print cmd
        cl.wait()
        cont = cl.communicate()[0].strip().split('\n')

        terms = [line.strip().split(' ')[2:] for line in cont if line.startswith('##')]
        head = map(lambda x:x.replace(',',' '),terms[0])

        head = ['ligand','position'] + map(lambda x:'"%s"' % x, head)
        data = [ligand_name, 0] + map(lambda x:float(x),terms[1])
        data = [data]
        db.insert_or_replace('scoring_terms',data,head)


def get_similar_ligands(ligand_path,finger_print='FP4'):
    """
    calculate tanimoto similarity
    return the ligands with high tanimoto similarity
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_ligand = os.path.join(config.smina_std_path, receptor, '_'.join([receptor, lig, resid, 'ligand.pdb']))

    ligands_list = glob(os.path.join(os.path.dirname(crystal_ligand), '*.pdb'))

    ligands_for_same_receptor = list(set(ligands_list) - set([crystal_ligand]))



    similar_ligands = []
    for lig_path in ligands_for_same_receptor:
        cmd = 'babel -d {} {} -ofpt -xf{}'.format(ligand_path, lig_path, finger_print)
        ls = os.popen(cmd).read()
        tanimoto_similarity = re.split('=|\n', ls)[2]
        print tanimoto_similarity

        lig_pair = [ligand_name_of(ligand_path), ligand_name_of(lig_path)]
        lig_pair = lig_pair if lig_pair[0]<lig_pair[1] else [lig_pair[1],lig_pair[0]]
        data = lig_pair + [finger_print, tanimoto_similarity]
        data = [data]
        db.insert_or_replace('similarity',data)
        log('tanimoto_similarity.csv',
            '{},{},{}'.format(ligand_name_of(ligand_path), ligand_name_of(lig_path), tanimoto_similarity),
            head='lig_a,lig_b,finger_print, tanimoto')
        if tanimoto_similarity > config.tanimoto_cutoff:
            similar_ligands.append([lig_path, tanimoto_similarity, finger_print])

    return crystal_ligand, similar_ligands

def get_same_ligands(ligand_path):
    """
    get the path of ligands, they are the same ligand docked to the same receptor,
    but different binding site.
    :param ligand_path:
    :return:
    """
    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    receptor, lig, resid, _ = ligand_name.split('_')
    same_ligands = glob(os.path.join(config.smina_std_path,receptor, '{}_{}_*.pdb'.format(receptor, lig)))
    return same_ligands

def calculate_clash(docked_ligand, crystal_ligand, similarity):
    """
    for all position in docked_ligand
    calculate if they are clash with crystal ligand
    :param docked_ligand: path
    :param crystal_ligand: path
    :return:
    """


    try:
        docked = prody.parsePDB(docked_ligand).getCoordsets()
    except Exception as e:
        log("overlap_failed.log",
            "{},{}".format(ligand_name_of(docked_ligand), str(e)),
            head='ligand,exception')
        return 
    try:
        crystal = prody.parsePDB(crystal_ligand).getCoords()
    except Exception as e:
        log("overlap_failed.log",
            "{},{}".format(ligand_name_of(crystal_ligand), str(e)),
            head='ligand,exception')
        return

    expanded_docked = np.expand_dims(docked, -2)
    diff = expanded_docked - crystal
    distance = np.sqrt(np.sum(np.power(diff, 2), axis=-1))
    all_clash = (distance < config.clash_cutoff_A).astype(float)
    atom_clash = (np.sum(all_clash, axis=-1) > 0).astype(float)
    position_clash = np.mean(atom_clash, axis=-1) > config.clash_size_cutoff
    position_clash_ratio = np.mean(atom_clash, axis=-1)

    clash_ratio = sum(position_clash.astype(float))/len(position_clash)
    clash = position_clash.astype(int).astype(str)

    data = []
    for i,ratio in enumerate(position_clash_ratio):
        datum = [ligand_name_of(docked_ligand), ligand_name_of(crystal_ligand), i+1, ratio]
        data.append(datum)

    db.insert_or_replace('overlap',data)

    head = ['docked_ligand','crystal_ligand','similarity','positon','precent_of_overlaping_atom']
    data = []
    datum = [ligand_name_of(docked_ligand), ligand_name_of(crystal_ligand), str(similarity)]
    for i, pcr in enumerate(position_clash_ratio):
        data.append(','.join(datum+[str(i+1), str(pcr)]))
    
    
    log("ligand_pair_overlap.csv",
        data,
        head = ','.join(head))
    
 #   log("single_overlap.tsv",
 #       '\t'.join([ligand_name_of(docked_ligand), ligand_name_of(crystal_ligand), str(clash_ratio), ','.join(clash)]),
 #       head='\t'.join(['docked_ligand','crystal_ligand','overlap_ratio','overlap_status']),
 #       lock=lock)
    return position_clash

def detect_overlap(ligand_path):
    """
    if docked result overlap with other crystal ligands
    splited from the same receptor
    note crystal ligand should reordered by smina
    :param ligand_path: path
    :return:
    """

    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]


    crystal_ligand, similar_ligands = get_similar_ligands(ligand_path)
    if len(similar_ligands) == 0:
        log("overlap_ligand_skip.log",
            "{}, doesn't have similar ligands".format(ligand_name),
            head='ligand,exception')

    position_clash = None
    for lig_path, similarity, finger_print in similar_ligands:
        cur_position_clash = calculate_clash(ligand_path, lig_path, similarity)
        if position_clash == None:
            position_clash = cur_position_clash
        else:
            position_clash += cur_position_clash

    #print similar_ligands
    clash = list(position_clash.astype(int).astype(str))
    clash_ratio = sum(position_clash.astype(float))/len(position_clash)

    
    log("overlap.tsv",
        '\t'.join([ligand_name, str(clash_ratio), ','.join(clash)]),
        head='\t'.join(['ligand','overlap_ratio','overlap_status']))

def count_rotable_bond(ligand_path):
    """
    count the number of rotable bond in ligand
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]

    obConversion = openbabel.OBConversion()
    OBligand = openbabel.OBMol()
    rot_bond = 0

    if not obConversion.ReadFile(OBligand, ligand_path):
        log("rotbond_failed.log",
            "{}, cannot parse by openbabel".format(ligand_name),
            head='ligand,exception')
        return

    for bond in openbabel.OBMolBondIter(OBligand):
        if  not bond.IsSingle() or bond.IsAmide() or bond.IsInRing():
            continue
        elif bond.GetBeginAtom().GetValence() == 1 or bond.GetEndAtom().GetValence() == 1:
            continue
        else:
            rot_bond += 1


    data = [ligand_name, rot_bond]
    data = [data]
    db.insert_or_replace('rotable_bond',data)
    log("rotbond.csv",
        "{},{}".format(ligand_name, rot_bond),
        head='ligand,rotbond')

def calculate_rmsd(ligand_path):
    """
    calculate rmsd between docked ligands and crystal ligand
    note when calculate rmsd, crystal ligand should be reordered by smina
    :param ligand_path: path
    :return: 
    """

    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_ligands = get_same_ligands(ligand_path)

    try:
        docked_coords = prody.parsePDB(ligand_path).select('not hydrogen').getCoordsets()
        for crystal_ligand in crystal_ligands:

            crystal_coord = prody.parsePDB(crystal_ligand).select('not hydrogen').getCoords()
            rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=-1), axis=-1))

            
            data = []
            for i,rd in enumerate(rmsd):
                datum = [ligand_name, ligand_name_of(crystal_ligand), i+1, rd ]
                data.append(datum)

            db.insert_or_replace('rmsd',data)

            data = [ligand_name, ligand_name_of(crystal_ligand), 1, 'success']
            data = [data]
            db.insert_or_replace('rmsd_state',data)
            
            rmsd_str = ','.join(rmsd.astype(str))
            head = ['ligand','crystal_ligand','rmsd']

            log('rmsd.tsv',
                '\t'.join([ligand_name, ligand_name_of(crystal_ligand), rmsd_str]),
                head=','.join(head))

    except Exception as e:

        log('rmsd_failed.csv',
            '{},{}'.format(ligand_name, str(e)),
            head='ligand,exception')
        return

def calculate_native_contact(ligand_path):
    """
    calculate native contact between the ligand and receptor
    notice when calculating native contact, we ignore all hydrogen
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    print ligand_name
    
    receptor, res_name, res_id, _  = ligand_name.split('_')
    crystal_ligand_dir = os.path.dirname(ligand_path).replace('/docked_ligands/','/crystal_ligands/')
    crystal_ligand_path = os.path.join(crystal_ligand_dir,'_'.join([receptor, res_name, res_id, 'ligand.pdb']) )


    receptor_path = receptor_path_of(ligand_path)

    try:
        parsed_ligands = prody.parsePDB(ligand_path).select('not hydrogen')
        parsed_crystal = prody.parsePDB(crystal_ligand_path).select('not hydrogen')
        parsed_receptor = prody.parsePDB(receptor_path).select('not hydrogen')

        cry_atom_num = parsed_crystal.numAtoms()
        ligand_atom_num = parsed_ligands.numAtoms()

        assert cry_atom_num == ligand_atom_num

        ligands_coords = parsed_ligands.getCoordsets()
        crystal_coord = parsed_crystal.getCoords()
        receptor_coord = parsed_receptor.getCoords()

        exp_crystal_coord = np.expand_dims(crystal_coord, -2)
        cry_diff = exp_crystal_coord - receptor_coord
        cry_distance = np.sqrt(np.sum(np.square(cry_diff), axis=-1))
        
        exp_ligands_coords = np.expand_dims(ligands_coords, -2)
        lig_diff = exp_ligands_coords - receptor_coord
        lig_distance = np.sqrt(np.sum(np.square(lig_diff),axis=-1))

        num_native_contact = None
        
        for threshold in np.linspace(4,8,9):
            
            cry_contact = ( cry_distance < threshold ).astype(int)
            
            num_contact = np.sum(cry_contact).astype(float)

            lig_contact = (lig_distance < threshold).astype(int)

            contact_ratio = np.sum(cry_contact * lig_contact, axis=(-1,-2)) / num_contact

            

            if num_native_contact is None:
                num_native_contact = contact_ratio
            else:
                num_native_contact = np.dstack((num_native_contact, contact_ratio))

        # after dstack shape become [1, x, y]
        
        num_native_contact = num_native_contact[0]
        
        data = []
        for i, nts in enumerate(num_native_contact):
            datum = [ligand_name, i+1, nts[0], nts[1]]
            data.append(datum)

        
        db.insert_or_replace('native_contact',data)

        head = ['ligand','heavy_atom_num', 'position'] + list(np.linspace(4,8,9).astype(str))
        data = []
        for i in range(len(num_native_contact)):
            data.append(','.join([ligand_name, str(ligand_atom_num)  ,str(i+1)]+ list(num_native_contact[i].astype(int).astype(str))))
            #log("native_contact.csv", ','.join([ligand_name, str(ligand_atom_num)  ,str(i+1)]+ list(num_native_contact[i].astype(int).astype(str))), head=','.join(head))
            #log("ratio_native_contact.csv", ','.join([ligand_name, str(i+1)]+ list(ratio_native_contact[i].astype(str))), head=','.join(head))
        log('native_contact.csv', data, head= ','.join(head))

            
    except Exception as e:
        log('native_contact_failed.csv',
            '{},{},{}'.format(receptor_of(ligand_name), ligand_name, str(e)),
            head=','.join(['receptor','ligand','exception']))



def add_hydrogen(input_dir, output_dir, input_file):
    """
    add hydrogens to molecule
    """
    try:
        input_file = input_file.strip()
        output_file = input_file.replace(input_dir, output_dir)
        if not os.path.exists(os.path.dirname(output_file)):
            mkdir(os.path.dirname(output_file))
        if not os.path.exists(output_file):
            cmd = 'obabel -ipdb {} -opdb -O {} -h'.format(input_file, output_file)
            os.system(cmd)
    except Exception as e:
        print e


def smina_dock(input_dir, output_dir, smina_pm, ligand_path):
    """
    dock ligands
    """
    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    receptor_path = receptor_path_of(ligand_path)
    receptor_name = ligand_name.split('_')[0]

    docked_ligand_name = os.path.basename(ligand_path).replace('ligand', smina_pm.name)
    docked_ligand_path = os.path.join(output_dir, receptor_name, docked_ligand_name)

    kw = {
        'receptor':receptor_path,
        'ligand'  :ligand_path,
        'autobox_ligand' : ligand_path,
        'out'     :docked_ligand_path
    }

    if os.path.exists(docked_ligand_path):
        pass
    if not os.path.exists(docked_ligand_path):
        mkdir(os.path.dirname(docked_ligand_path))
        cmd = smina_pm.make_command(**kw)
        #print cmd
        os.system(cmd)
        
        c_cmd = 'wc -l'


def clean_empty_ligand(ligand_path):
    """
    for some reason, dock failed and nothing in the output file
    remove empty file, and change dock status to failure
    """

    ligand_path = ligand_path.strip()
    ligand_name = ligand_name_of(ligand_path)
    
    if not count_lines(ligand_path):
        cmd = 'rm %s' % ligand_path
        os.system(cmd)
        data = [ligand_name, 0, 'empty']
        db.insert_or_replace('dock_state',data)

  
@profile
def run(target_list, func):
    func(target_list[3])
    pool = multiprocessing.Pool(config.process_num)
    pool.map_async(func, target_list)
    pool.close()
    pool.join()

def main():

    if FLAGS.initdb:
        db.backup_and_reset_db()
    if FLAGS.download:
        print "Downloading pdb from rcsb..."
        download_list = open(config.list_of_PDBs_to_download).readline().strip().split(', ')
        mkdir(config.pdb_download_path)
        run(download_list, download_pdb)
    if FLAGS.split:
        print "Spliting receptor and ligand..."
        structure_list = glob(os.path.join(config.pdb_download_path, '*.pdb'))
        mkdir(config.splited_receptors_path)
        run(structure_list, split_structure)
    if FLAGS.reorder:
        print "Reorder ligands atom..."
        smina_pm = smina_param()
        smina_pm.ser_smina(config.smina)
        smina_pm.set_name('reorder')
        smina_pm.load_param(*config.reorder_pm['arg'], **config.reorder_pm['kwarg'])


        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        input_dir = config.splited_ligand_path.rstrip('/')
        run(ligands_list, reorder_ligand)
    

    if FLAGS.rotbond:
        print "Counting rotable bonds..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run(ligands_list, count_rotable_bond)

    if FLAGS.cleanempty:
        print "Cleaning empty docking result"
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands' , '*' ,'*.pdb'))
        run(ligands_list, clean_empty_ligand)

    if FLAGS.overlap:
        print "Overlap detecting..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))
        #ligands_list = ['/home/xander/affinityDB/data/vinardo/1a0b/1a0b_ZN_117_vinardo.pdb']
        run(ligands_list, detect_overlap)

    if FLAGS.rmsd:
        print "Calculating rmsd..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))
        run(ligands_list, calculate_rmsd)

    if FLAGS.contact:
        print "Calculating native contact between receptor and ligand..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))
        run(ligands_list, calculate_native_contact)
        #print "Calculating native contact between ligands..."
        #run(ligands_list, ligands_pair_native_contact)

    if FLAGS.addh:
        print "Add Hydrogens to receptor..."
        receptors_list = glob(os.path.join(config.splited_receptors_path,'*.pdb'))
        input_dir = config.splited_receptors_path.rstrip('/')
        output_dir = input_dir + '_hydrogens'
        run(receptors_list,partial(add_hydrogen, input_dir, output_dir))

        print "Add Hydrogens to original ligands ..."
        ligands_list = glob(os.path.join(config.splited_ligands_path,'*','*.pdb'))
        input_dir = config.splited_ligands_path.rstrip('/')
        output_dir = input_dir + '_hydrogens'
        run(ligands_list, partial(add_hydrogen, input_dir, output_dir))

        print "Add Hydrogens to smina ligands ..."
        ligands_list = glob(os.path.join(config.smina_std_path,'*','*.pdb'))
        input_dir = config.smina_std_path.rstrip('/')
        output_dir = input_dir + '_hydrogens'
        run(ligands_list, partial(add_hydrogen, input_dir, output_dir))

    if FLAGS.vinardo:
        print "Reorder ligands atom..."
        smina_pm = smina_param()
        smina_pm.set_smina(config.smina)
        smina_pm.set_name('reorder')
        smina_pm.load_param(*config.reorder_pm['arg'], **config.reorder_pm['kwarg'])


        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        input_dir = config.splited_ligands_path.rstrip('/')
        output_dir = os.path.join(config.vinardo_docked_path,'crystal_ligands')
        output_dir = output_dir.rstrip('/')
        run(ligands_list, partial(reorder_ligand, input_dir, output_dir, smina_pm))

        print "Smina Vinardo dokcing..."
        smina_pm = smina_param()
        smina_pm.set_smina(config.smina)
        smina_pm.set_name('vinardo')
        smina_pm.load_param(*config.vinardo_pm['arg'],**config.vinardo_pm['kwarg'])

        ligands_list = glob(os.path.join(config.splited_ligands_path,'*','*.pdb'))
        input_dir = config.splited_ligands_path.rstrip('/')
        output_dir = os.path.join(config.vinardo_docked_path,'docked_ligands')
        output_dir = output_dir.rstrip('/')

        run(ligands_list, partial(smina_dock, input_dir, output_dir, smina_pm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Database Create Option")

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--vinardo', action='store_true')
    parser.add_argument('--rotbond', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--rmsd', action='store_true')
    parser.add_argument('--contact', action='store_true')
    parser.add_argument('--addh', action='store_true')
    parser.add_argument('--initdb', action='store_true')
    parser.add_argument('--cleanempty', action='store_true')
    #parser.add_argument('--dock', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    main()
