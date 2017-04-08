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

# short function for convenient
_mkdir = lambda path: os.system('mkdir -p {}'.format(path))
_ligand_name_of = lambda x:os.path.basename(x).split('.')[0]
_receptor_of = lambda x:os.path.basename(x).split('_')[0]
_name_of = lambda x:os.path.basename(x).split('.')[0]
_receptor_path_of = lambda x:os.path.join(config.splited_receptors_path, _receptor_of(x) + '.pdb')

_identifier_of = lambda x:'_'.join(os.path.basename(x.rstrip('/')).split('_')[1:])


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

        return



    hetero = parsed.select(
        '(hetero and not water) or resname ATP or resname ADP or resname AMP or resname GTP or resname GDP or resname GMP')

    receptor = parsed.select('protein or nucleic ')
    if receptor is None:
        data = [pdb_name, 0, "No receptor found"]
        data = [data]
        db.insert_or_replace('split_state',data)
        log("select_failed.log", "No receptor found")
        return
    if hetero is None:
        data = [pdb_name, 0, "No ligand found"]
        data = [[data]]
        db.insert_or_replace('split_state',data)
        #log("select_failed.log", "No ligand found")
        return

    # write ligand into file
    for each in prody.HierView(hetero).iterResidues():
        ResId = each.getResindex()
        ResName = each.getResname()
        ligand_path = os.path.join(config.splited_ligands_path,
                                   pdb_name,
                                   "{}_{}_{}_ligand.pdb".format(pdb_name, ResName, ResId))
        _mkdir(os.path.dirname(ligand_path))
        prody.writePDB(ligand_path, each)
        atom_num = each.select('not hydrogen').numAtoms()

        data = ["{}_{}_{}".format(pdb_name, ResName, ResId), atom_num]
        data = [data]
        db.insert_or_replace('ligand_atom_num',data)
        #log('atom_num.csv',
        #    "{}_{}_{},{}".format(pdb_name, ResName, ResId, atom_num),
        #    head='ligand,atom_num')

    receptor_path = os.path.join(config.splited_receptors_path, pdb_name + '.pdb')
    prody.writePDB(receptor_path, receptor)
    
    receptor_atom_num = receptor.select('not hydrogen').numAtoms()

    try:

        header = prody.parsePDBHeader(pdb_path)
        data = [pdb_name, header['experiment'], header['resolution'], receptor_atom_num ,len(header['chemicals']), len(header['biomoltrans']['1'][0])]
        data = [data]
        db.insert_or_replace('receptor_info', data)

    except Exception as e:
        data = [pdb_name, 0, str(e)]
        data = [data]
        db.insert_or_replace('split_state', data)
        return

    data = [pdb_name, 1, 'success']
    data = [data]
    db.insert_or_replace('split_state',data)


def reorder_ligand(input_dir, output_dir, smina_pm, identifier, ligand_path):
    """
    smina output file change the order of atom,
    inorder to get correct rmsd value, we just read the
    crystal ligand and write it by smina to keep the 
    atom order the same as docking result
    """

    lignad_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)
    receptor_path = _receptor_path_of(ligand_path)
    receptor_name = _receptor_of(ligand_path)

    out_ligand_path = ligand_path.replace(input_dir, output_dir)

    kw = {
        'receptor':receptor_path,
        'ligand'  :ligand_path,
        'out'     :out_ligand_path
    }

    if not db.if_success('reorder_state',[ligand_name, identifier]):
        print 'reorder %s ...' % ligand_name
    #if not os.path.exists(out_ligand_path):
        _mkdir(os.path.dirname(out_ligand_path))
        cmd = smina_pm.make_command(**kw)

        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        cont = cl.communicate()[0].strip().split('\n')

        terms = [line.strip().split(' ')[2:] for line in cont if line.startswith('##')]
        head = map(lambda x:x.replace(',',' '),terms[0])

        head = ['ligand','position'] + map(lambda x:'"%s"' % x, head)
        data = [ligand_name, 0] + map(lambda x:float(x),terms[1])
        data = [data]
        db.insert_or_replace('scoring_terms',data,head)

        data = [ligand_name, identifier, 1, 'success']
        data = [data]
        db.insert_or_replace('reorder_state',data)

def _get_similar_ligands(ligand_path, finger_print):
    """
    calculate tanimoto similarity
    return the ligands with similarty > config.tanimoto_cutoff
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_dir = os.path.dirname(ligand_path).replace('docked_ligands','crystal_ligands')
    crystal_ligand = os.path.join(crystal_dir , '_'.join([receptor, lig, resid, 'ligand.pdb']))

    ligands_list = glob(os.path.join(os.path.dirname(crystal_ligand), '*.pdb'))

    ligands_for_same_receptor = list(set(ligands_list) - set([crystal_ligand]))


    similar_ligands = []
    for lig_path in ligands_for_same_receptor:
        cmd = 'babel -d {} {} -ofpt -xf{}'.format(crystal_ligand, lig_path, finger_print)
        print cmd
        ls = os.popen(cmd).read()
        tanimoto_similarity = re.split('=|\n', ls)[2]

        try:
            float(tanimoto_similarity)
        except:
            continue

        lig_pair = [_ligand_name_of(crystal_ligand), _ligand_name_of(lig_path)]
        lig_pair = lig_pair if lig_pair[0]<lig_pair[1] else [lig_pair[1],lig_pair[0]]

        data = lig_pair + [finger_print, tanimoto_similarity]
        data = [data]
        db.insert_or_replace('similarity',data)
        #log('tanimoto_similarity.csv',
        #    '{},{},{}'.format(_ligand_name_of(ligand_path), _ligand_name_of(lig_path), tanimoto_similarity),
        #    head='lig_a,lig_b,finger_print, tanimoto')
        if tanimoto_similarity > config.tanimoto_cutoff:
            similar_ligands.append([lig_path, tanimoto_similarity, finger_print])

    return crystal_ligand, similar_ligands

def _get_ligands_from_same_receptor(ligand_path):
    """
    get the path of ligands, they are splited from the same receptor as the input

    """

    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_dir = os.path.dirname(ligand_path).replace('docked_ligands','crystal_ligands')
    crystal_ligand = os.path.join(crystal_dir , '_'.join([receptor, lig, resid, 'ligand.pdb']))

    ligands_list = glob(os.path.join(os.path.dirname(crystal_ligand), '*.pdb'))

    ligands_from_same_receptor = list(set(ligands_list) - set([crystal_ligand]))

    return crystal_ligand, ligands_from_same_receptor
    

def _get_same_ligands(ligand_path):
    """
    get the path of ligands, they are the same ligand docked to the same receptor,
    but different binding site.
    :param ligand_path:
    :return:
    """
    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)
    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_dir = os.path.dirname(ligand_path).replace("docked_ligands","crystal_ligands")
    same_ligands = glob(os.path.join(crystal_dir, '{}_{}_*.pdb'.format(receptor, lig)))
    return same_ligands

def calculate_similarity(identifier, finger_print, ligand_path):
    """
    calculate tanimoto similarity
    return the ligands with similarty > config.tanimoto_cutoff
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_dir = os.path.dirname(ligand_path).replace('docked_ligands','crystal_ligands')
    crystal_ligand = os.path.join(crystal_dir , '_'.join([receptor, lig, resid, 'ligand.pdb']))

    ligands_list = glob(os.path.join(os.path.dirname(crystal_ligand), '*.pdb'))

    ligands_for_same_receptor = list(set(ligands_list) - set([crystal_ligand]))


    for lig_path in ligands_for_same_receptor:

        lig_pair = [_ligand_name_of(crystal_ligand), _ligand_name_of(lig_path)]
        lig_pair = lig_pair if lig_pair[0] < lig_pair[1] else [lig_pair[1], lig_pair[0]]
        data = lig_pair + [finger_print]
        if db.if_success('similarity_state',data):
            continue

        cmd = 'babel -d {} {} -ofpt -xf{}'.format(crystal_ligand, lig_path, finger_print)
        print cmd
        ls = os.popen(cmd).read()
        tanimoto_similarity = re.split('=|\n', ls)[2]

        try:
            float(tanimoto_similarity)
        except Exception as e:
            data = data + [0,'babel similarity failed']
            data = [data]
            db.insert_or_replace('similarity_state',data)
            continue

        data = data + [1, 'success']
        data = [data]
        db.insert_or_replace('similarity_state',data)

        data = lig_pair + [finger_print, tanimoto_similarity]
        data = [data]
        db.insert_or_replace('similarity',data)


def overlap_with_ligand(docked_ligand, crystal_ligand, identifier):
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
            "{},{}".format(_ligand_name_of(docked_ligand), str(e)),
            head='ligand,exception')
        return 
    try:
        crystal = prody.parsePDB(crystal_ligand).getCoords()
    except Exception as e:
        log("overlap_failed.log",
            "{},{}".format(_ligand_name_of(crystal_ligand), str(e)),
            head='ligand,exception')
        return

    data = [_ligand_name_of(docked_ligand), _ligand_name_of(crystal_ligand), identifier]
    if db.if_success('overlap_state',data):
        print '%s %s overlap already' % (data[0], data[1])
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
        datum = [_ligand_name_of(docked_ligand), _ligand_name_of(crystal_ligand), i + 1, config.clash_cutoff_A , identifier, ratio]
        data.append(datum)

    db.insert_or_replace('overlap',data)

    data = [_ligand_name_of(docked_ligand), _ligand_name_of(crystal_ligand), identifier ,1,'success']
    data = [data]
    db.insert_or_replace('overlap_state',data)
    

    return position_clash


def overlap_with_ligands(identifier, ligand_path, finger_print='FP4'):
    """
    if docked result overlap with other crystal ligands
    splited from the same receptor
    note crystal ligand should reordered by smina
    :param ligand_path: path
    :return:
    """

    ligand_path = ligand_path.strip()
    ligand_name = os.path.basename(ligand_path).split('.')[0]

    if not db.if_success('dock_state',[ligand_name, identifier ]):
        print "%s doesn't docked correctly" % ligand_name
        return 

    crystal_ligand, other_ligands = _get_ligands_from_same_receptor(ligand_path)
    crystal_ligand, similar_ligands = _get_similar_ligands(ligand_path, finger_print)
    if len(other_ligands) == 0:
        return 
    
    for lig_path in other_ligands:
        cur_position_clash = overlap_with_ligand(ligand_path, lig_path, identifier)

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

def calculate_rmsd(identifier, ligand_path):
    """
    calculate rmsd between docked ligands and crystal ligand
    note when calculate rmsd, crystal ligand should be reordered by smina
    :param ligand_path: path
    :return: 
    """

    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)

    receptor, lig, resid, _ = ligand_name.split('_')
    crystal_ligands = _get_same_ligands(ligand_path)

    if not db.if_success('dock_state',[ligand_name, identifier]):
        print "%s doesn't docked correctly" % ligand_name
        return

    try:
        docked_coords = prody.parsePDB(ligand_path).select('not hydrogen').getCoordsets()
        for crystal_ligand in crystal_ligands:
            data = [ligand_name, _ligand_name_of(crystal_ligand), identifier]
            if db.if_success('rmsd_state',data):
                print "%s %s rmsd already" % (data[0], data[1])
                continue
            crystal_coord = prody.parsePDB(crystal_ligand).select('not hydrogen').getCoords()
            rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=-1), axis=-1))

            
            data = []
            for i,rd in enumerate(rmsd):
                datum = [ligand_name, _ligand_name_of(crystal_ligand), i + 1,identifier ,rd]
                data.append(datum)

            db.insert_or_replace('rmsd',data)

            data = [ligand_name, _ligand_name_of(crystal_ligand), identifier, 1, 'success']
            data = [data]
            db.insert_or_replace('rmsd_state',data)
            
    except Exception as e:
        data = [ligand_name, _ligand_name_of(crystal_ligand), identifier, 0, str(e)]
        data = [data]
        db.insert_or_replace('rmsd_state',data)
        return

def calculate_native_contact(identifier, ligand_path):
    """
    calculate native contact between the ligand and receptor
    notice when calculating native contact, we ignore all hydrogen
    :param ligand_path: path
    :return: 
    """
    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)
    
    if not db.if_success('dock_state',[ligand_name, identifier]):
        print "%s doesn't docked correctly" % ligand_name
        return 
    
    if db.if_success('native_contact_state',[ligand_name, identifier]):
        print "%s native contact already" % ligand_name
        return 

    receptor, res_name, res_id, _  = ligand_name.split('_')
    crystal_ligand_dir = os.path.dirname(ligand_path).replace('/docked_ligands/','/crystal_ligands/')
    crystal_ligand_path = os.path.join(crystal_ligand_dir,'_'.join([receptor, res_name, res_id, 'ligand.pdb']) )


    receptor_path = _receptor_path_of(ligand_path)

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
            datum = [ligand_name, i+1, identifier] + list(nts)
            data.append(datum)

        
        db.insert_or_replace('native_contact',data)

        data = [ligand_name, identifier,  1, 'success']
        data = [data]
        db.insert_or_replace('native_contact_state',data)


            
    except Exception as e:
        data = [ligand_name, identifier, 0, str(e)]
        data = [data]
        db.insert_or_replace('native_contact_state',data)

def add_hydrogen(input_dir, output_dir, identifier ,input_file):
    """
    add hydrogens to molecule
    """
    try:
        input_file = input_file.strip()
        output_file = input_file.replace(input_dir, output_dir)

        if not db.if_success('add_hydrogens_state',[_name_of(input_file), identifier]):
            _mkdir(os.path.dirname(output_file))
            cmd = 'obabel -ipdb {} -opdb -O {} -h'.format(input_file, output_file)
            os.system(cmd)

            if count_lines(output_file):
                data = [_name_of(input_file), identifier, 1, 'success']
                data = [data]
                db.insert_or_replace('add_hydrogens_state',data)
    except Exception as e:
        print e

def minimize_hydrogens(input_dir, output_dir, identifier, input_file):
    """
    
    :param input_dir: 
    :param output_dir: 
    :param input_file: 
    :return: 
    """

    try:
        import openbabel as ob
        input_file = input_file.strip()
        output_file = input_file.replace(input_dir, output_dir)

        data = [_name_of(input_file), identifier]
        if not db.if_success('minimize_state',data):
            conv = ob.OBConversion()
            mol = ob.OBMol()
            conv.ReadFile(mol,input_file)
            mol.AddHydrogens()

            heavy_atoms = [ atom for atom in ob.OBMolAtomIter(mol) if not atom.IsHydrogen()]

            print "number of heavy atoms ",len(heavy_atoms)
            print "number of atoms ",len(list(ob.OBMolAtomIter(mol)))
            heavy_atoms_idx = [ atom.GetId() for atom in heavy_atoms ]
            constraints = ob.OBFFConstraints()
            map(lambda idx: constraints.AddAtomConstraint(idx), heavy_atoms_idx)

            forcefield = ob.OBForceField.FindForceField("MMFF94")
            forcefield.Setup(mol, constraints)
            forcefield.SetConstraints(constraints)

            forcefield.ConjugateGradients(500)
            forcefield.GetCoordinates(mol)

            _mkdir(os.path.dirname(output_file))
            conv.WriteFile(mol, output_file)

            if count_lines(output_file):
                data = [_name_of(input_file), identifier, 1, 'success']
                data = [data]
                db.insert_or_replace('minimize_state',data)
    except Exception as e:
        print e
        


def smina_dock(input_dir, output_dir, smina_pm, identifier, ligand_path):
    """
    dock ligands by smina
    """
    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)
    receptor_path = _receptor_path_of(ligand_path)
    receptor_name = ligand_name.split('_')[0]

    docked_ligand_name = os.path.basename(ligand_path).replace('ligand', smina_pm.name)
    docked_ligand_path = os.path.join(output_dir, receptor_name, docked_ligand_name)

    kw = {
        'receptor':receptor_path,
        'ligand'  :ligand_path,
        'autobox_ligand' : ligand_path,
        'out'     :docked_ligand_path
    }

    
    if not db.if_success('dock_state',[ligand_name, identifier]):
        _mkdir(os.path.dirname(docked_ligand_path))
        cmd = smina_pm.make_command(**kw)
        #print cmd
        os.system(cmd)
        
        if count_lines(docked_ligand_path):
            data = [_ligand_name_of(docked_ligand_name), identifier, 1, 'success']
            data = [data]
            db.insert_or_replace('dock_state',data)
    else:
        print "%s dock already" % ligand_name

def clean_empty_ligand(identifier, ligand_path):
    """
    for some reason, dock failed and nothing fin the output file
    remove empty file, and change dock status to failure
    """

    ligand_path = ligand_path.strip()
    ligand_name = _ligand_name_of(ligand_path)
    
    if not count_lines(ligand_path):
        os.remove(ligand_path)
        data = [ligand_name, identifier, 0, 'empty']
        data = [data]
        db.insert_or_replace('dock_state',data)
    else:
        try:
            parsed = prody.parsePDB(ligand_path)
            
        except:
            os.remove(ligand_path)
            data = [ligand_name, identifier, 0, 'incomplete']
            data = [data]
            db.insert_or_replace('dock_state',data)
            return 
        
        data = [ligand_name, identifier, 1, 'success']
        data = [data]
        db.insert_or_replace('dock_state',data)
            


#@profile
def run_multiprocess(target_list, func):
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
        _mkdir(config.pdb_download_path)
        run_multiprocess(download_list, download_pdb)
    if FLAGS.split:
        print "Spliting receptor and ligand..."
        structure_list = glob(os.path.join(config.pdb_download_path, '*.pdb'))
        _mkdir(config.splited_receptors_path)
        run_multiprocess(structure_list, split_structure)

    if FLAGS.vinardo_dock or FLAGS.smina_dock:
        print "Reorder ligands atom..."

        if FLAGS.vinardo_dock:
            base_out_dir = config.vinardo_docked_path.rstrip('/')
            smina_pm = smina_param('vinardo')
        elif FLAGS.smina_dock:
            base_out_dir = config.smina_docked_path.rstrip('/')
            smina_pm = smina_param('smina')

        
        smina_pm.load_param(*config.reorder_pm['arg'], **config.reorder_pm['kwarg'])

        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        input_dir = config.splited_ligands_path.rstrip('/')
        output_dir = os.path.join(base_out_dir, 'crystal_ligands')
        output_dir = output_dir.rstrip('/')

        identifier = _identifier_of(base_out_dir)

        run_multiprocess(ligands_list, partial(reorder_ligand, input_dir, output_dir, smina_pm, identifier))

    if FLAGS.rotbond:
        try:
            import openbabel as op
        except:
            print "Cannot load openbabel"
            return 

        print "Counting rotable bonds..."
        ligands_list = glob(os.path.join(config.splited_ligands_path, '*', '*.pdb'))
        run_multiprocess(ligands_list, count_rotable_bond)

    if FLAGS.cleanempty:
        print "Cleaning empty docking result"
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands' , '*' ,'*.pdb'))
        identifier = _identifier_of(config.vinardo_docked_path)
        run_multiprocess(ligands_list, partial(clean_empty_ligand, identifier))

    if FLAGS.similarity:
        print "Claculate ligands-pair similarity"
        ligands_list = glob(os.path.join(config.vinardo_docked_path, 'docked_ligands', '*', '*.pdb'))
        identifier = _identifier_of(config.vinardo_docked_path)
        run_multiprocess(ligands_list, partial(calculate_similarity, identifier, 'FP4'))

    if FLAGS.overlap:
        print "Overlap detecting..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))

        identifier = _identifier_of(config.vinardo_docked_path)

        run_multiprocess(ligands_list, partial(overlap_with_ligands, identifier))

    if FLAGS.rmsd:
        print "Calculating rmsd..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))
        identifier = _identifier_of(config.vinardo_docked_path)
        run_multiprocess(ligands_list, partial(calculate_rmsd, identifier))

    if FLAGS.contact:
        print "Calculating native contact between receptor and ligand..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'docked_ligands', '*', '*.pdb'))
        identifier = _identifier_of(config.vinardo_docked_path)
        run_multiprocess(ligands_list, partial(calculate_native_contact, identifier))


    if FLAGS.addh:
        print "Add Hydrogens to splited structure"
        receptors_list = glob(os.path.join(config.splited_receptors_path,'*.pdb'))
        ligands_list = glob(os.path.join(config.splited_ligands_path,'*','*.pdb'))
        input_dir = config.splited_path.rstrip('/')
        output_dir = input_dir + '_hydrogens'

        identifier = _identifier_of(input_dir)
        run_multiprocess(receptors_list, partial(add_hydrogen, input_dir, output_dir, identifier))
        run_multiprocess(ligands_list, partial(add_hydrogen, input_dir, output_dir, identifier))

        print "Add Hydrogens to vinardo docked ..."
        
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'*','*','*.pdb'))
        input_dir = config.vinardo_docked_path.rstrip('/')
        output_dir = input_dir + '_hydrogens'

        identifier = _identifier_of(input_dir)
        run_multiprocess(ligands_list, partial(add_hydrogen, input_dir, output_dir, identifier))



    if FLAGS.minimize:
        try:
            import openbabel as ob
        except:
            print "Cannot load openbabel"
            return 


        print "Minimize splited structures ..."
        receptors_list = glob(os.path.join(config.splited_receptors_path,'*.pdb'))
        ligands_list = glob(os.path.join(config.splited_ligands_path,'*','*.pdb'))
        input_dir = config.splited_path.rstrip('/')
        output_dir = input_dir + '_minimize'

        identifier = _identifier_of(input_dir)
        
        run_multiprocess(ligands_list, partial(minimize_hydrogens, input_dir, output_dir, identifier))
        run_multiprocess(receptors_list, partial(minimize_hydrogens, input_dir, output_dir, identifier))

        print "Minimize to smina ligands ..."
        ligands_list = glob(os.path.join(config.vinardo_docked_path,'*','*','*.pdb'))
        input_dir = config.vinardo_docked_path.rstrip('/')
        output_dir = input_dir + '_minimize'

        identifier = _identifier_of(input_dir)
        run_multiprocess(ligands_list, partial(minimize_hydrogens, input_dir, output_dir, identifier))


    if FLAGS.vinardo_dock:

        print "Smina Vinardo dokcing..."
        smina_pm = smina_param('vinardo')
        smina_pm.load_param(*config.vinardo_pm['arg'],**config.vinardo_pm['kwarg'])

        ligands_list = glob(os.path.join(config.splited_ligands_path,'*','*.pdb'))
        input_dir = config.splited_ligands_path.rstrip('/')
        output_dir = os.path.join(config.vinardo_docked_path,'docked_ligands')
        output_dir = output_dir.rstrip('/')

        identifier = _identifier_of(config.vinardo_docked_path)
        run_multiprocess(ligands_list, partial(smina_dock, input_dir, output_dir, smina_pm, identifier))

    if FLAGS.export:
        db.export()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Database Create Option")

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--vinardo_dock', action='store_true')
    parser.add_argument('--smina_dock', action='store_true')
    parser.add_argument('--rotbond', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument('--similarity',action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--rmsd', action='store_true')
    parser.add_argument('--contact', action='store_true')
    parser.add_argument('--addh', action='store_true')
    parser.add_argument('--initdb', action='store_true')
    parser.add_argument('--cleanempty', action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--minimize', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    main()
