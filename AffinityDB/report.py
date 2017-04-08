"""
generate report for the database
"""

import os
import sys
from copy import copy
from collections import Counter
import numpy as np 
import pandas as pd 
import config
from db import database


db = database()


dataframe_dict = {}

def load_dataframe():
    #db.export()
    for table in db.tables.keys():
        try:
            csv_path = os.path.join(config.table_dir, table+'.csv')
        except IOError as e:
            print e
            continue
        dataframe_dict[table] = pd.read_csv(csv_path)


def gather_state():
    """
    gather report for state
    """

    state_tables = [tab for tab in dataframe_dict.keys() if tab.endswith('state')]
    state_tables = sorted(state_tables)

    records = []

    for table in state_tables:
        
        csv_path = os.path.join(config.table_dir,table+'.csv')

        df = pd.read_csv(csv_path)
        success_entrys = df[df['state'] == 1]
        success_num = len(success_entrys)

        failed_entrys = df[df['state'] == 0]
        failed_group = failed_entrys.groupby('comment')
        failed_count = []
        for name, group in failed_group:
            failed_count.append(name, len(group))

        print table
        print 'success : ', success_num
        
        if len(failed_count):
            print 'failed : ' 
            for reason, num in failed_count:
                print "\t%d\t%s" % (num, reason)

        print '\n\n'

def report_for_state_by_id(dataframe, identifier, total_num):
    
    df_success = dataframe[dataframe['state']==1]
    df_failed = dataframe[dataframe['state']==0]

    print "\ttotal : %d\n" % total_num
    success = len(df_success)
    print "\tfinished : %d\tpercent: %.2f" % (success, 100.0 * success/total_num)
    rest = total_num - success
    print "\trest     : %d\tpercent: %.2f " % ( rest, 100.0 * rest/ total_num)

    failed_group = df_failed.groupby('comment')
    failed_count = []
    for name, group in failed_group:
        failed_count.append([name, len(group)])

    if len(failed_count):
        print 
        for reason, failed_num in failed_count:
            print "\t%d failed for : %s" % (failed_num, reason)
        



def report_for_state(dataframe, name, total_num, key='identifier',cont='For %s\n'):
    
    print '\n\n==========\n%s\n==========\n' % name
    data_groups = dataframe.groupby(key)
    for identifier, group in data_groups:
        print cont % identifier
        report_for_state_by_id(group, identifier, total_num)



def reporter_test():
    
    ligands_num = len(dataframe_dict['ligand_atom_num'])
    report_for_state(dataframe_dict['dock_state'], 'dock', ligands_num,
                                    'identifier','result for %s\n')

    receptor_info = dataframe_dict['receptor_info']

    pair_cumsum = lambda x: (x-1)*x/2 if x>1 else 0
    similarity_pair_num = sum(receptor_info['residue_num'].apply(pair_cumsum))

    report_for_state(dataframe_dict['similarity_state'], 
                                    'similarity', 
                                    similarity_pair_num, 
                                    'finger_print',
                                    'use finger print %s\n')

    overlap_ligands_pairs = sum(receptor_info['residue_num'].apply(lambda x:x*x))

    report_for_state(dataframe_dict['overlap_state'],'overlap',overlap_ligands_pairs,
                                    'identifier',
                                    'for docking result of %s\n')

    ligand_atom_num = copy(dataframe_dict['ligand_atom_num'])
    ligand_atom_num['rlpair'] = ligand_atom_num['name'].apply(lambda x:'_'.join(x.split('_')[:2]))
    ligand_count = Counter(ligand_atom_num['rlpair'])
    pair_cumsum = lambda x: x*x 
    rmsd_count_num = sum(map(pair_cumsum, ligand_count.values()))

    report_for_state(dataframe_dict['rmsd_state'], 'rmsd', rmsd_count_num,
                                    'identifier',
                                    'for docking result of %s\n')

    report_for_state(dataframe_dict['native_contact_state'], 'native_contact', ligands_num,
                                    'identifier',
                                    'for docking result of %s\n' )



def reporter_gen():

    ligands_atom = len(dataframe_dict['ligand_atom_num'])

    dock_df = dataframe_dict['dock_state']
    dock_success = dock_df[dock_df['state']==1]
    dock_failed = dock_df[dock_df['state']==0]
    failed_group = dock_failed.groupby('comment')
    failed_count = []
    for name, group in failed_group:
        failed_count.append([name, len(group)])

    print "\n\n=========\ndocking\n=========\n"
    ligands_num = ligands_atom
    print "total ligands : %d" % ligands_num
    success = len(dock_success)
    print "finished: %d\tpercents: %f" % (success, 100.0 * success/ligands_num)
    rest = ligands_num - success
    print "rest: %d\tpercent: %f " % ( rest, 100.0 * rest/ ligands_num)

    print '\n'
    for reason, failed_num in failed_count:
        print "%d ligands failed docking due to %s" % (failed_num, reason)


    receptor_info = dataframe_dict['receptor_info']

    pair_cumsum = lambda x: (x-1)*x/2 if x>1 else 0
    ligands_pair_num = sum(receptor_info['residue_num'].apply(pair_cumsum))


    """
    similarity
    """

    similar_df = dataframe_dict['similarity']

    similar_group = similar_df.groupby(['finger_print'])
    similar_count = []
    for name, group in similar_group:
        similar_count.append([name, len(group)])


    print "\n\n=========\nsimilarity\n============\n"
    for finger_print, finish in similar_count:
        print "calculate similarity use finger print %s" % finger_print
        print "finished: %d\tpercent: %f" % (finish, 100.0 * finish / ligands_pair_num)
        rest = ligands_pair_num - finish
        print "rest: %d\tpercent: %f" % (rest, 100.0 * rest / ligands_pair_num)



    """
    overlap
    """

    overlap_ligands_pairs = sum(receptor_info['residue_num'].apply(lambda x:x*x))
    
    overlap_state = dataframe_dict['overlap_state']

    print "\n\n=========\noverlap\n=========\n"

    overlap_groups = overlap_state.groupby('identifier')
    for identifier, overlap_group in overlap_groups :
        overlap_success = overlap_group[overlap_group['state']==1]
        overlap_failed = overlap_group[overlap_group['state']==1]
        overlap_count = []
        for identifier, group in overlap_group:
            overlap_count.append([identifier, len(group)])

        
        print "total ligands pairs : %d" % overlap_ligands_pairs
        for identifer , finish in overlap_count:

            print "overlap result for %s" % identifier
            print "finished: %d\tpercent: %f " % (finish, 100.0 * finish / overlap_ligands_pairs)
            rest = overlap_ligands_pairs - finish
            print "rest: %d\tpercent: %f" % (rest, 100.0 * rest / overlap_ligands_pairs )


    """
    rmsd
    """

    print "\n\n=========\nrmsd\n=========\n"

    ligand_atom_num = copy(dataframe_dict['ligand_atom_num'])
    ligand_atom_num['rlpair'] = ligand_atom_num['name'].apply(lambda x:'_'.join(x.split('_')[:2]))
    ligand_count = Counter(ligand_atom_num['rlpair'])
    pair_cumsum = lambda x: x*x 
    rmsd_count_num = sum(map(pair_cumsum, ligand_count.values()))
    print rmsd_count_num


    rmsd_state = dataframe_dict['rmsd_state']
    rmsd_success = rmsd_state[rmsd_state['state']== 1][['docked_ligand','crystal_ligand']]
    rmsd_success =rmsd_success.drop_duplicates()
    rmsd_failed = rmsd_state[rmsd_state['state'] == 0][['docked_ligand','crystal_ligand','comment']]
    rmsd_failed = rmsd_failed.drop_duplicates()

    failed_group = rmsd_failed.groupby('comment')
    failed_count = []
    for name, group in failed_group:
        failed_count.append([name, len(group)])


    success = len(rmsd_success)
    print "rmsd"
    print "finished: %d\t percent: %f" % (success, 100.0 * success/rmsd_count_num)
    rest = rmsd_count_num - success
    print "rest: %d\t percent: %f" % (rest, 100.0 * rest/rmsd_count_num)

    for reason, failed_num in failed_count:
        print "%d pairs failed due to %s" % (failed_num, reason)

    """
    overlap
    """


def _reporter_gen():

    total_ligands_num = len(dataframe_dict['ligand_atom_num'])

    dock_group = dataframe_dict['dock_state'].groupby('identifier')



def rmsd_report():
    pass

        
load_dataframe()
reporter_test()