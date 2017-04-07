"""
generate report for the database
"""

import os
import sys
import numpy as np 
import pandas as pd 
import config
from db import database

db = database()


dataframe_dict = {}

def load_dataframe():
    db.export()
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

def reporter_gen():

    ligands_atom = len(dataframe_dict['ligand_atom_num'])

    dock_df = dataframe_dict['dock_state']
    dock_success = dock_df[dock_df['state']==1]
    dock_failed = dock_df[dock_df['state']==0]
    failed_group = dock_failed.groupby('comment')
    failed_count = []
    for name, group in failed_group:
        failed_count.append([name, len(group)])

    print "=========\ndocking\n=========\n\n"
    ligands_num = len(ligands_atom)
    print "total ligands : %d" % ligands_num
    success = len(dock_success)
    print "finished: %d\tpercents: %f" % (success, 100.0 * success/ligands_num)
    rest = ligands_num - success
    print "rest: %d\tpercent: %f " % ( rest, 10* rest/ ligands_num)

    for reason, failed_num in failed_count:
        print "%d ligands failed docking due to %s" % (failed_num, reason)


    receptor_info = dataframe_dict['receptor_info']

    pair_cumsum = lambda x: (x-1)*x/2 if x>1 else 0
    ligands_pair_num = sum(receptor_info['residue_num'].apply(pair_cumsum))


    """
    similarity
    """

    similar_df = dataframe_dict['similarity']

    similar_group = similar_df.groupby('finger_print')
    similar_count = []
    for name, group in similar_group:
        similar_count.append([name, len(group)])


    print "=========\nsimilarity\n============\n\ns"
    for finger_print, finish in similar_count:
        print "calculate similarity use finger print %s" % finger_print
        print "finished: %d\tpercent: %f" % (finish, 100.0 * finish / ligands_pair_num)
        rest = ligands_pair_num - finish
        print "rest: %d\tpercent: %f" % (rest, 100.0 * rest / ligands_pair_num)



    """
    overlap
    """

    #overlap_count_pairs = similar_df[similar_df['similarity']>config.tanimoto_cutoff]
    overlap_state = dataframe_dict['overlap_state']

    overlap_group = overlap_state.groupby('finger_print')
    overlap_count = []
    for name, group in overlap_group:
        overlap_count.append([name, len(group)])

    print "overlap"
    for finger_print, finish in overlap_count:

        print "overlap between ligands"

    """
    rmsd
    """

    rmsd_count_pairs = similar_df[similar_df['similarity']>config.tanimoto_cutoff]


    rmsd_state = dataframe_dict['rmsd_state']
    rmsd_success = rmsd_state[rmsd_state['state']== 1]['docked_ligand','crystal_ligand']
    rmsd_success = set(rmsd_success)
    rmsd_failed = rmsd_state[rmsd_state['state'] == 0]['docked_lgiand','crystal_ligand']
    rmsd_failed = set(rmsd_failed)

    failed_group = rmsd_failed.groupby('comment')
    failed_count = []
    for name, group in failed_group:
        failed_count.append([name, len(group)])

    rmsd_count_num = len(rmsd_count_pairs)
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
reporter_gen()
