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
        
load_dataframe()
gather_state()
