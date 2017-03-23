'''
Download PDB from http://www.rcsb.org/
'''
import os,sys
import numpy as np
import multiprocessing
import config
from utils import mkdir



def download_pdb(pdb_name):
    if not os.path.exists(os.path.join(config.pdb_download_path, pdb_name + '.pdb')):
        download_address = 'https://files.rcsb.org/download/{}.pdb'.format(pdb_name)
        os.system('wget -P {}  {}'.format(config.pdb_download_path, download_address))
        print "Download ", pdb_name

def download(target_list):
    # create folder to store pdb
    mkdir(config.pdb_download_path)

    #pool = multiprocessing.Pool(config.process_num)
    #pool.map_async(download_pdb, pdb_list)
    #pool.join()
    #pool.close()

    map(download_pdb,target_list)


if __name__ == '__main__':


    try:
        target_list = open(config.target_list_file).readline().strip().split(', ')

        download(target_list)

    except Exception as e:
        print e
