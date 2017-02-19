import os,sys
import pdb_target
import multiprocessing
import config
# download PDB from http://www.rcsb.org/


def download_pdb(pdb_name):
    if not os.path.exists(os.path.join(config.pdb_download_path, pdb_name + '.pdb')):
        download_address = 'https://files.rcsb.org/download/' + pdb_name + '.pdb'
        os.system('wget -P {}  {}'.format(config.pdb_download_path, download_address))
        print "Download ", pdb_name

def download():
    # create folder to store pdb
    os.system('mkdir -p {}'.format(config.pdb_download_path))

    # get the list of receptor need to be download
    # here we choose the structure with both ligands and binding affinity information
    pdb_list = list(set(pdb_target.has_ligands) & set(pdb_target.binding_affinity))

    map(download_pdb,pdb_list)

