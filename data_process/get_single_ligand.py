import numpy as np
import pandas as pd
import config
import os,sys
import shutil

'''
In our current database docking result of same crystal ligands stored in one file
This code use obabel to convert specific ligand from file
and store them orderly in given path
The souce is a csv file contain columns ['PDBname','PDBResId']
'''


def get_receptors_and_crystal_ligands(input_file):

    # source to place crystal ligand
    crystal_source = source_crystal_ligands
    # dest to place converted ligand
    crystal_dest = os.path.join(dest_database_path, 'crystal_ligands')
    if not os.path.exists(crystal_dest):
        os.mkdir(crystal_dest)

    # source to place pdb
    pdb_source = source_receptors
    # dest to place pdb
    pdb_dest = os.path.join(dest_database_path, 'receptors')
    if not os.path.exists(pdb_dest):
        os.mkdir(pdb_dest)

    #dirname = os.path.dirname(input_file)
    #receptor = os.path.basename(dirname)
    receptor = input_file.split('_')[0]
    filename = input_file+'_ligand'

    receptor_in = os.path.join(pdb_source,receptor,receptor+'.pdb')
    shutil.copy(receptor_in,pdb_dest)

    crystal_in = os.path.join(crystal_source,receptor,filename+'.pdb')
    crystal_path = os.path.join(crystal_dest,receptor)
    if not os.path.exists(crystal_path):
        os.mkdir(crystal_path)

    crystal_out = os.path.join(crystal_path,filename+'.pdb')
    if not os.path.exists(crystal_out):
        cmd = 'obabel -ipdb %s -opdb -O %s -d'%(crystal_in,crystal_out)
        os.system(cmd)

def convert(item):
    PDB =item['PDBname']
    RS = item['PDBResId']
    RES,Id = RS.split('_')

    source_base = source_docked_ligands
    source_file_path= os.path.join(source_base,PDB,'_'.join([PDB,RES,'ligand','fast.mol']))

    dest_base = os.path.join(dest_database_path,'docked_ligands')
    if not os.path.exists(dest_base):
        os.mkdir(dest_base)
    dest_path = os.path.join(dest_base,PDB)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    dest_file_path = os.path.join(dest_path,'_'.join([PDB,RES,'ligand','fast',Id+'.pdb']))

    cmd = 'obabel -imol2 %s -f %s -l %s -opdb -O %s '%(source_file_path,Id,Id,dest_file_path)

    os.system(cmd)

def run(base, offset):
    df = pd.read_csv(input_csv)
    df.apply(convert,axis=1)
    df['ID'].apply(get_receptors_and_crystal_ligands, axis=1)



input_csv = '/home/xl198/remark/dec_1.csv'
dest_database_path = '/n/scratch2/xl198/dataset/'
source_docked_ligands = '/n/scratch2/xl198/data/result'
source_crystal_ligands = '/n/scratch2/xl198/data/H/data'
source_receptors = '/n/scratch2/xl198/data/H/addH'

def main():
    args = sys.argv
    if len(args) >= 3:
        base = int(args[1])
        offset = int(args[2])
        print 'base %d offset %d' % (base, offset)
        run(base, offset)

if __name__ == '__main__':
    main()
