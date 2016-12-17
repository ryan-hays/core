'''
    Generate vina score for every ligand-pdb target

'''

from vector_gen import pdb_container
from Config import *
import os,csv

FILE_NAME = 'score_onlyinbox.csv'

def score_one_by_vina(PDBname, CLEAN=False):
    '''
    Record score of possible pair
    :param PDBname: the name of pdb
    :return: nothing but will generate rows in FILE_NAME
    '''

    if CLEAN:
        files = os.listdir('data/')
        for filename in files:
            loc = os.path.join('data/') + filename
            if os.path.exists(loc):
                os.remove(loc)

    source_pdb_loc = os.path.join(pdb_PREFIX,PDBname+'.pdb.gz')
    pdb = pdb_container(PDBname,filepos=source_pdb_loc)
    pdb.set_all_vina_benchmark()
    writer = file('result/{}'.format(FILE_NAME), 'a')
    w = csv.writer(writer)
    for k,v in pdb.heterodict.items():
        one_line= [PDBname,k,v['ligand'].getResname(),v['vina_score']]
        w.writerow(one_line)

    writer.flush()
    writer.close()

def initiate_score_file():
    writer = file('result/{}'.format(FILE_NAME), 'wb')
    w = csv.writer(writer)
    w.writerow(['PDBname','Target Ligang ID', 'Target Ligand NAME', 'Vina Score(kcal/mol)'])

if __name__ == '__main__':
    initiate_score_file()
    for each in PDB_tar:
        each = each.lower()
        score_one_by_vina(each,CLEAN=True)
