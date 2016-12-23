__author__ = 'wy'

from Ligand_container import Ligand_container
from Receptor_container import Receptor_container
from atomgroupIO import writePDB,parsePDB,writeMOL2,parseMOL2
from data_process.preprocess.Config import *



class PDB_container:
    '''
    This is the class that will calculate:
    1. autodock results
    2. ligand-ligand attributes
    3. receptor-ligand attributes
    And then bundle them into a dict and provide API to lookup

    I hope this class will only have methods as APIs for other programs or users,
    so it will not deal with files and hopefully return some nice results.
    '''

    def __init__(self,filename,fileloc=None,pdbname=None):
        pass

    def __repr__(self):
        pass

    def set_config(self,config):
        '''

        :param config: a dict for configuration that will be applied
        :return:
        '''

    instance_index = 0
    @classmethod
    def get_num_of_instance(cls):
        cls.instance_index += 1
        return cls.instance_index

    def mount_a_ligand(self):
        pass

    def get_pdb_info_dict(self):
        pass

    def get_pdb_info_str(self):
        pass

    def get_ligand_ids(self):
        pass

    def get_ligands_info_dict(self):
        pass

    def get_ligand_info_dict(self,ligand_id):
        pass

    def get_ligand_info_str(self,ligand_id):
        pass

    def get_ligands_similarity(self,ligand_id, src_file, src_filedir=None):
        pass

    def get_ligand_similarity(self,ligand_id, src_file, src_filedir=None):
        pass

    def get_similarity_idlist(self,src_file,src_filedir=None,method='tanimoto',threshold=0.85):
        pass

    def get_ligand_autovina_score(self,ligand_id):
        pass

    def get_ligand_autogrid_maps(self,ligand_id):
        pass


class Docking_Analyzer(PDB_container):
    '''
    Inherit from PDB_container, but specialized with docking remark generation
    '''

    def __set_coord_benchmark(self,ligand_id,bench_file,bench_dir=None):
        pass

    def mount_ligands_from_docking(self,src_file,src_dir=None):
        pass

    def get_ligand_info_dict(self,ligand_id):
        pass





