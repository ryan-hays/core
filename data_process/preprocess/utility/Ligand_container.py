
__author__ = 'wy'

import os.path as ospath
import os.remove as osremove
import os.listdir as oslistdir
import os.rmdir as osrmdir
import os.urandom as osurandom
from data_process.preprocess.Config import temp_pdb_PREFIX
from atomgroupIO import writePDB

class Ligand_container:
    '''
    Use to store the ligand information and define methods interact with ligands (without selector and
    '''
    __ligand = None
    __name = None
    __ligand_type = None
    __temp_file_list = []
    __temp_filefolder = '.'

    instance_index = 0

    def __init__(self,parser,pdbname,ligand_type='Artificial',**kwargs):
        '''
        receive input from parser

        :param parser:
        :param token:
        :param type: 'Experimental' or 'Artificial' or 'Docked'
        :param suffix : when type is 'Docked' , you need this for nomenclature
        :param receptor_dir(optional) : the temp direction of this ligand's receptor
        'Experimental' : real exists cocrystal structure
        'Artificial' : Compound might not be natural one
        'Docked' : It is the prediction made by docking algorithm, need a suffix to distinguish with others
        (Better do that, otherwise parallel computing and file manager will become nasty)
        '''
        self.__ligand= parser.select('hetero and not water')

        self.__ligand_type = ligand_type

        try:
            name_prefix = str(parser.getResindex())
        except:
            raise ValueError('This is not a single ligand!')

        if ligand_type == 'Artificial' or ligand_type == 'Experimental':
            self.__name= '%s_%s'%(pdbname,name_prefix)
        elif ligand_type == 'Docked':
            if 'suffix' in kwargs:
                self.__name = '%s_%s_%s'%(pdbname,name_prefix,kwargs['suffix'])
            else:
                raise ValueError('When type is Docked, must provide a distinguish suffix,'
                                 'otherwise multiple docking results will confuse people if they will be considered'
                                 'at the same time')

        else:
            raise ValueError('ligand_type can only be : Experimental, Artificial or Docked, and '
                             'Docked file should have suffix attribute to distinguish them from others')

        if 'receptor_dir' is not None:
            self.__temp_filefolder= kwargs['receptor_dir']
        else:
            print 'Warning! No receptor_dir value is provided, the program will create a temp direction but receptor and' \
                  'ligand will be stored in seperate place. It is strongly recommend to store them in a same direction because' \
                  'some scripts (like autodock_vina) will have severe bug if two files are in separate locations.'
            self.__temp_filefolder= ospath.join(temp_pdb_PREFIX,pdbname+'_'.join(map(lambda xx:(hex(ord(xx))[2:]),osurandom(16))))

    @classmethod
    def get_num_of_instance(cls):
        cls.instance_index += 1
        return cls.instance_index

    @property
    def temp_file_dir(self):
        if self.__temp_filefolder is '.':
            print 'Warning, the temporary file path is not configured!'
        return self.__temp_filefolder

    @property
    def ligand_name(self):
        if self.__name is None:
            raise AttributeError('No liangd is allocated!')
        return self.__name

    @property
    def ligand_type(self):
        if self.__ligand_type is None:
            raise AttributeError('No liangd is allocated!')
        return self.__ligand_type


    def write_file(self,format='pdb',location=None):
        '''
        Write atoms into temporary file location or other specific location
        For future implementation, it might be able to support different type of format, e.g. mol2
        :param format:
        :param location: Someplace other than designated temporary file location
        :return:
        '''
        pass

    def get_vector_box(self):
        pass

    def get_vector_box_with_receptor(self,receptor):
        pass


    def prepare_autodock_ligand(self):
        pass

    def prepare_autodock_receptor(self):
        '''
        For auto grid
        :return:
        '''
        pass



    def __del__(self):
        '''
        Remove all temporary files
        :return:
        '''
        temp_folder = self.__temp_filefolder
        for each in self.__temp_file_list:
            abs_dir = ospath.join(temp_folder,each)
            if ospath.exists(abs_dir):
                osremove(abs_dir)
        if len(oslistdir(temp_folder))==0:
            osrmdir(temp_folder)



class Ligands_container:
    '''
    This is for multiple ligands which might share similar features (like results in one docking process,
    evaluation results for MD analysis)
    '''
    #TODO implement this
    pass