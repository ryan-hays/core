__author__ = 'wy'

import os.path as ospath
import os.remove as osremove
import os.listdir as oslistdir
import os.rmdir as osrmdir
import os.urandom as osurandom
from data_process.preprocess.Config import temp_pdb_PREFIX
from atomgroupIO import writePDB

class Receptor_container:
    '''
    Use to store the receptor information
    The reason to write a class to do so(even now we use prody) is to use this class to manage files
    and ensure methods in upper level is not too nasty (since we need lots of temporary files)
    '''
    __receptor_type = None
    __receptor = None
    __name = None
    __temp_file_list = []
    __temp_filefolder = '.'
    instance_index = 0

    def __init__(self,parser,pdbname=None):
        '''
        receive input from parser
        '''
        self.__receptor= parser.select('protein or nucleic')

        # Set type of receptor
        if parser.select('protein') is None:
            if parser.select('nucleic') is not None:
                self.__receptor_type = 'Nucleic'
        else:
            if parser.select('nucleic') is None:
                self.__receptor_type = 'Protein'
            else:
                self.__receptor_type = 'Protein_Nucleic_Complex'
        pass

        if pdbname is None:
            self.__temp_filefolder= ospath.join(temp_pdb_PREFIX,''.join(map(lambda xx:(hex(ord(xx))[2:]),osurandom(16))))
            self.__name = 'Receptor_%s' % str(self.instance_index)
        else:
            assert isinstance(pdbname,str)
            self.__temp_filefolder= ospath.join(temp_pdb_PREFIX,pdbname+'_'.join(map(lambda xx:(hex(ord(xx))[2:]),osurandom(16))))
            self.__name = '%s' % (pdbname)



    @classmethod
    def get_num_of_instance(cls):
        cls.instance_index += 1
        return cls.instance_index

    @property
    def receptor_name(self):
        '''
        This is to make sure this variable is read-only. The same to the other
        :return:
        '''
        if self.__name is None:
            raise AttributeError('No receptor is allocated to this container!')
        return self.__name

    @property
    def receptor_type(self):
        if self.__receptor_type is None:
            raise AttributeError('No receptor is allocated to this container!')
        return self.__receptor_type

    @property
    def temp_file_dir(self):
        if self.__temp_filefolder is '.':
            print 'Warning, the temporary file path is not configured! It will share the directory with this script.'
        return self.__temp_filefolder

    def Is_pure_protein(self):
        return self.__receptor_type == 'Protein'

    def Is_pure_nucleic(self):
        return self.__receptor_type == 'Nucleic'

    def Is_protein_nucleic_complex(self):
        return self.__receptor_type == 'Protein_Nucleic_Complex'



    def write_file(self,format='pdb',location=None):
        '''
        Write atoms into temporary filelocation
        For future implementation, it might be able to support different type of format, e.g. mol2
        :param format:
        :param location: Someplace other than designated temporary file location
        :return:
        '''
        pass

    def prepare_autodock_receptor(self):
        '''

        :return: filename for autodock (pdbqt file)
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


