'''
Scripts that will generate:
1. autodock gridmaps (vector/file output)
2. autodock docking records (file)
3. autovina score reports (dict)

Need at least following components to make this works:
1. mgltools, pythonsh environment
2. autodock vina exec
Please refer readme to understand these two dependencies

'''

class Docking_container:
    '''
    This class shall only receive the following input:
    A. filename for ligand
    B. filename for receptor
    C. Configuration
    '''
    __dump_filelist = []
    __dump_dir = '.'

    def __init__(self):
        pass

    def prepare_ligand_file(self):
        pass

    def prepare_receptor_file(self):
        pass

    def do_autogrid(self):
        pass

    def get_autogrid_vectors(self,applied_suffix=['A','C','d','e','HD','N','NA','OA']):
        pass

    def do_autodock(self):
        pass

    def do_autovina(self):
        pass

    def get_autovina_scoredict(self):
        pass

    def __del__(self):
        pass