__author__ = 'wy'

from Ligand_container import Ligand_container
from Receptor_container import Receptor_container
from atomgroupIO import writePDB,parsePDB,writeMOL2,parseMOL2

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