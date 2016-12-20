'''
This script is used to generate docking result (electronic map) and calculating docking scores
'''

import sys,io,os
from functools import wraps
import time
import re
import commands
from Autodock_Config import autodock_store_dir,pythonsh_dir
temp_pdb_PREFIX = '/tmp'
#temp_pdb_PREFIX = '/home/wy/Documents/BCH_coding/pdb_data_extracter/data'
import gzip

WORK_DIR = os.getcwd()
CURRENT_DIR = os.getcwd()+'/autodock_utility'
#os.chdir(CURRENT_DIR)

BUFSIZE = 1024 * 8

class GZipTool:
    def __init__(self, bufSize):
        self.bufSize = bufSize
        self.fin = None
        self.fout = None

    def compress(self, src, dst):
        self.fin = open(src, 'rb')
        self.fout = gzip.open(dst, 'wb')

        self.__in2out()

    def decompress(self, gzFile, dst):
        self.fin = gzip.open(gzFile, 'rb')
        self.fout = open(dst, 'wb')

        self.__in2out()

    def cp(self, src ,dst):
        self.fin = open(src,'rb')
        self.fout = open(dst,'wb')
        self.__in2out()

    def __in2out(self, ):
        while True:
            buf = self.fin.read(self.bufSize)
            if len(buf) < 1:
                break
            self.fout.write(buf)

        self.fin.close()
        self.fout.close()

def pdb_to_mol2(src,tar, addH=False):
    '''
    convert pdb ligands into mol2 files
    :param src:
    :param tar:
    :return:
    '''
    if addH==True:
        cmd = 'babel -h -ipdb {} -omol2 {} '.format(src, tar)
    else:
        cmd = 'babel -ipdb {} -omol2 {} '.format(src, tar)
    #print cmd
    os.system(cmd)
    return True

def set_new_folder(PDBname,storedir):
    '''
    :param PDBname:
    :param storedir:
    :return:
    '''
    #os.chdir(storedir)
    if not os.path.exists(os.path.join(storedir,PDBname)):
        os.mkdir(os.path.join(storedir,PDBname))
    #os.chdir(os.getcwd())

def fn_timer(function):
    '''
    This is the decorator used for time counting issue
    Need not understand this one. It has nothing to do with generating files
    :param function:
    :return: no return. just print and record the time the decorated program ran.
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1 - t0))
               )
        return result

    return function_timer

def copy_pdbfile(filepos,tarpos,zipped=False):

    zipped = (filepos.split('.')[-1]=='gz')
    if zipped:
        tool = GZipTool(BUFSIZE)
        try:
            tool.decompress(filepos,tarpos)
        except:
            raise TypeError("unable to decompress at "+filepos)
    else:
        tool = GZipTool(BUFSIZE)
        try:
            tool.cp(filepos, tarpos)
        except:
            raise TypeError


@fn_timer
def repair_pdbfile(filename,pdbname,OVERWRITE=False):
    '''
    repair pdbfiles with add hydrogen
    :param filename:
    :param OVERWRITE:
    :return:
    '''

    #os.chdir(CURRENT_DIR)
    #cmd =os.path.join(pythonsh_dir, 'pythonsh') + ' prepare_receptor4.py -v -r {0} -o {0}qt -A bonds_hydrogens -U nphs_lps_waters'.format(real_filepos)
    cmd ='babel -h {} {} '.format(filename,filename)
    stat ,out = commands.getstatusoutput(cmd)

    if stat == 256:
        print out
        return filename

    #cmd = os.path.join(pythonsh_dir, 'pythonsh') + ' pdbqt_to_pdb.py -f {0}qt -o {0}'.format(real_filepos)
    # cmd ='obabel {} -opdb -O {} -h'.format(real_filepos,real_filepos)
    #stat, out = commands.getstatusoutput(cmd)
    #print stat, out
    #if stat == 256:
    #    print out
    #    return 'NA'

    #os.chdir(WORK_DIR)

    #print real_filepos
    #Convert into pdb files
    #os.system('cut -c-66 {} > {}'.format(real_filepos+'qt',real_filepos))
    #os.remove(real_filepos+'qt')

    #return os.path.join(os.getcwd(),real_filepos)
    return filename

def prepare_receptor(filedir,filename,pdbname,pdbresid='',OVERWRITE=True,repair=False):
    '''
    prepare receptor pdbqt files
    :param filename: the file name
    :param pdbname:  pdbname (used for naming)
    :param OVERWRITE: False = not overwrite existing file , True= can overwrite
    :return:
    '''
    if filename.split('.')[-1]!='pdb':
        print 'Error! when prepare receptor'
        return False
    real_dir = filedir
    real_filepos= os.path.join(real_dir,filename.split('/')[-1])+'qt'
    if not os.path.exists(real_filepos) or OVERWRITE:
        os.chdir(CURRENT_DIR)
        if repair == True:
            cmd = os.path.join(pythonsh_dir,
                               'pythonsh') + ' prepare_receptor4.py -r {0} -o {1} -A bonds_hydrogens -U nphs_lps_waters'.format(filename,real_filepos)
        else:
            cmd =os.path.join(pythonsh_dir, 'pythonsh') + ' prepare_receptor4.py -r {0} -o {1} -U nphs_lps_waters'.format(filename, real_filepos)
        #print cmd
        stat ,out = commands.getstatusoutput(cmd)
        #print stat,out
        os.chdir(WORK_DIR)
        if stat==256:
            print out
            return False
    #print 'Ok'
    return True


def prepare_ligand(filedir,filename,pdbname,pdbresid='',OVERWRITE=False):
    '''
    prepare ligand pdbqt files ( note different from receptor's)
    :param filename: the file name
    :param pdbname: pdbname (used for naming)
    :param OVERWRITE: False = not overwrite existing file , True= can overwrite
    :return:
    '''

    if filename.split('.')[-1]!='pdb':
        print 'Error! when prepare ligand'
        return False
    real_dir =  filedir
    real_filepos = os.path.join(real_dir, filename.split('/')[-1]) + 'qt'
    if not os.path.exists(real_filepos) or OVERWRITE:
        os.chdir(CURRENT_DIR)
        cmd =os.path.join(pythonsh_dir, 'pythonsh') + ' prepare_ligand4.py -A bonds_hydrogens -l {0} -o {1} -g'.format(filename, real_filepos)
        stat ,out = commands.getstatusoutput(cmd)
        os.chdir(WORK_DIR)
        if stat==256:
            print out
            return False

    #print 'Ok'
    return True

@fn_timer
def do_auto_grid(filedir,receptor,ligand,center=None,BOX_size=1,BOX_num=21):
    #extract names
    rname = receptor.split('/')[-1]
    lname = ligand.split('/')[-1]
    pdbname = rname.split('_')[0]
    pdbresid = rname.split('_')[1]

    if not os.path.exists(receptor) or not os.path.exists(ligand):
        return False

    #prepare receptor
    if rname.split('.')[-1]=='pdb':
        if not prepare_receptor(filedir,receptor,pdbname,pdbresid):
            return False
        rname+='qt'
    else:
        if rname.split('.')[-1]!='pdbqt':
            return False

    #prepare ligand
    if lname.split('.')[-1] == 'pdb':
        if not prepare_ligand(filedir,ligand,pdbname,pdbresid):
            return False
        lname+='qt'
    else:
        if lname.split('.')[-1] != 'pdbqt':
            return False

    # get absolute names and locations
    naming = "".join(rname.split('.')[:-1])
    real_dir = filedir
    glg_output_dir = os.path.join(real_dir,naming)

    rloc = os.path.join(real_dir,rname)
    lloc = os.path.join(real_dir,lname)

    os.chdir(CURRENT_DIR)

    # prepare gpf files with customized parameters
    if center is None:
        cmd = os.path.join(pythonsh_dir, 'pythonsh') + \
              ' prepare_gpf4.py -l {} -r {} -o {}.gpf -p spacing={} -p npts=\"{},{},{}\" '.format(lloc,
                rloc,glg_output_dir,BOX_size,BOX_num,BOX_num,BOX_num)
        stat, out = commands.getstatusoutput(cmd)
        if stat == 256:
            os.chdir(WORK_DIR)
            return False
    else:
        cmd = os.path.join(pythonsh_dir,'pythonsh') + \
                  ' prepare_gpf4.py -l {} -r {} -o {}.gpf -p spacing={} ' \
                  '-p npts=\"{},{},{}\" -p gridcenter=\"{},{},{}\" '.format(lloc,
                    rloc ,glg_output_dir, BOX_size,BOX_num,BOX_num,BOX_num,center[0],center[1],center[2])
        stat, out = commands.getstatusoutput(cmd)
        if stat == 256:
            os.chdir(WORK_DIR)
            return False

    #Suppose autogrid and autodock has installed
    os.chdir(real_dir)
    cmd = 'autogrid4 -p {0}.gpf -l {0}.glg'.format(naming)

    #Anything goes wrong , return False
    stat, out = commands.getstatusoutput(cmd)
    os.chdir(WORK_DIR)
    if stat==256:
        print out
        return False


    #print 'Ok'
    return True


@fn_timer
def do_auto_dock(filedir,receptor,ligand,center=None):
    rname = receptor.split('/')[-1]
    lname = ligand.split('/')[-1]
    pdbname = rname.split('_')[0]
    pdbresid = rname.split('_')[1]

    if not os.path.exists(receptor) or not os.path.exists(ligand):
        return False

    #first prepare auto_grid maps
    do_auto_grid(filedir,receptor,ligand,center)

    #prepare receptor
    if rname.split('.')[-1] == 'pdb':
        if not prepare_receptor(filedir,receptor,pdbname,pdbresid):
            return False
        rname += 'qt'
    else:
        if rname.split('.')[-1] != 'pdbqt':
            return False

    #prepare ligand
    if lname.split('.')[-1] == 'pdb':
        if not prepare_ligand(filedir,ligand,pdbname,pdbresid):
            return False
        lname += 'qt'
    else:
        if lname.split('.')[-1] != 'pdbqt':
            return False
    os.chdir(CURRENT_DIR)

    #This part is just to get the absolute direction
    #Because some scripts can only detect files in their direction
    #which is not a good news
    naming = "".join(rname.split('.')[:-1])
    filedir = receptor.split('/')[:-1]
    dlg_output_dir = os.path.join(filedir, naming)

    rloc = os.path.join(filedir, rname)
    lloc = os.path.join(filedir, lname)

    #prepare dpf files
    cmd=os.path.join(pythonsh_dir,'pythonsh') + \
        ' prepare_dpf4.py -l {} -r {} -o {}.dpf'.format(lloc, rloc, dlg_output_dir )
    stat, out = commands.getstatusoutput(cmd)
    # If anything goes wrong , return False
    if stat == 256:
        os.chdir(WORK_DIR)
        return False

    # Suppose autogrid and autodock has installed
    os.chdir(filedir)
    # Do real auto dock
    cmd = 'autodock4 -p {0}.dpf -l {0}.dlg'.format(naming)
    stat, out = commands.getstatusoutput(cmd)
    os.chdir(WORK_DIR)
    if stat == 256:
        return False


    return True


@fn_timer
def do_auto_vina_score(filedir,receptor,ligand,center,Box=20):
    # receptor_file_loc = os.path.join('data/',self.PDBname+'_{}_2.pdb'.format(ResId))
    #extract filename we want
    rname = receptor.split('/')[-1]
    lname = ligand.split('/')[-1]
    pdbname = rname.split('_')[0]
    pdbresid = rname.split('_')[1]
    #prepare receptor
    if not os.path.exists(receptor) or not os.path.exists(ligand):
        return 'NA'
    if rname.split('.')[-1]=='pdb':
        if not prepare_receptor(filedir,receptor,pdbname,pdbresid):
            return 'NA'
        rname+='qt'
    else:
        if rname.split('.')[-1]!='pdbqt':
            return 'NA'

    #prepare ligand
    if lname.split('.')[-1] == 'pdb':
        #print 'here'
        if not prepare_ligand(filedir,ligand,pdbname,pdbresid,OVERWRITE=True):
            return 'NA'
        lname+='qt'
    else:
        if lname.split('.')[-1] != 'pdbqt':
            return 'NA'

    #print 'here'
    # get the absolute location
    real_dir = filedir
    os.chdir(real_dir)
    # write config files
    with open('vina_config.txt', 'w') as f:
        f.write('    receptor = {}\n'.format(rname))
        f.write('    ligand = {}\n'.format(lname))
        f.write('    center_x = {}\n'.format(center[0]))
        f.write('    center_y = {}\n'.format(center[1]))
        f.write('    center_z = {}\n'.format(center[2]))
        f.write('    size_x = {}\n'.format(Box))
        f.write('    size_y = {}\n'.format(Box))
        f.write('    size_z = {}\n'.format(Box))
        #f.write('    cpu = 1\n')
        f.close()

    # Now do docking:
    # Suppose vina is installed
    command = os.popen('vina --config vina_config.txt --score_only')

    os.chdir(WORK_DIR)

    dict_key = ('Affinity','gauss 1','gauss 2','repulsion','hydrophobic','Hydrogen')

    Ans = {}

    for k in dict_key:
        Ans[k]='NA'

    # find the score in result
    ls = command.read()
    print ls
    for line in ls.split('\n'):
        if '#' in line:
            continue
        for each in dict_key:
            if each in line:
                # find the real number in this line
                real_num = re.compile(r"[-+]?\d+\.\d+")
                score = real_num.search(line.split(':')[1])
                if score:
                    Ans[each]=float(score.group())
                else:
                    Ans[each]='NA'

    return Ans

def vector_from_gridmap(mapfilename,BOX=21):
    '''
    Get the
    :param mapfilename:
    :return:
    '''
    try:
        with open(mapfilename,'rb') as f:
            ct=0
            answer=[]
            for line in f:
                ct+=1
                if ct>=7:
                    answer.append(float(line.rstrip('\n')))
            return answer
    except:
        return 'NA'

def fetch_gridmaps(filedir, map_prefix ,BOX=21):
    '''
    Convert group of gridmaps into vectors (8*21*21*21 for now)
    :param map_prefix: the file prefix , since autogrid's naming rules is same, so just provide anyname except .[].map
    :param BOX:  Boxsize
    :return:
    '''
    type= ['A','C','d','e','HD','N','NA','OA']
    vectors= []
    try:
        for each in type:
            real_dir = filedir
            real_pos= os.path.join(real_dir,map_prefix+'.'+each+'.map')
            vectors.append(vector_from_gridmap(real_pos,BOX=BOX))
        return vectors
    except:
        return 'NA'

if __name__=='__main__':
    #Example on how to finish auto docking process

    set_new_folder('1j8q','/home/wy/Documents/BCH_coding/pdb_data_extracter/result')

    #protein only
    do_auto_dock('/home/wy/Documents/BCH_coding/pdb_data_extracter/data/1j8q_147_pure.pdb',
                 '/home/wy/Documents/BCH_coding/pdb_data_extracter/data/fake-ligand.pdb',center=[21.36,10.47,81.86])

    #ligand only
    do_auto_dock('/home/wy/Documents/BCH_coding/pdb_data_extracter/data/1j8q_147_ligand.pdb',
                 '/home/wy/Documents/BCH_coding/pdb_data_extracter/data/fake-ligand.pdb', center=[21.36, 10.47, 81.86])

    #protein-ligand complex
    do_auto_dock('/home/wy/Documents/BCH_coding/pdb_data_extracter/data/1j8q_147_complex.pdb',
                 '/home/wy/Documents/BCH_coding/pdb_data_extracter/data/fake-ligand.pdb', center=[21.36, 10.47, 81.86])