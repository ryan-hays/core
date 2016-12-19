'''
1. To parse pdb file and write pdb file
2. To convert output into some other format like .mol2 .sdf
3. To parse results from smina (.mol2)

'''

import prody as pd
import os.path as ospath
import subprocess
from os import remove as osremove
from os import getcwd as osgetcwd

BABEL_EXCECUTE = './bin/babel'

def writePDB(parse,filename,dir=None):
    '''
    Replace prody.writePDB, can be implemented later here
    :param parse:
    :param filename:
    :param dir:
    :return:
    '''
    try:
        if dir is None:
            pd.writePDB(filename,parse)
        else:
            pd.writePDB(ospath.join(dir,filename),parse)
    except:
        raise IOError('Cannot write %s into pdb file!' % repr(parse))

def parsePDB(pdbname,filename=None,dir=None,header=False):
    '''
    Replace prody.parsePDB here, can be implemented later here.
    If something is wrong with name or filelocation, the program will not halt
    but will instead return None as result
    :param parse:
    :param file:
    :return: parse:atomgroup header:information in header of the .pdb file
    '''

    try:
        if filename is not None:
            if dir is None:
                if header==True:
                    parse, header = pd.parsePDB(filename, header=header)
                else:
                    parse = pd.parsePDB(filename)
                    header= None
            else:
                if header==True:
                    parse, header = pd.parsePDB(ospath.join(dir,filename), header=header)
                else:
                    parse = pd.parsePDB(ospath.join(dir,filename))
                    header= None
        else:
            #try parse pdb by name
            if header == True:
                parse, header = pd.parsePDB(filename, header=header)
            else:
                parse = pd.parsePDB(filename)
                header = None
            if ospath.exists('%s.pdb.gz' % pdbname):
                osremove('%s.pdb.gz' % pdbname)
    except:
        return None,None
    return parse,header

def writeMOL2(parse,filename,dir=None):
    '''

    :param parse:
    :param filename:
    :param dir:
    :return:
    '''
    assert isinstance(filename,str)

    if dir is None:
        filepos = filename
    else:
        filepos = ospath.join(dir, filename)
    filepath,tempfilename= ospath.split(filepos)
    text,ext= ospath.splitext(tempfilename)
    pdb_filename= ospath.join(filepath,'%s.pdb'%text)


    tag=ospath.exists(pdb_filename)

    if tag==True:
        try:
            f = open(filepos,'w')
            test = subprocess.call('%s -ipdb %s -omol2 %s' % (BABEL_EXCECUTE, pdb_filename, filepos),
                                   shell=True,stdout=f)
            if test==0:
                return
        except:
            print 'File with .pdb is not a valid file, need to generate .pdb file first then convert it into mol2 file!'

    #Now tag==False or cannot use .pdb file appropriately
    pdb_filename+='.tmp'
    writePDB(parse,filename=pdb_filename,dir=dir)
    try:
        f = open(filepos, 'w')
        test =subprocess.call('%s -ipdb %s -omol2 %s' % (BABEL_EXCECUTE,pdb_filename,filepos),shell=True,stdout=f)
        if test!=0:
            raise ValueError('subprocess of babel did not exit properly!')
    except:
        if ospath.exists(pdb_filename):
            osremove(pdb_filename)
        raise RuntimeError('Errors while running babels, check your file or the filelocation you provided!')

    #Remove temp file
    if ospath.exists(pdb_filename):
        osremove(pdb_filename)

def parseMOL2(token,filename,dir=None):
    '''

    :param token: The name or the tag for the file (i.e. sth like 1avd_248, chemical name , terminology, etc.
    :param filename:
    :param dir:
    :return:
    '''


    assert isinstance(filename, str)

    if dir is None:
        filepos = filename
    else:
        filepos = ospath.join(dir, filename)
    filepath, tempfilename = ospath.split(filepos)
    text, ext = ospath.splitext(tempfilename)
    pdb_filename = ospath.join(filepath, '%s.pdb.tmp' % text)

    tag = ospath.exists(pdb_filename)

    #First convert mol2 file into pdb file:
    if tag == False:
        try:
            f = open(pdb_filename, 'w')
            subprocess.call('%s -imol2 %s -opdb %s' % (BABEL_EXCECUTE, filepos, pdb_filename), shell=True, stdout=f)
        except:
            print 'File with .mol2 is not a valid file!'

    try:
        parse, header = parsePDB(token, filename=pdb_filename)
    except:
        if ospath.exists(pdb_filename):
            osremove(pdb_filename)
        raise RuntimeError('Errors while parsing pdb files! Needs debug')

    # Remove temp file
    if ospath.exists(pdb_filename):
        osremove(pdb_filename)

    return parse, header

if __name__=='__main__':
    path = ospath.join(ospath.dirname(osgetcwd()),'data')
    parse, header = parsePDB('xxxx',filename='fake-ligand.pdb',dir=path)
    #print parse,header
    writeMOL2(parse,filename='fake-ligand.mol2',dir=path)
    parse, header = parseMOL2('fake',filename='fake-ligand.mol2',dir=path)
    print parse
    writePDB(parse, filename='fake-ligand.pdb.backup',dir=path)
