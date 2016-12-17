'''
Add H in PDB protein files
'''
import os
import commands
from prody import *
from Config import PDB_tar,pdb_PREFIX,temp_pdb_PREFIX

from_dir = pdb_PREFIX
to_dir = temp_pdb_PREFIX


def add_hydrogens(filedir,pdbfilename,index):
    real_dir = os.path.join(filedir,pdbfilename)

    #cmd = 'babel -h {0} {0}'.format(real_dir)
    #os.system(cmd)
    #cmd = 'babel -d {0} {0}'.format(real_dir)
    #os.system(cmd)

    with open('repair.sh','w') as w:
        w.write('# !/bin/bash\n')
        w.write('# BSUB -n 2\n')
        w.write('# BSUB -W 100:00\n')
        w.write('# BSUB -J reapir_{}\n'.format(index))
        w.write('# BSUB -o /home/yw174/job/addH/{}.out\n'.format(index))
        w.write('# BSUB -e /home/yw174/job/addH/{}.err\n'.format(index))
        w.write('# BSUB -q long\n')
        w.write('export PATH =$PATH:/home/yw174/usr/babel/bin/\n')
        w.write('cd /home/yw174/program/pdb_sth\n')
        cmd = 'obminimize -cg -ff MMFF94 -h -n 500 {0}.pdb > {0}_hydro.pdb'.format(index)
        w.write(cmd+'\n')
    os.system('bsub < repair.sh')

def split_receptors(pdbname,src,tardir):
    print src
    try:
        parse= parsePDB(src)
    except:
        return 0
    print 'here'

    if parse.select('nucleic') is not None:
        return 1
    protein = parse.select('protein')
    if protein is None:
        return 2


    return 3

def write_it(pdbname,src,tardir,index):
    parse = parsePDB(src)
    protein = parse.select('protein')
    if not os.path.exists(tardir):
        os.makedirs(tardir)
    writePDB(os.path.join(tardir,str(index)+'.pdb'),protein)
    return True

if __name__=='__main__':

    N = 11549

    prefix = '/home/yw174/pdb_data/addHdata'
    os.remove('error.txt')

    Succ = 0
    Fail = 0

    for i in range(N):
        try:
            file = os.path.join(prefix, '{}_hydro.pdb'.format(i + 1))
            parsePDB(file)
            print str(i + 1) + ' is OK'
            Succ += 1
        except:
            print str(i + 1) + ' fails' +' ,try to move long queue'
            add_hydrogens(prefix,str(i+1)+'.pdb', str(i+1))
            with open('error.txt', 'a') as w:
                w.write(str(i + 1) + '\n')
            Fail += 1

    print 'Succ: {}, Fail: {}'.format(Succ, Fail)

