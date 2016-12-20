__author__ = 'wy'

import csv
import logging
import os
import time
import urllib
from functools import wraps

from Config import *
from data_process.preprocess.utility.autodock_utility import repair_pdbfile
from data_process.preprocess.utility.Receptor_container import pdb_container

'''
The main program to extract molecules in .sdf files and compare with ligands on PDB files.
Then accept all pairs with similarity >= 0.85 and generate the corresponding vectors.
'''

#This part is used to set debug log
#This will generate a log that record every content logged with a specific security levels
fileHandler = logging.FileHandler('debug.log',mode='w')
fileHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('LINE %(lineno)-4d  %(levelname)-8s %(message)s', '%m-%d %H:%M')
fileHandler.setFormatter(formatter)
logging.getLogger('').addHandler(fileHandler)

def list_formatter(table):
    '''
    I don't know if there is a better solution to format a list into string
    :param table:
    :return:
    '''
    try:
        output='['+str(table[0])
        for i in range(len(table)-1):
            output+=(','+str(table[i+1]))
        output+=']'
    except:
        raise TypeError('This object is not iterable!')
    return output



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
        logging.warning ("Total time running %s: %s seconds" %
               (function.func_name, str(t1 - t0))
               )
        return result

    return function_timer



@fn_timer
def mol_ligand_tar_generator(src,filepos,statistic_csv=None,CLEAN=False,fileforbabel='a.sdf'):
    '''

    :param src: pdb name
    :param statistic_csv: the report csv file's name
    :param CLEAN: Wipe temporary pdb files or not. Note I will not give options to wipe results. That's dangerous
    :return: True: If everything works fine
             False: Unexpected error happens. Note if there is no reuslt, it will return True because everything runs fine.
    '''

    #Wipe the pdb temporary files if you wish:
    if CLEAN:
        files = os.listdir('data/')
        for filename in files:
            loc = os.path.join('data/') + filename
            if os.path.exists(loc):
                os.remove(loc)

    # write the result

    result_file_name ='filter_{}'.format(src.split('/')[-1].split('.')[0])+'.csv'
    filedir = os.path.join(result_PREFIX,result_file_name)
    if not os.path.isfile(filedir):
        if not os.path.exists(result_PREFIX):
            os.mkdir(result_PREFIX)

    # in case for overwriting
    '''
    if os.path.exists(filedir):
        print '{} already done.'.format(src)
        logging.info('{} already done'.format(src))
        return True
    '''

    # csv writer
    writer = file(filedir, 'w')
    w = csv.writer(writer)
    w.writerow(experiment_part+PDB_part)
    print len(experiment_part+PDB_part)

    # combine as file direction
    sdfone = filedir_PREFIX + src.upper() + '.sdf'

    #open the source molecule files
    #Naming format [PDB name].sdf all lowercase
    try:
        input_sdf = open(sdfone,'r')
    except:
        logging.error('PDB {} with ligands sdf not found!'.format(src))
        return False


    # This variables are used for counting and statistic issue.
    active_count=0
    count=0
    bad_one=0

    # Combine as pdb file address
    # We generate a class to store each ligands as a dict, and use a method
    # to find the similar ones by tanimoto comparing scores to specific input files
    PDBindex= pdb_container(src,filepos=filepos)
    if PDBindex.get_pdb_type()!='Protein':
        return False
    # In each time
    # We write one single molecule in to a sdf file called a.sdf
    # Then use this file to compare with the one we extract from pdb files
    # Since we use scripts so I just extract them to make sure that is what
    # we really want to compare
    # (But there should be a smarter way to do so)

    try:
        mol = ''
        LINE_BEGIN=True
        Wait_Signal= 0
        experimental_data = {}
        one_line=['']*len(experiment_part)
        #print 'here'
        for line in input_sdf:
            mol+=line

            #This part is finding columns need to be recorded in sdf files
            if LINE_BEGIN:
                one_line[0] = line.lstrip(' ').rstrip('\n')
                LINE_BEGIN = False

            # just lazy
            if Wait_Signal>0:
                if Wait_Signal==999:
                    one_line[1]=line.lstrip(' ').rstrip('\n')
                    monomerID = one_line[1]
                else:
                    one_line[2+Wait_Signal]= line.lstrip(' ').rstrip('\n')
                Wait_Signal= 0

            for i in range(len(key)):
                if key[i] in line:
                    Wait_Signal=i+1
                    break

            if NAME in line:
                Wait_Signal=999

            if '$$$$' in line:
                #end of a molecule
                assert monomerID is not None
                if monomerID not in experimental_data:
                    fileforbabel = temp_pdb_PREFIX+'/{}/{}_{}.sdf'.format(src,src,monomerID)
                    o = open(fileforbabel, "wb")
                    o.write(mol)
                    o.close()
                    experimental_data[monomerID]=one_line
                    one_line = [''] * len(experiment_part)
                else:
                    #combine experimental data together
                    for i in range(len(key)):
                        if experimental_data[monomerID][3+i]=='':
                            experimental_data[monomerID][3 + i] = one_line[3+i]
                        else:
                            if len(one_line[3+i])>0:
                                experimental_data[monomerID][3+i] += '|'+one_line[3+i]




                mol = ''
                LINE_BEGIN=False
                monomerID = None

        for k,v in experimental_data.items():
            #print k,v
            fileforbabel  = temp_pdb_PREFIX+'/{0}/{0}_{1}.sdf'.format(src,k)
            # Find pairs with at least 85% similarity scores
            ans_list = PDBindex.find_similar_target(fileforbabel)
            #print 'here'
            count += 1
            for eachone in ans_list:
                # Combine each part together
                v[2] = eachone['cp']
                active_count += 1
                w.writerow(v + PDBindex.bundle_result(eachone['id']))

            if len(ans_list) == 0:
                bad_one += 1
                logging.info('not found ligand here: {}_{}.'.format(src, one_line[1]))

    except:
        #raise TypeError
        logging.error('Unknown error here!')
        return False
    logging.warning('{} bad ligands found'.format(bad_one))
    logging.warning('{} molecules are detected, and {} pairs are recorded.'.format(count,active_count))

    #Discard unused one
    PDBindex.clean_temp_data()

    writer.flush()
    writer.close()

    #Do some record
    if statistic_csv is not None:
        writer = file(statistic_csv, 'ab')
        w = csv.writer(writer)
        w.writerow([src,count,count-bad_one,bad_one,active_count])
        writer.flush()
        writer.close()
    return True

def do_one_pdb(pdb,filename=None,REPORTCSV=None,index=0):
    '''
    For each target-complex pdb , this program check if .pdb file exists
    if not ,download first then call the function to match all possible target-ligands with
    molecules in sdf files in one single pdb
    :param pdb:name
    :parameter REPORTCSV: sometimes generate a list of report with the filename this one
    :return:
    '''

    # For each pdb name , there should be one corresponding molecule files.
    # This will generate one result file.
    pdb = pdb.lower()

    #if os.path.exists(os.path.join(temp_pdb_PREFIX,pdb)):
    #    return

    if filename is None:
        filename = os.path.join(pdb_PREFIX,'{}.pdb.gz'.format(pdb))
    if os.path.exists(filename):
        # pdbfile exists
        logging.info(pdb + ' has already exists')
        return mol_ligand_tar_generator(pdb,filename,statistic_csv=REPORTCSV,fileforbabel='{}.sdf'.format(index))

    else:
        # Not exists, download from the internet
        urllib.urlretrieve(url_prefix + '{}.pdb.gz'.format(pdb.lower()), filename)
        # Wait for 1 second from rejection on connection.
        time.sleep(1)

        # This is to check whether we download the file successfully
        o = open(filename, 'r')
        for l in o:
            if l.find('DOCTYPE') != -1:
                print 'download {} failed'.format(pdb)
                logging.error('download {} failed'.format(pdb))
                return False
            else:
                #If we download files successfully, then we will run the program
                print 'download {} successfully'.format(pdb)
                logging.info('download {} successfully'.format(pdb))
                return mol_ligand_tar_generator(pdb,filename,statistic_csv=REPORTCSV,fileforbabel='{}.sdf'.format(index))
        o.close()

def initiate_report():
    csv_name = 'report.csv'
    writer = file(csv_name, 'wb')
    w = csv.writer(writer)
    w.writerow(['filename','pdb Name','molecules','paired','bad_one','pairtimes'])
    return csv_name

def quick_split(pdb):
    pdb = pdb.lower()
    fake_pdb_container(pdb,filepos=os.path.join(pdb_PREFIX,pdb+'.pdb.gz'))


if __name__ == '__main__':

    DONE=[]
    FAIL=[]
    ct=0
    report = initiate_report()

    for pdb in PDB_tar[0:100]:
        #dirty way to do small scale tests
        #Use a count variable
        pdb =pdb.lower()
        real_dir = repair_pdbfile(os.path.join(pdb_PREFIX,'{}.pdb'.format(pdb)))

        if do_one_pdb(pdb,filename=real_dir,REPORTCSV=report):
            DONE.append(pdb)
        else:
            FAIL.append(pdb)
        ct+=1

    print ct
    logging.info('total: {}'.format(ct))
