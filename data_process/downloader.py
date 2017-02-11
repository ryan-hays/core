import os, sys
import prody
import re
import getopt
import pandas as pd
import numpy  as np
import multiprocessing
import threading



class parseRCSB:
    def __init__(self):
       pass



    def error_log(self, content):
        with open(FLAGS.log_file, 'a') as fout:
            fout.write(content)

    def downloads(self, item):

        os.system('cd {}'.format(FLAGS.rowdata_folder))
        address = FLAGS.address(item)
        os.system('wget {}'.format(address))

        pdbname = item.lower()
        ligand_folder = os.path.join(FLAGS.splited_ligand_folder, pdbname)
        if not os.path.exists(ligand_folder):
            os.mkdir(ligand_folder)

        try:
            parsed = prody.parsePDB(os.path.join(FLAGS.rowdata_folder, item + '.pdb'))
        except:
            self.error_log('can not parse {}.\n'.format(item))
            return None

        hetero = parsed.select('(hetero and not water) or resname ATP or resname ADP')
        receptor = parsed.select('protein or nucleic')

        ligand_flags = False
        for each in prody.HierView(hetero).iterResidues():
            if each.numAtoms() <= 10:
                continue
            else:
                ligand_flags = True
                ResId = each.getResindex()
                ligand_path = os.path.join(FLAGS.splited_ligand_folder, pdbname,
                                           "{}_{}_ligand.pdb".format(pdbname, ResId))
                if not os.path.exists(os.path.dirname(ligand_path)):
                    os.mkdir(os.path.dirname(ligand_path))
                prody.writePDB(ligand_path, each)

        if ligand_flags:
            receptor_path = os.path.join(FLAGS.splited_receptor_folder, pdbname + '.pdb')
            prody.writePDB(receptor_path, receptor)
        else:
            self.error_log("{} doesn't convert, not ligand have more than 10 atoms.\n")


    def thread_convert(self, func, dataframe, index):

        for i in index:
            func(dataframe[i])

    def process_convert(self, func, dataframe, index):

        # linspace contain end value but range don't
        # so we use edge[i+1] to select value in index
        # end should be len(index)-1


        if len(index) < FLAGS.thread_num:
            for i in index:
                func(dataframe[i])
            return

        edge = np.linspace(0, len(index) - 1, FLAGS.thread_num + 1).astype(int)

        thread_list = [threading.Thread(target=self.thread_convert,
                                        args=(func,
                                              dataframe,
                                              range(index[edge[i]],
                                                    index[edge[i + 1]])))
                       for i in range(FLAGS.thread_num)]

        for t in thread_list:
            # print "thread start: ",t
            t.start()

        for t in thread_list:
            t.join()

    def convert(self):
        '''
        according the result of 'database_from_csv'
        running multiprocess to get result

        :param dataframe: pandas.DataFrame
                          string
        :param coded: bool
        :return:
        '''

        convert_func = self.downloads



        # when there's not enough entry to comvert , decrease thread's num
        if len(FLAGS.pdb_list) < FLAGS.process_num * FLAGS.thread_num:

            for i in range(len(FLAGS.pdb_list)):
                convert_func(FLAGS.pdb_list[i])
            return
        edge = np.linspace(0, len(FLAGS.pdb_list), FLAGS.process_num + 1).astype(int)
        process_list = [multiprocessing.Process(target=self.process_convert,
                                                args=(convert_func,
                                                      FLAGS.pdb_list,
                                                      range(edge[i],
                                                            edge[i + 1])))
                        for i in range(FLAGS.process_num)]

        for p in process_list:
            print "process start: ", p
            p.start()

        for p in process_list:
            print "process end: ", p
            p.join()


class FLAGS:
    workplace = '/n/scratch2/xl198/data/rcsb'
    rowdata_folder = os.path.join(workplace, 'row')
    splited_receptor_folder = os.path.join(workplace, 'row_receptor')
    splited_ligand_folder = os.path.join(workplace, 'ligands')
    address = lambda PDB: 'https://files.rcsb.org/download/' + PDB + '.pdb'
    log_file = 'error.log'
    thread_num = 16
    process_num = 12


def parse_FLAG():


    content = open('target_PDB.txt').readline()
    content = content.split(',')
    content = map(lambda x: x.strip(), content)
    FLAGS.pdb_list = content


if __name__ == '__main__':
    parse_FLAG()
    parser = parseRCSB()
    parser.convert()