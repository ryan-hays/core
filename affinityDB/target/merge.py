"""

remove redundant structure and ligand entry

usage: python merge.py
"""
import os
import sys


def merge():
    """
    find all ligands record file under this folder and merge them
    find all structures record file under this folder and merge them
    :return:
    """

    FILE_LIST = os.listdir(os.getcwd())

    strucs = set([])
    for file_name in FILE_LIST:
        if file_name.endswith('_structures.txt'):
            tmp = open(file_name).readline().strip().split(', ')
            tmp = set(tmp)
            print "read {}, get {} structure entrys.".format(file_name, len(tmp))
            strucs = strucs | tmp

    with open("structures.txt","w") as fout:
        fout.write(','.join(strucs))

    print "Get {} unique structure entry.".format(len(strucs))

    ligands = set([])
    for file_name in FILE_LIST:
        if file_name.endswith('_ligands.txt'):
            tmp = open(file_name).readline().strip().split(' ')
            tmp = set(tmp)
            print "read {}, get {} ligand entrys.".format(file_name, len(tmp))
            ligands = ligands | tmp

    with open("ligands.txt","w") as fout:
        fout.write(','.join(ligands))

    print "Get {} unique ligands entry.".format(len(ligands))



merge()

