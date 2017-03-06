import os, sys
import numpy as np
import prody
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r',
        dest='receptor',
        type=str,
        default='t_1.pdb',
        help='Path of receptor pdb file.')
    parser.add_argument(
        '-l',
        dest='ligand',
        type=str,
        default='tt_1.pdb',
        help='Path of ligand pdb file.')

    parser.add_argument(
        '-d',
        dest='debug',
        type=str,
        default='print',
        help='debug information option [print,log,off]')

    parser.add_argument(
        '--log',
        dest='log',
        type=str,
        default='eval.log',
        help='Paht of the log file')

    return parser