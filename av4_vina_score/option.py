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

def visualize_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        dest='input_file',
        type=str,
        help='path of npy file for visualize'
    )

    parser.add_argument(
        '-d',
        dest='dest_folder',
        type=str,
        default='images',
        help='path of folder to store result'
    )

    parser.add_argument(
        '-a',
        dest='fixed_axis',
        type=str,
        default='X',
        help='Slice by which axis [X,Y,Z]'
    )

    parser.add_argument(
        '-n',
        dest='images_num',
        type=int,
        default=10,
        help='Draw how many images as result'
    )

    return parser
