# MIT Affinity --- Data Preprocessing


## Intro

This is the program to convert source file into the vector
that will be selected and feed into the network

The program aims at the following goals:
*   Get source sdf files and split them by pdbnames
*   For each RCSB protein ID (i.e. 4 character ID), prepare experimental data
*   In the meantime, seperate ligans on pdb and send them to the pipeline that will be docking to generate unlabeled samples
*   Get result from smina and mark them as the input files.

## Usage

To be continue