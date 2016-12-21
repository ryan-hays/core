'''

Pipeline part I:

How to extract data from Experimental one and other ligands on pdb (cocrystal structure for complex bindings
yet they might not have experimental data like: IC50, EC50

The aim of this script is to return a tuple dict that contains three parts:

PDB information
Ligand information
Experimental data and similarity scores (>=0.85 thredshold with tanimoto similarity)

The usage example will show how to use this information (maybe exactly as how we did for creating a csv files)

The format to output is merely decided by the user(hopefully)

'''

__author__ = 'wy'