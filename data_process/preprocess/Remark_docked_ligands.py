'''

Pipeline part II:

For each specific ligand-receptor pair, you will have a record on RCSB protein bank in a pdb file. This program aims
at:

1. do docking at this pair (now we are trying to do docking in four method:
fast , rigorous_siteonly, rigorous_wholeprotein and random docking)
2. For the result, we will calculate its difference with original one by calculating RMSD and native contact, and
use Autodock vina to score this docking result. (For training, evaluating, kinds of purpose)

As a usage example, it will read and remarks on mol2 files so that everyone can parse remarks to get these features

#TODO implement the API for docking decoration
'''