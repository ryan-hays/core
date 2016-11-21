# Affinity Core V3

Affinity core is a starting code for training deep convolutional neural networks on crystallographic images of proteins to predict drug binding. In the simplest case, the training set consists of two sets of structures of complexes of proteins together with small molecules labeled either 1 or 0 (binders and decoys). Networks can be trained to distinguish binders from decoys when given a set of unlabeled ligand-protein structures, and to rank complexes by probability to be stable.

Co-crystal structures of proteins and small molecules (binders) can be obtained from [Protein Data Bank](http://www.rcsb.org/).

### scripts:

[av3_database_master.py](./av3_database_master.py)
crawls and indexes directoris, parses protein structures (PDB files) into numpy arrays with 4 columns. The first three columns store coordinates of all atoms, fourth column stores an atom tag (float) corresponding to a particular element. Hashtable to determine the correspondence between chemical elements and numbers is sourced from [av3_atomdict.py](./av2_atomdict.py)

[

