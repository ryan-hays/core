# Affinity Core V3

Affinity core is a starting code for training deep convolutional neural networks on crystallographic images of proteins to predict drug binding. In the simplest case, the training set consists of two folders of co-crystal structures of proteins and small molecules labeled either 0 or 1 (binders and decoys). Network can be trained to distinguish between those two classes when given an uknown 3D structure of the complex in the future.

# scripts:

av3_database_master.py 
crawls and indexes directoris, parses protein structures (PDB files) into numpy arrays with 4 columns. The first three columns store coordinates of all atoms, fourth column stores an atom tag (float) corresponding to a particular element. Hashtable to determine the correspondence between chemical elements and numbers is sourced from av3_atomdict.py

