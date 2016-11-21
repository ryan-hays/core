# Affinity Core V3

Affinity core is a starting code for training deep convolutional neural networks on crystallographic images of proteins to predict drug binding. In the simplest case, the training set consists of two sets of structures of complexes of proteins together with small molecules labeled either 1 or 0 (binders and decoys). Networks can be trained to distinguish binders from decoys when given a set of unlabeled ligand-protein structures, and to rank complexes by probability to be strong.

### scripts:

***[av3_database_master.py]***(./av3_database_master.py)
crawls and indexes directoris, parses protein and ligand structures (.pdb files) into numpy (.npy) arrays with 4 columns. The first three columns store coordinates of all atoms, fourth column stores an atom tag (float) corresponding to a particular element. Hashtable that determines the correspondence between chemical elements and numbers is sourced from [av3_atomdict.py](./av2_atomdict.py). 

Because the most commonly used to store molecule structures .pdb format is inherently unstable, some (~0.5%) of the structures may fail to parse. Database master handles the errors in this case. After .npy arrays have been generated, database master creates label-ligand.npy-protein.npy trios and writes them into database_index.csv file. 

Finally, database master reads database_index.csv file, shuffles it and safely splits it into training and testing sets.



### benchmark:
We have tested the 

Co-crystal structures of proteins and small molecules (binders) can be obtained from [Protein Data Bank](http://www.rcsb.org/).
