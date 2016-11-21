# Affinity Core V3

Affinity core is a starting code for training deep convolutional neural networks on crystallographic images of proteins to predict drug binding. In the simplest case, the training set consists of two sets of structures of complexes of proteins together with small molecules labeled either 1 or 0 (binders and decoys). Networks can be trained to distinguish binders from decoys when given a set of unlabeled ligand-protein structures, and to rank complexes by probability to be strong.

### scripts:

[av3_database_master.py](./av3_database_master.py)
crawls and indexes directoris, parses protein and ligand structures (.pdb files) into numpy (.npy) arrays with 4 columns. The first three columns store the coordinates of all atoms, fourth column stores an atom tag (float) corresponding to a particular element. Hashtable that determines the correspondence between chemical elements and numbers is sourced from [av3_atomdict.py](./av2_atomdict.py). 

Because the most commonly used to store molecule structures .pdb format is inherently unstable, some (~0.5%) of the structures may fail to parse. Database master handles the errors in this case. After .npy arrays have been generated, database master creates label-ligand.npy-protein.npy trios and writes them into database_index.csv file. 

Finally, database master reads database_index.csv file, shuffles it and safely splits the data into training and testing sets.

[av3.py](./av3.py)
the main script. Takes database index (train_set.csv), and the database with .npy arrays as an input. Performs training and basic evaluation of the network. Depends on av3_input.py which fills the queue with images. By default, av3 is optimizing weighted cross-entropy for a two-class sclassification problem with FP upweighted 10X compared to FN.

While running, the main script creates directoris with various outputs:

summaries/logs" - stores some of the outputs of performance
summaries/netstate" - stores the saved weights of the network
summaries/test" - stores some of the variable states for visualization in tensorboard 
summaries/train" - stores some of the variable states for visualization in tensorboard 

[av3_input.py](./av3_input.py)
generates the image queue and fills it with images
affine transform
custom filename coordinator class
multiple threads


### benchmark:
We have tested the 

Co-crystal structures of proteins and small molecules (binders) can be obtained from [Protein Data Bank](http://www.rcsb.org/).
