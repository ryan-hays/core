# Affinity Core V3

Affinity core is a starting code for training deep convolutional neural networks on crystallographic images of proteins to predict drug binding. In the simplest case, the training set consists of two sets of structures of complexes of proteins together with small molecules labeled either 1 or 0 (binders and decoys). Networks can be trained to distinguish binders from decoys when given a set of unlabeled ligand-protein structures, and to rank complexes by probability to be strong.

### scripts:
**database_master** [av3_database_master.py](./av3_database_master.py)

crawls and indexes directoris, parses protein and ligand structures (.pdb files) into numpy (.npy) arrays with 4 columns. The first three columns store the coordinates of all atoms, fourth column stores an atom tag (float) corresponding to a particular element. Hashtable that determines the correspondence between chemical elements and numbers is sourced from [av3_atomdict.py](./av2_atomdict.py). 

Because the most commonly used to store molecule structures .pdb format is inherently unstable, some (~0.5%) of the structures may fail to parse. Database master handles the errors in this case. After .npy arrays have been generated, database master creates label-ligand.npy-protein.npy trios and writes them into database_index.csv file. 

Finally, database master reads database_index.csv file, shuffles it and safely splits the data into training and testing sets.

***av3*** [av3.py](./av3.py)

the main script. Takes database index (train_set.csv), and the database with .npy arrays as an input. Performs training and basic evaluation of the network. Depends on av3_input.py which fills the queue with images. By default, av3 is optimizing weighted cross-entropy for a two-class sclassification problem with FP upweighted 10X compared to FN.
<pre>
tf.nn.weighted_cross_entropy_with_logits()
</pre>

While running, the main script creates directoris with various outputs:
<pre>
/summaries/x_logs       # stores some of the outputs of performance
/summaries/x_netstate   # stores state of the network
/summaries/x_test       # stores some of the variable states for visualization in tensorboard 
/summaries/x_train      # stores some of the variable states for visualization in tensorboard
</pre> 

***av3_input*** [av3_input.py](./av3_input.py)

handles data preprocessing, starts multiple background threads to convert protein and drug coordinates into 3d images of pixels. Each of the background workers performs the following procedures:

1. reads the ligand from .npy file
2. randomly initializes the box nearby the center of mass of the ligand
3. rotates and shifts the box until all of the ligand atoms can fit
4. reads the protein
5. crops the protein to the cube
6. rounds coordinates and converts sparse tensor(atoms) to dense(pixels)
7. enqueues image and label

In order to organize the reading of each protein-ligand pairs in random order, but only once by a single worker during one epoch, and also to count epchs, custom

<pre>
filename_coordinator_class()
</pre> 
controls the process. After a specified number of epochs has been reached, filename coordinator closes the main loop and orders enqueue workers to stop.

***av3_eval*** [av3_eval.py](./av3_eval.py)

restores the network from the saved state and performs its evaluation. At first, it runs throught the dataset several times to accumulate predictions. After evaluations, it averages all of the predictions and ranks and reorders the dataset by descending prediction averages.

Finally, it calculates several prediction measures such as top_100_score, confusion matrix, Area Under the Curve and writes sorted predictions, and computed metrics correspondingly into:
<pre>
/summaries/x_logs/x_predictions.txt
/summaries/x_logs/x_scores.txt
</pre>

### benchmark:
We have tested the 

Co-crystal structures of proteins and small molecules (binders) can be obtained from [Protein Data Bank](http://www.rcsb.org/).
