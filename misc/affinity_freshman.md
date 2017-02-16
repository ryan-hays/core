###Quick Introduction to Affinity  
_The aim of the resources on this page is to allow anyone, even without specific machine learning background, to quickly get up to speeed with Affinity Core virtual screening engine. The estimated time for completions is under two weeks._

####Background Readings
_If you are familiar with all of the concepts the list: weights, biases, activation function, ReLU, softmax, convolution, pooling, layers of depth, batch, gradient descent, backpropagation (and chain rule), AdamOptimizer, please feel free to skip to the next section._  

CS231n "Convolutional Neural Networks for Visual Recognition"  
http://cs231n.github.io/  
Please, read through  
Module 1, Neural Networks      
Module 2, Convolutional Neural Networks   


####TensorFlow tutorials to complete
_If you are familiar with all of the concepts in this list: tensor graph, session, tf.global_variable_initializer, tf.train.coordinator, tf.train.start_queue_runners, tf.nn.sparse_softmax_cross_entropy_with_logits, tf.saver, tf.summary, tf.summary.FileWriter, tf.name_scope, please, feel free to skip to the next section._


[MNIST](https://www.tensorflow.org/tutorials/mnist/beginners/)  
[Deep MNIST](https://www.tensorflow.org/tutorials/mnist/pros/)  
[Understandig TensorFlow's workflow](https://www.tensorflow.org/tutorials/mnist/tf/)  
[CIFAR10](https://www.tensorflow.org/tutorials/deep_cnn/)  
[Deep Generative Adversarial Models](https://github.com/carpedm20/DCGAN-tensorflow)  

####Introduction to Affinity Core

_Structure based virtual screening is an approach that allows to retrieve a very small percent, usually few dozens of molecules, from the large database, of millioons of chemical structures. The process can be imagined as a google search for a flexible key (ligand) with a 3D image of a rigid lock (receptor,protein). Search can be broken into two parts. Since the most optimal relative position of the drug and protein is not known, it has to be estimated (docking). Afterwards, many static protein-ligand complexes have to be ranked by their predicted relative binding affinity (sorting). Usually, 25,000-200,000 pose evaluations are done during docking, and a single pose evaluation is done during ranking. Because Tesla K80 GPU can only evaluate 100-200 images/second, position search for a single ligand may take anywhere between 3 and 35 minutes, docking the average-size database of 1,000,000 of molecules may take 1.2 GPU years. In this example we only apply the network to the previously docked with AutoDock Smina positions, IE: ranking._

####Step 1: teaching the network
You will need four scripts, and the database
```
av4_networks.py
av4_main.py
av4_input.py
av4_utils.py
labeled_av4
```
where `labeled_av4` is an already prepared database of the ligands and proteins in av4 binary format.


```
av4_networks.py 
# is a library of many different network architectures
# each of the networks accepts batch of images, and outputs batch of unscaled probabilities (logits)
#
# crucial part 1: the network itself
# convolutional layers IE: tf.nn.conv3d
# pooling layers IE: tf.nn.max_pool3d or tf.nn.avg_pool3d
# Rectifier Linear Regression Units, or ReLUs IE: tf.nn.relu
#
# crucial part 2: rules for variable initialization
# IE: bias_variable 
# tf.constant(0.01, shape=shape)
# or weight variable
# tf.truncated_normal(shape, stddev=0.005)
# initial weights and biases for trainable variables are usually initialized with small random positive 
# values. Deep networks can easily run out of control and owerflow the floats in higher layers 
# if not initialized properly.
# That's why it's important to initialize variables very accurately
# especially deep networks can be hierarchically constructed 
# when the new layer(s) is added to the top of existing trained network
#
# crucial part 3: variable summaries
# IE: tf.summary.histogram, tf.summary.scalar, 
# and tf.name_scope (groups variables together under a common name)
# variable summaries are written in a separate file and help to monitor the state and evolution 
# of the network during training or testing


av4_input.py
# is script that reads and indexes the database in av4 format, and creates batches of images
# in 3D that can be fed to the network
# av4 database consists of thousands of folders - one for each protein
# each protein can have many ligands
# each ligand av4 file can have many positions (frames), and every frame has it's label
# the name of each folder IE 1QGT, 4G93 each correspond to a particular PDB id  
# original PDB file with coordinates can be found at http://www.rcsb.org/
#
# crucial part 1: index_the_database_into_queue
# crawls of all the folders in the database and creates tensor of filenames
#
# crucial part 2: read_receptor_and_ligand
# reads a single example of receptor and ligand from the database
# returns coordinates and name (label) of every atom (only one frame depending on epoch counter)
#
# crucial part 3: convert_protein_and_ligand_to_image
# creates an image (sparse or dense) from input atom coordinates
# dense image is cubic, has only some atoms of the protein, and describes pixels (not atoms)
# empty spaces in dense image are filled with zeros
#
# crucial part 4: tf.train.batch (for three reasons)
# reason 1:
# it runs multiple independent threads of image creation
# reason 2:
# it batches images together
# single images are small and tensor operations in them (such as convolution or pooling)
# are not efficient. 
# reason 3:
# gradient descent optimizer gets much better gradient from multiple images. 
# Ideally, every single gradient descent step would be applied to a representative sample 
# of a whole database; this is better achievable with larger batches


# av4_main.py
# is a main script that assembles all of the parts together 
# 
# crucial part 1: FLAGS
# class that stores many global parameters of the script
# such as FLAGS.pixels_size - determines the size of the pixel generated by av4_input
# 
# crucial part 2: train
# assembles all of the parts together:
# 1. input image creation pipeline              IE: ....something = image_and_label_queue 
# 2. network that makes predictions             IE: intuit_net or nilai_net
# 3. cost function to minimize                  IE: tf.nn.sparse_softmax_cross_entropy_with_logits
# 4. optimizer(applies gradient descent steps)  IE: tf.train.AdamOptimizer
# 5. run while loop pipe the tensor graph       IE: sess.run(train_step_run)
# graph initialization commands:
# 6. initializer of variables from placeholders IE: tf.global_variable_initializer 
# 7. fill the graph skeleton with data inputs   IE: tf.train.start_queue_runners
# 8. also has saver of variable states          IE: tf.train.saver
# 9. variable summaries to visualize            IE: tf.summaries.writer
# 10. an epoch counter that counts every read through all proteins in database as one epoch

av4_utils.py
# stores various utilities to support functions that are not natively present in TensorFlow
``` 
here is how a typical session on our Amazon graphical instance with K80 GPU would look like:

```
# log into our remote machine 
# email maksym to get the key
ssh -i P2_key.pem ubuntu@awsinstance.com
# every member of the group should have his or her folder
cd maksym
# clone affinity core into your working directory 
ubuntu@ip-172-31-4-5:~/maksym$ git clone https://github.com/mitaffinity/core.git  
cd core 
python av4_main.py  
# point the script to the location of the database
vi (or any other command line text file editor; some people like nano) 
# the database has already been donloaded to the instance
# change the database path under flags to   
# /home/ubuntu/common/data/labeled_av4  

# does not work; needs latest tensorflow  
# tensorflow12 is hidden in an envoronmental variable 
# source $TF12  
# if you are interested what $TF12 it is:  
echo $TF12 
/home/ubuntu/common/venv/tf12/bin/activate  

# start training
python av4_main.py 
# seems to work, now it's time to launch this process for a while
# the key is to launch it on the background, so it does not die when you log off
# from your remote host. Use the '&' sign
python av4_main.py &  
# now background process will persist when you exit the session  

# now the nasty problem: TensorFlow tends not to die and hog on the GPU even after it's been terminated
# also, there are many of us using GPU instance at the same time, but with TF's default settings
# only one process will capture all VRAM on the GPU 
# see if anything is running on the GPU  
nvidia-smi  
# should show the running processes, and how much VRAM each of them takes
# you can also use top to monitor RAM and CPU
top
# since it's a development instance, it is ok to kill all python processes with
# be carefull as it kills all the python processes that other people are running 
# it's ok to do it on our instance since it's consired to be only development zone for debugging
pkill -9 python
```
The network training may take hours, or days depending on your dataset and architecture of the network. It's important to note that in our code the epoch is counted by protein-ligand pairs, not by images. Every protein-ligand pair may have multiple incorrect positions of the ligand 50-400, and a single correct, crystal position. In this case, it takes 100 epochs to only show all of the negatives to the network once. That is different from classical understanding of epochs in image recognition when images can't have multiple frames.
Running the code should have resulted in four folders with outputs:
```
1_logs   
1_netstate   
1_test   
1_train  
```
`1_logs` might be empty, and will be used to write outputs during evaluation.
`1_netstate` will store the saved weights and biases for every trainable variable of the network 
(and also all other variables, such as epoch counter)
`1_train` and `1_test` should store summaries for variable states during training and testing that can
be visualized. Let's expect the outputs of in the foders

```
# log into our instance
ssh -i P2_key.pem ubuntu@awsinstance.com
# now I am
# ubuntu@ip-172-31-4-5:~$
# cd maksym
cd /core/summaries
cd 1_netstate
ls -l
# should show all of the files together with their size
# IE: 96789276 Jan 29 16:55 saved_state-60999.data-00000-of-00001
cd ../1_train
ls
# should show 
# events.out.tfevents.1485708632.ip-172-31-4-5
# which is a tensorflow summaries file
# let's try to visualize it:
# load tensorflow 0.12 (default version in the environment is 0.10)
source $TF12
# it's important to launch the tensorboard on port 80. By default internet browsers, such as chrome,
# will connect to port 80. You can read more here: 
# https://en.wikipedia.org/wiki/Port_(computer_networking)
# by default port 80 is not available to the user (the error is port is busy) that's why we use sudo
sudo python -m tensorflow.tensorboard --logdir=. --port=80
# now you can navigate your browser to awsinstance.com
```
 you should be able to see the following:
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/cross_entropy.png)
Cross entropy (our cost function) goes down as we are training the network. 
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/sparsity.png)
Sparsity of Rectifier Linear Unit is a percentage of zero-valued outputs of the layer. 
In chain rule for backpropagation, the derivative on sparse neuron is 0, and the derivative on downstream 
neurons is also 0. If the sparsity for the layer is exactly 1, backpropagation does not work, and weights 
can't be updated. That is what frequently happens when the network "explodes" because of the incorrect weight initialization.
![alt_tag](https://github.com/mitaffinity/core/blob/master/misc/histogram.png)
Biases that all vere all initialized at 0.001 diverge as we are training our network. 


####Step 2: evaluating the network

In addition to four folders resulting from our previous step, you will need these three scripts:
```
1_logs   
1_netstate   
1_test   
1_train  

av4_eval.py
av4_input.py
av4_utils.py
```
the only new script is `av4_eval`  
```
av4_eval.py
# av4_eval script is very similar to av4_main
#
# crucial part 1:
# assembles all of the parts together:
# 1. input image creation pipeline              IE: ....something = image_and_label_queue 
# 2. network that makes predictions             IE: intuit_net or nilai_net
# 3. softmax instead of cost function in main   IE: tf.nn.softmax
# 4. no optimizer, variables are loaded         IE: saver.restore
# 5. run while loop pipe the tensor graph       IE: sess.run(train_step_run)
#
# crucial part 2: class store_predictions
# stores, sorts, and saves predictions
# estimates different evaluation parameters for straightforward binary classification such as 
# Area Under Curve https://en.wikipedia.org/wiki/Receiver_operating_characteristic
# Confusion Matrix https://en.wikipedia.org/wiki/Confusion_matrix
#  
# crucial part 3: av4_eval can be used for two different tasks
# 1. distinguishing the correct position within many positions as in docking
# 2. sorting ligands each of which has many positions as in sorting
# 1 is very straightforward since our training consits of correct and incorrect positions, 
# we only need to score all of the available positions
# our performance on task 1 is very high (AUC > 0.94)
# 2 is not straighforward. Since many of the docked positions given to the network are not correct
# (sometimes all of them)
```

####Step 3: database preparation (optional)
data and .av4 format
av4_database_master
av4_atom_dictionary

####Step 4: running affinity on Bridges, XSEDE national supercomputer
