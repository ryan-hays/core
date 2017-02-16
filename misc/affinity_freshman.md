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

####Step 1: teaching the network to distinguish binders from non-binders.
You will need four scripts 
```
av4_networks.py
av4_main.py
av4_input.py
av4_utils.py
```
and an already prepared database of the ligand positions and proteins in av4 binary format.

```labeled_av4```




a library of different networks  
all of the networks accept  
keeps together hyperparameters of the model such as: batch size,


####Step 2: evaluating the network
av4_eval

data and .av4 format
av4_database_master
av4_atom_dictionary

For development purposes we host an AWS instance with a single Tesla K80 GPU

`#clone affinity core into your working directory 
ubuntu@ip-172-31-4-5:~/maksym$ git clone https://github.com/mitaffinity/core.git  
cd core`  
python av4_main.py  
vi 
#To change the database path under flags to   
#/home/ubuntu/common/data/labeled_av4  

Does not work needs latest tensorflow  

Source $TF12  

If you are interested what it is:  
/home/ubuntu/common/venv/tf12/bin/activate  
  
The training should start now  
python av4_main.py  

The process will break when one exits the terminal  

Launch on the background  

python av4_main.py &  
Background process will persist regardless if you are in ssh session or not  

Only one person can use the GPU with TF at the same time by default.  
To see if anything is running  
nvidia-smi  
top  

Development zone kill processes with  

pkill -9 python  

Understanding the outputs:  

1_logs   
Usually evaluations are written here  
1_netstate saves the trained weights of the network  
1_test saves visualization of testing  
1_train saves visualization of train  

Visualizing the network  

sudo python -m tensorflow.tensorboard --logdir=. --port=80  
`
Open  
http://awsinstance.com/  
In your browser to visualize the network. This thing can crawl all the directories  

Heavy lifting  
Clusters  



