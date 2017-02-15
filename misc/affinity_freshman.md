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

Running the Network
```av4_main
# something
av4_input
av4_networks
av4_utils
labeled_av4
```

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



