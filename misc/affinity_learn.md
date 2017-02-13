For the introduction to convolutional networks:   
http://cs231n.github.io/  
complete  
Module 1  
Module 2  

ConvNets  
https://www.tensorflow.org/tutorials/mnist/beginners/  
https://www.tensorflow.org/tutorials/mnist/pros/  
https://www.tensorflow.org/tutorials/mnist/tf/  
https://www.tensorflow.org/tutorials/deep_cnn/  
https://github.com/carpedm20/DCGAN-tensorflow  
  
Start with digit generation:  
  
ubuntu@ip-172-31-4-5:~/maksym$ git clone https://github.com/mitaffinity/core.git  
cd core  
python av4_main.py  

vi 

To change the database path under flags to   
/home/ubuntu/common/data/labeled_av4  

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

Open  
http://awsinstance.com/  
In your browser to visualize the network. This thing can crawl all the directories  

Heavy lifting  
Clusters  



