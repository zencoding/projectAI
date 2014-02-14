projectAI
=========

##The auto encoder
The auto encoder as used by our project can be found in the file [aevb.py](aevb.py).

This class takes the following parameters:

class aevb(HU_decoder, HU_encoder, dimX, dimZ, batch_size, L=1, learning_rate=0.01)

There are three methods necessary to use:

* initH(miniBatch), which initializes H for AdaGrad
	
* iterate(data)
	
* getLowerbound(data), for specifically obtaining a lowerbound of the log likelihood. Example use is for obtaining lowerbound for test set.

###Scikit implementation
In the scikit implementation folder there is a version of the auto encoder following the guidelines of Scikit Learn. There is also an accompanying script showing how it works.


##Datasets

####Chinese characters

The CASIA Offline Chinese Handwriting Database is available from [here](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html)

The scripts to read in the .gnt files and preprocess them are available in the file [readchinese.py](chinese/readchinese.py). The size of the output can be set inside this script, also binarization can be toggled.

####MNIST

We used a Python Pickle provided by [Theano](http://deeplearning.net/tutorial/gettingstarted.html)

Description of the set can be found [here](http://yann.lecun.com/exdb/mnist/)

####Frey Face

We converted a matlab file from [NYU](http://www.cs.nyu.edu/~roweis/data.html) to a Python Pickle. Preprocess includes normalizing to 0-1 range and taking the negative.

##Training
Scripts for training the auto-encoder on specific datasets reside in the folder [train](train/). Inside the script it is possible to set the dimensionality of the latent space, the amount of hidden units, the batch_size and the learning_rate

It takes three parameters:

* "-p" for passing a filename (ending in .npy) to read the parameters from. It expects three files: lowerboundfilename, filename, hfilename

* "-s" for passing a filename (ending in .npy) to save the parameters to

* "-d" for passing a filename to read the parameters from for training a double auto-encoder (only for trainmnist)

Example (run from root folder):

    python -m train.trainmnist -s mnist.npy

which will create the following files:
lowerboundmnist.npy (to keep track of progress)
mnist.npy (contains all weights and biases)
hmnist.npy (contains values for AdaGrad)
    
##Classification with Log Regression
Scripts for performing log regression using our [own implementation](log_regression.py) of log regression that we wrote for the Machine Learning class.

Log regression has two useful methods:

- sgd_iter, which takes one datapoint and its label
- calculate_percentage, which takes a list of datapoints and their labels. It classifies the datapoints and computes the percentage correct by checking it with the labels.

Each log regression file takes the parameter "-p" for passing a filename (ending in .npy) to read the parameters from. If this parameter is passed it will automatically transform the data and perform log regression on it. If it is not given, log regression is performed on the raw pixel data.

Specifically for [logregmnist.py](logregression/logregmnist.py):

* "-s" for passing a filename (ending in .npy) to save the scores of the training set to
* "-d" for passing a filename to read the parameters from for classifying on double transformed data

Example (run from root folder):

	python -m logregression.logregmnist









