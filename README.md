projectAI
=========

##Training
Scripts for training the auto-encoder on specific datasets reside in the folder train/. Inside the script it is possible to set the dimensionality of the latent space, the amount of hidden units, the batch_size and the learning_rate

It takes three parameters:
"-p" for passing a filename (ending in npy) to read the parameters from
    - expects three files: lowerboundfilename, filename, hfilename
"-s" for passing a filename (ending in npy) to save the parameters to
"-d" for passing a filename to read the parameters from for training a double auto-encoder (only for trainmnist)

example (run from root folder):

python -m train.trainmnist -s mnist.npy

which will create the following files:
lowerboundmnist.npy (to keep track of progress)
mnist.npy (contains all weights and biases)
hmnist.npy (contains values for AdaGrad)
    
## 
