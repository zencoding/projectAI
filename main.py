import aevb
from data import load_mnist
from plot import plot
import numpy as np


if __name__ == "__main__":
	print "Loading MNIST data"
	(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
	data = np.concatenate((x_train,x_valid))
	
	[N,dimX] = data.shape
	HU_decoder = 100
	HU_encoder = 100
	dimZ = 2
	L = 1
	learning_rate = 0.1

	batchSize = 100

	encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,L,learning_rate)

	print "Initializing weights and biases"
	encoder.initParams()
		
	print "Creating Theano functions"
	encoder.createGradientFunctions()
	
	print "Iterating"
	batches = np.linspace(0,N,N/batchSize+1)

	for j in xrange(10):
		print 'iteration ', j
		for i in xrange(0,len(batches)-2):
			miniBatch = data[batches[i]:batches[i+1]]
			encoder.iterate(miniBatch.T, N)

	
	plot(encoder.params)
