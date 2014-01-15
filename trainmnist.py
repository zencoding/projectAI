"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import aevb
from data import load_mnist
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)

args = parser.parse_args()

print "Loading MNIST data"
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
data = np.concatenate((x_train,x_valid))

[N,dimX] = data.shape
HU_decoder = 500
HU_encoder = 500
dimZ = 2
L = 1
learning_rate = 0.1

batchSize = 100

encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,L,learning_rate)

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
if args.params:
	print "Loading params from: {0}".format(args.params)
	encoder.params = np.load(args.params)
	encoder.h = np.load('h'+args.params)
	lowerbound = np.load('lowerbound'+args.params)
else:
	encoder.initParams()
	for i in xrange(0,10):
		encoder.initH(data[batchSize*i:batchSize*(i+1)].T)
	lowerbound = []

print "Iterating"
batches = np.linspace(0,N,N/batchSize+1)

for j in xrange(200):
	print 'Iteration:', j
	encoder.lowerbound = 0
	for i in xrange(0,len(batches)-2):
		miniBatch = data[batches[i]:batches[i+1]]
		encoder.iterate(miniBatch.T, N)
	print encoder.lowerbound
	lowerbound.append(encoder.lowerbound)
	if args.save:
		print "Saving params"
		np.save(args.save,encoder.params)	
		np.save('h' + args.save,encoder.h)
		np.save('lowerbound' + args.save,lowerbound)
