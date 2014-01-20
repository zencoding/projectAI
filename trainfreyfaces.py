"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import aevb
from data import load_ff
import numpy as np
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)

args = parser.parse_args()

print "Loading data"
data = load_ff()

[N,dimX] = data.shape
HU_decoder = 200
HU_encoder = 200

dimZ = 2
L = 1
learning_rate = 0.01

batch_size = 131

encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)
encoder.continuous = True

encoder.data_sigma = np.std(data,0).T

cov = np.cov(data.T)
invcov = np.linalg.inv(cov)
encoder.invcov = invcov
(sign,logdetcov) = np.linalg.slogdet(cov)
encoder.logdetcov = sign*logdetcov

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
	for i in xrange(0,14):
		encoder.initH(data[batch_size*i:batch_size*(i+1)].T)
	lowerbound = []

print "Iterating"

for j in xrange(2000):
	print 'Iteration:', j
	encoder.lowerbound = 0
	encoder.iterate(data)
	print encoder.lowerbound/N
	lowerbound = np.append(lowerbound,encoder.lowerbound/N)
	if args.save:
		print "Saving params"
		np.save(args.save,encoder.params)	
		np.save('h' + args.save,encoder.h)
		np.save('lowerbound' + args.save,lowerbound)
