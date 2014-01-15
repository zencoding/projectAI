"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

'Trains parameters for aevb on the CIFAR dataset (unsupervised!)'

import aevb
from data import load_cifar
import argparse
import numpy as np
from plot import plot

parser = argparse.ArgumentParser()

parser.add_argument("-g", "--grey", help="convert to grey", default = False, action="store_true")
parser.add_argument("-plot", "--plot", help="plot manifold", default = False, action="store_true")
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)

args = parser.parse_args()

print 'loading CIFAR-100 data'
data = load_cifar()
data = data/256

if args.grey:
	data = np.sqrt( np.square(data[:,:1024]) + np.square(data[:,1024:2048]) + np.square(data[:,2048:]) )

print "Initializing weights and biases"

[N,dimX] = data.shape
HU_decoder = 200
HU_encoder = 200
dimZ = 2
L = 1
learning_rate = 0.05
batchSize = 100

encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,L,learning_rate)

if args.params:
    print "Loading params from: {0}".format(args.params)
    encoder.params = np.load(args.params)
else:
    encoder.initParams()
        
print "Creating Theano functions"
encoder.createGradientFunctions()

print "Iterating"
batches = np.linspace(0,N,N/batchSize+1)

for j in xrange(30):
        print 'iteration ', j
        for i in xrange(0,len(batches)-2):
                miniBatch = data[batches[i]:batches[i+1]]
                encoder.iterate(miniBatch.T, N)

if args.save:
    print "Saving params in: {0}".format(args.save)
    np.save(args.save,encoder.params)	

if args.plot:
	if args.grey:
		plot(encoder.params,32)
	else:
		print 'plot doesnt support rgb values (yet)	'