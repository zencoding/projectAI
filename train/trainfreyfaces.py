"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""This script trains an auto-encoder on the frey face dataset and keeps track of the lowerbound"""

#example: python -m train.trainfreyfaces -s freyfaces.npy

import aevb
from loadsave import load_ff,save_notest
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
learning_rate = 0.02

batch_size = 100

encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)
encoder.continuous = True

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
        encoder.initH(data[batch_size*i:batch_size*(i+1)].T)
    lowerbound = []

print "Iterating"

for iteration in xrange(1,35000):
    print 'Iteration:', iteration
    encoder.lowerbound = 0
    np.random.shuffle(data)
    encoder.iterate(data)
    print encoder.lowerbound/N
    lowerbound = np.append(lowerbound,encoder.lowerbound/N)
    if args.save:
        print "Saving params"
        save_notest(args.save,encoder.params,encoder.h,lowerbound)