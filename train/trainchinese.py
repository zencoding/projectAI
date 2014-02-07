"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""A script for training an auto-encoder on a subset of the Chinese dataset"""

#Example: python -m train.trainchinese -s chinese.npy

import aevb
from loadsave import load_notest, save_notest,load_filtered_chinese
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)

args = parser.parse_args()


dimZ = 200
HU_decoder = 500
HU_encoder = HU_decoder

batch_size = 100
L = 1
learning_rate = 0.01


(data, t_train) = load_filtered_chinese()
[N, dimX] = data.shape
encoder = aevb.AEVB(HU_decoder, HU_encoder, dimX, dimZ, batch_size, L, learning_rate)

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
if args.params:
    print "Loading params from: {0}".format(args.params)
    encoder.params, encoder.h, lowerbound = load_notest(args.params)
else:
    encoder.initParams()
    for i in xrange(0, 10):
        encoder.initH(data[batch_size*i:batch_size*(i+1)].T)
    lowerbound = np.array([])

for j in xrange(20000):
    print 'Iteration:', j
    encoder.lowerbound = 0
    encoder.iterate(data)
    print encoder.lowerbound/N
    lowerbound = np.append(lowerbound, encoder.lowerbound/N)

    if args.save:
        print "Saving params"
        save_notest(args.save, encoder.params, encoder.h, lowerbound)
