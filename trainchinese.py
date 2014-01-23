"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import aevb
from data import load_chinese
from loadsave import load_notest, save_notest
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)

args = parser.parse_args()


dimZ = 200
HU_decoder = 1000
HU_encoder = HU_decoder

batch_size = 100
L = 1
learning_rate = 0.01


(data, t_train) = load_chinese(1)
[N, dimX] = data.shape
encoder = aevb.AEVB(HU_decoder, HU_encoder, dimX, dimZ, batch_size, L, learning_rate)

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
if args.params:
    print "Loading params from: {0}".format(args.params)
    encoder.params, encoder.h, lowerbound = load_notest(args.params+'.npy')
else:
    encoder.initParams()
    for i in xrange(0, 10):
            encoder.initH(data[batch_size*i:batch_size*(i+1)].T)
    lowerbound = np.array([])
    # testlowerbound = np.array([])

for j in xrange(2000):
    (x_train, t_train) = load_chinese(1)
    print 'Iteration:', j
    lowerbound_total = 0
    for file_id in xrange(1, 90):
        encoder.lowerbound = 0
        print "file: ", file_id
        (x_train, t_train) = load_chinese(file_id)
        encoder.iterate(x_train)
        lowerbound_total += encoder.lowerbound/N
        print lowerbound_total/file_id
        if args.save:
            print "Saving params"
            save_notest(args.save, encoder.params, encoder.h, lowerbound)
    lowerbound = np.append(lowerbound, lowerbound_total/89)

    # if j % 5 == 0:
        # print "Saving test lowerbound"
        # testlowerbound = np.append(testlowerbound, encoder.getLowerBound(x_test))

