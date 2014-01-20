"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import aevb
from data import load_mnist
from loadsave import *
import numpy as np
import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.stats as sp

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-s", "--save", help="Specify file to save params", default = False)
parser.add_argument("-d","--double", help="Train on hidden layer of previously trained AE", default = False)

args = parser.parse_args()

print "Loading MNIST data"
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
data = np.concatenate((x_train,x_valid))

if args.double:
    print 'cumputing hidden layer to train new AE on'
    prev_params = np.load(args.double)
    data = np.tanh(data.dot(prev_params[2].T) + prev_params[7].T)
    x_test = np.tanh(x_test.dot(prev_params[2].T) + prev_params[7].T)

dimZ = 20
HU_decoder = 500
HU_encoder = HU_decoder

batch_size = 100
L = 1
learning_rate = 0.1

[N,dimX] = data.shape
encoder = aevb.AEVB(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate)

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
if args.params:
    print "Loading params from: {0}".format(args.params)
    encoder.params, encoder.h, lowerbound, testlowerbound = load()
else:
    encoder.initParams()
    for i in xrange(0,10):
            encoder.initH(data[batch_size*i:batch_size*(i+1)].T)
    lowerbound = np.array([])
    testlowerbound = np.array([])

for j in xrange(1,2000):
    print 'Iteration:', j
    encoder.lowerbound = 0
    encoder.iterate(data)
    print encoder.lowerbound/N
    lowerbound = np.append(lowerbound,encoder.lowerbound/N)
    if j % 5 == 0:
        print "creating manifold"
        gridSize = 20
        gridValues = np.linspace(0.01,0.99,gridSize)
        gs = gridspec.GridSpec(gridSize, gridSize)

        fig = plt.figure()
        W1 = encoder.params[0]
        W2 = encoder.params[1]

        b1 = encoder.params[5]
        b2 = encoder.params[6]
        for a in xrange(gridSize):
            for b in xrange(gridSize):
                z = np.matrix([sp.norm.ppf(gridValues[a]),sp.norm.ppf(gridValues[b])]).T
                y = 1 / (1 + np.exp(-(W2.dot(np.tanh(W1.dot(z) + b1)) + b2)))
                ax = fig.add_subplot(gs[a,b])
                ax.imshow(y.reshape((28,28)), interpolation='nearest', cmap='Greys')
                plt.axis('off')

        fig.patch.set_facecolor('white')
        plt.savefig('manifolds/manifold'+str(j)+'.png')
        plt.close()
    if j % 10 == 0:
        print "Saving test lowerbound"
        testlowerbound = np.append(testlowerbound,encoder.getLowerBound(x_test))
    if args.save:
        print "Saving params"
        save(args.save,encoder.params,encoder.h,lowerbound,testlowerbound)
