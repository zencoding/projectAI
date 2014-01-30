"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from loadsave import load_mnist

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", help="Specify file to save results", default = False)
parser.add_argument("-p", "--params", help="Specify param file", default = False)
parser.add_argument("-d", "--double", help = "On Double AE?", default = False)

args = parser.parse_args()

iterations = 500

print 'loading data'
(h_train, t_train), (h_valid, t_valid), (h_test, t_test) = load_mnist()

datasetsize = h_train.shape[1]


if args.params:
    print 'creating h from saved params'

    params = np.load(args.params)

    hidden = lambda x: (np.tanh(x.dot(params[0].T) + params[5].T) + 1 )/2
    h_train = hidden(h_train)
    h_test = hidden(h_test)
    h_valid = hidden(h_valid)

if args.double:
    print 'calculating output of 2nd hidden layer'
    params = np.load(args.double)
    hidden2 = lambda sp: np.log(1+np.exp(sp.dot(params[0].T) + params[6].T))
    sigmoid = lambda si: 1/(1+np.exp(-si.dot(params[0].T + params[6])))

    h_train = hidden2(h_train)
    h_test = hidden2(h_test)
    h_valid = hidden2(h_valid)


(N,dimh) = h_train.shape

w = np.zeros([10,dimh])
b = np.zeros([10])


train = []
valid = []

for i in xrange(iterations):
    print 'iteration: ', i
    for j in xrange(N):
            w,b = sgd_iter(h_train[j],t_train[j],w,b)
    
    valid_correct = calculate_percentage(h_valid,t_valid,w,b)   
    print 'valid correct = ', valid_correct

    if args.save:
        print "Saving results"
        valid.append(valid_correct) 
        np.save(args.save + '_val',valid)

test_correct = calculate_percentage(h_test,t_test,w,b)

print 'percentage of test set correct: ', test_correct
if args.save:
    np.save(args.save + '_test', test_correct)

