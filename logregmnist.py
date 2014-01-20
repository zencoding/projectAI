"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_mnist

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", help="Specify file to save results", default = False)
parser.add_argument("-p", "--params", help="Specify param file", default = False)

args = parser.parse_args()

iterations = 50

print 'loading data'
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

print 'creating h from saved params'

params = np.load(args.params)

hidden = lambda x: np.tanh(x.dot(params[2].T) + params[7].T)
h_train = hidden(x_train)
h_test = hidden(x_test)
h_valid = hidden(x_valid)


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
	train_correct = calculate_percentage(h_train,t_train,w,b)
	print 'valid correct = ', valid_correct
	print 'train correct = ', train_correct

	if args.save:
	    print "Saving results"
	    valid.append(valid_correct)
	    train.append(train_correct)
	    np.save(args.save + '_train',train)	
	    np.save(args.save + '_val',valid)

test_correct = calculate_percentage(h_test,t_test,w,b)

print 'percentage of training set correct: ', train_correct
print 'percentage of test set correct: ', test_correct
if args.save:
	np.save(args.save + '_test', test_correct)
	'creating and saving figure'
	from plot_logreg import plot_accuracy
	plot_accuracy(args.save, 'Accuracy of Log Reg on MNIST \n using AEVB hidden space ( N = ' + str(dimh) + ')')

