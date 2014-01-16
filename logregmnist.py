"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_mnist

iterations = 30

print 'loading and shuffling data'
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

print 'creating h from saved params'
params = np.load('params.npy')
h_train = np.tanh(x_train.dot(params[2].T)+params[7].T)
h_test = np.tanh(x_test.dot(params[2].T)+params[7].T)
h_valid = np.tanh(x_valid.dot(params[2].T)+params[7].T)


w = np.zeros([h_train.shape[1], 10])
b = np.zeros([10])

for i in xrange(iterations):
	print 'iteration: ', i
	w,b = sgd_iter(h_train,t_train,w,b)

	train_correct = calculate_percentage(h_train,t_train,w,b)
	print 'train correct = ', train_correct
	valid_correct = calculate_percentage(h_valid,t_valid,w,b)
	print 'valid correct = ', valid_correct

train_correct = calculate_percentage(h_train,t_train,w,b)
test_correct = calculate_percentage(h_test,t_test,w,b)

print 'percentage of training set correct: ', train_correct
print 'percentage of test set correct: ', test_correct