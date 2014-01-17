"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_mnist

iterations = 30

print 'loading data'
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

print 'creating h from saved params'

params = np.load('params.npy')

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

	print 'valid correct = ', calculate_percentage(h_valid,t_valid,w,b)
	print 'train correct = ', calculate_percentage(h_train,t_train,w,b)

train_correct = calculate_percentage(h_train,t_train,w,b)
test_correct = calculate_percentage(h_test,t_test,w,b)

print 'percentage of training set correct: ', train_correct
print 'percentage of test set correct: ', test_correct
np.save('logreg_resultsFinal_dim20',test_correct)
