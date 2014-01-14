"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_mnist

iterations = 10

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

params = np.load('params.npy')
h_train = np.tanh(x_train.dot(params[2].T))
h_test = np.tanh(x_test.dot(params[2].T))

w = np.zeros([len(h_train), 10])
b = np.zeros([10])

for i in xrange(iterations):
	print 'iteration: ', i
	w,b = sgd_iter(h_train,t_train,w,b)

train_correct = calculate_percentage(h_train,t_train,w,b)
test_correct = calculate_percentage(h_test,t_test,w,b)

print 'percentage of training set correct: ', train_correct
print 'percentage of test set correct: ', test_correct
