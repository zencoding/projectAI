"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_filtered_chinese
import cPickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = True)

args = parser.parse_args()

iterations = 200

print 'loading data'
(x_train, t_train) = load_filtered_chinese()
[N,dimx] = x_train.shape

x_valid = x_train[10000:]
t_valid = t_train[10000:]
x_train = x_train[0:10000]
t_train = t_train[0:10000]

if args.params:
	print 'creating h from saved params'

	params = np.load(args.params)

	hidden = lambda x: (np.tanh(x.dot(params[0].T) + params[5].T) + 1 )/2
	x_train = hidden(x_train).T
	x_valid = hidden(x_valid).T
        [dimx,N] = x_train.shape
        print dimx

pickle_file = open('chinesecharacterids.pkl','rb')
id_list = cPickle.load(pickle_file)

w = np.zeros([200,dimx])
b = np.zeros([200])

train = []
valid = []

for i in xrange(iterations):
    print 'iteration: ', i
    for j in xrange(x_train.shape[1]):
        w,b = sgd_iter(x_train[:,j],id_list[t_train[j]],w,b)
    print calculate_percentage(x_valid.T,[id_list[x] for x in t_valid],w,b)


