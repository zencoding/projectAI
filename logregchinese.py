"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
from log_regression import *
from data import load_chinese
import cPickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", help="Specify param file", default = True)

args = parser.parse_args()

iterations = 200

print 'loading data'
(x_train, t_train) = load_chinese(1)

pickle_file = open('chinesecharacterids.pkl','rb')
id_list = cPickle.load(pickle_file)

print len(id_list)

w = np.zeros([3755,1600])
b = np.zeros([3755])

train = []
valid = []

for i in xrange(iterations):
	print 'iteration: ', i
        for file_id in xrange(1,90):
            (x_train, t_train) = load_chinese(file_id)
            x_train = x_train.T
            [N,dimx] = x_train.shape
            for j in xrange(N):
                print j
                w,b = sgd_iter(x_train[j],id_list[t_train[j]],w,b)
            print calculate_percentage(x_train[0:500],[id_list[x] for x in t_train[0:500]],w,b)


