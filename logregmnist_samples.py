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
parser.add_argument("-p", "--params", help="Specify param file", default = True)
parser.add_argument("-d", "--double", help = "On Double AE?", default = False)

args = parser.parse_args()

print 'loading data'
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

print x_train.shape

print 'creating h from saved params'

params = np.load(args.params)

#hidden = lambda x: np.tanh(x.dot(params[0].T) + params[5].T)
hidden = lambda x: x
h_train = hidden(x_train)
h_test = hidden(x_test)
h_valid = hidden(x_valid)

print args.double

if args.double:
    print 'calculating output of 2nd hidden layer'
    params = np.load(args.double)
    hidden2 = lambda sp: np.log(1+np.exp(sp.dot(params[0].T) + params[5].T))
    sigmoid = lambda si: 1/(1+np.exp(-si.dot(params[0].T + params[5])))

    h_train = hidden2((h_train+1)/2)
    h_test = hidden2((h_test+1)/2)
    h_valid = hidden2((h_valid+1)/2)

(N,dimh) = h_train.shape

w = np.zeros([10,dimh])
b = np.zeros([10])


train = []
valid = []
scores = []
stepsize = 50

h_valid = h_valid[:1000]
i = 0
max_iter_dataset = 1000

for j in xrange(1):
    print 'sample dataset: ', j
    valid = []
    datasetsize = (j+1)*stepsize
    h_trainS = h_train[1:datasetsize,:]
    t_trainS = t_train[1:datasetsize]
    (NS,dimhS) = h_trainS.shape
    

    while True:
    	i +=1
    	for k in xrange(NS):
    		w,b = sgd_iter(h_trainS[k],t_trainS[k],w,b)
    	
    	valid_correct = calculate_percentage(h_valid,t_valid,w,b)	
    	#train_correct = calculate_percentage(h_train,t_train,w,b)
    	#print 'valid correct = ', valid_correct
    	#print 'train correct = ', train_correct
        print valid_correct
    	valid.append(valid_correct)
    	#train.append(train_correct)
        if i>1:
            if valid[i]<valid[i-1]
                lower += 1
                if lower == 2:
                    i = max_iter_dataset-1
            else:
                lower_once = 0


    maxscore = max(valid)
    np.append(scores,maxscore)
    print 'score = ', maxscore

if args.save:
    np.save(args.save + '_scores', scores)