"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np

def logreg_gradients(x,t,w,b):
    logq = w.T.dot(x) + b
        
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    deltab = -p
    deltab[t] = 1-p[t]
    
    deltaw = np.outer(x,deltab)
    
    return (deltaw,deltab)

def sgd_iter(x_train, t_train, w, b):
    learningrate = 0.001
    all_indices = np.arange(len(x_train),dtype=int)
    np.random.shuffle(all_indices)
        
    for i in all_indices:
        deltaw,deltab = logreg_gradients(x_train[i],t_train[i],w,b)
        w += learningrate * deltaw
        b += learningrate * deltab
                
    return (w,b)

def check_correct(x,t,w,b):
    logq = w.T.dot(x) + b
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    return (np.argmax(p) == t)

def calculate_percentage(X,T,w,b):
	numCorrect = 0
	for i in xrange(size(x,0)):
	    num_correct += check_correct(X[i],T[i],w,b)

	return num_correct*100/size(x,0)