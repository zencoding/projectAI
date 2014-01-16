"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np

def logreg_gradients(x,t,w,b):
    print w.shape, x.shape, b.shape
    logq = w.dot(x) + b
    
        
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    deltab = -p
    deltab[t] = 1-p[t]
    
    deltaw = np.outer(x,deltab)
    
    return (deltaw,deltab)

def sgd_iter(x_train, t_train, w, b):

    batchSize, dimH = x_train.shape
    learningrate = 0.001*np.sqrt(batchSize)
    
    deltaw,deltab = logreg_gradients(x_train,t_train,w,b)
    w += learningrate * deltaw
    b += learningrate * deltab
                
    return (w,b)

def check_correct(x,t,w,b):
    logq = w.T.dot(x) + b
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    return (np.argmax(p) == t)

def calculate_percentage(X,T,w,b):
    num_correct = 0.
    for i in xrange(X.shape[0]):
	    num_correct += check_correct(X[i],T[i],w,b)

    return num_correct*100./X.shape[0]
