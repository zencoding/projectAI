#contains functions for performing logistic regression on data X and classes T

"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np

def logreg_gradients(x,t,w,b):
    """
    Calculates and returns gradients of weight matrix w and bias vector b for a given (x,t)
    """
    logq = w.dot(x) + b
    
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    deltab = -p
    deltab[t] = 1-p[t]
    
    deltaw = np.outer(deltab,x)
    
    return (deltaw,deltab)

def sgd_iter(x_train, t_train, w, b):
    """
    Performs one Gradient Descent update step of Logistic Regression (on one datapoint).
    Returns new weights w and biases b
    """
    learningrate = 0.001
    
    deltaw,deltab = logreg_gradients(x_train,t_train,w,b)
    w += learningrate * deltaw
    b += learningrate * deltab
                
    return (w,b)

def check_correct(x,t,w,b):
    """
    Checks whether the learned weights correctly predicts the class of datapoint x, belonging to class t
    Returns True or False
    """
    logq = w.dot(x) + b
    p = np.exp(logq - np.log(np.sum(np.exp(logq))))
    
    return (np.argmax(p) == t)

def calculate_percentage(X,T,w,b):
    """
    Calculates and returns percentage of correctly classified datapoints
    """
    num_correct = 0.
    for i in xrange(X.shape[0]):
	    num_correct += check_correct(X[i],T[i],w,b)

    return num_correct*100/X.shape[0]
