"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

""" Contains functions for loading data from different datasets and for loading/saving paramaters """

import gzip, cPickle
import numpy as np

def load_mnist():
    f = gzip.open('dataset/mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def load_ff():
    f = open('dataset/freyfaces.pkl','rb')
    data = cPickle.load(f)
    f.close()
    return data

def load_filtered_chinese():
    f = gzip.open('dataset/chinesefiltered.pkl.gz','rb')
    data = cPickle.load(f)
    f.close()
    return data

def save(name,params,h,lowerbound,testlowerbound):
    """saves all variables during training aevb
    """
    np.save(name,params)    
    np.save('h' + name,h)
    np.save('lowerbound' + name,lowerbound)
    np.save('testlowerbound' + name,testlowerbound)
        

def save_notest(name,params,h,lowerbound):
    """same as save, but does save lowerbound on test data
    """
    np.save(name,params)    
    np.save('h' + name,h)
    np.save('lowerbound' + name,lowerbound)
        
def load(name):
    """loads all variables for continuing training aevb
    """
    params = np.load(name)
    h = np.load('h'+name)
    lowerbound = np.load('lowerbound'+name)
    testlowerbound = np.load('testlowerbound'+name)

    return (params,h,lowerbound,testlowerbound)

def load_notest(name):
    """same as load, but does not load lowerbound on test data
    """
    params = np.load(name)
    h = np.load('h'+name)
    lowerbound = np.load('lowerbound'+name)

    return (params,h,lowerbound)


