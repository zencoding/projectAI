import numpy as np

def load(name):
    params = np.load(name)
    h = np.load('h'+name)
    lowerbound = np.load('lowerbound'+name)
    testlowerbound = np.load('testlowerbound'+name)

    return (params,h,lowerbound,testlowerbound)

def load_notest(name):
    params = np.load(name)
    h = np.load('h'+name)
    lowerbound = np.load('lowerbound'+name)

    return (params,h,lowerbound)

def save(name,params,h,lowerbound,testlowerbound):
    np.save(name,params)    
    np.save('h' + name,h)
    np.save('lowerbound' + name,lowerbound)
    np.save('testlowerbound' + name,testlowerbound)
        

def save_notest(name,params,h,lowerbound):
    np.save(name,params)    
    np.save('h' + name,h)
    np.save('lowerbound' + name,lowerbound)
        
