"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""
import gzip,cPickle
import numpy as np

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data

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

	pct_correct = num_correct*100/size(x,0)

	return pct_correct


def logistic_regression():
	
	iterations = 10

	(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
	wEncode = load json
	h_train = np.tanh(x_train.dot(wEncode.T))
	h_test = np.tanh(x_test.dot(wEncode.T))

	#Durk --> initialize 0?
	w = np.zeros([len(h_train), 10])
	b = np.zeros([10])
    
	for i in xrange(iterations):
 		print 'iteration: ', i
        w,b = sgd_iter(h_train,t_train,w,b)

	train_correct = calculate_percentage(h_train,t_train,w,b)
	test_correct = calculate_percentage(h_test,t_test,w,b)

	print 'percentage of training set correct: ', train_correct
	print 'percentage of test set correct: ', test_correct


if __name__ == "__main__":
    logistic_regression()