"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import gzip, cPickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import theano as th

import theano.tensor as T


def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def initialize(HU_Decoder, HU_Encoder, dimX, dimZ):
    sigmaInit = 0.01
    #initialize theta for decoder
    W1 = np.random.normal(0,sigmaInit,(HU_Decoder,dimZ))
    b1 = np.zeros([HU_Decoder,1])

    W2 = np.random.normal(0,sigmaInit,(dimX,HU_Decoder))
    b2 = np.zeros([dimX,1])

    #initialize phi for encoder
    W3 = np.random.normal(0,sigmaInit,(HU_Encoder,dimX))
    b3 = np.zeros([HU_Encoder,1])

    W4 = np.random.normal(0,sigmaInit,(dimZ,HU_Encoder))
    b4 = np.zeros([dimZ,1])

    W5 = np.random.normal(0,sigmaInit,(dimZ,HU_Encoder))
    b5 = np.zeros([dimZ,1])

    #Create one list with parameters
    params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

    return params

def initGrad(dimZ):
    #Create the Theano variables
    W1,W2,W3,W4,W5,x,eps = T.dmatrices("W1","W2","W3","W4","W5","x","eps")

    #Create biases as cols so they can be broadcasted for minibatches
    b1,b2,b3,b4,b5 = T.dcols("b1","b2","b3","b4","b5")

    #Set up the equations for encoding
    #Something here is wrong, W3 gradient is fully zero
    h = T.tanh(T.dot(W3,x) + b3)

    mu = T.dot(W4,h) + b4
    sigma = T.sqrt(T.exp(T.dot(W5,h) + b5))

    #Find the hidden variable z
    z = mu + sigma*eps

    #Set up the equation for decoding
    y = 1. / (1 + T.exp(-(T.dot(W2,T.tanh(T.dot(W1,z) + b1)) + b2)))

    # y = th.printing.Print('value of y:')(y)

    #Set up likelihood
    logpxz = T.sum(x*T.log(y) - (1-x)*T.log(1 - y))
    
    #Set up q (??) 
    logqzx = T.sum(-(z - mu)**2/(2.*sigma**2) - 0.5 * T.log(2. * np.pi * sigma**2))

    #Choose prior
    logpz = T.sum(-(z**2)/2 - 0.5 * np.log(2 * np.pi))
    
    #Define lowerbound
    logp = logpxz + logpz - logqzx

    #Compute all the gradients
    derivatives = T.grad(logp,[W1,W2,W3,W4,W5,b1,b2,b3,b4,b5])

    f = th.function([W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,x,eps], derivatives, on_unused_input='ignore')

    return f

def iterate(params, f, miniBatch, L):
    """Compute the gradients for one miniBatch and return the updated parameters"""
    totalGradients = [None] * 10
    for l in xrange(L):
        dimZ = params[9].shape[0]
        batchSize = miniBatch.shape[1]

        e = np.random.normal(0,1,[dimZ,batchSize])
        gradients = f(*(params),x=miniBatch,eps=e)
        print "W3",gradients[2]

        for i in xrange(len(gradients)):
            # print i,gradients[i]
            if np.isnan(np.sum(gradients[i])):
                print "The gradients contain nans, that cannot be right"
                exit()

            if totalGradients[i] == None:
                totalGradients[i] = gradients[i]
            else:
                totalGradients[i] += gradients[i]

    return totalGradients

def plot(dimZ, params):
    W1 = params[0]
    W2 = params[1]

    b1 = params[5]
    b2 = params[6]

    z = np.matrix([sp.stats.invgauss.cdf(0.1,1),sp.stats.invgauss.cdf(0.1,1)]).T
    y = 1 / (1 + np.exp(-(W2.dot(np.tanh(W1.dot(z) + b1)) + b2)))

    plt.imshow(y.reshape((28,28)), interpolation='nearest', cmap='Greys')
    plt.show()


def sga():
    HU_Decoder = 100
    HU_Encoder = 100

    L = 3
    dimZ = 2

    learningrate = 0.01

    h = [0.0001]*10

    print "Loading MNIST data"
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

    print "Initializing weights and biases"
    [N,dimX] = x_train.shape

    params = initialize(HU_Decoder, HU_Encoder, dimX, dimZ)

    print "Creating Theano functions"
    f = initGrad(dimZ)

    print "Iterating"
    batchSize = 100
    dataSamples = 20000

    batches = np.linspace(0,dataSamples,dataSamples/batchSize+1)
    # batches = np.linspace(0,2,3)

    for j in xrange(5):
        for i in xrange(0,len(batches)-2):
            miniBatch = x_train[batches[i]:batches[i+1]]
            totalGradients = iterate(params, f, miniBatch.T, L)

            #Update the parameters
            for i in xrange(len(params)):
                if h[i] == None:
                    h[i] = totalGradients[i]*totalGradients[i]
                else:
                    h[i] += totalGradients[i]*totalGradients[i]

                prior = 0
                #Include adagrad
                params[i] = params[i] + (learningrate/h[i]) * totalGradients[i] + prior

    print "Plotting"
    plot(dimZ,params)

if __name__ == "__main__":
    sga()
