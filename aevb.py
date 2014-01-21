"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
import theano as th
import theano.tensor as T

class AEVB:
    def __init__(self, HU_decoder, HU_encoder, dimX, dimZ, batch_size, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder
        self.HU_encoder = HU_encoder

        self.dimX = dimX
        self.dimZ = dimZ
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.sigmaInit = 0.01
        self.h = [0.0001] * 10
        self.lowerbound = 0

        self.continuous = False

    def initParams(self):
                #initialize theta for decoder
        W1 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b1 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.dimX,1))

        #initialize phi for encoder
        W3 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,self.dimX))
        b3 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,1))

        W4 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b4 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))

        W5 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b5 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))

        #Create one list with parameters
        self.params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

    def initH(self,miniBatch):
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]


    def createGradientFunctions(self):
        #Create the Theano variables
        W1,W2,W3,W4,W5,x,eps = T.dmatrices("W1","W2","W3","W4","W5","x","eps")

        #Create biases as cols so they can be broadcasted for minibatches
        b1,b2,b3,b4,b5 = T.dcols("b1","b2","b3","b4","b5")

        if self.continuous:
            h = T.nnet.softplus(T.dot(W3,x) + b3)
        else:   
            h = T.tanh(T.dot(W3,x) + b3)

        mu = T.dot(W4,h) + b4
        # logsigma = 0.5*(T.dot(W5,h) + b5)
        sigma = T.exp(0.5*(T.dot(W5,h) + b5))

        #Find the hidden variable z
        z = mu + sigma*eps

        #Set up the equation for decoding
        y = T.nnet.sigmoid(T.dot(W2,T.tanh(T.dot(W1,z) + b1)) + b2)

        #Set up likelihood
        if self.continuous:
            logpxz = T.sum(-(0.5 * np.log(np.pi) + T.log(self.data_sigma)) - 0.5 * ((x - y) / self.data_sigma)**2)
        else:
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()


        #Set up q 
        logqzx = T.sum(-(z - mu)**2/(2.*sigma**2) - 0.5 * T.log(2. * np.pi * sigma**2))

        #Compute prior
        logpz = T.sum(-(z**2)/2 - 0.5 * np.log(2 * np.pi))

        #Define lowerbound
        logp = logpxz + logpz - logqzx

        #Compute all the gradients
        derivatives = T.grad(logp,[W1,W2,W3,W4,W5,b1,b2,b3,b4,b5])

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        self.gradientfunction = th.function([W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,x,eps], derivatives, on_unused_input='ignore')

    def iterate(self, data):
        """Compute the gradients and update parameters"""
        [N,dimX] = data.shape
        batches = np.linspace(0,N,N/self.batch_size+1)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            totalGradients = self.getGradients(miniBatch.T)
            self.updateParams(totalGradients,N)

    def getLowerBound(self,data):
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.linspace(0,N,N/self.batch_size+1)

        for i in xrange(0,len(batches)-2):
            e = np.random.normal(0,1,[self.dimZ,self.batch_size])
            miniBatch = data[batches[i]:batches[i+1]]
            gradients = self.gradientfunction(*(self.params),x=miniBatch.T,eps=e)
            lowerbound += gradients[10]

            return lowerbound/N


    def getGradients(self,miniBatch):
        totalGradients = [0] * 10
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,self.batch_size])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[10]

            for i in xrange(len(totalGradients)):
                if np.isnan(np.sum(gradients[i])):
                    print "The gradients contain nans, that cannot be right"
                    exit()

                totalGradients[i] += gradients[i]

        return totalGradients

    def updateParams(self,totalGradients,N):
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            prior = 0.5*self.params[i]*(i<5)

            #Include adagrad, include prior for weights
            self.params[i] = self.params[i] + (self.learning_rate)/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(self.batch_size/N))
