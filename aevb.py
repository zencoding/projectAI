"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
import theano as th
import theano.tensor as T

class AEVB:
    def __init__(self, HU_decoder, HU_encoder, dimX, dimZ, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder
        self.HU_encoder = HU_encoder

        self.dimX = dimX
        self.dimZ = dimZ
        self.L = L
        self.learning_rate = learning_rate

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
        batchSize = miniBatch.shape[1]
        totalGradients = self.getGradients(miniBatch,batchSize)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]


    def createGradientFunctions(self):
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
        y = T.nnet.sigmoid(T.dot(W2,T.tanh(T.dot(W1,z) + b1)) + b2)

        # y = th.printing.Print('value of y:')(y)

        #Set up likelihood
        if self.continuous:
            logpxz = T.sum(-0.5 * (self.dimZ * np.log(2*np.pi) + self.logdetcov) - T.dot(T.dot((x-y).T,self.invcov),(x - y))/2)
        else:
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()

        #Set up q (??) 
        logqzx = T.sum(-(z - mu)**2/(2.*sigma**2) - 0.5 * T.log(2. * np.pi * sigma**2))

        #Choose prior
        logpz = T.sum(-(z**2)/2 - 0.5 * np.log(2 * np.pi))

        #Define lowerbound
        logp = logpxz + logpz - logqzx

        #Compute all the gradients
        derivatives = T.grad(logp,[W1,W2,W3,W4,W5,b1,b2,b3,b4,b5])

        derivatives.append(logp)

        self.gradientfunction = th.function([W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,x,eps], derivatives, on_unused_input='ignore')

    def iterate(self, miniBatch, N):
        """Compute the gradients for one miniBatch and return the updated parameters"""
        batchSize = miniBatch.shape[1]
        totalGradients = self.getGradients(miniBatch,batchSize)
        self.updateParams(totalGradients,batchSize,N)

    def getGradients(self,miniBatch,batchSize):
        totalGradients = [0] * 10
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,batchSize])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[10]

            for i in xrange(len(totalGradients)):
                if np.isnan(np.sum(gradients[i])):
                    print "The gradients contain nans, that cannot be right"
                    exit()

                totalGradients[i] += gradients[i]

        return totalGradients

    def updateParams(self,totalGradients,batchSize,N):
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            prior = self.params[i]*(i<5)

            #Include adagrad, include prior for weights
            self.params[i] = self.params[i] + (self.learning_rate/np.sqrt(self.h[i])) * (totalGradients[i] - prior*(batchSize/N))

