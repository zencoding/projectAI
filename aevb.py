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
        self.lowerbound = 0

        self.continuous = False

    def initParams(self):
        W1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,self.dimX))
        b1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))

        W3 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b3 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))
        
        W4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W5 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
        b5 = np.random.normal(0,self.sigmaInit,(self.dimX,1))

        self.params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]
        if self.continuous:
            W6 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
            b6 = np.random.normal(0,self.sigmaInit,(self.dimX,1))
            self.params = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]

    def initH(self,miniBatch):
        self.h = [0.01] * len(self.params)
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]

    def createGradientFunctions(self):
        #Create the Theano variables
        W1,W2,W3,W4,W5,W6,x,eps = T.dmatrices("W1","W2","W3","W4","W5","W6","x","eps")

        #Create biases as cols so they can be broadcasted for minibatches
        b1,b2,b3,b4,b5,b6 = T.dcols("b1","b2","b3","b4","b5","b6")

        if self.continuous:
            h_encoder = T.nnet.softplus(T.dot(W1,x) + b1)
        else:   
            h_encoder = T.tanh(T.dot(W1,x) + b1)

        mu_encoder = T.dot(W2,h_encoder) + b2
        log_sigma_encoder = 0.5*(T.dot(W3,h_encoder) + b3)

        #Find the hidden variable z
        z = mu_encoder + T.exp(log_sigma_encoder)*eps

        #Set up decoding layer
        if self.continuous:
            h_decoder = T.nnet.softplus(T.dot(W4,z) + b4)
            mu_decoder = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            log_sigma_decoder = 0.5*(T.dot(W6,h_decoder) + b6)
            logpxz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)
        else:
            h_decoder = T.tanh(T.dot(W4,z) + b4)
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()

        #define analytical terms for logqzx and logpz, and calculate total logp
        #NB BROARDCAST THE 1! ALSO, CHECK FOR L>1 
        rest = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder) )
        logp = logpxz + rest


        #Set up q 
        #logqzx = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_encoder) - 0.5 * ((z - mu_encoder)/T.exp(log_sigma_encoder))**2)

        #Compute prior
        #logpz = T.sum(-0.5*(z**2) - 0.5 * np.log(2 * np.pi))

        #Define lowerbound
        #logp = logpxz + logpz - logqzx

        #Compute all the gradients
        if self.continuous:
            gradvariables = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

        derivatives = T.grad(logp,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        self.gradientfunction = th.function(gradvariables + [x,eps], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [x,eps], logp, on_unused_input='ignore')

    def iterate(self, data):
        """Compute the gradients and update parameters"""
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            totalGradients = self.getGradients(miniBatch.T)
            self.updateParams(totalGradients,N,miniBatch.shape[0])

    def getLowerBound(self,data):
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            lowerbound += self.lowerboundfunction(*(self.params),x=miniBatch,eps=e)

        return lowerbound/N


    def getGradients(self,miniBatch):
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[1]])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[-1]

            for i in xrange(len(self.params)):
                if np.isnan(np.sum(gradients[i])):
                    print "The gradients contain nans, that cannot be right"
                    exit()

                totalGradients[i] += gradients[i]

        return totalGradients

    def updateParams(self,totalGradients,N,batch_size):
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            prior = 0.5*self.params[i]*(i<5)

            #Include adagrad, include prior for weights
            self.params[i] = self.params[i] + self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(batch_size/N))
