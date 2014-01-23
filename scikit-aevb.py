"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np
import theano as th
import theano.tensor as T

class AEVB:
    """Auto-encoding variational Bayes (AEVB).

    An auto-encoder with variational Bayes inference.

    Parameters
    ----------
    n_components_decoder : int, optional
        Number of binary hidden units for decoder.

    n_components_encoder : int, optional
        Number of binary hidden units for encoder.

    n_hidden_variables : int, optional
    	The dimensionality of Z

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    sampling_rounds : int, optional
    	Number of sampling rounds done on the minibatch

    continuous : boolean, optional
    	Set what type of data the auto-encoder should model

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    `params` : list-like, list of weights and biases.

  

    Examples
    --------

    ----------
    References

    [1] Kingma D.P., Welling M. Stochastic Gradient VB and the Variational Auto-Encoder
    Arxiv, preprint. http://arxiv.org/pdf/1312.6114v6.pdf
    """
    
    def __init__(self, n_components_decoder = 200, n_components_encoder = 200, n_hidden_variables = 20, learning_rate=0.01,batch_size = 100, n_iter = 10, sampling_rounds = 1, continuous = False, verbose = False, random_state = None):
        self.n_components_decoder = n_components_decoder
        self.n_components_encoder= n_components_encoder
        self.n_hidden_variables = n_hidden_variables

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.sampling_rounds = sampling_steps
        self.verbose = verbose
        self.random_state = random_state

        self.continuous = continuous

    def initParams(self,dimX):
        """Create all weight and bias parameters in the right dimension"""
    	sigmaInit = 0.01
        W1 = np.random.normal(0,self.sigmaInit,(self.n_components_encoder,dimX))
        b1 = np.random.normal(0,self.sigmaInit,(self.n_components_encoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.n_hidden_variables,self.n_components_encoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.n_hidden_variables,1))

        W3 = np.random.normal(0,self.sigmaInit,(self.n_hidden_variables,self.n_components_encoder))
        b3 = np.random.normal(0,self.sigmaInit,(self.n_hidden_variables,1))
        
        W4 = np.random.normal(0,self.sigmaInit,(self.n_components_decoder,self.n_hidden_variables))
        b4 = np.random.normal(0,self.sigmaInit,(self.n_components_decoder,1))

        W5 = np.random.normal(0,self.sigmaInit,(dimX,self.n_components_decoder))
        b5 = np.random.normal(0,self.sigmaInit,(dimX,1))

        self.params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]
        if self.continuous:
            W6 = np.random.normal(0,self.sigmaInit,(dimX,self.n_components_decoder))
            b6 = np.random.normal(0,self.sigmaInit,(dimX,1))
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

        mu = T.dot(W2,h_encoder) + b2
        sigma = T.exp(0.5*(T.dot(W3,h_encoder) + b3))

        #Find the hidden variable z
        z = mu + sigma*eps

        #Set up likelihood
        h_decoder = T.tanh(T.dot(W4,z) + b4)

        if self.continuous:
            decoder_mu = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            decoder_sigma = T.exp(0.5*(T.dot(W6,h_decoder) + b6))
            logpxz = T.sum(-(0.5 * np.log(np.pi) + T.log(decoder_sigma)) - 0.5 * ((x - decoder_mu) / decoder_sigma)**2)
        else:
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()

        #Set up q 
        logqzx = T.sum(-(z - mu)**2/(2.*sigma**2) - 0.5 * T.log(2. * np.pi * sigma**2))

        #Compute prior
        logpz = T.sum(-(z**2)/2 - 0.5 * np.log(2 * np.pi))

        #Define lowerbound
        logp = logpxz + logpz - logqzx

        #Compute all the gradients
        if self.continuous:
            gradvariables = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

        derivatives = T.grad(logp,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        self.gradientfunction = th.function(gradvariables + [x,eps], derivatives, on_unused_input='ignore')

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
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,self.batch_size])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[-1]

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
