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

    TO DO (NEEDS SCIKIT LEARN UTIL FILE)
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
    
    def __init__(self, n_components_decoder = 200, n_components_encoder = 200, 
            n_hidden_variables = 20, learning_rate=0.01,batch_size = 100, 
            n_iter = 10, sampling_rounds = 1, continuous = False, verbose = False):
        self.n_components_decoder = n_components_decoder
        self.n_components_encoder= n_components_encoder
        self.n_hidden_variables = n_hidden_variables

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.sampling_rounds = sampling_steps
        self.verbose = verbose

        self.continuous = continuous

    def _initParams(self,dimX):
        """Create all weight and bias parameters with the right dimensions

        Parameters
        ----------
        dimX : scalar
            The dimensionality of the input data X
        """
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

    def _initH(self,miniBatch):
        """Initialize H for AdaGrad

        Parameters
        ----------

        miniBatch: array-like, shape (batch_size, n_features)
            The data to use for computing gradients
        """
        self.h = [0.01] * len(self.params)
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]

    def _createGradientFunctions(self):
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
            rest = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder) )
            logp = logpxz + rest
            gradvariables = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
            h_decoder = T.tanh(T.dot(W4,z) + b4)
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()
            logqzx = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_encoder) - 0.5 * ((z - mu_encoder)/T.exp(log_sigma_encoder))**2)
            logpz = T.sum(-0.5*(z**2) - 0.5 * np.log(2 * np.pi))
            logp = logpxz + logpz - logqzx
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

        #Compute all the gradients
        derivatives = T.grad(logp,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        gradientfunction = th.function(gradvariables + [x,eps], derivatives, on_unused_input='ignore')
        lowerboundfunction = th.function(gradvariables + [x,eps], logp, on_unused_input='ignore')

        return (gradientfunction,lowerboundfunction)

    def _getLowerBound(self,data):
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            lowerbound += self.lowerboundfunction(*(self.params),x=miniBatch.T,eps=e)

            return lowerbound/N


    def _getGradients(self,miniBatch,gradientfunction):
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[1]])
            gradients = gradientfunction(*(self.params),x=miniBatch,eps=e)

            for i in xrange(len(self.params)):
                if np.isnan(np.sum(gradients[i])):
                    print "The gradients contain nans, that cannot be right"
                    exit()

                totalGradients[i] += gradients[i]

        return (totalGradients,gradients[-1])

    def _updateParams(self,totalGradients,N,current_batch_size):
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))

    def fit(self,data,iterations):
        [N,dimX] = data.shape
        self._initParams(self,dimX)
        list_lowerbound = np.array([])

        (gradientfunction,lowerboundfunction) = self._createGradientFunctions()

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(10):
            miniBatch = data[batches[i]:batches[i+1]]
            self._initH(self,miniBatch)
        for i in xrange(iterations):
            iteration_lowerbound = 0
            for j in xrange(0,len(batches)-2):
                totalGradients,lowerbound = self._getGradients(miniBatch.T,gradientfunction)
                iteration_lowerbound += lowerbound
                self._updateParams(totalGradients,N)
            list_lowerbound = np.append(list_lowerbound,iteration_lowerbound)

    def transform(self,data):
        if self.continuous:
            return np.log(1+np.exp(data.dot(self.params[0].T) + params[6].T))
        else:
            return np.tanh(data.dot(self.params[0].T) + self.params[5].T)

    def fit_transform(self,data,iterations):
        self.fit(data.iterations)
        return self.transform(data)
