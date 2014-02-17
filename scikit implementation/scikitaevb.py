"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

import numpy as np


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
        self.sampling_rounds = sampling_rounds
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
        W1 = np.random.normal(0,sigmaInit,(self.n_components_encoder,dimX))
        b1 = np.random.normal(0,sigmaInit,(self.n_components_encoder,1))

        W2 = np.random.normal(0,sigmaInit,(self.n_hidden_variables,self.n_components_encoder))
        b2 = np.random.normal(0,sigmaInit,(self.n_hidden_variables,1))

        W3 = np.random.normal(0,sigmaInit,(self.n_hidden_variables,self.n_components_encoder))
        b3 = np.random.normal(0,sigmaInit,(self.n_hidden_variables,1))
        
        W4 = np.random.normal(0,sigmaInit,(self.n_components_decoder,self.n_hidden_variables))
        b4 = np.random.normal(0,sigmaInit,(self.n_components_decoder,1))

        W5 = np.random.normal(0,sigmaInit,(dimX,self.n_components_decoder))
        b5 = np.random.normal(0,sigmaInit,(dimX,1))

        if self.continuous:
            W6 = np.random.normal(0,sigmaInit,(dimX,self.n_components_decoder))
            b6 = np.random.normal(0,sigmaInit,(dimX,1))
            self.params = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
            self.params = {"W1": W1, "W2" : W2, "W3": W3, "W4":W4, "W5": W5, "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5}

        self.h = dict()
        for key in self.params:
            self.h[key] = 0.01

    def _initH(self,minibatch):
        """Initialize H for AdaGrad

        Parameters
        ----------

        minibatch: array-like, shape (batch_size, n_features)
            The data to use for computing gradients
        """

        e = np.random.normal(0,1,[self.n_hidden_variables,minibatch.shape[0]])
        gradients,lowerbound = self._computeGradients(minibatch.T,e)

        for key in gradients:
            self.h[key] += gradients[key]**2

    def _computeGradients(self,x,eps):
        W1, W2, W3, W4, W5 = self.params["W1"], self.params["W2"], self.params["W3"], self.params["W4"], self.params["W5"]
        b1, b2, b3, b4, b5 = self.params["b1"], self.params["b2"], self.params["b3"], self.params["b4"], self.params["b5"]

        sigmoid = lambda x: 1/(1+np.exp(-x))

        h_encoder = np.tanh(W1.dot(x) + b1)
        mu_encoder = W2.dot(h_encoder) + b2
        log_sigma_encoder = 0.5*(W3.dot(h_encoder) + b3)

        z = mu_encoder + np.exp(log_sigma_encoder)*eps

        h_decoder = np.tanh(W4.dot(z) + b4)
        y = sigmoid(W5.dot(h_decoder) + b5)

        logpxz = np.sum(x * np.log(y) + (1 - x) * np.log(1-y))
        KLD = 0.5 * np.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - np.exp(2*log_sigma_encoder)) 
        logp = logpxz + KLD

        #now sum over batch
        h_encoder = np.sum(h_encoder,1,keepdims=True)
        h_decoder = np.sum(h_decoder,1,keepdims=True)
        z = np.sum(z,1,keepdims=True)
        #log_sigma_encoder staat nu later, maar kan evt hier. y staat ook later want we willen x niet middelen (toch?)

        #z: 20x1
        #W1: 400x784
        #W2: 20x400
        #W3: 20x400
        #W4: 400x20
        #W5: 784x400
        #h_encoder: 400x1

        #784x100
        #784x1
        dp_dy = np.sum((x/y - (x - 1)/(y - 1)),1, keepdims=True)
        #784x400
        dy_dHd = np.multiply(W5,(sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5))))
        #400x20
        dHd_dz = np.multiply(-(np.tanh(W4.dot(z) + b4)**2 - 1),W4)
        #20x1
        dz_dmue = np.ones_like(z)
        #20x1
        #kunnen we niet net zo goed 1 keer noise samplen met kleinere variantie??
        dz_dlogsige = np.sum(eps * np.exp(log_sigma_encoder),1,keepdims=True) 
        #20x400
        dmue_dHe = W2
        #20x400
        dlogsige_dHe = (0.5 * W3)

        # dDKL_dlogsige = 1 - np.exp(2*log_sigma_encoder)
        # dDKL_dmue = -mu_encoder

        #20x400
        dmue_dW2 = h_encoder.dot(np.ones((1,20))).T
        #400x784
        dHe_dW1 = (-(np.tanh(W1.dot(x) + b1)**2 - 1).dot(x.T))

        #Add h_decoder transposed at end
        dp_dW5 = dp_dy * ((sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5))).dot(h_decoder.T))
        dp_db5 = dp_dy * sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5))
        
        #1x400
        dp_dHd = dp_dy.T.dot(dy_dHd)
        
        #400x20
        dp_dW4 = dp_dHd.T * (-(np.tanh(W4.dot(z) + b4)**2 - 1).dot(z.T))
        # print W4.shape, dp_dW4.shape
        
        #400x1
        dp_db4 = dp_dHd.T*(1 - np.tanh(W4.dot(z) + b4)**2)
        # print b4.shape, dp_db4.shape
        
        #1x20
        dp_dz = dp_dHd.dot(dHd_dz)
        # print '(1, 20)', dp_dz.shape
        
        #20x1
        dp_dmue = dp_dz.T*dz_dmue
        # print '(20, 1)', dp_dmue.shape
        
        #20x400
        dp_dW2 = dp_dmue*dmue_dW2
        # print W2.shape, dp_dW2.shape
        
        #dmue_db2 is 1, 20x1
        dp_db2 = dp_dmue
        # print '(20,1)',dp_db2.shape

        #Part one of z, dz_dmue is 1     

        # print W1.shape, dHe_dW1.shape
        
        #20x1
        dp_dmue = dp_dz.T * dz_dmue
        # print '(20,1)',dp_dmue.shape
        
        #1x400
        dp_dHe = dp_dmue.T.dot(dmue_dHe)
        # print '(1x400)', dp_dHe.shape
        
        #400x784
        dp_dW1_1 = dp_dHe.T*dHe_dW1
        # print W1.shape, dp_dW1_1.shape
        
        #400x1
        dHe_db1 = np.sum(1 - np.tanh(W1.dot(x) + b1)**2,1,keepdims=True)
        # print '(400,1)', dHe_db1.shape
        
        #400x1
        dp_db1_1 = dp_dHe.T*dHe_db1
        # print '(400,1)',dp_db1_1.shape 
        
        #20x1
        dp_dlogsige = dp_dz.T*dz_dlogsige
        # print '(20,1)', dp_dlogsige.shape
        
        #20x1
        dp_dW3 = dp_dlogsige.dot(0.5 * h_decoder.T)
        # print W3.shape, dp_dW3.shape

        dp_db3 = dp_dlogsige*0.5
        # print b3.shape, dp_dlogsige.shape

        #Part two of z 
        #400x1
        dp_dHe_2 = dlogsige_dHe.T.dot(dp_dlogsige)
        # print '(400,1)', dp_dHe_2.shape

        #400x784
        dp_dW1_2 = dp_dHe_2*dHe_dW1
        # print W1.shape, dp_dW1_2.shape


        dp_db1_2 = dp_dHe_2*dHe_db1
        # print b1.shape, dp_db1_2.shape

        #weird dimension mismatch, there is something really wrong
        #400x784
        dp_dW1 = dp_dW1_1 + dp_dW1_2
        # print W1.shape, dp_dW1.shape

        #400x1
        dp_db1 = dp_db1_1 + dp_db1_2
        # print b1.shape, dp_db1.shape

        return {"W1": dp_dW1, "W2": dp_dW2, "W3": dp_dW3, "W4": dp_dW4, "W5": dp_dW5,
        "b1": dp_db1, "b2": dp_db2, "b3": dp_db3, "b4": dp_db4, "b5": dp_db5}, logp
        

    def _computeLB(self,data, eps):
        sigmoid = lambda x: 1/1+np.exp(-x)
        h_encoder = np.tanh(W1.dot(x) + b1)
        mu_encoder = W2.dot(h_encoder) + b2
        log_sigma_encoder = 0.5*(W3.dot(h_encoder) + b3)

        z = mu_encoder + np.exp(log_sigma_encoder)*eps

        logqzx = sum(sum(-(0.5 * log(2 * pi) + log_sigma_encoder) - 0.5 * ((z - mu_encoder)/exp(log_sigma_encoder))^2));
        logpz = sum(sum(-0.5*(z^2) - 0.5 * log(2 * pi)));
        logp = logpxz + logpz - logqzx;

        return logp

    def _getLowerBound(self,data):
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            minibatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,minibatch.shape[0]])
            lowerbound += self._computeLB(data,eps)

            return lowerbound/N


    def _updateParams(self,minibatch,N):
        for l in xrange(self.sampling_rounds):
            e = np.random.normal(0,1,[self.n_hidden_variables,minibatch.shape[0]])
            gradients,lowerbound = self._computeGradients(minibatch.T,e)

            if 'total_gradients' not in locals():
                total_gradients = gradients
            else:
                for key in gradients:
                    if np.isnan(np.sum(gradients[key])):
                        print "The gradient " + key + " contain nans, that cannot be right"
                        exit()

                    total_gradients[key] += gradients[key]

        for key in self.params:
            self.h[key] += total_gradients[key]**2
            if "W" in key:
                prior = 0.5*self.params[key]
            else:
                prior = 0

            self.params[key] += self.learning_rate/np.sqrt(self.h[key]) * (total_gradients[key] - prior*(minibatch.shape[0]/N))

        return lowerbound

    def fit(self,data):
        [N,dimX] = data.shape
        self._initParams(dimX)
        list_lowerbound = np.array([])

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        if self.verbose:
            print "Initialize H"
        for i in xrange(10):
            minibatch = data[batches[i]:batches[i+1]]
            self._initH(minibatch)

        for i in xrange(self.n_iter):
            if self.verbose:
                print "iteration:", i
            iteration_lowerbound = 0
            for j in xrange(0,len(batches)-2):
                lowerbound = self._updateParams(minibatch, N)
                iteration_lowerbound += lowerbound
            print iteration_lowerbound/N
            list_lowerbound = np.append(list_lowerbound,iteration_lowerbound/N)
        return list_lowerbound

    def transform(self,data):
        if self.continuous:
            return np.log(1+np.exp(data.dot(self.params["W1"].T) + params["b1"].T))
        else:
            return np.tanh(data.dot(self.params["W1"].T) + self.params["b1"].T)

    def fit_transform(self,data,iterations):
        self.fit(data.iterations)
        return self.transform(data)
