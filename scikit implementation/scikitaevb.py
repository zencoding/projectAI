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

    def _initH(self,miniBatch):
        """Initialize H for AdaGrad

        Parameters
        ----------

        miniBatch: array-like, shape (batch_size, n_features)
            The data to use for computing gradients
        """

        e = np.random.normal(0,1,[self.n_hidden_variables,miniBatch.shape[0]])
        gradients,lowerbound = self._computeGradients(miniBatch.T,e)
        for key in gradients:
            self.h[key] += totalGradients[key]**2

    def _computeGradients(self,x,eps):
        W1, W2, W3, W4, W5 = self.params["W1"], self.params["W2"], self.params["W3"], self.params["W4"], self.params["W5"]
        b1, b2, b3, b4, b5 = self.params["b1"], self.params["b2"], self.params["b3"], self.params["b4"], self.params["b5"]

        sigmoid = lambda x: 1/1+np.exp(-x)

        h_encoder = np.tanh(W1.dot(x) + b1)
        mu_encoder = W2.dot(h_encoder) + b2
        log_sigma_encoder = 0.5*(W3.dot(h_encoder) + b3)

        z = mu_encoder + np.exp(log_sigma_encoder)*eps

        h_decoder = np.tanh(W4.dot(z) + b4)
        y = sigmoid(W5.dot(h_decoder) + b5)

        #z: 20x100
        #W1: 400x784
        #W2: 20x400
        #W3: 20x400
        #W4: 400x20
        #W5: 784x400
        #h_encoder: 400x100

        #784x100
        #1x1
        dp_dy = np.sum((x/y - (x - 1)/(y - 1)), keepdims=True)
        #400x100
        dy_dHd = W5.T.dot(sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5)))
        #100x20
        dHd_dz = -(np.tanh(W4.dot(z) + b4)**2 - 1).T.dot(W4)
        #1x1
        dz_dmue = 1
        #20x100
        dz_dlogsige = eps * np.exp(log_sigma_encoder)
        #400x20
        dmue_dHe = W2
        #400x20
        dlogsige_dHe = (0.5 * W3).T

        #Add h_decoder transposed at end
        dp_dW5 = dp_dy * ((sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5))).dot(h_decoder.T))
        dp_db5 = dp_dy * sigmoid(W5.dot(h_decoder) + b5) * (1 - sigmoid(W5.dot(h_decoder) + b5))
        print dp_dW5.shape, W5.shape
        print dp_db5.shape, b5.shape #need to sum here?

        dp_dHd = dp_dy * dy_dHd
        
        #Here suddenly need element wise with broadcasting, INCOSISTENT AAAAAH
        dp_dW4 = np.sum(dp_dHd, axis=1,keepdims=True)*(-(np.tanh(W4.dot(z) + b4)**2 - 1).dot(z.T))
        dp_db4 = np.sum(dp_dHd*(1 - np.tanh(W4.dot(z) + b4)**2),1,keepdims = True)
        print W4.shape,dp_dW4.shape
        print b4.shape,dp_db4.shape

        dp_dz = dp_dHd.dot(dHd_dz)

        #dz_dmue is 1
        dp_dmue = dp_dz.dot(1)

        dp_dW2 = dp_dmue * np.sum(h_encoder,axis=1,keepdims=True)
        #dmue_db2 is 1
        dp_db2 = dp_dmue

        #Part one of z, dz_dmue is 1
        dz_dHe = dmue_dHe
        dz_dW1 = dz_dHe.dot(-(np.tanh(W1.dot(x) + b1)**2 - 1).dot(x.T))
        dz_db1 = dz_dHe.dot(1 - np.tanh(W1.dot(x) + b1)**2)

        dp_dlogsige = dp_dz.dot(dz_dlogsige)

        dp_dW3 = dp_dlogsige * np.sum(0.5 * h_decoder, axis=1,keepdims=True)
        dp_db3 = dp_dlogsige.dot(0.5)

        #Part two of z 
        dhd_dHe_2 = dz_dlogsige.T.dot(dmue_dHe)
        dz_dW1_2 = dhd_dHe_2.dot(-(np.tanh(W1.dot(x) + b1)**2 - 1).dot(x.T))
        dz_db1_2 = dhd_dHe_2.dot(1 - np.tanh(W1.dot(x) + b1)**2)

        #weird dimension mismatch, there is something really wrong
        dp_dW1 = dp_dz.dot(dz_dW1 + dz_dW1_2)
        dp_db1 = dp_dz.dot(dz_db1 + dz_db1_2)

        return {"W1": dp_dW1, "W2": dp_dW2, "W3": dp_dW3, "W4": dp_dW4, "W5": dp_dW5,
        "b1": dp_db1, "b2": dp_db2, "b3": dp_db3, "b4": dp_db4, "b5": dp_db5}
        

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
            miniBatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            lowerbound += self._computeLB(data,eps)

            return lowerbound/N


    def _updateParams(self,minibatch,N):
        for l in xrange(self.sampling_rounds):
            e = np.random.normal(0,1,[self.n_hidden_variables,miniBatch.shape[0]])
            gradients = self._computeGradients(miniBatch.T,eps)

            if not total_gradients:
                total_gradients = gradients
            else:
                for key in gradients:
                    if np.isnan(gradients[key]):
                        print "The gradient " + key + " contain nans, that cannot be right"
                        exit()

                    totalGradients[key] += gradients[key]

        for key in self.params:
            self.h[key] += totalGradients[key]**2
            if "W" in key:
                prior = 0.5*self.params[key]
            else:
                prior = 0

            self.params[key] += self.learning_rate/np.sqrt(self.h[key]) * (totalGradients[key] - prior*(minibatch.shape[0]/N))

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
            miniBatch = data[batches[i]:batches[i+1]]
            self._initH(miniBatch)

        for i in xrange(self.n_iter):
            if self.verbose:
                print "iteration:", i
            # iteration_lowerbound = 0
            for j in xrange(0,len(batches)-2):
                # iteration_lowerbound += lowerbound
                self._updateParams(miniBatch, N)
            # print iteration_lowerbound/N
            # list_lowerbound = np.append(list_lowerbound,iteration_lowerbound/N)
        # return list_lowerbound

    def transform(self,data):
        if self.continuous:
            return np.log(1+np.exp(data.dot(self.params["W1"].T) + params["b1"].T))
        else:
            return np.tanh(data.dot(self.params["W1"].T) + self.params["b1"].T)

    def fit_transform(self,data,iterations):
        self.fit(data.iterations)
        return self.transform(data)
