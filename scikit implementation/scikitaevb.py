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
        np.random.seed(42)
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

        np.random.seed(42)
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

        #Calculate lowerbound probability
        logpxz = np.sum(x * np.log(y) + (1 - x) * np.log(1-y))
        KLD = 0.5 * np.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - np.exp(2*log_sigma_encoder)) 
        logp = logpxz + KLD

        #W5: This is correct
        dp_dy    = (x/y - (1 - x)/(1 - y))
        dy_dSig   = (y * (1 - y))
        dp_dW5   = (dp_dy * dy_dSig).dot(h_decoder.T)
        dp_db5   = np.sum(dp_dy * dy_dSig,1,keepdims=True)

        dSig_dHd = W5
        #400x100
        dp_dHd   = ((dp_dy * dy_dSig).T.dot(dSig_dHd)).T
        dHd_dtanh   = 1 - h_decoder**2

        #W4: This is correct
        dp_dW4   = (dp_dHd * dHd_dtanh).dot(z.T)
        dp_db4   = np.sum(dp_dHd * dHd_dtanh, axis = 1, keepdims = True)


        dtanh_dz = W4
        #Maybe just 1 is also good
        dz_dmue  = np.ones_like(mu_encoder)
        dmue_dW2 = h_encoder
        dlogsige_dHe = 0.5 * W3
        dmue_db2 = np.ones_like(b2)

        dp_dz    = (dp_dHd *dHd_dtanh).T.dot(dtanh_dz)
        dp_dmue  = dp_dz.T * dz_dmue

        dp_dW2   = dp_dmue.dot(dmue_dW2.T)
        dp_db2   = np.sum(dp_dmue, axis = 1, keepdims = True)

        dKLD_dmue = -mu_encoder
        dKLD_dW2 = dKLD_dmue.dot(dmue_dW2.T)
        dKLD_db2 = np.sum(dKLD_dmue * dmue_db2, axis = 1, keepdims = True)

        #W2: This is correct
        dp_dW2 += dKLD_dW2
        dp_db2 += dKLD_db2



        dmue_dHe = W2
        dz_dlogsige = np.sum(eps * np.exp(log_sigma_encoder),1,keepdims=True) 


        dHe_dW1  = (1 - (np.tanh(W1.dot(x) + b1)**2).dot(x.T))
        dHe_db1  = np.sum(1 - np.tanh(W1.dot(x) + b1)**2,1,keepdims=True)


        

        #Part one of z    
        dp_dmue  = dp_dz.T * dz_dmue
        dp_dHe   = dp_dmue.T.dot(dmue_dHe)
        dp_dW1_1 = dp_dHe.T*dHe_dW1
        dp_db1_1 = dp_dHe.T*dHe_db1

        #Part two of z
        dp_dlogsige = dp_dz.T*dz_dlogsige 
        dp_dHe_2 = dlogsige_dHe.T.dot(dp_dlogsige)
        dp_dW1_2 = dp_dHe_2*dHe_dW1
        dp_db1_2 = dp_dHe_2*dHe_db1
        dp_dW1   = dp_dW1_1 + dp_dW1_2
        dp_db1   = dp_db1_1 + dp_db1_2
        dlogsige_dW3 = np.sum(0.5*h_decoder,1,keepdims=True).T * np.ones_like(W3)
        dlogsige_db3 = 0.5 * np.ones_like(b3)
        dp_dW3   = dp_dlogsige * dlogsige_dW3
        dp_db3   = dp_dlogsige * dlogsige_db3

        #gradients of KL divergence term
        dKLD_dlogsige = np.sum(1 - np.exp(2*log_sigma_encoder),1,keepdims=True)
        #ADD SUMS
        dKLD_dHe_1 = dlogsige_dHe.T.dot(dKLD_dlogsige)
        dKLD_dHe_2 = dmue_dHe.T.dot(dKLD_dmue)

        dKLD_dW1_1 = dHe_dW1*dKLD_dHe_1
        dKLD_db1_1 = dHe_db1*dKLD_dHe_1
        dKLD_dW1_2 = dHe_dW1*dKLD_dHe_2
        dKLD_db1_2 = dHe_db1*dKLD_dHe_2
        dKLD_dW1 = dKLD_dW1_1 + dKLD_dW1_2
        dKLD_db1 = dKLD_db1_1 + dKLD_db1_2

       

        dKLD_dW3 = dKLD_dlogsige*dlogsige_dW3
        dKLD_db3 = dKLD_dlogsige*dlogsige_db3

        #add terms together to compute total gradients

        #print dKLD_db1.shape   
        dp_dW1 += dKLD_dW1
        dp_db1 += dKLD_db1
        dp_dW2 += dKLD_dW2
        dp_db2 += dKLD_db2
        dp_dW3 += dKLD_dW3
        dp_db3 += dKLD_db3

        #print dp_dW1.shape, dp_db1.shape, dp_dW2.shape, dp_db2.shape, dp_dW3.shape, dp_db3.shape

        #norms = [np.linalg.norm(dp_dW1), np.linalg.norm(dp_db1),np.linalg.norm(dp_dW2),np.linalg.norm(dp_db2),
        #np.linalg.norm(dp_dW3),np.linalg.norm(dp_db3),np.linalg.norm(dp_dW4),np.linalg.norm(dp_db4),
        #np.linalg.norm(dp_dW5),np.linalg.norm(dp_db5)]

        #average = np.sum(norms)/10 
        #print norms/average
        return {"W1": dp_dW1, "W2": dp_dW2, "W3": dp_dW3, "W4": dp_dW4, "W5": dp_dW5,
        "b1": dp_db1, "b2": dp_db2, "b3": dp_db3, "b4": dp_db4, "b5": dp_db5}, logp

    def _updateParams(self,minibatch,N):
        for l in xrange(self.sampling_rounds):
            np.random.seed(42)
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
            raw_input()
            print key, total_gradients[key]
            print key, np.linalg.norm(total_gradients[key])
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
        # for i in xrange(10):
        #     minibatch = data[batches[i]:batches[i+1]]
        #     self._initH(minibatch)
        #
        for i in xrange(self.n_iter):
            if self.verbose:
                print "iteration:", i
            iteration_lowerbound = 0
            for j in xrange(0,len(batches)-2):
                minibatch = data[batches[j]:batches[j+1]]
                print np.sum(minibatch[0])
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
