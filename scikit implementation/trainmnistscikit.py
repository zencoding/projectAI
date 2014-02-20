"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""A proof of concept with scikit learn implemenation of AEVB"""

#example: python trainffscikit.py

import scikitaevb
import numpy as np
import gzip,cPickle

print "Loading data"
f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

[N,dimX] = data.shape
HU_decoder = 400
HU_encoder = 400

dimZ = 20
L = 1
learning_rate = 0.01

batch_size = 100
continuous = False

encoder = scikitaevb.AEVB(HU_decoder,HU_encoder,dimZ,learning_rate,batch_size,10,L,continuous,True)


print "Iterating"
# np.random.shuffle(data)
encoder.fit(data)
