"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""A proof of concept with scikit learn implemenation of AEVB"""

#example: python trainffscikit.py

import scikitaevb
import numpy as np

print "Loading data"
f = open('dataset/freyfaces.pkl','rb')
data = cPickle.load(f)
f.close()


[N,dimX] = data.shape
HU_decoder = 200
HU_encoder = 200

dimZ = 2
L = 1
learning_rate = 0.02

batch_size = 100
continuous = True

encoder = scikitaevb.AEVB(HU_decoder,HU_encoder,dimZ,learning_rate,batch_size,10,L,continuous,True)


print "Iterating"
np.random.shuffle(data)
lowerbound = encoder.fit(data)
print lowerbound
