import matplotlib.pyplot as plt
import numpy as np

# params = np.load('chinese.npy')
params = np.load('mnist.npy')
weights = params[4]

# size = (40,40)
size = (28,28)

for i in xrange(20):
	plt.imshow(weights[:,i].reshape(size), interpolation='nearest', cmap='Greys')
	plt.savefig('weights/'+str(i))
	plt.close()
