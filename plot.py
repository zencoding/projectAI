import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp


def plot(dimZ, params):
	W1 = params[0]
	W2 = params[1]

	b1 = params[5]
	b2 = params[6]

	gridSize = 15
	gridValues = np.linspace(0.01,2,gridSize)
	gs = gridspec.GridSpec(gridSize, gridSize)

	fig = plt.figure()
	for i in xrange(gridSize):
		for j in xrange(gridSize):
			z = np.matrix([sp.invgauss.cdf(gridValues[i],1),sp.invgauss.cdf(gridValues[j],1)]).T
			y = 1 / (1 + np.exp(-(W2.dot(np.tanh(W1.dot(z) + b1)) + b2)))
			ax = fig.add_subplot(gs[i,j])
			ax.imshow(y.reshape((28,28)), interpolation='nearest', cmap='Greys')
			plt.axis('off')

	fig.patch.set_facecolor('white')
	plt.savefig('manifold.png')
