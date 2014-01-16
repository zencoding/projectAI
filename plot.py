import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import argparse

def plot(params, numPixels):
	W1 = params[0]
	W2 = params[1]

	b1 = params[5]
	b2 = params[6]
	
	size_x,size_y = numPixels

	gridSize = 8
	gridValues = np.linspace(0.1,1,gridSize)
	gs = gridspec.GridSpec(gridSize, gridSize)

	fig = plt.figure()
	for i in xrange(gridSize):
		for j in xrange(gridSize):
			z = np.matrix([sp.invgauss.cdf(gridValues[i],1),sp.invgauss.cdf(gridValues[j],1)]).T
			y = 1 / (1 + np.exp(-(W2.dot(np.tanh(W1.dot(z) + b1)) + b2)))
			ax = fig.add_subplot(gs[i,j])
			ax.imshow(y.reshape((size_x,size_y)), interpolation='nearest', cmap='Greys')
			plt.axis('off')

	fig.patch.set_facecolor('white')
	plt.savefig('manifold.png')


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", help="Specify dataset", default = 'mnist', type = str.lower, choices = ['mnist', 'freyfaces'])

args = parser.parse_args()

if args.data == "freyfaces":
	params = np.load("ff.npy")
	plot(params,(28,20))
if args.data == "mnist":
	params = np.load("mnist.npy")
	plot(params,(28,28))
