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

    gridSize = 10
    gridValues = np.linspace(0.01,0.99,gridSize)
    gs = gridspec.GridSpec(gridSize, gridSize)

    fig = plt.figure()
    for i in xrange(gridSize):
        for j in xrange(gridSize):
            z = np.matrix([sp.norm.ppf(gridValues[i]),sp.norm.ppf(gridValues[j])]).T
            y = 1 / (1 + np.exp(-(W2.dot(np.tanh(W1.dot(z) + b1)) + b2)))
            ax = fig.add_subplot(gs[i,j])
            ax.imshow(y.reshape((size_x,size_y)), interpolation='nearest', cmap='Greys')
            plt.axis('off')

    fig.patch.set_facecolor('white')
    plt.savefig('manifold.png')
    plt.close()


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", help="Specify dataset", default = 'mnist', type = str.lower, choices = ['mnist', 'freyfaces'])
parser.add_argument("-n", "--name", help="Specify name of parameters")

args = parser.parse_args()

params = np.load(args.name)

if args.data == "freyfaces":
    plot(params,(28,20))
if args.data == "mnist":
    plot(params,(28,28))
