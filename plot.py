from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import argparse

def plot(params, num_pixels):
    (W4,W5,b4,b5) = params

    height,width = num_pixels

    gridSize = 10
    gridValues = np.linspace(0.01,0.99,gridSize)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (gridSize, gridSize), axes_pad=0.)
    for i in xrange(gridSize):
        for j in xrange(gridSize):
            z = np.matrix([sp.norm.ppf(gridValues[i]),sp.norm.ppf(gridValues[j])]).T
            y = 1 / (1 + np.exp(-(W5.dot(np.tanh(W4.dot(z) + b4)) + b5)))
            grid[i*10+j].imshow(y.reshape((height,width)), interpolation='nearest', cmap='Greys')
            grid[i*10+j].set_axis_off()

    plt.savefig('manifold.png', bbox_inches='tight')
    plt.close()


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", help="Specify dataset", default = 'mnist', type = str.lower, choices = ['mnist', 'freyfaces'])
parser.add_argument("-n", "--name", help="Specify name of parameters")

args = parser.parse_args()

learned_params = np.load(args.name)

if args.data == "freyfaces":
    inputparams = (learned_params[3],learned_params[4],learned_params[9],learned_params[10])
    plot(inputparams,(28,20))
if args.data == "mnist":
    inputparams = (learned_params[3],learned_params[4],learned_params[8],learned_params[9])
    plot(inputparams,(28,28))
