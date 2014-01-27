from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import argparse

def plot(params, num_pixels, continuous):
    (W4,W5,b4,b5) = params

    height,width = num_pixels

    gridSize = 10
    gridValues = np.linspace(0.05,0.95,gridSize)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (gridSize, gridSize), axes_pad=0.)
    for i in xrange(gridSize):
        for j in xrange(gridSize):
            z = np.matrix([sp.norm.ppf(gridValues[i]),sp.norm.ppf(gridValues[j])]).T
            if continuous:
                h_decoder = np.log(1 + np.exp(np.dot(W4,z) + b4))
                y = 1 / (1 + np.exp(-(W5.dot(h_decoder) + b5)))
            else:
                h_encoder = np.tanh(W4.dot(z) + b4)
                y = 1 / (1 + np.exp(-(W5.dot(h_encoder) + b5)))
            grid[i*10+j].imshow(y.reshape((height,width)), interpolation='nearest', cmap='Greys')
            grid[i*10+j].set_axis_off()

    plt.savefig('manifold.png', bbox_inches='tight')
    plt.close()


def plot_accuracy(filename, title):
    acc_train = np.load(filename + '_train.npy')    
    acc_val = np.load(filename + '_val.npy')
    acc_test = np.load(filename + '_test.npy')

    plt.plot(acc_train, 'k-', acc_val, 'b-', 49, acc_test, 'rx')
    plt.axis([0, 50, 90, 100])
    plt.title(title)
    plt.xlabel('Iterations (= 50.000 datapoints)')
    plt.ylabel('Accuracy')
    plt.legend( ('Train Set', 'Validation Set', 'Test Set') )
    plt.savefig(title + '.png')
    plt.close()


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", help="Specify dataset", default = 'mnist', type = str.lower, choices = ['mnist', 'freyfaces'])
parser.add_argument("-n", "--name", help="Specify name of parameters")

args = parser.parse_args()

learned_params = np.load(args.name)

if args.data == "freyfaces":
    inputparams = (learned_params[3],learned_params[4],learned_params[9],learned_params[10])
    plot(inputparams,(28,20),True)
if args.data == "mnist":
    inputparams = (learned_params[3],learned_params[4],learned_params[8],learned_params[9])
    plot(inputparams,(28,28),False)
