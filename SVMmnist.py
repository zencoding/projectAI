"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

# Trains and tests a SVM with RBF kernel on multiple subsets of MNIST, both on raw data and on trained encoding parameters
# Creates 

from sklearn import svm
import numpy as np
from log_regression import *
from loadsave import load_mnist
import matplotlib.pyplot as plt
import matplotlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", help="Specify file to save results", default = False)
parser.add_argument("-a", "--aevb", help="Specify param file", default = False)
parser.add_argument("-g", "--gamma", help="Specify gamma (precision of RBF kernel)", default = False)


args = parser.parse_args()

sizes = np.array([10,20,40,75,150,300,600,1000])

print 'loading data'
(x_train_load, t_train_load), (x_valid_load, t_valid_load), (x_test_load, t_test_load) = load_mnist()

results_train = []
results_valid = []
results_features_train = []
results_features_valid = []

print 'Training SVM'

if args.gamma:
	gam = args.gamma
	print 'gamma = ', gam
else:
	print 'Using default settings for gamma'
	gam = 0

for size in sizes:
	print 'dataset size ', size

	x_train = x_train_load[:size,:]
	t_train = t_train_load[:size]
	x_valid = x_valid_load[:1000,:]
	t_valid = t_valid_load[:1000]


	clf = svm.SVC(gamma=float(gam))	
	clf.fit(x_train, t_train)

	dec = clf.predict(x_train)
	result_train = sum(dec == t_train)*100/len(t_train)
	results_train.append(result_train)
	print 'result, train: ', result_train

	dec = clf.predict(x_valid)
	result_valid = sum(dec == t_valid)*100/len(t_valid)
	results_valid.append(result_valid)
	print 'result, validation: ', result_valid


	if args.aevb:
		print 'creating h from saved params'

		params = np.load(args.aevb)

		hidden = lambda x: (np.tanh(x.dot(params[0].T) + params[5].T) + 1 )/2
		x_train = hidden(x_train)
		x_valid = hidden(x_valid)

		print 'Training SVM'
		clf = svm.SVC(gamma = float(gam))	
		clf.fit(x_train, t_train)


		dec = clf.predict(x_train)
		result_features_train = sum(dec == t_train)*100/len(t_train)
		results_features_train.append(result_features_train)
		print 'result on features, train: ', result_features_train

		dec = clf.predict(x_valid)
		result_features_valid = sum(dec == t_valid)*100/len(t_valid)
		results_features_valid.append(result_features_valid)
		print 'result on features, validation: ', result_features_valid

print 'making plots for comparison'

print results_train, sizes

plt.plot(sizes, results_train, 'r-', linewidth = 3)
plt.plot(sizes, results_valid, 'r--', linewidth = 3)
plt.plot(sizes, results_features_train, 'b-', linewidth = 3)
plt.plot(sizes, results_features_valid, 'b--', linewidth = 3)
plt.axis([-5, 1000, 0, 105])
plt.title('Gamma = '+ str(gam))
plt.xlabel('Size of Dataset')
plt.ylabel('Accuracy')
plt.legend( ('Train set, pixel values', 'Test set, pixel values', 'Train set, features', 'Test set, features'),loc=7,prop={'size':15} )
matplotlib.rcParams.update({'font.size': 20})
plt.tight_layout()

if args.save:
	plt.savefig(args.save + '.png')
else:
	plt.show()

