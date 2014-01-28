from sklearn import svm
import numpy as np
from log_regression import *
from data import load_mnist
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", help="Specify file to save results", default = False)
parser.add_argument("-a", "--aevb", help="Specify param file", default = False)



args = parser.parse_args()

sizes = np.array([10,20,40,75,150,300,600,1000])

print 'loading data'
(x_train_load, t_train_load), (x_valid_load, t_valid_load), (x_test_load, t_test_load) = load_mnist()

results = []
results_features = []

for size in sizes:
	print 'dataset size ', size

	x_train = x_train_load[:size,:]
	t_train = t_train_load[:size]
	x_valid = x_valid_load[:1000,:]
	t_valid = t_valid_load[:1000]

	print 'Training SVM'

	gam = 0.1
	clf = svm.SVC(gamma=gam)	
	clf.fit(x_train, t_train)

	print 'calculating results on validation set...'
	dec = clf.predict(x_valid)
	result = sum(dec == t_valid)*100/len(t_valid)
	results.append(result)
	print 'result: ', result


	if args.aevb:
		print 'creating h from saved params'

		params = np.load(args.aevb)

		hidden = lambda x: (np.tanh(x.dot(params[0].T) + params[5].T) + 1 )/2
		x_train = hidden(x_train)
		x_valid = hidden(x_valid)

		print 'Training SVM'
		clf = svm.SVC(gamma = gam)	
		clf.fit(x_train, t_train)

		print 'calculating results on validation set...'
		dec = 	clf.predict(x_valid)

		result_features = sum(dec == t_valid)*100/len(t_valid)
		results_features.append(result_features)
		print 'result on features: ', result_features

plt.plot(sizes, results, sizes, results_features)
plt.axis([0, 1000, 0, 100])
plt.title('Gamma = '+ str(gam))
plt.xlabel('Size of Dataset')
plt.ylabel('Accuracy')
plt.legend( ('Au Naturel', 'On Features') )
if args.save:
	plt.savefig(title + '.png')
else:
	plt.show()

