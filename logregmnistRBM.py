import numpy as np
from sklearn import neural_network
from data import load_mnist
from log_regression import *

#Number of hidden units
n_components=400
learning_rate = 0.01
batch_size=100
n_iter=5

rbm = neural_network.BernoulliRBM(n_components,learning_rate,batch_size,n_iter)

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

h_train = rbm.fit_transform(x_train)
h_valid = rbm.transform(x_valid)
h_test = rbm.transform(x_test)

print rbm.get_params(True)

[N,dimH] = h_train.shape
print N,dimH

w = np.zeros([10,dimH])
b = np.zeros([10])
for i in xrange(5):
    print 'iteration: ', i
    for j in xrange(N):
        w,b = sgd_iter(h_train[j],t_train[j],w,b)

    valid_correct = calculate_percentage(h_valid,t_valid,w,b)	
    train_correct = calculate_percentage(h_train,t_train,w,b)
    print 'valid correct = ', valid_correct
    print 'train correct = ', train_correct

# if args.save:
    # print "Saving results"
    # valid.append(valid_correct)
    # train.append(train_correct)
    # np.save(args.save + '_train',train)	
    # np.save(args.save + '_val',valid)

test_correct = calculate_percentage(h_test,t_test,w,b)
print test_correct
