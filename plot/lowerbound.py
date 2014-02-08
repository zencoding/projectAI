import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.load('lowerbound400HU20Zmnist.npy')
x = np.arange(0,data.shape[0]*60000,60000)
plt.plot(x,data, linewidth = 3)

datatest = np.load('testlowerbound400HU20Zmnist.npy')
datatest[:360] = datatest[:360]*100
for i in xrange(3,1000):
	datatest[i] = np.mean(datatest[i-2:i+3])

xtest = np.arange(0,datatest.shape[0]*60000,60000)
plt.plot(xtest,datatest, linewidth = 3)

plt.axis((100000,100000000,-150,-95))
matplotlib.rcParams.update({'font.size': 22})
plt.title('MNIST, Nz = 20')
plt.legend(('LB of train set', 'LB of test set'),loc=4,prop={'size':20})
plt.xscale('log')


plt.savefig('lowerboundAEVBMNIST.png')
plt.close()
