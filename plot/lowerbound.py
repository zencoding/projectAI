import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.load('lowerboundchinese.npy')
x = np.arange(0,data.shape[0]*10000,10000)
plt.plot(x,data, linewidth = 3)

datatest = np.load('testlowerboundchinese.npy')
# datatest[:360] = datatest[:360]*100
# for i in xrange(3,160):
# 	datatest[i] = np.mean(datatest[i-2:i+3])

xtest = np.arange(0,datatest.shape[0]*50000,50000)
plt.plot(xtest,datatest, linewidth = 3)

plt.axis((100000,10000000,-700,-350))
matplotlib.rcParams.update({'font.size': 22})
plt.title('Chinese Characters')
plt.legend(('LB of train set', 'LB of test set'),loc=4,prop={'size':20})
plt.xscale('log')


plt.savefig('lowerboundchinese.png')
plt.close()
