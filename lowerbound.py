import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.load('lowerbound400HU20Zmnist.npy')
x = np.arange(0,data.shape[0]*60000,60000)
plt.plot(x,data, linewidth = 5)
plt.axis((100000,100000000,-150,-95))
matplotlib.rcParams.update({'font.size': 22})
plt.title('MNIST, Nz = 20')
plt.xscale('log')
#plt.show()

plt.savefig('lowerboundAEVBMNIST.png')
plt.close()