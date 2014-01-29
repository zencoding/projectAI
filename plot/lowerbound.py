import numpy as np
import matplotlib.pyplot as plt

data = np.load('lowerboundchinese2.npy')
print data.shape
x = np.arange(0,data.shape[0]*10000,10000)
plt.plot(x,data)
plt.xscale('log')
plt.show()
