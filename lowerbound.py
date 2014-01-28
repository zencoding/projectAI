import numpy as np
import matplotlib.pyplot as plt

data = np.load('lowerboundff4.npy')
x = np.arange(0,data.shape[0]*1965,1965)
plt.plot(x,data)
plt.xscale('log')
plt.show()
