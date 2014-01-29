import numpy as np
import matplotlib.pyplot as plt
import matplotlib

weights20 = np.load('400HU20Zmnist.npy')[3]
weights400 = np.load('400HU400Zmnist.npy')[3]

SS_per_dim_400 = np.mean(weights400**2,0)
SS_per_dim_norm_400 = SS_per_dim_400/max(SS_per_dim_400)
SS_per_dim_20  = np.mean(weights20 **2,0)
SS_per_dim_norm_20  = SS_per_dim_20 /max(SS_per_dim_20 )

values_SS = 10**np.linspace(-6,0,10)
counts_SS_20  = np.array([])
counts_SS_400 = np.array([])
for value in values_SS:
	counts_SS_400 = np.append(counts_SS_400, sum(SS_per_dim_norm_400>value))
	counts_SS_20  = np.append(counts_SS_20 , sum(SS_per_dim_norm_20 >value))

plt.plot(values_SS,counts_SS_20 , linewidth = 3)
plt.plot(values_SS,counts_SS_400, linewidth = 3)
plt.xscale('log'), plt.yscale('log')
plt.title('Distribution of size of weights (SS)')
plt.xlabel('Value (normalized)')
plt.ylabel('NDim with SS of weights > Value')
matplotlib.rcParams.update({'font.size': 16})	
plt.legend(('20 dimensions','400 dimensions'),loc=4,prop={'size':18} )
plt.tight_layout()	
plt.savefig('N_irrelevant_hidden_dim_SS.png')
#plt.show()


max_per_dim_400 = np.max(abs(weights400),0)
max_per_dim_norm_400 = max_per_dim_400/np.max(max_per_dim_400)

max_per_dim_20  = np.max(abs(weights20 ),0)
max_per_dim_norm_20  = max_per_dim_20 /np.max(max_per_dim_20 )

values_max = 10**np.linspace(-3,0,10)
counts_max_400 = np.array([])
counts_max_20  = np.array([])
for value in values_max:
	counts_max_400 = np.append(counts_max_400, sum(max_per_dim_norm_400>value))
	counts_max_20  = np.append(counts_max_20 , sum(max_per_dim_norm_20 >value))

plt.plot(values_max,counts_max_20 , linewidth = 3)
plt.plot(values_max,counts_max_400, linewidth = 3)
plt.xscale('log'), plt.yscale('log')
plt.title('Distribution of size of weights (max.)')
plt.xlabel('Weight Size (normalized)')
plt.ylabel('NDim with weight > WS')
matplotlib.rcParams.update({'font.size': 16})	
plt.legend(('20 dimensions','400 dimensions'),loc=4,prop={'size':18} )
plt.tight_layout()	
plt.savefig('N_irrelevant_hidden_dim_max.png')
#plt.show()
