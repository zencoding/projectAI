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


plt.hist(SS_per_dim_norm_20 , bins=np.logspace(-5, 0,8),color = 'green')
matplotlib.rcParams.update({'font.size': 20})
plt.gca().set_xscale("log")
plt.ylim((0,120))
plt.title('Contribution of dimensions - N = 20')
plt.xlabel('SS of weights of dimension')
plt.ylabel('# Dimensions')
plt.tight_layout()
plt.savefig('relevant_z_N20.png')
plt.close()

plt.hist(SS_per_dim_norm_400, bins=np.logspace(-5, 0, 10))
matplotlib.rcParams.update({'font.size': 20})
plt.gca().set_xscale("log")
plt.ylim((0,120))
plt.title('Contribution of dimensions - N = 400')
plt.xlabel('SS of weights of dimension')
plt.ylabel('# Dimensions')
plt.tight_layout()

plt.savefig('relevant_z_N400.png')
plt.close()