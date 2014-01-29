import numpy as np
import matplotlib.pyplot as plt
import matplotlib

baseline = np.load('test_logregmnist_samples_baseline_scores.npy')
features1 = np.load('test_logregmnist_samples_scores.npy')
features2 = np.load('test_logregmnist_samples_double_scores.npy')
datasetsizes = 4*2**np.linspace(0,9,10)

plt.plot(datasetsizes,baseline, linewidth = 3)
plt.plot(datasetsizes,features1, linewidth = 3)
plt.plot(datasetsizes,features2, linewidth = 3)

plt.xscale('log')
plt.title('Accuracy using log. reg.')
plt.xlabel('Size of Dataset')
plt.ylabel('Accuracy')
matplotlib.rcParams.update({'font.size': 20})	
plt.legend(('baseline','one-layer features','two-layer features'),loc=4,prop={'size':15} )
plt.tight_layout()	
plt.savefig('logreg_small_datasets.png')