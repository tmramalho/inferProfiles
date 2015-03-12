__author__ = 'tiago'

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

i=1
plt.figure(figsize=(30,50))
for s in np.linspace(0.1, 1, 3):
	for d in np.linspace(2, 0.1, 5):
		for n in [20, 50, 100, 1000, 10000]:
			plt.subplot(15,5,i)
			s1 = s
			s2 = s
			m1 = 1
			m2 = 1-d
			N = n
			samp_1 = np.random.normal(loc=m1, scale=s1, size=N)
			samp_2 = np.random.normal(loc=m2, scale=s2, size=N)
			_, p = stats.ks_2samp(samp_1, samp_2)
			plt.title("{0}/{1}, {2} {3} {4:.01e}".format(m1, m2, s, n, p))
			plt.hist(samp_1, bins=50, alpha=0.3)
			plt.hist(samp_2, bins=50, alpha=0.3)
			i += 1
plt.savefig('kstest.pdf')