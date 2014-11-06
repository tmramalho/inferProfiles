__author__ = 'tiago'
import scipy.io as sio
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.decomposition import PCA
import scipy.stats as stats

df = pd.read_csv('data/2xa_summary.csv', skipinitialspace=True)

labels = []
cols = []
sizes = df.n.values
ax = plt.gca()
next_col = ax._get_lines.color_cycle.next()
plt.scatter(df.p[:191], np.abs(df.r[:191]), s=sizes[:191].astype('int'),
            alpha=0.5, lw=0)
labels.append("2XA ind. sessions")
cols.append(next_col)
for i in range(192, 200):
	next_col = ax._get_lines.color_cycle.next()
	plt.scatter(df.p[i], np.abs(df.r[i]), s=sizes[i].astype('int'),
	            alpha=0.8, lw=0, c=next_col)
	labels.append(df.Linename[i])
	cols.append(next_col)
plt.plot((0.01,0.01), (0,1), 'r--', alpha=0.6)
plt.plot((0.05,0.05), (0,1), 'g--', alpha=0.6)
plt.xscale('log')
plt.xlim([1e-18, 1])
plt.ylim([0,1])
plt.xlabel('pvalue')
plt.ylabel('r squared')
rects = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in cols]
leg = plt.legend(rects, labels, loc='upper left')
leg.get_frame().set_alpha(0.5)
plt.savefig('plots/summary.pdf')

