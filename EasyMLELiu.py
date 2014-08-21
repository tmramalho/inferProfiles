'''
Created on Oct 25, 2013

@author: tiago
'''

import numpy as np
import time
import random
import scipy.stats
from scipy.interpolate import UnivariateSpline
from LiuHugeDatasetProcess import LiuHugeDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab

if __name__ == '__main__':
	fly = LiuHugeDataset()
	q = 2
	stdList = []
	posList = []
	bins, avs, stds = fly.getBinnedData(q, 100)
	bs = len(bins)
	rv = scipy.stats.lognorm
	avList = []
	for bn in range(bs):
		if len(bins[bn]) == 0:
			continue
		bid = bins[bn]
		pos = (bn+0.5)/float(bs)
		posList.append(pos)
		avList.append(np.average(np.log(bid)))
		stdList.append(np.var(np.log(bid)))
	ap = np.linspace(min(posList), max(posList), len(posList))
	m = UnivariateSpline(ap, avList)
	s = UnivariateSpline(ap, stdList)
	x = np.linspace(0,1,100)
	plt.scatter(posList, avList)
	plt.plot(x, m(x))
	plt.twinx()
	plt.scatter(posList, stdList, c='r')
	plt.plot(x, s(x), 'r-')
	plt.savefig("results/comp.pdf")
	plt.clf()
	for bn in range(bs):
		if len(bins[bn]) == 0:
			continue
		pos = (bn+0.5)/float(bs)
		bid = bins[bn]
		_, bx, _ = plt.hist(np.log(bid), bins=50, label=str(bn)+":"+str(pos), normed = True)
		ap = np.linspace(np.min(bx), np.max(bx), 100)
		print m(pos), s(pos)
		plt.plot(ap, mlab.normpdf(ap, m(pos), np.sqrt(s(pos))), 'r-', linewidth=10.5)
		plt.plot(ap, mlab.normpdf(ap, np.average(np.log(bid)), np.std(np.log(bid))), 'g-', linewidth=1.5)
		plt.savefig("results/test"+str(bn)+".pdf")
		plt.clf()
	