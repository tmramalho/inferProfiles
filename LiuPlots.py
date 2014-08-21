'''
Created on Nov 4, 2013

@author: tiago
'''

import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import LiuHugeDatasetProcess as ldp

def chi2pval(data):
	av = np.average(data)
	va = np.var(data)
	hist, binEdges = np.histogram(data, bins=50, density=True)
	rvn = stats.norm(loc = av, scale = np.sqrt(va))
	eHist = np.array([rvn.cdf(binEdges[i+1])-rvn.cdf(binEdges[i]) for i in range(len(binEdges)-1)])
	chi2 = np.sum(np.power(hist-eHist,2)/eHist)
	df = len(hist)-1
	rv = stats.chi2(df)
	print chi2, df, 1-rv.cdf(chi2)
	
def ksTest(data, dist='norm'):
	av = np.average(data)
	va = np.var(data)
	normed_data = (data - av)/np.sqrt(va)
	_,p = stats.kstest(normed_data, dist)
	return p

def logNormKSTest(data):
	rv = stats.lognorm
	av = np.average(data)
	st = np.std(data)
	mu = np.log(np.power(av,2)/np.sqrt(np.power(av,2) + np.power(st,2)))
	sig = np.log(np.sqrt(np.power(av,2) + np.power(st,2))/av)
	shape, _, scale = rv.fit(data, args=(np.sqrt(sig), 0, np.exp(mu)), floc=0)
	_,p = stats.kstest(data, 'lognorm', args=(shape,0,scale))
	return p

if __name__ == '__main__':
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Computer Modern']})
	fly = ldp.LiuHugeDataset()
	r = np.load("results/nbinParams.npy")
	plt.figure(figsize=(7,14))
	plt.subplot(611)
	p1 = plt.plot(r[:,0],r[:,2], 'r-', label=r'$\mu$')
	plt.ylabel(r'\#bursts $(\mu)$',fontsize=10)
	plt.twinx()
	b = r[:,1]/(1-r[:,1])
	p2 = plt.plot(r[:,0], b, 'b-', label='b')
	lns = p1+p2
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs)
	plt.xlabel(r'ap position')
	plt.ylabel(r'\#proteins from single rna (b)',fontsize=10)
	plt.subplot(612)
	p1 = plt.plot(r[:,0], r[:,2]*b, label = r'$\langle n \rangle$')
	plt.legend()
	plt.ylabel(r'\#molecules', fontsize=10)
	plt.twinx()
	p2 = plt.plot(r[:,0], r[:,2]*b*(1+b), 'r-', label = r'$\sigma_n^2$')
	lns = p1+p2
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs)
	plt.ylabel(r'\#molecules squared', fontsize=10)
	plt.xlabel(r'ap position')
	plt.subplot(613)
	sigma = lambda p, x: p[0]*(p[1]*x*x + x + p[2])
	errfunc = lambda p, x, y: 10000000 if np.any(p < 0) else sigma(p,x) - y
	par = np.array([0.001,10,0.1])
	par, success = optimize.leastsq(errfunc, par, args=(r[3:,2]*b[3:], r[3:,2]*b[3:]*(1+b[3:])), maxfev = 1800)
	print par
	plt.scatter(r[:,2]*b, r[:,2]*b*(1+b))
	plt.plot(r[:,2]*b, sigma(par, r[:,2]*b), 'r-')
	plt.title("mean vs av: ${0:.2e} * ({1:.2f}*av^2 + av + {2:.2f})$".format(par[0], par[1], par[2]), fontsize = 8)
	plt.subplot(614)
	bins, avs, stds = fly.getBinnedData(2, 100, length=True)
	num = []
	pval = []
	pval2 = []
	for bn in range(100):
		if len(bins[bn]) == 0:
			continue
		data = np.clip(bins[bn], 0.01, 1e10)*2000/128.12
		num.append(len(data))
		ksmirnov = ksTest(data)
		pval.append(ksmirnov)
		pval2.append(logNormKSTest(data))
	try:
		plt.bar(r[:,0], num, label=r"\#samples")
	except:
		print "you imported the wrong data"
	plt.ylabel(r'\#samples', fontsize=10)
	plt.subplot(615)
	plt.title("kolmogorov smirnoff test for gaussianity", fontsize = 10)
	plt.yscale('log')
	plt.bar(r[:,0], pval, log=True)
	plt.plot(r[:,0], np.ones(shape=r[:,0].shape)*0.05, "g--", label=r"5\% confidence")
	plt.plot(r[:,0], np.ones(shape=r[:,0].shape)*0.01, "r--", label=r"1\% confidence")
	plt.ylabel('pvalue', fontsize=10)
	leg = plt.legend(loc="lower left", fontsize = 8)
	leg.get_frame().set_alpha(0.5)
	plt.subplot(616)
	plt.title("kolmogorov smirnoff test for lognormal", fontsize = 10)
	plt.yscale('log')
	plt.bar(r[:,0], pval2, log=True)
	plt.plot(r[:,0], np.ones(shape=r[:,0].shape)*0.05, "g--", label=r"5\% confidence")
	plt.plot(r[:,0], np.ones(shape=r[:,0].shape)*0.01, "r--", label=r"1\% confidence")
	plt.ylabel('pvalue', fontsize=10)
	leg = plt.legend(loc="lower left", fontsize = 8)
	leg.get_frame().set_alpha(0.5)
	plt.savefig("results/mater.pdf")
