'''
Created on Nov 7, 2013

@author: tiago
'''

import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import LiuHugeDatasetProcess as ldp
import mpmath
import scipy.special as special

class negBin(object):
	def __init__(self, p = 0.1, r = 10):
		nbin_mpmath = lambda k, p, r: mpmath.gamma(k + r)/(mpmath.gamma(k+1)*mpmath.gamma(r))*np.power(1-p, r)*np.power(p, k)
		self.nbin = np.frompyfunc(nbin_mpmath, 3, 1)
		self.p = p
		self.r = r
		
	def mleFun(self, par, data, sm):
		'''
		Objective function for MLE estimate according to
		https://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
		
		Keywords:
		data -- the points to be fit
		sm -- \sum data / len(data)
		'''
		p = par[0]
		r = par[1]
		n = len(data)
		f0 = sm/(r+sm)-p
		f1 = np.sum(special.psi(data+r)) - n*special.psi(r) + n*np.log(r/(r+sm))
		return np.array([f0, f1])
		
	def fit(self, data, p = None, r = None):
		if p is None or r is None:
			av = np.average(data)
			va = np.var(data)
			r = (av*av)/(va-av)
			p = (va-av)/(va)
		sm = np.sum(data)/len(data)
		x = optimize.fsolve(self.mleFun, np.array([p, r]), args=(data, sm))
		self.p = x[0]
		self.r = x[1]
		
	def pdf(self, k):
		return self.nbin(k, self.p, self.r).astype('float64')
	
def compareGammaNbin(r, b):
	xm = 2*(r*b/(1-b))
	x = np.linspace(0, xm, 1000)
	distNbin = negBin(b, r)
	distGamma = stats.gamma(r, scale=(1/(1-b)))
	plt.title("r {0:f}, p {1:f}".format(r,b))
	plt.plot(x, distGamma.pdf(x), label='gamma')
	plt.plot(x, distNbin.pdf(x), label='nbin')
	plt.legend()
	plt.show()
	
def kullbackLeibler(p, q):
	lpq = np.log(p/q)
	logpq = np.where(np.isinf(lpq), 0, lpq)
	return np.sum(p*logpq)

def jensenShannon(p, q):
	m = (p + q) / 2
	return 0.5*kullbackLeibler(p, m) + 0.5*kullbackLeibler(q, m)

if __name__ == '__main__':
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Computer Modern']})
	fly = ldp.LiuHugeDataset()
	r = np.load("results/nbinParams.npy")
	bins, _, _ = fly.getBinnedData(2, 100)
	compute_nbin = True
	infoKL = []
	infoJS = []
	for i,b in enumerate(bins):
		if len(b) == 0:
			continue
		data = np.clip(b, 0.01, 1e10)*2000/128.12
		avG = np.average(data)
		stdG = np.std(data)
		distGauss = stats.norm(loc=avG, scale=stdG)
		avLN = np.average(np.log(data))
		stdLN = np.std(np.log(data))
		distLN = stats.lognorm(stdLN, scale=np.exp(avLN))
		shape, _, scale = stats.gamma.fit(data, floc=0)
		distGamma = stats.gamma(shape, scale=scale)
		if compute_nbin:
			distNbin = negBin()
			distNbin.fit(data)
			print shape, scale, distNbin.p, distNbin.r
		bc, bx, _ = plt.hist(data, bins=50, label=str(i)+":"+str(i/float(i)), normed = True)
		if compute_nbin:
			x = np.linspace(np.min(bx), np.max(bx), 100)
			plt.plot(x, distGauss.pdf(x), label='normal')
			plt.plot(x, distLN.pdf(x), label='lognormal')
			plt.plot(x, distGamma.pdf(x), label='gamma')
			plt.plot(x, distNbin.pdf(x), label='nbin')
			plt.legend()
			plt.savefig("results/distribs{0:d}.pdf".format(i))
			plt.clf()
			print distGamma.pdf(x)-distNbin.pdf(x)
		bxd = [(bx[j] + bx[j+1]) / 2 for j in range(len(bx)-1)]
		klG = kullbackLeibler(bc, distGauss.pdf(bxd))
		klL = kullbackLeibler(bc, distLN.pdf(bxd))
		klGa = kullbackLeibler(bc, distGamma.pdf(bxd))
		jsG = jensenShannon(bc, distGauss.pdf(bxd))
		jsL = jensenShannon(bc, distLN.pdf(bxd))
		jsGa = jensenShannon(bc, distGamma.pdf(bxd))
		if compute_nbin:
			klNb = kullbackLeibler(bc, distNbin.pdf(bxd))
			jsNb = jensenShannon(bc, distNbin.pdf(bxd))
			iv = [klG, klL, klGa, klNb]
			ij = [jsG, jsL, jsGa, jsNb]
		else:
			iv = [klG, klL, klGa]
			ij = [jsG, jsL, jsGa]
		infoKL.append(iv)
		infoJS.append(ij)
	plt.clf()
	xn = np.linspace(0, 1, len(infoKL))
	plt.yscale('log')
	lines = plt.plot(xn, infoKL)
	leg = plt.legend(lines, ['gauss', 'lognormal', 'gamma', 'nbin'])
	leg.get_frame().set_alpha(0.5)
	plt.savefig('results/kl.pdf')
	plt.clf()
	plt.yscale('log')
	plt.plot(xn, infoJS)
	leg = plt.legend(lines, ['gauss', 'lognormal', 'gamma', 'nbin'])
	leg.get_frame().set_alpha(0.5)
	plt.savefig('results/js.pdf')