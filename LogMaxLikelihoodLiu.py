'''
Created on Oct 18, 2013

@author: tiago
'''

import numpy as np
import time
import random
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from LiuHugeDatasetProcess import LiuHugeDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab

class MLLiu(object):
	'''
	classdocs
	'''


	def __init__(self, dataset, seed = 1234, xmin = 0, xmax = 1):
		'''
		Constructor
		'''
		self.dataset = dataset
		self.rng = np.random.RandomState(seed)
		random.seed(seed)
		self.xmin = xmin
		self.xmax = xmax
		self.loaded = False
	
	def loadData(self, quality = 1):
		self.loaded = True
		x, y = self.dataset.returnClippedPointCloudForQuality(quality, minx=self.xmin)
		self.data = [x.values, y.values]
		self.data[1] = np.log(self.data[1])
	
	def fitData(self, n = 10, quality = 1):
		self.n = n
		if not self.loaded:
			self.loadData(quality)
		bins, _, _ = self.dataset.getBinnedData(quality, 100)
		mu, _, _ = stats.binned_statistic(self.data[0], self.data[1], bins=n)
		sig, _, _ = stats.binned_statistic(self.data[0], self.data[1], statistic=
										lambda x: np.var(x), bins=n)
		self.best = np.concatenate((mu, sig))
		print self.best
		start = time.clock()
		self.trainDifferentialEvolution(fun=self.evalTrainCost, nGenerations = 1000)
		print "Done in", np.ceil(time.clock() - start), "s"
	
	def evalTrainCost(self, x):
		'''
		calculate evaluation cost, probably could merge some ifs
		'''
		if(x.min() <= 0):
			return 1e100
		pts = self.data
		
		mpts = x[:self.n]
		spts = x[self.n:]
		ap = np.linspace(self.xmin, self.xmax, self.n)
		m = UnivariateSpline(ap, mpts)
		s = UnivariateSpline(ap, spts)
		
		penalty = 0
		'''derPoints = np.linspace(0, 1, 30)
		mDer = m(derPoints, 2)
		if(np.any(mDer < 0)):
			mPts = np.select([mDer < 0], [mDer])
			penalty = 5000*np.sqrt(np.dot(mPts,mPts))
		'''
		
		try:
			res = (np.power(pts[1]-m(pts[0]),2)/(2*s(pts[0])) 
				+ 0.5*np.log(2*np.pi*s(pts[0])))
		except FloatingPointError:
			return 1e100

		val = np.sum(res)
		if np.isnan(val):
			print x
			return 1e100
			
		return penalty + val
	
	def fitLine(self, quality = 2):
		if not self.loaded:
			self.loadData(quality)
		start = time.clock()
		self.best = np.array((-5.08, 7.26, 0.06))
		print self.best
		#self.trainDifferentialEvolution(nGenerations = 100, fun=self.linecost)
		print self.best
		print "Done in", np.ceil(time.clock() - start), "s"
		
	def linecost(self, x):
		pts = self.data
		m = lambda p: x[0]*p+x[1]
		res = (np.power(pts[1]-m(pts[0]),2)/(2*x[2]) 
				+ 0.5*np.log(2*np.pi*x[2]))
		return np.sum(res)
		
	def trainDifferentialEvolution(self, fun, nGenerations = 1000, absVal = 1e-5, 
								popSize = 20, F = 0.8, CR = 0.9):
		'''
		simple differential evolution algorithm, should be self explanatory
		pop is clipped to make sure all parameters are > 0
		'''
		settler = self.best
		ns = settler.shape[0]
		pop = settler + self.rng.normal(scale = 0.1, size=ns*popSize).reshape((popSize, ns))
		pop[0] = settler
		#pop = np.clip(pop, 0.001, 1e10)
		population = []
		scores = []
		for n in xrange(pop.shape[0]):
			cost = fun(pop[n])
			population.append(pop[n])
			scores.append(cost)
		
		for _ in xrange(nGenerations):
			start = time.clock()
			for (j, p) in enumerate(population):
				targetIndex = self.rng.randint(0, ns)
				others = population[:j] + population[(j + 1):]
				(pa, pb, pc) = random.sample(others, 3)
				trial = np.copy(p)
				for n in xrange(ns):
					r = self.rng.rand()
					if n == targetIndex or r < CR:
						trial[n] = pa[n] + F*(pb[n] - pc[n])
				cost = fun(trial)
				if cost < scores[j]:
					scores[j] = cost
					population[j] = trial
			print "New generation with score", min(scores), "up to", max(scores), "in", (time.clock() - start)*1000
		
		bestIndex = scores.index(min(scores))
		self.best = population[bestIndex]
		
def fitSpline(ml):
	dosage = []
	params = []
	q = 2
	n = 8
	fit = True
	if fit is False:
		ml.fitData(n, q)
		np.save("results/best"+str(n)+"_"+str(q)+".npy", ml.best)
	else:
		ml.loadData(q)
		ml.best = np.load("results/best"+str(n)+"_"+str(q)+".npy")
	mpts = ml.best[:n]
	spts = ml.best[n:]
	ap = np.linspace(ml.xmin, ml.xmax, len(mpts))
	m = UnivariateSpline(ap, mpts)
	s = UnivariateSpline(ap, spts)
	gs = gridspec.GridSpec(2, 3)
	plt.suptitle("lognormal fit")
	mean = lambda m, s: np.exp(m + s/2)
	std = lambda m, s: np.sqrt((np.exp(s) - 1)*np.exp(2*m+s))
	fi = lambda a, sa, sp: 2*np.power(sa, 2)/ (np.power(a,2)*2*sa+np.power(sp,2))
	fip = lambda a, sa, sp: sa / (np.power(a,2))
	plt.subplot(gs[0,0])
	plt.scatter(ml.data[0][:10000], np.exp(ml.data[1][:10000]), s=1, facecolor='0.5', lw = 0)
	plt.plot(ap, mean(m(ap), s(ap)))
	plt.plot(ap, mean(m(ap), s(ap))+std(m(ap), s(ap)), color = 'r')
	plt.plot(ap, mean(m(ap), s(ap))-std(m(ap), s(ap)), color = 'r')
	plt.subplot(gs[0,1])
	plt.plot(ap, m(ap, 1))
	plt.twinx()
	plt.plot(ap, std(m(ap), s(ap)), color='r')
	plt.subplot(gs[1,1])
	plt.plot(ap, fi(m(ap, 1), s(ap), s(ap, 1)))
	plt.plot(ap, fip(m(ap, 1), s(ap), s(ap, 1)))
	plt.ylim([0,0.02])
	plt.subplot(gs[1,0])
	_, binAv, binSig = fly.getBinnedData(q, 50)
	plt.scatter(binAv, binSig)
	plt.plot(mean(m(ap), s(ap)), std(m(ap), s(ap)))
	plt.subplot(gs[0,2])
	plt.plot(m(ap), s(ap))
	plt.subplot(gs[1,2])
	plt.scatter(ml.data[0][:10000], ml.data[1][:10000], s=1, facecolor='0.5', alpha=0.5, lw = 0)
	plt.plot(ap, m(ap))
	plt.plot(ap, m(ap) + np.sqrt(s(ap)), color = 'r')
	plt.plot(ap, m(ap) - np.sqrt(s(ap)), color = 'r')
	plt.savefig("results/liuMassiveLognormalFit.pdf")
	plt.clf()
	exit()
	bins, avs, stds = fly.getBinnedData(q, 100)
	bs = len(bins)
	rv = stats.lognorm
	avList = []
	stdList = []
	posList = []
	for bn in range(bs):
		if len(bins[bn]) == 0:
			continue
		plt.subplot(211)
		pos = (bn+0.5)/float(bs)
		posList.append(pos)
		bid = bins[bn]
		_, bx, _ = plt.hist(bid, bins=50, label=str(bn)+":"+str(pos), normed = True)
		ap = np.linspace(np.min(bx), np.max(bx), 100)
		plt.plot(ap, mlab.normpdf(ap, np.average(bid), np.std(bid)), 'r:', linewidth=1.5)
		plt.plot(ap, rv.pdf(ap, np.sqrt(s(pos)), scale=np.exp(m(pos))), 'k--', linewidth=1.5)
		plt.legend()
		plt.subplot(212)
		_, bx, _ = plt.hist(np.log(bid), bins=50, label=str(bn)+":"+str(pos), normed = True)
		ap = np.linspace(np.min(bx), np.max(bx), 100)
		plt.plot(ap, mlab.normpdf(ap, m(pos), np.sqrt(s(pos))), 'r-', linewidth=1.5)
		avList.append(np.average(np.log(bid)))
		stdList.append(np.var(np.log(bid)))
		plt.plot(ap, mlab.normpdf(ap, np.average(np.log(bid)), np.std(np.log(bid))), 'g-', linewidth=1.5)
		plt.savefig("results/mlelognormal"+str(bn)+".pdf")
		plt.clf()
	ap = np.linspace(min(posList), max(posList), len(posList))
	m = UnivariateSpline(ap, avList)
	s = UnivariateSpline(ap, stdList)
	res = (np.power(ml.data[1]-m(ml.data[0]),2)/(2*s(ml.data[0])) + 0.5*np.log(2*np.pi*s(ml.data[0])))
	print np.sum(res)
	
def fitLine(ml):
	ml.loadData(2)
	ml.fitLine()
	m = lambda x: ml.best[0]*x + ml.best[1]
	s = lambda x: x*ml.best[2]/x
	mp = lambda x: x*ml.best[0]/x
	ap = np.linspace(ml.xmin, ml.xmax, 100)
	gs = gridspec.GridSpec(2, 3)
	plt.suptitle("lognormal fit")
	mean = lambda m, s: np.exp(m + s/2)
	std = lambda m, s: np.sqrt((np.exp(s) - 1)*np.exp(2*m+s))
	fi = lambda a, sa, sp: 2*np.power(sa, 2)/ (np.power(a,2)*2*sa+np.power(sp,2))
	fip = lambda a, sa, sp: sa / (np.power(a,2))
	plt.subplot(gs[0,0])
	plt.scatter(ml.data[0][:10000], np.exp(ml.data[1][:10000]), s=1, facecolor='0.5', lw = 0)
	plt.plot(ap, mean(m(ap), s(ap)))
	plt.plot(ap, mean(m(ap), s(ap))+std(m(ap), s(ap)), color = 'r')
	plt.plot(ap, mean(m(ap), s(ap))-std(m(ap), s(ap)), color = 'r')
	plt.subplot(gs[0,1])
	plt.plot(ap, std(m(ap), s(ap)), color='r')
	_, binAv, binSig = fly.getBinnedData(2, 50)
	plt.subplot(gs[1,1])
	sl = np.log(1+np.power(binSig,2)/np.power(binAv,2))
	al = np.log(binAv)-sl
	plt.plot(al, sl)
	plt.plot(m(ap), s(ap))
	plt.subplot(gs[1,0])
	plt.scatter(binAv, binSig)
	plt.plot(mean(m(ap), s(ap)), std(m(ap), s(ap)))
	plt.subplot(gs[0,2])
	plt.plot(m(ap), s(ap))
	plt.subplot(gs[1,2])
	plt.scatter(ml.data[0][:10000], ml.data[1][:10000], s=1, facecolor='0.5', alpha=0.5, lw = 0)
	plt.plot(ap, m(ap))
	plt.plot(ap, m(ap) + np.sqrt(s(ap)), color = 'r')
	plt.plot(ap, m(ap) - np.sqrt(s(ap)), color = 'r')
	plt.savefig("results/liuMassiveLognormalFit.pdf")
	plt.clf()
	
if __name__ == '__main__':
	fly = LiuHugeDataset()
	ml = MLLiu(fly, xmin=0.1, xmax=1)
	fitLine(ml)